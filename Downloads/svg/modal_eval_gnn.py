"""
modal_eval.py

Two-phase design to fit a 6-hour budget and avoid GPU memory contention:

  PHASE 1 (this file, custom): boot ONLY the vision server (real HF
  checkpoint Qwen/Qwen2.5-VL-3B-Instruct, not the MLX one), walk the
  dataset, populate vision_cache/ for every case. Then shut the vision
  server down and free its GPU memory.

  PHASE 2 (reuses your proven CLI path exactly): boot ONLY the text server,
  run `python3 -m svgpatchlab.cli evaluate --config ...` with
  vision_context.enabled=true. Since the cache is warm, VisionContextAnnotator
  hits `_read_cache` for every node -- NO live vision calls happen during
  phase 2, so its cost should closely match your prior text-only run
  (~30s/case observed).

WHY TWO PHASES INSTEAD OF ONE CONCURRENT RUN:
- Avoids needing to fit two model servers in GPU memory simultaneously.
- Phase 2 reuses the exact CLI invocation that already worked for you on
  Kaggle -- lower risk than a custom in-process eval loop.
- Vision annotation cost is now fully separated and measurable on its own,
  so if it's slower than estimated you can stop after phase 1 and still
  have something useful (a cache to reuse on a later, longer run) rather
  than losing partial progress mid-eval.

HONEST TIME ESTIMATE:
  500 cases, ~10 candidate nodes/case average -> ~5000 vision calls.
  At 1-2.5s/call: 1.4-3.5h for phase 1, plus ~4.2h for phase 2 (matching
  your observed 30s/case average) = 5.6-7.6h total. THIS CAN EXCEED 6
  HOURS. DEFAULT_LIMIT_PER_TASK below is set to 60 (300 cases, ~3.3-4.6h)
  for real margin -- raise it once you've seen phase 1's actual per-call
  latency in this specific environment.

STILL UNVERIFIED (I don't have these files):
- svgpatchlab/models/__init__.py's exact create_model()/ModelAdapter
  interface -- phase 1 below calls create_model() with the same config-dict
  pattern used in runner.py, so it should work if that function accepts
  plain dicts the way I've inferred from usage, but hasn't been tested
  against your real file.
- Whether your installed vLLM version supports Qwen2.5-VL-3B-Instruct as a
  served model -- broad support exists in recent vLLM releases for this
  Qwen family, but I haven't confirmed the specific pinned version. Watch
  phase 1's server boot log for an unsupported-architecture error.

Run with: modal run modal_eval.py
"""
import modal

app = modal.App("svgpatchlab-vision-eval")

DEFAULT_LIMIT_PER_TASK = 60  # 300 cases total across 5 supported tasks -- see time estimate above

weights_volume = modal.Volume.from_name("svgpatchlab-hf-weights", create_if_missing=True)
repo_volume = modal.Volume.from_name("svgpatchlab-repo-and-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libcairo2")
    .run_commands("ldconfig")
    .pip_install(
        # NOTE: still unpinned (>=0.8.0) deliberately. The real error showed
        # the resolved version is much newer than 0.8.x (flags like
        # --moe-backend/--speculative-config don't exist that early), and it
        # expects --limit-mm-per-prompt's value as JSON -- fixed below.
        # Pinning to an older version without confirming it ALSO expects the
        # JSON format would risk reintroducing the old key=value expectation
        # instead -- an unverified guess in the other direction. After your
        # next run, `pip show vllm` inside the container tells you the real
        # resolved version if you want to pin it deliberately later.
        "vllm>=0.8.0",
        "outlines",
        "requests",
        "qwen-vl-utils",
        # NOTE: repo's `pip install -e ".[vision]"` step (below) does NOT
        # actually install cairosvg -- confirmed by a real crash
        # (ModuleNotFoundError: cairosvg) despite that step running fine.
        # Installing directly here instead of trusting the extras group.
        "cairosvg",
    )
)

GITHUB_TOKEN = modal.Secret.from_name("github-token")  # set via `modal secret create github-token GITHUB_TOKEN=...`


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/root/.cache/huggingface": weights_volume,
        "/repo": repo_volume,
    },
    secrets=[GITHUB_TOKEN],
    timeout=7 * 60 * 60,  # hard ceiling above the 6h target, so a slow phase 1 doesn't silently truncate phase 2
)
def run_two_phase_eval(limit_per_task: int = DEFAULT_LIMIT_PER_TASK, skip_phase2: bool = False):
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    if not Path("/repo/EditSVG-patch-lab").exists():
        token = os.environ["GITHUB_TOKEN"]
        subprocess.run(
            ["git", "clone", "--recurse-submodules",
             f"https://{token}@github.com/smerarawal/EditSVG-patch-lab.git",
             "/repo/EditSVG-patch-lab"],
            check=True,
        )
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[vision]"],
                        cwd="/repo/EditSVG-patch-lab", check=True)
    os.chdir("/repo/EditSVG-patch-lab")
    sys.path.insert(0, "/repo/EditSVG-patch-lab")

    config_path = Path("configs/experiments/skeleton_patch_vision_72b.json")
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({
            "dataset": {"root": "SVGEditBench",
                        "tasks": ["change_color", "set_contour", "upside_down", "transparency", "crop_to_half"],
                        "limit": limit_per_task * 5, "limit_per_task": limit_per_task},
            "architecture": {"name": "skeleton_patch", "vision_context": {
                "enabled": True, "max_nodes": 48, "image_size": 384, "min_visible_pixels": 12,
                "include_root": False, "cache_dir": "runs/vision_cache",
                "model": {"config_file": "../models/qwen2.5-vl-3b-hf.json"}}},
            "model": {"config_file": "../models/qwen3.5-4b-openai.json"},
            "evaluation": {"render": True, "render_size": 72, "save_outputs": True,
                            "output_dir": "runs/skeleton_patch_vision_modal"},
        }, indent=2))

    # REAL (non-MLX) vision model config -- this is the actual fix vs. Kaggle
    vision_model_config_path = Path("configs/models/qwen2.5-vl-3b-hf.json")
    vision_model_config_path.write_text(json.dumps({
        "adapter": "openai_compatible", "base_url": "http://localhost:8001/v1",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct", "temperature": 0.0, "max_tokens": 256, "timeout": 180,
    }, indent=2))

    text_model_config_path = Path("configs/models/qwen3.5-4b-openai.json")
    text_model_config_path.write_text(json.dumps({
        "adapter": "openai_compatible", "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen3.5-4B", "temperature": 0.0, "max_tokens": 4096, "json_mode": False, "timeout": 900,
    }, indent=2))

    def boot_server(model_id, port, extra_args, log_path):
        log = open(log_path, "w")
        env = dict(os.environ)
        # FlashInfer's top-k/top-p sampler JIT-compiles a CUDA kernel on first
        # use, which requires nvcc (the CUDA *compiler* toolchain) -- this image
        # only has the CUDA runtime via torch/vllm, not nvcc. Confirmed root
        # cause: "Could not find nvcc and default cuda_home=... doesn't exist".
        # Disabling the flashinfer sampler falls back to vLLM's native PyTorch
        # sampling path, which needs no compilation. Applied to both servers --
        # the text server would hit the identical crash in phase 2 otherwise,
        # just after burning the entire (paid) phase 1 first.
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        proc = subprocess.Popen(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
             "--model", model_id, "--port", str(port), "--dtype", "half"] + extra_args,
            stdout=log, stderr=subprocess.STDOUT, env=env,
        )
        for _ in range(600):
            if proc.poll() is not None:
                full_log = Path(log_path).read_text()
                marker = "EngineCore"
                idx = full_log.find(marker)
                # Grab from just before EngineCore starts all the way to the end --
                # no arbitrary cutoff. If this is still too big to be useful, the
                # crash is happening later than expected and we need the full file.
                region = full_log[max(0, idx - 200):] if idx != -1 else full_log
                raise RuntimeError(
                    f"Server crashed booting {model_id}:\n"
                    f"--- from first '{marker}' mention to end of log ---\n{region}"
                )
            try:
                import urllib.request
                urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2)
                return proc
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"Server for {model_id} did not become ready in time")

    # ===================== PHASE 1: vision annotation pass =====================
    print("=" * 65)
    print(" PHASE 1 - booting vision server, populating vision_cache")
    print("=" * 65)
    t0 = time.time()
    vision_proc = boot_server("Qwen/Qwen2.5-VL-3B-Instruct", 8001,
                               # NOTE: newer vLLM (>=0.8.0 pulled the latest available) parses
                               # --limit-mm-per-prompt's value as JSON, not the old key=value
                               # syntax -- confirmed by the real error: "Value image=1 cannot
                               # be converted to <function loads ...>" (that's json.loads).
                               #
                               # NOTE 2: previous crash showed weights loading fine (6.99 GiB)
                               # then dying immediately after -- classic shape of a CUDA OOM
                               # during vLLM's post-load profiling/cudagraph-capture pass on a
                               # VL model, since no --gpu-memory-utilization was set here
                               # (defaulted to 0.9) and cudagraph capture sizes go up to 512.
                               # --enforce-eager skips graph capture entirely (safe, some
                               # latency cost); explicit --gpu-memory-utilization leaves more
                               # headroom for the vision encoder's activation memory.
                               ["--limit-mm-per-prompt", '{"image": 1}', "--max-model-len", "8192",
                                "--gpu-memory-utilization", "0.85", "--enforce-eager"],
                               "vllm_vision_server.log")
    print(f"Vision server up after {time.time()-t0:.0f}s")

    from svgpatchlab.core import build_scene
    from svgpatchlab.vision import VisionContextAnnotator
    from svgpatchlab.data import SVGEditBench
    from svgpatchlab.models import create_model, RecordingModelAdapter

    exp_cfg = json.loads(config_path.read_text())
    vision_cfg = exp_cfg["architecture"]["vision_context"]
    annotator = VisionContextAnnotator(vision_cfg)
    vision_model = RecordingModelAdapter(create_model({
        "adapter": "openai_compatible", "base_url": "http://localhost:8001/v1",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct", "temperature": 0.0, "max_tokens": 256, "timeout": 180,
    }))

    benchmark = SVGEditBench(exp_cfg["dataset"]["root"])
    n_annotated = 0
    t_phase1 = time.time()
    for case in benchmark.iter_cases(tasks=exp_cfg["dataset"]["tasks"], limit_per_task=limit_per_task):
        scene = build_scene(case.source_svg)
        annotator.annotate(case.source_svg, scene, vision_model, request_id=case.case_id)
        n_annotated += 1
        if n_annotated % 25 == 0:
            elapsed = time.time() - t_phase1
            rate = elapsed / n_annotated
            remaining = (limit_per_task * 5 - n_annotated) * rate
            print(f"  {n_annotated} cases annotated, {elapsed:.0f}s elapsed, "
                  f"~{remaining/60:.1f} min remaining in phase 1")

    print(f"Phase 1 complete: {n_annotated} cases in {(time.time()-t_phase1)/60:.1f} min")
    vision_proc.terminate()
    vision_proc.wait(timeout=30)
    repo_volume.commit()

    if skip_phase2:
        # GNN training only needs the warm vision_cache/, not the text-model
        # CLI eval -- skipping phase 2 saves ~2.5h (observed ~30s/case text
        # eval). Cache is already committed above and pullable via
        # `modal volume get svgpatchlab-repo-and-cache EditSVG-patch-lab/runs/vision_cache ./local_dir`
        print("skip_phase2=True: stopping after phase 1. vision_cache/ is committed to the volume.")
        return {"phase1_cases_annotated": n_annotated, "phase2_skipped": True}

    # ===================== PHASE 2: real CLI eval (reuses proven path) =====================
    print()
    print("=" * 65)
    print(" PHASE 2 - booting text server, running the real CLI eval")
    print("=" * 65)
    text_proc = boot_server("Qwen/Qwen3.5-4B", 8000,
                             ["--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.9",
                              "--max-model-len", "14336"],
                             "vllm_text_server.log")

    result = subprocess.run([
        "python3", "-m", "svgpatchlab.cli", "evaluate",
        "--config", str(config_path), "--limit-per-task", str(limit_per_task), "--render",
    ])
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed (exit code {result.returncode})")

    text_proc.terminate()
    summary_path = Path("runs/skeleton_patch_vision_modal/summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    repo_volume.commit()
    return summary


@app.local_entrypoint()
def main(limit_per_task: int = DEFAULT_LIMIT_PER_TASK, skip_phase2: bool = False):
    import json
    summary = run_two_phase_eval.remote(limit_per_task=limit_per_task, skip_phase2=skip_phase2)
    print(json.dumps(summary, indent=2))
