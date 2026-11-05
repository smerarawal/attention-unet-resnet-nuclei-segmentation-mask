"""prepare -> train -> evaluate. Everything driven by Config."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import viz
from .config import Config
from .data import CLASS_NAMES, EvalDS, TrainDS, denorm, describe, discover, find_data_root, load_all, make_splits, save_splits
from .losses import SegLoss, compute_class_weights
from .metrics import ConfMat, InstanceMeter
from .models import build_model
from .postprocess import instances_from_probs, mask_ignore, tune_postproc
from .utils import Timer, autocast, get_device, make_scaler, nanmean, save_json, seed_everything, worker_init_fn


# ------------------------------------------------------------------------------ data
def prepare(cfg: Config, need=("train", "val", "test")):
    root = find_data_root(cfg.data_root)
    records = discover(root)
    splits = make_splits(records, cfg.val_frac, cfg.test_frac, cfg.seed)
    describe(splits)
    save_splits(splits, os.path.join(cfg.out_dir, "splits.json"))
    return {k: load_all(splits[k], cfg.cache_dir, cfg.boundary_width, desc=f"cache/{k}") for k in need if splits[k]}


def make_loaders(cfg, data):
    tr = TrainDS(data["train"], cfg.crop, cfg.crops_per_image, cfg.fg_crop_p, cfg.hed_jitter_p)
    dl_tr = DataLoader(tr, cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers,
                       worker_init_fn=worker_init_fn, pin_memory=torch.cuda.is_available(),
                       persistent_workers=cfg.num_workers > 0)
    dl_va = DataLoader(EvalDS(data["val"]), 1, shuffle=False, num_workers=0)
    return tr, dl_tr, dl_va


# ------------------------------------------------------------------------------ inference
def _starts(L, t, step):
    if L <= t:
        return [0]
    s = list(range(0, L - t + 1, step))
    return s if s[-1] == L - t else s + [L - t]


@torch.no_grad()
def forward_probs(model, x, cfg, device, tta=False):
    """x: (1,3,H,W) normalised. Returns probs (C,H,W), boundary prob (H,W) as float32 tensors on CPU."""
    _, _, H, W = x.shape
    th, tw = min(cfg.tile, H), min(cfg.tile, W)
    step = max(32, cfg.tile - cfg.tile_overlap)

    def run(xb):
        with autocast(device, cfg.amp):
            s, b = model(xb)
        return s.float().softmax(1), b.float().sigmoid()

    def tiled(xb):
        if H <= cfg.tile and W <= cfg.tile:
            return run(xb)
        P = torch.zeros(1, cfg.n_classes, H, W, device=xb.device)
        B = torch.zeros(1, 1, H, W, device=xb.device)
        N = torch.zeros(1, 1, H, W, device=xb.device)
        for y in _starts(H, th, step):
            for xx in _starts(W, tw, step):
                p, b = run(xb[:, :, y:y + th, xx:xx + tw])
                P[:, :, y:y + th, xx:xx + tw] += p
                B[:, :, y:y + th, xx:xx + tw] += b
                N[:, :, y:y + th, xx:xx + tw] += 1
        return P / N, B / N

    flips = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
    P = B = 0
    for f in flips:
        xb = x.flip(f) if f else x
        p, b = tiled(xb)
        P = P + (p.flip(f) if f else p)
        B = B + (b.flip(f) if f else b)
    return (P / len(flips))[0].cpu(), (B / len(flips))[0, 0].cpu()


@torch.no_grad()
def validate(model, loader, loss_fn, cfg, device):
    model.eval()
    cm, losses = ConfMat(cfg.n_classes), []
    for x, sem, bnd, meta in loader:
        x, sem, bnd = x.to(device), sem.to(device), bnd.to(device)
        with autocast(device, cfg.amp):
            s, b = model(x)
        loss, _ = loss_fn(s, b, sem, bnd)
        losses.append(loss.item())
        cm.update(s.argmax(1)[0].cpu().numpy(), sem[0].cpu().numpy())
    dice, _ = cm.per_class()
    return float(np.mean(losses)), cm, dice


# ------------------------------------------------------------------------------ training
def _optimizer(cfg, model):
    if cfg.encoder == "resnet34" and cfg.pretrained:
        enc = list(model.encoder.parameters())
        ids = {id(p) for p in enc}
        groups = [{"params": enc, "lr": cfg.lr * cfg.encoder_lr_mult},
                  {"params": [p for p in model.parameters() if id(p) not in ids], "lr": cfg.lr}]
        return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay), [g["lr"] for g in groups]
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay), [cfg.lr]


def train(cfg: Config, data=None):
    cfg.resolve()
    seed_everything(cfg.seed)
    device = get_device()
    print(f"[train] device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    data = data or prepare(cfg)
    save_json(cfg.to_dict(), os.path.join(cfg.out_dir, "config.json"))

    cw, freq = compute_class_weights(data["train"], cfg.n_classes)
    print("[train] class pixel share:", dict(zip(CLASS_NAMES, np.round(freq, 4))), "| weights:", np.round(cw.numpy(), 2))
    viz.plot_class_weights(freq, cw.numpy(), cfg.out_dir)

    ds_tr, dl_tr, dl_va = make_loaders(cfg, data)
    viz.plot_aug_preview(ds_tr, 0, cfg.out_dir)

    model = build_model(cfg).to(device)
    loss_fn = SegLoss(cw.to(device), cfg.w_ce, cfg.w_dice, cfg.w_bnd).to(device)
    opt, max_lrs = _optimizer(cfg, model)
    total_steps = cfg.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lrs, total_steps=total_steps,
                                                pct_start=min(0.5, max(0.1, 3.0 / total_steps)))
    scaler = make_scaler(cfg.amp and device.type == "cuda")

    ck_last, ck_best = os.path.join(cfg.out_dir, "last.pt"), os.path.join(cfg.out_dir, "best.pt")
    hist, start, best, bad = [], 0, -1.0, 0
    if cfg.resume and os.path.exists(ck_last):
        ck = torch.load(ck_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); hist, start, best, bad = ck["hist"], ck["epoch"], ck["best"], ck["bad"]
        print(f"[train] resumed from epoch {start}")

    timer = Timer()
    for ep in range(start, cfg.epochs):
        model.train()
        t0 = time.time()
        agg = {"loss": [], "ce": [], "dice": [], "bnd": []}
        pbar = tqdm(dl_tr, desc=f"ep {ep + 1}/{cfg.epochs}", leave=False)
        for x, sem, bnd in pbar:
            x, sem, bnd = x.to(device, non_blocking=True), sem.to(device, non_blocking=True), bnd.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device, cfg.amp):
                s, b = model(x)
            loss, parts = loss_fn(s, b, sem, bnd)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            agg["loss"].append(loss.item())
            for k, v in parts.items():
                agg[k].append(v)
            pbar.set_postfix(loss=f"{np.mean(agg['loss']):.3f}")

        vloss, cm, vdice = validate(model, dl_va, loss_fn, cfg, device)
        mfd, (bd, bi) = cm.mean_fg_dice(), cm.binary_fg()
        row = {"epoch": ep + 1, "train_loss": np.mean(agg["loss"]), "train_ce": np.mean(agg["ce"]),
               "train_dice": np.mean(agg["dice"]), "train_bnd": np.mean(agg["bnd"]), "val_loss": vloss,
               "val_mean_fg_dice": mfd, "val_bin_dice": bd, "val_bin_iou": bi,
               "lr": opt.param_groups[-1]["lr"], "sec": time.time() - t0}
        row.update({f"val_dice_{c}": vdice[c] for c in range(1, cfg.n_classes)})
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(cfg.out_dir, "history.csv"), index=False)

        score = -1.0 if np.isnan(mfd) else mfd
        improved = score > best
        if improved:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": ep + 1, "val_mean_fg_dice": mfd}, ck_best)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "hist": hist, "epoch": ep + 1, "best": best, "bad": bad}, ck_last)
        print(f"ep {ep + 1:3d} | loss {row['train_loss']:.3f}/{vloss:.3f} | val mFgDice {mfd:.4f} binDice {bd:.4f} "
              f"| per-class {np.round(vdice[1:], 3)} | {row['sec']:.0f}s {'*best*' if improved else ''}")
        if bad >= cfg.patience:
            print(f"[train] early stop (no improvement for {cfg.patience} epochs)")
            break
        if cfg.time_budget_min and timer.minutes > cfg.time_budget_min:
            print(f"[train] time budget hit ({timer.minutes:.0f} min) — stopping; use resume=True to continue")
            break
    viz.plot_training_curves(hist, cfg.out_dir)
    return hist


# ------------------------------------------------------------------------------ evaluation
def load_model(cfg, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = dict(ck["cfg"])
    cfg2 = Config(**{k: v for k, v in c.items() if k in Config.__dataclass_fields__})
    cfg2.pretrained = False  # weights come from the checkpoint
    model = build_model(cfg2).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


@torch.no_grad()
def predict_split(model, samples, cfg, device, tta=True, desc="predict"):
    """-> list of (probs fp16 (C,h,w), bnd fp16 (h,w)) cropped back to original size."""
    out = []
    for x, sem, bnd, meta in tqdm(DataLoader(EvalDS(samples), 1, shuffle=False), desc=desc, leave=False):
        i, h, w = meta[0].tolist()
        p, b = forward_probs(model, x.to(device), cfg, device, tta)
        out.append((p[:, :h, :w].numpy().astype(np.float16), b[:h, :w].numpy().astype(np.float16)))
    return out


def evaluate(cfg: Config, split="test", ckpt="best.pt", data=None):
    cfg.resolve()
    device = get_device()
    data = data or prepare(cfg)
    if split not in data:
        raise ValueError(f"split '{split}' is empty; available: {list(data)}")
    model, ck = load_model(cfg, os.path.join(cfg.out_dir, ckpt), device)
    print(f"[eval] checkpoint epoch {ck['epoch']} (val mFgDice {ck['val_mean_fg_dice']:.4f})")

    pp = dict(fg_thr=cfg.fg_thr, bnd_thr=cfg.bnd_thr, min_area=cfg.min_area)
    if cfg.tune_postproc and "val" in data:
        vp = predict_split(model, data["val"], cfg, device, cfg.tta, "val preds (tuning)")
        pp, grid = tune_postproc(vp, data["val"], cfg.n_classes)
        viz.plot_postproc_grid(grid, cfg.out_dir)
        del vp
    save_json(pp, os.path.join(cfg.out_dir, "postproc.json"))

    samples = data[split]
    preds = predict_split(model, samples, cfg, device, cfg.tta, f"{split} preds")
    cm, meter, store = ConfMat(cfg.n_classes), InstanceMeter(cfg.n_classes), []
    for (p, b), d in zip(preds, samples):
        p32, b32 = p.astype(np.float32), b.astype(np.float32)
        pr_sem = p32.argmax(0).astype(np.uint8)
        cm.update(pr_sem, d["sem"])
        inst, cls = instances_from_probs(p32, b32, **pp)
        inst, cls = mask_ignore(inst, cls, d["sem"])
        meter.update(f"{d['patient']}/{d['stem']}", d["inst"], d["inst_class"], inst, cls)
        store.append(dict(pr_sem=pr_sem, pr_inst=inst, pr_cls=cls, bnd=b))

    dice, iou = cm.per_class()
    bd, bi = cm.binary_fg()
    summ = meter.summary(CLASS_NAMES)
    res = {"split": split, "n_images": len(samples), "binary_fg_dice": bd, "binary_fg_iou": bi,
           "mean_fg_dice": cm.mean_fg_dice(), "mean_fg_iou": nanmean(iou[1:]),
           **{f"dice_{CLASS_NAMES[c]}": dice[c] for c in range(cfg.n_classes)},
           **{f"iou_{CLASS_NAMES[c]}": iou[c] for c in range(cfg.n_classes)}, **summ, "postproc": pp}
    save_json(res, os.path.join(cfg.out_dir, f"results_{split}.json"))
    df = pd.DataFrame(meter.rows)
    df.to_csv(os.path.join(cfg.out_dir, f"per_image_{split}.csv"), index=False)
    print(f"[eval/{split}] binary Dice {bd:.4f} IoU {bi:.4f} | mean-fg Dice {res['mean_fg_dice']:.4f} | "
          f"AJI {summ['AJI']:.4f} | mPQ {summ['mPQ']:.4f}")
    print(pd.DataFrame({"dice": dice, "iou": iou}, index=CLASS_NAMES).round(4))

    viz.plot_confusion(cm.m, cfg.out_dir, f"eval_{split}_confusion.png")
    viz.plot_class_metrics(dice, iou, summ, cfg.out_dir, f"eval_{split}_class_metrics.png")
    viz.plot_image_metrics(df, cfg.out_dir, f"eval_{split}_per_image.png")

    order = np.argsort(df["aji"].fillna(-1).values)
    pick = {"worst": order[:3], "median": order[len(order) // 2 - 1:len(order) // 2 + 2], "best": order[-3:]}
    for tag, idxs in pick.items():
        items = []
        for i in idxs:
            d = samples[int(i)]
            items.append(dict(image=d["image"], gt_inst=d["inst"], gt_cls=d["inst_class"], gt_sem=d["sem"],
                              name=f"{d['patient']}/{d['stem']}", aji=float(df["aji"].iloc[int(i)]), **store[int(i)]))
        viz.plot_pred_panels(items, cfg.out_dir, f"eval_{split}_panels_{tag}.png", f"{split}: {tag} images by AJI")

    # attention gates on the median image
    mid = int(order[len(order) // 2])
    x, *_ = EvalDS([samples[mid]])[0]
    model.set_keep_attention(True)
    with torch.no_grad():
        model(x[None].to(device))
    h, w = samples[mid]["sem"].shape
    maps = {k: v[0, 0].float().cpu().numpy() for k, v in model.attention_maps().items()}
    import torch.nn.functional as F
    maps = {k: F.interpolate(torch.from_numpy(v)[None, None], size=(x.shape[1], x.shape[2]), mode="bilinear")[0, 0].numpy()[:h, :w]
            for k, v in maps.items()}
    model.set_keep_attention(False)
    viz.plot_attention(samples[mid]["image"], maps, cfg.out_dir, f"eval_{split}_attention.png")
    return res
