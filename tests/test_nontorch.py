"""Runs the torch-free logic (parsing, splits, metrics, post-processing, EDA) — stubs torch if absent."""
import sys, types, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import torch  # noqa
except ImportError:
    class _T: pass
    tm = types.ModuleType("torch"); tm.Tensor = _T
    tn = types.ModuleType("torch.utils"); td = types.ModuleType("torch.utils.data"); td.Dataset = object; td.DataLoader = object
    sys.modules.update({"torch": tm, "torch.utils": tn, "torch.utils.data": td, "torch.nn": types.ModuleType("torch.nn"),
                        "torch.nn.functional": types.ModuleType("torch.nn.functional")})
    tm.nn = sys.modules["torch.nn"]; tm.nn.Module = object; tm.nn.functional = sys.modules["torch.nn.functional"]
import numpy as np
import matplotlib; matplotlib.use("Agg")
from tests.make_synthetic import make
from nucseg.data.monusac import discover, load_all, CLASS_NAMES
from nucseg.data.splits import make_splits, describe
from nucseg.metrics import ConfMat, InstanceMeter
from nucseg.postprocess import instances_from_probs, mask_ignore, tune_postproc

root = "/tmp/fake_monusac_t"
import shutil; shutil.rmtree(root, ignore_errors=True)
make(root, n_patients=8, imgs_per=2, test_patients=2)
recs = discover(root); print(len(recs), "records; origins:", {r.origin for r in recs})
sp = make_splits(recs, 0.2, 0.2, 0); describe(sp)
assert all(r.origin == "test" for r in sp["test"]) and sp["test"] and sp["val"] and sp["train"]
data = {k: load_all(v, "/tmp/fake_cache", 2, True, k) for k, v in sp.items()}
d = data["train"][0]
print("sem uniq", np.unique(d["sem"]), "inst max", d["inst"].max(), "inst_class", d["inst_class"][:8], "bnd frac", d["bnd"].mean().round(3))
assert d["inst"].max() == len(d["inst_class"]) - 1

# oracle predictions built from GT -> metrics must be ~perfect
def oracle(d, noise=False):
    C = 5; p = np.zeros((C,) + d["sem"].shape, np.float32)
    s = d["sem"].copy(); s[s == 255] = 0
    for c in range(C): p[c] = (s == c)
    return p, d["bnd"].astype(np.float32)
cm, meter = ConfMat(5), InstanceMeter(5)
for d in data["val"]:
    p, b = oracle(d); cm.update(p.argmax(0).astype(np.uint8), d["sem"])
    inst, cls = instances_from_probs(p, b, 0.5, 0.5, 5); inst, cls = mask_ignore(inst, cls, d["sem"])
    meter.update(d["stem"], d["inst"], d["inst_class"], inst, cls)
print("oracle pixel dice", np.round(cm.per_class()[0], 3)); s = meter.summary(CLASS_NAMES); print({k: round(v, 3) for k, v in s.items() if k in ("AJI", "mPQ")})
assert cm.mean_fg_dice() > 0.99 and s["AJI"] > 0.9, "oracle should be near perfect (touching nuclei split via boundary)"
best, res = tune_postproc([(oracle(d)[0].astype(np.float16), oracle(d)[1].astype(np.float16)) for d in data["val"]], data["val"], 5,
                          grid=dict(fg_thr=[0.5], bnd_thr=[0.5], min_area=[5, 20]))
from nucseg.config import Config
cfg = Config(out_dir="/tmp/fake_out", cache_dir="/tmp/fake_cache", data_root=root).resolve()
from nucseg.eda import run_eda
img, nuc = run_eda(cfg, data)
print("ALL NON-TORCH TESTS PASSED")
