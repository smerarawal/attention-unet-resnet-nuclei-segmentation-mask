"""Semantic probs + boundary prob -> instance map + per-instance class (marker-controlled watershed)."""
import itertools

import numpy as np
import scipy.ndimage as ndi
from skimage.measure import label
from skimage.morphology import disk
from skimage.segmentation import relabel_sequential, watershed

from .data.monusac import IGNORE
from .metrics import InstanceMeter


def instances_from_probs(probs, bnd, fg_thr=0.5, bnd_thr=0.5, min_area=20):
    """probs: (C,H,W) softmax; bnd: (H,W) boundary prob. Returns (inst int32, inst_class uint8 indexed by id)."""
    fg = (1.0 - probs[0]) > fg_thr
    interior = ndi.binary_opening(fg & (bnd < bnd_thr), structure=disk(1))  # scipy: stable across skimage versions
    markers = label(interior)
    dist = ndi.distance_transform_edt(fg)
    inst = watershed(-dist, markers, mask=fg)
    orphan = fg & (inst == 0)  # fg components that never received a marker
    if orphan.any():
        o = label(orphan)
        inst = np.where(orphan, o + inst.max(), inst)
    if inst.max() > 0:
        areas = np.bincount(inst.ravel())
        small = areas < min_area
        small[0] = False
        inst[small[inst]] = 0
        inst, _, _ = relabel_sequential(inst)
    inst = inst.astype(np.int32)
    n = int(inst.max())
    cls = np.zeros(n + 1, np.uint8)
    if n:
        ids = np.arange(1, n + 1)
        votes = np.stack([ndi.sum(probs[c], inst, ids) for c in range(1, probs.shape[0])], 1)
        cls[1:] = votes.argmax(1) + 1
    return inst, cls


def mask_ignore(inst, cls, sem):
    """Drop predicted pixels that fall in ambiguous/ignore GT regions (not penalised, as in the challenge)."""
    inst = np.where(sem == IGNORE, 0, inst)
    return inst, cls


def tune_postproc(preds, samples, n_classes, grid=None, log=print):
    """Grid-search (fg_thr, bnd_thr, min_area) on a split by dataset AJI. preds[i] = (probs fp16, bnd fp16)."""
    grid = grid or dict(fg_thr=[0.4, 0.5, 0.6], bnd_thr=[0.3, 0.4, 0.5, 0.6, 0.7], min_area=[10, 20, 40])
    best, best_aji, results = None, -1, []
    for fg, bt, ma in itertools.product(grid["fg_thr"], grid["bnd_thr"], grid["min_area"]):
        num = den = 0.0
        from .metrics import aji
        for (p, b), d in zip(preds, samples):
            inst, cls = instances_from_probs(p.astype(np.float32), b.astype(np.float32), fg, bt, ma)
            inst, _ = mask_ignore(inst, cls, d["sem"])
            n_, d_ = aji(d["inst"], inst)
            num += n_
            den += d_
        score = num / den if den else 0
        results.append((fg, bt, ma, score))
        if score > best_aji:
            best, best_aji = dict(fg_thr=fg, bnd_thr=bt, min_area=ma), score
    log(f"[postproc] best {best} -> val AJI {best_aji:.4f}")
    return best, results
