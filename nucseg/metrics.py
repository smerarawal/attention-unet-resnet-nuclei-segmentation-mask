"""Pixel metrics (confusion matrix) + instance metrics (AJI, PQ/DQ/SQ, per class)."""
import numpy as np
from skimage.segmentation import relabel_sequential

from .data.monusac import IGNORE


class ConfMat:
    def __init__(self, n):
        self.n = n
        self.m = np.zeros((n, n), np.int64)  # rows = GT, cols = pred

    def update(self, pred, target):
        k = target != IGNORE
        self.m += np.bincount(target[k].astype(np.int64) * self.n + pred[k], minlength=self.n ** 2).reshape(self.n, self.n)

    def per_class(self):
        tp = np.diag(self.m).astype(float)
        gt, pr = self.m.sum(1), self.m.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            dice = np.where(gt > 0, 2 * tp / (gt + pr), np.nan)
            iou = np.where(gt > 0, tp / (gt + pr - tp), np.nan)
        return dice, iou

    def mean_fg_dice(self):
        d, _ = self.per_class()
        d = d[1:]
        return float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")

    def binary_fg(self):
        """Class-agnostic foreground Dice/IoU (what your original notebook measured)."""
        m = self.m
        tp = m[1:, 1:].sum()
        gt, pr = m[1:].sum(), m[:, 1:].sum()
        return 2 * tp / max(gt + pr, 1), tp / max(gt + pr - tp, 1)


def _contingency(gt, pr):
    gt, _, _ = relabel_sequential(gt)
    pr, _, _ = relabel_sequential(pr)
    ng, npr = int(gt.max()), int(pr.max())
    inter = np.bincount(gt.ravel().astype(np.int64) * (npr + 1) + pr.ravel(), minlength=(ng + 1) * (npr + 1)).reshape(ng + 1, npr + 1)
    return inter, ng, npr


def aji(gt, pr):
    """Aggregated Jaccard Index (Kumar et al. 2017). Returns (num, den) so datasets can aggregate."""
    inter, ng, npr = _contingency(gt, pr)
    if ng == 0:
        return 0.0, float(inter[0, 1:].sum())
    ag, ap = inter.sum(1), inter.sum(0)
    union = ag[:, None] + ap[None, :] - inter
    num = den = 0.0
    used = np.zeros(npr + 1, bool)
    for g in range(1, ng + 1):
        if npr == 0:
            den += ag[g]
            continue
        iou = inter[g, 1:] / np.maximum(union[g, 1:], 1)
        p = int(iou.argmax()) + 1
        if iou[p - 1] > 0:
            num += inter[g, p]
            den += union[g, p]
            used[p] = True
        else:
            den += ag[g]
    den += ap[1:][~used[1:]].sum()
    return float(num), float(den)


def pq_counts(gt, pr):
    """Matches at IoU>0.5 (unique by construction). Returns tp, fp, fn, sum of matched IoU."""
    inter, ng, npr = _contingency(gt, pr)
    if ng == 0 or npr == 0:
        return 0, npr, ng, 0.0
    ag, ap = inter.sum(1), inter.sum(0)
    union = ag[:, None] + ap[None, :] - inter
    iou = inter[1:, 1:] / np.maximum(union[1:, 1:], 1)
    m = iou > 0.5
    tp = int(m.sum())
    return tp, npr - tp, ng - tp, float(iou[m].sum())


def pq_from_sums(tp, fp, fn, siou):
    dq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-9) if (tp + fp + fn) else float("nan")
    sq = siou / tp if tp else (0.0 if (fp + fn) else float("nan"))
    return dq, sq, (dq * sq if np.isfinite(dq) and np.isfinite(sq) else float("nan"))


def class_instance_map(inst, inst_class, c):
    """Keep only instances of class c (others -> 0)."""
    keep = np.zeros(len(inst_class) + 1, bool)
    keep[np.flatnonzero(inst_class == c)] = True
    return np.where(keep[inst], inst, 0)


class InstanceMeter:
    """Accumulates dataset-level AJI and per-class PQ/DQ/SQ plus per-image rows."""

    def __init__(self, n_classes=5):
        self.n = n_classes
        self.aji_num = self.aji_den = 0.0
        self.pq = {c: np.zeros(4) for c in range(1, n_classes)}
        self.rows = []

    def update(self, name, gt_inst, gt_cls, pr_inst, pr_cls):
        num, den = aji(gt_inst, pr_inst)
        self.aji_num += num
        self.aji_den += den
        row = {"image": name, "aji": num / den if den else np.nan,
               "n_gt": int((gt_cls > 0).sum()), "n_pred": int((pr_cls > 0).sum())}
        tot = np.zeros(4)
        for c in range(1, self.n):
            cnt = np.array(pq_counts(class_instance_map(gt_inst, gt_cls, c), class_instance_map(pr_inst, pr_cls, c)), float)
            self.pq[c] += cnt
            tot += cnt
        row["pq"] = pq_from_sums(*tot)[2]
        self.rows.append(row)

    def summary(self, class_names):
        out = {"AJI": self.aji_num / self.aji_den if self.aji_den else float("nan")}
        pqs = []
        for c in range(1, self.n):
            dq, sq, pq = pq_from_sums(*self.pq[c])
            out[f"DQ_{class_names[c]}"], out[f"SQ_{class_names[c]}"], out[f"PQ_{class_names[c]}"] = dq, sq, pq
            pqs.append(pq)
        out["mPQ"] = float(np.nanmean(pqs)) if np.isfinite(pqs).any() else float("nan")
        return out
