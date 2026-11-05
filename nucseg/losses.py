import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data.monusac import IGNORE


def compute_class_weights(samples, n_classes=5, cap=10.0):
    """ENet-style 1/log(1.02 + freq), normalised to mean 1 and capped. Ignore pixels excluded."""
    cnt = np.zeros(n_classes)
    for d in samples:
        cnt += np.bincount(d["sem"][d["sem"] != IGNORE].ravel(), minlength=n_classes)[:n_classes]
    freq = cnt / max(cnt.sum(), 1)
    w = 1.0 / np.log(1.02 + freq)
    w = np.minimum(w / w.mean(), cap)
    return torch.tensor(w, dtype=torch.float32), freq


def soft_dice_multiclass(logits, target, ignore=IGNORE, smooth=1.0):
    C = logits.shape[1]
    valid = (target != ignore)
    t = target.clone()
    t[~valid] = 0
    onehot = F.one_hot(t, C).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    p = logits.softmax(1) * valid.unsqueeze(1)
    inter = (p * onehot).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + smooth) / (denom + smooth)).mean()


class SegLoss(nn.Module):
    """CE(class-weighted) + multiclass soft Dice + boundary BCE/Dice. All masked by the ignore label."""

    def __init__(self, class_weights, w_ce=1.0, w_dice=1.0, w_bnd=0.5):
        super().__init__()
        self.register_buffer("cw", class_weights)
        self.w = (w_ce, w_dice, w_bnd)

    def forward(self, sem_logits, bnd_logits, sem_t, bnd_t):
        sem_logits, bnd_logits = sem_logits.float(), bnd_logits.float().squeeze(1)
        ce = F.cross_entropy(sem_logits, sem_t, weight=self.cw, ignore_index=IGNORE)
        dice = soft_dice_multiclass(sem_logits, sem_t)
        valid = (sem_t != IGNORE).float()
        bce = (F.binary_cross_entropy_with_logits(bnd_logits, bnd_t, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
        p = bnd_logits.sigmoid() * valid
        bt = bnd_t * valid
        bdice = 1 - (2 * (p * bt).sum() + 1) / (p.sum() + bt.sum() + 1)
        total = self.w[0] * ce + self.w[1] * dice + self.w[2] * (bce + bdice)
        return total, {"ce": ce.item(), "dice": dice.item(), "bnd": (bce + bdice).item()}
