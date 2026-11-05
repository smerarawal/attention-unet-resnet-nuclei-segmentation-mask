"""All figures. Every function saves a PNG into <out_dir>/figures and shows it inline in notebooks."""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

from .data.monusac import CLASS_NAMES, IGNORE

PALETTE = np.array([(0, 0, 0), (230, 57, 70), (42, 157, 143), (244, 162, 97), (69, 123, 157)], np.uint8)
CLS_COLORS = [tuple(c / 255 for c in col) for col in PALETTE]


def _finish(fig, out_dir, name, show=True):
    path = os.path.join(out_dir, "figures", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    print(f"[viz] saved {path}")
    return path


def class_legend(ax, loc="lower right", classes=range(1, 5)):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CLS_COLORS[c], label=CLASS_NAMES[c]) for c in classes], loc=loc, fontsize=7, framealpha=0.8)


# ---------------------------------------------------------------------------- overlays
def overlay_instances(img, inst, cls, fill=0.25):
    out = img.astype(np.float32).copy()
    m = inst > 0
    col = PALETTE[cls[inst]].astype(np.float32)
    out[m] = (1 - fill) * out[m] + fill * col[m]
    b = find_boundaries(inst, mode="inner") & m
    out[b] = col[b]
    return out.astype(np.uint8)


def overlay_sem(img, sem, alpha=0.45):
    out = img.astype(np.float32).copy()
    for c in range(1, len(PALETTE)):
        m = sem == c
        out[m] = (1 - alpha) * out[m] + alpha * PALETTE[c]
    ig = sem == IGNORE
    out[ig] = 0.6 * out[ig] + 0.4 * 128
    return out.astype(np.uint8)


def error_map(gt_sem, pr_sem):
    """green=correct class, yellow=class confusion, red=false positive, blue=missed, grey=ignored."""
    h, w = gt_sem.shape
    out = np.full((h, w, 3), 255, np.uint8)
    valid = gt_sem != IGNORE
    gfg, pfg = (gt_sem > 0) & valid, (pr_sem > 0) & valid
    out[gfg & pfg & (gt_sem == pr_sem)] = (60, 180, 75)
    out[gfg & pfg & (gt_sem != pr_sem)] = (255, 225, 25)
    out[~gfg & pfg] = (230, 25, 75)
    out[gfg & ~pfg] = (0, 130, 200)
    out[~valid] = (190, 190, 190)
    return out


# ---------------------------------------------------------------------------- training
def plot_class_weights(freq, cw, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(CLASS_NAMES, freq * 100, color=CLS_COLORS)
    ax[0].set_yscale("log"); ax[0].set_title("Train pixel share (%)")
    ax[1].bar(CLASS_NAMES, np.asarray(cw), color=CLS_COLORS); ax[1].set_title("CE class weights")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _finish(fig, out_dir, "train_class_weights.png")


def plot_training_curves(hist, out_dir):
    import pandas as pd
    df = pd.DataFrame(hist)
    ep = df["epoch"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    ax[0, 0].plot(ep, df["train_loss"], label="train"); ax[0, 0].plot(ep, df["val_loss"], label="val")
    ax[0, 0].set_title("Total loss"); ax[0, 0].legend()
    for k in ("ce", "dice", "bnd"):
        ax[0, 1].plot(ep, df[f"train_{k}"], label=k)
    ax[0, 1].set_title("Train loss components"); ax[0, 1].legend()
    ax[0, 2].plot(ep, df["val_mean_fg_dice"], label="mean fg Dice (class-avg)")
    ax[0, 2].plot(ep, df["val_bin_dice"], label="binary fg Dice")
    b = df["val_mean_fg_dice"].idxmax()
    ax[0, 2].axvline(ep[b], ls="--", c="gray"); ax[0, 2].set_title(f"Val Dice (best ep {ep[b]})"); ax[0, 2].legend()
    for c in range(1, 5):
        if f"val_dice_{c}" in df:
            ax[1, 0].plot(ep, df[f"val_dice_{c}"], color=CLS_COLORS[c], label=CLASS_NAMES[c])
    ax[1, 0].set_title("Val Dice per class"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(ep, df["lr"]); ax[1, 1].set_title("Learning rate (decoder)")
    ax[1, 2].bar(ep, df["sec"], color="gray"); ax[1, 2].set_title("Seconds / epoch")
    for a in ax.ravel():
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.tight_layout()
    return _finish(fig, out_dir, "train_curves.png")


def plot_aug_preview(ds, sample_idx, out_dir, n=8):
    from .data.dataset import denorm
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for k in range(n):
        x, sem, bnd = ds[sample_idx]
        img = denorm(x)
        axes[0, k].imshow(overlay_sem(img, sem.numpy().astype(np.uint8)))
        axes[1, k].imshow(bnd.numpy(), cmap="gray")
        axes[0, k].axis("off"); axes[1, k].axis("off")
    axes[0, 0].set_title("augmented crop + class overlay", fontsize=8, loc="left")
    axes[1, 0].set_title("boundary target", fontsize=8, loc="left")
    fig.tight_layout()
    return _finish(fig, out_dir, "train_aug_preview.png")


# ---------------------------------------------------------------------------- evaluation
def plot_confusion(cm, out_dir, name="eval_confusion.png"):
    import seaborn as sns
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax[0], vmin=0, vmax=1)
    ax[0].set_title("Pixel confusion (row-normalised: recall)"); ax[0].set_xlabel("pred"); ax[0].set_ylabel("GT")
    fgm = cm[1:, 1:]
    sns.heatmap(fgm / np.maximum(fgm.sum(0, keepdims=True), 1), annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=CLASS_NAMES[1:], yticklabels=CLASS_NAMES[1:], ax=ax[1], vmin=0, vmax=1)
    ax[1].set_title("Foreground only (column-normalised: precision)"); ax[1].set_xlabel("pred"); ax[1].set_ylabel("GT")
    fig.tight_layout()
    return _finish(fig, out_dir, name)


def plot_class_metrics(dice, iou, inst_summary, out_dir, name="eval_class_metrics.png"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    x = np.arange(1, 5)
    ax[0].bar(x - 0.2, dice[1:], 0.4, label="Dice", color="#457b9d"); ax[0].bar(x + 0.2, iou[1:], 0.4, label="IoU", color="#a8dadc")
    ax[0].set_xticks(x); ax[0].set_xticklabels(CLASS_NAMES[1:], rotation=20); ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].set_title("Pixel metrics")
    for k, key in enumerate(("DQ", "SQ", "PQ")):
        vals = [inst_summary.get(f"{key}_{CLASS_NAMES[c]}", np.nan) for c in x]
        ax[1].bar(x + (k - 1) * 0.27, np.nan_to_num(vals), 0.27, label=key)
    ax[1].set_xticks(x); ax[1].set_xticklabels(CLASS_NAMES[1:], rotation=20); ax[1].set_ylim(0, 1); ax[1].legend()
    ax[1].set_title(f"Instance metrics (mPQ={inst_summary.get('mPQ', float('nan')):.3f})")
    ax[2].axis("off")
    ax[2].text(0, 0.9, f"AJI  {inst_summary['AJI']:.4f}\nmPQ  {inst_summary['mPQ']:.4f}", fontsize=16, va="top", family="monospace")
    fig.tight_layout()
    return _finish(fig, out_dir, name)


def plot_image_metrics(df, out_dir, name="eval_per_image.png"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].hist(df["aji"].dropna(), bins=20, color="#2a9d8f"); ax[0].set_title("Per-image AJI"); ax[0].axvline(df["aji"].median(), c="k", ls="--")
    ax[1].scatter(df["n_gt"], df["aji"], s=14, alpha=0.7); ax[1].set_xlabel("# GT nuclei"); ax[1].set_ylabel("AJI"); ax[1].set_title("Crowding vs AJI")
    mx = max(df["n_gt"].max(), df["n_pred"].max(), 1)
    ax[2].scatter(df["n_gt"], df["n_pred"], s=14, alpha=0.7); ax[2].plot([0, mx], [0, mx], "k--", lw=1)
    ax[2].set_xlabel("# GT"); ax[2].set_ylabel("# predicted"); ax[2].set_title("Count agreement")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return _finish(fig, out_dir, name)


def plot_pred_panels(items, out_dir, name, title=""):
    """items: dicts with image, gt_inst, gt_cls, gt_sem, pr_inst, pr_cls, pr_sem, bnd, name, aji."""
    n = len(items)
    fig, ax = plt.subplots(n, 5, figsize=(17, 3.5 * n), squeeze=False)
    heads = ["image", "GT instances (class colours)", "Pred instances", "semantic errors", "boundary prob"]
    for r, it in enumerate(items):
        ax[r, 0].imshow(it["image"])
        ax[r, 1].imshow(overlay_instances(it["image"], it["gt_inst"], it["gt_cls"]))
        ax[r, 2].imshow(overlay_instances(it["image"], it["pr_inst"], it["pr_cls"]))
        ax[r, 3].imshow(error_map(it["gt_sem"], it["pr_sem"]))
        ax[r, 4].imshow(it["bnd"], cmap="magma", vmin=0, vmax=1)
        ax[r, 0].set_ylabel(f"{it['name'][-24:]}\nAJI {it['aji']:.3f}", fontsize=8)
        for c in range(5):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(heads[c], fontsize=9)
    class_legend(ax[0, 1], "upper right")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return _finish(fig, out_dir, name)


def plot_attention(image, maps, out_dir, name="eval_attention.png"):
    """maps: {'i_j': (h,w) np psi}. Shows the full-resolution level-0 gates + deepest level-3 gate."""
    keys = [k for k in ("0_1", "0_2", "0_3", "0_4", "1_1", "2_1", "3_1") if k in maps]
    fig, ax = plt.subplots(1, len(keys) + 1, figsize=(2.6 * (len(keys) + 1), 3))
    ax[0].imshow(image); ax[0].set_title("image", fontsize=9); ax[0].axis("off")
    for a, k in zip(ax[1:], keys):
        a.imshow(image); a.imshow(maps[k], cmap="jet", alpha=0.55, vmin=0, vmax=1)
        a.set_title(f"gate x[{k[0]}][{k[2]}]", fontsize=9); a.axis("off")
    fig.tight_layout()
    return _finish(fig, out_dir, name)


def plot_postproc_grid(results, out_dir, name="eval_postproc_tuning.png"):
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(results, columns=["fg_thr", "bnd_thr", "min_area", "aji"])
    ms = sorted(df.min_area.unique())
    fig, ax = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 3.6), squeeze=False)
    for a, m in zip(ax[0], ms):
        sub = df[df.min_area == m].pivot(index="fg_thr", columns="bnd_thr", values="aji")
        sns.heatmap(sub, annot=True, fmt=".3f", cmap="viridis", ax=a, vmin=df.aji.min(), vmax=df.aji.max())
        a.set_title(f"val AJI | min_area={m}")
    fig.tight_layout()
    return _finish(fig, out_dir, name)
