"""MoNuSAC EDA: tidy tables + figures (replaces the R script for the MoNuSAC part)."""
from __future__ import annotations

import os

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from skimage.measure import regionprops

from . import viz
from .config import Config
from .data import CLASS_NAMES, discover, find_data_root, load_all, make_splits, describe
from .data.monusac import IGNORE

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
CLS_PAL = {CLASS_NAMES[c]: viz.CLS_COLORS[c] for c in range(1, 5)}


def build_tables(split_samples: dict):
    img_rows, nuc_rows = [], []
    for split, samples in split_samples.items():
        for d in samples:
            h, w = d["sem"].shape
            inst, icls = d["inst"], d["inst_class"]
            ids = np.arange(1, len(icls))
            counts = {f"n_{CLASS_NAMES[c]}": int((icls[1:] == c).sum()) for c in range(1, 5)}
            img_rows.append(dict(split=split, patient=d["patient"], image=d["stem"], h=h, w=w, n_nuclei=int((icls > 0).sum()),
                                 fg_frac=float(((d["sem"] > 0) & (d["sem"] != IGNORE)).mean()),
                                 ignore_frac=float((d["sem"] == IGNORE).mean()),
                                 mean_r=float(d["image"][..., 0].mean()), mean_g=float(d["image"][..., 1].mean()),
                                 mean_b=float(d["image"][..., 2].mean()), **counts))
            props = regionprops(inst)
            if not props:
                continue
            cents = np.array([p.centroid for p in props])
            nn = cKDTree(cents).query(cents, k=2)[0][:, 1] if len(cents) > 1 else np.full(len(cents), np.nan)
            for p, dist in zip(props, nn):
                nuc_rows.append(dict(split=split, patient=d["patient"], image=d["stem"], cls=CLASS_NAMES[int(icls[p.label])],
                                     area=p.area, eq_diam=p.equivalent_diameter_area if hasattr(p, "equivalent_diameter_area") else p.equivalent_diameter,
                                     eccentricity=p.eccentricity, solidity=p.solidity, nn_dist=dist))
    return pd.DataFrame(img_rows), pd.DataFrame(nuc_rows)


def run_eda(cfg: Config, split_samples=None):
    cfg.resolve()
    if split_samples is None:
        records = discover(find_data_root(cfg.data_root))
        splits = make_splits(records, cfg.val_frac, cfg.test_frac, cfg.seed)
        describe(splits)
        split_samples = {k: load_all(v, cfg.cache_dir, cfg.boundary_width, desc=f"cache/{k}") for k, v in splits.items() if v}
    img, nuc = build_tables(split_samples)
    img.to_csv(os.path.join(cfg.out_dir, "eda_images.csv"), index=False)
    nuc.to_csv(os.path.join(cfg.out_dir, "eda_nuclei.csv"), index=False)
    out = cfg.out_dir
    cls_names = CLASS_NAMES[1:]

    # 1. splits: size, patients, class mix  (leakage + distribution-shift check)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    g = img.groupby("split")
    g.size().plot.bar(ax=ax[0], color="#457b9d"); ax[0].set_title("Images per split")
    g["patient"].nunique().plot.bar(ax=ax[1], color="#2a9d8f"); ax[1].set_title("Patients per split")
    mix = nuc.groupby(["split", "cls"]).size().unstack(fill_value=0).reindex(columns=cls_names, fill_value=0)
    (mix.div(mix.sum(1), axis=0) * 100).plot.bar(stacked=True, ax=ax[2], color=[CLS_PAL[c] for c in cls_names])
    ax[2].set_title("Nucleus class mix per split (%)"); ax[2].legend(fontsize=7)
    for a in ax:
        a.tick_params(axis="x", rotation=0)
    fig.tight_layout(); viz._finish(fig, out, "eda_splits.png")

    # 2. class balance
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    cnt = nuc["cls"].value_counts().reindex(cls_names).fillna(0)
    ax[0].bar(cls_names, cnt.values, color=[CLS_PAL[c] for c in cls_names]); ax[0].set_yscale("log"); ax[0].set_title("# nuclei per class (log)")
    for i, v in enumerate(cnt.values):
        ax[0].text(i, max(v, 1), f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    area_share = nuc.groupby("cls")["area"].sum().reindex(cls_names).fillna(0)
    ax[1].bar(cls_names, area_share.values / max(area_share.sum(), 1) * 100, color=[CLS_PAL[c] for c in cls_names]); ax[1].set_title("Share of nucleus pixels (%)")
    fig.tight_layout(); viz._finish(fig, out, "eda_class_balance.png")

    # 3. morphology per class
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    pal = [CLS_PAL[c] for c in cls_names]
    sns.violinplot(data=nuc.assign(log10_area=np.log10(nuc["area"].clip(lower=1))), x="cls", y="log10_area", order=cls_names,
                   palette=pal, ax=ax[0, 0], cut=0, inner="quartile")
    ax[0, 0].set_title("log10 area (px²)")
    for c in cls_names:
        sns.kdeplot(nuc.loc[nuc.cls == c, "eq_diam"].dropna(), ax=ax[0, 1], label=c, color=CLS_PAL[c], fill=True, alpha=0.2, cut=0)
    ax[0, 1].set_title("Equivalent diameter (px)"); ax[0, 1].legend(fontsize=8)
    sns.boxplot(data=nuc, x="cls", y="eccentricity", order=cls_names, palette=pal, ax=ax[1, 0], fliersize=1); ax[1, 0].set_title("Eccentricity")
    sns.boxplot(data=nuc, x="cls", y="solidity", order=cls_names, palette=pal, ax=ax[1, 1], fliersize=1); ax[1, 1].set_title("Solidity (low = irregular / merged)")
    fig.tight_layout(); viz._finish(fig, out, "eda_morphology.png")

    # 4. density + crowding
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    order = img.groupby("patient")["n_nuclei"].median().sort_values().index
    sns.stripplot(data=img, y="patient", x="n_nuclei", order=order, hue="split", ax=ax[0], size=3, alpha=0.8)
    ax[0].set_title("Nuclei per image, by patient"); ax[0].tick_params(axis="y", labelsize=5)
    sns.histplot(img["n_nuclei"], bins=30, ax=ax[1], color="#e76f51"); ax[1].set_title("Nuclei per image")
    for c in cls_names:
        v = np.sort(nuc.loc[nuc.cls == c, "nn_dist"].dropna().values)
        if len(v):
            ax[2].plot(v, np.arange(1, len(v) + 1) / len(v), color=CLS_PAL[c], label=c)
    ax[2].set_xlim(0, 60); ax[2].set_title("Nearest-neighbour distance ECDF (px) — crowding"); ax[2].legend(fontsize=8)
    fig.tight_layout(); viz._finish(fig, out, "eda_density.png")

    # 5. image sizes
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.scatterplot(data=img, x="w", y="h", hue="split", ax=ax[0], s=25); ax[0].set_title("Image size (px) — drives crop / tile choice")
    sns.histplot(img["fg_frac"] * 100, bins=25, ax=ax[1], color="#264653"); ax[1].set_title("Foreground coverage per image (%)")
    fig.tight_layout(); viz._finish(fig, out, "eda_image_sizes.png")

    # 6. stain / scanner variability
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.scatterplot(data=img, x="mean_r", y="mean_b", hue="patient", style="split", ax=ax[0], s=30, legend=False)
    ax[0].set_title("Per-image mean R vs B (colour = patient) — stain shift")
    sns.kdeplot(data=img.assign(brightness=(img.mean_r + img.mean_g + img.mean_b) / 3), x="brightness", hue="split", ax=ax[1], common_norm=False, fill=True, alpha=0.25)
    ax[1].set_title("Mean brightness per split")
    fig.tight_layout(); viz._finish(fig, out, "eda_stain_variability.png")

    # 7. class composition per image (clustered)
    comp = np.log1p(img[[f"n_{c}" for c in cls_names]].values)
    if len(img) >= 3:
        cg = sns.clustermap(pd.DataFrame(comp, columns=cls_names), cmap="mako_r", figsize=(6, 8), yticklabels=False, cbar_pos=(0.02, 0.85, 0.03, 0.12))
        cg.fig.suptitle("log(1+count) per image — rare-class images cluster", y=1.0)
        viz._finish(cg.fig, out, "eda_class_composition.png")

    # 8. gallery
    rng = np.random.default_rng(cfg.seed)
    pool = split_samples.get("train") or next(iter(split_samples.values()))
    idx = rng.choice(len(pool), size=min(6, len(pool)), replace=False)
    fig, ax = plt.subplots(len(idx), 3, figsize=(10, 3.3 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        d = pool[int(i)]
        ax[r, 0].imshow(d["image"]); ax[r, 1].imshow(viz.overlay_instances(d["image"], d["inst"], d["inst_class"]))
        ax[r, 2].imshow(d["bnd"], cmap="gray"); ax[r, 0].set_ylabel(d["stem"][-20:], fontsize=7)
        for c in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[0, 0].set_title("image"); ax[0, 1].set_title("instances by class"); ax[0, 2].set_title("boundary target")
    viz.class_legend(ax[0, 1], "upper right")
    fig.tight_layout(); viz._finish(fig, out, "eda_gallery.png")

    summary = (nuc.groupby("cls").agg(n=("area", "size"), mean_area=("area", "mean"), median_area=("area", "median"),
                                       mean_ecc=("eccentricity", "mean")).reindex(cls_names).round(1))
    print(summary)
    return img, nuc
