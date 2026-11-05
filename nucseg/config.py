"""Single source of truth for every knob. Override from code or CLI (--epochs 80)."""
from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, asdict

ON_KAGGLE = os.path.exists("/kaggle/working")


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    data_root: str = "auto"          # "auto" scans /kaggle/input for the folder with the most .xml files
    out_dir: str = "/kaggle/working/run" if ON_KAGGLE else "./runs/run"
    cache_dir: str = ""              # "" -> <out_dir>/../cache_monusac
    encoder_weights_path: str = ""   # offline Kaggle: path to resnet34-*.pth uploaded as a dataset

    # ---- splits (always PATIENT-level: no leakage between crops of one slide) -
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15          # ignored if a folder with 'test' in its path exists (official test set)

    # ---- data --------------------------------------------------------------
    crop: int = 256
    crops_per_image: int = 4         # virtual epoch length multiplier
    fg_crop_p: float = 0.6           # prob. of centring a crop on a nucleus pixel
    boundary_width: int = 2          # px, inner-boundary thickness used as 2nd target
    hed_jitter_p: float = 0.5        # stain augmentation (HED space)
    num_workers: int = 2

    # ---- model -------------------------------------------------------------
    encoder: str = "resnet34"        # "resnet34" (ImageNet-init) | "scratch" (your original residual encoder)
    pretrained: bool = True          # falls back to random init with a warning if weights can't be fetched
    base_ch: int = 32                # decoder widths = base_ch * (1, 2, 4, 8)
    n_classes: int = 5               # bg + epithelial, lymphocyte, neutrophil, macrophage

    # ---- optimisation ------------------------------------------------------
    epochs: int = 60
    batch_size: int = 12
    lr: float = 1e-3
    encoder_lr_mult: float = 0.3     # pretrained encoder trains slower than the fresh decoder
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    patience: int = 15               # early stop on val mean fg Dice
    time_budget_min: int = 0         # >0: stop cleanly before Kaggle's session limit
    w_ce: float = 1.0
    w_dice: float = 1.0
    w_bnd: float = 0.5
    resume: bool = False

    # ---- inference / post-processing ---------------------------------------
    tile: int = 512                  # sliding-window size for large images
    tile_overlap: int = 64
    tta: bool = True                 # 4-way flip TTA at eval
    fg_thr: float = 0.5
    bnd_thr: float = 0.5
    min_area: int = 20
    tune_postproc: bool = True       # grid-search fg/bnd/min_area on VAL by AJI, apply to TEST

    def resolve(self) -> "Config":
        if not self.cache_dir:
            self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(self.out_dir)), "cache_monusac")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "figures"), exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        return self

    def to_dict(self):
        return asdict(self)


def _str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def parse_cli(argv=None) -> Config:
    """--any_field value for every Config field (bools: --amp false)."""
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        t = _str2bool if f.type in ("bool", bool) else {"int": int, "float": float}.get(str(f.type), str)
        p.add_argument(f"--{f.name}", type=t, default=f.default)
    args, _ = p.parse_known_args(argv)  # parse_known_args: harmless inside Jupyter
    return Config(**vars(args)).resolve()
