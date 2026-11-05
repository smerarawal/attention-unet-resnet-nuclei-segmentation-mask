# Attention-UNet++ on MoNuSAC (Kaggle T4)

Nucleus **segmentation + 4-class classification** (epithelial / lymphocyte / neutrophil / macrophage) with instance recovery.

## Run on Kaggle
1. Upload this folder as a Dataset (`nucseg-code`); add the MoNuSAC dataset to the notebook; GPU = **T4 x1**.
2. Open `kaggle_monusac.ipynb`, run top to bottom (or `python run_all.py --data_root <path>`).
3. ImageNet weights need Internet ON. Offline: upload torchvision `resnet34` weights and set `encoder_weights_path`.
4. Session limit: `time_budget_min=500` stops cleanly; rerun with `resume=True`.

Every knob is in `nucseg/config.py` and is also a CLI flag (`--epochs 80 --encoder scratch`).

## Layout
```
nucseg/config.py        all hyperparameters
nucseg/data/            monusac.py (XML->masks, npz cache) | splits.py (patient-level) | dataset.py (crops, HED stain jitter)
nucseg/models/          blocks.py | encoders.py (resnet34 | your scratch encoder) | unetpp.py (attention UNet++, 2 heads)
nucseg/losses.py        weighted CE + multiclass Dice + boundary BCE/Dice, ignore-aware
nucseg/metrics.py       Dice/IoU per class, AJI, PQ/DQ/SQ per class
nucseg/postprocess.py   watershed instances, class vote, val-tuned thresholds
nucseg/engine.py        prepare / train / evaluate
nucseg/eda.py, viz.py   all figures
tools/                  git repair + guard   |   tests/  synthetic-data smoke tests
```

## What changed vs the original notebook
- MoNuSAC XML parsed natively (cv2 polygon fill), classes kept, ambiguous regions ignored (label 255), not treated as background.
- **Patient-level** train/val/test split (old `random_split` could leak slides); official test folder used if present.
- Native-resolution random crops (old 128 -> 256 resize was lossy); no interpolation on masks.
- Optional ImageNet ResNet-34 encoder (`--encoder scratch` keeps your original for the ablation).
- Multi-class output + boundary head -> watershed instances; metrics are AJI / PQ, not just IoU.
- AMP, AdamW + OneCycle, grad clip, early stopping, resume, flip TTA, tiled inference for large images.

## Outputs (`<out_dir>/figures`)
EDA: splits, class balance, morphology, density, image sizes, stain variability, class composition, gallery.
Train: class weights, augmentation preview, curves. Eval: confusion matrix, per-class Dice/IoU/PQ, per-image AJI,
best/median/worst prediction panels, attention-gate maps, post-processing tuning grid.
Tables: `history.csv`, `eda_*.csv`, `per_image_*.csv`, `results_*.json`.

## Git: permanent fix
Root cause: the repo root was a parent folder (Downloads), so `git add .` swept in sibling projects and a nested repo.
```
cd <this folder>
bash tools/fix_stray_history.sh     # backs up, re-roots history to this folder, strips submodule refs
bash tools/install_guard.sh         # root == this folder; pre-commit guard blocks foreign paths, nested repos, >5 MB / weights / data
git remote add origin <url> && git push --force --all
```
