# Attention-UNet++ on MoNuSAC (Kaggle T4)

Nucleus **segmentation + 4-class classification** (epithelial / lymphocyte / neutrophil / macrophage) with instance recovery.

## Run on Kaggle
1. Upload this folder as a Dataset (`nucseg-code`); add the MoNuSAC dataset to the notebook; GPU = **T4 x1**.
2. Open `kaggle_monusac.ipynb`, run top to bottom (or `python run_all.py --data_root <path>`).
3. ImageNet weights need Internet ON. Offline: upload torchvision `resnet34` weights and set `encoder_weights_path`.
4. Session limit: `time_budget_min=500` stops cleanly; rerun with `resume=True`.

## Run the live laptop demo
The demo opens a local web page where you can run the trained model on the included example image or upload another H&E image. It uses the checkpoint produced during training; the weights are kept outside Git because the file is large.

Install a standard CPython 3.11 or 3.12 build from [python.org](https://www.python.org/downloads/windows/) first. Then, on Windows from this project folder:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run demo_app.py
```
The app opens in your browser. By default it looks for `D:\ds\monusac_run\best.pt`. If your checkpoint is elsewhere, paste its path into the **Model checkpoint** field in the sidebar. The demo runs on CUDA when available and otherwise uses CPU; CPU inference can take longer. Click **Run segmentation** to see instance outlines and per-class counts.

On your current `D:\ds` setup, the app crops its built-in example from `D:\ds\monusac_run\figures\eval_test_panels_best.png`; the image stays local and is not part of the GitHub repository. If you run the code on another computer, upload a PNG, JPEG, or TIFF H&E image instead. The example has no ground-truth mask in the app.

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

## Checkpoint and data
Keep the trained checkpoint and MoNuSAC dataset outside the Git repository. The `.gitignore` excludes model weights, caches, and data files.
