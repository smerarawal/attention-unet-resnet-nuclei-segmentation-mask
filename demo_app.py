"""Small local Streamlit app for interactive MoNuSAC inference."""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from skimage.segmentation import find_boundaries

from nucseg.config import Config
from nucseg.data.dataset import EvalDS
from nucseg.data.monusac import CLASS_NAMES
from nucseg.engine import forward_probs, load_model
from nucseg.postprocess import instances_from_probs
from nucseg.utils import get_device


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT.parents[1] / "monusac_run" / "best.pt"
EXAMPLE_PANEL = ROOT.parents[1] / "monusac_run" / "figures" / "eval_test_panels_best.png"
EXAMPLE_CROP = (51, 216, 457, 430)
CLASS_COLORS = np.array(
    [
        [0, 0, 0],       # background
        [239, 71, 111],  # epithelial
        [6, 214, 160],   # lymphocyte
        [255, 209, 102], # neutrophil
        [17, 138, 178],  # macrophage
    ],
    dtype=np.uint8,
)


@st.cache_resource(show_spinner="Loading the trained model…")
def get_model(checkpoint_path: str, device_name: str):
    device = torch.device(device_name)
    model, checkpoint = load_model(Config(), checkpoint_path, device)
    config = Config(**checkpoint["cfg"])
    return model, config, checkpoint


def read_rgb(file_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(file_bytes)).convert("RGB"))


def read_local_example() -> np.ndarray:
    with Image.open(EXAMPLE_PANEL) as panel:
        return np.asarray(panel.crop(EXAMPLE_CROP).convert("RGB"))


def paint_instances(image: np.ndarray, instances: np.ndarray, classes: np.ndarray) -> np.ndarray:
    pixel_class = classes[instances]
    class_color = CLASS_COLORS[pixel_class]
    foreground = instances > 0
    result = image.astype(np.float32).copy()
    result[foreground] = 0.48 * result[foreground] + 0.52 * class_color[foreground]
    edges = find_boundaries(instances, mode="outer") & foreground
    result[edges] = (255, 255, 255)
    return np.clip(result, 0, 255).astype(np.uint8)


def predict(image: np.ndarray, model, config: Config, device, tta: bool, fg_thr: float, bnd_thr: float, min_area: int):
    # Match evaluation: reflect-pad to a multiple of 32, then crop predictions back.
    sample = {"image": image, "sem": np.zeros(image.shape[:2], dtype=np.uint8), "bnd": np.zeros(image.shape[:2], dtype=np.uint8)}
    x, _, _, meta = EvalDS([sample])[0]
    height, width = image.shape[:2]
    with torch.inference_mode():
        probs, boundary = forward_probs(model, x.unsqueeze(0).to(device), config, device, tta=tta)
    probs = probs[:, :height, :width].numpy().astype(np.float32)
    boundary = boundary[:height, :width].numpy().astype(np.float32)
    instances, classes = instances_from_probs(probs, boundary, fg_thr, bnd_thr, min_area)
    return instances, classes


st.set_page_config(page_title="Nuclei Segmentation Demo", page_icon="🔬", layout="wide")
st.title("MoNuSAC Nuclei Segmentation")
st.caption("Attention U-Net++ with ResNet-34 · instance segmentation and 4 cell classes")

with st.sidebar:
    st.header("Demo settings")
    checkpoint_path = st.text_input("Model checkpoint", str(DEFAULT_CHECKPOINT))
    use_tta = st.checkbox("Flip test-time augmentation", value=False, help="More accurate averaging, but slower on CPU.")
    fg_thr = st.slider("Foreground threshold", 0.1, 0.9, 0.5, 0.05)
    bnd_thr = st.slider("Boundary threshold", 0.1, 0.9, 0.5, 0.05)
    min_area = st.slider("Minimum nucleus area (pixels)", 1, 100, 20, 1)

uploaded = st.file_uploader("Upload an H&E image, or use the built-in example", type=["png", "jpg", "jpeg", "tif", "tiff"])
if uploaded is None:
    if not EXAMPLE_PANEL.is_file():
        st.warning("The built-in example is not available here. Upload an H&E image to continue.")
        st.stop()
    input_image = read_local_example()
    st.info("Using the held-out example crop from your local saved evaluation panel. Upload another image to try it too.")
else:
    try:
        input_image = read_rgb(uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read that image: {exc}")
        st.stop()

if input_image.shape[0] * input_image.shape[1] > 30_000_000:
    st.error("This image is too large for the laptop demo (limit: 30 megapixels). Please use a crop.")
    st.stop()

if st.button("Run segmentation", type="primary"):
    model_file = Path(checkpoint_path).expanduser()
    if not model_file.is_file():
        st.error(f"Checkpoint not found: {model_file}\n\nExpected the trained file at {DEFAULT_CHECKPOINT}.")
        st.stop()
    device = get_device()
    with st.spinner(f"Running inference on {device.type.upper()}…"):
        try:
            model, config, checkpoint = get_model(str(model_file.resolve()), device.type)
            started = time.perf_counter()
            pred_instances, pred_classes = predict(input_image, model, config, device, use_tta, fg_thr, bnd_thr, min_area)
            elapsed = time.perf_counter() - started
        except Exception as exc:
            st.exception(exc)
            st.stop()

    overlay = paint_instances(input_image, pred_instances, pred_classes)
    total = int(pred_instances.max())
    st.success(f"Found {total:,} nuclei in {elapsed:.2f} seconds · checkpoint epoch {checkpoint['epoch']}")
    left, right = st.columns(2)
    left.image(input_image, caption="Input image", use_container_width=True)
    right.image(overlay, caption="Predicted instances and classes", use_container_width=True)

    counts = {CLASS_NAMES[c]: int(np.count_nonzero(pred_classes[1:] == c)) for c in range(1, len(CLASS_NAMES))}
    cols = st.columns(len(counts) + 1)
    cols[0].metric("Total nuclei", f"{total:,}")
    for col, (name, count) in zip(cols[1:], counts.items()):
        col.metric(name.title(), f"{count:,}")

    st.markdown("**Overlay colors:** epithelial · lymphocyte · neutrophil · macrophage (white outlines show instance boundaries).")
    st.caption("Research demonstration only; predictions are not for clinical diagnosis.")
else:
    st.image(input_image, caption=f"Ready to run · {input_image.shape[1]} × {input_image.shape[0]} pixels", use_container_width=True)
    st.caption("Select **Run segmentation** to see the model prediction.")
