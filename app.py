"""
Streamlit frontend for the CharCNN character classifier.

Run:
    streamlit run app.py

Then open the URL Streamlit prints (usually http://localhost:8501).
"""
import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from streamlit_drawable_canvas import st_canvas

from model import CharCNN
from labels import NUM_CLASSES, IDX_TO_CLASS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT = "checkpoints/best.pt"
DEFAULT_TOPK = 5

st.set_page_config(page_title="Cursive Character Recognition", page_icon="✍️", layout="centered")


# ---------------------------------------------------------------------------
# Model loading (cached so it happens once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    img_size = ckpt["img_size"]

    model = CharCNN(num_classes=NUM_CLASSES, img_size=img_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = T.Compose([
        T.Grayscale(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])
    return model, transform, img_size, device


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def autocrop_square(img: Image.Image, pad: int = 4) -> Image.Image:
    """
    Crop to the bounding box of dark pixels and pad to a square.
    Training images were padded-to-square + resized, so this matches
    the training distribution and makes the demo far more forgiving
    of users who draw off-center or small.
    """
    arr = np.array(img.convert("L"))
    mask = arr < 128
    if not mask.any():
        return img  # blank image; let the model do whatever it does

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1

    side = max(h, w) + 2 * pad
    out = Image.new("L", (side, side), 255)
    crop = img.convert("L").crop((x0, y0, x1 + 1, y1 + 1))
    out.paste(crop, ((side - w) // 2, (side - h) // 2))
    return out


def flatten_alpha(img: Image.Image) -> Image.Image:
    """Composite RGBA/LA onto white before grayscaling (canvas exports have alpha)."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    return img.convert("L")


@torch.no_grad()
def predict(model, transform, device, pil_img: Image.Image, topk: int = 5):
    img = flatten_alpha(pil_img)
    img = autocrop_square(img)
    tensor = transform(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idxs = probs.topk(topk)
    results = [
        {"char": IDX_TO_CLASS[idx], "prob": float(p)}
        for p, idx in zip(top_probs.tolist(), top_idxs.tolist())
    ]
    # Return the preprocessed 32x32 image too, so the UI can show what the model saw.
    preview = transform(img).squeeze(0) * 0.5 + 0.5  # undo Normalize
    return results, preview


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_results(preds):
    if not preds:
        st.info("No predictions.")
        return
    top = preds[0]
    st.markdown(f"### → `{top['char']}`  &nbsp;&nbsp;*{top['prob']*100:.1f}%*")

    max_prob = max(p["prob"] for p in preds) or 1.0
    for p in preds:
        col1, col2, col3 = st.columns([1, 6, 1])
        col1.markdown(
            f"<div style='font-family:monospace;font-size:1.4rem;text-align:center'>{p['char']}</div>",
            unsafe_allow_html=True,
        )
        col2.progress(min(p["prob"] / max_prob, 1.0))
        col3.markdown(
            f"<div style='color:#888;font-size:.85rem;text-align:right'>{p['prob']*100:.1f}%</div>",
            unsafe_allow_html=True,
        )


def render_preview(preview_tensor):
    """Show the 32x32 image the model actually consumed."""
    arr = (preview_tensor.clamp(0, 1).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize((128, 128), Image.NEAREST)
    st.image(img, caption="What the model saw (32×32)", width=128)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.title("✍️ Character Classifier")
    st.caption("Draw or upload a character — digits 0–9, lowercase a–z, uppercase A–Z.")

    if not Path(CHECKPOINT).exists():
        st.error(
            f"Checkpoint not found at `{CHECKPOINT}`.\n\n"
            "Train one with `python train.py --datasets ...` or update `CHECKPOINT` in `app.py`."
        )
        st.stop()

    model, transform, img_size, device = load_model(CHECKPOINT)
    st.caption(f"Model: CharCNN @ {img_size}×{img_size} · device: `{device}`")

    topk = st.sidebar.slider("Top-K predictions", 1, 10, DEFAULT_TOPK)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tips for good predictions**\n"
        "- Fill the drawing area\n"
        "- Black strokes, white background\n"
        "- Center the character"
    )

    tab_draw, tab_upload = st.tabs(["✏️ Draw", "📁 Upload"])

    # ------------------------------------------------------------------
    # Draw tab
    # ------------------------------------------------------------------
    with tab_draw:
        col_canvas, col_result = st.columns([1, 1])

        with col_canvas:
            stroke_width = st.slider("Brush width", 4, 30, 14)
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",  # transparent fill
                stroke_width=stroke_width,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
                return_image_data=True,
            )

        with col_result:
            st.markdown("#### Prediction")
            if canvas_result.image_data is not None:
                # st_canvas gives RGBA as numpy; only predict if something was drawn.
                arr = canvas_result.image_data.astype(np.uint8)
                alpha = arr[..., 3] if arr.shape[-1] == 4 else None
                drew_something = alpha is not None and (alpha > 0).any()

                if drew_something:
                    img = Image.fromarray(arr, mode="RGBA")
                    preds, preview = predict(model, transform, device, img, topk=topk)
                    render_results(preds)
                    st.markdown("---")
                    render_preview(preview)
                else:
                    st.info("Start drawing on the canvas.")

    # ------------------------------------------------------------------
    # Upload tab
    # ------------------------------------------------------------------
    with tab_upload:
        col_up, col_result = st.columns([1, 1])

        with col_up:
            uploaded = st.file_uploader(
                "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"]
            )
            if uploaded is not None:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded image", use_container_width=True)

        with col_result:
            st.markdown("#### Prediction")
            if uploaded is not None:
                img = Image.open(uploaded)
                preds, preview = predict(model, transform, device, img, topk=topk)
                render_results(preds)
                st.markdown("---")
                render_preview(preview)
            else:
                st.info("Upload an image to see predictions.")

    st.markdown("---")
    st.caption(
        "Trained on C-Cube / CHoiCe / EMNIST-derived character crops. "
        "Confusable cursive pairs (`a`/`o`, `n`/`u`, `C`/`c`) are the usual failure modes."
    )


if __name__ == "__main__":
    main()
