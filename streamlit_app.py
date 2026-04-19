import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cardiomegaly Detector",
    page_icon="🫀",
    layout="wide",
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMAGE_HEIGHT, IMAGE_WIDTH = 299, 299
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMG_SHAPE = IMG_SIZE + (3,)
MODEL_PATH = "export/Cardiomegaly"   # path to your saved TF SavedModel folder
THRESHOLD = 0.5


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    """Load the TensorFlow SavedModel from disk (cached across reruns)."""
    if not os.path.exists(path):
        return None
    return tf.saved_model.load(path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, convert to RGB and normalise for InceptionV3."""
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)           # (1, 299, 299, 3)
    arr = tf.keras.applications.inception_v3.preprocess_input(arr)
    return arr


def predict(model, arr: np.ndarray) -> float:
    """Return sigmoid confidence score (0–1) for Cardiomegaly."""
    infer = model.signatures["serving_default"]
    # Find the input tensor key
    input_key = list(infer.structured_input_signature[1].keys())[0]
    output = infer(**{input_key: tf.constant(arr)})
    output_key = list(output.keys())[0]
    score = float(output[output_key].numpy().flatten()[0])
    return score


def confidence_gauge(score: float):
    """Draw a simple horizontal gauge bar using matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 0.6))
    color = "#e74c3c" if score > THRESHOLD else "#2ecc71"
    ax.barh([0], [score], color=color, height=0.4)
    ax.barh([0], [1 - score], left=[score], color="#ecf0f1", height=0.4)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.axvline(x=THRESHOLD, color="#7f8c8d", linestyle="--", linewidth=1)
    ax.set_title(f"Confidence: {score:.1%}", fontsize=10, pad=4)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Chest_X-ray_NIH.png/320px-Chest_X-ray_NIH.png",
             caption="Sample chest X-ray", use_container_width=True)
    st.markdown("## ⚙️ Settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.0, max_value=1.0, value=THRESHOLD, step=0.05,
        help="Confidence above this value is classified as Cardiomegaly (positive)."
    )
    st.markdown("---")
    st.markdown(
        """
        ### About
        This app uses a fine-tuned **InceptionV3** model to detect
        **Cardiomegaly** (enlarged heart) in chest X-ray images.

        **Model details**
        - Architecture: InceptionV3 + custom head
        - Input size: 299 × 299
        - Training: Transfer learning on NIH chest X-ray data
        - Val accuracy: ~75 %

        **Disclaimer**  
        For research / educational use only.  
        Not a substitute for clinical diagnosis.
        """
    )

# ─────────────────────────────────────────────
# Main title
# ─────────────────────────────────────────────
st.title("🫀 Cardiomegaly Detection from Chest X-Rays")
st.caption("Upload a chest X-ray and the model will predict whether Cardiomegaly is present.")
st.markdown("---")

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
model = load_model(MODEL_PATH)

if model is None:
    st.warning(
        f"⚠️ No saved model found at `{MODEL_PATH}`.  \n"
        "Please place your exported TensorFlow SavedModel folder at that path and restart the app.  \n"
        "The model is exported by the training notebook as `content/export/Cardiomegaly`."
    )
    st.stop()

# ─────────────────────────────────────────────
# Upload section
# ─────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("📤 Upload X-Ray")
    uploaded_files = st.file_uploader(
        "Choose one or more chest X-ray images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
if uploaded_files:
    results = []
    for f in uploaded_files:
        image = Image.open(f)
        arr = preprocess_image(image)
        score = predict(model, arr)
        label = "🔴 Cardiomegaly Detected" if score > threshold else "🟢 No Cardiomegaly"
        results.append({
            "file": f,
            "image": image,
            "score": score,
            "label": label,
        })

    # ── Results grid ──────────────────────────
    st.markdown("---")
    st.subheader(f"📊 Results — {len(results)} image(s) analysed")

    cols_per_row = 2
    for i in range(0, len(results), cols_per_row):
        row_results = results[i: i + cols_per_row]
        cols = st.columns(cols_per_row, gap="medium")
        for col, r in zip(cols, row_results):
            with col:
                st.image(r["image"], caption=r["file"].name, use_container_width=True)
                st.markdown(f"### {r['label']}")
                st.pyplot(confidence_gauge(r["score"]), use_container_width=True)
                st.metric("Confidence score", f"{r['score']:.4f}")
                st.markdown("---")

    # ── Summary table ─────────────────────────
    if len(results) > 1:
        st.subheader("📋 Summary Table")
        import pandas as pd
        summary = pd.DataFrame([
            {
                "Filename": r["file"].name,
                "Prediction": "Positive" if r["score"] > threshold else "Negative",
                "Confidence": f"{r['score']:.4f}",
                "Result": r["label"],
            }
            for r in results
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    with col_result:
        st.info("👈 Upload one or more chest X-ray images to get started.")
        st.markdown(
            """
            #### What the model looks for
            Cardiomegaly is characterised by an **enlarged cardiac silhouette**
            visible on a chest X-ray, typically when the cardiothoracic ratio
            exceeds 0.5 on a PA view.

            #### Tips for best results
            - Use frontal (PA or AP) chest X-ray images
            - JPEG or PNG format, any resolution (resized to 299×299 internally)
            - Avoid heavily cropped or rotated images
            """
        )
