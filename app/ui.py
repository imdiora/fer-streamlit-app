from __future__ import annotations

import cv2
import streamlit as st

from emotion_ai.predictor import EmotionPredictor
from emotion_ai.utils import bgr_from_bytes


st.set_page_config(page_title="Emotion AI System", layout="wide")


@st.cache_resource
def load_engine() -> EmotionPredictor:
    """Load the model once per Streamlit server process."""
    return EmotionPredictor(config_path="config/config.yaml")


try:
    predictor = load_engine()
except Exception as e:
    st.error(f"Failed to initialize EmotionPredictor: {e}")
    st.stop()


st.title("Production-Ready Real-Time Emotion Intelligence")
st.caption("DDAMFN (PyTorch) + Face Detection + Streamlit Demo")


default_thresh = float(predictor.config.ui.default_confidence_threshold)
confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=default_thresh,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.write("Runtime")
st.sidebar.write(f"Device: `{predictor.device}`")
st.sidebar.write(f"Weights: `{predictor.config.model.weights_path}`")
st.sidebar.write(f"Input size: `{predictor.config.model.input_size}x{predictor.config.model.input_size}`")


tab_upload, tab_camera = st.tabs(["Image Upload", "Camera Snapshot"])


def render_results(image_bgr, detections):
    filtered = [d for d in detections if d["confidence"] >= confidence_threshold]
    annotated = predictor.annotate_image(image_bgr, filtered)

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="Annotated image",
            use_column_width=True,
        )
    with col2:
        st.subheader("Detections")
        st.write(f"Total: **{len(filtered)}** (after threshold)")
        st.json(filtered)


with tab_upload:
    st.write("Upload a photo. The system detects faces and predicts expressions per face.")
    uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image_bgr = bgr_from_bytes(uploaded.getvalue())
        detections = predictor.predict(image_bgr)
        st.write(f"Faces detected: **{len(detections)}**")
        render_results(image_bgr, detections)


with tab_camera:
    st.write("Use your webcam to take a snapshot (works on most Streamlit setups).")
    cam = st.camera_input("Take a picture")
    if cam is not None:
        image_bgr = bgr_from_bytes(cam.getvalue())
        detections = predictor.predict(image_bgr)
        st.write(f"Faces detected: **{len(detections)}**")
        render_results(image_bgr, detections)


st.markdown("---")
st.markdown(
    "Tip: run the API with `uvicorn app.api:app --reload` and send an image to `POST /predict`."
)
