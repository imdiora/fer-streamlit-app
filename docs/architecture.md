# System Architecture

High-level pipeline:

1. **Input**: image / camera frame (BGR)
2. **Face Detection**: OpenCV Haar cascade (configurable)
3. **Preprocess**: crop -> resize -> normalize
4. **Model Inference**: DDAMFN (PyTorch)
5. **Output**: JSON detections (box + label + confidence)
6. **Presentation**: Streamlit UI overlays boxes + labels

Mermaid diagram (also used in README):

```mermaid
flowchart LR
    A[Video / Image] --> B[Face Detect (OpenCV Haar)]
    B --> C[Crop + Resize + Normalize]
    C --> D[DDAMFN Model]
    D --> E[Softmax + Top-k]
    E --> F[FastAPI /predict]
    F --> G[Streamlit Demo]
```
