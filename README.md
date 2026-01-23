# Production-Ready Real-Time Emotion Intelligence System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

A production-minded facial expression recognition system built around a DDAMFN-style PyTorch model.

This repository is structured to send strong "engineer who ships" signals:
- 📦 Installable Python package (`pip install -e .`)
- 🚀 FastAPI inference service (`POST /predict`, `GET /health`)
- 🖼️ Streamlit demo UI (image upload + camera snapshot)
- 🐳 Docker and docker compose
- 🧪 Pytest test suite and GitHub Actions CI

**Model Weights:** [Hugging Face Hub](https://huggingface.co/imdiora/ddamfn-facial-expression-recognition)

---

## Demo

*(Optional: Add a GIF here of the streamlit UI in action)*

---

## Architecture

```mermaid
flowchart LR
    A[Video / Image] --> B[Face Detect (OpenCV Haar)]
    B --> C[Crop + Resize + Normalize]
    C --> D[DDAMFN Model]
    D --> E[Softmax + Top-1]
    E --> F[FastAPI /predict]
    F --> G[Streamlit Demo]