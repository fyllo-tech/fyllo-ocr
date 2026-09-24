# Deploy-ready image for the fyllo-ocr test API (CPU + NVIDIA GPU capable).
# GPU path: build the same image on a CUDA host after installing the
# paddlepaddle-gpu wheel (see requirements.txt), then run with --gpus all.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OCR_DEVICE=auto \
    OCR_PRELOAD=1

# System libs for opencv / onnxruntime / Pillow / PyMuPDF.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglx-mesa0 \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

EXPOSE 8000

# Model is warmed on server startup (lifespan preload), not first request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
