# fyllo-ocr — simple PaddleOCR test API

FastAPI server with two routes:

- `GET /health` → `{"status": "ok", "message": "fyllo-ocr is working"}`
- `POST /parse` → upload an image (`jpg/jpeg/png/bmp/tiff/webp`) or `pdf`,
  runs PaddleOCR (3.x, ONNX Runtime CPU, same logic as
  `soilreport/report_extraction/text_extraction/paddleocr_extractor.py`),
  returns text + per-line confidences + metrics.

Response shape:

```json
{
  "filename": "report.jpg",
  "text": "...",
  "lines": [{"text": "...", "confidence": 0.98}],
  "metrics": {
    "time_taken_sec": 12.3,
    "total_request_sec": 12.4,
    "page_count": 1,
    "pages_with_text": 1,
    "line_count": 10,
    "word_count": 80,
    "char_count": 500,
    "avg_confidence": 0.95,
    "avg_confidence_percent": 95.0,
    "min_confidence": 0.7,
    "max_confidence": 0.99,
    "is_empty": false,
    "engine": "paddleocr-onnxruntime"
  }
}
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docs: http://localhost:8000/docs

Test:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/parse -F "file=@sample.jpg"
```

## Run with Docker

```bash
docker build -t fyllo-ocr .
docker run -p 8000:8000 fyllo-ocr
# or
docker compose up --build
```

Note: the OCR model is loaded on server startup (lifespan preload,
~20–45s once per container) onto the GPU when CUDA is available;
subsequent calls are fast. Check `GET /health` for
`device` / `ocr_loaded` / `cuda_available`.

Env:

- `MAX_UPLOAD_MB` (default `20`) — max upload size.
- `OCR_DEVICE` (default `auto`) — `auto` uses `gpu:0` if CUDA is
  detectable (torch / paddle / nvidia-smi), else `cpu`. Force with
  `cpu`, `gpu:0`, `cuda`, `cuda:1`, `npu:0`, etc.
- `OCR_ENGINE` (default `auto`) — `auto` uses ONNX Runtime on CPU and
  PaddleInference on GPU. Override to force e.g. `onnxruntime`.
- `OCR_PRELOAD` (default `1`) — set `0` to skip startup warmup.

## GPU

```bash
# needs NVIDIA drivers + nvidia-container-toolkit
docker run --gpus all -p 8000:8000 -e OCR_DEVICE=auto fyllo-ocr
```

For full GPU speed install the matching CUDA build inside the image
(see `requirements.txt`, e.g. `paddlepaddle-gpu` cu118/cu126).
