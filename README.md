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

## Pest detection (`POST /detect`)

Runs the per-pest YOLO11 models from `pest-trap` on one trap image. It uses
the same 1024 px tiles (0.2 overlap), per-model `conf` from
`model_config.json`, and distance NMS as `image_group_processor_lambda.py`.
There is no S3, no Fyllo API call and no trap-classifier gate.

The weights (~385 MB) are not in this repo. Copy them in once:

```bash
cp -r ../pest-trap/pest_detection/model_r24 ./pest_models   # mounted by docker-compose
```

```bash
curl -X POST http://localhost:8000/detect -F "file=@trap.jpg"
curl -X POST http://localhost:8000/detect -F "file=@trap.webp" -F "models=thrips,whitefly"
```

Response: `width`, `height`, `tiles`, `device`, `detections`
(`model, class, cx, cy, w, h, score` in original-image pixels), `counts` per
model, and `metrics` (`bytes`, `decode_sec`, `infer_sec`, `per_model_sec`,
`total_request_sec`). JPEG, PNG, WebP and AVIF uploads are all accepted.

Env: `PEST_MODEL_DIR` (default `/code/pest_models`), `PEST_DEVICE`
(`auto` → `cuda:0` when available), `PEST_BATCH` (tiles per predict call,
default 16), `PEST_MAX_UPLOAD_MB` (default 60), `PEST_PRELOAD` (default `1`).
`GET /health` lists the loaded pest models under `pest`.

## GPU

```bash
# needs NVIDIA drivers + nvidia-container-toolkit
docker build -t fyllo-ocr .
docker build -f Dockerfile.gpu -t fyllo-ocr:gpu .
docker run --gpus all -p 8000:8000 -e OCR_DEVICE=gpu:0 fyllo-ocr:gpu
```

`Dockerfile.gpu` adds `paddlepaddle-gpu==3.3.1` (cu118) on top of the base
image. Verified on an A10G (g5.xlarge, driver 550 / CUDA 12.4) with
`paddleocr==3.7.0` + `paddlex==3.7.2`. `paddlepaddle-gpu==3.0.0` does not
work with these versions: every page fails with a pir `strides` attribute error
and `/parse` returns empty text.
