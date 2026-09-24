"""fyllo-ocr test API: health + OCR parse (image/PDF via PaddleOCR)."""
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.extractor import (
    SUPPORTED_EXTENSIONS,
    get_device_info,
    paddleocr_extract_with_details,
    preload_model,
)

logger = logging.getLogger("fyllo_ocr")
logging.basicConfig(level=logging.INFO)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# OCR_PRELOAD=0 to skip model warmup (e.g. lightweight health-only container).
OCR_PRELOAD = os.getenv("OCR_PRELOAD", "1") not in ("0", "false", "no")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if OCR_PRELOAD:
        t0 = time.perf_counter()
        try:
            device = preload_model()
            logger.info(f"OCR model warmed on {device} in {time.perf_counter() - t0:.1f}s")
        except Exception:
            logger.exception("OCR preload failed; first /parse will load lazily")
    else:
        logger.info("OCR_PRELOAD=0, skipping model warmup")
    yield


app = FastAPI(title="fyllo-ocr", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    info = get_device_info()
    return {
        "status": "ok",
        "message": "fyllo-ocr is working",
        "ocr_loaded": info["loaded"],
        "device": info["active"],
        "cuda_available": info["cuda_available"],
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "fyllo-ocr is working. See /docs for /health and /parse."}


@app.post("/parse")
def parse(file: UploadFile = File(...)):
    """Accept an image or PDF, run PaddleOCR, return text + metrics.

    Response:
    {
      "filename": str,
      "text": str,
      "lines": [{"text": str, "confidence": float}],
      "metrics": {
        "time_taken_sec": float, "word_count": int, "char_count": int,
        "line_count": int, "page_count": int, "pages_with_text": int,
        "avg_confidence": float, "avg_confidence_percent": float,
        "min_confidence": float, "max_confidence": float, ...
      }
    }
    """
    t0 = time.perf_counter()
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or '(none)'}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Read into memory (bounded) then spill to temp file for the extractor.
    try:
        blob = file.file.read()
    finally:
        try:
            file.file.close()
        except Exception:
            pass
    if not blob:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(blob) / 1048576:.1f} MB > {MAX_UPLOAD_MB} MB).",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(blob)
        tmp_path = tmp.name
    try:
        try:
            result = paddleocr_extract_with_details(tmp_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("OCR failed")
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

        total = round(time.perf_counter() - t0, 3)
        result["metrics"]["total_request_sec"] = total
        return JSONResponse(
            {
                "filename": filename,
                "text": result["text"],
                "lines": result["lines"],
                "metrics": result["metrics"],
            }
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
