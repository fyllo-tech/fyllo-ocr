"""PaddleOCR extraction backend (PaddleOCR 3.x, GPU-first with CPU fallback).

File path in, plain text + per-line confidences out. Images go straight
to the recognizer; PDF pages are rendered with PyMuPDF and read page by
page (first N max). Model init stays lazy (singleton, warmed on server
startup via app.main lifespan) so the GPU model is ready before the
first request.

Device selection via OCR_DEVICE env (default "auto"):
  auto  -> gpu:0 if CUDA is available, else cpu
  cpu   -> force CPU (ONNX Runtime engine, fastest on CPU)
  gpu / gpu:0 / cuda / cuda:0 -> force GPU (PaddleInference engine)
  npu:0 / xpu:0 / ... -> passed straight through to PaddleOCR

Ported from @fyllo-ai/soilreport/report_extraction/text_extraction/paddleocr_extractor.py
with added confidence scores + metrics for the fyllo-ocr test API.
"""
import logging
import os
import shutil
import subprocess
import tempfile
import time

import fitz  # PyMuPDF: PDF page rendering
from PIL import Image

try:
    import torch  # noqa: F401  (torch DLLs must load before paddle on Windows)
except ImportError:
    torch = None  # type: ignore[assignment]  # optional on Linux/Docker

try:
    import paddle
except ImportError:
    paddle = None  # type: ignore[assignment]  # only used for CUDA detection

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None  # type: ignore[assignment]  # surfaced as error in get_ocr()

logger = logging.getLogger("fyllo_ocr")

# Longest edge fed to the recognizer; boxes are positions, not pixels,
# so capping only buys speed. ~17s/image measured at this size.
INFER_LONG_EDGE = 1600

# Keep in sync with soilreport.report_extraction.config
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}

_ocr = None
_ocr_device = None


def _cuda_available() -> bool:
    """True if an NVIDIA GPU is usable. Cheap checks first."""
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return True
        except Exception:
            pass
    if paddle is not None:
        try:
            # paddle.device.cuda.device_count() exists on PaddlePaddle 3.x
            if paddle.device.cuda.device_count() > 0:
                return True
        except Exception:
            pass
    # Last resort: nvidia-smi present (driver-level, no framework needed).
    try:
        if shutil.which("nvidia-smi"):
            subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=True,
            )
            return True
    except Exception:
        pass
    return False


def resolve_device(explicit: str | None = None) -> str:
    """Resolve OCR_DEVICE env (or explicit value) to a PaddleOCR device string."""
    raw = (explicit if explicit is not None else os.getenv("OCR_DEVICE", "auto")).strip().lower()
    if raw in ("auto", "", "none"):
        return "gpu:0" if _cuda_available() else "cpu"
    if raw in ("cuda", "gpu"):
        return "gpu:0"
    if raw.startswith("cuda:"):
        return "gpu:" + raw.split(":", 1)[1]
    # cpu, gpu:N, npu:N, xpu:N, ... pass through
    return raw


def get_device_info() -> dict:
    return {
        "configured": os.getenv("OCR_DEVICE", "auto"),
        "active": _ocr_device or resolve_device(),
        "loaded": _ocr is not None,
        "cuda_available": _cuda_available(),
    }


def get_ocr(device: str | None = None):
    """Lazy singleton (GPU when available): model init (~20-45s once).

    Call once at server startup to warm the model; per-request calls reuse
    the cached instance and pay no init cost.
    """
    global _ocr, _ocr_device
    if _ocr is None:
        if PaddleOCR is None:
            raise RuntimeError(
                "PaddleOCR is not installed. Install it with: pip install -r requirements.txt"
            )

        resolved = resolve_device(device)
        _ocr_device = resolved
        engine = os.getenv("OCR_ENGINE", "auto").strip().lower()
        kwargs: dict = {
            "lang": "en",
            # Orientation stages off: upright lab photos don't need them.
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": resolved,
        }
        if engine not in ("auto", "", "none"):
            kwargs["engine"] = engine
        elif resolved == "cpu":
            # ONNX engine: ~5x faster than paddle_static on CPU and sidesteps
            # the PaddlePaddle 3.x OneDNN Windows crash. On GPU the default
            # PaddleInference backend is used instead (ONNX-GPU needs
            # onnxruntime-gpu + CUDA providers).
            kwargs["engine"] = "onnxruntime"
        t0 = time.perf_counter()
        _ocr = PaddleOCR(**kwargs)
        logger.info(f"[PaddleOCR] model loaded on {resolved} in {time.perf_counter() - t0:.1f}s")
    return _ocr


def preload_model() -> str:
    """Load the model now (server startup). Returns the active device."""
    get_ocr()
    assert _ocr_device is not None
    return _ocr_device


def _render_pdf_pages(file_path: str, dpi: int = 200, max_pages: int = 3):
    """Render the first max_pages PDF pages to temp PNGs. Caller deletes."""
    paths = []
    with fitz.open(file_path) as doc:
        for page in list(doc)[:max_pages]:
            pix = page.get_pixmap(dpi=dpi)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(pix.tobytes("png"))
                paths.append(tmp.name)
    return paths


def _cap_size(img_path: str):
    """Shrink a copy to INFER_LONG_EDGE; originals pass through untouched."""
    img = Image.open(img_path)
    s = min(1.0, INFER_LONG_EDGE / max(img.width, img.height))
    if s >= 1.0:
        return img_path, False
    small = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    small.save(tmp.name)
    tmp.close()
    return tmp.name, True


def _read_image_with_scores(img_path: str):
    """One image -> (texts, scores) in reading order."""
    infer_path, capped = _cap_size(img_path)
    try:
        texts: list[str] = []
        scores: list[float] = []
        for page in get_ocr().predict(infer_path):
            rec_texts = page.get("rec_texts", []) or []
            rec_scores = page.get("rec_scores", []) or []
            for i, t in enumerate(rec_texts):
                texts.append(str(t))
                try:
                    scores.append(float(rec_scores[i]) if i < len(rec_scores) else 0.0)
                except (TypeError, ValueError):
                    scores.append(0.0)
        return texts, scores
    finally:
        if capped:
            try:
                os.unlink(infer_path)
            except OSError:
                pass


def _read_image(img_path: str) -> str:
    """One image -> text lines in reading order (plain-text helper)."""
    texts, _ = _read_image_with_scores(img_path)
    return "\n".join(texts)


def _compute_metrics(text: str, scores: list[float], elapsed: float, page_count: int) -> dict:
    words = text.split()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = len(scores)
    if n:
        avg = sum(scores) / n
        lo, hi = min(scores), max(scores)
    else:
        avg = lo = hi = 0.0
    return {
        "time_taken_sec": round(elapsed, 3),
        "page_count": page_count,
        "pages_with_text": len(lines) and page_count,  # refined per-page below when pdf
        "line_count": len(lines),
        "word_count": len(words),
        "char_count": len(text),
        "avg_confidence": round(avg, 4),
        "avg_confidence_percent": round(avg * 100, 2),
        "min_confidence": round(lo, 4),
        "max_confidence": round(hi, 4),
        "is_empty": not bool(text.strip()),
    }


def paddleocr_extract(file_path: str) -> str:
    """PDF or image -> plain text (newline-joined recognition lines)."""
    return paddleocr_extract_with_details(file_path)["text"]


def paddleocr_extract_with_details(
    file_path: str, dpi: int = 200, max_pages: int = 3
) -> dict:
    """PDF or image -> {"text", "lines", "metrics"}.

    lines: [{text, confidence}] in reading order across pages.
    metrics: time_taken_sec, page/image counts, word/char/line counts,
             avg/min/max confidence (+ percent), engine info.
    Never raises on OCR failure: returns empty text with is_empty=True.
    """
    t0 = time.perf_counter()
    tmp_paths: list[str] = []
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            targets = [file_path]
            is_pdf = False
        elif ext == ".pdf":
            targets = tmp_paths = _render_pdf_pages(file_path, dpi=dpi, max_pages=max_pages)
            is_pdf = True
        else:
            raise ValueError(f"Unsupported file extension: {ext or '(none)'}")

        all_texts: list[str] = []
        all_scores: list[float] = []
        lines: list[dict] = []
        pages_with_text = 0
        for p in targets:
            try:
                texts, scores = _read_image_with_scores(p)
            except Exception as e:
                logger.error(f"[PaddleOCR] Error reading page {p}: {e}")
                continue
            page_text = "\n".join(texts).strip()
            if page_text:
                pages_with_text += 1
                all_texts.append(page_text)
            for t, s in zip(texts, scores):
                lines.append({"text": str(t), "confidence": round(float(s), 4)})

        # Fill gaps if recognizer returned texts without scores
        all_scores = [float(l["confidence"]) for l in lines]
        text = "\n".join(all_texts)
        elapsed = time.perf_counter() - t0

        if not text.strip():
            logger.warning(f"[Stage 2/PaddleOCR] Empty result for {file_path}")
        else:
            logger.info(
                f"[Stage 2/PaddleOCR] {len(targets)} page(s) in {elapsed:.1f}s"
            )

        metrics = _compute_metrics(text, all_scores, elapsed, len(targets))
        metrics["pages_with_text"] = pages_with_text
        metrics.update(
            {
                "engine": f"paddleocr-{_ocr_device or 'unknown'}",
                "device": _ocr_device,
                "is_pdf": is_pdf,
                "dpi": dpi if is_pdf else None,
                "max_pages": max_pages if is_pdf else None,
                "infer_long_edge": INFER_LONG_EDGE,
            }
        )
        return {"text": text, "lines": lines, "metrics": metrics}
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"[PaddleOCR] Error parsing {file_path}: {e}")
        elapsed = time.perf_counter() - t0
        return {
            "text": "",
            "lines": [],
            "metrics": {
                "time_taken_sec": round(elapsed, 3),
                "page_count": 0,
                "pages_with_text": 0,
                "line_count": 0,
                "word_count": 0,
                "char_count": 0,
                "avg_confidence": 0.0,
                "avg_confidence_percent": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "is_empty": True,
                "engine": f"paddleocr-{_ocr_device or 'unknown'}",
                "device": _ocr_device,
                "error": str(e),
            },
        }
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
