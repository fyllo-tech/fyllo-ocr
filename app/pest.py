"""Pest detection backend: per-pest YOLO11 models, tiled like production.

Mirrors pest-trap/pest_detection/image_group_processor_lambda.py
(run_tiled_inference + non_max_suppression_distance): 1024px tiles with
0.2 overlap, zero-padded edges, per-model conf from model_config.json,
then distance-based NMS across all models (same class only). No S3, no
Fyllo API, no trap classifier gate: image bytes in, detections out.

Models are not baked into the image. Mount pest-trap's
pest_detection/model_r24/ (the .pt files + model_config.json) and point
PEST_MODEL_DIR at it.

Env:
  PEST_MODEL_DIR (default /code/pest_models)
  PEST_DEVICE    (default auto) -> cuda:0 if CUDA is available, else cpu
  PEST_BATCH     (default 16)   -> tiles per predict() call
"""
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]  # surfaced as error in load_models()

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]  # surfaced as error in load_models()

logger = logging.getLogger("fyllo_ocr")

TILE_SIZE = 1024
OVERLAP_RATIO = 0.2
CONF_DEFAULT = 0.4
NMS_DEFAULT = 20

MODEL_DIR = Path(os.getenv("PEST_MODEL_DIR", "/code/pest_models"))
BATCH = max(1, int(os.getenv("PEST_BATCH", "16")))

_models: dict[str, tuple] = {}  # name -> (YOLO, {"conf", "nms"})
_device: str | None = None
# Ultralytics predictors are not thread-safe; FastAPI runs sync routes in a pool.
_lock = threading.Lock()


def resolve_device() -> str:
    raw = os.getenv("PEST_DEVICE", "auto").strip().lower()
    if raw in ("", "auto"):
        return "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
    return raw


def load_models() -> dict[str, tuple]:
    """Load every .pt in MODEL_DIR once (server startup). Idempotent."""
    global _device
    with _lock:
        if _models:
            return _models
        if YOLO is None or torch is None:
            raise RuntimeError("ultralytics/torch not installed. pip install -r requirements.txt")
        if not MODEL_DIR.is_dir():
            raise RuntimeError(f"PEST_MODEL_DIR {MODEL_DIR} not found")
        cfg_file = MODEL_DIR / "model_config.json"
        cfg_map = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        _device = resolve_device()
        for p in sorted(MODEL_DIR.glob("*.pt")):
            cfg = cfg_map.get(p.name, {})
            model = YOLO(str(p))
            model.to(_device)
            _models[p.stem] = (model, {"conf": float(cfg.get("conf", CONF_DEFAULT)),
                                       "nms": float(cfg.get("nms", NMS_DEFAULT))})
        if not _models:
            raise RuntimeError(f"No .pt models in {MODEL_DIR}")
        logger.info(f"[pest] loaded {sorted(_models)} on {_device}")
        return _models


def get_info() -> dict:
    return {
        "loaded": sorted(_models),
        "device": _device or resolve_device(),
        "model_dir": str(MODEL_DIR),
        "configs": {n: c for n, (_, c) in sorted(_models.items())},
    }


def decode_image(blob: bytes) -> np.ndarray:
    """Bytes -> BGR uint8 (as cv2.imread in production). Pillow fallback for AVIF etc."""
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        rgb = np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
    except Exception as e:
        raise ValueError(f"Could not decode image: {e}")
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _tiles(image: np.ndarray):
    h, w = image.shape[:2]
    step = int(TILE_SIZE * (1 - OVERLAP_RATIO))
    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = image[y:y + TILE_SIZE, x:x + TILE_SIZE]
            if tile.shape[:2] != (TILE_SIZE, TILE_SIZE):
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            yield x, y, tile


def non_max_suppression_distance(predictions: list) -> list:
    """Same as production: keep highest score, drop same-class centres closer than max(nms_a, nms_b)."""
    preds = sorted(predictions, key=lambda p: p[4], reverse=True)
    final = []
    while preds:
        cur = preds.pop(0)
        final.append(cur)
        preds = [o for o in preds
                 if o[5] != cur[5] or np.hypot(cur[0] - o[0], cur[1] - o[1]) >= max(cur[6], o[6])]
    return final


def detect(image: np.ndarray, model_names: list[str] | None = None) -> dict:
    """Tiled inference over the whole image with the selected models (default: all)."""
    models = load_models()
    names = model_names or sorted(models)
    unknown = [n for n in names if n not in models]
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Available: {sorted(models)}")

    h, w = image.shape[:2]
    tiles = list(_tiles(image))
    preds, per_model_sec = [], {}
    with _lock:
        for name in names:
            model, cfg = models[name]
            t0 = time.perf_counter()
            for i in range(0, len(tiles), BATCH):
                chunk = tiles[i:i + BATCH]
                results = model.predict([t for _, _, t in chunk], device=_device, conf=cfg["conf"], verbose=False)
                for (x, y, _), r in zip(chunk, results):
                    for box in r.boxes:
                        cx, cy, bw, bh = box.xywh[0].tolist()
                        gx, gy = x + cx, y + cy
                        if gx < w and gy < h:
                            cls_idx = int(box.cls[0])
                            cls_name = model.names.get(cls_idx, str(cls_idx)) if model.names else str(cls_idx)
                            preds.append([gx, gy, bw, bh, float(box.conf[0]), cls_name, cfg["nms"], name])
            per_model_sec[name] = round(time.perf_counter() - t0, 3)

    merged = non_max_suppression_distance(preds)
    counts = {n: 0 for n in names}
    for p in merged:
        counts[p[7]] += 1
    return {
        "width": w,
        "height": h,
        "tiles": len(tiles),
        "detections": [
            {"model": p[7], "class": p[5], "cx": round(p[0], 1), "cy": round(p[1], 1),
             "w": round(p[2], 1), "h": round(p[3], 1), "score": round(p[4], 4)}
            for p in merged
        ],
        "counts": counts,
        "per_model_sec": per_model_sec,
    }
