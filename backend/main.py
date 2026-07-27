import asyncio
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from birefnet_service import auto_remove_background, get_birefnet_model
from image_utils import fit_image_for_processing, image_to_png_bytes
from sam_service import embed_session_image, get_device, mask_to_png_bytes, segment_image

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _cors_origins() -> list[str]:
    raw = os.getenv("CUTOUT_ALLOWED_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return ["*"]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


sessions: dict[str, dict] = {}
MAX_SESSIONS = int(os.getenv("CUTOUT_MAX_SESSIONS", "150"))
SESSION_TTL_SECONDS = int(os.getenv("CUTOUT_SESSION_TTL", str(2 * 3600)))
_ml_semaphore = asyncio.Semaphore(int(os.getenv("CUTOUT_MAX_CONCURRENT_ML", "1")))


async def _run_ml(fn, *args, **kwargs):
    async with _ml_semaphore:
        return await asyncio.to_thread(fn, *args, **kwargs)


def _ensure_low_res_mask(session: dict) -> None:
    if session.get("low_res_mask") is None and session.get("mask") is not None:
        from sam_service import mask_to_sam_lowres

        session["low_res_mask"] = mask_to_sam_lowres(session["mask"])


class Point(BaseModel):
    x: int
    y: int
    label: int = Field(default=1, ge=0, le=1)


class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class SessionRequest(BaseModel):
    session_id: str


class SegmentRequest(BaseModel):
    session_id: str
    points: list[Point] = Field(default_factory=list)
    box: Optional[Box] = None
    mask_index: Optional[int] = Field(default=None, ge=0, le=2)


def _new_session_data(image_rgb: Image.Image) -> dict:
    return {
        "image": image_rgb,
        "image_np": np.asarray(image_rgb),
        "points": [],
        "box": None,
        "result": None,
        "mask": None,
        "low_res_mask": None,
        "auto_mask": None,
        "auto_result": None,
        "mode": "auto",
        "cached_masks": None,
        "cached_scores": None,
        "cached_low_res": None,
        "sam_embedded": False,
        "created_at": time.time(),
        "png_original": None,
        "png_preview": None,
        "png_mask": None,
    }


def _touch_session(session: dict) -> None:
    session["created_at"] = time.time()


def _prune_sessions() -> None:
    now = time.time()
    expired = [
        sid
        for sid, data in sessions.items()
        if now - float(data.get("created_at", now)) > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        sessions.pop(sid, None)

    if len(sessions) <= MAX_SESSIONS:
        return

    ordered = sorted(sessions.items(), key=lambda item: float(item[1].get("created_at", 0)))
    for sid, _ in ordered[: len(sessions) - MAX_SESSIONS]:
        sessions.pop(sid, None)


def _apply_auto_result(session: dict, result) -> None:
    session["result"] = result.result
    session["mask"] = result.mask
    session["low_res_mask"] = result.low_res_mask
    session["auto_mask"] = result.mask
    session["auto_result"] = result.result
    session["mode"] = "auto"
    session["points"] = []
    session["box"] = None
    session["cached_masks"] = None
    session["cached_scores"] = None
    session["cached_low_res"] = None
    session["png_preview"] = image_to_png_bytes(result.result, fast=True)
    session["png_mask"] = None
    _touch_session(session)


async def _read_upload_image(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are allowed")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception:
        raise HTTPException(400, "Could not read the image")

    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")

    return image


async def _session_prune_loop() -> None:
    while True:
        await asyncio.sleep(900)
        _prune_sessions()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cpu_threads = int(os.getenv("CUTOUT_CPU_THREADS", str(os.cpu_count() or 2)))
    cpu_threads = max(1, min(cpu_threads, os.cpu_count() or cpu_threads))
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True

    prune_task = asyncio.create_task(_session_prune_loop())
    print(f"Device: {get_device()}")
    print(f"PyTorch CPU threads: {cpu_threads}")
    print("Loading BiRefNet...")
    get_birefnet_model()
    print("BiRefNet ready. SAM2 loads on first refine request.")
    yield
    prune_task.cancel()
    sessions.clear()


app = FastAPI(
    title="Cutout Studio",
    description="Free, self-hosted background removal with AI auto-cutout and manual refinement",
    lifespan=lifespan,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": str(get_device()), "active_sessions": len(sessions)}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    image = await _read_upload_image(file)

    session_id = str(uuid.uuid4())
    image_rgb = fit_image_for_processing(image.convert("RGB"))
    _prune_sessions()
    sessions[session_id] = _new_session_data(image_rgb)

    w, h = image_rgb.size
    return {
        "session_id": session_id,
        "width": w,
        "height": h,
        "preview_url": f"/api/preview/{session_id}",
        "original_url": f"/api/original/{session_id}",
    }


@app.post("/api/upload-and-auto")
async def upload_and_auto(file: UploadFile = File(...)):
    """Upload + automatic cutout in one request."""
    image = await _read_upload_image(file)

    session_id = str(uuid.uuid4())
    image_rgb = fit_image_for_processing(image.convert("RGB"))
    _prune_sessions()
    sessions[session_id] = _new_session_data(image_rgb)

    try:
        result = await _run_ml(auto_remove_background, image_rgb)
    except Exception:
        sessions.pop(session_id, None)
        raise HTTPException(500, "Automatic processing failed.")

    session = sessions[session_id]
    _apply_auto_result(session, result)

    w, h = image_rgb.size
    return {
        "session_id": session_id,
        "width": w,
        "height": h,
        "mode": "auto",
        "preview_url": f"/api/preview/{session_id}",
        "original_url": f"/api/original/{session_id}",
        "result_url": f"/api/preview/{session_id}",
        "mask_url": f"/api/mask/{session_id}",
    }


@app.post("/api/prepare-refine/{session_id}")
async def prepare_refine(session_id: str):
    """Pre-compute SAM2 embeddings when the user opens refine mode."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("sam_embedded"):
        return {"ready": True}

    try:
        _ensure_low_res_mask(session)
        await _run_ml(embed_session_image, session_id, session["image"], session.get("image_np"))
        session["sam_embedded"] = True
        _touch_session(session)
    except Exception:
        raise HTTPException(500, "Could not prepare refine mode.")

    return {"ready": True}


@app.get("/api/original/{session_id}")
async def original_image(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    data = session.get("png_original")
    if data is None:
        data = image_to_png_bytes(session["image"], fast=True)
        session["png_original"] = data
    return Response(content=data, media_type="image/png")


@app.get("/api/preview/{session_id}")
async def preview_image(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    image = session["result"] or session["image"]
    if session.get("result") is None:
        data = session.get("png_original")
    else:
        data = session.get("png_preview")
    if data is None:
        data = image_to_png_bytes(image, fast=True)
        if session.get("result") is not None:
            session["png_preview"] = data
        else:
            session["png_original"] = data
    return Response(content=data, media_type="image/png")


@app.get("/api/mask/{session_id}")
async def get_mask_overlay(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    mask: Optional[np.ndarray] = session.get("mask")
    if mask is None:
        raise HTTPException(404, "No mask available")

    cached = session.get("png_mask")
    if cached is None:
        cached = mask_to_png_bytes(mask)
        session["png_mask"] = cached
    return Response(content=cached, media_type="image/png")


@app.post("/api/auto")
async def auto_segment(request: SessionRequest):
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    try:
        result = await _run_ml(auto_remove_background, session["image"])
    except Exception:
        raise HTTPException(500, "Automatic processing failed.")

    _apply_auto_result(session, result)

    return {
        "session_id": request.session_id,
        "mode": "auto",
        "result_url": f"/api/preview/{request.session_id}",
        "mask_url": f"/api/mask/{request.session_id}",
        "original_url": f"/api/original/{request.session_id}",
    }


@app.post("/api/segment")
async def segment(request: SegmentRequest):
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if not request.points and not request.box:
        raise HTTPException(400, "At least one point or a box is required")

    points = [p.model_dump() for p in request.points]
    box = request.box.model_dump() if request.box else None
    box_list = [box["x1"], box["y1"], box["x2"], box["y2"]] if box else None

    session["points"] = points
    session["box"] = box
    session["mode"] = "refine"

    base_mask = session.get("mask")
    mask_input = None
    if request.mask_index is None:
        _ensure_low_res_mask(session)
        if session.get("low_res_mask") is not None:
            mask_input = session["low_res_mask"]

    try:
        result = await _run_ml(
            segment_image,
            session_id=request.session_id,
            image=session["image"],
            points=points,
            box=box_list,
            mask_input=mask_input,
            mask_index=request.mask_index,
            base_mask=base_mask,
            cached_masks=session.get("cached_masks"),
            cached_scores=session.get("cached_scores"),
            cached_low_res=session.get("cached_low_res"),
            image_np=session.get("image_np"),
        )
    except Exception:
        raise HTTPException(500, "Could not process the image. Try again.")

    session["sam_embedded"] = True
    session["result"] = result.result
    session["mask"] = result.mask
    session["low_res_mask"] = result.low_res_mask
    session["cached_masks"] = result.cached_masks
    session["cached_scores"] = result.cached_scores
    session["cached_low_res"] = result.cached_low_res
    session["png_preview"] = image_to_png_bytes(result.result, fast=True)
    session["png_mask"] = None
    _touch_session(session)

    return {
        "session_id": request.session_id,
        "mode": "refine",
        "points_count": len(points),
        "score": round(result.score, 3),
        "result_url": f"/api/preview/{request.session_id}",
        "mask_url": f"/api/mask/{request.session_id}",
        "alternatives": result.alternatives or None,
    }


@app.post("/api/restore-auto/{session_id}")
async def restore_auto(session_id: str):
    """Restore the automatic BiRefNet result after manual edits."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if session.get("auto_result") is None:
        raise HTTPException(400, "No automatic result saved.")

    session["result"] = session["auto_result"]
    session["mask"] = session["auto_mask"]
    session["low_res_mask"] = None
    session["points"] = []
    session["box"] = None
    session["mode"] = "auto"
    session["cached_masks"] = None
    session["cached_scores"] = None
    session["cached_low_res"] = None
    session["png_preview"] = image_to_png_bytes(session["auto_result"], fast=True)
    session["png_mask"] = None
    _touch_session(session)

    return {
        "status": "restored",
        "mode": "auto",
        "result_url": f"/api/preview/{session_id}",
        "mask_url": f"/api/mask/{session_id}",
    }


@app.post("/api/reset/{session_id}")
async def reset_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    session["points"] = []
    session["box"] = None
    session["cached_masks"] = None
    session["cached_scores"] = None
    session["cached_low_res"] = None

    if session.get("auto_result") is not None:
        session["result"] = session["auto_result"]
        session["mask"] = session["auto_mask"]
        session["low_res_mask"] = None
        session["mode"] = "auto"
    else:
        session["result"] = None
        session["mask"] = None
        session["low_res_mask"] = None
        session["mode"] = "auto"

    preview = session.get("result") or session["image"]
    session["png_preview"] = (
        image_to_png_bytes(preview, fast=True) if session.get("result") is not None else None
    )
    session["png_mask"] = None
    _touch_session(session)

    return {"status": "reset", "mode": session["mode"], "preview_url": f"/api/preview/{session_id}"}


@app.get("/api/download/{session_id}")
async def download_result(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    result: Optional[Image.Image] = session.get("result")
    if result is None:
        raise HTTPException(400, "No result yet.")

    return Response(
        content=image_to_png_bytes(result, fast=False),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="cutout-{session_id[:8]}.png"'},
    )


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
