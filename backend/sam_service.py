import base64
import io
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import ndimage

MODEL_ID = os.getenv("SAM2_MODEL", "facebook/sam2.1-hiera-tiny")
FEATHER_SIGMA = float(os.getenv("MASK_FEATHER", "0.8"))

_device: Optional[torch.device] = None
_predictor: Optional[SAM2ImagePredictor] = None
_predictor_lock = threading.Lock()
_embedded_session_id: Optional[str] = None


@dataclass
class SegmentResult:
    result: Image.Image
    mask: np.ndarray
    low_res_mask: np.ndarray
    score: float
    alternatives: list[dict] = field(default_factory=list)
    cached_masks: Optional[list[np.ndarray]] = None
    cached_scores: Optional[list[float]] = None
    cached_low_res: Optional[list[np.ndarray]] = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    return _device


def get_predictor() -> SAM2ImagePredictor:
    global _predictor
    if _predictor is None:
        device = get_device()
        _predictor = SAM2ImagePredictor.from_pretrained(
            MODEL_ID, device=str(device)
        )
    return _predictor


def _image_to_rgb_array(image: Image.Image) -> np.ndarray:
    if image.mode == "RGB":
        return np.asarray(image)
    return np.array(image.convert("RGB"))


def embed_session_image(
    session_id: str,
    image: Image.Image,
    image_np: Optional[np.ndarray] = None,
) -> None:
    """Pre-compute image embeddings once per upload."""
    global _embedded_session_id
    if image_np is None:
        image_np = _image_to_rgb_array(image)
    predictor = get_predictor()
    with _predictor_lock:
        if _embedded_session_id != session_id:
            predictor.set_image(image_np)
            _embedded_session_id = session_id


def _ensure_embedded(
    session_id: str,
    image: Image.Image,
    image_np: Optional[np.ndarray] = None,
) -> np.ndarray:
    if image_np is None:
        image_np = _image_to_rgb_array(image)
    embed_session_image(session_id, image, image_np=image_np)
    return image_np


def _positive_points(points: list[dict]) -> list[tuple[int, int]]:
    return [(p["x"], p["y"]) for p in points if p.get("label", 1) == 1]


def _post_process_mask(mask: np.ndarray) -> np.ndarray:
    """Fill holes and smooth edges without removing existing regions."""
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_closing(mask, iterations=1)
    return mask


def _has_negative_points(points: list[dict]) -> bool:
    return any(p.get("label", 1) == 0 for p in points)


def _hybrid_merge(
    base_mask: np.ndarray,
    sam_mask: np.ndarray,
    points: list[dict],
) -> np.ndarray:
    """
    Merge SAM prediction with an existing mask (e.g. from auto mode).

    - Include clicks → union (add missed regions without losing auto cutout)
    - Exclude clicks → intersect (shrink from auto base)
    - Mixed → union additions then apply SAM refinement
    """
    base = base_mask.astype(bool)
    sam = sam_mask.astype(bool)
    positive = _positive_points(points)
    has_negative = _has_negative_points(points)

    if positive and not has_negative:
        merged = base | sam
    elif has_negative and not positive:
        merged = base & sam
    else:
        merged = (base | sam) & sam

    return _post_process_mask(merged)


def refine_mask(mask: np.ndarray, points: list[dict]) -> np.ndarray:
    """Keep components touching foreground points; fill holes and smooth edges."""
    positive = _positive_points(points)
    if not positive:
        return _post_process_mask(mask)

    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask

    keep = np.zeros(num + 1, dtype=bool)
    h, w = mask.shape
    for x, y in positive:
        if 0 <= y < h and 0 <= x < w:
            label = labeled[y, x]
            if label > 0:
                keep[label] = True

    if not keep.any():
        keep[int(np.argmax(np.bincount(labeled.ravel())))] = True

    refined = keep[labeled]
    return _post_process_mask(refined)


def mask_to_sam_lowres(mask: np.ndarray) -> np.ndarray:
    """Convert full-resolution mask to SAM low-res logits (1x256x256)."""
    mask_f = mask.astype(np.float32, copy=False)
    if mask_f.max() > 1.0:
        mask_f = mask_f / 255.0
    if mask_f.shape != (256, 256):
        pil = Image.fromarray((np.clip(mask_f, 0, 1) * 255).astype(np.uint8))
        mask_f = (
            np.asarray(pil.resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
        )
    eps = 1e-4
    logits = np.log((mask_f + eps) / (1.0 - mask_f + eps))
    return logits[None, :, :]


def apply_mask_to_image(
    image: Image.Image,
    mask: np.ndarray,
    feather: bool = True,
) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if feather and FEATHER_SIGMA > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=FEATHER_SIGMA))
    rgba.putalpha(alpha)
    return rgba


def _mask_thumbnail(image: Image.Image, mask: np.ndarray, size: int = 96) -> str:
    thumb = apply_mask_to_image(image, mask, feather=False)
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumb.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    points: list[dict],
) -> int:
    positive = _positive_points(points)
    if not positive:
        return int(np.argmax(scores))

    best_idx = 0
    best_rank = (-1.0, -1.0)
    h, w = masks.shape[1], masks.shape[2]

    for idx, mask in enumerate(masks):
        coverage = 0
        for x, y in positive:
            if 0 <= y < h and 0 <= x < w and mask[y, x]:
                coverage += 1
        rank = (coverage / len(positive), float(scores[idx]))
        if rank > best_rank:
            best_rank = rank
            best_idx = idx

    return best_idx if best_rank[0] > 0 else int(np.argmax(scores))


def _should_offer_alternatives(points: list[dict], box: Optional[list[int]]) -> bool:
    if box is not None:
        return False
    foreground = [p for p in points if p.get("label", 1) == 1]
    background = [p for p in points if p.get("label", 1) == 0]
    return len(foreground) == 1 and len(background) == 0


def _finalize_mask(
    sam_mask: np.ndarray,
    points: list[dict],
    base_mask: Optional[np.ndarray],
    used_prior: bool,
) -> np.ndarray:
    if base_mask is not None and used_prior:
        return _hybrid_merge(base_mask, sam_mask, points)
    return refine_mask(sam_mask, points)


def segment_image(
    session_id: str,
    image: Image.Image,
    points: list[dict],
    box: Optional[list[int]] = None,
    mask_input: Optional[np.ndarray] = None,
    mask_index: Optional[int] = None,
    base_mask: Optional[np.ndarray] = None,
    cached_masks: Optional[list[np.ndarray]] = None,
    cached_scores: Optional[list[float]] = None,
    cached_low_res: Optional[list[np.ndarray]] = None,
    image_np: Optional[np.ndarray] = None,
) -> SegmentResult:
    image_rgb = image if image.mode == "RGB" else image.convert("RGB")
    image_np = _ensure_embedded(session_id, image_rgb, image_np=image_np)

    if box is not None and not points:
        x1, y1, x2, y2 = box
        points = [{
            "x": int((min(x1, x2) + max(x1, x2)) / 2),
            "y": int((min(y1, y2) + max(y1, y2)) / 2),
            "label": 1,
        }]

    point_coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    point_labels = np.array([p.get("label", 1) for p in points], dtype=np.int32)

    box_np = None
    if box is not None:
        x1, y1, x2, y2 = box
        box_np = np.array(
            [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
            dtype=np.float32,
        )

    used_prior = mask_input is not None or base_mask is not None

    if (
        mask_index is not None
        and cached_masks is not None
        and 0 <= mask_index < len(cached_masks)
    ):
        sam_mask = cached_masks[mask_index]
        mask = _finalize_mask(sam_mask, points, base_mask, used_prior)
        score = cached_scores[mask_index] if cached_scores else 1.0
        low_res = mask_to_sam_lowres(mask)
        result = apply_mask_to_image(image_rgb, mask)
        return SegmentResult(
            result=result,
            mask=mask,
            low_res_mask=low_res,
            score=score,
        )

    multimask = _should_offer_alternatives(points, box) and base_mask is None
    if mask_input is not None and not multimask:
        if mask_input.ndim == 2:
            mask_input = mask_to_sam_lowres(mask_input)
        if mask_input.ndim == 3:
            mask_input = mask_input[None, :, :]

    predictor = get_predictor()
    device = get_device()

    with _predictor_lock:
        with torch.inference_mode():
            if device.type == "cuda":
                autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()

            with autocast_ctx:
                masks, scores, low_res_masks = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_np,
                    mask_input=mask_input,
                    multimask_output=multimask,
                )

    alternatives: list[dict] = []

    final_masks = [
        _finalize_mask(m, points, base_mask, used_prior) for m in masks
    ]
    best_idx = _select_best_mask(masks, scores, points)
    mask = final_masks[best_idx]
    low_res_mask = mask_to_sam_lowres(mask)
    score = float(scores[best_idx])

    cached_masks = None
    cached_scores = None
    cached_low_res = None

    if multimask and len(masks) > 1:
        cached_masks = final_masks
        cached_scores = [float(s) for s in scores]
        cached_low_res = [mask_to_sam_lowres(m) for m in final_masks]
        for idx, (final, mask_score) in enumerate(zip(final_masks, scores)):
            alternatives.append({
                "index": idx,
                "score": round(float(mask_score), 3),
                "selected": idx == best_idx,
                "thumbnail": _mask_thumbnail(image_rgb, final),
            })

    result = apply_mask_to_image(image_rgb, mask)
    return SegmentResult(
        result=result,
        mask=mask,
        low_res_mask=low_res_mask,
        score=score,
        alternatives=alternatives,
        cached_masks=cached_masks,
        cached_scores=cached_scores,
        cached_low_res=cached_low_res,
    )


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_png_bytes(image: Image.Image, *, fast: bool = False) -> bytes:
    from image_utils import image_to_png_bytes as _encode

    return _encode(image, fast=fast)