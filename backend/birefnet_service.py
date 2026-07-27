import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from sam_service import apply_mask_to_image

BIREFNET_MODEL = os.getenv("BIREFNET_MODEL", "ZhengPeng7/BiRefNet_lite")
_INFERENCE_STEPS = (512, 640, 768, 896, 1024)

_model = None
_model_lock = threading.Lock()
_transform_cache: dict[int, transforms.Compose] = {}


@dataclass
class AutoResult:
    result: Image.Image
    mask: np.ndarray
    low_res_mask: Optional[np.ndarray] = None


def get_device() -> torch.device:
    from sam_service import get_device as _device

    return _device()


def _max_inference_size() -> int:
    raw = os.getenv("BIREFNET_SIZE", "").strip()
    if raw:
        try:
            return max(512, min(int(raw), 1024))
        except ValueError:
            pass
    return 768 if get_device().type == "cpu" else 1024


def _pick_inference_size(width: int, height: int) -> int:
    """Match inference cost to image size — full 1024 only when it helps."""
    cap = _max_inference_size()
    longest = max(width, height)

    if longest <= 640:
        target = 512
    elif longest <= 960:
        target = 640
    elif longest <= 1280:
        target = 768
    else:
        target = cap

    target = min(target, cap)
    for size in _INFERENCE_STEPS:
        if size >= target:
            return min(size, cap)
    return cap


def _get_transform(size: int) -> transforms.Compose:
    transform = _transform_cache.get(size)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _transform_cache[size] = transform
    return transform


def get_birefnet_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from transformers import AutoModelForImageSegmentation

            device = get_device()
            model = AutoModelForImageSegmentation.from_pretrained(
                BIREFNET_MODEL, trust_remote_code=True
            )
            model.eval().to(device)
            _model = model
    return _model


def _predict_mask(image: Image.Image) -> np.ndarray:
    model = get_birefnet_model()
    device = get_device()
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    w, h = rgb.size
    infer_size = _pick_inference_size(w, h)

    tensor = _get_transform(infer_size)(rgb).unsqueeze(0).to(device)

    with _model_lock:
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda"):
                    preds = model(tensor)[-1].sigmoid().cpu()
            else:
                preds = model(tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze().numpy()
    mask_img = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8), mode="L")
    if (w, h) != mask_img.size:
        mask_img = mask_img.resize((w, h), Image.BILINEAR)
    return np.array(mask_img, dtype=np.uint8) > 128


def auto_remove_background(image: Image.Image) -> AutoResult:
    mask = _predict_mask(image)
    result = apply_mask_to_image(image, mask)
    return AutoResult(result=result, mask=mask)