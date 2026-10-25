"""Classical image enhancement methods operating on OpenCV BGR images."""
from __future__ import annotations

import cv2
import numpy as np

METHODS = ("clahe", "gamma", "unsharp")


def enhance_image(
    image: np.ndarray,
    *,
    method: str,
    gamma: float = 0.8,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    sharpen_amount: float = 1.0,
) -> np.ndarray:
    """Enhance a uint8 BGR image without mutating the input."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 BGR array")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")

    normalized_method = method.lower()
    if normalized_method not in METHODS:
        raise ValueError(f"Unsupported method {method!r}; choose one of {METHODS}")

    if normalized_method == "clahe":
        if clip_limit <= 0 or tile_grid_size < 1:
            raise ValueError("clip_limit and tile_grid_size must be positive")
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
        )
        enhanced_l = clahe.apply(lightness)
        enhanced = cv2.cvtColor(
            cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR
        )
    elif normalized_method == "gamma":
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be a finite positive number")
        lookup = np.clip(
            np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        enhanced = cv2.LUT(image, lookup)
    else:
        if not np.isfinite(sharpen_amount) or sharpen_amount < 0:
            raise ValueError("sharpen_amount must be a finite non-negative number")
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
        enhanced = cv2.addWeighted(
            image, 1.0 + float(sharpen_amount), blurred, -float(sharpen_amount), 0
        )

    return enhanced
