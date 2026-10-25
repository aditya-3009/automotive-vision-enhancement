"""Image-quality metrics with explicit reference-image requirements."""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compare_images(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Calculate PSNR and SSIM for same-sized, corresponding uint8 images.

    These full-reference metrics should only be interpreted when the reference
    and candidate depict the same scene and are appropriately aligned.
    """
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Images must have identical shapes, got {reference.shape} and {candidate.shape}"
        )
    if reference.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError("Both images must have dtype uint8")
    if reference.ndim not in (2, 3):
        raise ValueError("Images must be grayscale or color arrays")

    psnr = float(peak_signal_noise_ratio(reference, candidate, data_range=255))
    if reference.ndim == 2:
        ssim = float(structural_similarity(reference, candidate, data_range=255))
    else:
        if reference.shape[2] not in (3, 4):
            raise ValueError("Color images must have 3 or 4 channels")
        ssim = float(
            structural_similarity(
                reference, candidate, channel_axis=2, data_range=255
            )
        )
    return {"psnr": psnr, "ssim": ssim}
