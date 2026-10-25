import numpy as np
import pytest

from vision_enhance.metrics import compare_images


def test_identical_images_have_perfect_full_reference_scores():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    result = compare_images(image, image.copy())
    assert result["psnr"] == float("inf")
    assert result["ssim"] == pytest.approx(1.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="identical shapes"):
        compare_images(np.zeros((8, 8), dtype=np.uint8), np.zeros((9, 9), dtype=np.uint8))
