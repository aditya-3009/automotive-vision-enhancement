import numpy as np
import pytest

from vision_enhance.methods import enhance_image


@pytest.fixture
def sample_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


@pytest.mark.parametrize("method", ["clahe", "gamma", "unsharp"])
def test_enhancement_returns_valid_array_without_mutating_input(sample_image, method):
    original = sample_image.copy()
    result = enhance_image(sample_image, method=method)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(sample_image, original)


def test_gamma_must_be_positive(sample_image):
    with pytest.raises(ValueError, match="gamma"):
        enhance_image(sample_image, method="gamma", gamma=0)


def test_invalid_method_is_rejected(sample_image):
    with pytest.raises(ValueError, match="Unsupported"):
        enhance_image(sample_image, method="neural-magic")


def test_grayscale_input_is_rejected():
    with pytest.raises(ValueError, match="HxWx3"):
        enhance_image(np.zeros((10, 10), dtype=np.uint8), method="clahe")
