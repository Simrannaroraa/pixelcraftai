"""
Unit tests for Editing Service.
Tests natural-language instruction parser, local transformations,
background removal/replacement, and honest generative disclosure.
"""

import pytest
import numpy as np
from PIL import Image

from services import editing


@pytest.fixture
def test_image():
    # 200x200 red image with blue square in center
    img = Image.new("RGB", (200, 200), color=(240, 50, 50))
    arr = np.array(img)
    arr[60:140, 60:140] = [50, 50, 240]
    return Image.fromarray(arr)


def test_parse_nl_instructions():
    # Background removal
    p1 = editing.parse_natural_language_instruction("Please remove the background")
    assert p1["action"] == "background_removal"

    # Background replacement
    p2 = editing.parse_natural_language_instruction("Change the background to a snowy mountain")
    assert p2["action"] == "background_replacement"
    assert "snow" in p2["parameters"]["backdrop_type"] or "mountain" in p2["parameters"]["backdrop_type"]

    # Lighting enhancement
    p3 = editing.parse_natural_language_instruction("Make the image brighter and increase contrast")
    assert p3["action"] == "lighting_enhancement"
    assert p3["parameters"]["brightness"] > 1.0

    # Color adjustment
    p4 = editing.parse_natural_language_instruction("Make it warmer and black and white")
    assert p4["action"] == "color_adjustment"

    # Unsupported generative addition
    p5 = editing.parse_natural_language_instruction("Add a tree in the background")
    assert p5["action"] == "unsupported_generative"
    assert p5["is_generative"] is True


def test_lighting_enhancement(test_image):
    edited = editing.adjust_lighting(test_image, brightness=1.3, contrast=1.2)
    assert edited.size == test_image.size
    orig_arr = np.array(test_image, dtype=float)
    edit_arr = np.array(edited, dtype=float)
    assert np.mean(edit_arr) > np.mean(orig_arr)


def test_color_adjustment_bw(test_image):
    bw = editing.adjust_color(test_image, saturation=0.0)
    arr = np.array(bw)
    # In grayscale, R == G == B
    diff_rg = np.max(np.abs(arr[:, :, 0].astype(int) - arr[:, :, 1].astype(int)))
    assert diff_rg <= 2


def test_crop_transform(test_image):
    # Test 1:1 aspect ratio on a rectangular 200x100 image
    rect = Image.new("RGB", (200, 100), color=(100, 100, 100))
    cropped, _ = editing.crop_or_transform(rect, aspect_ratio="1:1")
    assert cropped.size == (100, 100)


def test_execute_edit_generative_honesty(test_image):
    # Asking for generative addition without an external API must fail gracefully with honest explanation
    res, method, params, success = editing.execute_edit(test_image, "Add a green sports car next to the building")
    assert success is False
    assert res is None
    assert "generative diffusion api" in method.lower()
