"""
Unit tests for Storage Service.
Tests file validation, content inspection, size limits, and safe deletion.
"""

import os
import tempfile
import pytest
from io import BytesIO
from PIL import Image

from services import storage


@pytest.fixture
def sample_image_bytes():
    """Create in-memory RGB PNG image bytes."""
    img = Image.new("RGB", (120, 80), color=(255, 100, 50))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_validate_valid_image(sample_image_bytes):
    valid, msg, meta = storage.validate_image_file(sample_image_bytes, "test_pic.png")
    assert valid is True
    assert msg == ""
    assert meta is not None
    assert meta["width"] == 120
    assert meta["height"] == 80
    assert meta["mime_type"] == "image/png"


def test_validate_corrupt_file():
    corrupt_bytes = b"NOT_A_REAL_IMAGE_HEADER_1234567890"
    valid, msg, meta = storage.validate_image_file(corrupt_bytes, "fake.jpg")
    assert valid is False
    assert "not a valid" in msg.lower()
    assert meta is None


def test_validate_empty_file():
    valid, msg, meta = storage.validate_image_file(b"", "empty.png")
    assert valid is False
    assert "empty" in msg.lower()
    assert meta is None


def test_validate_unsupported_extension(sample_image_bytes):
    valid, msg, meta = storage.validate_image_file(sample_image_bytes, "script.exe")
    assert valid is False
    assert "unsupported file extension" in msg.lower()


def test_save_and_load_image(sample_image_bytes):
    img_id, path, meta = storage.save_uploaded_image(sample_image_bytes, "sample.png")
    assert os.path.exists(path)
    assert meta["width"] == 120

    loaded = storage.load_image(path)
    assert loaded is not None
    assert loaded.size == (120, 80)

    # Clean up
    storage.delete_file_safely(path)
    assert not os.path.exists(path)
