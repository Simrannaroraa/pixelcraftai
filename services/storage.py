"""
Storage Service for AI-Powered Image Editing Platform.
Handles secure file validation, size enforcement, atomic saving, and disk management.
"""

import os
import uuid
import logging
from typing import Tuple, Dict, Any, Optional
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# Base storage paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
VERSIONS_DIR = os.path.join(DATA_DIR, "versions")
THUMBNAILS_DIR = os.path.join(DATA_DIR, "thumbnails")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
DEFAULT_MAX_UPLOAD_SIZE_MB = 15.0


def get_max_upload_size_bytes() -> int:
    """Return max upload size in bytes from environment or default."""
    mb = float(os.getenv("MAX_UPLOAD_SIZE_MB", DEFAULT_MAX_UPLOAD_SIZE_MB))
    return int(mb * 1024 * 1024)


def ensure_storage_dirs() -> None:
    """Ensure all required filesystem directories exist."""
    for d in [DATA_DIR, UPLOADS_DIR, VERSIONS_DIR, THUMBNAILS_DIR]:
        os.makedirs(d, exist_ok=True)


def validate_image_file(file_bytes: bytes, original_filename: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validate extension, file size, and image integrity via PIL.
    Returns (is_valid, error_message_or_empty, metadata_dict_or_none).
    """
    # 1. Check size
    max_size = get_max_upload_size_bytes()
    file_size = len(file_bytes)
    if file_size == 0:
        return False, "Uploaded file is empty (0 bytes).", None
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        return False, f"File exceeds maximum allowed size of {max_mb:.1f} MB.", None

    # 2. Check extension
    ext = ""
    if "." in original_filename:
        ext = original_filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", None

    # 3. Verify image content with Pillow
    try:
        bio = BytesIO(file_bytes)
        img = Image.open(bio)
        # verify() checks header without decoding full pixels
        img.verify()

        # Reopen to read dimensions and format (verify() alters stream pointer)
        bio.seek(0)
        img_reopened = Image.open(bio)
        width, height = img_reopened.size
        img_format = (img_reopened.format or ext).upper()

        mime_map = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp"
        }
        mime_type = mime_map.get(img_format, f"image/{ext}")

        metadata = {
            "width": width,
            "height": height,
            "format": img_format,
            "mime_type": mime_type,
            "file_size": file_size
        }
        return True, "", metadata
    except Exception as e:
        logger.warning(f"Image validation failed for '{original_filename}': {e}")
        return False, "File is not a valid or readable image.", None


def save_uploaded_image(file_bytes: bytes, original_filename: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Save validated upload to disk with a random UUID filename.
    Returns (image_id, file_path, metadata).
    """
    ensure_storage_dirs()
    is_valid, err, metadata = validate_image_file(file_bytes, original_filename)
    if not is_valid or metadata is None:
        raise ValueError(err)

    image_id = str(uuid.uuid4())
    ext = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else "png"
    safe_filename = f"{image_id}.{ext}"
    target_path = os.path.join(UPLOADS_DIR, safe_filename)

    with open(target_path, "wb") as f:
        f.write(file_bytes)

    return image_id, target_path, metadata


def save_version_image(image: Image.Image, image_id: str, version_number: int, fmt: str = "PNG") -> Tuple[str, str, Dict[str, Any]]:
    """
    Save an edited PIL image as a new version.
    Returns (version_id, file_path, metadata).
    """
    ensure_storage_dirs()
    version_id = str(uuid.uuid4())
    ext = fmt.lower()
    if ext == "jpeg":
        ext = "jpg"
    safe_filename = f"v_{image_id}_num{version_number}_{version_id[:8]}.{ext}"
    target_path = os.path.join(VERSIONS_DIR, safe_filename)

    # Convert RGBA to RGB if saving as JPEG
    save_img = image
    if ext in ["jpg", "jpeg"] and image.mode in ("RGBA", "LA", "P"):
        save_img = image.convert("RGB")

    save_img.save(target_path, format=fmt, quality=95)
    file_size = os.path.getsize(target_path)
    width, height = save_img.size

    metadata = {
        "width": width,
        "height": height,
        "file_size": file_size,
        "format": fmt
    }
    return version_id, target_path, metadata


def delete_file_safely(file_path: str) -> bool:
    """Delete a file from disk if it exists, logging any issue."""
    if not file_path:
        return False
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Deleted file safely: %s", file_path)
            return True
    except Exception as e:
        logger.error(f"Error deleting file '{file_path}': {e}")
    return False


def load_image(file_path: str) -> Optional[Image.Image]:
    """Safely open and return a PIL Image from disk."""
    if not os.path.exists(file_path):
        logger.error("Requested file path does not exist: %s", file_path)
        return None
    try:
        img = Image.open(file_path)
        img.load()  # Read pixels into memory
        return img
    except Exception as e:
        logger.error(f"Failed to load image from {file_path}: {e}")
        return None
