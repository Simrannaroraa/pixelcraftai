"""
Version History Service for AI-Powered Image Editing Platform.
Coordinates version persistence, lineage branching, rollbacks, and before/after previews
while strictly preserving the original image.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from services import database, storage, editing

logger = logging.getLogger(__name__)


def get_active_image_state(image_id: str, version_id: Optional[str] = None) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """
    Get the PIL image and metadata for either a specific version or the latest version (or original).
    Returns (pil_image, info_dict).
    """
    image_row = database.get_image(image_id)
    if not image_row:
        raise ValueError(f"Image with ID {image_id} not found.")

    if version_id:
        v_row = database.get_version(version_id)
        if v_row and v_row["image_id"] == image_id:
            img = storage.load_image(v_row["file_path"])
            return img, {
                "type": "version",
                "version_id": v_row["id"],
                "version_number": v_row["version_number"],
                "file_path": v_row["file_path"],
                "edit_prompt": v_row["edit_prompt"],
                "method": v_row["method"],
                "parameters": v_row["parameters"],
                "created_at": v_row["created_at"],
                "width": v_row["width"],
                "height": v_row["height"],
                "file_size": v_row["file_size"],
                "original": image_row
            }

    # If no version_id specified, try latest version
    latest_v = database.get_latest_version(image_id)
    if latest_v:
        img = storage.load_image(latest_v["file_path"])
        return img, {
            "type": "version",
            "version_id": latest_v["id"],
            "version_number": latest_v["version_number"],
            "file_path": latest_v["file_path"],
            "edit_prompt": latest_v["edit_prompt"],
            "method": latest_v["method"],
            "parameters": latest_v["parameters"],
            "created_at": latest_v["created_at"],
            "width": latest_v["width"],
            "height": latest_v["height"],
            "file_size": latest_v["file_size"],
            "original": image_row
        }

    # Default to original
    img = storage.load_image(image_row["file_path"])
    return img, {
        "type": "original",
        "version_id": None,
        "version_number": 0,
        "file_path": image_row["file_path"],
        "edit_prompt": "Original Upload",
        "method": "original_upload",
        "parameters": {},
        "created_at": image_row["created_at"],
        "width": image_row["width"],
        "height": image_row["height"],
        "file_size": image_row["file_size"],
        "original": image_row
    }


def create_new_version(
    image_id: str,
    instruction: str,
    parent_version_id: Optional[str] = None,
    custom_params: Optional[Dict[str, Any]] = None,
    format_type: str = "PNG"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Apply an edit to the current or selected parent version, save output to disk,
    and persist a new version record. Original file remains untouched.
    Returns (success, message_or_error, version_dict_or_none).
    """
    # 1. Retrieve the parent image to edit
    try:
        parent_img, parent_info = get_active_image_state(image_id, version_id=parent_version_id)
    except Exception as e:
        return False, f"Could not load source image: {e}", None

    if parent_img is None:
        return False, "Failed to decode source image from storage.", None

    # 2. Execute edit
    edited_img, method_desc, params, success = editing.execute_edit(
        parent_img, instruction, custom_params=custom_params
    )

    if not success or edited_img is None:
        # Failed or unsupported edit: DO NOT create version record
        return False, method_desc, None

    # 3. Save new version to disk
    next_ver_num = database.get_next_version_number(image_id)
    try:
        ver_id, file_path, meta = storage.save_version_image(
            edited_img, image_id, next_ver_num, fmt=format_type
        )
    except Exception as e:
        logger.error(f"Failed to save version image file: {e}")
        return False, f"Disk write failure: {e}", None

    # 4. Save version to database
    try:
        actual_parent_id = parent_info.get("version_id")
        ver_record = database.insert_version(
            version_id=ver_id,
            image_id=image_id,
            parent_version_id=actual_parent_id,
            version_number=next_ver_num,
            file_path=file_path,
            edit_prompt=instruction,
            method=method_desc,
            parameters=params,
            width=meta["width"],
            height=meta["height"],
            file_size=meta["file_size"]
        )
        return True, "Version created successfully.", ver_record
    except Exception as e:
        # Clean up orphaned file on database error
        storage.delete_file_safely(file_path)
        logger.error(f"Failed to record version in DB: {e}")
        return False, f"Database transaction error: {e}", None


def get_version_history(image_id: str) -> List[Dict[str, Any]]:
    """Return complete chronological version history for an image."""
    return database.list_versions_for_image(image_id)


def get_comparison_pair(
    image_id: str,
    version_id: Optional[str] = None
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Dict[str, Any]]:
    """
    Get (original_or_parent_image, edited_image, comparison_metadata) for before/after view.
    """
    img_row = database.get_image(image_id)
    if not img_row:
        return None, None, {}

    original_img = storage.load_image(img_row["file_path"])

    if version_id:
        v_row = database.get_version(version_id)
        if v_row:
            after_img = storage.load_image(v_row["file_path"])
            # If version has a parent version, use parent as "before"
            before_img = original_img
            if v_row["parent_version_id"]:
                p_row = database.get_version(v_row["parent_version_id"])
                if p_row:
                    before_img = storage.load_image(p_row["file_path"])

            return before_img, after_img, {
                "version_number": v_row["version_number"],
                "prompt": v_row["edit_prompt"],
                "method": v_row["method"],
                "timestamp": v_row["created_at"]
            }

    # Latest version compared to original
    latest_v = database.get_latest_version(image_id)
    if latest_v:
        after_img = storage.load_image(latest_v["file_path"])
        return original_img, after_img, {
            "version_number": latest_v["version_number"],
            "prompt": latest_v["edit_prompt"],
            "method": latest_v["method"],
            "timestamp": latest_v["created_at"]
        }

    return original_img, original_img, {
        "version_number": 0,
        "prompt": "Original Image",
        "method": "original",
        "timestamp": img_row["created_at"]
    }
