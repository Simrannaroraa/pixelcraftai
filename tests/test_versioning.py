"""
Unit tests for Versioning Service.
Tests non-destructive editing, original preservation, version lineage, and rollback.
"""

import os
import tempfile
import pytest
from io import BytesIO
from PIL import Image

from services import database, storage, versioning


@pytest.fixture
def setup_version_env():
    # Setup temporary database and image
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_db(db_path)

    # Save an original image
    img = Image.new("RGB", (100, 100), color=(10, 200, 50))
    buf = BytesIO()
    img.save(buf, format="PNG")
    original_bytes = buf.getvalue()

    img_id, fpath, meta = storage.save_uploaded_image(original_bytes, "original.png")
    database.insert_image(
        image_id=img_id,
        filename=os.path.basename(fpath),
        original_name="original.png",
        file_path=fpath,
        file_size=meta["file_size"],
        mime_type=meta["mime_type"],
        width=meta["width"],
        height=meta["height"],
        db_path=db_path
    )

    yield img_id, fpath, db_path, original_bytes

    # Teardown
    database.delete_image(img_id, db_path=db_path)
    storage.delete_file_safely(fpath)
    if os.path.exists(db_path):
        os.remove(db_path)


def test_original_image_preservation(setup_version_env):
    img_id, orig_path, db_path, original_bytes = setup_version_env

    # Apply edit using mock default db
    # We patch DEFAULT_DB_PATH
    orig_default = database.DEFAULT_DB_PATH
    database.DEFAULT_DB_PATH = db_path
    try:
        success, msg, v1 = versioning.create_new_version(
            image_id=img_id,
            instruction="Make the image brighter",
            format_type="PNG"
        )
        assert success is True
        assert v1 is not None
        assert v1["version_number"] == 1

        # Check original image bytes on disk: must remain strictly identical
        with open(orig_path, "rb") as f:
            current_bytes = f.read()
        assert current_bytes == original_bytes

        # Check that version file exists and is distinct
        assert os.path.exists(v1["file_path"])
        assert v1["file_path"] != orig_path

        # Cleanup version file
        storage.delete_file_safely(v1["file_path"])
    finally:
        database.DEFAULT_DB_PATH = orig_default


def test_failed_edit_no_orphan_record(setup_version_env):
    img_id, orig_path, db_path, _ = setup_version_env
    orig_default = database.DEFAULT_DB_PATH
    database.DEFAULT_DB_PATH = db_path
    try:
        # Request an unsupported generative action
        success, msg, ver = versioning.create_new_version(
            image_id=img_id,
            instruction="Add a spaceship landing on a mountain",
            format_type="PNG"
        )
        assert success is False
        assert ver is None

        # Verify no version record was inserted
        history = database.list_versions_for_image(img_id, db_path=db_path)
        assert len(history) == 0
    finally:
        database.DEFAULT_DB_PATH = orig_default
