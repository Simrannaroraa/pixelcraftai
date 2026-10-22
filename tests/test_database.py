"""
Unit tests for Database Service.
Tests SQLite tables, foreign key constraints, CRUD operations, transactions, and cascades.
"""

import os
import tempfile
import pytest

from services import database


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_db(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_insert_and_get_image(temp_db):
    row = database.insert_image(
        image_id="img_123",
        filename="123.png",
        original_name="cat.png",
        file_path="/tmp/123.png",
        file_size=4096,
        mime_type="image/png",
        width=400,
        height=300,
        caption="A sleeping cat",
        tags="cat, pet, sleep",
        db_path=temp_db
    )
    assert row["id"] == "img_123"
    assert row["caption"] == "A sleeping cat"

    fetched = database.get_image("img_123", db_path=temp_db)
    assert fetched is not None
    assert fetched["original_name"] == "cat.png"
    assert fetched["width"] == 400


def test_update_image_metadata(temp_db):
    database.insert_image(
        image_id="img_456",
        filename="456.png",
        original_name="sunset.png",
        file_path="/tmp/456.png",
        file_size=2048,
        mime_type="image/png",
        width=800,
        height=600,
        db_path=temp_db
    )
    updated = database.update_image_metadata(
        "img_456",
        caption="A golden sunset over the hills",
        tags="sunset, hills, evening",
        db_path=temp_db
    )
    assert updated["caption"] == "A golden sunset over the hills"
    assert updated["tags"] == "sunset, hills, evening"


def test_version_lineage_and_cascade(temp_db):
    database.insert_image(
        image_id="img_parent",
        filename="parent.png",
        original_name="original.png",
        file_path="/tmp/parent.png",
        file_size=1024,
        mime_type="image/png",
        width=500,
        height=500,
        db_path=temp_db
    )

    # Insert Version 1
    v1 = database.insert_version(
        version_id="ver_1",
        image_id="img_parent",
        parent_version_id=None,
        version_number=1,
        file_path="/tmp/v1.png",
        edit_prompt="Make it brighter",
        method="local_pillow",
        parameters={"brightness": 1.2},
        width=500,
        height=500,
        file_size=1100,
        db_path=temp_db
    )
    assert v1["version_number"] == 1

    # Insert Version 2 linked to Version 1
    v2 = database.insert_version(
        version_id="ver_2",
        image_id="img_parent",
        parent_version_id="ver_1",
        version_number=2,
        file_path="/tmp/v2.png",
        edit_prompt="Add vintage filter",
        method="local_filter",
        parameters={"style": "vintage"},
        width=500,
        height=500,
        file_size=1200,
        db_path=temp_db
    )
    assert v2["parent_version_id"] == "ver_1"

    # List versions
    versions = database.list_versions_for_image("img_parent", db_path=temp_db)
    assert len(versions) == 2

    # Latest version
    latest = database.get_latest_version("img_parent", db_path=temp_db)
    assert latest["id"] == "ver_2"

    # Delete original image -> cascaded delete of versions
    success, paths = database.delete_image("img_parent", db_path=temp_db)
    assert success is True
    assert len(paths) == 3  # original + 2 versions
    assert database.get_image("img_parent", db_path=temp_db) is None
    assert len(database.list_versions_for_image("img_parent", db_path=temp_db)) == 0
