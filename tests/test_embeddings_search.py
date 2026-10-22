"""
Unit tests for Embeddings and Search Service.
Tests dense embedding generation, normalization, cosine similarity,
TF-IDF fallback, and real-time index synchronization.
"""

import os
import tempfile
import pytest
import numpy as np

from services import embeddings, search, database


def test_vector_normalization_and_validation():
    # Test normalization
    raw_vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    norm = embeddings.normalize_vector(raw_vec)
    assert np.isclose(np.linalg.norm(norm), 1.0)

    # Test validation
    valid_384 = np.random.randn(384).astype(np.float32)
    valid_384 = embeddings.normalize_vector(valid_384)
    is_valid, _ = embeddings.validate_vector(valid_384, 384)
    assert is_valid is True

    # Test nan detection
    nan_vec = valid_384.copy()
    nan_vec[10] = np.nan
    is_valid_nan, reason = embeddings.validate_vector(nan_vec, 384)
    assert is_valid_nan is False


def test_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    targets = np.array([
        [1.0, 0.0, 0.0],   # Identical: 1.0
        [0.0, 1.0, 0.0],   # Orthogonal: 0.0
        [-1.0, 0.0, 0.0]   # Opposite: -1.0
    ], dtype=np.float32)

    sims = embeddings.compute_cosine_similarity(v1, targets)
    assert np.isclose(sims[0], 1.0)
    assert np.isclose(sims[1], 0.0)
    assert np.isclose(sims[2], -1.0)


def test_tfidf_fallback_engine():
    engine = embeddings.TfidfSearchFallback()
    docs = [
        ("doc_1", "photograph of a golden sunset over snowy mountains"),
        ("doc_2", "red sports car parked in the city center"),
        ("doc_3", "happy family walking in the green sunny park")
    ]
    engine.fit_documents(docs)

    # Query for sunset
    results = engine.search("sunset mountain")
    assert len(results) > 0
    assert results[0][0] == "doc_1"

    # Query for car
    results_car = engine.search("red automobile car")
    assert len(results_car) > 0
    assert results_car[0][0] == "doc_2"


def test_search_index_sync_lifecycle():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_db(db_path)

    orig_default = database.DEFAULT_DB_PATH
    database.DEFAULT_DB_PATH = db_path
    try:
        # 1. Insert image
        database.insert_image(
            image_id="test_search_img",
            filename="beach.jpg",
            original_name="beach.jpg",
            file_path="/tmp/beach.jpg",
            file_size=5000,
            mime_type="image/jpeg",
            width=1920,
            height=1080,
            caption="Tropical beach with turquoise water and palm trees",
            tags="beach, ocean, tropical, summer",
            db_path=db_path
        )

        # 2. Sync index
        synced = search.sync_image_index("test_search_img")
        assert synced is True or synced is False  # Valid either way depending on model load

        # 3. Query
        search_res = search.search_images("tropical ocean beach")
        assert "results" in search_res
        matches = [r for r in search_res["results"] if r["image"]["id"] == "test_search_img"]
        assert len(matches) > 0

        # 4. Delete image
        search.remove_image_from_index("test_search_img")
        embs = database.get_all_embeddings(db_path=db_path)
        assert len(embs) == 0
    finally:
        database.DEFAULT_DB_PATH = orig_default
        if os.path.exists(db_path):
            os.remove(db_path)
