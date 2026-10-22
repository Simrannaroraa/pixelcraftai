"""
Search Service for AI-Powered Image Editing Platform.
Orchestrates index synchronization across uploads, edits, and deletions.
Executes semantic vector search and graceful TF-IDF fallback with honest UI indicators.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from services import database, embeddings, storage

logger = logging.getLogger(__name__)

# Fallback TF-IDF cache
_tfidf_engine = embeddings.TfidfSearchFallback()
_tfidf_synced = False


def build_searchable_text(image_record: Dict[str, Any], latest_version: Optional[Dict[str, Any]] = None) -> str:
    """
    Construct rich, unified text representation of an image for indexing.
    Combines caption, description, tags, original filename, and edit history.
    """
    parts = []
    if image_record.get("caption"):
        parts.append(image_record["caption"])
    if image_record.get("description"):
        parts.append(image_record["description"])
    if image_record.get("tags"):
        parts.append(f"Tags: {image_record['tags']}")
    if image_record.get("original_name"):
        parts.append(f"Filename: {image_record['original_name']}")

    if latest_version and latest_version.get("edit_prompt"):
        parts.append(f"Latest Edit: {latest_version['edit_prompt']}")

    return " | ".join(parts).strip()


def sync_image_index(image_id: str) -> bool:
    """
    Synchronize the embedding index for a single image.
    Generates embedding if SentenceTransformers is available, and updates TF-IDF state.
    """
    global _tfidf_synced
    _tfidf_synced = False  # Mark TF-IDF cache dirty

    img = database.get_image(image_id)
    if not img:
        return False

    latest_v = database.get_latest_version(image_id)
    text_to_index = build_searchable_text(img, latest_v)

    if not text_to_index:
        text_to_index = f"Image {img.get('original_name', 'upload')}"

    # Try dense semantic embedding
    vec = embeddings.generate_embedding(text_to_index)
    if vec is not None:
        blob = embeddings.serialize_vector(vec)
        database.save_embedding(
            image_id=image_id,
            vector_blob=blob,
            dim=embeddings.EMBEDDING_DIM,
            model_name=embeddings.PRIMARY_MODEL_NAME,
            text_indexed=text_to_index
        )
        logger.info("Saved semantic embedding for image %s", image_id)
        return True
    else:
        logger.info("Semantic model unavailable; image %s indexed for TF-IDF fallback", image_id)
        return False


def remove_image_from_index(image_id: str) -> None:
    """Remove image embedding from index."""
    global _tfidf_synced
    _tfidf_synced = False
    database.delete_embedding(image_id)


def rebuild_full_index() -> Dict[str, Any]:
    """
    Rebuild the entire search index from all images in the database.
    Useful when changing embedding models or re-synchronizing after migrations.
    """
    global _tfidf_synced
    all_images = database.list_images(limit=10000)
    indexed_embeddings = 0
    total = len(all_images)

    for img in all_images:
        success = sync_image_index(img["id"])
        if success:
            indexed_embeddings += 1

    _tfidf_synced = False
    return {
        "total_images": total,
        "semantic_indexed": indexed_embeddings,
        "engine": embeddings.get_active_embedding_engine_info()
    }


def _ensure_tfidf_index():
    """Ensure in-memory TF-IDF index is populated with current database content."""
    global _tfidf_engine, _tfidf_synced
    if not _tfidf_synced:
        all_images = database.list_images(limit=10000)
        docs = []
        for img in all_images:
            latest_v = database.get_latest_version(img["id"])
            text = build_searchable_text(img, latest_v)
            docs.append((img["id"], text))
        _tfidf_engine.fit_documents(docs)
        _tfidf_synced = True


def search_images(
    query: str,
    top_k: int = 20,
    min_score: float = 0.05
) -> Dict[str, Any]:
    """
    Execute natural-language search over the image library.
    Selects semantic vector search or TF-IDF fallback based on availability.
    Returns:
    {
        "query": str,
        "engine_used": str,
        "results": [
            {
                "image": dict,
                "latest_version": dict or None,
                "score": float,
                "score_percent": int,
                "display_path": str
            }, ...
        ]
    }
    """
    query = query.strip()
    if not query:
        return {"query": "", "engine_used": "none", "results": []}

    engine_info = embeddings.get_active_embedding_engine_info()
    results = []

    # 1. Attempt Semantic Embedding Search
    if engine_info["mode"] == "semantic_embedding":
        query_vec = embeddings.generate_embedding(query)
        if query_vec is not None:
            all_embs = database.get_all_embeddings()
            if all_embs:
                valid_ids = []
                vec_list = []
                for row in all_embs:
                    v = embeddings.deserialize_vector(row["vector_blob"], dim=row["dim"])
                    if v is not None:
                        valid_ids.append(row["image_id"])
                        vec_list.append(v)

                if vec_list:
                    target_matrix = np.vstack(vec_list)
                    sims = embeddings.compute_cosine_similarity(query_vec, target_matrix)
                    ranked_indices = np.argsort(-sims)

                    for idx in ranked_indices[:top_k]:
                        score = float(sims[idx])
                        if score >= min_score:
                            img_id = valid_ids[idx]
                            img_row = database.get_image(img_id)
                            if img_row:
                                latest_v = database.get_latest_version(img_id)
                                display_path = latest_v["file_path"] if latest_v else img_row["file_path"]
                                results.append({
                                    "image": img_row,
                                    "latest_version": latest_v,
                                    "score": round(score, 4),
                                    "score_percent": int(max(0, min(100, (score + 1.0) / 2.0 * 100)) if score < 0 else min(100, score * 100)),
                                    "display_path": display_path
                                })

                    return {
                        "query": query,
                        "engine_used": "Semantic Vector Search (all-MiniLM-L6-v2)",
                        "engine_mode": "semantic",
                        "results": results
                    }

    # 2. Graceful Fallback: Scikit-learn TF-IDF Search
    _ensure_tfidf_index()
    tfidf_matches = _tfidf_engine.search(query, top_k=top_k)

    for img_id, score in tfidf_matches:
        if score >= min_score:
            img_row = database.get_image(img_id)
            if img_row:
                latest_v = database.get_latest_version(img_id)
                display_path = latest_v["file_path"] if latest_v else img_row["file_path"]
                results.append({
                    "image": img_row,
                    "latest_version": latest_v,
                    "score": round(score, 4),
                    "score_percent": int(min(100, score * 100)),
                    "display_path": display_path
                })

    # If TF-IDF yielded no results, perform basic case-insensitive substring search
    if not results:
        all_imgs = database.list_images(limit=100)
        q_lower = query.lower()
        for img_row in all_imgs:
            latest_v = database.get_latest_version(img_row["id"])
            full_text = build_searchable_text(img_row, latest_v).lower()
            if q_lower in full_text:
                display_path = latest_v["file_path"] if latest_v else img_row["file_path"]
                results.append({
                    "image": img_row,
                    "latest_version": latest_v,
                    "score": 0.5,
                    "score_percent": 50,
                    "display_path": display_path
                })

    return {
        "query": query,
        "engine_used": "Keyword TF-IDF Search (Scikit-Learn Fallback)",
        "engine_mode": "tfidf",
        "results": results
    }
