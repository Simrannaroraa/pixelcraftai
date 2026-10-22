"""
Embedding Generation and Vector Similarity Service for AI-Powered Image Editing Platform.
Provides local dense embeddings via SentenceTransformers ('all-MiniLM-L6-v2'),
Scikit-learn TF-IDF fallback, L2 vector normalization, and cosine-similarity ranking.
100% free and locally runnable.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

logger = logging.getLogger(__name__)

# Model configurations
PRIMARY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Global cached model instance
_ST_MODEL = None
_MODEL_INITIALIZED = False
_MODEL_LOAD_ERROR = None


def get_embedding_model():
    """
    Lazy load SentenceTransformer model. Returns instance or None if unavailable.
    Caches the model instance in memory.
    """
    global _ST_MODEL, _MODEL_INITIALIZED, _MODEL_LOAD_ERROR
    if _MODEL_INITIALIZED:
        return _ST_MODEL

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model '%s'...", PRIMARY_MODEL_NAME)
        _ST_MODEL = SentenceTransformer(PRIMARY_MODEL_NAME)
        _MODEL_INITIALIZED = True
        _MODEL_LOAD_ERROR = None
        logger.info("SentenceTransformer model loaded successfully.")
        return _ST_MODEL
    except Exception as e:
        _MODEL_INITIALIZED = True
        _MODEL_LOAD_ERROR = str(e)
        _ST_MODEL = None
        logger.warning("SentenceTransformer failed to load: %s. Using TF-IDF fallback.", e)
        return None


def get_active_embedding_engine_info() -> Dict[str, Any]:
    """Return status and details of the active embedding engine."""
    model = get_embedding_model()
    if model is not None:
        return {
            "mode": "semantic_embedding",
            "model_name": PRIMARY_MODEL_NAME,
            "dimension": EMBEDDING_DIM,
            "status": "active",
            "description": "Dense semantic vectors via all-MiniLM-L6-v2"
        }
    else:
        return {
            "mode": "tfidf_fallback",
            "model_name": "scikit-learn/TfidfVectorizer",
            "dimension": "dynamic_vocab",
            "status": "fallback",
            "error": _MODEL_LOAD_ERROR,
            "description": "TF-IDF keyword cosine similarity fallback"
        }


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length (L2 norm) for cosine similarity."""
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec
    return vec / norm


def validate_vector(vec: np.ndarray, expected_dim: int = EMBEDDING_DIM) -> Tuple[bool, str]:
    """Validate numerical properties and dimensionality of embedding vector."""
    if not isinstance(vec, np.ndarray):
        return False, "Vector is not a NumPy array."
    if vec.shape != (expected_dim,):
        return False, f"Invalid vector shape {vec.shape}, expected ({expected_dim},)."
    if not np.all(np.isfinite(vec)):
        return False, "Vector contains NaN or infinite values."
    return True, "Valid"


def generate_embedding(text: str) -> Optional[np.ndarray]:
    """
    Generate a 384-d normalized embedding vector for the input text.
    Returns None if the model is unavailable or text is empty.
    """
    if not text or not text.strip():
        return None

    model = get_embedding_model()
    if model is None:
        return None

    try:
        raw_vec = model.encode(text.strip(), convert_to_numpy=True, show_progress_bar=False)
        vec_f32 = np.asarray(raw_vec, dtype=np.float32)
        norm_vec = normalize_vector(vec_f32)
        is_valid, reason = validate_vector(norm_vec, EMBEDDING_DIM)
        if not is_valid:
            logger.error(f"Generated vector failed validation: {reason}")
            return None
        return norm_vec
    except Exception as e:
        logger.error(f"Error generating embedding with SentenceTransformers: {e}")
        return None


def serialize_vector(vec: np.ndarray) -> bytes:
    """Serialize a NumPy float32 array to raw bytes for database BLOB storage."""
    return vec.astype(np.float32).tobytes()


def deserialize_vector(blob: bytes, dim: int = EMBEDDING_DIM) -> Optional[np.ndarray]:
    """Deserialize raw bytes back to a NumPy float32 array."""
    if not blob:
        return None
    try:
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.shape != (dim,):
            logger.warning(f"Deserialized vector has unexpected size {vec.shape}, expected ({dim},)")
            return None
        return vec
    except Exception as e:
        logger.error(f"Failed to deserialize vector blob: {e}")
        return None


def compute_cosine_similarity(query_vec: np.ndarray, target_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between a normalized query vector and multiple target vectors.
    Since vectors are unit-normalized, cosine similarity equals the dot product.
    """
    if target_vecs.ndim == 1:
        target_vecs = target_vecs.reshape(1, -1)
    # Dot product with clipping to [-1.0, 1.0]
    sims = np.dot(target_vecs, query_vec)
    return np.clip(sims, -1.0, 1.0)


# ==================== Scikit-learn TF-IDF Fallback ====================

class TfidfSearchFallback:
    """
    Scikit-learn TF-IDF search fallback engine when SentenceTransformers is unavailable.
    Performs tokenization, sublinear TF scaling, and cosine similarity.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2)
        )
        self.fitted = False
        self.doc_ids: List[str] = []
        self.tfidf_matrix = None

    def fit_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Fit TF-IDF index on list of (doc_id, text) pairs.
        """
        if not documents:
            self.fitted = False
            self.doc_ids = []
            self.tfidf_matrix = None
            return

        self.doc_ids = [d[0] for d in documents]
        texts = [d[1].strip() if d[1].strip() else "image photograph" for d in documents]

        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.fitted = True
        except ValueError:
            # Empty vocabulary fallback
            self.fitted = False

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Compute TF-IDF cosine similarity scores for query.
        Returns list of (doc_id, similarity_score).
        """
        if not self.fitted or not self.doc_ids or not query.strip():
            return []

        try:
            query_vec = self.vectorizer.transform([query.strip()])
            scores = sklearn_cosine_sim(query_vec, self.tfidf_matrix).flatten()
            ranked_indices = np.argsort(-scores)

            results = []
            for idx in ranked_indices[:top_k]:
                score = float(scores[idx])
                if score > 0.001:  # Non-zero similarity
                    results.append((self.doc_ids[idx], score))
            return results
        except Exception as e:
            logger.error(f"TF-IDF search error: {e}")
            return []
