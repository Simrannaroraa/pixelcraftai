"""
Database Service for AI-Powered Image Editing Platform.
Handles SQLite connection, schema migrations, and CRUD operations for images,
versions, and embeddings with ACID transactions and integrity constraints.
"""

import os
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "image_editor.db"
)


def get_actual_db_path(db_path: Optional[str] = None) -> str:
    """Return db_path if provided, else current DEFAULT_DB_PATH."""
    return db_path if db_path is not None else DEFAULT_DB_PATH


@contextmanager
def get_db_connection(db_path: Optional[str] = None):
    """Context manager for SQLite database connection with row factory and foreign keys."""
    actual_path = get_actual_db_path(db_path)
    os.makedirs(os.path.dirname(actual_path), exist_ok=True)
    conn = sqlite3.connect(actual_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction rolled back due to error: {e}")
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database schema with tables, indexes, and triggers."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                caption TEXT DEFAULT '',
                description TEXT DEFAULT '',
                tags TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                parent_version_id TEXT,
                version_number INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                edit_prompt TEXT NOT NULL,
                method TEXT NOT NULL,
                parameters TEXT DEFAULT '{}',
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (parent_version_id) REFERENCES versions(id) ON DELETE SET NULL
            );
        """)

        # 3. Embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL UNIQUE,
                vector_blob BLOB NOT NULL,
                dim INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                text_indexed TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            );
        """)

        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_at DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_image_id ON versions(image_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_parent ON versions(parent_version_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_image_id ON embeddings(image_id);")

        actual_path = get_actual_db_path(db_path)
        logger.info("Database initialized successfully at %s", actual_path)


# ==================== Images CRUD ====================

def insert_image(
    image_id: str,
    filename: str,
    original_name: str,
    file_path: str,
    file_size: int,
    mime_type: str,
    width: int,
    height: int,
    caption: str = "",
    description: str = "",
    tags: str = "",
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Insert a new image record into the database."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO images (
                id, filename, original_name, file_path, file_size,
                mime_type, width, height, caption, description, tags, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            image_id, filename, original_name, file_path, file_size,
            mime_type, width, height, caption, description, tags
        ))
    return get_image(image_id, db_path=db_path)


def get_image(image_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve an image by its unique ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_images(
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "created_at DESC",
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List images with pagination and sorting."""
    allowed_sorts = {
        "created_at DESC": "created_at DESC",
        "created_at ASC": "created_at ASC",
        "original_name ASC": "original_name ASC",
        "file_size DESC": "file_size DESC"
    }
    order_clause = allowed_sorts.get(sort_by, "created_at DESC")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        query = f"SELECT * FROM images ORDER BY {order_clause} LIMIT ? OFFSET ?"
        cursor.execute(query, (limit, offset))
        return [dict(row) for row in cursor.fetchall()]


def count_images(db_path: Optional[str] = None) -> int:
    """Return total count of images in database."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM images")
        return cursor.fetchone()[0]


def update_image_metadata(
    image_id: str,
    caption: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None,
    db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Update caption, description, or tags for an image."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        updates = []
        params = []
        if caption is not None:
            updates.append("caption = ?")
            params.append(caption)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if tags is not None:
            updates.append("tags = ?")
            params.append(tags)

        if not updates:
            return get_image(image_id, db_path=db_path)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(image_id)
        cursor.execute(f"UPDATE images SET {', '.join(updates)} WHERE id = ?", params)
    return get_image(image_id, db_path=db_path)


def delete_image(image_id: str, db_path: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    Delete an image and cascade delete versions and embeddings.
    Returns (success, list_of_file_paths_to_clean_up).
    """
    paths_to_delete = []
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        # Find original file
        cursor.execute("SELECT file_path FROM images WHERE id = ?", (image_id,))
        img_row = cursor.fetchone()
        if not img_row:
            return False, []
        paths_to_delete.append(img_row["file_path"])

        # Find version files
        cursor.execute("SELECT file_path FROM versions WHERE image_id = ?", (image_id,))
        for v_row in cursor.fetchall():
            paths_to_delete.append(v_row["file_path"])

        # Delete from DB (foreign keys cascade)
        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        return True, paths_to_delete


# ==================== Versions CRUD ====================

def insert_version(
    version_id: str,
    image_id: str,
    parent_version_id: Optional[str],
    version_number: int,
    file_path: str,
    edit_prompt: str,
    method: str,
    parameters: Dict[str, Any],
    width: int,
    height: int,
    file_size: int,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Record an image edit as a new persistent version."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO versions (
                id, image_id, parent_version_id, version_number,
                file_path, edit_prompt, method, parameters,
                width, height, file_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version_id, image_id, parent_version_id, version_number,
            file_path, edit_prompt, method, json.dumps(parameters),
            width, height, file_size
        ))
    return get_version(version_id, db_path=db_path)


def get_version(version_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a version by its ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM versions WHERE id = ?", (version_id,))
        row = cursor.fetchone()
        if row:
            d = dict(row)
            try:
                d["parameters"] = json.loads(d["parameters"]) if d["parameters"] else {}
            except Exception:
                d["parameters"] = {}
            return d
        return None


def list_versions_for_image(
    image_id: str,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List all versions for a given original image, in chronological order."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM versions WHERE image_id = ? ORDER BY version_number ASC",
            (image_id,)
        )
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            try:
                d["parameters"] = json.loads(d["parameters"]) if d["parameters"] else {}
            except Exception:
                d["parameters"] = {}
            results.append(d)
        return results


def get_latest_version(
    image_id: str,
    db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Retrieve the latest version for an image, or None if only the original exists."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM versions WHERE image_id = ? ORDER BY version_number DESC LIMIT 1",
            (image_id,)
        )
        row = cursor.fetchone()
        if row:
            d = dict(row)
            try:
                d["parameters"] = json.loads(d["parameters"]) if d["parameters"] else {}
            except Exception:
                d["parameters"] = {}
            return d
        return None


def get_next_version_number(image_id: str, db_path: Optional[str] = None) -> int:
    """Calculate the next version sequence number for an image."""
    latest = get_latest_version(image_id, db_path=db_path)
    return (latest["version_number"] + 1) if latest else 1


def delete_version(
    version_id: str,
    db_path: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Delete a specific version. Returns (success, file_path_to_clean_up)."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM versions WHERE id = ?", (version_id,))
        row = cursor.fetchone()
        if not row:
            return False, None
        path = row["file_path"]
        cursor.execute("DELETE FROM versions WHERE id = ?", (version_id,))
        return True, path


# ==================== Embeddings CRUD ====================

def save_embedding(
    image_id: str,
    vector_blob: bytes,
    dim: int,
    model_name: str,
    text_indexed: str,
    db_path: Optional[str] = None
) -> None:
    """Store or update embedding vector for an image."""
    emb_id = f"emb_{image_id}"
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO embeddings (id, image_id, vector_blob, dim, model_name, text_indexed, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(image_id) DO UPDATE SET
                vector_blob = excluded.vector_blob,
                dim = excluded.dim,
                model_name = excluded.model_name,
                text_indexed = excluded.text_indexed,
                updated_at = CURRENT_TIMESTAMP
        """, (emb_id, image_id, vector_blob, dim, model_name, text_indexed))


def get_all_embeddings(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all embeddings for in-memory similarity matching."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM embeddings")
        return [dict(row) for row in cursor.fetchall()]


def delete_embedding(image_id: str, db_path: Optional[str] = None) -> bool:
    """Remove embedding record for an image."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM embeddings WHERE image_id = ?", (image_id,))
        return cursor.rowcount > 0
