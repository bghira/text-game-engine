from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_LOCK = threading.Lock()
_MAX_INPUT_CHARS = 512
_SOURCE_SNIPPET_MAX_CHARS = 1200

_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
_DB_PATH = os.path.join(_DB_DIR, "tge_source_embeddings.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_material_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id     TEXT    NOT NULL,
    document_key    TEXT    NOT NULL,
    document_label  TEXT    NOT NULL,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT    NOT NULL,
    embedding       BLOB    NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tge_sm_campaign ON source_material_chunks(campaign_id);
CREATE INDEX IF NOT EXISTS idx_tge_sm_campaign_doc ON source_material_chunks(campaign_id, document_key);
"""


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _MODEL


def _embed(text: str) -> bytes:
    import numpy as np

    model = _get_model()
    vector = model.encode((text or "")[:_MAX_INPUT_CHARS], normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32).tobytes()


def _bytes_to_vector(blob: bytes):
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32)


class SourceMaterialMemory:
    _conn_local = threading.local()

    @classmethod
    def _get_conn(cls) -> sqlite3.Connection:
        conn = getattr(cls._conn_local, "conn", None)
        if conn is not None:
            return conn
        os.makedirs(_DB_DIR, exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA_SQL)
        cls._conn_local.conn = conn
        return conn

    @staticmethod
    def _normalize_source_document_key(value: str) -> str:
        key = str(value or "").strip().lower()
        key = "".join(ch if ch.isalnum() else "-" for ch in key)
        key = "-".join(part for part in key.split("-") if part)
        return key[:80] or "source-material"

    @classmethod
    def _split_source_line_fragments(cls, text: str) -> List[str]:
        clean = str(text or "").strip()
        if not clean:
            return []
        fragments = [line.strip() for line in clean.splitlines() if line.strip()]
        if not fragments:
            fragments = [clean]
        out: List[str] = []
        for fragment in fragments:
            if len(fragment) <= _SOURCE_SNIPPET_MAX_CHARS:
                out.append(fragment)
                continue
            words = fragment.split()
            current: List[str] = []
            current_len = 0
            for word in words:
                wlen = len(word) + (1 if current else 0)
                if current and current_len + wlen > _SOURCE_SNIPPET_MAX_CHARS:
                    out.append(" ".join(current).strip())
                    current = [word]
                    current_len = len(word)
                else:
                    current.append(word)
                    current_len += wlen
            if current:
                out.append(" ".join(current).strip())
        return [s for s in out if s]

    @classmethod
    def source_material_units_from_chunks(cls, chunks: List[str]) -> List[str]:
        units: List[str] = []
        for chunk in chunks or []:
            raw = str(chunk or "")
            if not raw.strip():
                continue
            units.extend(cls._split_source_line_fragments(raw))

        deduped: List[str] = []
        seen = set()
        for unit in units:
            key = " ".join(unit.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(unit[:8000])
        return deduped

    @classmethod
    def list_source_material_documents(
        cls,
        campaign_id: str,
        limit: int = 20,
    ) -> List[Dict[str, object]]:
        try:
            conn = cls._get_conn()
            rows = conn.execute(
                """
                SELECT
                    document_key,
                    document_label,
                    COUNT(*) AS n,
                    MAX(created_at) AS last_at
                FROM source_material_chunks
                WHERE campaign_id = ?
                GROUP BY document_key, document_label
                ORDER BY last_at DESC, n DESC
                LIMIT ?
                """,
                (str(campaign_id), max(1, int(limit))),
            ).fetchall()
            out: List[Dict[str, object]] = []
            for document_key, document_label, count, last_at in rows:
                out.append(
                    {
                        "document_key": str(document_key or ""),
                        "document_label": str(document_label or ""),
                        "chunk_count": int(count or 0),
                        "last_at": str(last_at or ""),
                    }
                )
            return out
        except Exception:
            logger.exception(
                "Source material: list documents failed for campaign %s",
                campaign_id,
            )
            return []

    @classmethod
    def store_source_material_chunks(
        cls,
        campaign_id: str,
        *,
        document_label: str,
        chunks: List[str],
        replace_document: bool = True,
    ) -> Tuple[int, str]:
        try:
            label = " ".join(str(document_label or "").strip().split())[:120]
            if not label:
                label = "source-material"
            document_key = cls._normalize_source_document_key(label)
            clean_chunks = [
                str(chunk or "").strip()[:8000]
                for chunk in (chunks or [])
                if str(chunk or "").strip()
            ]
            if not clean_chunks:
                return 0, document_key
            sentence_units = cls.source_material_units_from_chunks(clean_chunks)
            if not sentence_units:
                return 0, document_key

            conn = cls._get_conn()
            if replace_document:
                conn.execute(
                    """
                    DELETE FROM source_material_chunks
                    WHERE campaign_id = ? AND document_key = ?
                    """,
                    (str(campaign_id), document_key),
                )
            for idx, chunk_text in enumerate(sentence_units, start=1):
                conn.execute(
                    """
                    INSERT INTO source_material_chunks
                    (campaign_id, document_key, document_label, chunk_index, chunk_text, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(campaign_id),
                        document_key,
                        label,
                        idx,
                        chunk_text,
                        _embed(chunk_text),
                    ),
                )
            conn.commit()
            return len(sentence_units), document_key
        except Exception:
            logger.exception(
                "Source material: store failed for campaign %s",
                campaign_id,
            )
            return 0, "source-material"

    @classmethod
    def search_source_material(
        cls,
        query: str,
        campaign_id: str,
        *,
        document_key: Optional[str] = None,
        top_k: int = 5,
        before_lines: int = 0,
        after_lines: int = 0,
    ) -> List[Tuple[str, str, int, str, float]]:
        try:
            import numpy as np

            query_vec = _bytes_to_vector(_embed(query or ""))
            conn = cls._get_conn()
            before_n = max(0, min(50, int(before_lines)))
            after_n = max(0, min(50, int(after_lines)))
            key = str(document_key or "").strip()
            if key:
                rows = conn.execute(
                    """
                    SELECT document_key, document_label, chunk_index, chunk_text, embedding
                    FROM source_material_chunks
                    WHERE campaign_id = ? AND document_key = ?
                    """,
                    (str(campaign_id), key),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT document_key, document_label, chunk_index, chunk_text, embedding
                    FROM source_material_chunks
                    WHERE campaign_id = ?
                    """,
                    (str(campaign_id),),
                ).fetchall()
            if not rows:
                return []

            by_doc: Dict[str, Dict[int, str]] = {}
            scored: List[Tuple[str, str, int, str, float]] = []
            for row_key, row_label, row_chunk_idx, row_chunk_text, row_blob in rows:
                doc_key = str(row_key or "")
                chunk_idx = int(row_chunk_idx or 0)
                chunk_text = str(row_chunk_text or "")
                if chunk_idx > 0 and chunk_text:
                    by_doc.setdefault(doc_key, {})[chunk_idx] = chunk_text
                vec = _bytes_to_vector(row_blob)
                score = float(np.dot(query_vec, vec))
                scored.append(
                    (
                        doc_key,
                        str(row_label or ""),
                        chunk_idx,
                        chunk_text,
                        score,
                    )
                )
            scored.sort(key=lambda t: t[4], reverse=True)
            selected = scored[: max(1, int(top_k))]
            expanded: List[Tuple[str, str, int, str, float]] = []
            mark_center = bool(before_n or after_n)
            for doc_key, doc_label, center_idx, center_text, score in selected:
                doc_chunks = by_doc.get(doc_key, {})
                if center_idx <= 0:
                    expanded.append((doc_key, doc_label, center_idx, center_text, score))
                    continue
                start_idx = max(1, center_idx - before_n)
                end_idx = center_idx + after_n
                window_parts: List[str] = []
                for idx in range(start_idx, end_idx + 1):
                    part = str(doc_chunks.get(idx) or "").strip()
                    if not part:
                        continue
                    if idx == center_idx and mark_center:
                        window_parts.append(f">> {part}")
                    else:
                        window_parts.append(part)
                if not window_parts:
                    window_parts = [center_text]
                expanded.append(
                    (
                        doc_key,
                        doc_label,
                        center_idx,
                        "\n".join(window_parts),
                        score,
                    )
                )
            return expanded
        except Exception:
            logger.exception(
                "Source material: search failed for campaign %s",
                campaign_id,
            )
            return []
