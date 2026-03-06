from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import hashlib
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
    def _normalize_source_unit_mode(cls, mode: str) -> str:
        mode_clean = str(mode or "line").strip().lower()
        if mode_clean in {"story", "paragraph", "paragraphs", "scene"}:
            return "story"
        if mode_clean in {"rulebook", "line", "lines"}:
            return "rulebook"
        if mode_clean in {"generic", "chunk", "chunked", "dump"}:
            return "generic"
        return "line"

    @classmethod
    def _dedupe_source_units(cls, units: List[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for unit in units:
            key = " ".join(str(unit or "").lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(str(unit).strip()[:8000])
        return deduped

    @staticmethod
    def _is_rulebook_fact_line(line: str) -> bool:
        stripped = str(line or "").strip()
        if ":" not in stripped:
            return False
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            return False
        if len(key) > 140:
            return False
        return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _/\-()&'.]*", key))

    @classmethod
    def source_material_units_from_chunks(cls, chunks: List[str]) -> List[str]:
        return cls.source_material_units_from_chunks_with_mode(chunks, mode="line")

    @classmethod
    def source_material_units_from_chunks_with_mode(
        cls, chunks: List[str], *, mode: str = "line"
    ) -> List[str]:
        """Convert source-material chunks into source lookup units.

        Supported modes:
        - line (default): one unit per non-empty line.
        - story: split each chunk into paragraph units.
        - generic: preserve chunk boundaries.
        """
        source_mode = cls._normalize_source_unit_mode(mode)
        units: List[str] = []
        for chunk in chunks or []:
            raw = str(chunk or "")
            if not raw.strip():
                continue
            if source_mode == "rulebook":
                current_fact: str | None = None
                for line in raw.splitlines():
                    line_clean = line.strip()
                    if not line_clean:
                        continue
                    if cls._is_rulebook_fact_line(line_clean):
                        if current_fact:
                            units.append(current_fact)
                        current_fact = line_clean
                        continue
                    if current_fact:
                        current_fact = f"{current_fact} {line_clean}"
                if current_fact:
                    units.append(current_fact)
                continue
            if source_mode == "story":
                paragraphs = [line.strip() for line in raw.split("\n\n") if line.strip()]
                if not paragraphs:
                    paragraphs = [raw]
                for paragraph in paragraphs:
                    paragraph_unit = " ".join(paragraph.split())
                    if paragraph_unit:
                        units.extend(cls._split_source_line_fragments(paragraph_unit))
                continue

            compact = " ".join(raw.split())
            if compact:
                units.extend(cls._split_source_line_fragments(compact))

        return cls._dedupe_source_units(units)

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
                sample_chunk = ""
                if document_key:
                    sample_rows = conn.execute(
                        """
                        SELECT chunk_text
                        FROM source_material_chunks
                        WHERE campaign_id = ? AND document_key = ?
                        ORDER BY chunk_index ASC
                        LIMIT 6
                        """,
                        (str(campaign_id), document_key),
                    ).fetchall()
                    sample_parts = [
                        str(sample_row[0] or "").strip()
                        for sample_row in sample_rows
                        if str(sample_row[0] or "").strip()
                    ]
                    if sample_parts:
                        sample_chunk = "\n".join(sample_parts)
                out.append(
                    {
                        "document_key": str(document_key or ""),
                        "document_label": str(document_label or ""),
                        "chunk_count": int(count or 0),
                        "last_at": str(last_at or ""),
                        "sample_chunk": sample_chunk,
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
    def get_source_material_document_units(
        cls,
        campaign_id: str,
        document_key: str,
    ) -> List[str]:
        try:
            key = str(document_key or "").strip()
            if not key:
                return []
            conn = cls._get_conn()
            rows = conn.execute(
                """
                SELECT chunk_text
                FROM source_material_chunks
                WHERE campaign_id = ? AND document_key = ?
                ORDER BY chunk_index ASC
                """,
                (str(campaign_id), key),
            ).fetchall()
            return [
                str(row[0] or "").strip()
                for row in rows
                if str(row[0] or "").strip()
            ]
        except Exception:
            logger.exception(
                "Source material: get document units failed for campaign %s key %s",
                campaign_id,
                document_key,
            )
            return []

    @classmethod
    def _source_units_signature(cls, units: List[str]) -> str:
        normalized = "\n".join(
            " ".join(str(unit or "").strip().lower().split())
            for unit in units
            if str(unit or "").strip()
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def find_duplicate_source_material_document(
        cls,
        campaign_id: str,
        *,
        chunks: List[str],
        source_mode: str = "line",
    ) -> Optional[Dict[str, object]]:
        try:
            candidate_units = cls.source_material_units_from_chunks_with_mode(
                chunks,
                mode=source_mode,
            )
            if not candidate_units:
                return None
            candidate_sig = cls._source_units_signature(candidate_units)
            for row in cls.list_source_material_documents(campaign_id, limit=200):
                document_key = str(row.get("document_key") or "").strip()
                if not document_key:
                    continue
                existing_units = cls.get_source_material_document_units(
                    campaign_id,
                    document_key,
                )
                if not existing_units:
                    continue
                if cls._source_units_signature(existing_units) != candidate_sig:
                    continue
                return {
                    "document_key": document_key,
                    "document_label": str(row.get("document_label") or ""),
                    "chunk_count": int(row.get("chunk_count") or 0),
                }
            return None
        except Exception:
            logger.exception(
                "Source material: duplicate document lookup failed for campaign %s",
                campaign_id,
            )
            return None

    @classmethod
    def delete_source_material_document(
        cls,
        campaign_id: str,
        document_key: str,
    ) -> int:
        try:
            key = str(document_key or "").strip()
            if not key:
                return 0
            conn = cls._get_conn()
            cur = conn.execute(
                """
                DELETE FROM source_material_chunks
                WHERE campaign_id = ? AND document_key = ?
                """,
                (str(campaign_id), key),
            )
            conn.commit()
            return int(getattr(cur, "rowcount", 0) or 0)
        except Exception:
            logger.exception(
                "Source material: delete document failed for campaign %s key %s",
                campaign_id,
                document_key,
            )
            return 0

    @classmethod
    def clear_source_material_documents(cls, campaign_id: str) -> int:
        try:
            conn = cls._get_conn()
            cur = conn.execute(
                """
                DELETE FROM source_material_chunks
                WHERE campaign_id = ?
                """,
                (str(campaign_id),),
            )
            conn.commit()
            return int(getattr(cur, "rowcount", 0) or 0)
        except Exception:
            logger.exception(
                "Source material: clear documents failed for campaign %s",
                campaign_id,
            )
            return 0

    @classmethod
    def store_source_material_chunks(
        cls,
        campaign_id: str,
        *,
        document_label: str,
        chunks: List[str],
        source_mode: str = "line",
        replace_document: bool = True,
    ) -> Tuple[int, str]:
        try:
            label = " ".join(str(document_label or "").strip().split())[:120]
            if not label:
                label = "source-material"
            document_key = cls._normalize_source_document_key(label)
            mode = cls._normalize_source_unit_mode(source_mode)
            clean_chunks = [
                str(chunk or "").strip()
                for chunk in (chunks or [])
                if str(chunk or "").strip()
            ]
            if not clean_chunks:
                return 0, document_key
            sentence_units = cls.source_material_units_from_chunks_with_mode(
                clean_chunks, mode=mode
            )
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

    # ------------------------------------------------------------------
    # Browse source keys (rulebook key-snippet index)
    # ------------------------------------------------------------------

    @classmethod
    def browse_source_keys(
        cls,
        campaign_id: str,
        *,
        document_key: Optional[str] = None,
        wildcard: str = "%",
        limit: int = 255,
    ) -> List[str]:
        """Return a compact source index or matching raw source lines.

        When *wildcard* is omitted / broad (``*`` or ``%``), return a compact
        key listing so the model can see the document taxonomy without burning
        context on full fact bodies. When *wildcard* is specific, return the
        raw matching source lines.
        """
        try:
            conn = cls._get_conn()
            pattern = str(wildcard or "%").strip()
            if not pattern or pattern == "*":
                pattern = "%"
            else:
                pattern = pattern.replace("*", "%")
            broad_browse = pattern in {"%", "%%"}
            key = str(document_key or "").strip()
            if key:
                rows = conn.execute(
                    """
                    SELECT document_key, chunk_text
                    FROM source_material_chunks
                    WHERE campaign_id = ? AND document_key = ?
                      AND chunk_text LIKE ? ESCAPE '\\'
                    ORDER BY chunk_index ASC
                    LIMIT ?
                    """,
                    (str(campaign_id), key, pattern, max(1, int(limit))),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT document_key, chunk_text
                    FROM source_material_chunks
                    WHERE campaign_id = ?
                      AND chunk_text LIKE ? ESCAPE '\\'
                    ORDER BY document_key ASC, chunk_index ASC
                    LIMIT ?
                    """,
                    (str(campaign_id), pattern, max(1, int(limit))),
                ).fetchall()
            cleaned_rows = []
            for row_doc_key, row_chunk_text in rows:
                chunk_text = str(row_chunk_text or "").strip()
                if not chunk_text:
                    continue
                cleaned_rows.append((str(row_doc_key or "").strip(), chunk_text))
            if not broad_browse:
                return [chunk_text for _, chunk_text in cleaned_rows]

            compact: List[str] = []
            seen = set()
            for row_doc_key, chunk_text in cleaned_rows:
                key_text = chunk_text
                if ":" in chunk_text:
                    key_text = chunk_text.split(":", 1)[0].strip() or chunk_text
                if key:
                    line = key_text
                else:
                    line = f"{row_doc_key}: {key_text}" if row_doc_key else key_text
                normalized = " ".join(line.lower().split())
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                compact.append(line)
            return compact
        except Exception:
            logger.exception(
                "Source material: browse_source_keys failed for campaign %s",
                campaign_id,
            )
            return []
