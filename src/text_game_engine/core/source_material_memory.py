from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import hashlib
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_model_minilm = None
_model_snowflake = None
_MODEL_LOCK = threading.Lock()
_MAX_INPUT_CHARS = 512
_SOURCE_SNIPPET_MAX_CHARS = 1200
_EMBED_DIM = 384  # dimension shared by MiniLM-L6-v2 and Snowflake arctic-embed-s
_EMBED_FALLBACK_WARNED = False

EMBED_SOURCE_MINILM = "minilm"
EMBED_SOURCE_SNOWFLAKE = "snowflake"
EMBED_SOURCE_DEFAULT = EMBED_SOURCE_SNOWFLAKE

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
    embed_source    TEXT NOT NULL DEFAULT 'minilm',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tge_sm_campaign ON source_material_chunks(campaign_id);
CREATE INDEX IF NOT EXISTS idx_tge_sm_campaign_doc ON source_material_chunks(campaign_id, document_key);

CREATE TABLE IF NOT EXISTS source_material_digests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id     TEXT    NOT NULL,
    document_key    TEXT    NOT NULL,
    digest_text     TEXT    NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tge_smd_campaign_doc
    ON source_material_digests(campaign_id, document_key);
"""


_VALID_EMBED_SOURCES = {EMBED_SOURCE_MINILM, EMBED_SOURCE_SNOWFLAKE}


def _get_model(source: str = EMBED_SOURCE_DEFAULT):
    global _model_minilm, _model_snowflake
    source = (source or "").strip().lower()
    if source not in _VALID_EMBED_SOURCES:
        logger.warning("Unknown embed source %r, falling back to %s", source, EMBED_SOURCE_DEFAULT)
        source = EMBED_SOURCE_DEFAULT
    if source == EMBED_SOURCE_MINILM:
        if _model_minilm is not None:
            return _model_minilm
        with _MODEL_LOCK:
            if _model_minilm is not None:
                return _model_minilm
            from sentence_transformers import SentenceTransformer

            _model_minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return _model_minilm
    else:
        if _model_snowflake is not None:
            return _model_snowflake
        with _MODEL_LOCK:
            if _model_snowflake is not None:
                return _model_snowflake
            from sentence_transformers import SentenceTransformer

            _model_snowflake = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
            return _model_snowflake


def _embed(text: str, source: str = EMBED_SOURCE_DEFAULT) -> bytes:
    import numpy as np

    try:
        model = _get_model(source)
    except Exception:
        global _EMBED_FALLBACK_WARNED
        with _MODEL_LOCK:
            if not _EMBED_FALLBACK_WARNED:
                logger.warning(
                    "sentence-transformers not available; storing zero-vector embeddings. "
                    "Install text-game-engine[embeddings] for semantic search."
                )
                _EMBED_FALLBACK_WARNED = True
        return np.zeros(_EMBED_DIM, dtype=np.float32).tobytes()
    vector = model.encode((text or "")[:_MAX_INPUT_CHARS], normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32).tobytes()


def _bytes_to_vector(blob: bytes):
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32)


_ALLOWED_EMBED_TABLES = {"source_material_chunks"}


def _campaign_embed_sources(conn, table: str, campaign_id) -> set:
    """Return the set of distinct embed_source values for a campaign in *table*."""
    if table not in _ALLOWED_EMBED_TABLES:
        raise ValueError(f"Invalid table name for embed source query: {table!r}")
    rows = conn.execute(
        f"SELECT DISTINCT embed_source FROM {table} WHERE campaign_id = ?",
        (str(campaign_id),),
    ).fetchall()
    return {str(r[0] or EMBED_SOURCE_MINILM) for r in rows}


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
        cls._ensure_schema(conn)
        cls._conn_local.conn = conn
        return conn

    @classmethod
    def _ensure_schema(cls, conn: sqlite3.Connection) -> None:
        cols = conn.execute("PRAGMA table_info(source_material_chunks)").fetchall()
        existing = {str(r[1] or "").strip() for r in cols}
        if "embed_source" not in existing:
            conn.execute(
                "ALTER TABLE source_material_chunks "
                "ADD COLUMN embed_source TEXT NOT NULL DEFAULT 'minilm'"
            )
            conn.commit()

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
    def _normalize_rulebook_fact_key(cls, value: str) -> str:
        return " ".join(str(value or "").strip().split())

    @classmethod
    def list_rulebook_entries(
        cls,
        campaign_id: str,
        document_key: str,
    ) -> List[Dict[str, str]]:
        try:
            units = cls.get_source_material_document_units(campaign_id, document_key)
            entries: List[Dict[str, str]] = []
            for unit in units:
                text = str(unit or "").strip()
                if not cls._is_rulebook_fact_line(text):
                    continue
                key, value = text.split(":", 1)
                key_clean = cls._normalize_rulebook_fact_key(key)
                value_clean = " ".join(value.strip().split())
                if not key_clean or not value_clean:
                    continue
                entries.append({"key": key_clean, "value": value_clean})
            return entries
        except Exception:
            logger.exception(
                "Source material: list_rulebook_entries failed for campaign %s key %s",
                campaign_id,
                document_key,
            )
            return []

    @classmethod
    def get_rulebook_entry(
        cls,
        campaign_id: str,
        document_key: str,
        rule_key: str,
    ) -> Dict[str, str] | None:
        normalized_key = cls._normalize_rulebook_fact_key(rule_key).lower()
        if not normalized_key:
            return None
        for entry in cls.list_rulebook_entries(campaign_id, document_key):
            if cls._normalize_rulebook_fact_key(entry.get("key") or "").lower() == normalized_key:
                return entry
        return None

    @classmethod
    def put_rulebook_entry(
        cls,
        campaign_id: str,
        *,
        document_label: str,
        rule_key: str,
        rule_text: str,
        replace_existing: bool = False,
    ) -> Dict[str, object]:
        label = " ".join(str(document_label or "").strip().split())[:120] or "campaign-rulebook"
        document_key = cls._normalize_source_document_key(label)
        key_clean = cls._normalize_rulebook_fact_key(rule_key)
        value_clean = " ".join(str(rule_text or "").strip().split())
        if not key_clean or not value_clean:
            return {
                "ok": False,
                "document_key": document_key,
                "document_label": label,
                "reason": "invalid",
            }

        entries = cls.list_rulebook_entries(campaign_id, document_key)
        match_index = -1
        old_value = None
        normalized_lookup = key_clean.lower()
        for idx, entry in enumerate(entries):
            entry_key = cls._normalize_rulebook_fact_key(entry.get("key") or "")
            if entry_key.lower() == normalized_lookup:
                match_index = idx
                key_clean = entry_key or key_clean
                old_value = str(entry.get("value") or "").strip()
                break

        if match_index >= 0 and not replace_existing:
            return {
                "ok": False,
                "document_key": document_key,
                "document_label": label,
                "reason": "exists",
                "key": key_clean,
                "old_value": old_value or "",
                "new_value": value_clean,
            }

        if match_index >= 0:
            entries[match_index] = {"key": key_clean, "value": value_clean}
        else:
            entries.append({"key": key_clean, "value": value_clean})

        lines = [f"{entry['key']}: {entry['value']}" for entry in entries if entry.get("key") and entry.get("value")]
        stored_count, stored_key = cls.store_source_material_chunks(
            campaign_id,
            document_label=label,
            chunks=["\n".join(lines)],
            source_mode="rulebook",
            replace_document=True,
        )
        return {
            "ok": stored_count > 0,
            "document_key": stored_key,
            "document_label": label,
            "key": key_clean,
            "old_value": old_value or "",
            "new_value": value_clean,
            "created": match_index < 0,
            "replaced": match_index >= 0,
        }

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
            conn.execute(
                """
                DELETE FROM source_material_digests
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
            conn.execute(
                """
                DELETE FROM source_material_digests
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
        embed_source: str = EMBED_SOURCE_DEFAULT,
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
                    (campaign_id, document_key, document_label, chunk_index, chunk_text, embedding, embed_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(campaign_id),
                        document_key,
                        label,
                        idx,
                        chunk_text,
                        _embed(chunk_text, source=embed_source),
                        embed_source,
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
    def store_source_material_digest(
        cls,
        campaign_id: str,
        document_key: str,
        digest_text: str,
    ) -> bool:
        try:
            key = cls._normalize_source_document_key(document_key)
            text = str(digest_text or "").strip()
            if not key or not text:
                return False
            conn = cls._get_conn()
            conn.execute(
                """
                INSERT INTO source_material_digests (campaign_id, document_key, digest_text)
                VALUES (?, ?, ?)
                ON CONFLICT(campaign_id, document_key)
                DO UPDATE SET digest_text = excluded.digest_text,
                              created_at = CURRENT_TIMESTAMP
                """,
                (str(campaign_id), key, text),
            )
            conn.commit()
            return True
        except Exception:
            logger.exception(
                "Source material: store digest failed for campaign %s key %s",
                campaign_id,
                document_key,
            )
            return False

    @classmethod
    def get_source_material_digest(
        cls,
        campaign_id: str,
        document_key: str,
    ) -> Optional[str]:
        try:
            key = cls._normalize_source_document_key(document_key)
            if not key:
                return None
            conn = cls._get_conn()
            row = conn.execute(
                """
                SELECT digest_text
                FROM source_material_digests
                WHERE campaign_id = ? AND document_key = ?
                """,
                (str(campaign_id), key),
            ).fetchone()
            if row and str(row[0] or "").strip():
                return str(row[0]).strip()
            return None
        except Exception:
            logger.exception(
                "Source material: get digest failed for campaign %s key %s",
                campaign_id,
                document_key,
            )
            return None

    @classmethod
    def get_all_source_material_digests(
        cls,
        campaign_id: str,
    ) -> Dict[str, str]:
        try:
            conn = cls._get_conn()
            rows = conn.execute(
                """
                SELECT document_key, digest_text
                FROM source_material_digests
                WHERE campaign_id = ?
                """,
                (str(campaign_id),),
            ).fetchall()
            return {
                str(row[0]): str(row[1]).strip()
                for row in rows
                if str(row[0] or "").strip() and str(row[1] or "").strip()
            }
        except Exception:
            logger.exception(
                "Source material: get all digests failed for campaign %s",
                campaign_id,
            )
            return {}

    @classmethod
    def delete_source_material_digest(
        cls,
        campaign_id: str,
        document_key: str,
    ) -> bool:
        try:
            key = cls._normalize_source_document_key(document_key)
            if not key:
                return False
            conn = cls._get_conn()
            conn.execute(
                """
                DELETE FROM source_material_digests
                WHERE campaign_id = ? AND document_key = ?
                """,
                (str(campaign_id), key),
            )
            conn.commit()
            return True
        except Exception:
            logger.exception(
                "Source material: delete digest failed for campaign %s key %s",
                campaign_id,
                document_key,
            )
            return False

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

            conn = cls._get_conn()
            sources = _campaign_embed_sources(conn, "source_material_chunks", campaign_id)
            if not sources:
                return []
            before_n = max(0, min(50, int(before_lines)))
            after_n = max(0, min(50, int(after_lines)))
            key = str(document_key or "").strip()

            # Build by_doc index from all rows (for window expansion).
            if key:
                all_rows = conn.execute(
                    """
                    SELECT document_key, chunk_index, chunk_text
                    FROM source_material_chunks
                    WHERE campaign_id = ? AND document_key = ?
                    """,
                    (str(campaign_id), key),
                ).fetchall()
            else:
                all_rows = conn.execute(
                    """
                    SELECT document_key, chunk_index, chunk_text
                    FROM source_material_chunks
                    WHERE campaign_id = ?
                    """,
                    (str(campaign_id),),
                ).fetchall()
            by_doc: Dict[str, Dict[int, str]] = {}
            for r_key, r_idx, r_text in all_rows:
                doc_k = str(r_key or "")
                ci = int(r_idx or 0)
                ct = str(r_text or "")
                if ci > 0 and ct:
                    by_doc.setdefault(doc_k, {})[ci] = ct

            scored: List[Tuple[str, str, int, str, float]] = []
            for source in sources:
                query_vec = _bytes_to_vector(_embed(query or "", source=source))
                if key:
                    rows = conn.execute(
                        """
                        SELECT document_key, document_label, chunk_index, chunk_text, embedding
                        FROM source_material_chunks
                        WHERE campaign_id = ? AND document_key = ? AND embed_source = ?
                        """,
                        (str(campaign_id), key, source),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT document_key, document_label, chunk_index, chunk_text, embedding
                        FROM source_material_chunks
                        WHERE campaign_id = ? AND embed_source = ?
                        """,
                        (str(campaign_id), source),
                    ).fetchall()

                for row_key, row_label, row_chunk_idx, row_chunk_text, row_blob in rows:
                    doc_key_str = str(row_key or "")
                    chunk_idx = int(row_chunk_idx or 0)
                    chunk_text = str(row_chunk_text or "")
                    vec = _bytes_to_vector(row_blob)
                    score = float(np.dot(query_vec, vec))
                    scored.append(
                        (
                            doc_key_str,
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
            for doc_key_s, doc_label, center_idx, center_text, score in selected:
                doc_chunks = by_doc.get(doc_key_s, {})
                if center_idx <= 0:
                    expanded.append((doc_key_s, doc_label, center_idx, center_text, score))
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
                        doc_key_s,
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
