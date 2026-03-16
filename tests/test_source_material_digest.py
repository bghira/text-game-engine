"""Tests for source material digest storage and retrieval."""

from __future__ import annotations

import sqlite3
import struct
import threading
from unittest.mock import MagicMock, patch

import numpy as np

from text_game_engine.core.source_material_memory import (
    EMBED_SOURCE_MINILM,
    EMBED_SOURCE_SNOWFLAKE,
    SourceMaterialMemory,
    _EMBED_DIM,
    _MAX_INPUT_CHARS,
    _SNOWFLAKE_QUERY_PREFIX,
    _embed,
)


def _fresh_conn():
    """Create a fresh in-memory connection for testing."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    from text_game_engine.core.source_material_memory import _SCHEMA_SQL
    conn.executescript(_SCHEMA_SQL)
    return conn


class TestDigestStorage:
    def setup_method(self):
        self._conn = _fresh_conn()
        self._patcher = patch.object(
            SourceMaterialMemory, "_get_conn", return_value=self._conn
        )
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()
        self._conn.close()

    def test_store_and_get_digest(self):
        ok = SourceMaterialMemory.store_source_material_digest(
            "camp1", "my-doc", "A rich narrative about dragons."
        )
        assert ok
        digest = SourceMaterialMemory.get_source_material_digest("camp1", "my-doc")
        assert digest == "A rich narrative about dragons."

    def test_get_nonexistent_digest(self):
        digest = SourceMaterialMemory.get_source_material_digest("camp1", "nope")
        assert digest is None

    def test_upsert_digest(self):
        SourceMaterialMemory.store_source_material_digest(
            "camp1", "doc", "Version 1."
        )
        SourceMaterialMemory.store_source_material_digest(
            "camp1", "doc", "Version 2."
        )
        digest = SourceMaterialMemory.get_source_material_digest("camp1", "doc")
        assert digest == "Version 2."

    def test_get_all_digests(self):
        SourceMaterialMemory.store_source_material_digest(
            "camp1", "doc-a", "Digest A."
        )
        SourceMaterialMemory.store_source_material_digest(
            "camp1", "doc-b", "Digest B."
        )
        SourceMaterialMemory.store_source_material_digest(
            "camp2", "doc-c", "Digest C."
        )
        all_digests = SourceMaterialMemory.get_all_source_material_digests("camp1")
        assert len(all_digests) == 2
        assert all_digests["doc-a"] == "Digest A."
        assert all_digests["doc-b"] == "Digest B."

    def test_delete_digest(self):
        SourceMaterialMemory.store_source_material_digest(
            "camp1", "doc", "Will be deleted."
        )
        ok = SourceMaterialMemory.delete_source_material_digest("camp1", "doc")
        assert ok
        assert SourceMaterialMemory.get_source_material_digest("camp1", "doc") is None

    def test_delete_document_cascades_digest(self):
        # Directly insert a chunk row to simulate stored document (no ML model needed)
        import struct
        fake_embedding = struct.pack("f" * 384, *([0.0] * 384))
        self._conn.execute(
            """
            INSERT INTO source_material_chunks
            (campaign_id, document_key, document_label, chunk_index, chunk_text, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("camp1", "my-doc", "My Doc", 1, "test chunk", fake_embedding),
        )
        self._conn.commit()
        SourceMaterialMemory.store_source_material_digest(
            "camp1", "my-doc", "Digest to cascade."
        )
        SourceMaterialMemory.delete_source_material_document("camp1", "my-doc")
        assert SourceMaterialMemory.get_source_material_digest("camp1", "my-doc") is None

    def test_clear_documents_cascades_digests(self):
        SourceMaterialMemory.store_source_material_digest(
            "camp1", "doc-a", "Digest A."
        )
        SourceMaterialMemory.store_source_material_digest(
            "camp1", "doc-b", "Digest B."
        )
        SourceMaterialMemory.clear_source_material_documents("camp1")
        all_digests = SourceMaterialMemory.get_all_source_material_digests("camp1")
        assert len(all_digests) == 0

    def test_empty_digest_rejected(self):
        ok = SourceMaterialMemory.store_source_material_digest("camp1", "doc", "")
        assert not ok
        ok = SourceMaterialMemory.store_source_material_digest("camp1", "doc", "   ")
        assert not ok

    def test_empty_key_normalizes_to_default(self):
        # Empty key normalizes to "source-material" via _normalize_source_document_key
        ok = SourceMaterialMemory.store_source_material_digest("camp1", "", "content")
        assert ok
        digest = SourceMaterialMemory.get_source_material_digest("camp1", "source-material")
        assert digest == "content"


class TestEmbedQueryPrefix:
    """Verify that _embed() applies the Snowflake query prefix correctly."""

    def _stub_model(self):
        """Return a mock model whose .encode() captures its input string."""
        model = MagicMock()
        model.encode = MagicMock(
            side_effect=lambda text, **_kw: np.zeros(_EMBED_DIM, dtype=np.float32)
        )
        return model

    def test_snowflake_query_prepends_prefix(self):
        model = self._stub_model()
        with patch(
            "text_game_engine.core.source_material_memory._get_model",
            return_value=model,
        ):
            _embed("sword in the cave", source=EMBED_SOURCE_SNOWFLAKE, query=True)
        encoded_text = model.encode.call_args[0][0]
        assert encoded_text.startswith(_SNOWFLAKE_QUERY_PREFIX)
        assert "sword in the cave" in encoded_text

    def test_snowflake_document_no_prefix(self):
        model = self._stub_model()
        with patch(
            "text_game_engine.core.source_material_memory._get_model",
            return_value=model,
        ):
            _embed("sword in the cave", source=EMBED_SOURCE_SNOWFLAKE, query=False)
        encoded_text = model.encode.call_args[0][0]
        assert not encoded_text.startswith(_SNOWFLAKE_QUERY_PREFIX)
        assert encoded_text == "sword in the cave"

    def test_minilm_query_no_prefix(self):
        model = self._stub_model()
        with patch(
            "text_game_engine.core.source_material_memory._get_model",
            return_value=model,
        ):
            _embed("sword in the cave", source=EMBED_SOURCE_MINILM, query=True)
        encoded_text = model.encode.call_args[0][0]
        assert not encoded_text.startswith(_SNOWFLAKE_QUERY_PREFIX)
        assert encoded_text == "sword in the cave"

    def test_query_truncation_respects_max_chars(self):
        """Total encoded string (prefix + text) must not exceed _MAX_INPUT_CHARS."""
        model = self._stub_model()
        long_text = "a" * (_MAX_INPUT_CHARS + 100)
        with patch(
            "text_game_engine.core.source_material_memory._get_model",
            return_value=model,
        ):
            _embed(long_text, source=EMBED_SOURCE_SNOWFLAKE, query=True)
        encoded_text = model.encode.call_args[0][0]
        assert len(encoded_text) <= _MAX_INPUT_CHARS
        assert encoded_text.startswith(_SNOWFLAKE_QUERY_PREFIX)

    def test_document_truncation_respects_max_chars(self):
        model = self._stub_model()
        long_text = "b" * (_MAX_INPUT_CHARS + 100)
        with patch(
            "text_game_engine.core.source_material_memory._get_model",
            return_value=model,
        ):
            _embed(long_text, source=EMBED_SOURCE_SNOWFLAKE, query=False)
        encoded_text = model.encode.call_args[0][0]
        assert len(encoded_text) <= _MAX_INPUT_CHARS
        assert not encoded_text.startswith(_SNOWFLAKE_QUERY_PREFIX)
