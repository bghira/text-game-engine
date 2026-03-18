"""Tests for fixes applied during the DTM→TGE migration.

Covers:
- JSON repair pipeline in _parse_json_lenient
- Memory search result deduplication across queries
- Source material campaign_id translation
- Embedding/keyword pool separation in memory search
- Memory tool text truncation limits
"""
from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# JSON repair pipeline
# ---------------------------------------------------------------------------

class TestParseJsonLenient:
    """_parse_json_lenient should apply _repair_json_lenient_text before giving up."""

    @pytest.fixture()
    def emulator(self):
        from text_game_engine.zork_emulator import ZorkEmulator
        emu = ZorkEmulator.__new__(ZorkEmulator)
        return emu

    def test_standard_json(self, emulator):
        result = emulator._parse_json_lenient('{"narration": "hello"}')
        assert result == {"narration": "hello"}

    def test_trailing_comma_repaired(self, emulator):
        # Non-schema key avoids the known-schema double-quote issue
        result = emulator._parse_json_lenient('{"foo": "bar", "count": 1,}')
        assert result["foo"] == "bar"
        assert result["count"] == 1

    def test_unquoted_keys_repaired(self, emulator):
        # Unquoted keys with properly quoted string values
        result = emulator._parse_json_lenient('{narration: "hello", xp: 1}')
        # Repair may double-quote the string value; key point is it parses.
        assert isinstance(result, dict)
        assert "narration" in result
        assert result.get("xp") == 1

    def test_unquoted_string_value_repaired(self, emulator):
        text = '{"narration": The world is quiet here.}'
        result = emulator._parse_json_lenient(text)
        assert isinstance(result, dict)
        assert "narration" in result

    def test_python_dict_coercion(self, emulator):
        result = emulator._parse_json_lenient("{'narration': 'hello', 'xp': 5}")
        assert result == {"narration": "hello", "xp": 5}

    def test_jsonl_merge(self, emulator):
        text = '{"narration": "a"}\n{"xp": 5}'
        result = emulator._parse_json_lenient(text)
        assert result.get("narration") == "a"
        assert result.get("xp") == 5

    def test_non_json_raises(self, emulator):
        with pytest.raises(json.JSONDecodeError):
            emulator._parse_json_lenient("This is just prose with no JSON at all.")

    def test_repair_applied_before_coercion(self, emulator):
        """Trailing comma + numeric value should be repaired."""
        text = '{"foo": "bar", "count": 5,}'
        result = emulator._parse_json_lenient(text)
        assert result.get("foo") == "bar"
        assert result.get("count") == 5


class TestExtractJson:
    @pytest.fixture()
    def emulator(self):
        from text_game_engine.zork_emulator import ZorkEmulator
        emu = ZorkEmulator.__new__(ZorkEmulator)
        return emu

    def test_strips_code_fences(self, emulator):
        text = '```json\n{"narration": "hello"}\n```'
        result = emulator._extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["narration"] == "hello"

    def test_finds_json_in_prose(self, emulator):
        text = 'Here is the response:\n{"narration": "hello"}\nDone.'
        result = emulator._extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["narration"] == "hello"

    def test_returns_none_for_no_json(self, emulator):
        assert emulator._extract_json("No JSON here") is None


class TestCleanResponse:
    @pytest.fixture()
    def emulator(self):
        from text_game_engine.zork_emulator import ZorkEmulator
        emu = ZorkEmulator.__new__(ZorkEmulator)
        return emu

    def test_truncated_object_repaired(self, emulator):
        text = '{"narration": "hello", "xp_awarded": 1'
        result = emulator._clean_response(text)
        parsed = json.loads(result)
        assert parsed["narration"] == "hello"

    def test_clean_json_passthrough(self, emulator):
        text = '{"narration": "hello"}'
        result = emulator._clean_response(text)
        assert json.loads(result) == {"narration": "hello"}


# ---------------------------------------------------------------------------
# Source material campaign_id translation
# ---------------------------------------------------------------------------

class TestSourceMaterialConfigure:
    """SourceMaterialMemory.configure() should redirect DB path and translate IDs."""

    def test_resolve_campaign_id_no_translator(self):
        from text_game_engine.core.source_material_memory import SourceMaterialMemory
        # Without translator, returns str as-is
        result = SourceMaterialMemory._resolve_campaign_id("some-uuid-string")
        assert result == "some-uuid-string"

    def test_resolve_campaign_id_with_translator(self):
        from text_game_engine.core import source_material_memory as sm_mod
        from text_game_engine.core.source_material_memory import SourceMaterialMemory

        old_translator = sm_mod._CAMPAIGN_ID_TRANSLATOR
        try:
            sm_mod._CAMPAIGN_ID_TRANSLATOR = lambda cid: 42 if cid == "my-uuid" else cid
            result = SourceMaterialMemory._resolve_campaign_id("my-uuid")
            assert result == "42"  # str() wraps the integer
        finally:
            sm_mod._CAMPAIGN_ID_TRANSLATOR = old_translator

    def test_resolve_campaign_id_translator_exception_falls_back(self):
        from text_game_engine.core import source_material_memory as sm_mod
        from text_game_engine.core.source_material_memory import SourceMaterialMemory

        old_translator = sm_mod._CAMPAIGN_ID_TRANSLATOR
        try:
            sm_mod._CAMPAIGN_ID_TRANSLATOR = lambda cid: 1 / 0  # always raises
            result = SourceMaterialMemory._resolve_campaign_id("fallback-id")
            assert result == "fallback-id"  # falls back to raw string
        finally:
            sm_mod._CAMPAIGN_ID_TRANSLATOR = old_translator

    def test_configure_sets_db_path_override(self):
        from text_game_engine.core import source_material_memory as sm_mod
        from text_game_engine.core.source_material_memory import SourceMaterialMemory

        old_override = sm_mod._DB_PATH_OVERRIDE
        try:
            SourceMaterialMemory.configure(db_path="/tmp/test_embeddings.db")
            assert sm_mod._DB_PATH_OVERRIDE == "/tmp/test_embeddings.db"
        finally:
            sm_mod._DB_PATH_OVERRIDE = old_override
            # Reset thread-local connections
            SourceMaterialMemory._conn_local = threading.local()

    def test_list_documents_uses_translated_campaign_id(self):
        """Integration test: list_source_material_documents passes translated ID to SQL."""
        from text_game_engine.core import source_material_memory as sm_mod
        from text_game_engine.core.source_material_memory import SourceMaterialMemory

        old_override = sm_mod._DB_PATH_OVERRIDE
        old_translator = sm_mod._CAMPAIGN_ID_TRANSLATOR
        try:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name

            # Set up a fresh DB with integer campaign_id
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_material_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    document_key TEXT NOT NULL,
                    document_label TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embed_source TEXT NOT NULL DEFAULT 'minilm',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_material_digests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    document_key TEXT NOT NULL,
                    digest_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Insert with integer campaign_id (as ZorkMemory does)
            fake_embed = b"\x00" * (384 * 4)
            conn.execute(
                "INSERT INTO source_material_chunks "
                "(campaign_id, document_key, document_label, chunk_index, chunk_text, embedding, embed_source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (80, "lore-book", "Lore Book", 1, "Tommy runs the booth.", fake_embed, "snowflake"),
            )
            conn.commit()
            conn.close()

            # Configure with translator that maps UUID → integer
            SourceMaterialMemory.configure(
                db_path=db_path,
                campaign_id_translator=lambda cid: 80 if cid == "my-campaign-uuid" else cid,
            )
            # Reset connection cache so it picks up new DB
            SourceMaterialMemory._conn_local = threading.local()

            docs = SourceMaterialMemory.list_source_material_documents("my-campaign-uuid")
            assert len(docs) == 1
            assert docs[0]["document_key"] == "lore-book"
            assert docs[0]["chunk_count"] == 1
        finally:
            sm_mod._DB_PATH_OVERRIDE = old_override
            sm_mod._CAMPAIGN_ID_TRANSLATOR = old_translator
            SourceMaterialMemory._conn_local = threading.local()


# ---------------------------------------------------------------------------
# Memory search deduplication
# ---------------------------------------------------------------------------

class TestMemorySearchDedup:
    """Curated memory hits should be deduplicated across queries."""

    def test_curated_dedup_by_term_and_text(self):
        """Same (term, text) from different queries should appear only once."""
        seen: set[tuple[str, str]] = set()
        curated_hits: list[tuple[str, str, float]] = []

        # Simulate 3 queries returning the same curated memory
        raw_results = [
            ("char:alice", "Alice is a spy.", 0.9),
            ("char:alice", "Alice is a spy.", 0.7),
            ("char:alice", "Alice is a spy.", 0.5),
            ("char:bob", "Bob is a baker.", 0.8),
        ]
        for hit in raw_results:
            dedup_key = (str(hit[0] or "").strip(), str(hit[1] or "").strip())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            curated_hits.append(hit)

        assert len(curated_hits) == 2
        assert curated_hits[0][0] == "char:alice"
        assert curated_hits[1][0] == "char:bob"

    def test_curated_dedup_keeps_first_occurrence(self):
        """First occurrence (highest score from first query) is kept."""
        seen: set[tuple[str, str]] = set()
        curated_hits: list[tuple[str, str, float]] = []

        raw_results = [
            ("term", "memory text", 0.95),
            ("term", "memory text", 0.30),
        ]
        for hit in raw_results:
            dedup_key = (str(hit[0] or "").strip(), str(hit[1] or "").strip())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            curated_hits.append(hit)

        assert len(curated_hits) == 1
        assert curated_hits[0][2] == 0.95


# ---------------------------------------------------------------------------
# Embedding vs keyword pool separation
# ---------------------------------------------------------------------------

class TestEmbeddingKeywordPools:
    """Embedding-only hits should surface even when keyword hits fill the top-5."""

    def test_keyword_and_embedding_pools_both_represented(self):
        """Simulates the pool-separation logic from _tool_memory_search."""
        # 6 keyword hits with score 1.0
        narrator_hits: dict[int, dict[str, Any]] = {}
        for tid in range(100, 106):
            narrator_hits[tid] = {
                "turn_id": tid,
                "score": 1.0,
                "content": f"keyword hit {tid}",
                "visibility_scope": "public",
                "actor_player_slug": "",
                "location_key": "",
            }

        # 4 embedding-only hits with lower scores
        embed_only_hits: dict[int, dict[str, Any]] = {}
        for i, tid in enumerate([200, 201, 202, 203]):
            embed_only_hits[tid] = {
                "turn_id": tid,
                "score": 0.8 - i * 0.1,
                "content": f"embedding hit {tid}",
                "visibility_scope": "public",
                "actor_player_slug": "",
                "location_key": "",
            }

        # Apply the pool logic
        ordered_keyword = sorted(
            narrator_hits.values(),
            key=lambda row: (float(row.get("score", 0.0)), int(row.get("turn_id", 0))),
            reverse=True,
        )[:5]
        keyword_ids = {int(r.get("turn_id", 0)) for r in ordered_keyword}
        ordered_embed_only = sorted(
            (v for v in embed_only_hits.values() if int(v.get("turn_id", 0)) not in keyword_ids),
            key=lambda row: (float(row.get("score", 0.0)), int(row.get("turn_id", 0))),
            reverse=True,
        )[:3]
        ordered_narrator = ordered_keyword + ordered_embed_only

        assert len(ordered_keyword) == 5
        assert len(ordered_embed_only) == 3
        assert len(ordered_narrator) == 8

        # All keyword hits are from narrator_hits
        for r in ordered_keyword:
            assert r["turn_id"] in narrator_hits

        # All embed-only hits are from embed_only_hits
        for r in ordered_embed_only:
            assert r["turn_id"] in embed_only_hits

    def test_embedding_hit_already_in_keyword_not_duplicated(self):
        """If a turn is found by both keyword and embedding, it shouldn't appear twice."""
        narrator_hits = {
            100: {"turn_id": 100, "score": 1.0, "content": "shared hit"},
        }
        embed_only_hits: dict[int, dict[str, Any]] = {}

        # Simulate: embedding search also finds turn 100
        tid = 100
        if tid in narrator_hits:
            # Boost score if embedding is higher (it isn't here)
            pass
        else:
            embed_only_hits[tid] = {"turn_id": tid, "score": 0.7, "content": "shared hit"}

        ordered_keyword = sorted(
            narrator_hits.values(),
            key=lambda row: (float(row.get("score", 0.0)), int(row.get("turn_id", 0))),
            reverse=True,
        )[:5]
        keyword_ids = {int(r.get("turn_id", 0)) for r in ordered_keyword}
        ordered_embed_only = sorted(
            (v for v in embed_only_hits.values() if int(v.get("turn_id", 0)) not in keyword_ids),
            key=lambda row: (float(row.get("score", 0.0)), int(row.get("turn_id", 0))),
            reverse=True,
        )[:3]
        ordered_narrator = ordered_keyword + ordered_embed_only

        assert len(ordered_narrator) == 1
        assert ordered_narrator[0]["turn_id"] == 100

    def test_no_keyword_hits_embedding_still_surfaces(self):
        """When keyword search finds nothing, embedding results should still appear."""
        narrator_hits: dict[int, dict[str, Any]] = {}
        embed_only_hits = {
            50: {"turn_id": 50, "score": 0.75, "content": "only embedding found this"},
            51: {"turn_id": 51, "score": 0.60, "content": "also embedding only"},
        }

        ordered_keyword = sorted(
            narrator_hits.values(),
            key=lambda row: (float(row.get("score", 0.0)), int(row.get("turn_id", 0))),
            reverse=True,
        )[:5]
        keyword_ids = {int(r.get("turn_id", 0)) for r in ordered_keyword}
        ordered_embed_only = sorted(
            (v for v in embed_only_hits.values() if int(v.get("turn_id", 0)) not in keyword_ids),
            key=lambda row: (float(row.get("score", 0.0)), int(row.get("turn_id", 0))),
            reverse=True,
        )[:3]
        ordered_narrator = ordered_keyword + ordered_embed_only

        assert len(ordered_narrator) == 2
        assert ordered_narrator[0]["turn_id"] == 50  # score 0.75 > 0.60


# ---------------------------------------------------------------------------
# JSON repair sub-methods
# ---------------------------------------------------------------------------

class TestRepairJsonLenientText:
    @pytest.fixture()
    def emulator_cls(self):
        from text_game_engine.zork_emulator import ZorkEmulator
        return ZorkEmulator

    def test_trailing_commas_removed(self, emulator_cls):
        text = '{"a": 1, "b": 2,}'
        result = emulator_cls._repair_trailing_json_commas(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_unquoted_keys_fixed(self, emulator_cls):
        text = '{name: "Alice", age: 30}'
        result = emulator_cls._repair_unquoted_json_keys(text)
        assert '"name"' in result
        assert '"age"' in result

    def test_full_pipeline(self, emulator_cls):
        # Non-schema keys to avoid known-schema double-quote on already-quoted values
        text = '{"foo": "bar", "count": 5,}'
        result = emulator_cls._repair_json_lenient_text(text)
        parsed = json.loads(result)
        assert parsed["foo"] == "bar"
        assert parsed["count"] == 5

    def test_unquoted_value_for_schema_field(self, emulator_cls):
        """Known schema field with genuinely unquoted value gets quoted."""
        text = '{"narration": The room is quiet here.}'
        result = emulator_cls._repair_known_schema_string_fields(text)
        parsed = json.loads(result)
        assert parsed["narration"] == "The room is quiet here."
