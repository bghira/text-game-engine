from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import re
import time

from text_game_engine.core.types import GiveItemInstruction, LLMTurnOutput, TimerInstruction
from text_game_engine.core.engine import GameEngine
from text_game_engine.persistence.sqlalchemy.uow import SQLAlchemyUnitOfWork
from text_game_engine.persistence.sqlalchemy.models import Actor, Campaign, Player, Snapshot, Turn
from text_game_engine.tool_aware_llm import ToolAwareZorkLLM
from text_game_engine.zork_emulator import ZorkEmulator


class StubLLM:
    def __init__(self, output: LLMTurnOutput):
        self.output = output

    async def complete_turn(self, context):
        return self.output


class StubCompletionPort:
    async def complete(self, system_prompt, prompt, *, temperature=0.8, max_tokens=2048):
        if "Summarise the following text passage" in system_prompt:
            return "chunk summary --COMPLETED SUMMARY--"
        if "You classify whether text references a known published work" in system_prompt:
            return (
                '{"is_known_work": true, "work_type": "film", '
                '"work_description": "A hacker learns reality is simulated.", '
                '"suggested_title": "The Matrix"}'
            )
        if "creative game designer" in system_prompt:
            return (
                '{"variants":[{"id":"variant-1","title":"Canonical Matrix",'
                '"summary":"Neo awakens to the machine world and must choose between survival and freedom.",'
                '"main_character":"Neo","essential_npcs":["Morpheus","Trinity","Agent Smith"],'
                '"chapter_outline":[{"title":"Wake","summary":"Neo discovers the truth."},'
                '{"title":"Revolt","summary":"The crew strikes back."}]}]}'
            )
        if "world-builder for interactive text-adventure campaigns" in system_prompt:
            return (
                '{"summary":"Setup summary","setting":"Neo-noir Seattle","tone":"noir",'
                '"default_persona":"A focused hacker with dry wit.","landmarks":["Dock 9"],'
                '"story_outline":{"chapters":[{"title":"Wake"},{"title":"Revolt"}]},'
                '"start_room":{"room_title":"Dock 9","room_summary":"Wet steel pier","room_description":"Rain hisses on steel.","exits":["warehouse","alley"],"location":"dock-9"},'
                '"opening_narration":"Neon smears across rain slicks.","characters":{"guide":{"name":"Mira"}}}'
            )
        return '{"narration":"ok"}'


class CaptureMapCompletionPort:
    def __init__(self):
        self.calls = []

    async def complete(self, system_prompt, prompt, *, temperature=0.8, max_tokens=2048):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "prompt": prompt,
            }
        )
        return "@\nLEGEND:\n@ Neo"


class NovelIntentProbeCompletionPort:
    def __init__(self):
        self.initial_classify_calls = 0
        self.reclassify_calls = 0
        self.variant_prompts = []

    async def complete(self, system_prompt, prompt, *, temperature=0.8, max_tokens=2048):
        if "You classify whether text references a known published work" in system_prompt:
            self.initial_classify_calls += 1
            return (
                '{"is_known_work": true, "work_type": "film", '
                '"work_description": "A hacker learns reality is simulated.", '
                '"suggested_title": "The Matrix"}'
            )
        if system_prompt.startswith("Return JSON only: is_known_work"):
            self.reclassify_calls += 1
            return (
                '{"is_known_work": true, "work_type": "film", '
                '"work_description": "Reclassified known work.", '
                '"suggested_title": "Unexpected Sequel"}'
            )
        if "creative game designer" in system_prompt:
            self.variant_prompts.append(prompt)
            return (
                '{"variants":[{"id":"variant-1","title":"Original Arc",'
                '"summary":"A wholly original campaign premise.",'
                '"main_character":"Kara","essential_npcs":["Nox"],'
                '"chapter_outline":[{"title":"Awaken","summary":"The world opens."}]}]}'
            )
        if "world-builder for interactive text-adventure campaigns" in system_prompt:
            return (
                '{"summary":"Setup summary","setting":"Original setting","tone":"mystery",'
                '"default_persona":"A careful investigator.","landmarks":["Harbor"],'
                '"story_outline":{"chapters":[{"title":"Awaken"}]},'
                '"start_room":{"room_title":"Harbor","room_summary":"Fog and bells","room_description":"Fog rolls in.","exits":["market"],"location":"harbor"},'
                '"opening_narration":"The bell tolls as fog closes in.","characters":{"guide":{"name":"Nox"}}}'
            )
        return '{"narration":"ok"}'


class GuardRetryCompletionPort:
    def __init__(self):
        self.calls = 0

    async def complete(self, system_prompt, prompt, *, temperature=0.8, max_tokens=2048):
        self.calls += 1
        if self.calls == 1:
            return "first attempt without guard"
        return "second attempt with guard --COMPLETED SUMMARY--"


class FailingCondenseCompletionPort:
    async def complete(self, system_prompt, prompt, *, temperature=0.8, max_tokens=2048):
        raise RuntimeError("condense failed")


class ReadyToWriteCompletionPort:
    def __init__(self):
        self.calls = []

    async def complete(self, system_prompt, prompt, *, temperature=0.8, max_tokens=2048):
        self.calls.append({"system_prompt": system_prompt, "prompt": prompt})
        if len(self.calls) == 1:
            return '{"tool_call":"ready_to_write"}'
        return (
            '{"reasoning":"finalized","narration":"Final scene.","state_update":{"game_time":{"day":1,"hour":9,"minute":0,"day_of_week":"monday","period":"morning","date_label":"Monday, Day 1, Morning"}},"summary_update":"done"}'
        )


class StubTimerEffects:
    def __init__(self):
        self.edits = []
        self.emits = []

    async def edit_timer_line(self, channel_id: str, message_id: str, replacement: str) -> None:
        self.edits.append((channel_id, message_id, replacement))

    async def emit_timed_event(self, campaign_id: str, channel_id: str, actor_id: str | None, narration: str) -> None:
        self.emits.append((campaign_id, channel_id, actor_id, narration))


class StubIMDB:
    def search(self, query: str, max_results: int = 3):
        return [{"title": "The Matrix", "year": 1999, "imdb_id": "tt0133093"}][:max_results]

    def enrich(self, results):
        enriched = []
        for entry in results:
            item = dict(entry)
            item["description"] = "A hacker learns reality is simulated."
            enriched.append(item)
        return enriched


class StubAttachment:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self._data = data
        self.size = len(data)

    async def read(self) -> bytes:
        return self._data


class StubMediaPort:
    def __init__(self):
        self.scene_calls = []
        self.avatar_calls = []
        self.available = True

    def gpu_worker_available(self) -> bool:
        return self.available

    async def enqueue_scene_generation(
        self,
        *,
        actor_id: str,
        prompt: str,
        model: str,
        reference_images=None,
        metadata=None,
        channel_id=None,
    ) -> bool:
        self.scene_calls.append(
            {
                "actor_id": actor_id,
                "prompt": prompt,
                "model": model,
                "reference_images": list(reference_images or []),
                "metadata": dict(metadata or {}),
                "channel_id": channel_id,
            }
        )
        return True

    async def enqueue_avatar_generation(
        self,
        *,
        actor_id: str,
        prompt: str,
        model: str,
        metadata=None,
        channel_id=None,
    ) -> bool:
        self.avatar_calls.append(
            {
                "actor_id": actor_id,
                "prompt": prompt,
                "model": model,
                "metadata": dict(metadata or {}),
                "channel_id": channel_id,
            }
        )
        return True


class StubCtx:
    class _Author:
        id = "actor-1"
        display_name = "Neo"

    class _Guild:
        id = "default"

    class _Channel:
        id = "chan-1"

    author = _Author()
    guild = _Guild()
    channel = _Channel()
    message = None


class LegacyCtx:
    class _Author:
        def __init__(self, actor_id: str):
            self.id = actor_id
            self.display_name = "Neo"

    class _Guild:
        def __init__(self, guild_id: str):
            self.id = guild_id

    class _Channel:
        def __init__(self, channel_id: str):
            self.id = channel_id

    class _Message:
        def __init__(self, attachments):
            self.attachments = attachments

    def __init__(self, actor_id: str, guild_id: str = "default", channel_id: str = "main", attachments=None):
        self.author = LegacyCtx._Author(actor_id)
        self.guild = LegacyCtx._Guild(guild_id)
        self.channel = LegacyCtx._Channel(channel_id)
        self.message = LegacyCtx._Message(attachments or [])


def test_rename_player_character_updates_state_actor_and_roster(
    session_factory,
    uow_factory,
    seed_campaign_and_actor,
):
    llm = StubLLM(LLMTurnOutput(narration="ok"))
    engine = GameEngine(uow_factory=uow_factory, llm=llm)
    compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

    actor_id = seed_campaign_and_actor["actor_id"]
    campaign_id = seed_campaign_and_actor["campaign_id"]
    compat.get_or_create_actor(actor_id, display_name="Old Name")
    player = compat.get_or_create_player(campaign_id, actor_id)

    with session_factory() as session:
        player_row = session.get(Player, player.id)
        assert player_row is not None
        player_row.state_json = compat._dump_json({"character_name": "Old Name"})
        campaign_row = session.get(Campaign, campaign_id)
        assert campaign_row is not None
        campaign_row.characters_json = compat._dump_json(
            {"old-name": {"name": "Old Name", "location": "harbor"}}
        )
        session.commit()

    result = compat.rename_player_character(campaign_id, actor_id, "New Name")
    assert result["ok"] is True
    assert result["old_name"] == "Old Name"
    assert result["name"] == "New Name"
    assert result["migrated_roster_slug"] == "new-name"

    refreshed = compat.get_or_create_player(campaign_id, actor_id)
    refreshed_state = compat.get_player_state(refreshed)
    assert refreshed_state.get("character_name") == "New Name"

    with session_factory() as session:
        actor_row = session.get(Actor, actor_id)
        assert actor_row is not None
        assert actor_row.display_name == "New Name"
        campaign_row = session.get(Campaign, campaign_id)
        characters = json.loads(campaign_row.characters_json or "{}")
        assert "old-name" not in characters
        assert characters["new-name"]["name"] == "New Name"


class FakeHTTPResponse:
    def __init__(self, payload: bytes, status: int = 200):
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_compat(session_factory, completion_port=None, timer_effects=None, imdb_port=None, media_port=None):
    llm = StubLLM(LLMTurnOutput(narration="Compat narration"))
    engine = GameEngine(
        uow_factory=lambda: SQLAlchemyUnitOfWork(session_factory),
        llm=llm,
    )
    return ZorkEmulator(
        game_engine=engine,
        session_factory=session_factory,
        completion_port=completion_port,
        timer_effects_port=timer_effects,
        imdb_port=imdb_port,
        media_port=media_port,
    )


def test_player_stats_tracking(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

    t1 = datetime(2026, 2, 21, 12, 0, 0)
    t2 = t1 + timedelta(seconds=120)
    compat.record_player_message(player, observed_at=t1)
    stats = compat.record_player_message(player, observed_at=t2)

    assert stats[compat.PLAYER_STATS_MESSAGES_KEY] == 2
    assert stats[compat.PLAYER_STATS_ATTENTION_SECONDS_KEY] == 120
    summary = compat.get_player_statistics(player)
    assert summary["attention_hours"] == round(120 / 3600.0, 2)


def test_guardrails_onrails_timed_events_toggles(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

    assert compat.set_guardrails_enabled(campaign, True) is True
    assert compat.set_on_rails(campaign, True) is True
    assert compat.set_timed_events_enabled(campaign, False) is True

    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    assert compat.is_guardrails_enabled(campaign) is True
    assert compat.is_on_rails(campaign) is True
    assert compat.is_timed_events_enabled(campaign) is False


def test_json_parsing_helpers(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)

    payload = compat._parse_json_lenient("{'a': 1, 'b': null, 'c': true}")
    assert payload["a"] == 1
    assert payload["b"] is None
    assert payload["c"] is True

    payload2 = compat._parse_json_lenient('{"a":1}{"b":2}')
    assert payload2 == {"a": 1, "b": 2}

    cleaned = compat._clean_response("prefix ```json\n{\"x\":1}\n``` suffix")
    assert cleaned == '{"x":1}'

    truncated = '{"tool_call":"memory_search","queries":["Elizabeth"]'
    repaired = compat._clean_response(truncated)
    parsed = compat._parse_json_lenient(repaired)
    assert parsed.get("tool_call") == "memory_search"
    assert parsed.get("queries") == ["Elizabeth"]


def test_build_prompt_shape(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])

    system_prompt, user_prompt = compat.build_prompt(campaign, player, "look", turns)
    assert "You are the ZorkEmulator" in system_prompt
    assert "STRUCTURE REQUIREMENT:" in system_prompt
    assert "CALENDAR_REMINDERS" in user_prompt
    assert "ACTIVE_PLAYER_LOCATION:" in user_prompt
    assert "CAMPAIGN:" in user_prompt
    assert "CURRENT_GAME_TIME:" in user_prompt
    assert "SPEED_MULTIPLIER:" in user_prompt
    assert "MEMORY_LOOKUP_ENABLED:" in user_prompt
    assert "RECENT_TURNS_LOADED: true" in user_prompt
    assert "CALENDAR:" in user_prompt
    assert "RECENT_TURNS:\n" in user_prompt
    assert "PLAYER_ACTION" in user_prompt
    assert "other_player_state_updates" in system_prompt
    assert "not plot-armored" in system_prompt
    assert '{"tool_call": "memory_terms"' not in system_prompt
    assert '{"tool_call": "sms_list"' not in system_prompt
    assert "CALENDAR & GAME TIME SYSTEM:" not in system_prompt
    assert "CHARACTER ROSTER & PORTRAITS:" not in system_prompt


def test_build_prompt_bootstrap_stage_shape(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])

    system_prompt, user_prompt = compat.build_prompt(
        campaign,
        player,
        "look",
        turns,
        prompt_stage=compat.PROMPT_STAGE_BOOTSTRAP,
    )

    assert "continuity bootstrapper" in system_prompt
    assert '{"tool_call": "recent_turns"' in system_prompt
    assert "Do NOT narrate yet" in system_prompt
    assert "RECENT_TURNS_LOADED: false" in user_prompt
    assert "WORLD_SUMMARY:" not in user_prompt
    assert "WORLD_STATE:" not in user_prompt
    assert "CALENDAR:" not in user_prompt
    assert "STORY_CONTEXT:" not in user_prompt
    assert "RECENT_TURNS:\n" not in user_prompt
    assert "WORLD_CHARACTERS:" in user_prompt
    assert "PLAYER_CARD:" in user_prompt
    assert "PARTY_SNAPSHOT:" in user_prompt
    assert "ACTIVE_PLAYER_LOCATION:" in user_prompt
    assert "PLAYER_ACTION" in user_prompt


def test_build_prompt_research_stage_shape(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])

    system_prompt, user_prompt = compat.build_prompt(
        campaign,
        player,
        "look",
        turns,
        prompt_stage=compat.PROMPT_STAGE_RESEARCH,
    )

    assert "research planner" in system_prompt
    assert '{"tool_call": "ready_to_write", "speakers": [' in system_prompt
    assert (
        '{"tool_call": "memory_search"' in system_prompt
        or "EARLY-CAMPAIGN MEMORY MODE:" in system_prompt
    )
    assert '{"tool_call": "sms_list"' in system_prompt
    assert "CALENDAR & GAME TIME SYSTEM:" in system_prompt
    assert "Do NOT output planning prose" in system_prompt
    assert "roughly 20 minutes per turn" in system_prompt
    assert "RECENT_TURNS_LOADED: true" in user_prompt
    assert "RECENT_TURNS:\n" in user_prompt
    assert "No planning prose or self-talk in research phase" in user_prompt
    assert "WORLD_SUMMARY:" in user_prompt


def test_ready_to_write_finalization_uses_final_stage_system_prompt(
    session_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        completion = ReadyToWriteCompletionPort()
        engine = GameEngine(
            uow_factory=lambda: SQLAlchemyUnitOfWork(session_factory),
            llm=StubLLM(LLMTurnOutput(narration="unused")),
        )
        compat = ZorkEmulator(engine, session_factory, completion_port=completion)
        tool_llm = ToolAwareZorkLLM(
            session_factory=session_factory,
            completion_port=completion,
            temperature=0.8,
            max_tokens=2048,
        )
        tool_llm.bind_emulator(compat)

        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
        turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])
        research_system_prompt, research_user_prompt = compat.build_prompt(
            campaign,
            player,
            "look",
            turns,
            prompt_stage=compat.PROMPT_STAGE_RESEARCH,
        )
        final_system_prompt, final_user_prompt = compat.build_prompt(
            campaign,
            player,
            "look",
            turns,
            prompt_stage=compat.PROMPT_STAGE_FINAL,
        )

        payload = await tool_llm._resolve_payload(  # noqa: SLF001
            seed_campaign_and_actor["campaign_id"],
            seed_campaign_and_actor["actor_id"],
            "look",
            research_system_prompt,
            research_user_prompt,
            final_system_prompt,
            final_user_prompt,
        )

        assert payload is not None
        assert payload.get("narration") == "Final scene."
        assert len(completion.calls) == 2
        assert '{"tool_call": "ready_to_write"}' in completion.calls[0]["system_prompt"]
        assert '{"tool_call": "ready_to_write"}' not in completion.calls[1]["system_prompt"]

    asyncio.run(run_test())


def test_memory_search_supports_last_results_full_text_and_context_pruning(
    session_factory,
    seed_campaign_and_actor,
):
    class MemoryRefineCompletionPort:
        def __init__(self, keep_turn_id: int):
            self.keep_turn_id = keep_turn_id
            self.calls = []

        async def complete(self, system_prompt, prompt, *, temperature=0.8, max_tokens=2048):
            self.calls.append({"system_prompt": system_prompt, "prompt": prompt})
            if len(self.calls) == 1:
                return '{"tool_call":"memory_search","queries":["alpha"]}'
            if len(self.calls) == 2:
                return json.dumps(
                    {
                        "tool_call": "memory_search",
                        "search_within": "last_results",
                        "queries": ["beta"],
                        "full_text": True,
                        "keep_memory_turns": [self.keep_turn_id],
                    }
                )
            if len(self.calls) == 3:
                return '{"tool_call":"ready_to_write","speakers":[],"listeners":[]}'
            return json.dumps(
                {
                    "reasoning": "done",
                    "scene_output": {
                        "location_key": "dock-9",
                        "context_key": "",
                        "beats": [{"text": "Done."}],
                    },
                    "state_update": {
                        "game_time": {
                            "day": 1,
                            "hour": 9,
                            "minute": 0,
                            "day_of_week": "monday",
                            "period": "morning",
                            "date_label": "Monday, Day 1, Morning",
                        }
                    },
                    "summary_update": "done",
                }
            )

    async def run_test():
        compat = _build_compat(session_factory)
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])

        keep_tail = "KEEP-FULL-TEXT-TAIL"
        drop_tail = "DROP-FULL-TEXT-TAIL"
        outside_tail = "OUTSIDE-LAST-RESULTS-TAIL"
        compat._record_simple_turn_pair(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            session_id=None,
            action_text="remember alpha",
            narration=("alpha beta " + ("keep " * 240) + keep_tail),
        )
        compat._record_simple_turn_pair(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            session_id=None,
            action_text="remember alpha again",
            narration=("alpha only " + ("drop " * 240) + drop_tail),
        )
        compat._record_simple_turn_pair(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            session_id=None,
            action_text="remember beta elsewhere",
            narration=("beta outside " + ("outside " * 240) + outside_tail),
        )

        with session_factory() as session:
            narrator_turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign.id)
                .filter(Turn.kind == "narrator")
                .order_by(Turn.id.asc())
                .all()
            )
        keep_turn_id = int(narrator_turns[0].id)
        drop_turn_id = int(narrator_turns[1].id)

        completion = MemoryRefineCompletionPort(keep_turn_id)
        tool_llm = ToolAwareZorkLLM(
            session_factory=session_factory,
            completion_port=completion,
            temperature=0.8,
            max_tokens=2048,
        )
        tool_llm.bind_emulator(compat)

        payload = await tool_llm._resolve_payload(  # noqa: SLF001
            campaign.id,
            seed_campaign_and_actor["actor_id"],
            "remember alpha",
            "research system",
            "PLAYER_ACTION: remember alpha\nmemory_lookup_enabled: true",
            "final system",
            "PLAYER_ACTION: remember alpha\nmemory_lookup_enabled: true",
        )

        assert payload is not None
        assert payload.get("summary_update") == "done"
        assert len(completion.calls) == 4

        first_augmented_prompt = completion.calls[1]["prompt"]
        assert f'"turn_id":{keep_turn_id}' in first_augmented_prompt
        assert f'"turn_id":{drop_turn_id}' in first_augmented_prompt
        assert '"full_text":"' not in first_augmented_prompt
        assert keep_tail not in first_augmented_prompt
        assert drop_tail not in first_augmented_prompt
        assert outside_tail not in first_augmented_prompt

        second_augmented_prompt = completion.calls[2]["prompt"]
        assert f'"turn_id":{keep_turn_id}' in second_augmented_prompt
        assert f'"turn_id":{drop_turn_id}' not in second_augmented_prompt
        assert '"kind":"memory_context_retained"' in second_augmented_prompt
        assert '"full_text":"' in second_augmented_prompt
        assert keep_tail in second_augmented_prompt
        assert drop_tail not in second_augmented_prompt
        assert outside_tail not in second_augmented_prompt

    asyncio.run(run_test())


def test_attachment_setup_length_error_for_short_upload(session_factory):
    compat = _build_compat(session_factory)
    short_text = "One short paragraph about a character.\n\nAnother short paragraph."
    msg = compat._attachment_setup_length_error(short_text)
    assert msg is not None
    assert "too short for setup ingestion" in msg
    assert "4" in msg


def test_recent_turns_include_turn_number_and_in_game_time(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    with session_factory() as session:
        row = session.get(Campaign, campaign.id)
        state = json.loads(row.state_json or "{}")
        state["game_time"] = {"day": 2, "hour": 14, "minute": 30, "period": "afternoon"}
        row.state_json = compat._dump_json(state)
        session.commit()

    compat._record_simple_turn_pair(
        campaign_id=campaign.id,
        actor_id=seed_campaign_and_actor["actor_id"],
        session_id=None,
        action_text="look around",
        narration="A brass gate hums with static.",
    )
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])
    _, user_prompt = compat.build_prompt(campaign, player, "look", turns)

    assert "[TURN #" in user_prompt
    assert "Day 2 14:30" in user_prompt


def test_sms_thread_roundtrip(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

    ok, status = compat.write_sms_thread(
        campaign.id,
        thread="saul",
        sender="Dale",
        recipient="Saul",
        message="Meet me at Dock 9.",
    )
    assert ok is True
    assert status == "stored"

    listing = compat.list_sms_threads(campaign.id, wildcard="sa*", limit=10)
    assert listing
    assert listing[0]["thread"] == "saul"
    assert listing[0]["last_preview"].startswith("Meet me")

    thread_key, thread_label, messages = compat.read_sms_thread(campaign.id, "saul", limit=10)
    assert thread_key == "saul"
    assert thread_label is not None
    assert messages
    assert messages[-1]["from"] == "Dale"
    assert messages[-1]["to"] == "Saul"
    assert "Dock 9" in messages[-1]["message"]


def test_extract_inline_sms_intent_parses_action(session_factory):
    compat = _build_compat(session_factory)
    parsed = compat._extract_inline_sms_intent('i sms elizabeth: "what\'s your plan"')
    assert parsed is not None
    recipient, message = parsed
    assert recipient == "elizabeth"
    assert message == "what's your plan"


def test_sms_write_deduplicates_identical_message_in_same_minute(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

    with session_factory() as session:
        row = session.get(Campaign, campaign.id)
        state = compat.get_campaign_state(row)
        state["game_time"] = {"day": 3, "hour": 9, "minute": 15}
        row.state_json = compat._dump_json(state)
        session.commit()

    ok1, status1 = compat.write_sms_thread(
        campaign.id,
        thread="elizabeth",
        sender="Deshawn",
        recipient="Elizabeth",
        message="what's your plan",
    )
    ok2, status2 = compat.write_sms_thread(
        campaign.id,
        thread="elizabeth",
        sender="Deshawn",
        recipient="Elizabeth",
        message="what's your plan",
    )
    assert ok1 is True and status1 == "stored"
    assert ok2 is True and status2 == "stored"

    _thread_key, _label, messages = compat.read_sms_thread(campaign.id, "elizabeth", limit=20)
    assert len(messages) == 1


def test_sms_read_merges_related_threads_by_participant(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

    ok1, status1 = compat.write_sms_thread(
        campaign.id,
        thread="elizabeth",
        sender="Deshawn",
        recipient="Elizabeth",
        message="what you up to?",
    )
    ok2, status2 = compat.write_sms_thread(
        campaign.id,
        thread="deshawn",
        sender="Elizabeth",
        recipient="Deshawn",
        message="in the writers room. hungry soon?",
    )
    assert ok1 is True and status1 == "stored"
    assert ok2 is True and status2 == "stored"

    thread_key, thread_label, messages = compat.read_sms_thread(campaign.id, "elizabeth", limit=20)
    assert thread_key is not None
    assert thread_label is not None
    assert len(messages) >= 2
    assert any(str(row.get("from")) == "Deshawn" and str(row.get("to")) == "Elizabeth" for row in messages)
    assert any(str(row.get("from")) == "Elizabeth" and str(row.get("to")) == "Deshawn" for row in messages)


def test_sms_list_and_read_prefer_roster_contact_name_over_legacy_pair_thread(
    session_factory,
    seed_campaign_and_actor,
):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])

    with session_factory() as session:
        player = (
            session.query(Player)
            .filter(Player.campaign_id == campaign.id)
            .filter(Player.actor_id == seed_campaign_and_actor["actor_id"])
            .first()
        )
        assert player is not None
        player_state = json.loads(player.state_json or "{}")
        player_state["character_name"] = "Chris"
        player.state_json = compat._dump_json(player_state)

        row = session.get(Campaign, campaign.id)
        characters = json.loads(row.characters_json or "{}")
        characters["crawly"] = {"name": "Crawly"}
        row.characters_json = compat._dump_json(characters)
        session.commit()

    ok, status = compat.write_sms_thread(
        campaign.id,
        thread="chris-crawly",
        sender="Chris",
        recipient="Crawly",
        message="you around?",
        owner_actor_id=seed_campaign_and_actor["actor_id"],
    )
    assert ok is True and status == "stored"

    listed = compat.list_sms_threads(
        campaign.id,
        wildcard="*",
        limit=20,
        viewer_actor_id=seed_campaign_and_actor["actor_id"],
    )
    assert listed
    assert listed[0]["thread"] == "crawly"
    assert listed[0]["label"] == "Crawly"

    canonical, label, messages = compat.read_sms_thread(
        campaign.id,
        "crawly",
        limit=20,
        viewer_actor_id=seed_campaign_and_actor["actor_id"],
    )
    assert canonical == "crawly"
    assert label == "Crawly"
    assert len(messages) == 1
    assert str(messages[0]["thread"]).endswith("actor-actor-1")


def test_sms_schedule_delivers_later(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

    async def run_case():
        ok, reason, delay = compat.schedule_sms_thread_delivery(
            campaign.id,
            thread="elizabeth",
            sender="Elizabeth",
            recipient="Deshawn",
            message="On my way.",
            delay_seconds=0,
        )
        assert ok is True
        assert reason == "scheduled"
        assert delay == 0

        _thread_key, _label, before = compat.read_sms_thread(campaign.id, "elizabeth", limit=20)
        assert before == []

        await asyncio.sleep(0.05)
        _thread_key, _label, after = compat.read_sms_thread(campaign.id, "elizabeth", limit=20)
        assert len(after) == 1
        assert str(after[0].get("from")) == "Elizabeth"
        assert str(after[0].get("to")) == "Deshawn"
        assert str(after[0].get("message")) == "On my way."

    asyncio.run(run_case())


def test_sms_schedule_can_be_cancelled(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

    async def run_case():
        ok, reason, _delay = compat.schedule_sms_thread_delivery(
            campaign.id,
            thread="saul",
            sender="Saul",
            recipient="Dale",
            message="Hold tight.",
            delay_seconds=1,
        )
        assert ok is True
        assert reason == "scheduled"
        cancelled = compat.cancel_pending_sms_deliveries(campaign.id)
        assert cancelled >= 1
        await asyncio.sleep(0.1)
        _thread_key, _label, messages = compat.read_sms_thread(campaign.id, "saul", limit=20)
        assert messages == []

    asyncio.run(run_case())


def test_memory_search_usage_hint_recommends_roster_key_after_repeats(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

    hints1 = compat.record_memory_search_usage(campaign.id, ["elizabeth"])
    hints2 = compat.record_memory_search_usage(campaign.id, ["elizabeth"])
    hints3 = compat.record_memory_search_usage(campaign.id, ["elizabeth"])

    assert hints1 == []
    assert hints2 == []
    assert hints3
    assert hints3[0]["slug"] == "elizabeth"
    assert int(hints3[0]["count"]) >= 3


def test_memory_search_usage_hint_skips_if_rostered(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

    with session_factory() as session:
        row = session.get(Campaign, campaign.id)
        row.characters_json = compat._dump_json({"elizabeth": {"name": "Elizabeth Chen"}})
        session.commit()

    hints = []
    for _ in range(3):
        hints = compat.record_memory_search_usage(campaign.id, ["elizabeth"])
    assert hints == []


def test_recent_turns_keep_full_content(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    long_narration = "BEGIN_MARKER " + ("alpha " * 260) + "END_MARKER"

    with session_factory() as session:
        session.add(
            Turn(
                campaign_id=campaign.id,
                session_id=None,
                actor_id=player.actor_id,
                kind="narrator",
                content=long_narration,
            )
        )
        session.commit()

    turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])
    _system_prompt, user_prompt = compat.build_prompt(campaign, player, "look", turns)
    assert "BEGIN_MARKER" in user_prompt
    assert "END_MARKER" in user_prompt
    assert "...[truncated]" not in user_prompt


def test_build_prompt_seeds_default_game_time(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])

    _, user_prompt = compat.build_prompt(campaign, player, "look", turns)
    state = json.loads(campaign.state_json or "{}")
    game_time = state.get("game_time", {})

    assert game_time.get("day") == 1
    assert game_time.get("hour") == 8
    assert game_time.get("minute") == 0
    assert game_time.get("day_of_week") == "monday"
    assert game_time.get("period") == "morning"
    assert game_time.get("date_label") == "Monday, Day 1, Morning"
    assert '"day": 1' in user_prompt
    assert "WORLD_CLOCK:" in user_prompt
    assert "CURRENT_GAME_TIME:" in user_prompt


def test_format_game_time_label_preserves_existing_weekday_without_campaign_state(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)

    label = compat._format_game_time_label(
        {
            "day": 3,
            "hour": 8,
            "minute": 0,
            "day_of_week": "sunday",
            "date_label": "Sunday, Day 3, Morning",
        }
    )

    assert label == "Sunday, Day 3, Morning"


def test_set_player_known_game_time_uses_campaign_clock_weekday(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    campaign_state = compat.get_campaign_state(campaign)
    campaign_state["clock_start_day_of_week"] = "friday"
    campaign.state_json = compat._dump_json(campaign_state)

    compat._set_player_known_game_time(
        player,
        {"day": 3, "hour": 8, "minute": 0},
        campaign_state=campaign_state,
    )

    player_state = compat.get_player_state(player)
    assert player_state["game_time"]["day_of_week"] == "sunday"
    assert player_state["game_time"]["date_label"] == "Sunday, Day 3, Morning"


def test_build_prompt_persists_default_clock_start_day_when_missing(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])

    with session_factory() as session:
        campaign_row = session.get(Campaign, campaign.id)
        campaign_row.state_json = compat._dump_json(
            {
                "game_time": {
                    "day": 1,
                    "hour": 8,
                    "minute": 0,
                    "day_of_week": "monday",
                    "period": "morning",
                    "date_label": "Monday, Day 1, Morning",
                }
            }
        )
        session.commit()

    compat.build_prompt(campaign, player, "look", turns)
    state = json.loads(campaign.state_json or "{}")
    assert state.get("clock_start_day_of_week") == "monday"


def test_build_prompt_includes_weekday_in_world_clock(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])

    with session_factory() as session:
        campaign_row = session.get(Campaign, campaign.id)
        campaign_row.state_json = compat._dump_json(
            {
                "clock_start_day_of_week": "friday",
                "game_time": {"day": 3, "hour": 8, "minute": 0},
            }
        )
        session.commit()

    _, user_prompt = compat.build_prompt(campaign, player, "look", turns)

    assert '"day_of_week": "sunday"' in user_prompt
    assert '"date_label": "Sunday, Day 3, Morning"' in user_prompt
    assert "WORLD_CLOCK:" in user_prompt


def test_story_context_includes_next_three_and_coerces_progress_indices(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)

    campaign_state = {
        "current_chapter": "1",
        "current_scene": "2",
        "story_outline": {
            "chapters": [
                {"title": "Ch1", "summary": "one", "scenes": [{"title": "A1"}]},
                {"title": "Ch2", "summary": "two", "scenes": [{"title": "B1"}, {"title": "B2"}, {"title": "B3"}]},
                {"title": "Ch3", "summary": "three", "scenes": [{"title": "C1"}]},
                {"title": "Ch4", "summary": "four", "scenes": [{"title": "D1"}]},
                {"title": "Ch5", "summary": "five", "scenes": [{"title": "E1"}]},
            ]
        },
    }

    story_context = compat._build_story_context(campaign_state)
    assert story_context is not None
    assert "CURRENT CHAPTER: Ch2" in story_context
    assert "Scene 3: B3 >>> CURRENT SCENE <<<" in story_context
    assert "NEXT CHAPTER: Ch3" in story_context
    assert "UPCOMING CHAPTER 2: Ch4" in story_context
    assert "UPCOMING CHAPTER 3: Ch5" in story_context


def test_story_context_uses_offrails_active_chapters(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)

    campaign_state = {
        "on_rails": False,
        "chapters": [
            {
                "slug": "friday-night-after",
                "title": "Friday Night, After",
                "summary": "Everything is different and nothing shows.",
                "current_scene": "boundary-test",
                "scenes": ["boundary-test", "the-cat-knows", "closing-approaches"],
                "status": "active",
            },
            {
                "slug": "health-tax",
                "title": "The Health Tax",
                "summary": "Monet cuts him off because she noticed.",
                "current_scene": "the-cutoff",
                "scenes": ["the-cutoff", "what-follows"],
                "status": "active",
            },
        ],
    }

    story_context = compat._build_story_context(campaign_state)
    assert story_context is not None
    assert "CURRENT CHAPTER: Friday Night, After" in story_context
    assert "Scene 1: Boundary Test >>> CURRENT SCENE <<<" in story_context
    assert "NEXT CHAPTER: The Health Tax" in story_context


def test_build_prompt_places_story_context_above_world_summary_and_composes_summary(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])

    with session_factory() as session:
        campaign_row = session.get(Campaign, campaign.id)
        campaign_row.summary = "lel without elaboration.\nInventory: stale pretzel\nKevin accepted the drink."
        campaign_row.state_json = compat._dump_json(
            {
                "game_time": {"day": 1, "hour": 8, "minute": 0, "day_of_week": "monday", "period": "morning", "date_label": "Monday, Day 1, Morning"},
                "story_outline": {
                    "chapters": [
                        {
                            "title": "Chapter One",
                            "summary": "A real chapter summary.",
                            "scenes": [{"title": "Opening"}],
                        }
                    ]
                },
                "current_chapter": 0,
                "current_scene": 0,
            }
        )
        session.commit()

    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    _system_prompt, user_prompt = compat.build_prompt(campaign, player, "look", turns)
    assert "WORLD_SUMMARY: Kevin accepted the drink." in user_prompt
    assert "lel without elaboration." not in user_prompt
    assert "Inventory: stale pretzel" not in user_prompt
    assert user_prompt.index("STORY_CONTEXT:") < user_prompt.index("WORLD_SUMMARY:")


def test_generate_map_prompt_uses_authoritative_location_keys(session_factory, seed_campaign_and_actor):
    async def run_test():
        map_port = CaptureMapCompletionPort()
        compat = _build_compat(session_factory, completion_port=map_port)
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        primary = compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])
        other = compat.get_or_create_player(campaign.id, "actor-2")
        with session_factory() as session:
            primary_row = session.get(Player, primary.id)
            primary_row.state_json = compat._dump_json(
                {
                    "character_name": "Dale",
                    "location": "chp-cruiser",
                    "room_title": "CHP Cruiser - En Route",
                    "room_summary": "Back seat of a patrol cruiser heading toward LA.",
                }
            )
            other_row = session.get(Player, other.id)
            other_row.state_json = compat._dump_json(
                {
                    "character_name": "Arsipea",
                    "location": "victorville-facility",
                    "room_title": "Victorville Federal Facility - Visitor B",
                    "room_summary": "Institutional visitor room under fluorescent lights.",
                }
            )
            campaign_row = session.get(Campaign, campaign.id)
            campaign_row.characters_json = compat._dump_json(
                {
                    "warden-rollins": {"name": "Rollins", "location": "victorville-facility"},
                    "driver-hobbs": {"name": "Hobbs", "location": "chp-cruiser"},
                }
            )
            session.commit()

        map_text = await compat.generate_map(campaign.id, actor_id=seed_campaign_and_actor["actor_id"])
        assert map_text
        assert map_port.calls
        prompt = map_port.calls[-1]["prompt"]
        assert "PLAYER_LOCATION_KEY:" in prompt
        assert "WORLD_CHARACTER_LOCATIONS:" in prompt
        assert '"location_key"' in prompt
        assert "MAP_SPATIAL_RULES:" in prompt

    asyncio.run(run_test())


def test_prompt_budget_constants_match_upstream_latest():
    assert ZorkEmulator.MAX_SUMMARY_CHARS == 10000
    assert ZorkEmulator.MAX_STATE_CHARS == 10000
    assert ZorkEmulator.MAX_NARRATION_CHARS == 23500
    assert ZorkEmulator.MAX_CHARACTERS_CHARS == 8000


def test_setup_flow_with_attachment_and_confirm(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(
            session_factory,
            completion_port=StubCompletionPort(),
            imdb_port=StubIMDB(),
        )
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

        msg = await compat.start_campaign_setup(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            raw_name="Matrix",
            on_rails=True,
        )
        assert "I recognize" in msg

        genre_msg = await compat.handle_setup_message(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            message_text="yes",
            attachments=[StubAttachment("lore.txt", b"Neo wakes up in a false city.\n\nAgents hunt him.")],
        )
        assert "genre direction" in genre_msg.lower()

        variants_msg = await compat.handle_setup_message(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            message_text="noir",
        )
        assert "Choose a storyline variant" in variants_msg
        assert "retry: <guidance>" in variants_msg
        assert "retry: make it darker" in variants_msg

        done_msg = await compat.handle_setup_message(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            message_text="1",
        )
        assert "is ready" in done_msg

        campaign2 = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        assert compat.is_in_setup_mode(campaign2) is False
        state = compat.get_campaign_state(campaign2)
        assert state.get("setting") == "Neo-noir Seattle"

    asyncio.run(run_test())


def test_classify_confirm_negative_with_novel_guidance_skips_reclassify(
    session_factory, seed_campaign_and_actor
):
    async def run_test():
        probe = NovelIntentProbeCompletionPort()
        compat = _build_compat(
            session_factory,
            completion_port=probe,
            imdb_port=StubIMDB(),
        )
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

        msg = await compat.start_campaign_setup(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            raw_name="Matrix",
            on_rails=True,
        )
        assert "Is this correct?" in msg
        assert probe.initial_classify_calls == 1

        genre_msg = await compat.handle_setup_message(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            message_text="no, i'd rather do a novel thing where the moon is a prison colony",
        )
        assert "genre direction" in genre_msg.lower()
        assert probe.reclassify_calls == 0

        variants_msg = await compat.handle_setup_message(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            message_text="noir",
        )
        assert "Choose a storyline variant" in variants_msg
        assert probe.variant_prompts
        assert "moon is a prison colony" in probe.variant_prompts[-1].lower()

        with session_factory() as session:
            row = session.get(Campaign, campaign.id)
            state = json.loads(row.state_json or "{}")
            setup = state.get("setup_data", {})
            assert setup.get("is_known_work") is False
            assert setup.get("imdb_results") == []

    asyncio.run(run_test())


def test_timer_runtime_emits_effect(session_factory, seed_campaign_and_actor):
    async def run_test():
        timer_effects = StubTimerEffects()
        compat = _build_compat(
            session_factory,
            timer_effects=timer_effects,
        )
        compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

        compat._schedule_timer(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            channel_id="chan-1",
            delay_seconds=0,
            event_description="The floor collapses.",
            interruptible=True,
        )
        await asyncio.sleep(0.05)
        assert timer_effects.emits
        campaign_id, channel_id, actor_id, narration = timer_effects.emits[-1]
        assert campaign_id == seed_campaign_and_actor["campaign_id"]
        assert channel_id == "chan-1"
        assert actor_id == seed_campaign_and_actor["actor_id"]
        assert narration is not None

        with session_factory() as session:
            turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == seed_campaign_and_actor["campaign_id"])
                .order_by(Turn.id.asc())
                .all()
            )
            assert turns
            assert all(t.kind == "narrator" for t in turns)

    asyncio.run(run_test())


def test_room_scene_image_store_get_clear(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

    assert compat.record_room_scene_image_url_for_channel(
        guild_id="default",
        channel_id="main",
        room_key="dock-9",
        image_url="https://example.com/scene.png",
        campaign_id=campaign.id,
        scene_prompt="wet steel pier",
    )
    assert compat.get_room_scene_image_url(campaign, "dock-9") == "https://example.com/scene.png"
    assert compat.clear_room_scene_image_url(campaign, "dock-9") is True
    assert compat.get_room_scene_image_url(campaign, "dock-9") is None


def test_avatar_pending_accept_decline(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

    assert compat.record_pending_avatar_image_for_campaign(
        campaign_id=seed_campaign_and_actor["campaign_id"],
        user_id=seed_campaign_and_actor["actor_id"],
        image_url="https://example.com/avatar.png",
        avatar_prompt="mysterious detective",
    )
    ok, msg = compat.accept_pending_avatar(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    assert ok is True
    assert "Avatar accepted" in msg

    ok2, msg2 = compat.decline_pending_avatar(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    assert ok2 is False
    assert "No pending avatar" in msg2


def test_inventory_delta_sanitization(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    previous = {
        "location": "dock",
        "inventory": [{"name": "Rusty Key", "origin": "Found in locker"}],
        "room_title": "Dock 9",
        "room_description": "Rain hisses on steel.",
    }
    update = {
        "inventory_add": ["Lantern"],
        "inventory_remove": ["Rusty Key"],
        "location": "warehouse",
    }
    cleaned = compat._sanitize_player_state_update(previous, update, action_text="enter warehouse")
    names = [item["name"] for item in cleaned["inventory"]]
    assert "Lantern" in names
    assert "Rusty Key" not in names
    assert cleaned["room_description"] is None
    assert cleaned["room_title"] is None
    assert cleaned["room_summary"] is None


def test_media_enqueue_hooks(session_factory, seed_campaign_and_actor):
    async def run_test():
        media = StubMediaPort()
        compat = _build_compat(session_factory, media_port=media)
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

        ok, message = await compat.enqueue_avatar_generation(
            StubCtx(),
            campaign=campaign,
            player=player,
            requested_prompt="long coat, noir hero",
        )
        assert ok is True
        assert "queued" in message.lower()
        assert media.avatar_calls

        ok2 = await compat.enqueue_scene_composite_from_seed(
            channel=StubCtx().channel,
            campaign_id=campaign.id,
            room_key="dock-9",
            user_id=seed_campaign_and_actor["actor_id"],
            scene_prompt="Neo meets Mira under sodium lights.",
            base_image_url="https://example.com/base-scene.png",
        )
        assert ok2 is True
        assert media.scene_calls

    asyncio.run(run_test())


def test_character_portrait_helpers(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    with session_factory() as session:
        row = session.get(Campaign, campaign.id)
        row.characters_json = compat._dump_json({"mira-guide": {"name": "Mira"}})
        session.commit()

    assert compat.record_character_portrait_url(
        campaign_id=campaign.id,
        character_slug="mira-guide",
        image_url="https://example.com/mira.png",
    )
    refreshed = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    characters = compat.get_campaign_characters(refreshed)
    assert characters["mira-guide"]["image_url"] == "https://example.com/mira.png"
    prompt = compat._compose_character_portrait_prompt("Mira", "scar across one cheek")
    assert "Character portrait of Mira." in prompt


def test_apply_character_updates_remove_slug_on_null(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    existing = {
        "mira-guide": {"name": "Mira", "location": "Dock 9"},
        "jet-smuggler": {"name": "Jet", "location": "Market"},
    }
    updated = compat._apply_character_updates(
        existing,
        {"mira-guide": None, "jet-smuggler": {"current_status": "watchful"}},
    )
    assert "mira-guide" not in updated
    assert updated["jet-smuggler"]["current_status"] == "watchful"
    updated2 = compat._apply_character_updates(
        {"mira-guide": {"name": "Mira", "location": "Dock 9"}},
        {"Mira Guide": {"remove": True}},
    )
    assert "mira-guide" not in updated2
    derived = compat._character_updates_from_state_nulls(
        {"Mira": None, "irrelevant_state_key": None},
        {"mira-guide": {"name": "Mira", "location": "Dock 9"}},
    )
    assert derived == {"mira-guide": None}


def test_new_character_auto_enqueues_portrait(
    uow_factory,
    session_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        media = StubMediaPort()
        llm = StubLLM(
            LLMTurnOutput(
                narration="A wary scout steps out of the fog.",
                character_updates={
                    "mira-guide": {
                        "name": "Mira",
                        "appearance": "Lean build, stormcloak, and a scar over one brow.",
                        "location": "dock-9",
                    }
                },
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory, media_port=media)
        compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

        out = await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="look",
        )
        assert out is not None
        assert media.avatar_calls
        call = media.avatar_calls[-1]
        assert call["metadata"].get("zork_store_character_portrait") is True
        assert call["metadata"].get("zork_character_slug") == "mira-guide"
        assert "Character portrait of Mira." in call["prompt"]

    asyncio.run(run_test())


def test_attachment_helpers_summarise_chunk_guard_retry(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(session_factory, completion_port=GuardRetryCompletionPort())
        out = await compat._summarise_chunk(
            "chunk payload",
            summarise_system="Summarise the following text passage for a text-adventure campaign.",
            summary_max_tokens=900,
            guard="--COMPLETED SUMMARY--",
        )
        assert out == "second attempt with guard"
        assert isinstance(compat._completion_port, GuardRetryCompletionPort)
        assert compat._completion_port.calls == 2

    asyncio.run(run_test())


def test_attachment_helpers_condense_fallback_on_error(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(session_factory, completion_port=FailingCondenseCompletionPort())
        idx, condensed = await compat._condense(
            4,
            "summary text",
            target_tokens_per=500,
            target_chars_per=1500,
            guard="--COMPLETED SUMMARY--",
        )
        assert idx == 4
        assert condensed == "summary text"

    asyncio.run(run_test())


def test_legacy_begin_turn_and_play_action_signatures(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(session_factory)
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        player = compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])
        compat.enable_channel("default", "main", seed_campaign_and_actor["actor_id"])
        with session_factory() as session:
            row = session.get(Player, player.id)
            state = compat.get_player_state(player)
            state["party_status"] = "main_party"
            row.state_json = compat._dump_json(state)
            session.commit()
        ctx = LegacyCtx(seed_campaign_and_actor["actor_id"], guild_id="default", channel_id="main")

        campaign_id, error = await compat.begin_turn(ctx, command_prefix="!")
        assert error is None
        assert campaign_id == campaign.id
        compat.end_turn(campaign.id, seed_campaign_and_actor["actor_id"])

        narration = await compat.play_action(
            ctx,
            "look around",
            command_prefix="!",
            campaign_id=campaign.id,
        )
        assert narration is not None
        assert narration.startswith("Compat narration")
        assert "\n\nInventory: empty" in narration

    asyncio.run(run_test())


def test_context_onboarding_requires_party_choice(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(session_factory)
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])
        compat.enable_channel("default", "main", seed_campaign_and_actor["actor_id"])
        ctx = LegacyCtx(seed_campaign_and_actor["actor_id"], guild_id="default", channel_id="main")

        response = await compat.play_action(
            ctx,
            "look around",
            command_prefix="!",
            campaign_id=campaign.id,
        )
        assert response is not None
        assert "Mission rejected until path is selected." in response

        player = compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])
        state = compat.get_player_state(player)
        assert state.get("onboarding_state") == "await_party_choice"

    asyncio.run(run_test())


def test_context_shortcuts_look_and_inventory(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(session_factory)
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        player = compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])
        compat.enable_channel("default", "main", seed_campaign_and_actor["actor_id"])
        with session_factory() as session:
            row = session.get(Player, player.id)
            row.state_json = compat._dump_json(
                {
                    "party_status": "main_party",
                    "room_title": "Dock 9",
                    "room_description": "Rain hisses on steel.",
                    "exits": ["warehouse", "alley"],
                    "inventory": [{"name": "Lantern", "origin": "locker"}],
                }
            )
            session.commit()

        ctx = LegacyCtx(seed_campaign_and_actor["actor_id"], guild_id="default", channel_id="main")
        look_resp = await compat.play_action(
            ctx,
            "look",
            command_prefix="!",
            campaign_id=campaign.id,
        )
        assert "Dock 9" in (look_resp or "")
        assert "Rain hisses on steel." in (look_resp or "")
        assert "Exits: warehouse, alley" in (look_resp or "")
        assert "Inventory: Lantern" in (look_resp or "")

        inv_resp = await compat.play_action(
            ctx,
            "inventory",
            command_prefix="!",
            campaign_id=campaign.id,
        )
        assert inv_resp == "Inventory: Lantern"

    asyncio.run(run_test())


def test_context_shortcuts_calendar_and_roster(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(session_factory)
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        player = compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])
        compat.enable_channel("default", "main", seed_campaign_and_actor["actor_id"])
        with session_factory() as session:
            player_row = session.get(Player, player.id)
            player_row.state_json = compat._dump_json({"party_status": "main_party"})
            campaign_row = session.get(Campaign, campaign.id)
            campaign_state = compat.get_campaign_state(campaign)
            campaign_state["game_time"] = {"day": 3, "hour": 19, "period": "evening"}
            campaign_state["calendar"] = [
                {
                    "name": "Moonrise Ceremony",
                    "fire_day": 3,
                    "fire_hour": 22,
                    "description": "Lanterns gather at the old plaza",
                }
            ]
            campaign_row.state_json = compat._dump_json(campaign_state)
            campaign_row.characters_json = compat._dump_json(
                {
                    "mira-guide": {
                        "name": "Mira",
                        "location": "Dock 9",
                        "current_status": "watchful",
                        "background": "A veteran smuggler. Knows every back alley.",
                        "image_url": "https://example.com/mira.png",
                    }
                }
            )
            session.commit()

        ctx = LegacyCtx(seed_campaign_and_actor["actor_id"], guild_id="default", channel_id="main")

        calendar_resp = await compat.play_action(
            ctx,
            "calendar",
            command_prefix="!",
            campaign_id=campaign.id,
        )
        assert calendar_resp is not None
        assert "**Game Time:** Day 3, Evening" in calendar_resp
        assert "Moonrise Ceremony" in calendar_resp
        assert "fires in 3 hour(s)" in calendar_resp

        roster_resp = await compat.play_action(
            ctx,
            "roster",
            command_prefix="!",
            campaign_id=campaign.id,
        )
        assert roster_resp is not None
        assert "**Character Roster:**" in roster_resp
        assert "Mira" in roster_resp
        assert "Dock 9" in roster_resp
        assert "Portrait:" not in roster_resp
        assert "https://example.com/mira.png" not in roster_resp

    asyncio.run(run_test())


def test_calendar_reminders_suppress_non_milestone_hours(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    text = compat._calendar_reminder_text(
        [
            {
                "name": "Media Exposure Fallout",
                "hours_remaining": 30,
                "fire_day": 5,
                "fire_hour": 14,
            }
        ]
    )
    assert text == "None"


def test_calendar_reminders_include_milestone_hours(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    text = compat._calendar_reminder_text(
        [
            {
                "name": "Media Exposure Fallout",
                "hours_remaining": 24,
                "fire_day": 5,
                "fire_hour": 14,
            }
        ]
    )
    assert "SOON: Media Exposure Fallout" in text
    assert "fires in 24 hour(s)" in text


def test_calendar_reminders_respect_known_by_filter(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    text = compat._calendar_reminder_text(
        [
            {
                "name": "Blood Test",
                "hours_remaining": 24,
                "fire_day": 5,
                "fire_hour": 11,
                "known_by": ["Elizabeth", "Sasha"],
            }
        ],
        active_scene_names=["Deshawn"],
    )
    assert text == "None"


def test_calendar_reminders_surface_when_known_character_present(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    text = compat._calendar_reminder_text(
        [
            {
                "name": "Blood Test",
                "hours_remaining": 24,
                "fire_day": 5,
                "fire_hour": 11,
                "known_by": ["Elizabeth", "Sasha"],
            }
        ],
        active_scene_names=["Elizabeth"],
    )
    assert "SOON: Blood Test" in text
    assert "fires in 24 hour(s)" in text


def test_main_party_location_sync_updates_other_players(
    uow_factory,
    session_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="You enter Visitor Room B.",
                player_state_update={
                    "location": "visitor-room-b",
                    "room_title": "Visitor Room B",
                    "room_summary": "Inside Visitor Room B with Arsipea.",
                    "room_description": "A bolted table under fluorescent lights.",
                    "exits": ["hallway"],
                },
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        _p1 = compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])
        p2 = compat.get_or_create_player(campaign.id, "actor-2")
        with session_factory() as session:
            p1_row = (
                session.query(Player)
                .filter(Player.campaign_id == campaign.id)
                .filter(Player.actor_id == seed_campaign_and_actor["actor_id"])
                .first()
            )
            p1_row.state_json = compat._dump_json({"party_status": "main_party"})
            p2_row = session.get(Player, p2.id)
            p2_row.state_json = compat._dump_json(
                {
                    "party_status": "main_party",
                    "location": "chp-cruiser",
                    "room_title": "CHP Cruiser",
                    "room_summary": "Back seat of a patrol cruiser.",
                }
            )
            session.commit()

        out = await compat.play_action(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            action="look",
        )
        assert out is not None

        with session_factory() as session:
            p2_row = session.get(Player, p2.id)
            p2_state = json.loads(p2_row.state_json or "{}")
            assert p2_state.get("location") == "visitor-room-b"
            assert p2_state.get("room_title") == "Visitor Room B"
            assert "Visitor Room B" in str(p2_state.get("room_summary") or "")

    asyncio.run(run_test())


def test_calendar_update_keeps_overdue_and_requires_explicit_remove(
    session_factory, seed_campaign_and_actor
):
    compat = _build_compat(session_factory)
    campaign_state = {
        "game_time": {"day": 2, "hour": 9},
        "calendar": [
            {
                "name": "Moonrise Ceremony",
                "fire_day": 2,
                "description": "Late but still relevant",
            }
        ],
    }

    updated = compat._apply_calendar_update(
        campaign_state,
        {
            "add": [
                {
                    "name": "Moonrise Ceremony",
                    "time_remaining": -1,
                    "time_unit": "days",
                    "description": "Consequences escalating",
                }
            ]
        },
    )
    calendar = updated.get("calendar", [])
    assert isinstance(calendar, list)
    assert len([e for e in calendar if e.get("name") == "Moonrise Ceremony"]) == 1
    assert any(
        e.get("name") == "Moonrise Ceremony" and e.get("fire_day") == 1
        for e in calendar
    )

    removed = compat._apply_calendar_update(updated, {"remove": ["Moonrise Ceremony"]})
    assert all(e.get("name") != "Moonrise Ceremony" for e in removed.get("calendar", []))


def test_legacy_setup_signatures(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(
            session_factory,
            completion_port=StubCompletionPort(),
            imdb_port=StubIMDB(),
        )
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        ctx = LegacyCtx(
            seed_campaign_and_actor["actor_id"],
            guild_id="default",
            channel_id="main",
            attachments=[StubAttachment("lore.txt", b"Neo wakes up in a false city.\n\nAgents hunt him.")],
        )

        msg = await compat.start_campaign_setup(
            campaign,
            "Matrix",
            attachment_summary="short source summary",
        )
        assert "I recognize" in msg

        genre_msg = await compat.handle_setup_message(
            ctx,
            "yes",
            campaign,
            command_prefix="!",
        )
        assert "genre direction" in genre_msg.lower()

        variants = await compat.handle_setup_message(
            ctx,
            "noir",
            campaign,
            command_prefix="!",
        )
        assert "Choose a storyline variant" in variants

        done = await compat.handle_setup_message(
            ctx,
            "1",
            campaign,
            command_prefix="!",
        )
        assert "is ready" in done

    asyncio.run(run_test())


def test_imdb_progressive_fallback_and_formatting(monkeypatch, session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    calls: list[str] = []
    responses = [
        {"d": []},
        {
            "d": [
                {
                    "id": "tt0133093",
                    "l": "The Matrix",
                    "y": 1999,
                    "q": "feature",
                    "s": "Keanu Reeves",
                }
            ]
        },
    ]

    def fake_urlopen(request, timeout=0):
        calls.append(request.full_url)
        payload = responses.pop(0)
        return FakeHTTPResponse(json.dumps(payload).encode("utf-8"), status=200)

    monkeypatch.setattr("text_game_engine.zork_emulator.urllib_request.urlopen", fake_urlopen)
    results = compat._imdb_search("The Matrix S01E01", max_results=3)
    assert len(calls) >= 2
    assert results
    assert results[0]["title"] == "The Matrix"
    assert results[0]["imdb_id"] == "tt0133093"

    formatted = compat._format_imdb_results(results)
    assert "- The Matrix (1999) [feature] — Keanu Reeves" in formatted


def test_imdb_detail_jsonld_parsing(monkeypatch, session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    html = """
    <html><head></head><body>
    <script type="application/ld+json">
    {"description":"A hacker learns reality is simulated.","genre":["Action","Sci-Fi"],"actor":[{"name":"Keanu Reeves"},{"name":"Carrie-Anne Moss"}]}
    </script>
    </body></html>
    """.strip()

    def fake_urlopen(request, timeout=0):
        return FakeHTTPResponse(html.encode("utf-8"), status=200)

    monkeypatch.setattr("text_game_engine.zork_emulator.urllib_request.urlopen", fake_urlopen)
    details = compat._imdb_fetch_details("tt0133093")
    assert details["description"] == "A hacker learns reality is simulated."
    assert details["genre"] == ["Action", "Sci-Fi"]
    assert "Keanu Reeves" in details["actors"]


def test_inflight_turn_claim_lifecycle(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(session_factory)
        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])

        compat._clear_inflight_turn(campaign.id, seed_campaign_and_actor["actor_id"])
        cid, err = await compat.begin_turn(campaign.id, seed_campaign_and_actor["actor_id"])
        assert err is None
        assert cid == campaign.id

        cid2, err2 = await compat.begin_turn(campaign.id, seed_campaign_and_actor["actor_id"])
        assert err2 is None
        assert cid2 is None

        compat.end_turn(campaign.id, seed_campaign_and_actor["actor_id"])
        cid3, err3 = await compat.begin_turn(campaign.id, seed_campaign_and_actor["actor_id"])
        assert err3 is None
        assert cid3 == campaign.id
        compat.end_turn(campaign.id, seed_campaign_and_actor["actor_id"])

    asyncio.run(run_test())


def test_timed_event_race_guard_skips_when_recent_player_turn(
    session_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        timer_effects = StubTimerEffects()
        compat = _build_compat(session_factory, timer_effects=timer_effects)
        player = compat.get_or_create_player(
            seed_campaign_and_actor["campaign_id"],
            seed_campaign_and_actor["actor_id"],
        )
        assert player is not None

        with session_factory() as session:
            session.add(
                Turn(
                    campaign_id=seed_campaign_and_actor["campaign_id"],
                    actor_id=seed_campaign_and_actor["actor_id"],
                    kind="player",
                    content="look",
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            session.commit()

        await compat._execute_timed_event(
            seed_campaign_and_actor["campaign_id"],
            "chan-1",
            "The floor collapses.",
        )

        assert timer_effects.emits == []

        with session_factory() as session:
            turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == seed_campaign_and_actor["campaign_id"])
                .all()
            )
            assert len(turns) == 1

    asyncio.run(run_test())


def test_zork_log_writes_file(monkeypatch, tmp_path, session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    log_path = tmp_path / "zork.log"
    monkeypatch.setattr("text_game_engine.zork_emulator._ZORK_LOG_PATH", str(log_path))

    compat._zork_log("TEST SECTION", "hello world")

    text = log_path.read_text(encoding="utf-8")
    assert "TEST SECTION" in text
    assert "hello world" in text


def test_play_action_appends_inventory_and_timer_and_persists(
    uow_factory,
    session_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="Storm gathers over the pier.",
                player_state_update={"inventory": [{"name": "Lantern", "origin": "Found in locker"}]},
                timer_instruction=TimerInstruction(
                    delay_seconds=60,
                    event_text="A rain wall slams across Dock 9.",
                    interruptible=True,
                ),
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)
        compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
        session_row = compat.get_or_create_session(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            surface="discord",
            surface_key="discord:test:main",
            surface_channel_id="main",
        )

        narration = await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="wait",
            session_id=session_row.id,
        )
        assert narration is not None
        assert "Inventory: Lantern" in narration
        assert "⏰ <t:" in narration
        assert "(act to prevent!)" in narration

        with session_factory() as session:
            campaign = session.get(Campaign, seed_campaign_and_actor["campaign_id"])
            assert campaign is not None
            assert campaign.last_narration == narration
            narrator_turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == seed_campaign_and_actor["campaign_id"])
                .filter(Turn.kind == "narrator")
                .order_by(Turn.id.desc())
                .first()
            )
            assert narrator_turn is not None
            assert narrator_turn.content == narration
            snapshot = session.query(Snapshot).filter(Snapshot.turn_id == narrator_turn.id).first()
            assert snapshot is not None
            assert snapshot.campaign_last_narration == narration

    asyncio.run(run_test())


def test_local_timer_only_owner_can_interrupt(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(session_factory)
        campaign_id = seed_campaign_and_actor["campaign_id"]
        owner_actor_id = seed_campaign_and_actor["actor_id"]
        other_actor_id = "actor-2"
        compat.get_or_create_player(campaign_id, owner_actor_id)
        compat.get_or_create_player(campaign_id, other_actor_id)

        compat._schedule_timer(
            campaign_id=campaign_id,
            channel_id="chan-1",
            delay_seconds=300,
            event_description="Guards close in on the alley.",
            interruptible=True,
            interrupt_scope="local",
            interrupt_actor_id=owner_actor_id,
        )

        narration_other = await compat.play_action(
            campaign_id=campaign_id,
            actor_id=other_actor_id,
            action="wait",
        )
        assert narration_other is not None
        pending_after_other = compat._pending_timers.get(campaign_id)
        assert pending_after_other is not None
        assert pending_after_other.get("interrupt_scope") == "local"

        narration_owner = await compat.play_action(
            campaign_id=campaign_id,
            actor_id=owner_actor_id,
            action="duck behind crates",
        )
        assert narration_owner is not None
        assert compat._pending_timers.get(campaign_id) is None

    asyncio.run(run_test())


def test_global_timer_can_be_interrupted_by_any_player(session_factory, seed_campaign_and_actor):
    async def run_test():
        compat = _build_compat(session_factory)
        campaign_id = seed_campaign_and_actor["campaign_id"]
        owner_actor_id = seed_campaign_and_actor["actor_id"]
        other_actor_id = "actor-2"
        compat.get_or_create_player(campaign_id, owner_actor_id)
        compat.get_or_create_player(campaign_id, other_actor_id)

        compat._schedule_timer(
            campaign_id=campaign_id,
            channel_id="chan-1",
            delay_seconds=300,
            event_description="The alarm countdown begins.",
            interruptible=True,
            interrupt_scope="global",
            interrupt_actor_id=owner_actor_id,
        )

        narration_other = await compat.play_action(
            campaign_id=campaign_id,
            actor_id=other_actor_id,
            action="slam the console switch",
        )
        assert narration_other is not None
        assert compat._pending_timers.get(campaign_id) is None

    asyncio.run(run_test())


def test_ooc_action_does_not_record_player_turn(
    uow_factory,
    session_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="Meta acknowledged."))
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)
        compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

        narration = await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="[OOC] calibrate tone",
        )
        assert narration is not None

        with session_factory() as session:
            player_turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == seed_campaign_and_actor["campaign_id"])
                .filter(Turn.kind == "player")
                .all()
            )
            narrator_turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == seed_campaign_and_actor["campaign_id"])
                .filter(Turn.kind == "narrator")
                .all()
            )
            assert player_turns == []
            assert len(narrator_turns) == 1

    asyncio.run(run_test())


def test_give_item_fallback_infers_transfer_from_narration(
    uow_factory,
    session_factory,
):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="<@2> catches the Rusty Key you toss across the room.",
                player_state_update={"inventory": []},
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        compat.get_or_create_actor("1")
        compat.get_or_create_actor("2")
        campaign = compat.get_or_create_campaign("default", "main", "1")
        source = compat.get_or_create_player(campaign.id, "1")
        compat.get_or_create_player(campaign.id, "2")
        with session_factory() as session:
            source_row = session.get(Player, source.id)
            source_row.state_json = compat._dump_json(
                {"inventory": [{"name": "Rusty Key", "origin": "Found in locker"}]}
            )
            session.commit()

        narration = await compat.play_action(
            campaign_id=campaign.id,
            actor_id="1",
            action="I toss the Rusty Key to <@2>",
        )
        assert narration is not None

        src_player = compat.get_or_create_player(campaign.id, "1")
        dst_player = compat.get_or_create_player(campaign.id, "2")
        src_names = [entry["name"] for entry in compat._get_inventory_rich(compat.get_player_state(src_player))]
        dst_items = compat._get_inventory_rich(compat.get_player_state(dst_player))
        dst_names = [entry["name"] for entry in dst_items]
        assert "Rusty Key" not in src_names
        assert "Rusty Key" in dst_names
        key_row = next(entry for entry in dst_items if entry["name"] == "Rusty Key")
        assert "Received from <@1>" in key_row.get("origin", "")

    asyncio.run(run_test())


def test_give_item_explicit_transfers_without_inventory_remove(
    uow_factory,
    session_factory,
):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="<@2> nods and pockets the Rusty Key.",
                give_item=GiveItemInstruction(item="Rusty Key", to_discord_mention="<@2>"),
                player_state_update={},
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        compat.get_or_create_actor("1")
        compat.get_or_create_actor("2")
        campaign = compat.get_or_create_campaign("default", "main", "1")
        source = compat.get_or_create_player(campaign.id, "1")
        compat.get_or_create_player(campaign.id, "2")
        with session_factory() as session:
            source_row = session.get(Player, source.id)
            source_row.state_json = compat._dump_json(
                {"inventory": [{"name": "Rusty Key", "origin": "Found in locker"}]}
            )
            session.commit()

        narration = await compat.play_action(
            campaign_id=campaign.id,
            actor_id="1",
            action="I give the Rusty Key to <@2>",
        )
        assert narration is not None

        src_player = compat.get_or_create_player(campaign.id, "1")
        dst_player = compat.get_or_create_player(campaign.id, "2")
        src_names = [entry["name"] for entry in compat._get_inventory_rich(compat.get_player_state(src_player))]
        dst_items = compat._get_inventory_rich(compat.get_player_state(dst_player))
        dst_names = [entry["name"] for entry in dst_items]
        assert "Rusty Key" not in src_names
        assert "Rusty Key" in dst_names
        key_row = next(entry for entry in dst_items if entry["name"] == "Rusty Key")
        assert "Received from <@1>" in key_row.get("origin", "")

    asyncio.run(run_test())


def test_give_item_fallback_respects_pushes_back_refusal(
    uow_factory,
    session_factory,
):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="<@2> pushes it back and refuses to take the Rusty Key.",
                player_state_update={"inventory": []},
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        compat.get_or_create_actor("1")
        compat.get_or_create_actor("2")
        campaign = compat.get_or_create_campaign("default", "main", "1")
        source = compat.get_or_create_player(campaign.id, "1")
        compat.get_or_create_player(campaign.id, "2")
        with session_factory() as session:
            source_row = session.get(Player, source.id)
            source_row.state_json = compat._dump_json(
                {"inventory": [{"name": "Rusty Key", "origin": "Found in locker"}]}
            )
            session.commit()

        narration = await compat.play_action(
            campaign_id=campaign.id,
            actor_id="1",
            action="I hand the Rusty Key to <@2>",
        )
        assert narration is not None

        dst_player = compat.get_or_create_player(campaign.id, "2")
        dst_names = [
            entry["name"]
            for entry in compat._get_inventory_rich(compat.get_player_state(dst_player))
        ]
        assert "Rusty Key" not in dst_names

    asyncio.run(run_test())


def test_narration_footer_is_stripped_before_persist(
    uow_factory,
    session_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="A hidden panel slides open.\n---\nXP Awarded: 3\nState Update: {}",
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)
        compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

        narration = await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="search wall",
        )
        assert narration is not None
        assert "XP Awarded" not in narration
        assert narration.startswith("A hidden panel slides open.")

    asyncio.run(run_test())


def test_speed_multiplier_scales_timer_delay_and_rendered_line(
    uow_factory,
    session_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="The lights dim ominously.",
                timer_instruction=TimerInstruction(delay_seconds=120, event_text="The vault seals"),
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
        compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])
        session_row = compat.get_or_create_session(
            campaign_id=campaign.id,
            surface="discord_channel",
            surface_key="discord:default:chan-speed",
            surface_channel_id="chan-speed",
        )
        assert compat.set_speed_multiplier(campaign, 2.0) is True
        assert compat.get_speed_multiplier(campaign) == 2.0

        narration = await compat.play_action(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            action="wait",
            session_id=session_row.id,
        )
        assert narration is not None
        pending = compat._pending_timers.get(campaign.id)
        assert pending is not None
        assert int(pending.get("delay", 0)) == 60

        timer_match = re.search(r"<t:(\d+):R>", narration)
        assert timer_match is not None
        expiry_ts = int(timer_match.group(1))
        delta = expiry_ts - int(time.time())
        assert 50 <= delta <= 65

        compat.cancel_pending_timer(campaign.id)

    asyncio.run(run_test())


# ---------------------------------------------------------------------------
# SMS "text back" syntax regression tests
# ---------------------------------------------------------------------------


def test_extract_inline_sms_intent_text_back_colon(session_factory):
    """'text back to <recipient>: <message>' should parse via Pattern 1."""
    compat = _build_compat(session_factory)

    parsed = compat._extract_inline_sms_intent("text back to elizabeth: I'll be there soon")
    assert parsed is not None
    recipient, message = parsed
    assert recipient == "elizabeth"
    assert message == "I'll be there soon"


def test_extract_inline_sms_intent_text_back_colon_no_to(session_factory):
    """'text back elizabeth: <message>' (no 'to') should also parse."""
    compat = _build_compat(session_factory)

    parsed = compat._extract_inline_sms_intent("text back elizabeth: on my way")
    assert parsed is not None
    recipient, message = parsed
    assert recipient == "elizabeth"
    assert message == "on my way"


def test_extract_inline_sms_intent_sms_back_colon(session_factory):
    """'sms back to <recipient>: <message>' variant."""
    compat = _build_compat(session_factory)

    parsed = compat._extract_inline_sms_intent("sms back to saul: Meet at dock 9")
    assert parsed is not None
    recipient, message = parsed
    assert recipient == "saul"
    assert message == "Meet at dock 9"


def test_extract_inline_sms_intent_i_text_back(session_factory):
    """'i text back to <recipient>: <message>' variant."""
    compat = _build_compat(session_factory)

    parsed = compat._extract_inline_sms_intent("i text back to Doc: thanks for the tip")
    assert parsed is not None
    recipient, message = parsed
    assert recipient == "Doc"
    assert message == "thanks for the tip"


def test_extract_inline_sms_intent_original_patterns_still_work(session_factory):
    """Verify the original patterns are unbroken by the 'back' addition."""
    compat = _build_compat(session_factory)

    # Pattern 1: colon-delimited without "back"
    p1 = compat._extract_inline_sms_intent("text elizabeth: hello there")
    assert p1 is not None
    assert p1[0] == "elizabeth"
    assert p1[1] == "hello there"

    # Pattern 2: space-delimited
    p2 = compat._extract_inline_sms_intent("text Doc hello")
    assert p2 is not None
    assert p2[0] == "Doc"
    assert p2[1] == "hello"


# ---------------------------------------------------------------------------
# Character field immutability tests
# ---------------------------------------------------------------------------


def test_apply_character_updates_immutable_fields_preserved(session_factory):
    """Foundational fields on existing characters must not be overwritten."""
    compat = _build_compat(session_factory)

    existing = {
        "saul": {
            "name": "Saul Goodman",
            "personality": "Charming, fast-talking",
            "background": "Former lawyer",
            "appearance": "Slicked-back hair, flashy suit",
            "speech_style": "Rapid-fire legalese",
            "location": "Strip mall office",
            "current_status": "scheming",
        },
    }

    updates = {
        "saul": {
            "name": "James McGill",  # immutable — should be ignored
            "personality": "Reformed",  # immutable — should be ignored
            "background": "New backstory",  # immutable — should be ignored
            "appearance": "Casual clothes",  # immutable — should be ignored
            "speech_style": "Soft-spoken",  # immutable — should be ignored
            "location": "courthouse",  # mutable — should update
            "current_status": "on trial",  # mutable — should update
            "evolving_personality": "Worn down by guilt",  # mutable new field
        },
    }

    result = compat._apply_character_updates(existing, updates)

    # Immutable fields unchanged
    assert result["saul"]["name"] == "Saul Goodman"
    assert result["saul"]["personality"] == "Charming, fast-talking"
    assert result["saul"]["background"] == "Former lawyer"
    assert result["saul"]["appearance"] == "Slicked-back hair, flashy suit"
    assert result["saul"]["speech_style"] == "Rapid-fire legalese"

    # Mutable fields updated
    assert result["saul"]["location"] == "courthouse"
    assert result["saul"]["current_status"] == "on trial"
    assert result["saul"]["evolving_personality"] == "Worn down by guilt"


def test_apply_character_updates_new_char_gets_all_fields(session_factory):
    """New characters should receive all fields including foundational ones."""
    compat = _build_compat(session_factory)

    existing = {}
    updates = {
        "mira": {
            "name": "Mira",
            "personality": "Quiet observer",
            "background": "Unknown origins",
            "appearance": "Dark cloak",
            "speech_style": "Whispered fragments",
            "location": "alley",
        },
    }

    result = compat._apply_character_updates(existing, updates)

    assert result["mira"]["name"] == "Mira"
    assert result["mira"]["personality"] == "Quiet observer"
    assert result["mira"]["background"] == "Unknown origins"
    assert result["mira"]["appearance"] == "Dark cloak"
    assert result["mira"]["speech_style"] == "Whispered fragments"
    assert result["mira"]["location"] == "alley"


# ---------------------------------------------------------------------------
# tool_calls allowlist filtering and execution tests
# ---------------------------------------------------------------------------


def test_payload_to_output_filters_tool_calls_to_allowlist(session_factory, seed_campaign_and_actor):
    """Only sms_write and sms_schedule should survive the allowlist filter."""
    completion = StubCompletionPort()
    tool_llm = ToolAwareZorkLLM(
        session_factory=session_factory,
        completion_port=completion,
        temperature=0.8,
        max_tokens=2048,
    )
    compat = _build_compat(session_factory)
    tool_llm.bind_emulator(compat)

    payload = {
        "narration": "Test narration",
        "state_update": {},
        "tool_calls": [
            {"tool_call": "sms_write", "thread": "saul", "from": "Saul", "to": "Dale", "message": "On my way."},
            {"tool_call": "sms_schedule", "thread": "saul", "from": "Saul", "to": "Dale", "message": "Later.", "delay_seconds": 60},
            {"tool_call": "memory_search", "queries": ["danger"]},  # NOT in allowlist
            {"tool_call": "plot_plan"},  # NOT in allowlist
        ],
    }

    output = tool_llm._payload_to_output(payload, actor_id=seed_campaign_and_actor["actor_id"])
    assert len(output.tool_calls) == 2
    assert output.tool_calls[0]["tool_call"] == "sms_write"
    assert output.tool_calls[1]["tool_call"] == "sms_schedule"


def test_tool_calls_sms_write_executed_in_complete_turn(session_factory, seed_campaign_and_actor):
    """tool_calls with sms_write should actually persist the message."""

    class SmsToolCallCompletionPort:
        async def complete(self, system_prompt, prompt, *, temperature=0.8, max_tokens=2048):
            return json.dumps({
                "narration": "Saul fires off a quick text.",
                "state_update": {
                    "game_time": {"day": 1, "hour": 10, "minute": 0, "day_of_week": "monday", "period": "morning", "date_label": "Monday, Day 1, Morning"},
                },
                "summary_update": "Saul texted Dale.",
                "tool_calls": [
                    {
                        "tool_call": "sms_write",
                        "thread": "dale",
                        "from": "Saul",
                        "to": "Dale",
                        "message": "Meet me at Dock 9.",
                    },
                ],
            })

    completion = SmsToolCallCompletionPort()
    tool_llm = ToolAwareZorkLLM(
        session_factory=session_factory,
        completion_port=completion,
        temperature=0.8,
        max_tokens=2048,
    )
    compat = _build_compat(session_factory, completion_port=completion)
    tool_llm.bind_emulator(compat)

    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    # Ensure a Player row exists — complete_turn queries Player by campaign+actor.
    compat.get_or_create_player(campaign.id, seed_campaign_and_actor["actor_id"])

    async def run_test():
        # Duck-typed context — complete_turn only reads campaign_id,
        # actor_id, action, and session_id from the context object.
        class _Ctx:
            def __init__(self, campaign_id, actor_id, action):
                self.campaign_id = campaign_id
                self.actor_id = actor_id
                self.action = action
                self.session_id = None

        ctx = _Ctx(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            action="text Dale",
        )
        output = await tool_llm.complete_turn(ctx)
        assert output.narration == "Saul fires off a quick text."
        assert len(output.tool_calls) == 1

        # Verify the SMS was actually persisted
        _key, _label, messages = compat.read_sms_thread(campaign.id, "dale", limit=10)
        assert len(messages) >= 1
        assert any(m["from"] == "Saul" and "Dock 9" in m["message"] for m in messages)

    asyncio.run(run_test())


# ---------------------------------------------------------------------------
# SMS empty-thread filtering tests
# ---------------------------------------------------------------------------


def test_sms_list_excludes_threads_with_zero_messages(session_factory):
    """Threads that have 0 messages (or only empty-text messages) must not
    appear in the sms_list output."""
    compat = _build_compat(session_factory)

    # Simulate campaign state with a mixture of real and ghost threads.
    campaign_state = {
        compat.SMS_STATE_KEY: {
            "harry": {"label": "Harry", "messages": []},  # empty — should be excluded
            "dawn": {
                "label": "Dawn",
                "messages": [{"message": "", "from": "Dawn", "to": "You"}],  # blank text
            },
            "saul": {
                "label": "Saul",
                "messages": [
                    {
                        "from": "Saul",
                        "to": "Dale",
                        "message": "Meet at Dock 9.",
                        "day": 1,
                        "hour": 10,
                        "minute": 0,
                        "turn_id": 1,
                        "seq": 1,
                    }
                ],
            },
        }
    }

    threads = compat._sms_threads_from_state(campaign_state)
    assert "harry" not in threads, "Empty-messages thread should be excluded from _sms_threads_from_state"
    assert "dawn" not in threads, "Blank-text-only thread should be excluded from _sms_threads_from_state"
    assert "saul" in threads, "Thread with real messages should be present"

    listed = compat._sms_list_threads(campaign_state, wildcard="*", limit=20)
    thread_keys = [t["thread"] for t in listed]
    assert "harry" not in thread_keys, "Empty thread should not appear in sms_list"
    assert "dawn" not in thread_keys, "Blank-text thread should not appear in sms_list"
    assert "saul" in thread_keys, "Thread with messages should appear in sms_list"
    assert listed[0]["count"] == 1
    assert "Dock 9" in listed[0]["last_preview"]
