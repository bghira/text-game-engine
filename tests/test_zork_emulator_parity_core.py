from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import json
import re
import time

from text_game_engine.core.types import GiveItemInstruction, LLMTurnOutput, TimerInstruction
from text_game_engine.core.engine import GameEngine
from text_game_engine.persistence.sqlalchemy.uow import SQLAlchemyUnitOfWork
from text_game_engine.persistence.sqlalchemy.models import Campaign, Player, Snapshot, Turn
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


def test_build_prompt_shape(session_factory, seed_campaign_and_actor):
    compat = _build_compat(session_factory)
    campaign = compat.get_or_create_campaign("default", "main", seed_campaign_and_actor["actor_id"])
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
    turns = compat.get_recent_turns(seed_campaign_and_actor["campaign_id"])

    system_prompt, user_prompt = compat.build_prompt(campaign, player, "look", turns)
    assert "You are the ZorkEmulator" in system_prompt
    assert "STRUCTURE REQUIREMENT:" in system_prompt
    assert "USE memory_search AGGRESSIVELY" in system_prompt
    assert "CALENDAR & GAME TIME SYSTEM:" in system_prompt
    assert "CRITICAL — calendar_update.remove rules:" in system_prompt
    assert "Do NOT remove events just because they are overdue." in system_prompt
    assert "stores fire_day" in system_prompt
    assert "CALENDAR_REMINDERS" in user_prompt
    assert "CHARACTER ROSTER & PORTRAITS:" in system_prompt
    assert "CAMPAIGN:" in user_prompt
    assert "CURRENT_GAME_TIME:" in user_prompt
    assert "SPEED_MULTIPLIER:" in user_prompt
    assert "CALENDAR:" in user_prompt
    assert "PLAYER_ACTION:" in user_prompt


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
    assert game_time.get("period") == "morning"
    assert game_time.get("date_label") == "Day 1, Morning"
    assert '"day": 1' in user_prompt
    assert "CURRENT_GAME_TIME:" in user_prompt


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

        variants_msg = await compat.handle_setup_message(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            message_text="yes",
            attachments=[StubAttachment("lore.txt", b"Neo wakes up in a false city.\n\nAgents hunt him.")],
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

        variants_msg = await compat.handle_setup_message(
            campaign_id=campaign.id,
            actor_id=seed_campaign_and_actor["actor_id"],
            message_text="no, i'd rather do a novel thing where the moon is a prison colony",
        )
        assert "Choose a storyline variant" in variants_msg
        assert probe.reclassify_calls == 0
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
            campaign_state["game_time"] = {"day": 3, "period": "evening"}
            campaign_state["calendar"] = [
                {
                    "name": "Moonrise Ceremony",
                    "fire_day": 5,
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

        variants = await compat.handle_setup_message(
            ctx,
            "yes",
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
                    created_at=datetime.utcnow(),
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
