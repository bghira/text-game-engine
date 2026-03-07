from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json

from sqlalchemy import select

from text_game_engine.core.engine import GameEngine
from text_game_engine.core.types import GiveItemInstruction, LLMTurnOutput, ResolveTurnInput, TimerInstruction
from text_game_engine.persistence.sqlalchemy.models import Actor, Campaign, OutboxEvent, Snapshot, Timer, Turn


class StubLLM:
    def __init__(self, output: LLMTurnOutput):
        self.output = output

    async def complete_turn(self, context):
        return self.output


class QueueLLM:
    def __init__(self, outputs: list[LLMTurnOutput]):
        self.outputs = list(outputs)
        self.contexts = []

    async def complete_turn(self, context):
        self.contexts.append(context)
        if not self.outputs:
            raise AssertionError("No queued LLM outputs left.")
        return self.outputs.pop(0)


def test_phase_c_cas_conflict_rolls_back_all_writes(
    session_factory,
    uow_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="A scene happens.",
                state_update={"k": "v"},
                scene_image_prompt="describe scene",
                timer_instruction=TimerInstruction(delay_seconds=60, event_text="Boom"),
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm, max_conflict_retries=0)

        async def bump_version(_context, _attempt):
            with uow_factory() as uow:
                c = uow.campaigns.get(seed_campaign_and_actor["campaign_id"])
                c.row_version += 1
                c.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                uow.commit()

        result = await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="look",
            ),
            before_phase_c=bump_version,
        )
        assert result.status == "conflict"

        with session_factory() as session:
            assert session.execute(select(Turn)).scalars().all() == []
            assert session.execute(select(Snapshot)).scalars().all() == []
            assert session.execute(select(Timer)).scalars().all() == []
            assert session.execute(select(OutboxEvent)).scalars().all() == []

    asyncio.run(run_test())


def test_single_auto_retry_then_conflict_response(
    session_factory,
    uow_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="retry me"))
        engine = GameEngine(uow_factory=uow_factory, llm=llm, max_conflict_retries=1)
        calls = {"n": 0}

        async def always_bump(_context, _attempt):
            calls["n"] += 1
            with uow_factory() as uow:
                c = uow.campaigns.get(seed_campaign_and_actor["campaign_id"])
                c.row_version += 1
                c.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                uow.commit()

        result = await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="look",
            ),
            before_phase_c=always_bump,
        )
        assert calls["n"] == 2
        assert result.status == "conflict"

    asyncio.run(run_test())


def test_timer_transition_idempotency(uow_factory, seed_campaign_and_actor):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with uow_factory() as uow:
        timer = uow.timers.schedule(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            session_id=None,
            due_at=now + timedelta(seconds=60),
            event_text="Explosion",
            interruptible=True,
            interrupt_action=None,
        )
        uow.commit()
        timer_id = timer.id

    with uow_factory() as uow:
        assert uow.timers.attach_message(timer_id, "msg-1", "chan-1", None) is True
        assert uow.timers.attach_message(timer_id, "msg-2", "chan-1", None) is True
        assert uow.timers.mark_expired(timer_id, datetime.now(timezone.utc).replace(tzinfo=None)) is True
        assert uow.timers.mark_expired(timer_id, datetime.now(timezone.utc).replace(tzinfo=None)) is False
        assert uow.timers.mark_consumed(timer_id, datetime.now(timezone.utc).replace(tzinfo=None)) is True
        assert uow.timers.mark_consumed(timer_id, datetime.now(timezone.utc).replace(tzinfo=None)) is False
        uow.commit()


def test_timer_outbox_includes_interrupt_scope(session_factory, uow_factory, seed_campaign_and_actor):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="A siren ramps up.",
                timer_instruction=TimerInstruction(
                    delay_seconds=90,
                    event_text="Blast doors lock.",
                    interruptible=True,
                    interrupt_scope="local",
                ),
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)

        result = await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="wait",
            )
        )
        assert result.status == "ok"

        with session_factory() as session:
            outbox = (
                session.query(OutboxEvent)
                .filter(OutboxEvent.campaign_id == seed_campaign_and_actor["campaign_id"])
                .filter(OutboxEvent.event_type == "timer_scheduled")
                .order_by(OutboxEvent.id.desc())
                .first()
            )
            assert outbox is not None
            payload = json.loads(outbox.payload_json or "{}")
            assert payload.get("interrupt_scope") == "local"

    asyncio.run(run_test())


def test_recent_turns_visibility_filters_private_and_limited(
    session_factory,
    uow_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        with session_factory() as session:
            session.add_all(
                [
                    Actor(id="actor-2", display_name="Tester Two", kind="human", metadata_json="{}"),
                    Actor(id="actor-3", display_name="Tester Three", kind="human", metadata_json="{}"),
                ]
            )
            session.commit()

        llm = QueueLLM(
            [
                LLMTurnOutput(
                    narration="A private exchange.",
                    turn_visibility={"scope": "private"},
                ),
                LLMTurnOutput(
                    narration="A limited aside.",
                    turn_visibility={
                        "scope": "limited",
                        "player_slugs": ["player-actor-2"],
                    },
                ),
                LLMTurnOutput(narration="Observer turn."),
                LLMTurnOutput(narration="Targeted observer turn."),
                LLMTurnOutput(narration="Excluded observer turn."),
            ]
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)

        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="whisper to myself",
            )
        )
        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="wave actor two over",
            )
        )
        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id="actor-2",
                action="look around",
            )
        )
        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id="actor-3",
                action="look around",
            )
        )
        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="recap quietly",
            )
        )

        actor2_context = llm.contexts[2]
        actor3_context = llm.contexts[3]
        actor1_context = llm.contexts[4]

        actor2_seen = [row["content"] for row in actor2_context.recent_turns]
        actor3_seen = [row["content"] for row in actor3_context.recent_turns]
        actor1_seen = [row["content"] for row in actor1_context.recent_turns]

        assert "A private exchange." not in actor2_seen
        assert "whisper to myself" not in actor2_seen
        assert "A limited aside." in actor2_seen
        assert "wave actor two over" in actor2_seen

        assert "A private exchange." not in actor3_seen
        assert "A limited aside." not in actor3_seen
        assert "whisper to myself" not in actor3_seen
        assert "wave actor two over" not in actor3_seen

        assert "A private exchange." in actor1_seen
        assert "A limited aside." in actor1_seen
        assert "whisper to myself" in actor1_seen
        assert "wave actor two over" in actor1_seen

    asyncio.run(run_test())


def test_summary_update_only_persists_for_public_turns(
    session_factory,
    uow_factory,
    seed_campaign_and_actor,
):
    async def run_test():
        llm = QueueLLM(
            [
                LLMTurnOutput(
                    narration="A private note.",
                    summary_update="Private summary should not persist.",
                    turn_visibility={"scope": "private"},
                ),
                LLMTurnOutput(
                    narration="A limited note.",
                    summary_update="Limited summary should not persist.",
                    turn_visibility={
                        "scope": "limited",
                        "player_slugs": ["player-actor-2"],
                    },
                ),
                LLMTurnOutput(
                    narration="A public note.",
                    summary_update="Public summary should persist.",
                ),
            ]
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)

        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="keep this private",
            )
        )
        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="keep this limited",
            )
        )
        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="say it out loud",
            )
        )

        with session_factory() as session:
            campaign = session.get(Campaign, seed_campaign_and_actor["campaign_id"])
            assert campaign is not None
            summary = campaign.summary or ""
            assert "Private summary should not persist." not in summary
            assert "Limited summary should not persist." not in summary
            assert "Public summary should persist." in summary

    asyncio.run(run_test())


def test_memory_visibility_filter_after_rewind(session_factory, uow_factory, seed_campaign_and_actor):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="Turn narration"))
        engine = GameEngine(uow_factory=uow_factory, llm=llm)

        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="go north",
            )
        )
        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="go south",
            )
        )

        with session_factory() as session:
            turns = session.execute(select(Turn).order_by(Turn.id.asc())).scalars().all()
            # player,narrator,player,narrator
            target_turn_id = turns[1].id

        rewind_result = engine.rewind_to_turn(seed_campaign_and_actor["campaign_id"], target_turn_id)
        assert rewind_result.status == "ok"

        with session_factory() as session:
            campaign = session.get(Campaign, seed_campaign_and_actor["campaign_id"])
            assert campaign.memory_visible_max_turn_id == target_turn_id
            remaining_turns = session.execute(select(Turn).order_by(Turn.id.asc())).scalars().all()
            assert len(remaining_turns) == 2
            assert all(t.id <= target_turn_id for t in remaining_turns)

        filtered = engine.filter_memory_hits_by_visibility(
            seed_campaign_and_actor["campaign_id"],
            [
                {"turn_id": target_turn_id - 1, "content": "older"},
                {"turn_id": target_turn_id + 10, "content": "future"},
            ],
        )
        assert filtered == [{"turn_id": target_turn_id - 1, "content": "older"}]

    asyncio.run(run_test())


def test_give_item_unresolved_nonfatal_compat(session_factory, uow_factory, seed_campaign_and_actor):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="You try to hand it over.",
                give_item=GiveItemInstruction(item="rusty key", to_discord_mention="<@999999>"),
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm, actor_resolver=None)

        result = await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="give key",
            )
        )
        assert result.status == "ok"

        with session_factory() as session:
            events = session.execute(select(OutboxEvent).where(OutboxEvent.event_type == "give_item_unresolved")).scalars().all()
            assert len(events) == 1

    asyncio.run(run_test())


def test_engine_fallback_narration_uses_state_updates(session_factory, uow_factory, seed_campaign_and_actor):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="",
                player_state_update={"room_summary": "Hotel room with ocean view."},
                state_update={"victorville": {"visitation_timer": "26_minutes_remaining"}},
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)

        result = await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="look",
            )
        )
        assert result.status == "ok"
        assert result.narration == "Hotel room with ocean view."

    asyncio.run(run_test())


def test_calendar_update_ops_are_applied_and_not_persisted_as_patch_key(
    session_factory, uow_factory, seed_campaign_and_actor
):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="Calendar updated.",
                state_update={
                    "calendar_update": {
                        "add": [
                            {
                                "name": "Eclipse",
                                "time_remaining": 3,
                                "time_unit": "days",
                                "description": "A shadow crosses the city",
                            }
                        ]
                    }
                },
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)

        result = await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="check horizon",
            )
        )
        assert result.status == "ok"

        with session_factory() as session:
            campaign = session.get(Campaign, seed_campaign_and_actor["campaign_id"])
            assert campaign is not None
            state_text = campaign.state_json
            assert "\"calendar_update\"" not in state_text
            assert "\"calendar\"" in state_text
            assert "Eclipse" in state_text
            state = json.loads(state_text or "{}")
            calendar = state.get("calendar", [])
            assert isinstance(calendar, list) and calendar
            eclipse = next((entry for entry in calendar if entry.get("name") == "Eclipse"), None)
            assert eclipse is not None
            assert eclipse.get("fire_day") == 4
            assert eclipse.get("fire_hour") == 8

    asyncio.run(run_test())


def test_story_progress_state_update_coerces_string_indices_and_clamps(
    session_factory, uow_factory, seed_campaign_and_actor
):
    async def run_test():
        with session_factory() as session:
            campaign = session.get(Campaign, seed_campaign_and_actor["campaign_id"])
            campaign.state_json = json.dumps(
                {
                    "current_chapter": 0,
                    "current_scene": 0,
                    "story_outline": {
                        "chapters": [
                            {"title": "One", "scenes": [{"title": "S1"}]},
                            {"title": "Two", "scenes": [{"title": "S2-1"}, {"title": "S2-2"}]},
                        ]
                    },
                }
            )
            session.commit()

        llm = StubLLM(
            LLMTurnOutput(
                narration="Advance chapter and scene.",
                state_update={"current_chapter": "1", "current_scene": "99"},
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)

        result = await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="continue",
            )
        )
        assert result.status == "ok"

        with session_factory() as session:
            campaign = session.get(Campaign, seed_campaign_and_actor["campaign_id"])
            state = json.loads(campaign.state_json or "{}")
            assert state.get("current_chapter") == 1
            # chapter 2 has two scenes -> max valid index is 1
            assert state.get("current_scene") == 1

    asyncio.run(run_test())


def test_character_updates_null_removes_character(session_factory, uow_factory, seed_campaign_and_actor):
    async def run_test():
        with session_factory() as session:
            campaign = session.get(Campaign, seed_campaign_and_actor["campaign_id"])
            campaign.characters_json = json.dumps(
                {
                    "mira-guide": {"name": "Mira", "location": "Dock 9"},
                    "jet-smuggler": {"name": "Jet", "location": "Market"},
                }
            )
            session.commit()

        llm = StubLLM(
            LLMTurnOutput(
                narration="Mira departs the story.",
                character_updates={"mira-guide": None},
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)

        result = await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="continue",
            )
        )
        assert result.status == "ok"

        with session_factory() as session:
            campaign = session.get(Campaign, seed_campaign_and_actor["campaign_id"])
            characters = json.loads(campaign.characters_json or "{}")
            assert "mira-guide" not in characters
            assert "jet-smuggler" in characters

        with session_factory() as session:
            campaign = session.get(Campaign, seed_campaign_and_actor["campaign_id"])
            campaign.characters_json = json.dumps(
                {
                    "rhea-sage": {"name": "Rhea Sage", "location": "Tower"},
                    "jet-smuggler": {"name": "Jet", "location": "Market"},
                }
            )
            session.commit()

        llm2 = StubLLM(
            LLMTurnOutput(
                narration="Rhea exits the campaign.",
                state_update={"Rhea": None},
            )
        )
        engine2 = GameEngine(uow_factory=uow_factory, llm=llm2)

        result2 = await engine2.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="continue",
            )
        )
        assert result2.status == "ok"

        with session_factory() as session:
            campaign = session.get(Campaign, seed_campaign_and_actor["campaign_id"])
            characters = json.loads(campaign.characters_json or "{}")
            assert "rhea-sage" not in characters
            assert "jet-smuggler" in characters

    asyncio.run(run_test())


def test_rewind_requires_snapshot_from_same_campaign(session_factory, uow_factory, seed_campaign_and_actor):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="Turn narration"))
        engine = GameEngine(uow_factory=uow_factory, llm=llm)

        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="look around",
            )
        )

        with session_factory() as session:
            other_campaign = Campaign(
                id="campaign-2",
                namespace="default",
                name="side",
                name_normalized="side",
                created_by_actor_id=seed_campaign_and_actor["actor_id"],
                summary="",
                state_json="{}",
                characters_json="{}",
                row_version=1,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
            session.add(other_campaign)
            session.add(
                Actor(
                    id="actor-2",
                    display_name="Other",
                    kind="human",
                    metadata_json="{}",
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            session.commit()

        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id="campaign-2",
                actor_id="actor-2",
                action="go",
            )
        )

        with session_factory() as session:
            other_narrator_turn = (
                session.execute(
                    select(Turn)
                    .where(Turn.campaign_id == "campaign-2")
                    .where(Turn.kind == "narrator")
                    .order_by(Turn.id.desc())
                )
                .scalars()
                .first()
            )
            assert other_narrator_turn is not None

        result = engine.rewind_to_turn(seed_campaign_and_actor["campaign_id"], other_narrator_turn.id)
        assert result.status == "error"
        assert result.reason == "snapshot_not_found"

    asyncio.run(run_test())


def test_rewind_same_target_is_idempotent(session_factory, uow_factory, seed_campaign_and_actor):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="Turn narration"))
        engine = GameEngine(uow_factory=uow_factory, llm=llm)

        await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="go north",
            )
        )

        with session_factory() as session:
            target_turn = (
                session.execute(
                    select(Turn)
                    .where(Turn.campaign_id == seed_campaign_and_actor["campaign_id"])
                    .where(Turn.kind == "narrator")
                    .order_by(Turn.id.desc())
                )
                .scalars()
                .first()
            )
            assert target_turn is not None
            target_turn_id = target_turn.id

        first = engine.rewind_to_turn(seed_campaign_and_actor["campaign_id"], target_turn_id)
        second = engine.rewind_to_turn(seed_campaign_and_actor["campaign_id"], target_turn_id)
        assert first.status == "ok"
        assert second.status == "ok"

    asyncio.run(run_test())
