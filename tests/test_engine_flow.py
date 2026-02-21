from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from sqlalchemy import select

from text_game_engine.core.engine import GameEngine
from text_game_engine.core.types import GiveItemInstruction, LLMTurnOutput, ResolveTurnInput, TimerInstruction
from text_game_engine.persistence.sqlalchemy.models import Actor, Campaign, OutboxEvent, Snapshot, Timer, Turn


class StubLLM:
    def __init__(self, output: LLMTurnOutput):
        self.output = output

    async def complete_turn(self, context):
        return self.output


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
                c.updated_at = datetime.utcnow()
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
                c.updated_at = datetime.utcnow()
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
    now = datetime.utcnow()
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
        assert uow.timers.mark_expired(timer_id, datetime.utcnow()) is True
        assert uow.timers.mark_expired(timer_id, datetime.utcnow()) is False
        assert uow.timers.mark_consumed(timer_id, datetime.utcnow()) is True
        assert uow.timers.mark_consumed(timer_id, datetime.utcnow()) is False
        uow.commit()


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
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(other_campaign)
            session.add(
                Actor(
                    id="actor-2",
                    display_name="Other",
                    kind="human",
                    metadata_json="{}",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
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
