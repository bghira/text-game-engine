from __future__ import annotations

import asyncio

from sqlalchemy import select

from text_game_engine.core.engine import GameEngine
from text_game_engine.core.types import LLMTurnOutput, ResolveTurnInput, TimerInstruction
from text_game_engine.persistence.sqlalchemy.models import Snapshot, Turn, Timer
from text_game_engine.zork_emulator import ZorkEmulator


class StubLLM:
    def __init__(self, output: LLMTurnOutput):
        self.output = output

    async def complete_turn(self, context):
        return self.output


def test_zork_compat_play_action_returns_narration(uow_factory, session_factory, seed_campaign_and_actor):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="Compat narration"))
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

        out = await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="look",
            manage_claim=True,
        )
        assert out == "Compat narration"

    asyncio.run(run_test())


def test_zork_compat_begin_end_claim(uow_factory, session_factory, seed_campaign_and_actor):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="noop"))
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        campaign_id, err = await compat.begin_turn(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
        assert campaign_id == seed_campaign_and_actor["campaign_id"]
        assert err is None

        # duplicate begin while claimed -> old behavior shape returns (None, None)
        campaign_id2, err2 = await compat.begin_turn(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
        assert campaign_id2 is None
        assert err2 is None

        compat.end_turn(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
        campaign_id3, err3 = await compat.begin_turn(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])
        assert campaign_id3 == seed_campaign_and_actor["campaign_id"]
        assert err3 is None

    asyncio.run(run_test())


def test_zork_compat_register_timer_message(uow_factory, session_factory, seed_campaign_and_actor):
    async def run_test():
        llm = StubLLM(
            LLMTurnOutput(
                narration="timer scene",
                timer_instruction=TimerInstruction(delay_seconds=60, event_text="Event"),
            )
        )
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

        result = await engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=seed_campaign_and_actor["campaign_id"],
                actor_id=seed_campaign_and_actor["actor_id"],
                action="wait",
            )
        )
        assert result.status == "ok"

        ok = compat.register_timer_message(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            message_id="12345",
            channel_id="chan-a",
        )
        assert ok is True

        with session_factory() as session:
            timer = session.execute(select(Timer).where(Timer.campaign_id == seed_campaign_and_actor["campaign_id"])) .scalars().first()
            assert timer is not None
            assert timer.status == "scheduled_bound"
            assert timer.external_message_id == "12345"
            assert timer.external_channel_id == "chan-a"

    asyncio.run(run_test())


def test_zork_compat_returns_live_objects_post_commit(uow_factory, session_factory, seed_campaign_and_actor):
    llm = StubLLM(LLMTurnOutput(narration="noop"))
    engine = GameEngine(uow_factory=uow_factory, llm=llm)
    compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

    campaign = compat.get_or_create_campaign(
        namespace="default",
        name="main",
        created_by_actor_id=seed_campaign_and_actor["actor_id"],
    )
    player = compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

    assert campaign.name == "main"
    assert player.level == 1
    assert player.xp == 0


def test_zork_compat_record_turn_message_ids_and_rewind_by_message_id(
    uow_factory, session_factory, seed_campaign_and_actor
):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="Compat narration"))
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

        await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="turn 1",
        )
        compat.record_turn_message_ids(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            user_message_id="u-1",
            bot_message_id="b-1",
        )

        await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="turn 2",
        )
        compat.record_turn_message_ids(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            user_message_id="u-2",
            bot_message_id="b-2",
        )

        rewind_result = compat.execute_rewind(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            target_discord_message_id="b-1",
        )
        assert rewind_result is not None
        rewind_turn_id, deleted_count = rewind_result
        assert rewind_turn_id == 2
        assert deleted_count == 2

        with session_factory() as session:
            turns = (
                session.execute(
                    select(Turn)
                    .where(Turn.campaign_id == seed_campaign_and_actor["campaign_id"])
                    .order_by(Turn.id.asc())
                )
                .scalars()
                .all()
            )
            assert [t.id for t in turns] == [1, 2]

    asyncio.run(run_test())


def test_zork_compat_rewind_fallback_uses_user_message_id(
    uow_factory, session_factory, seed_campaign_and_actor
):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="Compat narration"))
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

        await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="turn 1",
        )
        compat.record_turn_message_ids(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            user_message_id="u-1",
            bot_message_id="b-1",
        )

        await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="turn 2",
        )
        compat.record_turn_message_ids(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            user_message_id="u-2",
            bot_message_id="b-2",
        )

        with session_factory() as session:
            first_narrator = (
                session.execute(
                    select(Turn)
                    .where(Turn.campaign_id == seed_campaign_and_actor["campaign_id"])
                    .where(Turn.kind == "narrator")
                    .where(Turn.external_user_message_id == "u-1")
                    .order_by(Turn.id.asc())
                )
                .scalars()
                .first()
            )
            assert first_narrator is not None
            first_narrator.external_message_id = None
            session.commit()

        rewind_result = compat.execute_rewind(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            target_discord_message_id="u-1",
        )
        assert rewind_result is not None
        rewind_turn_id, deleted_count = rewind_result
        assert rewind_turn_id == 2
        assert deleted_count == 2

    asyncio.run(run_test())


def test_zork_compat_rewind_channel_scope_only_deletes_that_surface(
    uow_factory, session_factory, seed_campaign_and_actor
):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="Compat narration"))
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        compat.get_or_create_player(seed_campaign_and_actor["campaign_id"], seed_campaign_and_actor["actor_id"])

        session_a = compat.get_or_create_session(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            surface="discord",
            surface_key="discord:100",
            surface_channel_id="100",
        )
        session_b = compat.get_or_create_session(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            surface="discord",
            surface_key="discord:200",
            surface_channel_id="200",
        )

        await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="a1",
            session_id=session_a.id,
        )
        compat.record_turn_message_ids(seed_campaign_and_actor["campaign_id"], "u-1", "b-1")

        await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="b1",
            session_id=session_b.id,
        )
        compat.record_turn_message_ids(seed_campaign_and_actor["campaign_id"], "u-2", "b-2")

        await compat.play_action(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            actor_id=seed_campaign_and_actor["actor_id"],
            action="a2",
            session_id=session_a.id,
        )
        compat.record_turn_message_ids(seed_campaign_and_actor["campaign_id"], "u-3", "b-3")

        rewind_result = compat.execute_rewind(
            campaign_id=seed_campaign_and_actor["campaign_id"],
            target_discord_message_id="b-1",
            channel_id="100",
        )
        assert rewind_result is not None
        rewind_turn_id, deleted_count = rewind_result
        assert rewind_turn_id == 2
        assert deleted_count == 2

        with session_factory() as session:
            turns = (
                session.execute(
                    select(Turn)
                    .where(Turn.campaign_id == seed_campaign_and_actor["campaign_id"])
                    .order_by(Turn.id.asc())
                )
                .scalars()
                .all()
            )
            assert [t.id for t in turns] == [1, 2, 3, 4]
            assert [t.session_id for t in turns] == [session_a.id, session_a.id, session_b.id, session_b.id]

            snapshots = (
                session.execute(
                    select(Snapshot)
                    .where(Snapshot.campaign_id == seed_campaign_and_actor["campaign_id"])
                    .order_by(Snapshot.turn_id.asc())
                )
                .scalars()
                .all()
            )
            assert [s.turn_id for s in snapshots] == [2, 4]

    asyncio.run(run_test())


def test_zork_compat_begin_turn_missing_campaign_returns_error(uow_factory, session_factory):
    async def run_test():
        llm = StubLLM(LLMTurnOutput(narration="noop"))
        engine = GameEngine(uow_factory=uow_factory, llm=llm)
        compat = ZorkEmulator(game_engine=engine, session_factory=session_factory)

        campaign_id, err = await compat.begin_turn("missing-campaign", "actor-1")
        assert campaign_id is None
        assert err == "Campaign not found."

    asyncio.run(run_test())
