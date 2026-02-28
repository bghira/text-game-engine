from __future__ import annotations

import asyncio

from text_game_engine.core.engine import GameEngine
from text_game_engine.core.types import LLMTurnOutput
from text_game_engine.persistence.sqlalchemy import (
    SQLAlchemyUnitOfWork,
    build_engine,
    build_session_factory,
    create_schema,
)
from text_game_engine.zork_emulator import ZorkEmulator


class StoryLLM:
    async def complete_turn(self, context):
        action = context.action.strip().lower()
        if "door" in action:
            return LLMTurnOutput(
                narration="The bronze door groans open, revealing a lantern-lit stair.",
                player_state_update={
                    "room_title": "Service Stairwell",
                    "room_summary": "A narrow stairwell descending into the keep.",
                    "location": "service stairwell",
                    "exits": ["downstairs", "back to hallway"],
                },
                state_update={"last_discovery": "service stairwell"},
                summary_update="A hidden stairwell was discovered behind the bronze door.",
                xp_awarded=3,
                scene_image_prompt=(
                    "A lantern-lit medieval stone stairwell behind an opened bronze door, "
                    "dust in the air, narrow steps descending into shadow."
                ),
            )
        return LLMTurnOutput(
            narration="You stand in a marble hallway lined with bronze reliefs.",
            player_state_update={
                "room_title": "Marble Hallway",
                "room_summary": "A cold hall with a sealed bronze door.",
                "location": "marble hallway",
                "exits": ["inspect bronze door", "return to courtyard"],
            },
            state_update={"current_location": "marble hallway"},
            summary_update="The party reached a hallway with a sealed bronze door.",
            xp_awarded=1,
        )


def build_runtime():
    engine = build_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    session_factory = build_session_factory(engine)

    def uow_factory():
        return SQLAlchemyUnitOfWork(session_factory)

    game_engine = GameEngine(uow_factory=uow_factory, llm=StoryLLM())
    emulator = ZorkEmulator(game_engine=game_engine, session_factory=session_factory)
    return emulator


async def main() -> None:
    emulator = build_runtime()

    actor = emulator.get_or_create_actor("actor-1", display_name="Player One")
    campaign = emulator.get_or_create_campaign(
        namespace="demo",
        name="castle-run",
        created_by_actor_id=actor.id,
    )
    emulator.get_or_create_player(campaign.id, actor.id)

    first = await emulator.play_action(
        campaign_id=campaign.id,
        actor_id=actor.id,
        action="look",
    )
    second = await emulator.play_action(
        campaign_id=campaign.id,
        actor_id=actor.id,
        action="open the bronze door",
    )

    print("Turn 1:")
    print(first)
    print()
    print("Turn 2:")
    print(second)


if __name__ == "__main__":
    asyncio.run(main())
