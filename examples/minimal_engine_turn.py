from __future__ import annotations

import asyncio
import json

from text_game_engine.core.engine import GameEngine
from text_game_engine.core.types import LLMTurnOutput, ResolveTurnInput
from text_game_engine.persistence.sqlalchemy import (
    SQLAlchemyUnitOfWork,
    build_engine,
    build_session_factory,
    create_schema,
)
from text_game_engine.persistence.sqlalchemy.models import Actor, Campaign


class DemoLLM:
    async def complete_turn(self, context):
        game_time = context.campaign_state.get("game_time", {})
        current_day = int(game_time.get("day", 1))
        return LLMTurnOutput(
            narration=f"You scout the old road. The story advances to Day {current_day + 1}.",
            state_update={
                "game_time": {
                    "day": current_day + 1,
                    "hour": 18,
                    "period": "Evening",
                    "date_label": f"Day {current_day + 1}, Evening",
                },
                "calendar_update": {
                    "add": [
                        {
                            "name": "City gate closes",
                            "time_remaining": 2,
                            "time_unit": "days",
                            "description": "After this day, entry requires forged papers.",
                        }
                    ]
                },
            },
            summary_update="The party moved closer to the city and noted an upcoming gate deadline.",
            xp_awarded=2,
            player_state_update={"location": "old road"},
        )


def make_uow_factory():
    engine = build_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    session_factory = build_session_factory(engine)

    with session_factory() as session:
        actor = Actor(id="actor-1", display_name="Player One", kind="human", metadata_json="{}")
        campaign = Campaign(
            id="campaign-1",
            namespace="demo",
            name="city-escape",
            name_normalized="city-escape",
            created_by_actor_id=actor.id,
            summary="",
            state_json=json.dumps(
                {
                    "game_time": {"day": 1, "hour": 8, "period": "Morning", "date_label": "Day 1, Morning"},
                    "calendar": [],
                }
            ),
            characters_json="{}",
            row_version=1,
        )
        session.add(actor)
        session.add(campaign)
        session.commit()

    def _uow_factory():
        return SQLAlchemyUnitOfWork(session_factory)

    return _uow_factory, session_factory


async def main() -> None:
    uow_factory, session_factory = make_uow_factory()
    engine = GameEngine(uow_factory=uow_factory, llm=DemoLLM())

    result = await engine.resolve_turn(
        ResolveTurnInput(
            campaign_id="campaign-1",
            actor_id="actor-1",
            action="head toward the city",
        )
    )

    print("resolve_turn status:", result.status)
    print("narration:", result.narration)

    with session_factory() as session:
        campaign = session.get(Campaign, "campaign-1")
        state = json.loads(campaign.state_json or "{}")
        print("persisted game_time:", state.get("game_time"))
        print("persisted calendar:", state.get("calendar"))


if __name__ == "__main__":
    asyncio.run(main())
