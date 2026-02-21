from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import text

from text_game_engine.persistence.sqlalchemy.db import build_engine, build_session_factory, create_schema
from text_game_engine.persistence.sqlalchemy.models import Actor, Campaign
from text_game_engine.persistence.sqlalchemy.uow import SQLAlchemyUnitOfWork


@pytest.fixture()
def session_factory():
    engine = build_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    sf = build_session_factory(engine)
    with sf() as session:
        session.execute(text("PRAGMA foreign_keys=ON"))
        session.commit()
    return sf


@pytest.fixture()
def uow_factory(session_factory):
    def _factory():
        return SQLAlchemyUnitOfWork(session_factory)

    return _factory


@pytest.fixture()
def seed_campaign_and_actor(session_factory):
    with session_factory() as session:
        actor = Actor(id="actor-1", display_name="Tester", kind="human", metadata_json="{}")
        campaign = Campaign(
            id="campaign-1",
            namespace="default",
            name="main",
            name_normalized="main",
            created_by_actor_id=actor.id,
            summary="",
            state_json="{}",
            characters_json="{}",
            row_version=1,
            memory_visible_max_turn_id=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(actor)
        session.add(campaign)
        session.commit()
    return {"campaign_id": "campaign-1", "actor_id": "actor-1"}
