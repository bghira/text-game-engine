from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from .repos import (
    CampaignRepo,
    InflightTurnRepo,
    OutboxRepo,
    PlayerRepo,
    SnapshotRepo,
    TimerRepo,
    TurnRepo,
)


class SQLAlchemyUnitOfWork:
    def __init__(self, session_factory: sessionmaker[Session]):
        self._session_factory = session_factory
        self.session: Session | None = None

    def __enter__(self) -> "SQLAlchemyUnitOfWork":
        self.session = self._session_factory()
        self.campaigns = CampaignRepo(self.session)
        self.players = PlayerRepo(self.session)
        self.turns = TurnRepo(self.session)
        self.snapshots = SnapshotRepo(self.session)
        self.timers = TimerRepo(self.session)
        self.inflight = InflightTurnRepo(self.session)
        self.outbox = OutboxRepo(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.session is None:
            return
        if exc_type is not None:
            self.rollback()
        self.session.close()

    def commit(self) -> None:
        assert self.session is not None
        self.session.commit()

    def rollback(self) -> None:
        assert self.session is not None
        self.session.rollback()
