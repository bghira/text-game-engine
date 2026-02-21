from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


TurnIDType = BigInteger().with_variant(Integer, "sqlite")


class Campaign(TimestampMixin, Base):
    __tablename__ = "tge_campaigns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    namespace: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    name_normalized: Mapped[str] = mapped_column(String(128), nullable=False)
    created_by_actor_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("tge_actors.id"), nullable=True)

    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    state_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    characters_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    last_narration: Mapped[str | None] = mapped_column(Text, nullable=True)

    memory_visible_max_turn_id: Mapped[int | None] = mapped_column(TurnIDType, nullable=True)
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    __table_args__ = (
        UniqueConstraint("namespace", "name_normalized", name="uq_tge_campaign_namespace_name_norm"),
    )


class Session(TimestampMixin, Base):
    __tablename__ = "tge_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_campaigns.id"), nullable=False)

    surface: Mapped[str] = mapped_column(String(32), nullable=False)
    surface_key: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    surface_guild_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    surface_channel_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    surface_thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class Actor(TimestampMixin, Base):
    __tablename__ = "tge_actors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    kind: Mapped[str] = mapped_column(String(24), nullable=False, default="human")
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class ActorExternalRef(TimestampMixin, Base):
    __tablename__ = "tge_actor_external_refs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    actor_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_actors.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    __table_args__ = (
        UniqueConstraint("provider", "external_id", name="uq_tge_actor_external_ref_provider_external"),
        UniqueConstraint("actor_id", "provider", "external_id", name="uq_tge_actor_external_ref_actor_provider_external"),
    )


class Player(TimestampMixin, Base):
    __tablename__ = "tge_players"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_campaigns.id"), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_actors.id"), nullable=False)

    level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    xp: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attributes_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    state_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("campaign_id", "actor_id", name="uq_tge_player_campaign_actor"),
    )


class Turn(Base):
    __tablename__ = "tge_turns"

    id: Mapped[int] = mapped_column(TurnIDType, primary_key=True, autoincrement=True)
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_campaigns.id"), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("tge_sessions.id"), nullable=True)
    actor_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("tge_actors.id"), nullable=True)

    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    external_message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    external_user_message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


Index("ix_tge_turn_campaign_id_desc", Turn.campaign_id, Turn.id.desc())
Index("ix_tge_turn_campaign_external_msg", Turn.campaign_id, Turn.external_message_id)


class Snapshot(Base):
    __tablename__ = "tge_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    turn_id: Mapped[int] = mapped_column(TurnIDType, ForeignKey("tge_turns.id"), nullable=False, unique=True)
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_campaigns.id"), nullable=False)

    campaign_state_json: Mapped[str] = mapped_column(Text, nullable=False)
    campaign_characters_json: Mapped[str] = mapped_column(Text, nullable=False)
    campaign_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    campaign_last_narration: Mapped[str | None] = mapped_column(Text, nullable=True)
    players_json: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


Index("ix_tge_snapshot_campaign_turn", Snapshot.campaign_id, Snapshot.turn_id.desc())


class Timer(TimestampMixin, Base):
    __tablename__ = "tge_timers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_campaigns.id"), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("tge_sessions.id"), nullable=True)

    status: Mapped[str] = mapped_column(String(24), nullable=False, default="scheduled_unbound")
    event_text: Mapped[str] = mapped_column(Text, nullable=False)
    interruptible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    interrupt_action: Mapped[str | None] = mapped_column(Text, nullable=True)

    due_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    fired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    external_message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    external_channel_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    external_thread_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    meta_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    __table_args__ = (
        CheckConstraint(
            "status IN ('scheduled_unbound','scheduled_bound','cancelled','expired','consumed')",
            name="timer_status_valid",
        ),
    )


Index("ix_tge_timer_campaign_status_due", Timer.campaign_id, Timer.status, Timer.due_at)
Index(
    "uq_tge_timer_one_active_per_campaign",
    Timer.campaign_id,
    unique=True,
    sqlite_where=text("status IN ('scheduled_unbound','scheduled_bound')"),
    postgresql_where=text("status IN ('scheduled_unbound','scheduled_bound')"),
)


class InflightTurn(Base):
    __tablename__ = "tge_inflight_turns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_campaigns.id"), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_actors.id"), nullable=False)
    claim_token: Mapped[str] = mapped_column(String(64), nullable=False)
    claimed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint("campaign_id", "actor_id", name="uq_tge_inflight_campaign_actor"),
    )


Index("ix_tge_inflight_expiry", InflightTurn.expires_at)


class MediaRef(TimestampMixin, Base):
    __tablename__ = "tge_media_refs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_campaigns.id"), nullable=False)
    player_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("tge_players.id"), nullable=True)

    ref_type: Mapped[str] = mapped_column(String(32), nullable=False)
    room_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class Embedding(Base):
    __tablename__ = "tge_embeddings"

    turn_id: Mapped[int] = mapped_column(TurnIDType, ForeignKey("tge_turns.id"), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_campaigns.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


Index("ix_tge_embedding_campaign", Embedding.campaign_id)


class OutboxEvent(TimestampMixin, Base):
    __tablename__ = "tge_outbox_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id: Mapped[str] = mapped_column(String(36), ForeignKey("tge_campaigns.id"), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("tge_sessions.id"), nullable=True)
    session_scope: Mapped[str] = mapped_column(String(36), nullable=False, default="__none__")

    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "campaign_id",
            "session_scope",
            "event_type",
            "idempotency_key",
            name="uq_tge_outbox_campaign_session_event_key",
        ),
    )


Index("ix_tge_outbox_status_next_created", OutboxEvent.status, OutboxEvent.next_attempt_at, OutboxEvent.created_at)
