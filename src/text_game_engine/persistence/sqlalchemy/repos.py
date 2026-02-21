from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import (
    Campaign,
    InflightTurn,
    OutboxEvent,
    Player,
    Snapshot,
    Timer,
    Turn,
)


class CampaignRepo:
    def __init__(self, session: Session):
        self.session = session

    def get(self, campaign_id: str) -> Campaign | None:
        return self.session.get(Campaign, campaign_id)

    def cas_apply_update(
        self,
        campaign_id: str,
        expected_row_version: int,
        values: dict[str, object],
    ) -> bool:
        update_values = dict(values)
        update_values["row_version"] = Campaign.row_version + 1
        update_values["updated_at"] = datetime.utcnow()
        stmt = (
            update(Campaign)
            .where(Campaign.id == campaign_id)
            .where(Campaign.row_version == expected_row_version)
            .values(**update_values)
        )
        result = self.session.execute(stmt)
        return result.rowcount == 1


class PlayerRepo:
    def __init__(self, session: Session):
        self.session = session

    def get_by_campaign_actor(self, campaign_id: str, actor_id: str) -> Player | None:
        stmt = (
            select(Player)
            .where(Player.campaign_id == campaign_id)
            .where(Player.actor_id == actor_id)
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def create(self, campaign_id: str, actor_id: str, state_json: str = "{}") -> Player:
        row = Player(campaign_id=campaign_id, actor_id=actor_id, state_json=state_json)
        self.session.add(row)
        self.session.flush()
        return row

    def list_by_campaign(self, campaign_id: str) -> list[Player]:
        stmt = select(Player).where(Player.campaign_id == campaign_id)
        return list(self.session.execute(stmt).scalars().all())


class TurnRepo:
    def __init__(self, session: Session):
        self.session = session

    def add(
        self,
        campaign_id: str,
        session_id: str | None,
        actor_id: str | None,
        kind: str,
        content: str,
        meta_json: str = "{}",
    ) -> Turn:
        row = Turn(
            campaign_id=campaign_id,
            session_id=session_id,
            actor_id=actor_id,
            kind=kind,
            content=content,
            meta_json=meta_json,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def recent(self, campaign_id: str, limit: int) -> list[Turn]:
        stmt = (
            select(Turn)
            .where(Turn.campaign_id == campaign_id)
            .order_by(Turn.id.desc())
            .limit(limit)
        )
        rows = list(self.session.execute(stmt).scalars().all())
        rows.reverse()
        return rows

    def delete_after(self, campaign_id: str, turn_id: int) -> int:
        stmt = delete(Turn).where(Turn.campaign_id == campaign_id).where(Turn.id > turn_id)
        return self.session.execute(stmt).rowcount or 0


class SnapshotRepo:
    def __init__(self, session: Session):
        self.session = session

    def add(
        self,
        turn_id: int,
        campaign_id: str,
        campaign_state_json: str,
        campaign_characters_json: str,
        campaign_summary: str,
        campaign_last_narration: str | None,
        players_json: str,
    ) -> Snapshot:
        row = Snapshot(
            turn_id=turn_id,
            campaign_id=campaign_id,
            campaign_state_json=campaign_state_json,
            campaign_characters_json=campaign_characters_json,
            campaign_summary=campaign_summary,
            campaign_last_narration=campaign_last_narration,
            players_json=players_json,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def get_by_turn_id(self, turn_id: int) -> Snapshot | None:
        stmt = select(Snapshot).where(Snapshot.turn_id == turn_id).limit(1)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_campaign_turn_id(self, campaign_id: str, turn_id: int) -> Snapshot | None:
        stmt = (
            select(Snapshot)
            .where(Snapshot.campaign_id == campaign_id)
            .where(Snapshot.turn_id == turn_id)
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def delete_after_turn(self, campaign_id: str, turn_id: int) -> int:
        subq = select(Turn.id).where(Turn.campaign_id == campaign_id).where(Turn.id > turn_id)
        stmt = delete(Snapshot).where(Snapshot.turn_id.in_(subq))
        return self.session.execute(stmt).rowcount or 0


class TimerRepo:
    ACTIVE = ("scheduled_unbound", "scheduled_bound")

    def __init__(self, session: Session):
        self.session = session

    def get_active_for_campaign(self, campaign_id: str) -> Timer | None:
        stmt = (
            select(Timer)
            .where(Timer.campaign_id == campaign_id)
            .where(Timer.status.in_(self.ACTIVE))
            .order_by(Timer.created_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def schedule(
        self,
        campaign_id: str,
        session_id: str | None,
        due_at: datetime,
        event_text: str,
        interruptible: bool,
        interrupt_action: str | None,
    ) -> Timer:
        row = Timer(
            campaign_id=campaign_id,
            session_id=session_id,
            due_at=due_at,
            event_text=event_text,
            interruptible=interruptible,
            interrupt_action=interrupt_action,
            status="scheduled_unbound",
        )
        self.session.add(row)
        self.session.flush()
        return row

    def attach_message(
        self,
        timer_id: str,
        external_message_id: str,
        external_channel_id: str | None,
        external_thread_id: str | None,
    ) -> bool:
        stmt = (
            update(Timer)
            .where(Timer.id == timer_id)
            .where(Timer.status.in_(self.ACTIVE))
            .values(
                status="scheduled_bound",
                external_message_id=external_message_id,
                external_channel_id=external_channel_id,
                external_thread_id=external_thread_id,
                updated_at=datetime.utcnow(),
            )
        )
        result = self.session.execute(stmt)
        return result.rowcount == 1

    def cancel_active(self, campaign_id: str, cancelled_at: datetime) -> int:
        stmt = (
            update(Timer)
            .where(Timer.campaign_id == campaign_id)
            .where(Timer.status.in_(self.ACTIVE))
            .values(status="cancelled", cancelled_at=cancelled_at, updated_at=cancelled_at)
        )
        return self.session.execute(stmt).rowcount or 0

    def mark_expired(self, timer_id: str, fired_at: datetime) -> bool:
        stmt = (
            update(Timer)
            .where(Timer.id == timer_id)
            .where(Timer.status.in_(self.ACTIVE))
            .values(status="expired", fired_at=fired_at, updated_at=fired_at)
        )
        return (self.session.execute(stmt).rowcount or 0) == 1

    def mark_consumed(self, timer_id: str, consumed_at: datetime) -> bool:
        stmt = (
            update(Timer)
            .where(Timer.id == timer_id)
            .where(Timer.status == "expired")
            .values(status="consumed", updated_at=consumed_at)
        )
        return (self.session.execute(stmt).rowcount or 0) == 1


class InflightTurnRepo:
    def __init__(self, session: Session):
        self.session = session

    def acquire_or_steal(
        self,
        campaign_id: str,
        actor_id: str,
        claim_token: str,
        now: datetime,
        expires_at: datetime,
    ) -> bool:
        try:
            with self.session.begin_nested():
                row = InflightTurn(
                    campaign_id=campaign_id,
                    actor_id=actor_id,
                    claim_token=claim_token,
                    claimed_at=now,
                    heartbeat_at=now,
                    expires_at=expires_at,
                )
                self.session.add(row)
                self.session.flush()
                return True
        except IntegrityError as exc:
            message = str(exc).lower()
            if (
                "uq_tge_inflight_campaign_actor" in message
                or "tge_inflight_turns.campaign_id, tge_inflight_turns.actor_id" in message
            ):
                pass
            else:
                raise

        stmt = (
            update(InflightTurn)
            .where(InflightTurn.campaign_id == campaign_id)
            .where(InflightTurn.actor_id == actor_id)
            .where(InflightTurn.expires_at < now)
            .values(
                claim_token=claim_token,
                claimed_at=now,
                heartbeat_at=now,
                expires_at=expires_at,
            )
        )
        return (self.session.execute(stmt).rowcount or 0) == 1

    def validate_token(
        self,
        campaign_id: str,
        actor_id: str,
        claim_token: str,
        now: datetime,
    ) -> bool:
        stmt = (
            select(InflightTurn)
            .where(InflightTurn.campaign_id == campaign_id)
            .where(InflightTurn.actor_id == actor_id)
            .where(InflightTurn.claim_token == claim_token)
            .limit(1)
        )
        row = self.session.execute(stmt).scalar_one_or_none()
        if row is None:
            return False
        return row.expires_at >= now

    def heartbeat(
        self,
        campaign_id: str,
        actor_id: str,
        claim_token: str,
        now: datetime,
        expires_at: datetime,
    ) -> bool:
        stmt = (
            update(InflightTurn)
            .where(InflightTurn.campaign_id == campaign_id)
            .where(InflightTurn.actor_id == actor_id)
            .where(InflightTurn.claim_token == claim_token)
            .values(heartbeat_at=now, expires_at=expires_at)
        )
        return (self.session.execute(stmt).rowcount or 0) == 1

    def release(self, campaign_id: str, actor_id: str, claim_token: str) -> int:
        stmt = (
            delete(InflightTurn)
            .where(InflightTurn.campaign_id == campaign_id)
            .where(InflightTurn.actor_id == actor_id)
            .where(InflightTurn.claim_token == claim_token)
        )
        return self.session.execute(stmt).rowcount or 0


class OutboxRepo:
    def __init__(self, session: Session):
        self.session = session

    def add(
        self,
        campaign_id: str,
        session_id: str | None,
        event_type: str,
        idempotency_key: str,
        payload_json: str,
    ) -> None:
        scope = session_id or "__none__"
        try:
            with self.session.begin_nested():
                row = OutboxEvent(
                    campaign_id=campaign_id,
                    session_id=session_id,
                    session_scope=scope,
                    event_type=event_type,
                    idempotency_key=idempotency_key,
                    payload_json=payload_json,
                )
                self.session.add(row)
                self.session.flush()
        except IntegrityError as exc:
            message = str(exc).lower()
            if (
                "uq_tge_outbox_campaign_session_event_key" in message
                or "tge_outbox_events.campaign_id, tge_outbox_events.session_scope, tge_outbox_events.event_type, tge_outbox_events.idempotency_key" in message
            ):
                # Outbox keys are idempotent by design; duplicate inserts are no-ops.
                return
            raise
