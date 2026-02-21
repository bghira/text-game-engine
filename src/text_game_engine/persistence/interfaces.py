from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class CampaignRepo(Protocol):
    def get(self, campaign_id: str): ...
    def cas_bump_row_version(self, campaign_id: str, expected_row_version: int) -> bool: ...


class PlayerRepo(Protocol):
    def get_by_campaign_actor(self, campaign_id: str, actor_id: str): ...
    def create(self, campaign_id: str, actor_id: str, state_json: str = "{}"): ...
    def list_by_campaign(self, campaign_id: str): ...


class TurnRepo(Protocol):
    def add(
        self,
        campaign_id: str,
        session_id: str | None,
        actor_id: str | None,
        kind: str,
        content: str,
        meta_json: str = "{}",
    ): ...
    def recent(self, campaign_id: str, limit: int): ...
    def delete_after(self, campaign_id: str, turn_id: int) -> int: ...


class SnapshotRepo(Protocol):
    def add(
        self,
        turn_id: int,
        campaign_id: str,
        campaign_state_json: str,
        campaign_characters_json: str,
        campaign_summary: str,
        campaign_last_narration: str | None,
        players_json: str,
    ): ...
    def get_by_turn_id(self, turn_id: int): ...
    def get_by_campaign_turn_id(self, campaign_id: str, turn_id: int): ...
    def delete_after_turn(self, campaign_id: str, turn_id: int) -> int: ...


class TimerRepo(Protocol):
    def get_active_for_campaign(self, campaign_id: str): ...
    def schedule(
        self,
        campaign_id: str,
        session_id: str | None,
        due_at: datetime,
        event_text: str,
        interruptible: bool,
        interrupt_action: str | None,
    ): ...
    def attach_message(
        self,
        timer_id: str,
        external_message_id: str,
        external_channel_id: str | None,
        external_thread_id: str | None,
    ) -> bool: ...
    def cancel_active(self, campaign_id: str, cancelled_at: datetime) -> int: ...
    def mark_expired(self, timer_id: str, fired_at: datetime) -> bool: ...
    def mark_consumed(self, timer_id: str, consumed_at: datetime) -> bool: ...


class InflightTurnRepo(Protocol):
    def acquire_or_steal(
        self,
        campaign_id: str,
        actor_id: str,
        claim_token: str,
        now: datetime,
        expires_at: datetime,
    ) -> bool: ...
    def validate_token(
        self,
        campaign_id: str,
        actor_id: str,
        claim_token: str,
        now: datetime,
    ) -> bool: ...
    def heartbeat(
        self,
        campaign_id: str,
        actor_id: str,
        claim_token: str,
        now: datetime,
        expires_at: datetime,
    ) -> bool: ...
    def release(self, campaign_id: str, actor_id: str, claim_token: str) -> int: ...


class OutboxRepo(Protocol):
    def add(
        self,
        campaign_id: str,
        session_id: str | None,
        event_type: str,
        idempotency_key: str,
        payload_json: str,
    ) -> None: ...


class UnitOfWork(Protocol):
    campaigns: CampaignRepo
    players: PlayerRepo
    turns: TurnRepo
    snapshots: SnapshotRepo
    timers: TimerRepo
    inflight: InflightTurnRepo
    outbox: OutboxRepo

    def commit(self) -> None: ...
    def rollback(self) -> None: ...

    def __enter__(self) -> "UnitOfWork": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
