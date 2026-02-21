from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Tuple

from sqlalchemy import or_

from .core.engine import GameEngine
from .core.normalize import normalize_campaign_name, parse_json_dict
from .core.types import ResolveTurnInput
from .persistence.sqlalchemy.models import (
    Actor,
    Campaign,
    Player,
    Session as GameSession,
    Snapshot,
    Timer,
    Turn,
)


@dataclass
class TurnClaim:
    campaign_id: str
    actor_id: str


class ZorkEmulator:
    """Compatibility facade shaped after discord_tron_master's ZorkEmulator.

    This keeps call patterns and return contracts as close as possible while
    routing through the standalone engine + persistence layer.
    """

    BASE_POINTS = 10
    POINTS_PER_LEVEL = 5
    MAX_ATTRIBUTE_VALUE = 20
    MAX_SUMMARY_CHARS = 4000
    MAX_STATE_CHARS = 8000
    MAX_RECENT_TURNS = 24
    MAX_TURN_CHARS = 1200
    MAX_NARRATION_CHARS = 3500
    XP_BASE = 100
    XP_PER_LEVEL = 50

    MAIN_PARTY_TOKEN = "main party"
    NEW_PATH_TOKEN = "new path"

    ROOM_IMAGE_STATE_KEY = "room_scene_images"
    PLAYER_STATS_KEY = "zork_stats"

    def __init__(self, game_engine: GameEngine, session_factory):
        self._engine = game_engine
        self._session_factory = session_factory
        self._claims: dict[tuple[str, str], TurnClaim] = {}

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    @classmethod
    def total_points_for_level(cls, level: int) -> int:
        return cls.BASE_POINTS + max(level - 1, 0) * cls.POINTS_PER_LEVEL

    @classmethod
    def xp_needed_for_level(cls, level: int) -> int:
        return cls.XP_BASE + max(level - 1, 0) * cls.XP_PER_LEVEL

    @classmethod
    def points_spent(cls, attributes: dict[str, int]) -> int:
        total = 0
        for value in attributes.values():
            if isinstance(value, int):
                total += value
        return total

    @staticmethod
    def _dump_json(data: dict[str, Any]) -> str:
        return json.dumps(data, ensure_ascii=True)

    @staticmethod
    def _load_json(text: Optional[str], default):
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            return default

    # ------------------------------------------------------------------
    # Storage accessors
    # ------------------------------------------------------------------

    def get_or_create_actor(self, actor_id: str, display_name: str | None = None) -> Actor:
        with self._session_factory() as session:
            row = session.get(Actor, actor_id)
            if row is None:
                row = Actor(id=actor_id, display_name=display_name, kind="human", metadata_json="{}")
                session.add(row)
                session.commit()
            return row

    def get_or_create_campaign(
        self,
        namespace: str,
        name: str,
        created_by_actor_id: str,
        campaign_id: str | None = None,
    ) -> Campaign:
        normalized = normalize_campaign_name(name)
        with self._session_factory() as session:
            row = (
                session.query(Campaign)
                .filter(Campaign.namespace == namespace)
                .filter(Campaign.name_normalized == normalized)
                .first()
            )
            if row is None:
                row = Campaign(
                    id=campaign_id,
                    namespace=namespace,
                    name=normalized,
                    name_normalized=normalized,
                    created_by_actor_id=created_by_actor_id,
                    summary="",
                    state_json="{}",
                    characters_json="{}",
                    row_version=1,
                )
                session.add(row)
                session.commit()
            return row

    def list_campaigns(self, namespace: str) -> list[Campaign]:
        with self._session_factory() as session:
            return list(
                session.query(Campaign)
                .filter(Campaign.namespace == namespace)
                .order_by(Campaign.name.asc())
                .all()
            )

    def get_or_create_session(
        self,
        campaign_id: str,
        surface: str,
        surface_key: str,
        surface_guild_id: str | None = None,
        surface_channel_id: str | None = None,
        surface_thread_id: str | None = None,
    ) -> GameSession:
        with self._session_factory() as session:
            row = session.query(GameSession).filter(GameSession.surface_key == surface_key).first()
            if row is None:
                row = GameSession(
                    campaign_id=campaign_id,
                    surface=surface,
                    surface_key=surface_key,
                    surface_guild_id=surface_guild_id,
                    surface_channel_id=surface_channel_id,
                    surface_thread_id=surface_thread_id,
                    enabled=True,
                    metadata_json="{}",
                )
                session.add(row)
                session.commit()
            return row

    def get_or_create_player(self, campaign_id: str, actor_id: str) -> Player:
        self.get_or_create_actor(actor_id)
        with self._session_factory() as session:
            row = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == actor_id)
                .first()
            )
            if row is None:
                row = Player(campaign_id=campaign_id, actor_id=actor_id, state_json="{}", attributes_json="{}")
                session.add(row)
                session.commit()
            return row

    def get_player_state(self, player: Player) -> dict[str, Any]:
        return parse_json_dict(player.state_json)

    def get_player_attributes(self, player: Player) -> dict[str, int]:
        data = parse_json_dict(player.attributes_json)
        out: dict[str, int] = {}
        for key, value in data.items():
            if isinstance(value, int):
                out[str(key)] = value
        return out

    def get_campaign_state(self, campaign: Campaign) -> dict[str, Any]:
        return parse_json_dict(campaign.state_json)

    def get_campaign_characters(self, campaign: Campaign) -> dict[str, Any]:
        return parse_json_dict(campaign.characters_json)

    def set_attribute(self, player: Player, name: str, value: int) -> tuple[bool, str]:
        if value < 0 or value > self.MAX_ATTRIBUTE_VALUE:
            return False, f"Value must be between 0 and {self.MAX_ATTRIBUTE_VALUE}."
        attrs = self.get_player_attributes(player)
        attrs[name] = value
        total_points = self.total_points_for_level(player.level)
        if self.points_spent(attrs) > total_points:
            return False, f"Not enough points. You have {total_points} total points."
        with self._session_factory() as session:
            row = session.get(Player, player.id)
            row.attributes_json = self._dump_json(attrs)
            row.updated_at = datetime.utcnow()
            session.commit()
        return True, "Attribute updated."

    def level_up(self, player: Player) -> tuple[bool, str]:
        needed = self.xp_needed_for_level(player.level)
        if player.xp < needed:
            return False, f"Need {needed} XP to level up."
        with self._session_factory() as session:
            row = session.get(Player, player.id)
            row.xp -= needed
            row.level += 1
            row.updated_at = datetime.utcnow()
            session.commit()
            return True, f"Leveled up to {row.level}."

    def get_recent_turns(self, campaign_id: str, limit: int | None = None) -> list[Turn]:
        if limit is None:
            limit = self.MAX_RECENT_TURNS
        with self._session_factory() as session:
            rows = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .order_by(Turn.id.desc())
                .limit(limit)
                .all()
            )
            rows.reverse()
            return rows

    # ------------------------------------------------------------------
    # Turn lifecycle (compat signatures)
    # ------------------------------------------------------------------

    async def begin_turn(
        self,
        campaign_id: str,
        actor_id: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return None, "Campaign not found."
        key = (campaign_id, actor_id)
        if key in self._claims:
            return None, None
        # Claim is ultimately enforced by DB lease in resolve_turn; this keeps
        # classic begin_turn/end_turn call shape for callers.
        self._claims[key] = TurnClaim(campaign_id=campaign_id, actor_id=actor_id)
        return campaign_id, None

    def end_turn(self, campaign_id: str, actor_id: str):
        self._claims.pop((campaign_id, actor_id), None)

    async def play_action(
        self,
        campaign_id: str,
        actor_id: str,
        action: str,
        session_id: str | None = None,
        manage_claim: bool = True,
    ) -> Optional[str]:
        should_end = False
        if manage_claim:
            cid, error_text = await self.begin_turn(campaign_id, actor_id)
            if error_text is not None:
                return error_text
            if cid is None:
                return None
            should_end = True
        try:
            result = await self._engine.resolve_turn(
                ResolveTurnInput(
                    campaign_id=campaign_id,
                    actor_id=actor_id,
                    action=action,
                    session_id=session_id,
                )
            )
            if result.status == "ok":
                return result.narration
            if result.status == "busy":
                return None
            if result.status == "conflict":
                return "The world shifts under your feet. Please try again."
            return f"Engine error: {result.conflict_reason or 'unknown'}"
        finally:
            if should_end:
                self.end_turn(campaign_id, actor_id)

    def execute_rewind(
        self,
        campaign_id: str,
        target_discord_message_id: str | int,
        channel_id: str | None = None,
    ) -> Optional[Tuple[int, int]]:
        target_turn_id = self._resolve_rewind_target_turn_id(campaign_id, str(target_discord_message_id))
        if target_turn_id is None:
            return None

        if channel_id is not None:
            return self._execute_rewind_channel_scoped(campaign_id, target_turn_id, str(channel_id))

        result = self._engine.rewind_to_turn(campaign_id, target_turn_id)
        if result.status != "ok" or result.target_turn_id is None:
            return None
        return (result.target_turn_id, result.deleted_turns)

    def _resolve_rewind_target_turn_id(self, campaign_id: str, target_message_id: str) -> int | None:
        with self._session_factory() as session:
            target_turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .filter(Turn.kind == "narrator")
                .filter(Turn.external_message_id == target_message_id)
                .first()
            )
            if target_turn is None:
                player_turn = (
                    session.query(Turn)
                    .filter(Turn.campaign_id == campaign_id)
                    .filter(Turn.external_user_message_id == target_message_id)
                    .order_by(Turn.id.asc())
                    .first()
                )
                if player_turn is not None:
                    target_turn = (
                        session.query(Turn)
                        .filter(Turn.campaign_id == campaign_id)
                        .filter(Turn.kind == "narrator")
                        .filter(Turn.id >= player_turn.id)
                        .order_by(Turn.id.asc())
                        .first()
                    )
            if target_turn is None:
                return None
            return target_turn.id

    def _execute_rewind_channel_scoped(
        self,
        campaign_id: str,
        target_turn_id: int,
        channel_id: str,
    ) -> Optional[Tuple[int, int]]:
        with self._session_factory() as session:
            snapshot = (
                session.query(Snapshot)
                .filter(Snapshot.campaign_id == campaign_id)
                .filter(Snapshot.turn_id == target_turn_id)
                .first()
            )
            if snapshot is None:
                return None

            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return None

            campaign.state_json = snapshot.campaign_state_json
            campaign.characters_json = snapshot.campaign_characters_json
            campaign.summary = snapshot.campaign_summary
            campaign.last_narration = snapshot.campaign_last_narration
            campaign.memory_visible_max_turn_id = target_turn_id
            campaign.row_version = max(int(campaign.row_version), 0) + 1
            campaign.updated_at = datetime.utcnow()

            players_data = self._load_json(snapshot.players_json, [])
            if isinstance(players_data, dict):
                players_data = players_data.get("players", [])
            if not isinstance(players_data, list):
                players_data = []
            for pdata in players_data:
                actor_id = pdata.get("actor_id")
                if not actor_id:
                    continue
                player = (
                    session.query(Player)
                    .filter(Player.campaign_id == campaign_id)
                    .filter(Player.actor_id == actor_id)
                    .first()
                )
                if player is None:
                    continue
                player.level = int(pdata.get("level", player.level))
                player.xp = int(pdata.get("xp", player.xp))
                player.attributes_json = str(pdata.get("attributes_json", player.attributes_json))
                player.state_json = str(pdata.get("state_json", player.state_json))
                player.updated_at = datetime.utcnow()

            scoped_session_ids = [
                row.id
                for row in (
                    session.query(GameSession.id)
                    .filter(GameSession.campaign_id == campaign_id)
                    .filter(
                        or_(
                            GameSession.surface_channel_id == channel_id,
                            GameSession.surface_thread_id == channel_id,
                            GameSession.surface_key == channel_id,
                        )
                    )
                    .all()
                )
            ]

            turn_ids_to_delete: list[int] = []
            if scoped_session_ids:
                turn_ids_to_delete = [
                    row.id
                    for row in (
                        session.query(Turn.id)
                        .filter(Turn.campaign_id == campaign_id)
                        .filter(Turn.id > target_turn_id)
                        .filter(Turn.session_id.in_(scoped_session_ids))
                        .all()
                    )
                ]

            if turn_ids_to_delete:
                session.query(Snapshot).filter(Snapshot.turn_id.in_(turn_ids_to_delete)).delete(synchronize_session=False)
                deleted_count = (
                    session.query(Turn)
                    .filter(Turn.id.in_(turn_ids_to_delete))
                    .delete(synchronize_session=False)
                )
            else:
                deleted_count = 0

            session.commit()
            return (target_turn_id, int(deleted_count))

    def record_turn_message_ids(
        self,
        campaign_id: str,
        user_message_id: str | int,
        bot_message_id: str | int,
    ) -> None:
        user_id = str(user_message_id)
        bot_id = str(bot_message_id)
        with self._session_factory() as session:
            narrator_turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .filter(Turn.kind == "narrator")
                .order_by(Turn.id.desc())
                .first()
            )
            if narrator_turn is not None:
                narrator_turn.external_message_id = bot_id
                narrator_turn.external_user_message_id = user_id

            player_turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .filter(Turn.kind == "player")
                .order_by(Turn.id.desc())
                .first()
            )
            if player_turn is not None:
                player_turn.external_user_message_id = user_id

            session.commit()

    # ------------------------------------------------------------------
    # Timer integration compatibility
    # ------------------------------------------------------------------

    def register_timer_message(
        self,
        campaign_id: str,
        message_id: str,
        channel_id: str | None = None,
        thread_id: str | None = None,
    ) -> bool:
        with self._session_factory() as session:
            timer = (
                session.query(Timer)
                .filter_by(campaign_id=campaign_id)
                .filter(Timer.status.in_(["scheduled_unbound", "scheduled_bound"]))
                .order_by(Timer.created_at.desc())
                .first()
            )
            if timer is None:
                return False
            if timer.status not in ("scheduled_unbound", "scheduled_bound"):
                return False
            timer.status = "scheduled_bound"
            timer.external_message_id = str(message_id)
            timer.external_channel_id = str(channel_id) if channel_id is not None else None
            timer.external_thread_id = str(thread_id) if thread_id is not None else None
            timer.updated_at = datetime.utcnow()
            session.commit()
            return True

    # ------------------------------------------------------------------
    # Memory visibility compatibility
    # ------------------------------------------------------------------

    def filter_memory_hits_by_visibility(self, campaign_id: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._engine.filter_memory_hits_by_visibility(campaign_id, hits)
