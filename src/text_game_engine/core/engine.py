from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from .errors import StaleClaimError, TurnBusyError
from .normalize import apply_patch, dump_json, normalize_give_item, parse_json_dict
from .ports import ActorResolverPort, LLMPort
from .types import ResolveTurnInput, ResolveTurnResult, RewindResult, TurnContext


class GameEngine:
    AUTO_FIX_COUNTERS_KEY = "_auto_fix_counters"
    MIN_TURN_ADVANCE_MINUTES = 1
    DEFAULT_TURN_ADVANCE_MINUTES = 5
    MAX_TURN_ADVANCE_MINUTES = 180

    def __init__(
        self,
        uow_factory: Callable[[], Any],
        llm: LLMPort,
        actor_resolver: ActorResolverPort | None = None,
        clock: Callable[[], datetime] | None = None,
        lease_ttl_seconds: int = 90,
        max_conflict_retries: int = 1,
        player_state_sanitizer: Callable[
            [dict[str, Any], dict[str, Any], str, str], dict[str, Any]
        ] | None = None,
    ):
        self._uow_factory = uow_factory
        self._llm = llm
        self._actor_resolver = actor_resolver
        self._clock = clock or (lambda: datetime.now(timezone.utc).replace(tzinfo=None))
        self._lease_ttl_seconds = lease_ttl_seconds
        self._max_conflict_retries = max_conflict_retries
        self._player_state_sanitizer = player_state_sanitizer

    @classmethod
    def _increment_auto_fix_counter(
        cls,
        campaign_state: dict[str, Any],
        key: str,
        amount: int = 1,
    ) -> None:
        if not isinstance(campaign_state, dict):
            return
        safe_key = re.sub(r"[^a-z0-9_]+", "_", str(key or "").strip().lower()).strip("_")
        if not safe_key:
            return
        try:
            safe_amount = max(1, int(amount))
        except (TypeError, ValueError):
            safe_amount = 1
        counters = campaign_state.get(cls.AUTO_FIX_COUNTERS_KEY)
        if not isinstance(counters, dict):
            counters = {}
            campaign_state[cls.AUTO_FIX_COUNTERS_KEY] = counters
        current = cls._coerce_non_negative_int(counters.get(safe_key, 0), default=0)
        counters[safe_key] = min(10**9, current + safe_amount)

    async def resolve_turn(
        self,
        turn_input: ResolveTurnInput,
        before_phase_c: Callable[[TurnContext, int], Awaitable[None] | None] | None = None,
    ) -> ResolveTurnResult:
        for attempt in range(self._max_conflict_retries + 1):
            claim_token = uuid.uuid4().hex
            context: TurnContext | None = None
            try:
                context = self._phase_a(turn_input, claim_token)
                llm_output = await self._llm.complete_turn(context)

                if before_phase_c is not None:
                    maybe = before_phase_c(context, attempt)
                    if asyncio.iscoroutine(maybe):
                        await maybe

                return self._phase_c(turn_input, context, claim_token, llm_output)
            except TurnBusyError:
                return ResolveTurnResult(status="busy", conflict_reason="turn_inflight")
            except StaleClaimError:
                self._release_claim_best_effort(turn_input.campaign_id, turn_input.actor_id, claim_token)
                if attempt < self._max_conflict_retries:
                    continue
                return ResolveTurnResult(status="conflict", conflict_reason="stale_claim_or_row_version")
            except Exception as e:  # pragma: no cover - defensive surface
                self._release_claim_best_effort(turn_input.campaign_id, turn_input.actor_id, claim_token)
                return ResolveTurnResult(status="error", conflict_reason=str(e))

        return ResolveTurnResult(status="conflict", conflict_reason="max_retries_exhausted")

    def rewind_to_turn(self, campaign_id: str, target_turn_id: int) -> RewindResult:
        with self._uow_factory() as uow:
            campaign = uow.campaigns.get(campaign_id)
            if campaign is None:
                return RewindResult(status="error", reason="campaign_not_found")

            snapshot = uow.snapshots.get_by_campaign_turn_id(campaign_id, target_turn_id)
            if snapshot is None:
                return RewindResult(status="error", reason="snapshot_not_found")

            ok = uow.campaigns.cas_apply_update(
                campaign_id=campaign_id,
                expected_row_version=campaign.row_version,
                values={
                    "state_json": snapshot.campaign_state_json,
                    "characters_json": snapshot.campaign_characters_json,
                    "summary": snapshot.campaign_summary,
                    "last_narration": snapshot.campaign_last_narration,
                    "memory_visible_max_turn_id": target_turn_id,
                },
            )
            if not ok:
                uow.rollback()
                return RewindResult(status="conflict", reason="row_version_conflict")

            players_data = json.loads(snapshot.players_json)
            if isinstance(players_data, dict):
                players_data = players_data.get("players", [])
            if not isinstance(players_data, list):
                players_data = []
            for pdata in players_data:
                actor_id = pdata.get("actor_id")
                if not actor_id:
                    continue
                player = uow.players.get_by_campaign_actor(campaign_id, actor_id)
                if player is None:
                    continue
                player.level = int(pdata.get("level", player.level))
                player.xp = int(pdata.get("xp", player.xp))
                player.attributes_json = str(pdata.get("attributes_json", player.attributes_json))
                player.state_json = str(pdata.get("state_json", player.state_json))
                player.updated_at = self._clock()

            uow.snapshots.delete_after_turn(campaign_id, target_turn_id)
            deleted_turns = uow.turns.delete_after(campaign_id, target_turn_id)

            uow.outbox.add(
                campaign_id=campaign_id,
                session_id=None,
                event_type="memory_prune_requested",
                idempotency_key=f"rewind:{target_turn_id}",
                payload_json=dump_json({"campaign_id": campaign_id, "after_turn_id": target_turn_id}),
            )
            uow.commit()
            return RewindResult(status="ok", target_turn_id=target_turn_id, deleted_turns=deleted_turns)

    def filter_memory_hits_by_visibility(
        self,
        campaign_id: str,
        hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        with self._uow_factory() as uow:
            campaign = uow.campaigns.get(campaign_id)
            if campaign is None:
                return []
            watermark = campaign.memory_visible_max_turn_id
            if watermark is None:
                return hits
            out: list[dict[str, Any]] = []
            for hit in hits:
                try:
                    turn_id = int(hit.get("turn_id"))
                except Exception:
                    continue
                if turn_id <= watermark:
                    out.append(hit)
            return out

    @staticmethod
    def _player_slug_key(value: object) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:64]

    @staticmethod
    def _player_visibility_slug(actor_id: str) -> str:
        raw = str(actor_id or "").strip()
        return f"player-{raw}" if raw else ""

    @staticmethod
    def _default_visibility_scope(actor_location_key: str) -> str:
        location_key = str(actor_location_key or "").strip().lower()
        return "local" if location_key and location_key != "unknown-room" else "public"

    @staticmethod
    def _is_private_phone_command_line(value: object) -> bool:
        text = " ".join(str(value or "").strip().split())
        if not text:
            return False
        return bool(
            re.match(
                r"^(?:i\s+)?(?:(?:send|check|read|open|view|look\s+at)\s+)?"
                r"(?:(?:my|the)\s+)?(?:phone|sms|texts?|messages?)\b",
                text,
                flags=re.IGNORECASE,
            )
            or re.match(
                r"^(?:i\s+)?(?:send\s+)?(?:sms|text|message)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _redact_private_phone_command_lines(cls, text: str) -> tuple[str, bool]:
        if not text:
            return "", False
        kept_lines: list[str] = []
        redacted = False
        for raw_line in str(text).splitlines():
            if cls._is_private_phone_command_line(raw_line):
                redacted = True
                continue
            kept_lines.append(raw_line)
        return "\n".join(kept_lines).strip(), redacted

    @staticmethod
    def _force_private_visibility_for_phone_activity(
        actor_id: str,
        visibility: dict[str, Any],
    ) -> dict[str, Any]:
        actor_slug = GameEngine._player_visibility_slug(actor_id)
        reason = " ".join(str(visibility.get("reason") or "").split())[:240]
        return {
            "scope": "private",
            "actor_player_slug": str(visibility.get("actor_player_slug") or actor_slug).strip() or actor_slug,
            "actor_actor_id": str(visibility.get("actor_actor_id") or actor_id).strip() or actor_id,
            "visible_player_slugs": [actor_slug] if actor_slug else [],
            "visible_actor_ids": [str(actor_id).strip()] if str(actor_id).strip() else [],
            "location_key": None,
            "aware_npc_slugs": [],
            "reason": reason or "Private phone/SMS activity is actor-only unless explicitly shared.",
            "source": "auto-private-phone",
        }

    @classmethod
    def _promote_player_npc_slugs(
        cls,
        visibility: dict[str, Any],
        campaign_players: list[Any],
    ) -> dict[str, Any]:
        """Cross-reference aware_npc_slugs against real players.

        When the LLM puts a real player's character slug in npc_slugs,
        the turn should be visible to that player.  npc_slugs alone are
        informational and do NOT expose the turn to other players.

        This method auto-promotes any real player found in npc_slugs
        into visible_player_slugs / visible_actor_ids so the turn
        appears in their RECENT_TURNS context.
        """
        npc_slugs = visibility.get("aware_npc_slugs")
        if not npc_slugs or not isinstance(npc_slugs, list):
            return visibility
        scope = str(visibility.get("scope") or "").strip().lower()
        if scope == "public":
            # public turns are already visible to everyone
            return visibility

        # Build a lookup: character-name-slug → (actor_id, player_visibility_slug)
        player_by_char_slug: dict[str, tuple[str, str]] = {}
        for p in campaign_players:
            actor_id = str(getattr(p, "actor_id", "") or "").strip()
            if not actor_id:
                continue
            state = parse_json_dict(getattr(p, "state_json", "{}"))
            char_name = str(state.get("character_name") or "").strip()
            if char_name:
                char_slug = cls._player_slug_key(char_name)
                if char_slug:
                    player_by_char_slug[char_slug] = (
                        actor_id,
                        cls._player_visibility_slug(actor_id),
                    )

        if not player_by_char_slug:
            return visibility

        # Check each npc_slug against known players
        promoted = False
        vis_slugs = list(visibility.get("visible_player_slugs") or [])
        vis_ids = list(visibility.get("visible_actor_ids") or [])
        remaining_npc_slugs = []

        for npc_slug in npc_slugs:
            normalised = cls._player_slug_key(npc_slug)
            match = player_by_char_slug.get(normalised)
            if match is not None:
                aid, pslug = match
                # skip the acting player – already included
                if aid == str(visibility.get("actor_actor_id") or "").strip():
                    remaining_npc_slugs.append(npc_slug)
                    continue
                if pslug not in vis_slugs:
                    vis_slugs.append(pslug)
                if aid not in vis_ids:
                    vis_ids.append(aid)
                promoted = True
            else:
                remaining_npc_slugs.append(npc_slug)

        if not promoted:
            return visibility

        result = dict(visibility)
        result["visible_player_slugs"] = vis_slugs
        result["visible_actor_ids"] = vis_ids
        result["aware_npc_slugs"] = remaining_npc_slugs
        return result

    @staticmethod
    def _normalize_turn_visibility(
        actor_id: str,
        _actor_name: str,
        raw_visibility: object,
        *,
        actor_location_key: str,
    ) -> dict[str, Any]:
        actor_slug = GameEngine._player_visibility_slug(actor_id)
        default_scope = GameEngine._default_visibility_scope(actor_location_key)
        default_meta = {
            "scope": default_scope,
            "actor_player_slug": actor_slug,
            "actor_actor_id": actor_id,
            "visible_player_slugs": [],
            "visible_actor_ids": [],
            "location_key": (
                actor_location_key
                if default_scope == "local"
                and actor_location_key
                and actor_location_key.lower() != "unknown-room"
                else None
            ),
            "aware_npc_slugs": [],
        }
        if not isinstance(raw_visibility, dict):
            return default_meta
        scope = str(raw_visibility.get("scope") or "").strip().lower()
        if scope not in {"public", "private", "limited", "local"}:
            scope = default_scope
        visible_player_slugs = []
        raw_player_slugs = raw_visibility.get("player_slugs")
        if isinstance(raw_player_slugs, list):
            for item in raw_player_slugs:
                slug = GameEngine._player_slug_key(item)
                if slug and slug not in visible_player_slugs:
                    visible_player_slugs.append(slug)
        elif isinstance(raw_player_slugs, str):
            slug = GameEngine._player_slug_key(raw_player_slugs)
            if slug:
                visible_player_slugs.append(slug)
        # Keep the prompt contract aligned with the documented schema:
        # models specify player_slugs, not actor_ids. The engine still
        # maintains visible_actor_ids internally for the acting player.
        visible_actor_ids: list[str] = []
        if scope in {"private", "limited", "local"}:
            if actor_slug and actor_slug not in visible_player_slugs:
                visible_player_slugs.insert(0, actor_slug)
            if actor_id and actor_id not in visible_actor_ids:
                visible_actor_ids.insert(0, actor_id)
        aware_npc_slugs = []
        raw_npc_slugs = raw_visibility.get("npc_slugs")
        if isinstance(raw_npc_slugs, list):
            for item in raw_npc_slugs:
                slug = str(item or "").strip()
                if slug and slug not in aware_npc_slugs:
                    aware_npc_slugs.append(slug)
        elif isinstance(raw_npc_slugs, str) and raw_npc_slugs.strip():
            aware_npc_slugs.append(raw_npc_slugs.strip())
        reason = " ".join(str(raw_visibility.get("reason") or "").split())[:240]
        return {
            "scope": scope,
            "actor_player_slug": actor_slug,
            "actor_actor_id": actor_id,
            "visible_player_slugs": visible_player_slugs,
            "visible_actor_ids": visible_actor_ids,
            "location_key": str(default_meta.get("location_key") or "").strip() or None,
            "aware_npc_slugs": aware_npc_slugs,
            "reason": reason or None,
        }

    @staticmethod
    def _turn_visible_to_actor(
        turn: Any,
        actor_id: str,
        actor_slug: str,
        viewer_location_key: str,
    ) -> bool:
        meta = parse_json_dict(getattr(turn, "meta_json", "{}"))
        if bool(meta.get("suppress_context")):
            return False
        if str(getattr(turn, "actor_id", "") or "").strip() == str(actor_id or "").strip():
            return True
        visibility = meta.get("visibility") if isinstance(meta, dict) else None
        if not isinstance(visibility, dict):
            return True
        scope = str(visibility.get("scope") or "").strip().lower()
        raw_actor_ids = visibility.get("visible_actor_ids")
        actor_ids = set()
        if isinstance(raw_actor_ids, list):
            actor_ids = {str(item or "").strip() for item in raw_actor_ids if str(item or "").strip()}
        raw_player_slugs = visibility.get("visible_player_slugs")
        player_slugs = set()
        if isinstance(raw_player_slugs, list):
            player_slugs = {GameEngine._player_slug_key(item) for item in raw_player_slugs if GameEngine._player_slug_key(item)}
        has_explicit_participants = bool(actor_ids or player_slugs)
        is_participant = bool(
            str(actor_id or "").strip() in actor_ids
            or (actor_slug and actor_slug in player_slugs)
        )
        if scope in {"private", "limited"}:
            if has_explicit_participants and not is_participant:
                return False
            if has_explicit_participants:
                return is_participant
            return False
        if scope in {"", "public"}:
            return True
        if scope == "local":
            turn_location_key = str(
                visibility.get("location_key") or meta.get("location_key") or ""
            ).strip().lower()
            if viewer_location_key and turn_location_key and viewer_location_key == turn_location_key:
                return True
        if str(actor_id or "").strip() in actor_ids:
            return True
        if actor_slug and actor_slug in player_slugs:
            return True
        return False

    def _phase_a(self, turn_input: ResolveTurnInput, claim_token: str) -> TurnContext:
        now = self._clock()
        expires_at = now + timedelta(seconds=self._lease_ttl_seconds)

        with self._uow_factory() as uow:
            campaign = uow.campaigns.get(turn_input.campaign_id)
            if campaign is None:
                raise TurnBusyError("campaign_not_found")

            acquired = uow.inflight.acquire_or_steal(
                campaign_id=turn_input.campaign_id,
                actor_id=turn_input.actor_id,
                claim_token=claim_token,
                now=now,
                expires_at=expires_at,
            )
            if not acquired:
                raise TurnBusyError("turn_inflight")

            player = uow.players.get_by_campaign_actor(turn_input.campaign_id, turn_input.actor_id)
            if player is None:
                player = uow.players.create(turn_input.campaign_id, turn_input.actor_id)

            player_state = parse_json_dict(player.state_json)
            actor_name = str(player_state.get("character_name") or "").strip()
            actor_slug = self._player_visibility_slug(turn_input.actor_id)
            viewer_location_key = self._room_key_from_state(player_state)
            turns = uow.turns.recent(turn_input.campaign_id, limit=24)
            def _turn_meta_payload(turn_obj):
                meta = parse_json_dict(getattr(turn_obj, "meta_json", "{}"))
                if not isinstance(meta, dict):
                    return None
                game_time = meta.get("game_time")
                return game_time if isinstance(game_time, dict) else None
            context = TurnContext(
                campaign_id=turn_input.campaign_id,
                actor_id=turn_input.actor_id,
                session_id=turn_input.session_id,
                action=turn_input.action,
                campaign_state=parse_json_dict(campaign.state_json),
                campaign_summary=campaign.summary or "",
                campaign_characters=parse_json_dict(campaign.characters_json),
                player_state=parse_json_dict(player.state_json),
                player_level=player.level,
                player_xp=player.xp,
                recent_turns=[
                    {
                        "id": t.id,
                        "turn_number": t.id,
                        "kind": t.kind,
                        "actor_id": t.actor_id,
                        "content": t.content,
                        "in_game_time": _turn_meta_payload(t),
                        "created_at": t.created_at.isoformat() if t.created_at else None,
                    }
                    for t in turns
                    if self._turn_visible_to_actor(
                        t,
                        turn_input.actor_id,
                        actor_slug,
                        viewer_location_key,
                    )
                ],
                start_row_version=campaign.row_version,
                now=now,
            )
            uow.commit()
            return context

    def _phase_c(self, turn_input: ResolveTurnInput, context: TurnContext, claim_token: str, llm_output) -> ResolveTurnResult:
        now = self._clock()

        with self._uow_factory() as uow:
            valid = uow.inflight.validate_token(
                campaign_id=turn_input.campaign_id,
                actor_id=turn_input.actor_id,
                claim_token=claim_token,
                now=now,
            )
            if not valid:
                raise StaleClaimError("claim_invalid")

            campaign = uow.campaigns.get(turn_input.campaign_id)
            player = uow.players.get_by_campaign_actor(turn_input.campaign_id, turn_input.actor_id)
            if campaign is None or player is None:
                raise StaleClaimError("missing_campaign_or_player")

            if campaign.row_version != context.start_row_version:
                raise StaleClaimError("row_version_changed")

            campaign_state = parse_json_dict(campaign.state_json)
            campaign_characters = parse_json_dict(campaign.characters_json)
            player_state = parse_json_dict(player.state_json)
            pre_turn_game_time = self._extract_game_time_snapshot(context.campaign_state)

            campaign_state_update = dict(llm_output.state_update or {})
            campaign_state_update = self._normalize_story_progress_update(
                campaign_state,
                campaign_state_update,
            )
            resolution_context = (
                f"{turn_input.action}\n"
                f"{str(llm_output.narration or '')}\n"
                f"{str(llm_output.summary_update or '')}"
            )
            campaign_state_update = self._guard_state_null_character_prunes(
                campaign_state_update,
                campaign_characters,
                resolution_context=resolution_context,
                campaign_state=campaign_state,
            )
            state_null_character_updates = self._character_updates_from_state_nulls(
                campaign_state_update,
                campaign_characters,
            )
            merged_character_updates = dict(state_null_character_updates)
            if isinstance(llm_output.character_updates, dict):
                merged_character_updates.update(llm_output.character_updates)
            if merged_character_updates:
                merged_character_updates = self._sanitize_character_removals(
                    campaign_characters,
                    merged_character_updates,
                    resolution_context=resolution_context,
                    campaign_state=campaign_state,
                )
            calendar_update = campaign_state_update.pop("calendar_update", None)
            campaign_state = apply_patch(campaign_state, campaign_state_update)
            campaign_state = self._apply_calendar_update(
                campaign_state,
                calendar_update,
                resolution_context=resolution_context,
            )
            campaign_state = self._ensure_game_time_progress(
                campaign_state,
                pre_turn_game_time,
                action_text=turn_input.action,
                narration_text=llm_output.narration or "",
            )
            _on_rails = bool(campaign_state.get("on_rails"))
            campaign_characters = self._apply_character_updates(
                campaign_characters,
                merged_character_updates,
                on_rails=_on_rails,
            )
            raw_player_update = llm_output.player_state_update or {}
            if self._player_state_sanitizer is not None and isinstance(raw_player_update, dict):
                raw_player_update = self._player_state_sanitizer(
                    player_state,
                    raw_player_update,
                    turn_input.action,
                    llm_output.narration or "",
                )
            player_state = apply_patch(player_state, raw_player_update)
            post_turn_game_time = self._extract_game_time_snapshot(campaign_state)
            actor_location_key = self._room_key_from_state(player_state)
            turn_visibility = self._normalize_turn_visibility(
                turn_input.actor_id,
                str(player_state.get("character_name") or ""),
                getattr(llm_output, "turn_visibility", None),
                actor_location_key=actor_location_key,
            )
            # Auto-promote real players listed in npc_slugs so they
            # actually receive the turn in their RECENT_TURNS context.
            turn_visibility = self._promote_player_npc_slugs(
                turn_visibility,
                uow.players.list_by_campaign(turn_input.campaign_id),
            )
            stored_player_action, private_phone_redacted = self._redact_private_phone_command_lines(
                turn_input.action
            )
            if private_phone_redacted:
                turn_visibility = self._force_private_visibility_for_phone_activity(
                    turn_input.actor_id,
                    turn_visibility,
                )
                self._increment_auto_fix_counter(
                    campaign_state,
                    "private_phone_redacted",
                )
            suppress_recent_context = bool(private_phone_redacted)
            turn_is_public = str(turn_visibility.get("scope") or "").strip().lower() == "public"

            summary = campaign.summary or ""
            if (
                turn_is_public
                and isinstance(llm_output.summary_update, str)
                and llm_output.summary_update.strip()
            ):
                summary = (summary + "\n" + llm_output.summary_update.strip()).strip()

            narration = (llm_output.narration or "").strip()
            if not narration:
                narration = self._fallback_narration_from_updates(
                    summary_update=llm_output.summary_update,
                    state_update=llm_output.state_update,
                    player_state_update=llm_output.player_state_update,
                    character_updates=llm_output.character_updates,
                ) or "The world shifts, but nothing clear emerges."
            active_char_sync = self._sync_active_player_character_location(
                campaign_characters,
                player_state=player_state,
            )
            world_char_sync = self._auto_sync_character_locations(
                campaign_characters,
                player_state=player_state,
                narration_text=narration,
            )
            if active_char_sync:
                self._increment_auto_fix_counter(
                    campaign_state,
                    "location_auto_sync_active_character",
                    amount=active_char_sync,
                )
            if world_char_sync:
                self._increment_auto_fix_counter(
                    campaign_state,
                    "location_auto_sync_world_characters",
                    amount=world_char_sync,
                )

            # give_item compatibility path - unresolved targets are non-fatal.
            give_item_payload: dict[str, Any] | None = None
            if llm_output.give_item is not None:
                give_item_payload = asdict(llm_output.give_item)
            _, give_item_issue = normalize_give_item(give_item_payload, self._actor_resolver)

            if give_item_issue is not None:
                uow.outbox.add(
                    campaign_id=turn_input.campaign_id,
                    session_id=turn_input.session_id,
                    event_type="give_item_unresolved",
                    idempotency_key=f"give_item_unresolved:{turn_input.actor_id}:{now.isoformat()}",
                    payload_json=dump_json({
                        "campaign_id": turn_input.campaign_id,
                        "actor_id": turn_input.actor_id,
                        "issue": give_item_issue,
                        "give_item": give_item_payload or {},
                    }),
                )

            player.xp += max(int(llm_output.xp_awarded or 0), 0)
            player.state_json = dump_json(player_state)
            player.updated_at = now
            player.last_active_at = now

            if turn_input.record_player_turn and stored_player_action:
                player_turn_meta = {
                    "game_time": pre_turn_game_time,
                    "visibility": turn_visibility,
                    "location_key": self._room_key_from_state(player_state),
                    "suppress_context": suppress_recent_context,
                }
                uow.turns.add(
                    campaign_id=turn_input.campaign_id,
                    session_id=turn_input.session_id,
                    actor_id=turn_input.actor_id,
                    kind="player",
                    content=stored_player_action,
                    meta_json=dump_json(player_turn_meta),
                )
            reasoning_text: str | None = None
            if isinstance(getattr(llm_output, "reasoning", None), str):
                compact_reasoning = " ".join(llm_output.reasoning.strip().split())
                if compact_reasoning:
                    reasoning_text = compact_reasoning[:1200]
            narrator_turn_meta = {
                "game_time": post_turn_game_time,
                "visibility": turn_visibility,
                "actor_player_slug": self._player_visibility_slug(turn_input.actor_id),
                "location_key": self._room_key_from_state(player_state),
                "suppress_context": suppress_recent_context,
            }
            if reasoning_text:
                narrator_turn_meta["reasoning"] = reasoning_text
            narrator_turn = uow.turns.add(
                campaign_id=turn_input.campaign_id,
                session_id=turn_input.session_id,
                actor_id=turn_input.actor_id,
                kind="narrator",
                content=narration,
                meta_json=dump_json(narrator_turn_meta),
            )

            timer_instruction = llm_output.timer_instruction if turn_input.allow_timer_instruction else None
            if timer_instruction is not None:
                uow.timers.cancel_active(turn_input.campaign_id, now)
                due_at = now + timedelta(seconds=max(30, int(timer_instruction.delay_seconds)))
                timer = uow.timers.schedule(
                    campaign_id=turn_input.campaign_id,
                    session_id=turn_input.session_id,
                    due_at=due_at,
                    event_text=timer_instruction.event_text,
                    interruptible=bool(timer_instruction.interruptible),
                    interrupt_action=timer_instruction.interrupt_action,
                )
                uow.outbox.add(
                    campaign_id=turn_input.campaign_id,
                    session_id=turn_input.session_id,
                    event_type="timer_scheduled",
                    idempotency_key=f"timer_scheduled:{timer.id}",
                    payload_json=dump_json(
                        {
                            "timer_id": timer.id,
                            "campaign_id": turn_input.campaign_id,
                            "session_id": turn_input.session_id,
                            "due_at": due_at.isoformat(),
                            "event_text": timer_instruction.event_text,
                            "interruptible": bool(timer_instruction.interruptible),
                            "interrupt_scope": str(
                                getattr(timer_instruction, "interrupt_scope", "global")
                                or "global"
                            ),
                        }
                    ),
                )

            if isinstance(llm_output.scene_image_prompt, str) and llm_output.scene_image_prompt.strip():
                room_key = self._room_key_from_state(player_state)
                uow.outbox.add(
                    campaign_id=turn_input.campaign_id,
                    session_id=turn_input.session_id,
                    event_type="scene_image_requested",
                    idempotency_key=f"scene_image:{narrator_turn.id}:{room_key}",
                    payload_json=dump_json(
                        {
                            "campaign_id": turn_input.campaign_id,
                            "session_id": turn_input.session_id,
                            "actor_id": turn_input.actor_id,
                            "turn_id": narrator_turn.id,
                            "room_key": room_key,
                            "scene_image_prompt": llm_output.scene_image_prompt.strip(),
                        }
                    ),
                )

            players_data = []
            for p in uow.players.list_by_campaign(turn_input.campaign_id):
                players_data.append(
                    {
                        "player_id": p.id,
                        "actor_id": p.actor_id,
                        "level": p.level,
                        "xp": p.xp,
                        "attributes_json": p.attributes_json,
                        "state_json": p.state_json,
                    }
                )

            uow.snapshots.add(
                turn_id=narrator_turn.id,
                campaign_id=turn_input.campaign_id,
                campaign_state_json=dump_json(campaign_state),
                campaign_characters_json=dump_json(campaign_characters),
                campaign_summary=summary,
                campaign_last_narration=narration,
                players_json=dump_json({"players": players_data}),
            )

            cas_ok = uow.campaigns.cas_apply_update(
                campaign_id=turn_input.campaign_id,
                expected_row_version=context.start_row_version,
                values={
                    "summary": summary,
                    "state_json": dump_json(campaign_state),
                    "characters_json": dump_json(campaign_characters),
                    "last_narration": narration,
                    "memory_visible_max_turn_id": narrator_turn.id,
                },
            )
            if not cas_ok:
                raise StaleClaimError("cas_failed")

            uow.inflight.release(turn_input.campaign_id, turn_input.actor_id, claim_token)
            uow.commit()

            return ResolveTurnResult(
                status="ok",
                narration=narration,
                scene_image_prompt=(llm_output.scene_image_prompt or None),
                timer_instruction=timer_instruction,
                give_item=give_item_payload,
            )

    def _release_claim_best_effort(self, campaign_id: str, actor_id: str, claim_token: str) -> None:
        try:
            with self._uow_factory() as uow:
                uow.inflight.release(campaign_id, actor_id, claim_token)
                uow.commit()
        except Exception:
            return

    def _room_key_from_state(self, state: dict[str, Any]) -> str:
        for key in ("room_id", "location", "room_title", "room_summary"):
            raw = str(state.get(key) or "").strip().lower()
            if raw:
                return raw[:120]
        return "unknown-room"

    @staticmethod
    def _fallback_narration_from_updates(
        *,
        summary_update: object,
        state_update: object,
        player_state_update: object,
        character_updates: object,
    ) -> str:
        if isinstance(player_state_update, dict):
            room_summary = str(player_state_update.get("room_summary") or "").strip()
            if room_summary:
                return room_summary[:300]
            room_title = str(player_state_update.get("room_title") or "").strip()
            if room_title:
                return f"{room_title}."
        summary_line = str(summary_update or "").strip()
        if summary_line:
            return summary_line.splitlines()[0][:300]
        if isinstance(character_updates, dict) and character_updates:
            return "Character roster updated."
        if isinstance(state_update, dict) and state_update:
            return "Noted."
        if isinstance(player_state_update, dict) and player_state_update:
            return "Noted."
        return ""

    @classmethod
    def _apply_character_updates(
        cls,
        existing: dict[str, Any],
        updates: dict[str, Any],
        on_rails: bool = False,
    ) -> dict[str, Any]:
        merged = dict(existing) if isinstance(existing, dict) else {}
        if not isinstance(updates, dict):
            return merged
        for raw_slug, fields in updates.items():
            slug = str(raw_slug).strip()
            if not slug:
                continue

            target_slug = cls._resolve_existing_character_slug(merged, slug)

            delete_requested = (
                fields is None
                or (
                    isinstance(fields, str)
                    and fields.strip().lower() in {"delete", "remove", "null"}
                )
                or (
                    isinstance(fields, dict)
                    and bool(
                        fields.get("remove")
                        or fields.get("delete")
                        or fields.get("_delete")
                        or fields.get("deleted")
                    )
                )
            )
            if delete_requested:
                merged.pop(target_slug or slug, None)
                continue

            if not isinstance(fields, dict):
                continue
            if target_slug and target_slug in merged:
                # Existing character — only accept mutable fields.
                _IMMUTABLE = {
                    "name",
                    "personality",
                    "background",
                    "appearance",
                    "speech_style",
                }
                old_location = str(merged[target_slug].get("location") or "").strip().lower()
                for key, value in fields.items():
                    if key not in _IMMUTABLE:
                        merged[target_slug][key] = value
                # If location changed but current_status wasn't updated,
                # clear the stale status to avoid contradictions.
                if (
                    "location" in fields
                    and "current_status" not in fields
                    and "current_status" in merged[target_slug]
                ):
                    new_location = str(fields["location"] or "").strip().lower()
                    if old_location and new_location and old_location != new_location:
                        merged[target_slug]["current_status"] = ""
            else:
                if on_rails:
                    continue
                # New character — store everything.
                merged[slug] = dict(fields)
        return merged

    @classmethod
    def _resolve_existing_character_slug(
        cls,
        existing: dict[str, Any],
        raw_slug: object,
    ) -> str | None:
        slug = str(raw_slug or "").strip()
        if not slug:
            return None
        canonical = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
        if slug in existing:
            return slug
        if canonical and canonical in existing:
            return canonical
        partial_matches: list[str] = []
        for existing_slug, existing_fields in existing.items():
            existing_canonical = re.sub(
                r"[^a-z0-9]+", "-", str(existing_slug).lower()
            ).strip("-")
            if canonical and canonical == existing_canonical:
                return str(existing_slug)
            if canonical and (
                existing_canonical.startswith(canonical)
                or canonical in existing_canonical
            ):
                partial_matches.append(str(existing_slug))
            if isinstance(existing_fields, dict):
                name_canonical = re.sub(
                    r"[^a-z0-9]+", "-",
                    str(existing_fields.get("name") or "").lower(),
                ).strip("-")
                if canonical and canonical == name_canonical:
                    return str(existing_slug)
                if canonical and (
                    name_canonical.startswith(canonical)
                    or canonical in name_canonical
                ):
                    partial_matches.append(str(existing_slug))
        if canonical:
            unique_matches = list(dict.fromkeys(partial_matches))
            if len(unique_matches) == 1:
                return unique_matches[0]
        return None

    @classmethod
    def _character_updates_from_state_nulls(
        cls,
        state_update: dict[str, Any] | Any,
        existing_chars: dict[str, Any],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not isinstance(state_update, dict) or not isinstance(existing_chars, dict):
            return out
        for key, value in state_update.items():
            if value is not None:
                continue
            resolved = cls._resolve_existing_character_slug(existing_chars, key)
            if resolved:
                out[resolved] = None
        return out

    @classmethod
    def _character_delete_requested(cls, fields: object) -> bool:
        return bool(
            fields is None
            or (
                isinstance(fields, str)
                and fields.strip().lower() in {"delete", "remove", "null"}
            )
            or (
                isinstance(fields, dict)
                and bool(
                    fields.get("remove")
                    or fields.get("delete")
                    or fields.get("_delete")
                    or fields.get("deleted")
                )
            )
        )

    @classmethod
    def _character_delete_allowed(
        cls,
        *,
        raw_slug: str,
        fields: object,
        existing_row: dict[str, Any] | None,
        context_text: str,
    ) -> bool:
        context = " ".join(str(context_text or "").lower().split())
        if not context:
            return False
        if isinstance(fields, dict) and str(fields.get("deceased_reason") or "").strip():
            return True

        remove_cues = (
            "remove from roster",
            "roster remove",
            "remove character",
            "delete character",
            "drop character",
            "purge duplicate",
            "duplicate",
            "cleanup roster",
            "roster cleanup",
            "retcon",
            "written out",
            "no longer in story",
        )
        death_cues = (
            "dead",
            "dies",
            "died",
            "killed",
            "murdered",
            "executed",
            "corpse",
            "funeral",
            "deceased",
        )
        has_delete_intent = any(cue in context for cue in remove_cues) or any(
            cue in context for cue in death_cues
        )
        if not has_delete_intent:
            return False

        aliases: list[str] = []
        slug_alias = re.sub(r"[^a-z0-9]+", " ", str(raw_slug or "").lower()).strip()
        if slug_alias:
            aliases.append(slug_alias)
        if isinstance(existing_row, dict):
            name_alias = re.sub(
                r"[^a-z0-9]+", " ",
                str(existing_row.get("name") or "").lower(),
            ).strip()
            if name_alias:
                aliases.append(name_alias)
        for alias in aliases:
            if alias and alias in context:
                return True
            tokens = [t for t in alias.split() if len(t) >= 4]
            if any(token in context for token in tokens):
                return True
        return False

    @classmethod
    def _sanitize_character_removals(
        cls,
        existing_chars: dict[str, Any],
        updates: object,
        *,
        resolution_context: str = "",
        campaign_state: dict[str, Any] | None = None,
        counter_key: str = "character_remove_blocked",
    ) -> dict[str, Any]:
        if not isinstance(updates, dict):
            return {}
        # Character deletion is now fully model-controlled (reasoning + structured updates).
        # Keep this hook for compatibility, but do not block removals.
        return dict(updates)

    @classmethod
    def _guard_state_null_character_prunes(
        cls,
        state_update: object,
        existing_chars: dict[str, Any],
        *,
        resolution_context: str = "",
        campaign_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(state_update, dict):
            return {}
        if not isinstance(existing_chars, dict):
            return dict(state_update)
        candidate_deletes: dict[str, Any] = {}
        for raw_key, value in state_update.items():
            if value is not None:
                continue
            resolved = cls._resolve_existing_character_slug(existing_chars, raw_key)
            if resolved:
                candidate_deletes[str(raw_key)] = None
        if not candidate_deletes:
            return dict(state_update)
        allowed_deletes = cls._sanitize_character_removals(
            existing_chars,
            candidate_deletes,
            resolution_context=resolution_context,
            campaign_state=campaign_state,
            counter_key="state_character_prune_blocked",
        )
        out = dict(state_update)
        for raw_key in candidate_deletes.keys():
            if raw_key not in allowed_deletes:
                out.pop(raw_key, None)
        return out

    @staticmethod
    def _extract_game_time_snapshot(state: dict[str, Any] | None) -> dict[str, int]:
        game_time = state.get("game_time") if isinstance(state, dict) else {}
        if not isinstance(game_time, dict):
            game_time = {}
        try:
            day = int(game_time.get("day", 1))
        except (TypeError, ValueError):
            day = 1
        try:
            hour = int(game_time.get("hour", 8))
        except (TypeError, ValueError):
            hour = 8
        try:
            minute = int(game_time.get("minute", 0))
        except (TypeError, ValueError):
            minute = 0
        day = max(1, day)
        hour = min(23, max(0, hour))
        minute = min(59, max(0, minute))
        return {"day": day, "hour": hour, "minute": minute}

    @staticmethod
    def _coerce_non_negative_int(value: object, default: int = 0) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0 else default

    @staticmethod
    def _is_ooc_action_text(action_text: object) -> bool:
        return bool(re.match(r"\s*\[OOC\b", str(action_text or ""), re.IGNORECASE))

    @staticmethod
    def _game_period_from_hour(hour: int) -> str:
        if 5 <= hour <= 11:
            return "morning"
        if 12 <= hour <= 16:
            return "afternoon"
        if 17 <= hour <= 20:
            return "evening"
        return "night"

    @classmethod
    def _game_time_to_total_minutes(cls, game_time: dict[str, Any]) -> int:
        day = cls._coerce_non_negative_int(game_time.get("day", 1), default=1) or 1
        hour = min(
            23,
            max(0, cls._coerce_non_negative_int(game_time.get("hour", 0), default=0)),
        )
        minute = min(
            59,
            max(0, cls._coerce_non_negative_int(game_time.get("minute", 0), default=0)),
        )
        return ((max(1, day) - 1) * 24 * 60) + (hour * 60) + minute

    @classmethod
    def _game_time_from_total_minutes(cls, total_minutes: int) -> dict[str, Any]:
        total = max(0, int(total_minutes))
        day = (total // (24 * 60)) + 1
        within = total % (24 * 60)
        hour = within // 60
        minute = within % 60
        period = cls._game_period_from_hour(hour)
        return {
            "day": day,
            "hour": hour,
            "minute": minute,
            "period": period,
            "date_label": f"Day {day}, {period.title()}",
        }

    @classmethod
    def _speed_multiplier_from_state(cls, campaign_state: dict[str, Any]) -> float:
        raw = campaign_state.get("speed_multiplier", 1.0) if isinstance(campaign_state, dict) else 1.0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 1.0
        if value <= 0:
            return 1.0
        return max(0.1, min(10.0, value))

    @classmethod
    def _estimate_turn_time_advance_minutes(cls, action_text: str, narration_text: str) -> int:
        action_l = str(action_text or "").lower()
        combined = f"{action_l}\n{str(narration_text or '').lower()}"
        if any(token in combined for token in ("time skip", "timeskip", "time-skip")):
            return 60
        if any(
            token in combined
            for token in ("sleep", "rest", "nap", "wait", "travel", "drive", "ride", "fly", "journey")
        ):
            return 30
        if any(token in combined for token in ("fight", "combat", "attack", "shoot", "chase", "run")):
            return 8
        if any(token in action_l for token in ("look", "examine", "inspect", "ask", "say", "talk")):
            return 3
        return cls.DEFAULT_TURN_ADVANCE_MINUTES

    @classmethod
    def _ensure_game_time_progress(
        cls,
        campaign_state: dict[str, Any],
        pre_turn_game_time: dict[str, int],
        *,
        action_text: str,
        narration_text: str,
    ) -> dict[str, Any]:
        if not isinstance(campaign_state, dict):
            return campaign_state
        pre_snapshot = (
            pre_turn_game_time
            if isinstance(pre_turn_game_time, dict)
            else cls._extract_game_time_snapshot(campaign_state)
        )
        cur_snapshot = cls._extract_game_time_snapshot(campaign_state)
        pre_total = cls._game_time_to_total_minutes(pre_snapshot)
        cur_total = cls._game_time_to_total_minutes(cur_snapshot)

        if cur_total > pre_total:
            campaign_state["game_time"] = cls._game_time_from_total_minutes(cur_total)
            return campaign_state
        if cls._is_ooc_action_text(action_text):
            campaign_state["game_time"] = cls._game_time_from_total_minutes(cur_total)
            return campaign_state

        base_minutes = cls._estimate_turn_time_advance_minutes(action_text, narration_text)
        speed_multiplier = cls._speed_multiplier_from_state(campaign_state)
        scaled = int(round(base_minutes * speed_multiplier))
        delta_minutes = max(cls.MIN_TURN_ADVANCE_MINUTES, scaled)
        delta_minutes = min(cls.MAX_TURN_ADVANCE_MINUTES, delta_minutes)
        new_total = max(pre_total, cur_total) + delta_minutes
        campaign_state["game_time"] = cls._game_time_from_total_minutes(new_total)
        cls._increment_auto_fix_counter(campaign_state, "game_time_auto_advance")
        return campaign_state

    @staticmethod
    def _normalize_location_text(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @classmethod
    def _resolve_player_location_for_state_sync(cls, player_state: dict[str, Any]) -> str:
        if not isinstance(player_state, dict):
            return ""
        for key in ("location", "room_title", "room_summary"):
            text = cls._normalize_location_text(player_state.get(key))
            if text:
                return text[:160]
        return ""

    @staticmethod
    def _entity_name_candidates_for_sync(state_key: object, entity_state: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        raw_name = ""
        if isinstance(entity_state, dict):
            raw_name = str(entity_state.get("name") or "").strip().lower()
        if raw_name:
            candidates.append(re.sub(r"\s+", " ", raw_name))
        key_text = re.sub(r"[_\-]+", " ", str(state_key or "").strip().lower())
        key_text = re.sub(r"\s+", " ", key_text).strip()
        if key_text:
            candidates.append(key_text)
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if len(candidate) < 3 or candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    @classmethod
    def _narration_implies_entity_with_player(cls, narration_text: str, name_candidates: list[str]) -> bool:
        text = str(narration_text or "").strip().lower()
        if not text or not name_candidates:
            return False
        cues = (
            "at your heels",
            "by your side",
            "beside you",
            "with you",
            "follows you",
            "following you",
            "trailing you",
            "walks with you",
            "stays close",
        )
        if not any(cue in text for cue in cues):
            return False
        for name in name_candidates:
            if re.search(rf"\b{re.escape(name)}\b", text):
                return True
        return False

    @classmethod
    def _narration_mentions_entity_in_active_scene(
        cls,
        narration_text: str,
        name_candidates: list[str],
    ) -> bool:
        text = str(narration_text or "").strip().lower()
        if not text or not name_candidates:
            return False
        remote_cues = (
            "sms",
            "text message",
            "calls you",
            "on the phone",
            "voicemail",
            "news feed",
            "on tv",
            "radio says",
            "video call",
        )
        if any(cue in text for cue in remote_cues):
            return False
        presence_cues = (
            "is here",
            "in the room",
            "across from you",
            "beside you",
            "nearby",
            "waits",
            "stands",
            "sits",
            "arrives",
            "at the desk",
            "at reception",
        )
        if not any(cue in text for cue in presence_cues):
            return False
        for name in name_candidates:
            if re.search(rf"\b{re.escape(name)}\b", text):
                return True
        return False

    @classmethod
    def _sync_active_player_character_location(
        cls,
        campaign_characters: dict[str, Any],
        *,
        player_state: dict[str, Any],
    ) -> int:
        if not isinstance(campaign_characters, dict):
            return 0
        player_location = cls._resolve_player_location_for_state_sync(player_state)
        if not player_location:
            return 0
        character_name = cls._normalize_location_text(player_state.get("character_name")).lower()
        if not character_name:
            return 0
        target_slug = cls._resolve_existing_character_slug(campaign_characters, character_name)
        if target_slug is None:
            for slug, entry in campaign_characters.items():
                if not isinstance(entry, dict):
                    continue
                entry_name = cls._normalize_location_text(entry.get("name")).lower()
                if entry_name and entry_name == character_name:
                    target_slug = slug
                    break
        if target_slug is None:
            return 0
        entry = campaign_characters.get(target_slug)
        if not isinstance(entry, dict):
            return 0
        current_location = cls._normalize_location_text(entry.get("location"))
        if current_location == player_location:
            return 0
        entry["location"] = player_location
        return 1

    @classmethod
    def _auto_sync_character_locations(
        cls,
        campaign_characters: dict[str, Any],
        *,
        player_state: dict[str, Any],
        narration_text: str,
    ) -> int:
        if not isinstance(campaign_characters, dict):
            return 0
        player_location = cls._resolve_player_location_for_state_sync(player_state)
        if not player_location:
            return 0
        changed = 0
        for slug, entry in campaign_characters.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("deceased_reason"):
                continue
            current_location = cls._normalize_location_text(entry.get("location"))
            if not current_location or current_location == player_location:
                continue
            names = cls._entity_name_candidates_for_sync(slug, entry)
            if not (
                cls._narration_implies_entity_with_player(narration_text, names)
                or cls._narration_mentions_entity_in_active_scene(narration_text, names)
            ):
                continue
            entry["location"] = player_location
            changed += 1
        return changed

    def _normalize_story_progress_update(
        self,
        campaign_state: dict[str, Any],
        state_update: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(state_update, dict):
            return {}
        if not isinstance(campaign_state, dict):
            return state_update

        outline = campaign_state.get("story_outline")
        chapters = outline.get("chapters") if isinstance(outline, dict) else []
        if not isinstance(chapters, list) or not chapters:
            return state_update

        out = dict(state_update)
        old_ch = self._coerce_non_negative_int(campaign_state.get("current_chapter", 0), default=0)
        old_ch = min(old_ch, len(chapters) - 1)

        has_ch = "current_chapter" in out
        has_sc = "current_scene" in out

        if has_ch:
            ch = self._coerce_non_negative_int(out.get("current_chapter"), default=old_ch)
            out["current_chapter"] = min(ch, len(chapters) - 1)

        if has_sc:
            scene_ch = out.get("current_chapter", old_ch)
            scene_ch = self._coerce_non_negative_int(scene_ch, default=old_ch)
            scene_ch = min(scene_ch, len(chapters) - 1)
            raw_scene = out.get("current_scene")
            sc = self._coerce_non_negative_int(raw_scene, default=0)
            chapter_scenes = chapters[scene_ch].get("scenes", [])
            if isinstance(chapter_scenes, list) and chapter_scenes:
                sc = min(sc, len(chapter_scenes) - 1)
            else:
                sc = 0
            out["current_scene"] = sc

        # Chapter transition defaults to first scene unless model explicitly sets one.
        if has_ch and not has_sc and out.get("current_chapter") != old_ch:
            out["current_scene"] = 0

        return out

    @staticmethod
    def _calendar_resolve_fire_point(
        current_day: int,
        current_hour: int,
        time_remaining: object,
        time_unit: object,
    ) -> tuple[int, int]:
        try:
            day = int(current_day)
        except (TypeError, ValueError):
            day = 1
        try:
            hour = int(current_hour)
        except (TypeError, ValueError):
            hour = 8
        day = max(1, day)
        hour = min(23, max(0, hour))
        try:
            remaining = int(time_remaining)
        except (TypeError, ValueError):
            remaining = 1
        unit = str(time_unit or "days").strip().lower()
        base_hours = (day - 1) * 24 + hour
        if unit.startswith("hour"):
            fire_abs_hours = base_hours + remaining
        else:
            fire_abs_hours = base_hours + (remaining * 24)
        fire_abs_hours = max(0, int(fire_abs_hours))
        fire_day = (fire_abs_hours // 24) + 1
        fire_hour = fire_abs_hours % 24
        return max(1, int(fire_day)), min(23, max(0, int(fire_hour)))

    @staticmethod
    def _calendar_resolve_fire_day(
        current_day: int,
        current_hour: int,
        time_remaining: object,
        time_unit: object,
    ) -> int:
        fire_day, _ = GameEngine._calendar_resolve_fire_point(
            current_day=current_day,
            current_hour=current_hour,
            time_remaining=time_remaining,
            time_unit=time_unit,
        )
        return fire_day

    @staticmethod
    def _calendar_fix_ampm(fire_hour: int, description: str) -> int:
        """Fix fire_hour when description contains AM/PM that contradicts it.

        Examples:
        - fire_hour=7, description mentions "7pm"  → returns 19
        - fire_hour=19, description mentions "7am" → returns 7
        - fire_hour=12, description mentions "12pm" (noon) → returns 12
        - fire_hour=0, description mentions "12am" (midnight) → returns 0
        """
        if not description:
            return fire_hour
        text = description.lower()
        # Look for patterns like "7pm", "7 pm", "7:00pm", "7:30 pm"
        for m in re.finditer(r"\b(\d{1,2})(?:\s*:\s*\d{2})?\s*(am|pm)\b", text):
            desc_hour = int(m.group(1))
            ampm = m.group(2)
            if desc_hour < 1 or desc_hour > 12:
                continue
            # Convert described hour to 24h
            if ampm == "pm":
                expected_24h = desc_hour if desc_hour == 12 else desc_hour + 12
            else:
                expected_24h = 0 if desc_hour == 12 else desc_hour
            # If fire_hour looks like the raw 12h value (not yet converted)
            if fire_hour == desc_hour and fire_hour != expected_24h:
                fire_hour = expected_24h
                break
        return fire_hour

    @staticmethod
    def _calendar_fix_relative_day(fire_day: int, description: str, current_day: int) -> int:
        """Fix obvious relative-day mismatches for 'tomorrow' and 'today' wording."""
        if not description:
            return fire_day
        text = description.lower()
        if re.search(r"\btomorrow\b", text):
            expected = current_day + 1
            # If fire_day is current_day or more than 1 day off, fix it
            if fire_day == current_day:
                return expected
        elif re.search(r"\btoday\b", text):
            if fire_day > current_day:
                return current_day
        return fire_day

    @staticmethod
    def _calendar_name_key(value: object) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "", text)

    @classmethod
    def _calendar_known_by_from_event(cls, event: Any) -> list[str]:
        if not isinstance(event, dict):
            return []
        raw_known_by = event.get("known_by")
        items: list[object]
        if isinstance(raw_known_by, list):
            items = raw_known_by
        elif isinstance(raw_known_by, str):
            if "," in raw_known_by:
                items = [chunk.strip() for chunk in raw_known_by.split(",")]
            else:
                items = [raw_known_by]
        else:
            items = []
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            name = str(item or "").strip()
            if not name:
                continue
            key = cls._calendar_name_key(name)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(name[:80])
            if len(out) >= 24:
                break
        return out

    @classmethod
    def _calendar_target_tokens_from_event(cls, event: Any) -> list[str]:
        if not isinstance(event, dict):
            return []
        raw_values: list[object] = []
        for key in (
            "target_players",
            "target_player",
            "targets",
            "target",
            "players",
            "player",
            "player_id",
            "user_id",
            "target_user_id",
            "target_user_ids",
            "who",
        ):
            raw_value = event.get(key)
            if isinstance(raw_value, list):
                raw_values.extend(raw_value)
            elif raw_value is not None:
                raw_values.append(raw_value)
        out: list[str] = []
        seen: set[str] = set()
        for item in raw_values:
            text = str(item or "").strip()
            if not text:
                continue
            key = re.sub(r"\s+", " ", text.lower())[:160]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(text[:160])
            if len(out) >= 12:
                break
        return out

    @classmethod
    def _calendar_normalize_event(
        cls,
        event: Any,
        *,
        current_day: int,
        current_hour: int,
    ) -> dict[str, Any] | None:
        if not isinstance(event, dict):
            return None
        name = str(event.get("name") or "").strip()
        if not name:
            return None
        fire_day_raw = event.get("fire_day")
        fire_hour_raw = event.get("fire_hour")
        if (
            isinstance(fire_day_raw, (int, float))
            and not isinstance(fire_day_raw, bool)
            and isinstance(fire_hour_raw, (int, float))
            and not isinstance(fire_hour_raw, bool)
        ):
            fire_day = max(1, int(fire_day_raw))
            fire_hour = min(23, max(0, int(fire_hour_raw)))
        elif isinstance(fire_day_raw, (int, float)) and not isinstance(
            fire_day_raw, bool
        ):
            fire_day = max(1, int(fire_day_raw))
            # Backward compatibility for legacy day-only events.
            fire_hour = 23
        else:
            fire_day, fire_hour = cls._calendar_resolve_fire_point(
                current_day=current_day,
                current_hour=current_hour,
                time_remaining=event.get("time_remaining", 1),
                time_unit=event.get("time_unit", "days"),
            )
        # Cross-check fire_hour against description AM/PM mentions.
        # If description says "7pm" but fire_hour is 7 (AM), fix it.
        description = str(event.get("description") or "")[:200]
        fire_hour = cls._calendar_fix_ampm(fire_hour, description)
        # Cross-check fire_day against "tomorrow"/"today" in description.
        fire_day = cls._calendar_fix_relative_day(fire_day, description, current_day)

        normalized: dict[str, Any] = {
            "name": name,
            "fire_day": fire_day,
            "fire_hour": fire_hour,
            "description": description,
            "known_by": cls._calendar_known_by_from_event(event),
        }
        target_players = cls._calendar_target_tokens_from_event(event)
        if target_players:
            normalized["target_players"] = target_players
        for key in ("created_day", "created_hour"):
            raw = event.get(key)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                normalized[key] = int(raw)
        for key in ("fired_notice_key", "fired_notice_day", "fired_notice_hour"):
            raw = event.get(key)
            if raw is None:
                continue
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                normalized[key] = int(raw)
            elif isinstance(raw, str):
                normalized[key] = raw[:160]
        return normalized

    def _apply_calendar_update(
        self,
        campaign_state: dict[str, Any],
        calendar_update: Any,
        resolution_context: str = "",
    ) -> dict[str, Any]:
        if not isinstance(calendar_update, dict):
            return campaign_state

        calendar_raw = list(campaign_state.get("calendar") or [])
        game_time = campaign_state.get("game_time") or {}
        current_day = game_time.get("day", 1)
        current_hour = game_time.get("hour", 8)
        day_int = int(current_day) if isinstance(current_day, (int, float)) else 1
        hour_int = int(current_hour) if isinstance(current_hour, (int, float)) else 8
        calendar: list[dict[str, Any]] = []
        for event in calendar_raw:
            normalized = self._calendar_normalize_event(
                event,
                current_day=day_int,
                current_hour=hour_int,
            )
            if normalized is not None:
                calendar.append(normalized)

        to_remove = calendar_update.get("remove")
        if isinstance(to_remove, list):
            remove_set = {str(name).strip().lower() for name in to_remove if name}
            context_text = " ".join(str(resolution_context or "").lower().split())
            allowed_remove_set: set[str] = set()
            blocked = 0
            for event in calendar:
                name_raw = str(event.get("name", "")).strip()
                if not name_raw:
                    continue
                key = name_raw.lower()
                if key not in remove_set:
                    continue
                name_norm = re.sub(r"[^a-z0-9]+", " ", key).strip()
                name_tokens = [t for t in name_norm.split() if len(t) > 2]
                name_mentioned = (
                    name_norm in context_text
                    or any(token in context_text for token in name_tokens)
                )
                completion_cues = (
                    "completed",
                    "finished",
                    "resolved",
                    "result delivered",
                    "results delivered",
                    "outcome delivered",
                    "concluded",
                    "cancelled",
                    "abandoned",
                    "closed out",
                    "already left",
                    "already departed",
                    "already en route",
                    "cleared from your schedule",
                    "off your schedule",
                )
                cleanup_cues = (
                    "remove from calendar",
                    "remove it from calendar",
                    "take it off the calendar",
                    "take it off calendar",
                    "clear it from the calendar",
                    "clear from your schedule",
                    "overdue",
                    "already",
                    "done",
                )
                premature_cues = (
                    "arrives",
                    "arrived",
                    "in progress",
                    "processing",
                    "pending",
                    "awaiting",
                    "sample",
                    "blood drawn",
                    "not back yet",
                )
                has_completion = any(cue in context_text for cue in completion_cues)
                has_cleanup_intent = any(cue in context_text for cue in cleanup_cues)
                has_premature = any(cue in context_text for cue in premature_cues)
                fire_day = event.get("fire_day")
                fire_hour = event.get("fire_hour")
                event_is_past = False
                if isinstance(fire_day, (int, float)) and isinstance(fire_hour, (int, float)):
                    fire_day_int = int(fire_day)
                    fire_hour_int = int(fire_hour)
                    event_is_past = (
                        fire_day_int < day_int
                        or (fire_day_int == day_int and fire_hour_int <= hour_int)
                    )
                if event_is_past or (
                    name_mentioned and not has_premature and (has_completion or has_cleanup_intent)
                ):
                    allowed_remove_set.add(key)
                else:
                    blocked += 1
            calendar = [
                event
                for event in calendar
                if str(event.get("name", "")).strip().lower() not in allowed_remove_set
            ]
            if blocked > 0:
                self._increment_auto_fix_counter(
                    campaign_state,
                    "calendar_remove_blocked",
                    amount=blocked,
                )

        to_add = calendar_update.get("add")
        if isinstance(to_add, list):
            for entry in to_add:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                if not name:
                    continue
                fire_day = entry.get("fire_day")
                fire_hour = entry.get("fire_hour")
                if (
                    isinstance(fire_day, (int, float))
                    and not isinstance(fire_day, bool)
                    and isinstance(fire_hour, (int, float))
                    and not isinstance(fire_hour, bool)
                ):
                    resolved_fire_day = max(1, int(fire_day))
                    resolved_fire_hour = min(23, max(0, int(fire_hour)))
                elif isinstance(fire_day, (int, float)) and not isinstance(
                    fire_day, bool
                ):
                    resolved_fire_day = max(1, int(fire_day))
                    resolved_fire_hour = 23
                else:
                    resolved_fire_day, resolved_fire_hour = self._calendar_resolve_fire_point(
                        current_day=day_int,
                        current_hour=hour_int,
                        time_remaining=entry.get("time_remaining", 1),
                        time_unit=entry.get("time_unit", "days"),
                    )
                event = {
                    "name": name,
                    "fire_day": resolved_fire_day,
                    "fire_hour": resolved_fire_hour,
                    "created_day": current_day,
                    "created_hour": current_hour,
                    "description": str(entry.get("description") or "")[:200],
                    "known_by": self._calendar_known_by_from_event(entry),
                }
                target_players = self._calendar_target_tokens_from_event(entry)
                if target_players:
                    event["target_players"] = target_players
                calendar.append(event)

        if isinstance(to_add, list):
            seen_names: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for event in reversed(calendar):
                key = str(event.get("name", "")).strip().lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
                deduped.append(event)
            calendar = list(reversed(deduped))

        if len(calendar) > 10:
            calendar = calendar[-10:]

        campaign_state["calendar"] = calendar
        return campaign_state
