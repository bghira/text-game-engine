from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from .errors import StaleClaimError, TurnBusyError
from .normalize import apply_patch, dump_json, normalize_give_item, parse_json_dict
from .ports import ActorResolverPort, LLMPort
from .types import ResolveTurnInput, ResolveTurnResult, RewindResult, TurnContext


class GameEngine:
    def __init__(
        self,
        uow_factory: Callable[[], Any],
        llm: LLMPort,
        actor_resolver: ActorResolverPort | None = None,
        clock: Callable[[], datetime] | None = None,
        lease_ttl_seconds: int = 90,
        max_conflict_retries: int = 1,
    ):
        self._uow_factory = uow_factory
        self._llm = llm
        self._actor_resolver = actor_resolver
        self._clock = clock or datetime.utcnow
        self._lease_ttl_seconds = lease_ttl_seconds
        self._max_conflict_retries = max_conflict_retries

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
            calendar_update = campaign_state_update.pop("calendar_update", None)
            campaign_state = apply_patch(campaign_state, campaign_state_update)
            campaign_state = self._apply_calendar_update(campaign_state, calendar_update)
            state_null_character_updates = self._character_updates_from_state_nulls(
                campaign_state_update,
                campaign_characters,
            )
            merged_character_updates = dict(state_null_character_updates)
            if isinstance(llm_output.character_updates, dict):
                merged_character_updates.update(llm_output.character_updates)
            campaign_characters = self._apply_character_updates(
                campaign_characters,
                merged_character_updates,
            )
            player_state = apply_patch(player_state, llm_output.player_state_update or {})
            post_turn_game_time = self._extract_game_time_snapshot(campaign_state)

            summary = campaign.summary or ""
            if isinstance(llm_output.summary_update, str) and llm_output.summary_update.strip():
                summary = (summary + "\n" + llm_output.summary_update.strip()).strip()

            narration = (llm_output.narration or "").strip() or "The world shifts, but nothing clear emerges."

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

            if turn_input.record_player_turn:
                uow.turns.add(
                    campaign_id=turn_input.campaign_id,
                    session_id=turn_input.session_id,
                    actor_id=turn_input.actor_id,
                    kind="player",
                    content=turn_input.action,
                    meta_json=dump_json({"game_time": pre_turn_game_time}),
                )
            narrator_turn = uow.turns.add(
                campaign_id=turn_input.campaign_id,
                session_id=turn_input.session_id,
                actor_id=turn_input.actor_id,
                kind="narrator",
                content=narration,
                meta_json=dump_json({"game_time": post_turn_game_time}),
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

    @classmethod
    def _apply_character_updates(
        cls,
        existing: dict[str, Any],
        updates: dict[str, Any],
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
            merged[target_slug or slug] = fields
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
        normalized: dict[str, Any] = {
            "name": name,
            "fire_day": fire_day,
            "fire_hour": fire_hour,
            "description": str(event.get("description") or "")[:200],
        }
        for key in ("created_day", "created_hour"):
            raw = event.get(key)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                normalized[key] = int(raw)
        return normalized

    def _apply_calendar_update(
        self,
        campaign_state: dict[str, Any],
        calendar_update: Any,
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
            calendar = [
                event
                for event in calendar
                if str(event.get("name", "")).strip().lower() not in remove_set
            ]

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
                }
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
