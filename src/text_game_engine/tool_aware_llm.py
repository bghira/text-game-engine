from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import Any, Callable
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)

from .core.ports import ProgressCallback
from .core.types import GiveItemInstruction, LLMTurnOutput, TimerInstruction
from .persistence.sqlalchemy.models import Campaign, Player, Turn
from .zork_emulator import ZorkEmulator

_YOUTUBE_ALLOWED_HOSTS = frozenset({
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
})


def _is_allowed_youtube_url(text: str) -> bool:
    parsed = urlparse(str(text or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    host = str(parsed.netloc or "").strip().lower()
    if not host:
        return False
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    return host in _YOUTUBE_ALLOWED_HOSTS


def _neutralize_discord_mentions(text: object) -> str:
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return ""
    value = value.replace("@everyone", "@\u200beveryone")
    value = value.replace("@here", "@\u200bhere")
    value = re.sub(
        r"<@([!&]?\d+)>",
        lambda match: f"<@\u200b{match.group(1)}>",
        value,
    )
    return value


def _strip_reasoning_from_recent_turn_jsonl(text: object) -> str:
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            lines.append(line)
            continue
        if not isinstance(payload, dict):
            lines.append(line)
            continue
        payload.pop("reasoning", None)
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines).strip()


def _search_youtube_first_result(query: str) -> dict[str, str] | None:
    text = " ".join(str(query or "").split()).strip()
    if not text:
        return None
    parsed = urlparse(text)
    if _is_allowed_youtube_url(text):
        return {
            "title": text,
            "url": text,
            "channel": "",
        }
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return {
            "title": text,
            "url": f"https://www.youtube.com/results?search_query={quote_plus(text)}",
            "channel": "",
        }
    try:
        import yt_dlp
    except ImportError:
        logger.warning(
            "yt_dlp is not installed; YouTube search functionality is unavailable."
        )
        return {
            "title": text,
            "url": f"https://www.youtube.com/results?search_query={quote_plus(text)}",
            "channel": "",
        }

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "default_search": "ytsearch1",
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{text}", download=False)
    except Exception:
        return {
            "title": text,
            "url": f"https://www.youtube.com/results?search_query={quote_plus(text)}",
            "channel": "",
        }
    if not isinstance(info, dict):
        return None
    entries = info.get("entries")
    entry = None
    if isinstance(entries, list) and entries:
        first = entries[0]
        if isinstance(first, dict):
            entry = first
    if entry is None:
        entry = info if isinstance(info, dict) else None
    if not isinstance(entry, dict):
        return None
    url = str(entry.get("webpage_url") or entry.get("url") or "").strip()
    if url and not url.startswith("http"):
        video_id = str(entry.get("id") or "").strip()
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
    title = " ".join(str(entry.get("title") or "").split()).strip()
    channel = " ".join(
        str(
            entry.get("channel")
            or entry.get("uploader")
            or entry.get("channel_name")
            or ""
        ).split()
    ).strip()
    if not url:
        return {
            "title": title or text,
            "url": f"https://www.youtube.com/results?search_query={quote_plus(text)}",
            "channel": channel,
        }
    return {
        "title": title or text,
        "url": url,
        "channel": channel,
    }


async def _notify_progress(
    callback: ProgressCallback | None,
    phase: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Best-effort progress notification — never raises."""
    if callback is None:
        return
    try:
        maybe = callback(phase, metadata)
        if asyncio.iscoroutine(maybe):
            await maybe
    except Exception:
        return


class DeterministicLLM:
    """Simple fallback LLM for bring-up and failure paths."""

    @staticmethod
    def _speed_multiplier(campaign_state: dict[str, Any] | None) -> float:
        raw = campaign_state.get("speed_multiplier", 1.0) if isinstance(campaign_state, dict) else 1.0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 1.0
        if value <= 0:
            return 1.0
        return max(0.1, min(10.0, value))

    @staticmethod
    def _advance_time(
        game_time: dict[str, Any],
        *,
        speed_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        day = int(game_time.get("day", 1) or 1)
        hour = int(game_time.get("hour", 8) or 8)
        minute = int(game_time.get("minute", 0) or 0)

        base_minutes = int(getattr(ZorkEmulator, "DEFAULT_TURN_ADVANCE_MINUTES", 20) or 20)
        min_minutes = int(getattr(ZorkEmulator, "MIN_TURN_ADVANCE_MINUTES", base_minutes) or base_minutes)
        scaled_min_minutes = max(1, int(round(min_minutes * float(speed_multiplier or 1.0))))
        scaled_minutes = int(round(base_minutes * float(speed_multiplier or 1.0)))
        minute += max(scaled_min_minutes, scaled_minutes)
        if minute >= 60:
            hour += minute // 60
            minute = minute % 60
        if hour >= 24:
            day += hour // 24
            hour = hour % 24

        if hour < 12:
            period = "morning"
        elif hour < 17:
            period = "afternoon"
        elif hour < 21:
            period = "evening"
        else:
            period = "night"

        return {
            "day": day,
            "hour": hour,
            "minute": minute,
            "period": period,
            "date_label": f"Day {day}, {period.title()}",
        }

    async def complete_turn(self, context, *, progress: ProgressCallback | None = None) -> LLMTurnOutput:
        action = (context.action or "").strip()
        lowered = action.lower()
        campaign_state = context.campaign_state if isinstance(context.campaign_state, dict) else {}
        player_state = context.player_state if isinstance(getattr(context, "player_state", None), dict) else {}
        time_model = str(campaign_state.get("time_model") or "").strip().lower()
        if time_model == ZorkEmulator.TIME_MODEL_INDIVIDUAL_CLOCKS:
            current_time = (
                player_state.get("game_time")
                if isinstance(player_state.get("game_time"), dict)
                else campaign_state.get("game_time", {})
            )
        else:
            current_time = campaign_state.get("game_time", {})
        speed_multiplier = self._speed_multiplier(
            campaign_state
        )
        next_time = self._advance_time(
            current_time if isinstance(current_time, dict) else {},
            speed_multiplier=speed_multiplier,
        )

        if any(token in lowered for token in ["look", "scan", "observe", "examine"]):
            return LLMTurnOutput(
                narration="You pause and take stock. The room is quiet, details sharp at the edges.",
                state_update={"game_time": next_time},
                summary_update="You took a careful look around.",
                xp_awarded=1,
                player_state_update={
                    "room_title": "Holding Room",
                    "room_summary": "A functional room with steel desk, chair, and one narrow door.",
                    "location": "holding-room",
                    "exits": ["north door", "desk drawer"],
                },
                scene_image_prompt=(
                    "Sparse holding room, steel desk under fluorescent lighting, muted colors, "
                    "narrow door with worn paint, realistic perspective"
                ),
            )

        if any(token in lowered for token in ["wait", "rest", "sleep"]):
            return LLMTurnOutput(
                narration="Time passes in measured silence.",
                state_update={"game_time": next_time},
                summary_update="You waited and let the scene breathe.",
                xp_awarded=0,
                player_state_update={
                    "room_summary": "Still in the holding room. Nothing immediate changes."
                },
            )

        return LLMTurnOutput(
            narration=f"You {action}. The world answers with small, immediate consequences.",
            state_update={"game_time": next_time},
            summary_update=f"Action resolved: {action}",
            xp_awarded=1,
            player_state_update={"room_summary": "You remain in control of the next move."},
        )


class ToolAwareZorkLLM:
    """Model adapter that reuses ZorkEmulator prompt + tool call semantics."""

    AUTO_FIX_COUNTERS_KEY = "_auto_fix_counters"

    def __init__(
        self,
        *,
        session_factory,
        completion_port,
        temperature: float,
        max_tokens: int,
        max_tool_rounds: int = 4,
        turn_visibility_default_resolver: (
            Callable[[str | None, str | None], str] | None
        ) = None,
    ) -> None:
        self._session_factory = session_factory
        self._completion = completion_port
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._max_tool_rounds = int(max_tool_rounds)
        self._turn_visibility_default_resolver = turn_visibility_default_resolver
        self._fallback = DeterministicLLM()
        self._emulator: ZorkEmulator | None = None
        self._log_callback: Callable[[str, str], None] | None = None

    def bind_emulator(self, emulator: ZorkEmulator) -> None:
        self._emulator = emulator

    def set_log_callback(self, callback: Callable[[str, str], None] | None) -> None:
        """Set a ``(section, body) → None`` callback for request/response logging."""
        self._log_callback = callback

    def _zork_log(self, section: str, body: str = "") -> None:
        cb = self._log_callback
        if cb is not None:
            try:
                cb(section, body)
            except Exception:
                pass

    @staticmethod
    def _parse_json(text: str | None, default: Any) -> Any:
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            return default

    @staticmethod
    def _memory_tool_text_value(text: object, max_chars: int = 4000) -> str:
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(normalized) > max_chars:
            normalized = normalized[: max_chars - 3].rstrip() + "..."
        return normalized

    @staticmethod
    def _prune_recent_turn_tool_row(row: object) -> dict[str, Any]:
        if not isinstance(row, dict):
            return {}
        cleaned: dict[str, Any] = {}
        for key, value in row.items():
            if str(key) == "index":
                continue
            normalized = value
            if isinstance(normalized, str):
                normalized = normalized.strip()
                if key == "visibility" and normalized.lower() == "local":
                    continue
            elif isinstance(normalized, list):
                normalized = [
                    item.strip() if isinstance(item, str) else item
                    for item in normalized
                    if (item.strip() if isinstance(item, str) else item) not in ("", None)
                ]
            elif isinstance(normalized, dict):
                normalized = {
                    inner_key: inner_value
                    for inner_key, inner_value in normalized.items()
                    if inner_value not in ("", None, [], {})
                }
            if normalized in ("", None, [], {}):
                continue
            cleaned[str(key)] = normalized
        return cleaned

    @staticmethod
    def _memory_tool_jsonl(records: list[dict[str, Any]]) -> str:
        return "\n".join(
            json.dumps(
                row,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            for row in records
            if isinstance(row, dict)
        )

    @staticmethod
    def _memory_tool_bool_value(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = " ".join(str(value or "").strip().lower().split())
        return text in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _memory_tool_turn_id_list(
        value: object,
        *,
        limit: int = 24,
    ) -> list[int]:
        if isinstance(value, int):
            raw_values = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        elif isinstance(value, str) and value.strip():
            raw_values = re.findall(r"\d+", value)
        else:
            raw_values = []
        out: list[int] = []
        seen: set[int] = set()
        for raw in raw_values:
            try:
                turn_id = int(raw)
            except Exception:
                continue
            if turn_id <= 0 or turn_id in seen:
                continue
            seen.add(turn_id)
            out.append(turn_id)
            if len(out) >= max(1, int(limit)):
                break
        return out

    @classmethod
    def _memory_tool_records_from_text(cls, text: object) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if not stripped or not stripped.startswith("{"):
                continue
            try:
                row = json.loads(stripped)
            except Exception:
                continue
            if isinstance(row, dict):
                records.append(row)
        return records

    @classmethod
    def _memory_tool_turn_ids_from_text(cls, text: object) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for row in cls._memory_tool_records_from_text(text):
            row_kind = str(row.get("kind") or "").strip().lower()
            row_turn_id = 0
            if row_kind == "memory_turn_result":
                try:
                    row_turn_id = int(row.get("turn_id") or 0)
                except Exception:
                    row_turn_id = 0
            elif (
                row_kind == "memory_hit"
                and str(row.get("memory_type") or "").strip().lower() == "turn"
            ):
                try:
                    row_turn_id = int(row.get("turn_id") or 0)
                except Exception:
                    row_turn_id = 0
            if row_turn_id <= 0 or row_turn_id in seen:
                continue
            seen.add(row_turn_id)
            out.append(row_turn_id)
        return out

    def _memory_tool_excluded_turn_ids_from_recent_context(
        self,
        campaign_id: str,
        actor_id: str | None,
    ) -> set[int]:
        emulator = self._emulator
        actor_id_text = str(actor_id or "").strip()
        if emulator is None or not actor_id_text:
            return set()
        with emulator._session_factory() as session:  # noqa: SLF001
            campaign = session.get(Campaign, str(campaign_id))
            actor_row = (
                session.query(Player)
                .filter(Player.campaign_id == str(campaign_id))
                .filter(Player.actor_id == actor_id_text)
                .first()
            )
        if campaign is None or actor_row is None:
            return set()
        actor_state = self._parse_json(getattr(actor_row, "state_json", "") or "{}", {})
        registry = emulator._campaign_player_registry(str(campaign_id), emulator._session_factory)  # noqa: SLF001
        viewer_slug = str(
            (registry.get("by_actor_id", {}).get(actor_id_text) or {}).get("slug")
            or emulator._player_visibility_slug(actor_id_text)  # noqa: SLF001
        ).strip()
        viewer_location_key = emulator._room_key_from_player_state(actor_state)  # noqa: SLF001
        viewer_private_context = emulator._active_private_context_from_state(actor_state)  # noqa: SLF001
        viewer_private_context_key = str(
            (viewer_private_context or {}).get("context_key") or ""
        ).strip()
        recent_turn_limit = max(int(getattr(emulator, "MAX_RECENT_TURNS", 24) or 24), 1)
        turns = emulator.get_recent_turns(str(campaign_id), limit=recent_turn_limit)
        excluded: set[int] = set()
        for turn in turns:
            try:
                turn_id = int(getattr(turn, "id", 0) or 0)
            except Exception:
                turn_id = 0
            if turn_id <= 0 or str(getattr(turn, "kind", "") or "").strip().lower() != "narrator":
                continue
            if emulator._turn_visible_in_recent_turns_context(  # noqa: SLF001
                turn,
                viewer_actor_id=actor_id_text,
                viewer_slug=viewer_slug,
                viewer_location_key=viewer_location_key,
                viewer_private_context_key=viewer_private_context_key,
            ):
                excluded.add(turn_id)
        return excluded

    @classmethod
    def _pruned_memory_tool_text(
        cls,
        tool_name: str,
        text: object,
        keep_turn_ids: set[int],
    ) -> str:
        tool_key = str(tool_name or "").strip().lower()
        if tool_key not in {"memory_search", "memory_turn"}:
            return str(text or "")
        if not keep_turn_ids:
            return ""
        kept_rows: list[dict[str, Any]] = []
        for row in cls._memory_tool_records_from_text(text):
            row_kind = str(row.get("kind") or "").strip().lower()
            row_turn_id = 0
            if row_kind == "memory_turn_result":
                try:
                    row_turn_id = int(row.get("turn_id") or 0)
                except Exception:
                    row_turn_id = 0
            elif (
                row_kind == "memory_hit"
                and str(row.get("memory_type") or "").strip().lower() == "turn"
            ):
                try:
                    row_turn_id = int(row.get("turn_id") or 0)
                except Exception:
                    row_turn_id = 0
            if row_turn_id > 0 and row_turn_id in keep_turn_ids:
                kept_rows.append(row)
        if not kept_rows:
            return ""
        actual_turn_ids: set[int] = set()
        for row in kept_rows:
            try:
                tid = int(row.get("turn_id") or 0)
            except Exception:
                tid = 0
            if tid > 0:
                actual_turn_ids.add(tid)
        retained_meta = {
            "kind": "memory_context_retained",
            "retained_turn_ids": sorted(actual_turn_ids),
            "retained_count": len(kept_rows),
        }
        prefix = "MEMORY_TURN_RESULT:" if tool_key == "memory_turn" else "MEMORY_RECALL:"
        return prefix + "\n" + cls._memory_tool_jsonl([retained_meta, *kept_rows])

    @staticmethod
    def _tool_call_signature(payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        try:
            return json.dumps(
                payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
            )
        except Exception:
            return str(payload)

    def _resolve_turn_visibility_default(
        self,
        session_id: str | None,
        *,
        campaign_id: str | None = None,
    ) -> str:
        if self._turn_visibility_default_resolver is None:
            return "public"
        try:
            value = self._turn_visibility_default_resolver(session_id, campaign_id)
        except Exception:
            return "public"
        text = str(value or "").strip().lower()
        if text in {"public", "private", "limited", "local"}:
            return "private" if text == "limited" else text
        return "public"

    def _bump_auto_fix_counter(
        self,
        campaign_id: str,
        key: str,
        amount: int = 1,
    ) -> None:
        safe_key = re.sub(r"[^a-z0-9_]+", "_", str(key or "").strip().lower()).strip(
            "_"
        )
        if not safe_key:
            return
        try:
            safe_amount = max(1, int(amount))
        except Exception:
            safe_amount = 1
        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return
            state = self._parse_json(campaign.state_json, {})
            if not isinstance(state, dict):
                state = {}
            counters = state.get(self.AUTO_FIX_COUNTERS_KEY)
            if not isinstance(counters, dict):
                counters = {}
                state[self.AUTO_FIX_COUNTERS_KEY] = counters
            try:
                current = max(0, int(counters.get(safe_key, 0) or 0))
            except Exception:
                current = 0
            counters[safe_key] = min(10**9, current + safe_amount)
            campaign.state_json = json.dumps(state, ensure_ascii=True)
            campaign.updated_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()

    @staticmethod
    def _should_force_auto_memory_search(action_text: str) -> bool:
        if re.match(r"\s*\[OOC\b", str(action_text or ""), re.IGNORECASE):
            return False
        text = " ".join(str(action_text or "").strip().lower().split())
        if not text or text.startswith("!") or len(text) < 6:
            return False
        trivial = {
            "look",
            "l",
            "inventory",
            "inv",
            "i",
            "map",
            "yes",
            "y",
            "no",
            "n",
            "ok",
            "okay",
            "thanks",
            "thank you",
        }
        return text not in trivial

    def _derive_auto_memory_queries(
        self,
        campaign_id: str,
        actor_id: str,
        action_text: str,
        limit: int = 4,
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()

        def _push(raw: object) -> None:
            text = " ".join(str(raw or "").strip().split())
            if not text:
                return
            key = text.lower()
            if key in seen:
                return
            seen.add(key)
            out.append(text[:120])

        with self._session_factory() as session:
            player = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == actor_id)
                .first()
            )
            pstate = self._parse_json(
                player.state_json if player is not None else "{}", {}
            )
            if isinstance(pstate, dict):
                _push(pstate.get("location"))
                _push(pstate.get("room_title"))
                player_name = " ".join(
                    str(pstate.get("character_name") or "").strip().lower().split()
                )
            else:
                player_name = ""

            others = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .order_by(Player.actor_id.asc())
                .all()
            )
            for row in others[:6]:
                ostate = self._parse_json(row.state_json or "{}", {})
                if not isinstance(ostate, dict):
                    continue
                name = " ".join(str(ostate.get("character_name") or "").strip().split())
                if not name:
                    continue
                if name.lower() == player_name:
                    continue
                _push(name)
                if len(out) >= limit:
                    break

        _push(action_text)
        return out[: max(1, int(limit or 4))]

    @staticmethod
    def _is_emptyish_payload(payload: dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return True
        narration = " ".join(str(payload.get("narration") or "").strip().lower().split())
        trivial_narration = narration in {
            "",
            "the world shifts, but nothing clear emerges.",
            "a hollow silence answers. try again.",
            "a hollow silence answers.",
        }
        short_narration = len(narration) < 24
        state_update = payload.get("state_update")
        player_state_update = payload.get("player_state_update")
        other_player_state_updates = payload.get("other_player_state_updates")
        summary_update = payload.get("summary_update")
        character_updates = payload.get("character_updates")
        location_updates = payload.get("location_updates")
        calendar_update = payload.get("calendar_update")
        scene_image_prompt = payload.get("scene_image_prompt")
        scene_output = payload.get("scene_output")
        xp_awarded = payload.get("xp_awarded", 0)
        has_scene_output = (
            isinstance(scene_output, dict)
            and isinstance(scene_output.get("beats"), list)
            and any(
                isinstance(beat, dict) and str(beat.get("text") or "").strip()
                for beat in scene_output.get("beats")
            )
        )
        has_signal = (
            bool(state_update)
            or bool(player_state_update)
            or bool(other_player_state_updates)
            or bool(character_updates)
            or bool(location_updates)
            or bool(calendar_update)
        )
        has_signal = has_signal or has_scene_output
        has_signal = has_signal or bool(str(summary_update or "").strip()) or bool(
            str(scene_image_prompt or "").strip()
        )
        try:
            has_signal = has_signal or int(xp_awarded or 0) > 0
        except Exception:
            pass
        if trivial_narration and not has_signal:
            return True
        if short_narration and not has_signal:
            return True
        return False

    @staticmethod
    def _looks_like_major_narrative_beat(payload: dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        narration = " ".join(str(payload.get("narration") or "").lower().split())
        summary = " ".join(str(payload.get("summary_update") or "").lower().split())
        text = f"{narration} {summary}".strip()
        cues = (
            "reveals",
            "reveal",
            "confirms",
            "confirmed",
            "pregnant",
            "paternity",
            "dies",
            "dead",
            "betray",
            "arrest",
            "results",
            "test result",
            "identity",
            "truth",
            "confession",
            "escape",
            "ambush",
        )
        if any(cue in text for cue in cues):
            return True
        if isinstance(payload.get("character_updates"), dict):
            for row in payload.get("character_updates", {}).values():
                if isinstance(row, dict) and str(row.get("deceased_reason") or "").strip():
                    return True
        if isinstance(payload.get("other_player_state_updates"), dict):
            for row in payload.get("other_player_state_updates", {}).values():
                if isinstance(row, dict) and str(row.get("deceased_reason") or "").strip():
                    return True
        if isinstance(payload.get("calendar_update"), dict):
            cal = payload.get("calendar_update") or {}
            if isinstance(cal.get("add"), list) or isinstance(cal.get("remove"), list):
                return True
        if isinstance(payload.get("state_update"), dict):
            if "current_chapter" in payload.get("state_update", {}) or "current_scene" in payload.get(
                "state_update", {}
            ):
                return True
        return False

    @staticmethod
    def _action_requests_clock_time(action_text: str) -> bool:
        text = " ".join(str(action_text or "").strip().lower().split())
        if not text:
            return False
        return any(
            token in text
            for token in (
                "what time",
                "current time",
                "check time",
                "clock",
                "time is it",
            )
        )

    @staticmethod
    def _narration_has_explicit_clock_time(narration_text: str) -> bool:
        return bool(
            re.search(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", str(narration_text or ""))
        )

    def _fallback_memory_state(self, campaign_id: str) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return []
            state = self._parse_json(campaign.state_json, {})
        rows = state.get("_webui_curated_memory", []) if isinstance(state, dict) else []
        return rows if isinstance(rows, list) else []

    def _persist_fallback_memory_state(
        self,
        campaign_id: str,
        entries: list[dict[str, Any]],
    ) -> None:
        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return
            state = self._parse_json(campaign.state_json, {})
            if not isinstance(state, dict):
                state = {}
            state["_webui_curated_memory"] = entries[-500:]
            campaign.state_json = json.dumps(state, ensure_ascii=True)
            campaign.updated_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()

    def _parse_model_payload(self, response: str | None) -> dict[str, Any] | None:
        emulator = self._emulator
        if emulator is None:
            logger.warning("_parse_model_payload: emulator is None")
            return None
        if not response:
            logger.warning("_parse_model_payload: empty response from LLM")
            return None
        cleaned = emulator._clean_response(response)  # noqa: SLF001
        json_text = emulator._extract_json(cleaned)  # noqa: SLF001
        logger.debug(
            "_parse_model_payload: cleaned_len=%d json_text_len=%d has_fences=%s",
            len(cleaned or ""),
            len(json_text or ""),
            "```" in (response or ""),
        )
        try:
            payload = emulator._parse_json_lenient(json_text or cleaned)  # noqa: SLF001
        except Exception as exc:
            inferred = self._plaintext_search_intent_to_payload(cleaned or response or "")
            if inferred is not None:
                logger.warning(
                    "_parse_model_payload: recovered plain-text search intent as tool_call=%s queries=%s",
                    inferred.get("tool_call"),
                    inferred.get("queries"),
                )
                self._zork_log(
                    "PARSE RECOVERY",
                    f"mode=plain_text_search_intent\n"
                    f"raw_response=\n{response or '(empty)'}\n"
                    f"cleaned_response=\n{cleaned or '(empty)'}\n"
                    f"inferred_payload=\n{json.dumps(inferred, ensure_ascii=True)}",
                )
                return inferred
            logger.warning(
                "_parse_model_payload: JSON parse failed: %s\n"
                "RAW RESPONSE:\n%s\n"
                "CLEANED RESPONSE:\n%s\n"
                "EXTRACTED JSON TEXT:\n%s",
                exc,
                response or "(empty)",
                cleaned or "(empty)",
                json_text or "(empty)",
            )
            self._zork_log(
                "PARSE CHAIN DETAIL",
                f"response_len={len(response or '')}\n"
                f"cleaned_len={len(cleaned or '')}\n"
                f"json_text_len={len(json_text or '')}\n"
                f"had_fences={'```' in (response or '')}\n"
                f"raw_response=\n{response or '(empty)'}\n"
                f"cleaned_response=\n{cleaned or '(empty)'}\n"
                f"json_text=\n{json_text or '(empty)'}",
            )
            return None
        if not isinstance(payload, dict):
            logger.warning(
                "_parse_model_payload: parsed payload is %s, not dict",
                type(payload).__name__,
            )
            return None
        is_tool = emulator._is_tool_call(payload)  # noqa: SLF001
        logger.debug(
            "_parse_model_payload: success keys=%s is_tool_call=%s narration_len=%d",
            list(payload.keys())[:10],
            is_tool,
            len(str(payload.get("narration") or "")),
        )
        return payload

    @staticmethod
    def _plaintext_search_intent_to_payload(text: str | None) -> dict[str, Any] | None:
        raw = " ".join(str(text or "").strip().split())
        if not raw or "{" in raw or "}" in raw:
            return None
        lower = raw.lower()
        planning_markers = (
            "i need to",
            "we need to",
            "i should",
            "let me",
            "i will",
            "i'll",
        )
        search_markers = (
            "search",
            "look up",
            "find more context",
            "find context",
            "check context",
        )
        if not any(marker in lower for marker in planning_markers):
            return None
        if not any(marker in lower for marker in search_markers):
            return None

        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", raw) if part.strip()]
        candidate = ""
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if "context about" in sentence_lower or "context for" in sentence_lower:
                candidate = sentence
                break
            if any(marker in sentence_lower for marker in ("look up", "find more context", "find context")):
                candidate = sentence
                break
        if not candidate and sentences:
            candidate = sentences[0]
        query = candidate.strip().rstrip(".!?")
        query = re.sub(r"(?i)^(?:i|we)\s+(?:need|want|have)\s+to\s+", "", query).strip()
        query = re.sub(r"(?i)^i\s+should\s+", "", query).strip()
        query = re.sub(r"(?i)^let\s+me\s+", "", query).strip()
        query = re.sub(r"(?i)^i\s+will\s+", "", query).strip()
        query = re.sub(r"(?i)^i['’]ll\s+", "", query).strip()
        query = re.sub(
            r"(?i)^(?:find|search|look up|check)(?:\s+for)?\s+(?:more\s+)?context\s+(?:about|for|on)\s+",
            "",
            query,
        ).strip()
        query = re.sub(r"(?i)^(?:find|search|look up|check)(?:\s+for)?\s+", "", query).strip()
        query = query.strip(" -:;,")
        if not query:
            return None
        if query.lower() in {"that", "this", "it", "them", "him", "her"}:
            return None
        return {"tool_call": "memory_search", "queries": [query[:240]]}

    def _payload_to_output(
        self,
        payload: dict[str, Any],
        *,
        actor_id: str | None = None,
        character_name: str | None = None,
    ) -> LLMTurnOutput:
        emulator = self._emulator
        if emulator is None:
            return LLMTurnOutput(narration="")

        narration = payload.get("narration")
        if not isinstance(narration, str):
            narration = ""

        raw_scene_output = payload.get("scene_output")
        scene_output_text_fallback = ""
        if isinstance(raw_scene_output, str):
            scene_output_text_fallback = raw_scene_output.strip()
            scene_output = None
        elif isinstance(raw_scene_output, dict):
            scene_output = raw_scene_output
        else:
            scene_output = None
        if not narration and scene_output_text_fallback:
            narration = scene_output_text_fallback
        if not narration and scene_output is not None:
            try:
                rendered = emulator._scene_output_rendered_text(scene_output)  # noqa: SLF001
            except Exception:
                rendered = ""
            if isinstance(rendered, str) and rendered.strip():
                narration = rendered.strip()

        state_update = payload.get("state_update")
        if not isinstance(state_update, dict):
            state_update = {}

        player_state_update = payload.get("player_state_update")
        if not isinstance(player_state_update, dict):
            player_state_update = {}

        other_player_state_updates = payload.get("other_player_state_updates")
        if isinstance(other_player_state_updates, dict):
            normalized_other_player_state_updates: dict[str, dict[str, Any]] = {}
            actor_visibility_slug = (
                emulator._player_visibility_slug(actor_id) if actor_id else ""
            )  # noqa: SLF001
            actor_name_slug = ""
            if actor_id and character_name:
                actor_name_slug = emulator._player_slug_key(  # noqa: SLF001
                    character_name
                )
            for raw_slug, raw_update in other_player_state_updates.items():
                slug = str(raw_slug or "").strip().lower()
                if not slug or slug in {actor_visibility_slug, actor_name_slug}:
                    continue
                if not isinstance(raw_update, dict):
                    continue
                normalized_other_player_state_updates[slug] = dict(raw_update)
            other_player_state_updates = normalized_other_player_state_updates
        else:
            other_player_state_updates = {}

        co_located_player_slugs = payload.get("co_located_player_slugs")
        if isinstance(co_located_player_slugs, list):
            normalized_co_located_player_slugs: list[str] = []
            seen_co_located_player_slugs: set[str] = set()
            actor_visibility_slug = (
                emulator._player_visibility_slug(actor_id) if actor_id else ""
            )  # noqa: SLF001
            for item in co_located_player_slugs:
                slug = str(item or "").strip().lower()
                if not slug or slug == actor_visibility_slug or slug in seen_co_located_player_slugs:
                    continue
                seen_co_located_player_slugs.add(slug)
                normalized_co_located_player_slugs.append(slug)
            co_located_player_slugs = normalized_co_located_player_slugs
        else:
            co_located_player_slugs = []

        turn_visibility = payload.get("turn_visibility")
        if not isinstance(turn_visibility, dict):
            turn_visibility = None

        state_update, player_state_update = emulator._split_room_state(  # noqa: SLF001
            state_update, player_state_update
        )
        game_time_update = state_update.get("game_time")
        if isinstance(game_time_update, dict) and game_time_update:
            player_state_update.setdefault("game_time", game_time_update)

        try:
            state_update, extracted_calendar_update = emulator._extract_calendar_update_from_state_update(  # noqa: SLF001
                state_update,
                payload.get("calendar_update"),
            )
        except Exception:
            extracted_calendar_update = payload.get("calendar_update")
        if isinstance(extracted_calendar_update, dict):
            add_rows = extracted_calendar_update.get("add")
            if isinstance(add_rows, list):
                for row in add_rows:
                    if not isinstance(row, dict):
                        continue
                    visibility = str(row.get("visibility") or "").strip().lower()
                    target_players = row.get("target_players")
                    target_player = row.get("target_player")
                    participants = row.get("participants")
                    participant = row.get("participant")
                    has_participants = bool(participant) or (
                        isinstance(participants, list) and bool(participants)
                    )
                    has_targets = bool(target_player) or (
                        isinstance(target_players, list) and bool(target_players)
                    )
                    if (visibility == "private" or has_participants) and not has_targets and actor_id:
                        row["target_players"] = [str(actor_id)]
            state_update["calendar_update"] = extracted_calendar_update

        summary_update = payload.get("summary_update")
        if not isinstance(summary_update, str) or not summary_update.strip():
            summary_update = None
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, str):
            reasoning = " ".join(reasoning.strip().split())[:1200] or None
        else:
            reasoning = None

        xp_awarded_raw = payload.get("xp_awarded", 0)
        try:
            xp_awarded = max(0, min(10, int(xp_awarded_raw)))
        except Exception:
            xp_awarded = 0

        scene_image_prompt = payload.get("scene_image_prompt")
        if not isinstance(scene_image_prompt, str) or not scene_image_prompt.strip():
            image_prompt = payload.get("image_prompt")
            if isinstance(image_prompt, str) and image_prompt.strip():
                scene_image_prompt = image_prompt
            else:
                scene_image_prompt = None

        timer_instruction = None
        timer_delay = payload.get("set_timer_delay")
        timer_event = payload.get("set_timer_event")
        if (
            isinstance(timer_delay, (int, float))
            and isinstance(timer_event, str)
            and timer_event.strip()
        ):
            timer_interruptible = payload.get("set_timer_interruptible", True)
            timer_interrupt_action = payload.get("set_timer_interrupt_action")
            timer_scope = payload.get("set_timer_interrupt_scope", "global")
            if (
                not isinstance(timer_scope, str)
                or timer_scope.strip().lower() not in {"local", "global"}
            ):
                timer_scope = "global"
            timer_instruction = TimerInstruction(
                delay_seconds=max(1, int(timer_delay)),
                event_text=timer_event.strip(),
                interruptible=bool(timer_interruptible),
                interrupt_action=(
                    str(timer_interrupt_action).strip()
                    if timer_interrupt_action is not None
                    else None
                ),
                interrupt_scope=timer_scope,
            )

        give_item = None
        give_item_payload = payload.get("give_item")
        if isinstance(give_item_payload, dict):
            item = give_item_payload.get("item")
            if isinstance(item, str) and item.strip():
                to_actor_id = give_item_payload.get("to_actor_id")
                to_discord_mention = give_item_payload.get("to_discord_mention")
                give_item = GiveItemInstruction(
                    item=item.strip(),
                    to_actor_id=(
                        to_actor_id.strip()
                        if isinstance(to_actor_id, str) and to_actor_id.strip()
                        else None
                    ),
                    to_discord_mention=(
                        to_discord_mention.strip()
                        if isinstance(to_discord_mention, str)
                        and to_discord_mention.strip()
                        else None
                    ),
                )

        character_updates = payload.get("character_updates")
        if not isinstance(character_updates, dict):
            character_updates = {}
        location_updates = payload.get("location_updates")
        if not isinstance(location_updates, dict):
            location_updates = {}

        _TOOL_CALLS_ALLOWLIST = frozenset({
            "sms_write",
            "sms_schedule",
            "plot_plan",
            "chapter_plan",
            "song_search",
        })
        tool_calls_raw = payload.get("tool_calls")
        tool_calls: list[dict[str, Any]] = []
        if isinstance(tool_calls_raw, list):
            for tc in tool_calls_raw:
                if isinstance(tc, dict) and isinstance(tc.get("tool_call"), str):
                    if tc["tool_call"].strip().lower() in _TOOL_CALLS_ALLOWLIST:
                        tool_calls.append(tc)

        return LLMTurnOutput(
            narration=narration,
            reasoning=reasoning,
            scene_output=scene_output,
            state_update=state_update,
            summary_update=summary_update,
            xp_awarded=xp_awarded,
            player_state_update=player_state_update,
            other_player_state_updates=other_player_state_updates,
            co_located_player_slugs=co_located_player_slugs,
            turn_visibility=turn_visibility,
            scene_image_prompt=scene_image_prompt,
            timer_instruction=timer_instruction,
            character_updates=character_updates,
            location_updates=location_updates,
            give_item=give_item,
            tool_calls=tool_calls,
        )

    def _tool_memory_search(
        self,
        campaign_id: str,
        payload: dict[str, Any],
        *,
        actor_id: str | None = None,
    ) -> str:
        queries_raw = payload.get("queries")
        category_raw = payload.get("category")

        queries: list[str] = []
        if isinstance(queries_raw, list):
            for row in queries_raw:
                if isinstance(row, str) and row.strip():
                    queries.append(row.strip())
        elif isinstance(queries_raw, str) and queries_raw.strip():
            queries.append(queries_raw.strip())

        category = (
            category_raw.strip()
            if isinstance(category_raw, str) and category_raw.strip()
            else None
        )
        category_scope = " ".join((category or "").lower().split())
        interaction_participant_slug: str | None = None
        awareness_npc_slug: str | None = None
        visibility_scope_filter: str | None = None
        structured_turn_scope = False
        if category_scope in {"interaction", "interactions"}:
            structured_turn_scope = True
        elif category_scope.startswith("interaction:") and self._emulator is not None:
            structured_turn_scope = True
            interaction_participant_slug = self._emulator._player_slug_key(  # noqa: SLF001
                category_scope.split(":", 1)[1]
            )
        elif category_scope.startswith("awareness:"):
            structured_turn_scope = True
            awareness_npc_slug = category_scope.split(":", 1)[1].strip() or None
        elif category_scope.startswith("visibility:"):
            structured_turn_scope = True
            visibility_scope_filter = (
                category_scope.split(":", 1)[1].strip().lower() or None
            )
        if not queries:
            return "MEMORY_RECALL: No queries provided."
        self._zork_log(
            f"MEMORY_SEARCH campaign={campaign_id}",
            f"queries={queries!r}  category={category!r}  actor={actor_id}",
        )
        try:
            source_before_lines = int(payload.get("before_lines", 0))
        except Exception:
            source_before_lines = 0
        try:
            source_after_lines = int(payload.get("after_lines", 0))
        except Exception:
            source_after_lines = 0
        source_before_lines = max(0, min(50, source_before_lines))
        source_after_lines = max(0, min(50, source_after_lines))
        full_text = self._memory_tool_bool_value(payload.get("full_text"))
        raw_turn_id_scope = None
        if "search_within_turn_ids" in payload:
            raw_turn_id_scope = payload.get("search_within_turn_ids")
        elif "within_turn_ids" in payload:
            raw_turn_id_scope = payload.get("within_turn_ids")
        elif "turn_ids" in payload:
            raw_turn_id_scope = payload.get("turn_ids")
        search_within_turn_ids = self._memory_tool_turn_id_list(raw_turn_id_scope)
        search_within_turn_id_set = set(search_within_turn_ids)
        excluded_recent_turn_ids: set[int] = set()

        curated_hits: list[tuple[str, str, float]] = []
        source_docs: list[dict[str, Any]] = []
        has_source_material = False
        source_scope = False
        source_scope_key: str | None = None
        source_doc_formats: dict[str, str] = {}
        if self._emulator is not None:
            source_docs = self._emulator.list_source_material_documents(campaign_id, limit=8)
            has_source_material = bool(source_docs)
            for row in source_docs:
                doc_key = str(row.get("document_key") or "")
                if not doc_key:
                    continue
                doc_format = str(row.get("format") or "").strip().lower()
                if not doc_format:
                    doc_format = str(
                        self._emulator._source_material_format_heuristic(  # noqa: SLF001
                            str(row.get("sample_chunk") or "")
                        )
                    ).strip().lower()
                if not doc_format:
                    doc_format = "generic"
                source_doc_formats[doc_key] = doc_format
            if category_scope in {"source", "source-material"}:
                source_scope = True
            elif category_scope.startswith("source:"):
                source_scope = True
                source_scope_key = category_scope.split(":", 1)[1].strip() or None
            include_turn_hits = not source_scope
            include_manual_hits = not source_scope
            include_source_hits = source_scope or not category_scope
            curated_seen: set[tuple[str, str]] = set()
            if include_manual_hits:
                for query in queries[:4]:
                    for hit in self._emulator.search_curated_memories(
                        query=query,
                        campaign_id=campaign_id,
                        category=category,
                        top_k=5,
                    ):
                        dedup_key = (str(hit[0] or "").strip(), str(hit[1] or "").strip())
                        if dedup_key in curated_seen:
                            continue
                        curated_seen.add(dedup_key)
                        curated_hits.append(hit)
        else:
            include_turn_hits = True
            include_manual_hits = True
            include_source_hits = not category_scope

        if include_turn_hits and not search_within_turn_id_set:
            excluded_recent_turn_ids = self._memory_tool_excluded_turn_ids_from_recent_context(
                campaign_id,
                actor_id,
            )

        roster_hints: list[dict[str, Any]] = []
        if self._emulator is not None and hasattr(
            self._emulator, "record_memory_search_usage"
        ):
            try:
                roster_hints_raw = self._emulator.record_memory_search_usage(
                    campaign_id, queries[:8]
                )
                if isinstance(roster_hints_raw, list):
                    for row in roster_hints_raw:
                        if isinstance(row, dict):
                            roster_hints.append(row)
            except Exception:
                roster_hints = []

        narrator_hits: dict[int, dict[str, Any]] = {}
        if include_turn_hits:
            with self._session_factory() as session:
                turns = (
                    session.query(Turn)
                    .filter(Turn.campaign_id == campaign_id)
                    .filter(Turn.kind == "narrator")
                    .order_by(Turn.id.desc())
                    .all()
                )
                actor_row = None
                if actor_id:
                    actor_row = (
                        session.query(Player)
                        .filter(Player.campaign_id == campaign_id)
                        .filter(Player.actor_id == actor_id)
                        .first()
                    )
            actor_slug = None
            actor_location_key = ""
            if actor_row is not None and self._emulator is not None:
                actor_state = self._parse_json(actor_row.state_json, {})
                actor_slug = self._emulator._player_visibility_slug(actor_id)  # noqa: SLF001
                actor_location_key = self._emulator._room_key_from_player_state(  # noqa: SLF001
                    actor_state
                )
            for query in queries[:4]:
                q = query.lower().strip()
                parts = [token for token in re.split(r"\W+", q) if token]
                for turn in turns:
                    turn_id = int(turn.id or 0)
                    if search_within_turn_id_set and turn_id not in search_within_turn_id_set:
                        continue
                    if excluded_recent_turn_ids and turn_id in excluded_recent_turn_ids:
                        continue
                    content = str(turn.content or "")
                    if not content:
                        continue
                    meta = self._parse_json(turn.meta_json, {})
                    visibility = meta.get("visibility") if isinstance(meta, dict) else None
                    if actor_id and self._emulator is not None:
                        if not self._emulator._turn_visible_to_viewer(  # noqa: SLF001
                            turn, actor_id, actor_slug or "", actor_location_key.lower()
                        ):
                            continue
                    if structured_turn_scope:
                        actor_player_slug = str(meta.get("actor_player_slug") or "").strip()
                        if interaction_participant_slug:
                            visible_player_slugs = (
                                visibility.get("visible_player_slugs")
                                if isinstance(visibility, dict)
                                else []
                            )
                            visible_slug_set = {
                                self._emulator._player_slug_key(item)  # noqa: SLF001
                                for item in (
                                    visible_player_slugs
                                    if isinstance(visible_player_slugs, list)
                                    else []
                                )
                                if self._emulator is not None
                            }
                            if (
                                actor_player_slug != interaction_participant_slug
                                and interaction_participant_slug not in visible_slug_set
                            ):
                                continue
                        if awareness_npc_slug:
                            aware_npc_slugs = (
                                visibility.get("aware_npc_slugs")
                                if isinstance(visibility, dict)
                                else []
                            )
                            if awareness_npc_slug not in {
                                str(item or "").strip()
                                for item in (
                                    aware_npc_slugs
                                    if isinstance(aware_npc_slugs, list)
                                    else []
                                )
                            }:
                                continue
                        if visibility_scope_filter in {"public", "private", "limited", "local"}:
                            row_scope = str((visibility or {}).get("scope") or "public").strip().lower()
                            if row_scope != visibility_scope_filter:
                                continue
                    hay = content.lower()
                    score = 0.0
                    if q and q in hay:
                        score = 1.0
                    elif parts:
                        score = sum(1 for token in parts if token in hay) / len(parts)
                    if score <= 0.0:
                        continue
                    prior = narrator_hits.get(int(turn.id))
                    if prior is None or score > float(prior.get("score", 0.0)):
                        narrator_hits[turn_id] = {
                            "turn_id": turn_id,
                            "score": score,
                            "content": content,
                            "visibility_scope": str(
                                (visibility or {}).get("scope") or "public"
                            ),
                            "actor_player_slug": str(meta.get("actor_player_slug") or ""),
                            "location_key": str(meta.get("location_key") or ""),
                        }

        # Embedding-based turn search via the memory port (supplements keyword hits).
        embed_only_hits: dict[int, dict[str, Any]] = {}
        if (
            include_turn_hits
            and self._emulator is not None
            and self._emulator._memory_port is not None
        ):
            for query in queries[:4]:
                try:
                    embed_hits = self._emulator._memory_port.search(  # noqa: SLF001
                        query=query,
                        campaign_id=campaign_id,
                        top_k=5,
                    )
                    for turn_id, kind, content, score in embed_hits:
                        tid = int(turn_id)
                        if search_within_turn_id_set and tid not in search_within_turn_id_set:
                            continue
                        if excluded_recent_turn_ids and tid in excluded_recent_turn_ids:
                            continue
                        if tid in narrator_hits:
                            # Already found by keyword — boost score if embedding is higher.
                            prior = narrator_hits[tid]
                            if float(score) > float(prior.get("score", 0.0)):
                                prior["score"] = float(score)
                        else:
                            prior_embed = embed_only_hits.get(tid)
                            if prior_embed is None or float(score) > float(prior_embed.get("score", 0.0)):
                                embed_only_hits[tid] = {
                                    "turn_id": tid,
                                    "score": float(score),
                                    "content": str(content or ""),
                                    "visibility_scope": "public",
                                    "actor_player_slug": "",
                                    "location_key": "",
                                }
                except Exception:
                    pass

        # Take top keyword hits, then fill remaining slots with embedding-only hits.
        ordered_keyword = sorted(
            narrator_hits.values(),
            key=lambda row: (float(row.get("score", 0.0)), int(row.get("turn_id", 0))),
            reverse=True,
        )[:5]
        keyword_ids = {int(r.get("turn_id", 0)) for r in ordered_keyword}
        ordered_embed_only = sorted(
            (v for v in embed_only_hits.values() if int(v.get("turn_id", 0)) not in keyword_ids),
            key=lambda row: (float(row.get("score", 0.0)), int(row.get("turn_id", 0))),
            reverse=True,
        )[:3]
        ordered_narrator = ordered_keyword + ordered_embed_only
        source_hits_flat: list[tuple[str, str, int, str, float]] = []
        if self._emulator is not None and has_source_material and include_source_hits:
            for query in queries[:4]:
                source_hits_flat.extend(
                    self._emulator.search_source_material(
                        query,
                        campaign_id,
                        document_key=source_scope_key,
                        top_k=10 if source_scope else 6,
                        before_lines=source_before_lines,
                        after_lines=source_after_lines,
                    )
                )
        source_hits_unique: list[tuple[str, str, int, str, float]] = []
        seen_source = set()
        for row in source_hits_flat:
            row_key = (str(row[0] or ""), int(row[2] or 0))
            if row_key in seen_source:
                continue
            seen_source.add(row_key)
            source_hits_unique.append(row)
        source_hits_unique = source_hits_unique[:12]

        records: list[dict[str, Any]] = []
        total_hits = 0
        if include_turn_hits:
            for hit in ordered_narrator:
                total_hits += 1
                turn_text = str(hit.get("content") or "")
                records.append(
                    {
                        "kind": "memory_hit",
                        "memory_type": "turn",
                        "turn_id": int(hit.get("turn_id", 0) or 0),
                        "relevance": round(float(hit.get("score", 0.0) or 0.0), 4),
                        "actor_player_slug": str(hit.get("actor_player_slug") or "").strip(),
                        "visibility_scope": str(hit.get("visibility_scope") or "public").strip(),
                        "location_key": str(hit.get("location_key") or "").strip(),
                        "text": self._memory_tool_text_value(turn_text, max_chars=900),
                        **(
                            {
                                "full_text": self._memory_tool_text_value(
                                    turn_text,
                                    max_chars=12000,
                                )
                            }
                            if full_text
                            else {}
                        ),
                    }
                )
        if include_manual_hits:
            for term, memory, score in curated_hits[:5]:
                total_hits += 1
                records.append(
                    {
                        "kind": "memory_hit",
                        "memory_type": "manual",
                        "term": str(term or "").strip(),
                        "relevance": round(float(score or 0.0), 4),
                        "text": self._memory_tool_text_value(memory or "", max_chars=700),
                    }
                )
        for (
            source_doc_key,
            source_doc_label,
            source_chunk_index,
            source_chunk_text,
            source_score,
        ) in source_hits_unique:
            if float(source_score) < 0.40:
                continue
            total_hits += 1
            records.append(
                {
                    "kind": "memory_hit",
                    "memory_type": "source",
                    "document_key": str(source_doc_key or ""),
                    "document_label": str(source_doc_label or ""),
                    "chunk_index": int(source_chunk_index or 0),
                    "relevance": round(float(source_score or 0.0), 4),
                    "text": self._memory_tool_text_value(source_chunk_text or "", max_chars=1600),
                }
            )
        records.insert(
            0,
            {
                "kind": "memory_query_result",
                "queries": queries[:4],
                "category": category or "",
                "hit_count": total_hits,
                "full_text": full_text,
                "search_within_turn_ids": search_within_turn_ids,
                "excluded_recent_turn_count": len(excluded_recent_turn_ids),
            },
        )
        if has_source_material:
            total_snippets = 0
            for row in source_docs:
                try:
                    total_snippets += int(row.get("chunk_count") or 0)
                except Exception:
                    continue
            records.append(
                {
                    "kind": "source_index_meta",
                    "document_count": len(source_docs),
                    "snippet_count": total_snippets,
                }
            )
            for row in source_docs[:5]:
                row_format = str(row.get("format") or "").strip().lower() or source_doc_formats.get(
                    str(row.get("document_key") or ""), "generic"
                )
                records.append(
                    {
                        "kind": "source_index_entry",
                        "document_key": str(row.get("document_key") or ""),
                        "document_label": str(row.get("document_label") or ""),
                        "format": row_format,
                        "snippet_count": row.get("chunk_count"),
                    }
                )
        for hint in roster_hints[:6]:
            if not isinstance(hint, dict):
                continue
            records.append(
                {
                    "kind": "memory_roster_hint",
                    "term": str(hint.get("term") or hint.get("slug") or "").strip(),
                    "slug": str(hint.get("slug") or "").strip(),
                    "count": int(hint.get("count") or 0),
                }
            )
        # When searching a specific category, proactively append available
        # terms so the model knows what else is stored without a separate
        # memory_terms call.
        if category_scope and self._emulator is not None and not source_scope:
            try:
                all_terms = self._emulator.list_memory_terms(campaign_id, wildcard="%", limit=100)
                category_terms = [
                    t for t in all_terms
                    if isinstance(t, dict) and str(t.get("category") or "").strip().lower() == category_scope
                ]
                for t in category_terms[:15]:
                    records.append({
                        "kind": "category_term",
                        "category": str(t.get("category") or "").strip(),
                        "term": str(t.get("term") or "").strip(),
                        "count": t.get("count"),
                    })
            except Exception:
                pass

        if not records:
            self._zork_log(
                f"MEMORY_SEARCH RESULT campaign={campaign_id}",
                "No relevant memories found.",
            )
            return "MEMORY_RECALL: No relevant memories found."
        result_jsonl = self._memory_tool_jsonl(records)
        hit_count = sum(1 for r in records if r.get("kind") == "memory_hit")
        self._zork_log(
            f"MEMORY_SEARCH RESULT campaign={campaign_id}",
            f"hits={hit_count}  curated={len(curated_hits)}  "
            f"narrator={len(ordered_narrator)}  source={len(source_hits_unique)}  "
            f"roster_hints={len(roster_hints)}  excluded_recent={len(excluded_recent_turn_ids)}",
        )
        next_actions = [
            "MEMORY_RECALL_NEXT_ACTIONS:",
            '- Narrow within the last turn hits only: {"tool_call":"memory_search","search_within":"last_results","queries":["keyword"]}',
            '- Request expanded turn text only after narrowing: {"tool_call":"memory_search","search_within":"last_results","queries":["exact wording"],"full_text":true}',
            '- Prune prior memory-turn context you no longer need: {"tool_call":"memory_search","search_within":"last_results","queries":["keyword"],"keep_memory_turns":[123,456]}',
            '- Retrieve one exact turn directly: {"tool_call":"memory_turn","turn_id":1234}',
        ]
        return "MEMORY_RECALL:\n" + result_jsonl + "\n" + "\n".join(next_actions)

    def _tool_memory_terms(self, campaign_id: str, payload: dict[str, Any]) -> str:
        wildcard_raw = payload.get("wildcard")
        wildcard = (
            wildcard_raw.strip()
            if isinstance(wildcard_raw, str) and wildcard_raw.strip()
            else "*"
        )
        wildcard_sql = wildcard.replace("*", "%")

        terms: list[Any] = []
        if self._emulator is not None:
            terms = self._emulator.list_memory_terms(
                campaign_id, wildcard=wildcard_sql, limit=50
            )

        if not terms:
            fallback_rows = self._fallback_memory_state(campaign_id)
            distinct = sorted(
                {str(row.get("term") or "") for row in fallback_rows if row.get("term")}
            )
            terms = [term for term in distinct if term and fnmatch(term, wildcard)]

        rows: list[dict[str, Any]] = []
        if terms:
            for row in terms[:40]:
                if isinstance(row, dict):
                    rows.append(
                        {
                            "kind": "memory_term",
                            "category": str(row.get("category") or "").strip(),
                            "term": str(row.get("term") or "").strip(),
                            "count": row.get("count"),
                            "last_at": row.get("last_at"),
                        }
                    )
                else:
                    rows.append(
                        {
                            "kind": "memory_term",
                            "category": "",
                            "term": str(row),
                            "count": None,
                            "last_at": None,
                        }
                    )
        else:
            rows.append(
                {
                    "kind": "memory_term",
                    "category": "",
                    "term": "",
                    "count": 0,
                    "last_at": None,
                }
            )
        return "MEMORY_TERMS_RESULT:\n" + self._memory_tool_jsonl(rows)

    def _tool_source_browse(self, campaign_id: str, payload: dict[str, Any]) -> str:
        doc_key_raw = payload.get("document_key") or payload.get("document")
        document_key = str(doc_key_raw).strip()[:120] if isinstance(doc_key_raw, str) else ""
        wildcard_raw = payload.get("wildcard")
        wildcard = str(wildcard_raw).strip()[:120] if isinstance(wildcard_raw, str) else ""
        wildcard_provided = bool(wildcard)
        wildcard = wildcard or "*"
        wildcard_meta = f"wildcard={wildcard!r}"
        if not wildcard_provided:
            wildcard_meta = "wildcard=(omitted)"
        limit = 60
        try:
            limit = max(1, min(120, int(payload.get("limit") or 60)))
        except Exception:
            pass

        rows: list[str] = []
        if self._emulator is not None:
            browse = getattr(self._emulator, "browse_source_keys", None)
            if callable(browse):
                rows = browse(
                    campaign_id,
                    document_key=document_key or None,
                    wildcard=wildcard,
                    limit=limit,
                )

        if rows:
            return (
                f"SOURCE_BROWSE_RESULT "
                f"(document_key={document_key or '*'!r}, "
                f"{wildcard_meta}, "
                f"showing {len(rows)}):\n" + "\n".join(str(row) for row in rows)
            )
        return (
            f"SOURCE_BROWSE_RESULT "
            f"(document_key={document_key or '*'!r}, "
            f"{wildcard_meta}): no entries found"
        )

    def _tool_communication_rules(self, payload: dict[str, Any]) -> str:
        if self._emulator is None:
            return "COMMUNICATION_RULES_RESULT: unavailable"
        raw_keys = payload.get("keys") or []
        if isinstance(raw_keys, str):
            raw_keys = [raw_keys]
        requested_keys = [
            str(key or "").strip().upper()
            for key in raw_keys
            if str(key or "").strip()
        ][:8]
        requested_set = set(requested_keys)
        lines = [
            f"{rule_key}: {rule_text}"
            for rule_key, rule_text in self._emulator.DEFAULT_GM_COMMUNICATION_RULES.items()  # noqa: SLF001
            if rule_key in requested_set
        ]
        if lines:
            return "COMMUNICATION_RULES_RESULT:\n" + "\n".join(lines)
        return (
            "COMMUNICATION_RULES_RESULT: no matching keys found. Available keys: "
            + ", ".join(self._emulator.COMMUNICATION_RULE_KEYS)  # noqa: SLF001
        )

    def _tool_autobiography_append(self, campaign_id: str, payload: dict[str, Any]) -> str:
        emulator = self._emulator
        if emulator is None:
            return "AUTOBIOGRAPHY_APPEND_RESULT: unavailable"
        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return "AUTOBIOGRAPHY_APPEND_RESULT: campaign not found"
            characters = emulator.get_campaign_characters(campaign)
            state = self._parse_json(campaign.state_json, {})
            game_time = state.get("game_time") if isinstance(state, dict) else {}
            latest_turn_id = 0
            turn_rows = (
                session.query(Turn.id)
                .filter(Turn.campaign_id == campaign_id)
                .order_by(Turn.id.desc())
                .limit(1)
                .all()
            )
            if turn_rows:
                latest_turn_id = int(turn_rows[0][0] or 0)
            characters, applied_rows = emulator._apply_autobiography_update_to_characters(  # noqa: SLF001
                characters,
                payload,
                current_turn=latest_turn_id,
                game_time=game_time if isinstance(game_time, dict) else {},
            )
            if not applied_rows:
                return (
                    "AUTOBIOGRAPHY_APPEND_RESULT: nothing stored. "
                    "Use existing NPC slugs from CHARACTER_INDEX / WORLD_CHARACTERS and include a non-empty a/b/c delta."
                )
            campaign.characters_json = json.dumps(characters, ensure_ascii=True)
            campaign.updated_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
        lines = ["AUTOBIOGRAPHY_APPEND_RESULT:"]
        for row in applied_rows[:8]:
            suggestion = ""
            if int(row.get("raw_count") or 0) >= emulator.AUTOBIOGRAPHY_COMPRESS_TRIGGER_COUNT:  # noqa: SLF001
                suggestion = " [compression recommended]"
            lines.append(
                "- "
                f"{row.get('character')}: {row.get('importance')} delta stored "
                f"(raw_count={row.get('raw_count')}){suggestion}"
            )
        return "\n".join(lines)

    async def _tool_autobiography_compress(self, campaign_id: str, payload: dict[str, Any]) -> str:
        emulator = self._emulator
        if emulator is None:
            return "AUTOBIOGRAPHY_COMPRESS_RESULT: unavailable"
        raw_slug = str(
            payload.get("character")
            or payload.get("slug")
            or payload.get("npc")
            or ""
        ).strip()
        if not raw_slug:
            return "AUTOBIOGRAPHY_COMPRESS_RESULT: character not found. Use an existing NPC slug from CHARACTER_INDEX / WORLD_CHARACTERS."

        # Phase 1: read character data and build the LLM prompt.
        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return "AUTOBIOGRAPHY_COMPRESS_RESULT: campaign not found"
            characters = emulator.get_campaign_characters(campaign)
            resolved_slug = emulator._resolve_existing_character_slug(characters, raw_slug) or raw_slug  # noqa: SLF001
            character_row = dict(characters.get(resolved_slug) or {})
            if not character_row:
                return "AUTOBIOGRAPHY_COMPRESS_RESULT: character not found. Use an existing NPC slug from CHARACTER_INDEX / WORLD_CHARACTERS."
            raw_entries = emulator._normalize_autobiography_raw_entries(  # noqa: SLF001
                character_row.get(emulator.AUTOBIOGRAPHY_RAW_FIELD)  # noqa: SLF001
            )
            current_auto = str(character_row.get(emulator.AUTOBIOGRAPHY_FIELD) or "").strip()  # noqa: SLF001
        if not raw_entries and not current_auto:
            return f"AUTOBIOGRAPHY_COMPRESS_RESULT: {resolved_slug} has no autobiography material yet."
        entry_lines: list[str] = []
        for row in raw_entries[-16:]:
            if not isinstance(row, dict):
                continue
            a_text = emulator._sanitize_autobiography_text(  # noqa: SLF001
                row.get("a"),
                max_chars=emulator.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,  # noqa: SLF001
            )
            b_text = emulator._sanitize_autobiography_text(  # noqa: SLF001
                row.get("b"),
                max_chars=emulator.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,  # noqa: SLF001
            )
            c_text = emulator._sanitize_autobiography_text(  # noqa: SLF001
                row.get("c"),
                max_chars=emulator.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,  # noqa: SLF001
            )
            legacy_text = emulator._sanitize_autobiography_text(  # noqa: SLF001
                row.get("text"),
                max_chars=emulator.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,  # noqa: SLF001
            )
            if not (a_text or b_text or c_text or legacy_text):
                continue
            stamp = ""
            gt = row.get("game_time")
            if isinstance(gt, dict):
                day = max(0, emulator._coerce_int(gt.get("day"), 0))  # noqa: SLF001
                hour = min(23, max(0, emulator._coerce_int(gt.get("hour"), 0)))  # noqa: SLF001
                minute = min(59, max(0, emulator._coerce_int(gt.get("minute"), 0)))  # noqa: SLF001
                if day > 0:
                    stamp = f"Day {day} {hour:02d}:{minute:02d} "
            importance = " ".join(str(row.get("importance") or "").strip().split())[:40]
            trigger = " ".join(str(row.get("trigger") or "").strip().split())[:80]
            raw_row: dict[str, str] = {}
            if trigger:
                raw_row["trigger"] = trigger
            if importance:
                raw_row["importance"] = importance
            if a_text:
                raw_row["a"] = a_text
            if b_text:
                raw_row["b"] = b_text
            if c_text:
                raw_row["c"] = c_text
            if legacy_text and not raw_row.get("a") and not raw_row.get("b") and not raw_row.get("c"):
                raw_row["text"] = legacy_text
            entry_lines.append(f"- {stamp}{json.dumps(raw_row, ensure_ascii=True)}".strip())
        system_prompt = (
            "You are compressing a character autobiography. "
            "Output ONLY the rewritten autobiography text, in first person, in the character's own voice. "
            "Do not output JSON, labels, bullets, or explanation.\n"
            "Rules:\n"
            "- Preserve values, patterns, loyalties, and self-understanding the character still acts from.\n"
            "- Preserve unresolved contradictions as tension; do not resolve them unless the story already did.\n"
            "- Preserve relationship turns that changed the character's understanding of someone.\n"
            "- Compress repetition. Keep only what future narration needs to write the character accurately.\n"
            "- The autobiography is constitutional: growth is allowed, drift without reckoning is not.\n"
        )
        char_name = character_row.get("name") or resolved_slug
        user_prompt = (
            f"CHARACTER: {char_name} ({resolved_slug})\n"
            f"CURRENT_AUTOBIOGRAPHY: {current_auto or '(none)'}\n"
            "RAW_ENTRIES:\n"
            f"{chr(10).join(entry_lines) or '(none)'}\n"
            f"Write a compressed autobiography no longer than {emulator.MAX_AUTOBIOGRAPHY_TEXT_CHARS} characters."
        )

        # Phase 2: LLM call (no DB session held open).
        response = await self._completion.complete(
            system_prompt,
            user_prompt,
            temperature=0.3,
            max_tokens=min(self._max_tokens, 700),
        )
        cleaned = response or ""
        cleaned = re.sub(r"^```[\w-]*\s*", "", cleaned).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        compressed = emulator._sanitize_autobiography_text(  # noqa: SLF001
            cleaned,
            max_chars=emulator.MAX_AUTOBIOGRAPHY_TEXT_CHARS,  # noqa: SLF001
        )
        if not compressed:
            return f"AUTOBIOGRAPHY_COMPRESS_RESULT: failed for {resolved_slug}."

        # Phase 3: re-load campaign and persist the compressed autobiography.
        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return f"AUTOBIOGRAPHY_COMPRESS_RESULT: failed for {resolved_slug}."
            characters = emulator.get_campaign_characters(campaign)
            if resolved_slug not in characters:
                return f"AUTOBIOGRAPHY_COMPRESS_RESULT: failed for {resolved_slug}."
            latest_turn_id = 0
            turn_rows = (
                session.query(Turn.id)
                .filter(Turn.campaign_id == campaign_id)
                .order_by(Turn.id.desc())
                .limit(1)
                .all()
            )
            if turn_rows:
                latest_turn_id = int(turn_rows[0][0] or 0)
            characters, applied_row = emulator._apply_autobiography_compress_to_characters(  # noqa: SLF001
                characters,
                {"character": resolved_slug, "autobiography": compressed},
                current_turn=latest_turn_id,
            )
            if not applied_row:
                return f"AUTOBIOGRAPHY_COMPRESS_RESULT: failed for {resolved_slug}."
            campaign.characters_json = json.dumps(characters, ensure_ascii=True)
            campaign.updated_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
        return (
            "AUTOBIOGRAPHY_COMPRESS_RESULT:\n"
            f"- character={resolved_slug}\n"
            f"- raw_count={applied_row.get('raw_count')}\n"
            f"- autobiography={compressed}"
        )

    def _tool_name_generate(self, campaign_id: str, payload: dict[str, Any]) -> str:
        raw_origins = payload.get("origins") or []
        if isinstance(raw_origins, str):
            raw_origins = [raw_origins]
        origins = [str(o).strip().lower() for o in raw_origins if str(o or "").strip()][
            :4
        ]
        ng_gender = str(payload.get("gender") or "both").strip().lower()
        ng_count = 5
        try:
            ng_count = max(1, min(6, int(payload.get("count") or 5)))
        except (TypeError, ValueError):
            pass
        ng_context = str(payload.get("context") or "").strip()[:300]

        names: list[str] = []
        if self._emulator is not None:
            fetch = getattr(self._emulator, "_fetch_random_names", None)
            if callable(fetch):
                names = fetch(origins=origins or None, gender=ng_gender, count=ng_count)

        if names:
            result = (
                f"NAME_GENERATE_RESULT "
                f"(origins={origins or 'any'}, "
                f"gender={ng_gender}, "
                f"count={len(names)}):\n"
                + "\n".join(f"- {n}" for n in names)
                + "\n\nEvaluate these against your character concept"
            )
            if ng_context:
                result += f" ({ng_context})"
            result += (
                ". Pick the best fit, or call name_generate again "
                "with different origins/gender for more options."
            )
            return result
        return (
            f"NAME_GENERATE_RESULT "
            f"(origins={origins or 'any'}): no names returned — try broader origins "
            "or fewer filters."
        )

    def _tool_character_cards(
        self,
        campaign_id: str,
        payload: dict[str, Any],
        *,
        actor_id: str | None = None,
    ) -> str:
        emulator = self._emulator
        if emulator is None or not actor_id:
            return "CHARACTER_CARDS_RESULT: unavailable"

        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return "CHARACTER_CARDS_RESULT: campaign not found"
            player = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id, Player.actor_id == actor_id)
                .one_or_none()
            )
            if player is None:
                return "CHARACTER_CARDS_RESULT: player not found"
            player_state = emulator.get_player_state(player)
            campaign_state = emulator.get_campaign_state(campaign)
            characters = emulator.get_campaign_characters(campaign)

        raw_character = str(
            payload.get("character")
            or payload.get("slug")
            or payload.get("npc")
            or payload.get("name")
            or ""
        ).strip()

        if raw_character:
            resolved_slug = emulator._resolve_existing_character_slug(characters, raw_character) or raw_character  # noqa: SLF001
            source = dict(characters.get(resolved_slug) or {})
            if not source:
                return (
                    "CHARACTER_CARDS_RESULT: character not found. "
                    "Use an NPC slug from CHARACTER_INDEX / CHARACTER_CARDS."
                )
            source["_slug"] = resolved_slug
            cards = emulator._build_character_cards_for_prompt(  # noqa: SLF001
                [source],
                characters=characters,
                campaign_state=campaign_state,
                player_state=player_state,
                include_critical_fields=False,
            )
            return f"CHARACTER_CARDS_RESULT: {emulator._dump_json(cards)}"  # noqa: SLF001

        player_registry = emulator._campaign_player_registry(campaign_id, self._session_factory)  # noqa: SLF001
        excluded_character_keys = emulator._player_character_prompt_keys(  # noqa: SLF001
            [],
            player_registry=player_registry,
        )
        scene_characters = emulator._build_scene_characters_for_prompt(  # noqa: SLF001
            characters,
            player_state,
            excluded_character_keys=excluded_character_keys,
        )
        scene_characters = scene_characters[: emulator.MAX_CHARACTERS_IN_PROMPT]  # noqa: SLF001
        scene_characters = emulator._fit_characters_to_budget(  # noqa: SLF001
            scene_characters,
            emulator.MAX_CHARACTERS_CHARS,  # noqa: SLF001
        )
        cards = emulator._build_character_cards_for_prompt(  # noqa: SLF001
            scene_characters,
            characters=characters,
            campaign_state=campaign_state,
            player_state=player_state,
            include_critical_fields=False,
        )
        return f"CHARACTER_CARDS_RESULT: {emulator._dump_json(cards)}"  # noqa: SLF001

    def _tool_recent_turns(
        self,
        campaign_id: str,
        payload: dict[str, Any],
        *,
        actor_id: str | None = None,
    ) -> str:
        if self._emulator is None or not actor_id:
            return "RECENT_TURNS: unavailable"
        try:
            limit = max(1, min(40, int(payload.get("limit") or 24)))
        except Exception:
            limit = 24
        raw_player_slugs = payload.get("player_slugs")
        requested_player_slugs = {
            self._emulator._player_slug_key(item)  # noqa: SLF001
            for item in (raw_player_slugs if isinstance(raw_player_slugs, list) else [])
            if self._emulator._player_slug_key(item)  # noqa: SLF001
        }
        raw_npc_slugs = payload.get("npc_slugs")
        requested_npc_slugs = {
            str(item or "").strip()
            for item in (raw_npc_slugs if isinstance(raw_npc_slugs, list) else [])
            if str(item or "").strip()
        }

        with self._session_factory() as session:
            actor_row = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == actor_id)
                .first()
            )
            turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .order_by(Turn.id.desc())
                .limit(max(limit * 4, 40))
                .all()
            )
        turns.reverse()
        actor_state = self._parse_json(actor_row.state_json, {}) if actor_row is not None else {}
        viewer_slug = self._emulator._player_visibility_slug(actor_id)  # noqa: SLF001
        viewer_location_key = self._emulator._room_key_from_player_state(actor_state).lower()  # noqa: SLF001

        def _turn_relevant_to_receivers(turn: Turn) -> bool:
            meta = self._emulator._safe_turn_meta(turn)  # noqa: SLF001
            visibility = meta.get("visibility")
            if not isinstance(visibility, dict):
                return False
            scope = str(visibility.get("scope") or "").strip().lower()
            if scope not in {"private", "limited"}:
                return False
            visible_player_slugs = {
                self._emulator._player_slug_key(item)  # noqa: SLF001
                for item in list(visibility.get("visible_player_slugs") or [])
                if self._emulator._player_slug_key(item)  # noqa: SLF001
            }
            aware_npc_slugs = {
                str(item or "").strip()
                for item in list(visibility.get("aware_npc_slugs") or [])
                if str(item or "").strip()
            }
            player_match = True
            npc_match = True
            if requested_player_slugs:
                player_match = bool(visible_player_slugs.intersection(requested_player_slugs))
            if requested_npc_slugs:
                npc_match = bool(aware_npc_slugs.intersection(requested_npc_slugs))
            return player_match and npc_match

        rows: list[dict[str, Any]] = []
        location_history: list[str] = []
        visible_count = 0
        for turn in turns:
            visible = self._emulator._turn_visible_to_viewer(  # noqa: SLF001
                turn,
                actor_id,
                viewer_slug,
                viewer_location_key,
            )
            if (
                not visible
                and turn.actor_id == actor_id
                and (requested_player_slugs or requested_npc_slugs)
                and _turn_relevant_to_receivers(turn)
            ):
                visible = True
            if not visible:
                continue
            visible_count += 1
            meta = self._emulator._safe_turn_meta(turn)  # noqa: SLF001
            visibility = meta.get("visibility") if isinstance(meta, dict) else {}
            location_key = str(
                (visibility or {}).get("location_key") or meta.get("location_key") or ""
            ).strip()
            if location_key:
                location_history.append(location_key)
            scene_output = meta.get("scene_output")
            beats = scene_output.get("beats") if isinstance(scene_output, dict) else None
            if isinstance(beats, list) and beats:
                for index, beat in enumerate(beats):
                    if not isinstance(beat, dict):
                        continue
                    rows.append(
                        {
                            "kind": "beat",
                            "turn_id": int(turn.id),
                            "index": index,
                            "reasoning": str(beat.get("reasoning") or "").strip(),
                            "type": str(beat.get("type") or "narration").strip(),
                            "speaker": str(beat.get("speaker") or "narrator").strip(),
                            "actors": list(beat.get("actors") or []),
                            "listeners": list(beat.get("listeners") or []),
                            "visibility": str(beat.get("visibility") or "local").strip(),
                            "aware_actor_ids": list((beat.get("aware_actor_ids") or [])),
                            "aware_npc_slugs": list((beat.get("aware_npc_slugs") or [])),
                            "location_key": beat.get("location_key"),
                            "context_key": beat.get("context_key"),
                            "text": self._memory_tool_text_value(beat.get("text") or "", max_chars=2000),
                        }
                    )
            else:
                rows.append(
                    {
                        "kind": "beat",
                        "turn_id": int(turn.id),
                        "index": 0,
                        "reasoning": "Compatibility fallback from stored turn text.",
                        "type": "player_action" if str(turn.kind or "") == "player" else "narration",
                        "speaker": str(turn.actor_id or "narrator"),
                        "actors": [str(turn.actor_id or "").strip()] if str(turn.kind or "") == "player" else [],
                        "listeners": [],
                        "visibility": str((visibility or {}).get("scope") or "public").strip().lower(),
                        "aware_actor_ids": list((visibility or {}).get("visible_actor_ids") or []),
                        "aware_npc_slugs": list((visibility or {}).get("aware_npc_slugs") or []),
                        "location_key": location_key or None,
                        "context_key": str(
                            (visibility or {}).get("context_key") or meta.get("context_key") or ""
                        ).strip()
                        or None,
                        "text": self._memory_tool_text_value(turn.content or "", max_chars=2000),
                    }
                )
            if visible_count >= limit:
                break

        current_location = viewer_location_key or "unknown"
        last_other = "none"
        for key in reversed(location_history):
            if key and key.lower() != current_location.lower():
                last_other = key
                break
        header_lines = [
            "RECENT_TURNS_LOADED: true",
            "RECENT_TURNS_NOTE: Immediate visible continuity for the acting player.",
            "Local continuity is room-scoped; older local turns from other rooms may be missing.",
            f"RECENT_TURNS_LOCATIONS: current={current_location} last_other={last_other}",
            f"RECENT_TURNS_RECEIVERS: players={sorted(requested_player_slugs)} npcs={sorted(requested_npc_slugs)}",
            "RECENT_TURNS:",
        ]
        if not rows:
            header_lines.append("None")
            return "\n".join(header_lines)
        cleaned_rows = [
            self._prune_recent_turn_tool_row(row)
            for row in rows
            if isinstance(row, dict)
        ]
        return "\n".join(header_lines) + "\n" + self._memory_tool_jsonl(cleaned_rows)

    def _tool_memory_turn(
        self,
        campaign_id: str,
        payload: dict[str, Any],
        *,
        actor_id: str | None = None,
    ) -> str:
        turn_id_raw = payload.get("turn_id")
        try:
            turn_id = int(turn_id_raw)
        except Exception:
            return (
                "MEMORY_TURN_RESULT:\n"
                + self._memory_tool_jsonl(
                    [
                        {
                            "kind": "memory_turn_result",
                            "status": "invalid_turn_id",
                            "turn_id": str(turn_id_raw or ""),
                        }
                    ]
                )
            )

        with self._session_factory() as session:
            turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .filter(Turn.id == turn_id)
                .first()
            )
            actor_row = None
            if actor_id:
                actor_row = (
                    session.query(Player)
                    .filter(Player.campaign_id == campaign_id)
                    .filter(Player.actor_id == actor_id)
                    .first()
                )
        if turn is None:
            return (
                "MEMORY_TURN_RESULT:\n"
                + self._memory_tool_jsonl(
                    [
                        {
                            "kind": "memory_turn_result",
                            "status": "not_found",
                            "turn_id": turn_id,
                        }
                    ]
                )
            )
        if actor_id and self._emulator is not None:
            actor_state = (
                self._parse_json(actor_row.state_json, {})
                if actor_row is not None
                else {}
            )
            actor_slug = self._emulator._player_visibility_slug(actor_id)  # noqa: SLF001
            actor_location_key = self._emulator._room_key_from_player_state(  # noqa: SLF001
                actor_state
            )
            if not self._emulator._turn_visible_to_viewer(  # noqa: SLF001
                turn, actor_id, actor_slug or "", actor_location_key.lower()
            ):
                return (
                    "MEMORY_TURN_RESULT:\n"
                    + self._memory_tool_jsonl(
                        [
                            {
                                "kind": "memory_turn_result",
                                "status": "not_visible",
                                "turn_id": turn_id,
                            }
                        ]
                    )
                )

        return (
            "MEMORY_TURN_RESULT:\n"
            + self._memory_tool_jsonl(
                [
                    {
                        "kind": "memory_turn_result",
                        "status": "ok",
                        "turn_id": int(turn.id),
                        "turn_kind": str(turn.kind or ""),
                        "actor_id": str(turn.actor_id or ""),
                        "created_at": turn.created_at.isoformat() if turn.created_at else None,
                        "full_text": self._memory_tool_text_value(
                            turn.content or "", max_chars=12000
                        ),
                    }
                ]
            )
        )

    def _tool_memory_store(self, campaign_id: str, payload: dict[str, Any]) -> str:
        category = payload.get("category")
        term = payload.get("term")
        memory = payload.get("memory")
        if (
            not isinstance(category, str)
            or not category.strip()
            or not isinstance(memory, str)
            or not memory.strip()
        ):
            return "MEMORY_STORE_RESULT: invalid payload"

        if self._emulator is not None:
            ok, reason = self._emulator.store_memory(
                campaign_id,
                category=category.strip(),
                term=term.strip() if isinstance(term, str) and term.strip() else None,
                memory=memory.strip(),
            )
            if ok:
                return f"MEMORY_STORE_RESULT: stored via memory_port ({reason})"

        entries = self._fallback_memory_state(campaign_id)
        entry = {
            "id": len(entries) + 1,
            "category": category.strip(),
            "term": term.strip() if isinstance(term, str) and term.strip() else None,
            "memory": memory.strip(),
            "created_at": datetime.now(UTC).isoformat(),
            "source": "webui_fallback",
        }
        entries.append(entry)
        self._persist_fallback_memory_state(campaign_id, entries)
        return "MEMORY_STORE_RESULT: stored via fallback memory state"

    def _tool_sms_list(
        self,
        campaign_id: str,
        payload: dict[str, Any],
        *,
        actor_id: str | None = None,
    ) -> str:
        wildcard_raw = payload.get("wildcard")
        wildcard = (
            wildcard_raw.strip()
            if isinstance(wildcard_raw, str) and wildcard_raw.strip()
            else "*"
        )

        rows = (
            self._emulator.list_sms_threads(
                campaign_id,
                wildcard=wildcard,
                limit=20,
                viewer_actor_id=actor_id,
            )
            if self._emulator
            else []
        )
        lines = ["SMS_LIST:"]
        if rows:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                label = str(row.get("label") or row.get("thread") or "unknown")
                thread = str(row.get("thread") or "unknown")
                count = int(row.get("count") or 0)
                preview = str(row.get("last_preview") or "")
                lines.append(f"- {label} [{thread}] count={count} preview={preview}")
        else:
            lines.append("- (none)")

        lines.extend(
            [
                "NEXT_ACTIONS:",
                '- {"tool_call": "sms_read", "thread": "contact-slug", "limit": 20}',
                '- {"tool_call": "sms_write", "thread": "contact-slug", "from": "A", "to": "B", "message": "..."}',
                '- {"tool_call": "sms_schedule", "thread": "contact-slug", "from": "NPC", "to": "Player", "message": "...", "delay_seconds": 120}',
            ]
        )
        return "\n".join(lines)

    def _tool_sms_read(
        self,
        campaign_id: str,
        payload: dict[str, Any],
        *,
        actor_id: str | None = None,
    ) -> str:
        thread_raw = payload.get("thread")
        limit_raw = payload.get("limit", 20)
        thread = thread_raw.strip() if isinstance(thread_raw, str) and thread_raw.strip() else ""
        try:
            limit = max(1, min(40, int(limit_raw)))
        except Exception:
            limit = 20

        if not thread:
            return "SMS_READ_RESULT: invalid thread"

        canonical, matched, messages = (
            self._emulator.read_sms_thread(
                campaign_id,
                thread,
                limit=limit,
                viewer_actor_id=actor_id,
            )
            if self._emulator
            else (None, None, [])
        )
        lines = [f"SMS_READ_RESULT: thread={canonical or thread} matched={matched}"]
        if messages:
            for msg in messages[-limit:]:
                if not isinstance(msg, dict):
                    continue
                frm = str(msg.get("from") or "")
                to = str(msg.get("to") or "")
                text = str(msg.get("message") or "")
                try:
                    day = int(msg.get("day") or 0)
                except Exception:
                    day = 0
                try:
                    hour = max(0, min(23, int(msg.get("hour") or 0)))
                except Exception:
                    hour = 0
                try:
                    minute = max(0, min(59, int(msg.get("minute") or 0)))
                except Exception:
                    minute = 0
                lines.append(
                    f"- Day {day} {hour:02d}:{minute:02d} {frm} -> {to}: {text}"
                )
        else:
            lines.append("- (no messages)")
        return "\n".join(lines)

    def _tool_sms_write(
        self,
        campaign_id: str,
        payload: dict[str, Any],
        *,
        actor_id: str | None = None,
    ) -> str:
        # Handle batched messages array format that LLMs sometimes produce:
        #   {"tool_call": "sms_write", "thread": "elin", "messages": [
        #       {"from": "player-123", "text": "hello"},
        #       {"from": "elin-kivi", "text": "hi back"}]}
        messages_raw = payload.get("messages")
        if isinstance(messages_raw, list) and messages_raw:
            return self._tool_sms_write_batch(campaign_id, payload, messages_raw, actor_id=actor_id)

        thread_raw = payload.get("thread")
        sender_raw = payload.get("from", payload.get("sender"))
        recipient_raw = payload.get("to", payload.get("recipient"))
        message_raw = payload.get("message") or payload.get("content") or payload.get("text")
        direction_raw = str(payload.get("direction") or "").strip().lower()

        thread = thread_raw.strip() if isinstance(thread_raw, str) and thread_raw.strip() else ""
        sender = sender_raw.strip() if isinstance(sender_raw, str) and sender_raw.strip() else ""
        recipient = (
            recipient_raw.strip()
            if isinstance(recipient_raw, str) and recipient_raw.strip()
            else ""
        )
        message = message_raw.strip() if isinstance(message_raw, str) and message_raw.strip() else ""

        # Handle "direction: incoming" from final tool_calls — the model puts
        # the NPC in "recipient" but means the NPC is the *sender*.  Swap and
        # resolve the actual recipient from actor_id.  Thread key must stay
        # as the NPC name (the "other party"), not the player slug.
        if direction_raw == "incoming" and recipient and not sender:
            sender = recipient
            if not thread:
                thread = sender  # NPC name is the thread key
            if actor_id:
                emulator = self._emulator
                if emulator is not None:
                    player_slug = emulator._player_visibility_slug(actor_id)  # noqa: SLF001
                    if player_slug:
                        recipient = player_slug
                    else:
                        recipient = str(actor_id)
                else:
                    recipient = str(actor_id)
            else:
                recipient = ""

        # LLMs sometimes omit the thread field; default to recipient (standard pattern).
        if not thread:
            thread = recipient or sender

        if not thread or not sender or not recipient or not message:
            return "SMS_WRITE_RESULT: invalid payload"

        with self._session_factory() as session:
            latest_turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .order_by(Turn.id.desc())
                .first()
            )
            turn_id = int(latest_turn.id) if latest_turn is not None else 0

        if self._emulator is None:
            return "SMS_WRITE_RESULT: emulator unavailable"

        ok, reason = self._emulator.write_sms_thread(
            campaign_id,
            thread=thread,
            sender=sender,
            recipient=recipient,
            message=message,
            turn_id=turn_id,
            owner_actor_id=actor_id,
        )
        return f"SMS_WRITE_RESULT: stored={bool(ok)} reason={reason} thread={thread}"

    def _tool_sms_write_batch(
        self,
        campaign_id: str,
        payload: dict[str, Any],
        messages: list[Any],
        *,
        actor_id: str | None = None,
    ) -> str:
        thread_raw = payload.get("thread")
        thread = thread_raw.strip() if isinstance(thread_raw, str) and thread_raw.strip() else ""
        if not thread:
            return "SMS_WRITE_RESULT: invalid payload (no thread for messages batch)"

        with self._session_factory() as session:
            latest_turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .order_by(Turn.id.desc())
                .first()
            )
            turn_id = int(latest_turn.id) if latest_turn is not None else 0

        if self._emulator is None:
            return "SMS_WRITE_RESULT: emulator unavailable"

        # Collect all unique senders to derive recipients when not specified.
        senders: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                f = str(msg.get("from") or msg.get("sender") or "").strip()
                if f and f not in senders:
                    senders.append(f)

        stored = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            sender = str(msg.get("from") or msg.get("sender") or "").strip()
            text = str(msg.get("text") or msg.get("message") or "").strip()
            recipient = str(msg.get("to") or msg.get("recipient") or "").strip()
            if not recipient:
                # Derive: the other participant, or fall back to thread name.
                others = [s for s in senders if s != sender]
                recipient = others[0] if others else thread
            if not sender or not text:
                continue
            ok, _reason = self._emulator.write_sms_thread(
                campaign_id,
                thread=thread,
                sender=sender,
                recipient=recipient,
                message=text,
                turn_id=turn_id,
                owner_actor_id=actor_id,
            )
            if ok:
                stored += 1
        return f"SMS_WRITE_RESULT: stored={stored}/{len(messages)} thread={thread}"

    def _tool_sms_schedule(
        self,
        campaign_id: str,
        payload: dict[str, Any],
        *,
        actor_id: str | None = None,
    ) -> str:
        thread_raw = payload.get("thread")
        sender_raw = payload.get("from", payload.get("sender"))
        recipient_raw = payload.get("to", payload.get("recipient"))
        message_raw = payload.get("message")
        delay_raw = payload.get("delay_seconds", payload.get("delay"))
        delay_minutes_raw = payload.get("delay_minutes")

        thread = thread_raw.strip() if isinstance(thread_raw, str) and thread_raw.strip() else ""
        sender = sender_raw.strip() if isinstance(sender_raw, str) and sender_raw.strip() else ""
        recipient = (
            recipient_raw.strip()
            if isinstance(recipient_raw, str) and recipient_raw.strip()
            else ""
        )
        message = message_raw.strip() if isinstance(message_raw, str) and message_raw.strip() else ""

        # LLMs sometimes omit the thread field; default to recipient (standard pattern).
        if not thread:
            thread = recipient or sender

        if not thread or not sender or not recipient or not message:
            return "SMS_SCHEDULE_RESULT: invalid payload"

        if delay_raw is None and delay_minutes_raw is not None:
            try:
                delay_raw = int(delay_minutes_raw) * 60
            except Exception:
                delay_raw = None
        try:
            delay_seconds = int(delay_raw)
        except Exception:
            delay_seconds = 90

        if self._emulator is None:
            return "SMS_SCHEDULE_RESULT: emulator unavailable"

        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return "SMS_SCHEDULE_RESULT: campaign not found"
            latest_turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .order_by(Turn.id.desc())
                .first()
            )
        speed = self._emulator.get_timed_events_speed_multiplier(campaign) if campaign is not None else 1.0
        if speed > 0:
            delay_seconds = int(delay_seconds / speed)
        delay_seconds = max(15, min(86_400, delay_seconds))
        turn_id = int(latest_turn.id) if latest_turn is not None else 0
        ok, reason, applied_delay = self._emulator.schedule_sms_thread_delivery(
            campaign_id,
            thread=thread,
            sender=sender,
            recipient=recipient,
            message=message,
            delay_seconds=delay_seconds,
            turn_id=turn_id,
            owner_actor_id=actor_id,
        )
        return (
            "SMS_SCHEDULE_RESULT: "
            f"scheduled={bool(ok)} reason={reason} "
            f"thread={thread} delay_seconds={applied_delay if ok else delay_seconds} "
            "delivery_visibility=hidden_until_delivery interruptible=false"
        )

    def _tool_story_outline(self, campaign_id: str, payload: dict[str, Any]) -> str:
        chapter_key_raw = payload.get("chapter")
        chapter_key = (
            chapter_key_raw.strip().lower()
            if isinstance(chapter_key_raw, str) and chapter_key_raw.strip()
            else None
        )

        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return "STORY_OUTLINE: campaign not found"
            state = self._parse_json(campaign.state_json, {})

        outline = state.get("story_outline") if isinstance(state, dict) else None
        if not isinstance(outline, dict):
            return "STORY_OUTLINE: none"

        if chapter_key:
            chapters = outline.get("chapters", [])
            if isinstance(chapters, list):
                for index, chapter in enumerate(chapters):
                    if not isinstance(chapter, dict):
                        continue
                    title = str(chapter.get("title") or "").strip()
                    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
                    if chapter_key in {slug, str(index), str(index + 1)}:
                        data = {"index": index, "chapter": chapter}
                        return (
                            "STORY_OUTLINE_CHAPTER:\n"
                            f"{json.dumps(data, ensure_ascii=True)}"
                        )
            return "STORY_OUTLINE_CHAPTER: not found"

        current = {
            "current_chapter": state.get("current_chapter"),
            "current_scene": state.get("current_scene"),
            "story_outline": outline,
        }
        text = json.dumps(current, ensure_ascii=True)
        if len(text) > 10000:
            text = text[:9999] + "..."
        return f"STORY_OUTLINE:\n{text}"

    def _tool_plot_plan(self, campaign_id: str, payload: dict[str, Any]) -> str:
        plans = payload.get("plans")
        if isinstance(plans, dict):
            plans = [plans]
        if not isinstance(plans, list):
            return "PLOT_PLAN_RESULT: invalid payload"

        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return "PLOT_PLAN_RESULT: campaign not found"
            state = self._parse_json(campaign.state_json, {})
            if not isinstance(state, dict):
                state = {}
            threads = state.get("_plot_threads")
            if not isinstance(threads, dict):
                threads = {}
            updated = 0
            removed = 0
            for row in plans[:12]:
                if not isinstance(row, dict):
                    continue
                slug_raw = row.get("thread") or row.get("slug")
                slug = re.sub(r"[^a-z0-9]+", "-", str(slug_raw or "").strip().lower()).strip("-")[:80]
                if not slug:
                    continue
                if bool(row.get("remove") or row.get("delete") or row.get("_delete")):
                    if slug in threads:
                        threads.pop(slug, None)
                        removed += 1
                    continue
                item = dict(threads.get(slug) or {"thread": slug, "status": "active"})
                for field in ("setup", "intended_payoff", "resolution"):
                    if row.get(field) is not None:
                        item[field] = " ".join(str(row.get(field) or "").split())[:260]
                if row.get("target_turns") is not None:
                    try:
                        item["target_turns"] = max(1, min(250, int(row.get("target_turns"))))
                    except Exception:
                        item["target_turns"] = int(item.get("target_turns") or 8)
                deps = row.get("dependencies")
                if isinstance(deps, list):
                    clean = []
                    for dep in deps[:8]:
                        text = " ".join(str(dep or "").split())[:120]
                        if text:
                            clean.append(text)
                    item["dependencies"] = clean
                status = str(row.get("status") or item.get("status") or "active").strip().lower()
                if row.get("resolve"):
                    status = "resolved"
                if status not in {"active", "resolved"}:
                    status = "active"
                item["status"] = status
                threads[slug] = item
                updated += 1
            state["_plot_threads"] = threads
            campaign.state_json = json.dumps(state, ensure_ascii=True)
            campaign.updated_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
        return f"PLOT_PLAN_RESULT: updated={updated} removed={removed} total={len(threads)}"

    def _tool_chapter_plan(self, campaign_id: str, payload: dict[str, Any]) -> str:
        action = str(payload.get("action") or "create").strip().lower()
        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return "CHAPTER_PLAN_RESULT: campaign not found"
            state = self._parse_json(campaign.state_json, {})
            if not isinstance(state, dict):
                state = {}
            if bool(state.get("on_rails")):
                return "CHAPTER_PLAN_RESULT: ignored (on_rails enabled)"
            chapters = state.get("_chapter_plan")
            if not isinstance(chapters, dict):
                chapters = {}

            chapter_obj = payload.get("chapter")
            if isinstance(chapter_obj, dict):
                slug_raw = chapter_obj.get("slug") or chapter_obj.get("title")
            else:
                slug_raw = payload.get("chapter") or payload.get("slug")
            slug = re.sub(r"[^a-z0-9]+", "-", str(slug_raw or "").strip().lower()).strip("-")[:80]
            changed = 0

            if action in {"create", "update"}:
                if not slug:
                    return "CHAPTER_PLAN_RESULT: missing slug"
                row = dict(chapters.get(slug) or {"slug": slug, "status": "active"})
                if isinstance(chapter_obj, dict):
                    if chapter_obj.get("title") is not None:
                        row["title"] = " ".join(str(chapter_obj.get("title") or "").split())[:120]
                    if chapter_obj.get("summary") is not None:
                        row["summary"] = " ".join(str(chapter_obj.get("summary") or "").split())[:260]
                    scenes = chapter_obj.get("scenes")
                    if isinstance(scenes, list):
                        row["scenes"] = [
                            re.sub(r"[^a-z0-9]+", "-", str(scene or "").strip().lower()).strip("-")[:80]
                            for scene in scenes[:20]
                            if str(scene or "").strip()
                        ]
                    if chapter_obj.get("current_scene") is not None:
                        row["current_scene"] = re.sub(
                            r"[^a-z0-9]+",
                            "-",
                            str(chapter_obj.get("current_scene") or "").strip().lower(),
                        ).strip("-")[:80]
                    if chapter_obj.get("active") is not None:
                        row["status"] = (
                            "active" if bool(chapter_obj.get("active")) else "resolved"
                        )
                chapters[slug] = row
                changed += 1
            elif action == "advance_scene":
                if slug and slug in chapters:
                    row = dict(chapters.get(slug) or {})
                    to_scene = payload.get("to_scene") or payload.get("scene")
                    scene_slug = re.sub(r"[^a-z0-9]+", "-", str(to_scene or "").strip().lower()).strip("-")[:80]
                    if scene_slug:
                        row["current_scene"] = scene_slug
                        scenes = row.get("scenes")
                        if not isinstance(scenes, list):
                            scenes = []
                        if scene_slug not in scenes:
                            scenes.append(scene_slug)
                        row["scenes"] = scenes[:20]
                    row["status"] = "active"
                    chapters[slug] = row
                    changed += 1
            elif action in {"resolve", "close"}:
                if slug and slug in chapters:
                    row = dict(chapters.get(slug) or {})
                    row["status"] = "resolved"
                    row["resolution"] = " ".join(str(payload.get("resolution") or "").split())[:260]
                    chapters[slug] = row
                    changed += 1

            state["_chapter_plan"] = chapters
            campaign.state_json = json.dumps(state, ensure_ascii=True)
            campaign.updated_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
        return f"CHAPTER_PLAN_RESULT: updated={changed} total={len(chapters)}"

    def _tool_consequence_log(self, campaign_id: str, payload: dict[str, Any]) -> str:
        def _iter_rows(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, dict):
                return [value]
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
            return []

        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return "CONSEQUENCE_LOG_RESULT: campaign not found"
            state = self._parse_json(campaign.state_json, {})
            if not isinstance(state, dict):
                state = {}
            rows = state.get("_consequences")
            if not isinstance(rows, dict):
                rows = {}

            added = 0
            resolved = 0
            removed = 0
            for entry in _iter_rows(payload.get("add")):
                trigger = " ".join(str(entry.get("trigger") or "").split())[:240]
                consequence = " ".join(str(entry.get("consequence") or "").split())[:300]
                if not trigger or not consequence:
                    continue
                cid_raw = entry.get("id") or entry.get("slug") or trigger[:60]
                cid = re.sub(r"[^a-z0-9]+", "-", str(cid_raw or "").strip().lower()).strip("-")[:90]
                if not cid:
                    continue
                severity = str(entry.get("severity") or "low").strip().lower()
                if severity not in {"low", "moderate", "high", "critical"}:
                    severity = "low"
                row = dict(rows.get(cid) or {})
                row.update(
                    {
                        "id": cid,
                        "trigger": trigger,
                        "consequence": consequence,
                        "severity": severity,
                        "status": "active",
                        "resolution": str(row.get("resolution") or "")[:260],
                    }
                )
                rows[cid] = row
                added += 1

            for entry in _iter_rows(payload.get("resolve")):
                cid_raw = entry.get("id") or entry.get("slug") or entry.get("trigger")
                cid = re.sub(r"[^a-z0-9]+", "-", str(cid_raw or "").strip().lower()).strip("-")[:90]
                if not cid or cid not in rows:
                    continue
                row = dict(rows.get(cid) or {})
                row["status"] = "resolved"
                row["resolution"] = " ".join(
                    str(entry.get("resolution") or row.get("resolution") or "resolved").split()
                )[:260]
                rows[cid] = row
                resolved += 1

            remove_keys = payload.get("remove")
            if isinstance(remove_keys, list):
                for raw in remove_keys:
                    cid = re.sub(r"[^a-z0-9]+", "-", str(raw or "").strip().lower()).strip("-")[:90]
                    if cid and cid in rows:
                        rows.pop(cid, None)
                        removed += 1

            state["_consequences"] = rows
            campaign.state_json = json.dumps(state, ensure_ascii=True)
            campaign.updated_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
        return f"CONSEQUENCE_LOG_RESULT: added={added} resolved={resolved} removed={removed} total={len(rows)}"

    async def _tool_song_search(
        self,
        campaign_id: str,
        payload: dict[str, Any],
    ) -> str:
        query = " ".join(
            str(
                payload.get("query")
                or payload.get("song")
                or payload.get("search")
                or ""
            ).split()
        ).strip()
        if not query:
            return "SONG_SEARCH_RESULT: missing query"

        search_task = asyncio.create_task(
            asyncio.to_thread(_search_youtube_first_result, query)
        )
        try:
            result = await asyncio.wait_for(
                search_task,
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            search_task.cancel()
            logger.warning(
                "song_search timeout for campaign=%s query=%r",
                campaign_id,
                query,
            )
            return f"SONG_SEARCH_RESULT: query={query!r} found=false delivered=false timeout=true"
        if not isinstance(result, dict):
            return f"SONG_SEARCH_RESULT: query={query!r} found=false delivered=false"

        title = " ".join(str(result.get("title") or query).split()).strip()
        url = str(result.get("url") or "").strip()
        channel = " ".join(str(result.get("channel") or "").split()).strip()
        if not url:
            return f"SONG_SEARCH_RESULT: query={query!r} found=false delivered=false"

        sender = _neutralize_discord_mentions(
            payload.get("sender") or payload.get("from") or ""
        )
        message = _neutralize_discord_mentions(
            payload.get("message") or payload.get("caption") or ""
        )
        lines = []
        if sender:
            lines.append(f"**[Song] {sender}**")
        else:
            lines.append("**[Song]**")
        if message:
            lines.append(message)
        title_line = title
        if channel:
            title_line = f"{title_line} - {channel}"
        if title_line and title_line != url:
            lines.append(title_line)
        lines.append(url)
        delivery_text = "\n".join(line for line in lines if line).strip()

        delivered = False
        emulator = self._emulator
        notification_port = emulator.notification_port if emulator is not None else None
        if notification_port is not None:
            try:
                await notification_port.send_channel_message(
                    campaign_id=str(campaign_id),
                    message=delivery_text,
                )
                delivered = True
            except Exception:
                logger.debug(
                    "song_search delivery failed for campaign=%s query=%r",
                    campaign_id,
                    query,
                    exc_info=True,
                )

        title_json = json.dumps(title, ensure_ascii=True)
        channel_json = json.dumps(channel, ensure_ascii=True)
        url_json = json.dumps(url, ensure_ascii=True)
        return (
            "SONG_SEARCH_RESULT: "
            f'{{"query":{json.dumps(query, ensure_ascii=True)},"title":{title_json},"channel":{channel_json},'
            f'"url":{url_json},"delivered":{str(delivered).lower()}}}'
        )

    async def _execute_tool_call(
        self,
        campaign_id: str,
        payload: dict[str, Any],
        *,
        actor_id: str | None = None,
    ) -> str:
        tool = payload.get("tool_call")
        if not isinstance(tool, str):
            return "TOOL_ERROR: missing tool_call"
        name = tool.strip().lower()

        if name == "memory_search":
            return self._tool_memory_search(campaign_id, payload, actor_id=actor_id)
        if name == "memory_terms":
            return self._tool_memory_terms(campaign_id, payload)
        if name == "source_browse":
            return self._tool_source_browse(campaign_id, payload)
        if name.startswith("source") and name != "source_browse":
            # Catch all hallucinated source_* tool names and redirect to the real tools.
            # If the payload has a document_key, treat it as a source_browse.
            if payload.get("document_key"):
                return self._tool_source_browse(campaign_id, payload)
            # Otherwise treat as memory_search with source scope.
            queries = payload.get("queries") or payload.get("query") or payload.get("key")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list) or not queries:
                queries = [str(payload.get("keyword") or payload.get("term") or "")]
            redirected = {"tool_call": "memory_search", "category": "source", "queries": queries}
            return self._tool_memory_search(campaign_id, redirected, actor_id=actor_id)
        if name == "communication_rules":
            return self._tool_communication_rules(payload)
        if name in {"autobiography_append", "autobiography_update"}:
            return self._tool_autobiography_append(campaign_id, payload)
        if name == "autobiography_compress":
            return await self._tool_autobiography_compress(campaign_id, payload)
        if name == "name_generate":
            return self._tool_name_generate(campaign_id, payload)
        if name == "character_cards":
            return self._tool_character_cards(campaign_id, payload, actor_id=actor_id)
        if name == "memory_turn":
            return self._tool_memory_turn(campaign_id, payload, actor_id=actor_id)
        if name == "memory_store":
            return self._tool_memory_store(campaign_id, payload)
        if name == "sms_list":
            return self._tool_sms_list(campaign_id, payload, actor_id=actor_id)
        if name == "sms_read":
            return self._tool_sms_read(campaign_id, payload, actor_id=actor_id)
        if name == "sms_write":
            return self._tool_sms_write(campaign_id, payload, actor_id=actor_id)
        if name == "sms_schedule":
            return self._tool_sms_schedule(campaign_id, payload, actor_id=actor_id)
        if name == "story_outline":
            return self._tool_story_outline(campaign_id, payload)
        if name == "plot_plan":
            return self._tool_plot_plan(campaign_id, payload)
        if name == "chapter_plan":
            return self._tool_chapter_plan(campaign_id, payload)
        if name == "consequence_log":
            return self._tool_consequence_log(campaign_id, payload)
        if name == "song_search":
            return await self._tool_song_search(campaign_id, payload)
        if name == "ready_to_write":
            return "READY_TO_WRITE_ACK"
        if name == "recent_turns":
            return self._tool_recent_turns(campaign_id, payload, actor_id=actor_id)
        return f"TOOL_ERROR: unsupported tool_call '{name}'"

    async def _resolve_payload(
        self,
        campaign_id: str,
        actor_id: str,
        action_text: str,
        system_prompt: str,
        user_prompt: str,
        final_system_prompt: str,
        final_user_prompt: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any] | None:
        await _notify_progress(progress, "thinking")
        logger.debug("_resolve_payload: sending initial LLM request for campaign=%s actor=%s", campaign_id, actor_id)
        self._zork_log(
            f"RESEARCH REQUEST campaign={campaign_id}",
            f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
        )
        first = await self._completion.complete(
            system_prompt,
            user_prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        logger.debug("_resolve_payload: initial LLM response length=%d", len(first or ""))
        self._zork_log(f"RESEARCH RESPONSE campaign={campaign_id}", first or "(empty)")
        payload = self._parse_model_payload(first)
        if payload is None:
            logger.warning("_resolve_payload: initial payload parse failed → JSON retry")
            self._zork_log(f"PARSE FAILED campaign={campaign_id}", first or "(empty)")
            await _notify_progress(progress, "refining")
            retry_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Your previous response was not valid JSON.  "
                "Return ONLY a single JSON object with no markdown fences, "
                "no commentary, no prose.  Begin with {{ and end with }}."
            )
            retry_resp = await self._completion.complete(
                system_prompt,
                retry_prompt,
                temperature=max(0.1, self._temperature - 0.3),
                max_tokens=self._max_tokens,
            )
            self._zork_log(f"PARSE RETRY RESPONSE campaign={campaign_id}", retry_resp or "(empty)")
            payload = self._parse_model_payload(retry_resp)
            if payload is None:
                logger.warning("_resolve_payload: retry parse also failed → DeterministicLLM fallback")
                return None

        tool_history = ""
        tool_history_entries: list[dict[str, str]] = []
        used_tool_names: set[str] = set()
        seen_tool_signatures: set[str] = set()
        research_sms_writes: list[tuple[str, str]] = []  # (recipient_norm, content_norm)
        memory_obligation_met = False
        last_memory_turn_ids: list[int] = []
        emulator = self._emulator
        if emulator is None:
            logger.warning("_resolve_payload: emulator is None (second check)")
            return None

        def _refresh_tool_history() -> None:
            nonlocal tool_history
            tool_history = "".join(
                f"\n\n{str(entry.get('text') or '').strip()}"
                for entry in tool_history_entries
                if str(entry.get("text") or "").strip()
            )

        def _append_tool_history_entry(tool_name: str, text: str) -> None:
            nonlocal last_memory_turn_ids
            tool_key = str(tool_name or "").strip().lower()
            tool_history_entries.append(
                {
                    "tool_name": tool_key,
                    "text": str(text or "").strip(),
                }
            )
            candidate_turn_ids = self._memory_tool_turn_ids_from_text(text)
            if tool_key in {"memory_search", "memory_turn"}:
                last_memory_turn_ids = candidate_turn_ids
            elif candidate_turn_ids:
                last_memory_turn_ids = candidate_turn_ids
            _refresh_tool_history()

        def _apply_keep_memory_turns(keep_turn_ids: list[int]) -> None:
            nonlocal tool_history_entries, last_memory_turn_ids
            keep_set = {turn_id for turn_id in keep_turn_ids if turn_id > 0}
            if not keep_set:
                return
            pruned_entries: list[dict[str, str]] = []
            for entry in tool_history_entries:
                entry_tool = str(entry.get("tool_name") or "").strip().lower()
                entry_text = str(entry.get("text") or "")
                if entry_tool in {"memory_search", "memory_turn"}:
                    pruned_text = self._pruned_memory_tool_text(
                        entry_tool,
                        entry_text,
                        keep_set,
                    )
                    if pruned_text:
                        pruned_entries.append(
                            {"tool_name": entry_tool, "text": pruned_text}
                        )
                    continue
                pruned_entries.append(
                    {
                        "tool_name": entry_tool,
                        "text": entry_text.strip(),
                    }
                )
            tool_history_entries = pruned_entries
            last_memory_turn_ids = [
                turn_id for turn_id in last_memory_turn_ids if turn_id in keep_set
            ]
            _refresh_tool_history()

        is_initial_tool_call = emulator._is_tool_call(payload)  # noqa: SLF001
        logger.debug(
            "_resolve_payload: initial parse ok — is_tool_call=%s tool=%s has_narration=%s keys=%s",
            is_initial_tool_call,
            payload.get("tool_call", "(none)"),
            bool(payload.get("narration")),
            list(payload.keys())[:8],
        )
        memory_lookup_enabled = "memory_lookup_enabled: true" in user_prompt.lower()
        if (
            not emulator._is_tool_call(payload)  # noqa: SLF001
            and memory_lookup_enabled
            and self._should_force_auto_memory_search(action_text)
        ):
            forced_queries = self._derive_auto_memory_queries(
                campaign_id,
                actor_id,
                action_text,
                limit=4,
            )
            if forced_queries:
                tool_payload = {"tool_call": "memory_search", "queries": forced_queries}
                tool_result = await self._execute_tool_call(
                    campaign_id, tool_payload, actor_id=actor_id
                )
                _append_tool_history_entry("memory_search", tool_result)
                memory_obligation_met = True
                augmented_prompt = (
                    f"{user_prompt}\n"
                    f"{tool_history}\n\n"
                    'Use the memory results above. Continue with your next {"tool_call": "..."} or, if research is complete, '
                    'return {"tool_call": "ready_to_write", "speakers": [...], "listeners": [...]}. '
                    "Do NOT return final narration/state JSON during research."
                )
                self._zork_log(f"FORCED MEMORY SEARCH campaign={campaign_id}", augmented_prompt)
                nxt = await self._completion.complete(
                    system_prompt,
                    augmented_prompt,
                    temperature=max(0.1, self._temperature - 0.2),
                    max_tokens=self._max_tokens,
                )
                self._zork_log(f"FORCED MEMORY SEARCH RESPONSE campaign={campaign_id}", nxt or "(empty)")
                payload = self._parse_model_payload(nxt)
                self._bump_auto_fix_counter(campaign_id, "forced_memory_search")
                if payload is None:
                    logger.warning("_resolve_payload: forced memory search response unparseable")
                    return None

        for _round_idx in range(max(0, self._max_tool_rounds)):
            if not emulator._is_tool_call(payload):  # noqa: SLF001
                logger.debug("_resolve_payload: round %d — not a tool call, breaking loop", _round_idx)
                break
            tool_name = str(payload.get("tool_call") or "").strip().lower()
            logger.debug("_resolve_payload: round %d — tool_call=%s keys=%s", _round_idx, tool_name, list(payload.keys())[:8])
            if tool_name:
                used_tool_names.add(tool_name)
            if tool_name == "ready_to_write":
                break
            if not memory_lookup_enabled and tool_name.startswith("memory_"):
                _append_tool_history_entry(
                    "system_note",
                    "\n\nMEMORY_TOOLS_DISABLED: Long-term memory lookup is disabled for this turn "
                    "(early campaign context still fits prompt budget). "
                    "Do NOT call memory_* tools; continue with direct context or non-memory tools."
                )
                augmented_prompt = (
                    f"{user_prompt}\n"
                    f"{tool_history}\n\n"
                    'Continue with a non-memory {"tool_call": "..."} or, if research is complete, '
                    'return {"tool_call": "ready_to_write", "speakers": [...], "listeners": [...]}. '
                    "Do NOT return final narration/state JSON during research."
                )
                self._zork_log(f"MEMORY TOOL DISABLED AUGMENTED RESPONSE campaign={campaign_id}", augmented_prompt)
                nxt = await self._completion.complete(
                    system_prompt,
                    augmented_prompt,
                    temperature=max(0.1, self._temperature - 0.2),
                    max_tokens=self._max_tokens,
                )
                self._zork_log(f"MEMORY TOOL DISABLED RESPONSE campaign={campaign_id}", nxt or "(empty)")
                payload = self._parse_model_payload(nxt)
                if payload is None:
                    logger.warning("_resolve_payload: memory-disabled retry unparseable")
                    return None
                continue
            tool_signature = self._tool_call_signature(payload)
            if tool_signature and tool_signature in seen_tool_signatures:
                _append_tool_history_entry(
                    "system_note",
                    "\n\nTOOL_DEDUP_RESULT: duplicate tool_call payload already executed this turn; skipped. "
                    "Do NOT repeat identical tool calls. Use a distinct tool/payload or "
                    'return {"tool_call": "ready_to_write", "speakers": [...], "listeners": [...]} to proceed.'
                )
                augmented_prompt = (
                    f"{user_prompt}\n"
                    f"{tool_history}\n\n"
                    'Use a distinct {"tool_call": "..."} or, if research is complete, '
                    'return {"tool_call": "ready_to_write", "speakers": [...], "listeners": [...]}. '
                    "Do NOT return final narration/state JSON during research."
                )
                self._zork_log(f"TOOL DEDUP AUGMENTED RESPONSE campaign={campaign_id}", augmented_prompt)
                nxt = await self._completion.complete(
                    system_prompt,
                    augmented_prompt,
                    temperature=max(0.1, self._temperature - 0.2),
                    max_tokens=self._max_tokens,
                )
                self._zork_log(f"TOOL DEDUP RESPONSE campaign={campaign_id}", nxt or "(empty)")
                payload = self._parse_model_payload(nxt)
                if payload is None:
                    logger.warning("_resolve_payload: dedup retry unparseable")
                    return None
                continue
            if tool_signature:
                seen_tool_signatures.add(tool_signature)
            if tool_name == "memory_search":
                search_within = str(payload.get("search_within") or "").strip().lower()
                if search_within == "last_results" and last_memory_turn_ids:
                    payload = dict(payload)
                    payload["search_within_turn_ids"] = list(last_memory_turn_ids)
                elif search_within == "last_results" and not last_memory_turn_ids:
                    _append_tool_history_entry(
                        "system_note",
                        "MEMORY_SEARCH_WITHIN_RESULT: no previous turn hits are available to narrow within yet; search ran without last_results scoping.",
                    )
            await _notify_progress(progress, "tool_call", {"tool": tool_name})
            self._zork_log(f"TOOL CALL campaign={campaign_id}", f"tool={tool_name}\npayload={json.dumps(payload, default=str)}")
            tool_result = await self._execute_tool_call(
                campaign_id,
                payload,
                actor_id=actor_id,
            )
            self._zork_log(f"TOOL RESULT campaign={campaign_id}", f"tool={tool_name}\nresult={str(tool_result)}")
            # Track sms_write calls executed during research for dedup against
            # final inline tool_calls (models often re-include the outgoing SMS).
            if tool_name in ("sms_write", "sms_schedule") and isinstance(tool_result, str) and "stored=True" in tool_result:
                _sms_recipient = " ".join(str(payload.get("recipient") or payload.get("thread") or payload.get("to") or "").strip().lower().split())
                _sms_content = " ".join(str(payload.get("content") or payload.get("text") or payload.get("message") or "").strip().lower().split())
                if _sms_recipient and _sms_content:
                    research_sms_writes.append((_sms_recipient, _sms_content))
            _append_tool_history_entry(tool_name, tool_result)
            keep_memory_turns = self._memory_tool_turn_id_list(
                payload.get("keep_memory_turns")
            )
            if keep_memory_turns:
                _apply_keep_memory_turns(keep_memory_turns)
            if not memory_obligation_met and tool_name in (
                "recent_turns",
                "memory_search",
            ):
                memory_obligation_met = True
                # Strip MEMORY_RECALL_NEXT_ACTIONS from the tool result we just
                # appended — the memory-tool JSON examples tempt the model into
                # calling them again instead of proceeding to ready_to_write.
                for entry in tool_history_entries:
                    text = str(entry.get("text") or "")
                    marker = "MEMORY_RECALL_NEXT_ACTIONS:"
                    if marker in text:
                        entry["text"] = text[: text.index(marker)].rstrip()
                _refresh_tool_history()
                _append_tool_history_entry(
                    "system_note",
                    "\n\n[MEMORY OBLIGATION MET — you have satisfied the mandatory memory lookup for this turn. "
                    "Do not call additional memory tools unless you genuinely need specific older information. "
                    "You may now proceed to ready_to_write or use non-memory tools.]"
                )
            augmented_prompt = (
                f"{user_prompt}\n"
                f"{tool_history}\n\n"
                'Use the tool results above. Continue with your next {"tool_call": "..."} or, if research is complete, '
                'return {"tool_call": "ready_to_write", "speakers": [...], "listeners": [...]}. '
                "Do NOT return final narration/state JSON during research."
            )
            self._zork_log(f"AUGMENTED API REQUEST campaign={campaign_id} tool={tool_name}", augmented_prompt)
            nxt = await self._completion.complete(
                system_prompt,
                augmented_prompt,
                temperature=max(0.1, self._temperature - 0.2),
                max_tokens=self._max_tokens,
            )
            self._zork_log(f"AUGMENTED API RESPONSE campaign={campaign_id} tool={tool_name}", nxt or "(empty)")
            payload = self._parse_model_payload(nxt)
            if payload is None:
                logger.warning("_resolve_payload: post-tool-execution response unparseable (tool=%s)", tool_name)
                return None
            if not emulator._is_tool_call(payload):  # noqa: SLF001
                self._zork_log(
                    f"RESEARCH PHASE PROTOCOL VIOLATION campaign={campaign_id}",
                    "Model returned final JSON during research; requesting ready_to_write restage.",
                )
                restaged_payload: dict[str, Any] | None = None
                for attempt in range(2):
                    restage_prompt = (
                        f"{user_prompt}\n"
                        f"{tool_history}\n\n"
                        "RESEARCH_PHASE_PROTOCOL_VIOLATION: You returned final narration/state JSON during research.\n"
                        'Do NOT narrate yet. Return ONLY {"tool_call":"ready_to_write","speakers":[...],"listeners":[...]} now.\n'
                        "speakers = only characters who will actually speak or take meaningful visible action.\n"
                        "listeners = only the direct recipients/observers whose shared knowledge matters for those beats.\n"
                    )
                    self._zork_log(
                        f"RESEARCH RESTAGE REQUEST campaign={campaign_id} attempt={attempt + 1}",
                        restage_prompt,
                    )
                    restage_response = await self._completion.complete(
                        system_prompt,
                        restage_prompt,
                        temperature=max(0.1, self._temperature - 0.2),
                        max_tokens=self._max_tokens,
                    )
                    self._zork_log(
                        f"RESEARCH RESTAGE RESPONSE campaign={campaign_id} attempt={attempt + 1}",
                        restage_response or "(empty)",
                    )
                    candidate = self._parse_model_payload(restage_response)
                    if (
                        isinstance(candidate, dict)
                        and emulator._is_tool_call(candidate)  # noqa: SLF001
                        and str(candidate.get("tool_call") or "").strip().lower()
                        == "ready_to_write"
                    ):
                        restaged_payload = candidate
                        break
                if restaged_payload is None:
                    logger.warning("_resolve_payload: model refused ready_to_write restage after tool=%s", tool_name)
                    return None
                payload = restaged_payload

        if emulator._is_tool_call(payload) and str(payload.get("tool_call") or "").strip().lower() != "ready_to_write":  # noqa: E501, SLF001
            pending_tool_name = str(payload.get("tool_call") or "").strip().lower()
            logger.warning(
                "_resolve_payload: max tool rounds exceeded, still has tool_call=%s; forcing finalization",
                payload.get("tool_call"),
            )
            pending_tool_json = json.dumps(payload, ensure_ascii=False, default=str)
            _append_tool_history_entry(
                "system_note",
                "\n\nTOOL_ROUND_LIMIT_REACHED: Research/tool rounds are exhausted. "
                "Do NOT call any more research tools. Proceed to final narration/state JSON now. "
                "If the pending action is appropriate for final inline tool_calls "
                "(sms_write, sms_schedule, plot_plan, chapter_plan), move it under final JSON tool_calls instead of returning a standalone tool_call.\n"
                f"PENDING_TOOL_CALL: {pending_tool_json}",
            )
            payload = {
                "tool_call": "ready_to_write",
                "speakers": [],
                "listeners": [],
                "_forced_after_tool_round_limit": pending_tool_name or "unknown",
            }

        # Dedicated ready_to_write finalization: the payload is the tool call
        # itself (no narration), so issue the actual writing call with craft guidance.
        if emulator._is_tool_call(payload) and str(payload.get("tool_call") or "").strip().lower() == "ready_to_write":  # noqa: E501, SLF001
            await _notify_progress(progress, "writing")
            _pc_names = emulator.get_pc_names(campaign_id)
            _pc_reminder = ""
            if _pc_names:
                _pc_reminder = (
                    "\nPLAYER_CHARACTERS (real humans — do NOT write their dialogue, actions, decisions, "
                    f"emotional reactions, facial expressions, or movement): {', '.join(_pc_names)}\n"
                )
            _ready_speakers = payload.get("speakers") or []
            _ready_listeners = payload.get("listeners") or []
            if isinstance(_ready_speakers, str):
                _ready_speakers = [_ready_speakers]
            if isinstance(_ready_listeners, str):
                _ready_listeners = [_ready_listeners]
            # Auto-inject SMS recipient as speaker when action is an SMS.
            _sms_intent = emulator._extract_inline_sms_intent(action_text)  # noqa: SLF001
            if _sms_intent is not None:
                _sms_recipient_slug = emulator._player_slug_key(_sms_intent[0])  # noqa: SLF001
                if _sms_recipient_slug:
                    _all_ready = [
                        emulator._player_slug_key(s) for s in list(_ready_speakers) + list(_ready_listeners)  # noqa: SLF001
                    ]
                    if _sms_recipient_slug not in _all_ready:
                        _ready_speakers = list(_ready_speakers) + [_sms_intent[0]]
            _shared_context_block = ""
            _speaker_continuity_block = ""
            _final_character_cards_block = ""
            _final_location_cards_block = ""
            _final_prompt_body, _final_prompt_tail = emulator._split_prompt_tail(final_user_prompt)  # noqa: SLF001
            _clean_final_user_prompt = emulator._strip_recent_turns_prompt_sections(_final_prompt_body)  # noqa: SLF001
            _clean_tool_history = emulator._strip_recent_turns_prompt_sections(tool_history)  # noqa: SLF001
            with emulator._session_factory() as session:  # noqa: SLF001
                campaign = session.get(Campaign, str(campaign_id))
            player = emulator.get_or_create_player(campaign_id, actor_id)
            if campaign is not None and player is not None:
                campaign_state = emulator.get_campaign_state(campaign)  # noqa: SLF001
                campaign_characters = emulator.get_campaign_characters(campaign)  # noqa: SLF001
                player_state = emulator.get_player_state(player)
                viewer_location_key = emulator._room_key_from_player_state(player_state).lower()  # noqa: SLF001
                viewer_private_context = emulator._active_private_context_from_state(player_state)  # noqa: SLF001
                viewer_private_context_key = str(
                    (viewer_private_context or {}).get("context_key") or ""
                ).strip()
                registry = emulator._campaign_player_registry(campaign_id, emulator._session_factory)  # noqa: SLF001
                viewer_slug = str(
                    (registry.get("by_actor_id", {}).get(actor_id) or {}).get("slug")
                    or emulator._player_visibility_slug(actor_id)  # noqa: SLF001
                ).strip()
                player_slugs = {
                    emulator._player_slug_key(info.get("slug") or info.get("name") or "")  # noqa: SLF001
                    for info in registry.get("by_actor_id", {}).values()
                    if isinstance(info, dict)
                    and emulator._player_slug_key(info.get("slug") or info.get("name") or "")  # noqa: SLF001
                }
                speaker_npc_slugs = {
                    emulator._player_slug_key(raw_slug)  # noqa: SLF001
                    for raw_slug in list(_ready_speakers)
                    if emulator._player_slug_key(raw_slug)  # noqa: SLF001
                }
                speaker_npc_slugs = {
                    slug for slug in speaker_npc_slugs if slug and slug not in player_slugs and slug != viewer_slug
                }
                listener_npc_slugs = {
                    emulator._player_slug_key(raw_slug)  # noqa: SLF001
                    for raw_slug in list(_ready_listeners)
                    if emulator._player_slug_key(raw_slug)  # noqa: SLF001
                }
                listener_npc_slugs = {
                    slug for slug in listener_npc_slugs if slug and slug not in player_slugs and slug != viewer_slug
                }
                scene_npc_slugs = speaker_npc_slugs.union(listener_npc_slugs)
                final_character_entries: list[dict[str, Any]] = []
                for slug in sorted(scene_npc_slugs):
                    row = campaign_characters.get(slug) if isinstance(campaign_characters, dict) else None
                    if not isinstance(row, dict):
                        continue
                    prompt_row = dict(row)
                    prompt_row["_slug"] = slug
                    final_character_entries.append(prompt_row)
                if final_character_entries:
                    final_character_cards = emulator._build_character_cards_for_prompt(  # noqa: SLF001
                        final_character_entries,
                        characters=campaign_characters,
                        campaign_state=campaign_state,
                        player_state=player_state,
                        include_critical_fields=True,
                        fully_expand=True,
                    )
                    if final_character_cards:
                        _final_character_cards_block = (
                            "\nFINAL_CHARACTER_CARDS: "
                            f"{emulator._dump_json(final_character_cards)}\n"  # noqa: SLF001
                        )
                location_cards_map = emulator._location_cards_from_state(campaign_state, player_state)  # noqa: SLF001
                final_location_cards = emulator._build_location_cards_for_prompt(  # noqa: SLF001
                    location_cards_map,
                    player_state=player_state,
                    include_active_location=True,
                )
                if final_location_cards:
                    active_location = str(player_state.get("location") or "").strip().lower()
                    narrowed_location_cards = [
                        row for row in final_location_cards
                        if str(row.get("slug") or "").strip().lower() == active_location
                    ]
                    if not narrowed_location_cards:
                        narrowed_location_cards = final_location_cards[:1]
                    _final_location_cards_block = (
                        "\nFINAL_LOCATION_CARDS: "
                        f"{emulator._dump_json(narrowed_location_cards)}\n"  # noqa: SLF001
                    )
                turns = emulator.get_recent_turns(campaign_id, limit=0)
                shared_summary = emulator._compose_world_summary(  # noqa: SLF001
                    campaign,
                    campaign_state,
                    turns=turns,
                    viewer_actor_id=actor_id,
                    viewer_slug=viewer_slug,
                    viewer_location_key=viewer_location_key,
                    viewer_private_context_key=viewer_private_context_key,
                    scene_npc_slugs=scene_npc_slugs or None,
                    recent_turn_window=0,
                    max_chars=emulator.MAX_SUMMARY_CHARS,  # noqa: SLF001
                )
                shared_recent = emulator._recent_turns_text_for_viewer(  # noqa: SLF001
                    campaign,
                    turns,
                    viewer_actor_id=actor_id,
                    viewer_slug=viewer_slug,
                    viewer_location_key=viewer_location_key,
                    viewer_private_context_key=viewer_private_context_key,
                    requested_player_slugs=set(),
                    requested_npc_slugs=scene_npc_slugs,
                    scene_npc_slugs=scene_npc_slugs or None,
                )
                lcd_recent_limit = max(
                    int(getattr(emulator, "MAX_LCD_RECENT_TURNS", 8) or 8), 4
                )
                _recent_lines = [
                    ln for ln in str(shared_recent or "").splitlines() if ln.strip()
                ]
                if len(_recent_lines) > lcd_recent_limit:
                    shared_recent = "\n".join(_recent_lines[-lcd_recent_limit:])
                shared_recent_with_reasoning = shared_recent
                shared_recent = emulator._strip_reasoning_from_recent_turn_jsonl(shared_recent)  # noqa: SLF001
                shared_recent_texts: set[str] = set()
                for line in str(shared_recent or "").splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except Exception:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    text_value = str(payload.get("text") or "").strip()
                    if text_value:
                        shared_recent_texts.add(text_value)
                if shared_summary and not scene_npc_slugs:
                    deduped_summary_lines = [
                        line
                        for line in str(shared_summary).splitlines()
                        if line.strip() and line.strip() not in shared_recent_texts
                    ]
                    shared_summary = "\n".join(deduped_summary_lines).strip()
                if scene_npc_slugs:
                    _clean_final_user_prompt = re.sub(
                        r"(?m)^WORLD_SUMMARY(?:_FINAL)?:[^\n]*\n?",
                        "",
                        _clean_final_user_prompt,
                    )
                    _shared_context_block = (
                        f"\nSCENE_PARTICIPANTS_LCD: speakers={sorted(_ready_speakers)} listeners={sorted(_ready_listeners)}\n"
                        "The following WORLD_SUMMARY and RECENT_TURNS are filtered to the LOWEST COMMON DENOMINATOR — "
                        "only events that ALL named participants would plausibly know about. "
                        "Use this as the shared pool for common knowledge in the scene.\n"
                        f"\nWORLD_SUMMARY_LCD: {shared_summary or '(empty)'}\n"
                        f"\nRECENT_TURNS_LCD:\n{shared_recent_with_reasoning}\n"
                    )
                elif shared_recent and shared_recent != "None":
                    _shared_context_block = (
                        f"\nWORLD_SUMMARY_FINAL: {shared_summary or '(empty)'}\n"
                        f"\nRECENT_TURNS_FINAL:\n{shared_recent}\n"
                    )
                shared_recent_lines = {
                    line.strip()
                    for line in str(shared_recent or "").splitlines()
                    if line.strip()
                }
                speaker_line_limit = max(
                    int(getattr(emulator, "MAX_SPEAKER_CONTINUITY_TURNS", 12) or 12), 4
                )
                speaker_blocks: list[str] = []
                for speaker_slug in sorted(speaker_npc_slugs):
                    speaker_recent = emulator._recent_turns_text_for_viewer(  # noqa: SLF001
                        campaign,
                        turns,
                        viewer_actor_id=actor_id,
                        viewer_slug=viewer_slug,
                        viewer_location_key=viewer_location_key,
                        viewer_private_context_key=viewer_private_context_key,
                        requested_player_slugs=set(),
                        requested_npc_slugs={speaker_slug},
                        scene_npc_slugs=None,
                        focus_on_requested_receivers=True,
                    )
                    speaker_recent = emulator._strip_reasoning_from_recent_turn_jsonl(speaker_recent)  # noqa: SLF001
                    if speaker_recent and speaker_recent != "None":
                        deduped_speaker_lines = [
                            line
                            for line in str(speaker_recent).splitlines()
                            if line.strip() and line.strip() not in shared_recent_lines
                        ]
                        if not deduped_speaker_lines:
                            continue
                        if len(deduped_speaker_lines) > speaker_line_limit:
                            deduped_speaker_lines = deduped_speaker_lines[-speaker_line_limit:]
                        speaker_blocks.append(
                            f"SPEAKER_CONTINUITY[{speaker_slug}]:\n"
                            + "\n".join(deduped_speaker_lines)
                            + "\n"
                        )
                if speaker_blocks:
                    _speaker_continuity_block = (
                        "\nSPEAKER_CONTINUITY_RULES:\n"
                        "Each SPEAKER_CONTINUITY block is extra continuity for that named speaker only. "
                        "That speaker may use it to decide what to reveal, conceal, or emphasize. "
                        "Do NOT let listeners or silent bystanders act on a speaker's private continuity unless they were independently involved in those same events.\n\n"
                        + "\n".join(speaker_blocks)
                    )
            final_output_contract = (
                "RESEARCH_COMPLETE: Context gathering is complete.\n"
                "Do NOT call any more tools now. Return final narration/state JSON directly.\n"
                "REQUIRED fields: reasoning, scene_output, state_update (with game_time), summary_update.\n"
                "OPTIONAL: set_timer_delay + set_timer_event (include these in the SAME JSON object "
                "to schedule a timed event when the world has external pressure that would escalate without player input). "
                "Also optional: set_timer_interruptible, set_timer_interrupt_action, set_timer_interrupt_scope.\n"
                "scene_output MUST be an object with keys location_key, context_key, and beats.\n"
                "beats MUST be an array of 1 to 2 beat objects; scene_output as a plain string is invalid.\n"
                "narration is optional when scene_output is present; prefer omitting it and let the harness render beat text.\n"
            )
            finalize_prompt = (
                f"{_clean_final_user_prompt}\n"
                f"{_clean_tool_history}\n\n"
                + final_output_contract
                + _pc_reminder
                + _shared_context_block
                + _speaker_continuity_block
                + _final_character_cards_block
                + _final_location_cards_block
                + (f"\n{_final_prompt_tail}\n" if _final_prompt_tail else "")
                + emulator.WRITING_CRAFT_PROMPT
            )
            self._zork_log(f"FINALIZATION REQUEST campaign={campaign_id}", f"SYSTEM:\n{final_system_prompt}\n\nUSER:\n{finalize_prompt}")
            finalized = await self._completion.complete(
                final_system_prompt,
                finalize_prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            self._zork_log(f"FINALIZATION RESPONSE campaign={campaign_id}", finalized or "(empty)")
            finalized_payload = self._parse_model_payload(finalized)
            if finalized_payload is not None and not emulator._is_tool_call(finalized_payload):  # noqa: SLF001
                payload = finalized_payload

        final_output_contract = (
            "RESEARCH_COMPLETE: Context gathering is complete.\n"
            "Do NOT call any more tools now. Return final narration/state JSON directly.\n"
            "REQUIRED fields: reasoning, scene_output, state_update (with game_time), summary_update.\n"
            "OPTIONAL: set_timer_delay + set_timer_event (include these in the SAME JSON object "
            "to schedule a timed event when the world has external pressure that would escalate without player input). "
            "Also optional: set_timer_interruptible, set_timer_interrupt_action, set_timer_interrupt_scope.\n"
            "scene_output MUST be an object with keys location_key, context_key, and beats.\n"
            "beats MUST be an array of 1 to 2 beat objects; scene_output as a plain string is invalid.\n"
            "narration is optional when scene_output is present; prefer omitting it and let the harness render beat text.\n"
        )

        if self._is_emptyish_payload(payload):
            await _notify_progress(progress, "refining")
            for attempt in range(2):
                self._bump_auto_fix_counter(campaign_id, "empty_response_repair_retry")
                repair_prompt = (
                    f"{user_prompt}\n"
                    f"{tool_history}\n\n"
                    + final_output_contract
                    + "OUTPUT_VALIDATION_FAILED: previous response was too empty.\n"
                    "Follow the same JSON contract above.\n"
                    "If the turn is mostly OOC, low-motion, or intentionally minimal, one concise valid beat is enough.\n"
                    "Do not invent extra scene action just to satisfy this retry.\n"
                )
                self._zork_log(
                    f"EMPTY RESPONSE REPAIR campaign={campaign_id} attempt={attempt + 1}",
                    repair_prompt,
                )
                repaired = await self._completion.complete(
                    system_prompt,
                    repair_prompt,
                    temperature=max(0.1, self._temperature - 0.1),
                    max_tokens=self._max_tokens,
                )
                self._zork_log(
                    f"EMPTY RESPONSE REPAIR RESPONSE campaign={campaign_id} attempt={attempt + 1}",
                    repaired or "(empty)",
                )
                repaired_payload = self._parse_model_payload(repaired)
                if repaired_payload is not None and not emulator._is_tool_call(repaired_payload):  # noqa: SLF001
                    payload = repaired_payload
                    if not self._is_emptyish_payload(payload):
                        break

        narration = str(payload.get("narration") or "")
        if self._narration_has_explicit_clock_time(narration) and not self._action_requests_clock_time(action_text):
            await _notify_progress(progress, "refining")
            self._bump_auto_fix_counter(campaign_id, "clock_drift_retry")
            clock_prompt = (
                f"{user_prompt}\n"
                f"{tool_history}\n\n"
                "OUTPUT_VALIDATION_FAILED: Do not invent explicit HH:MM clock timestamps unless asked.\n"
                "Use canonical CURRENT_GAME_TIME or omit exact times.\n"
                "Return ONLY final JSON (no tool_call) with reasoning.\n"
            )
            self._zork_log(f"CLOCK DRIFT RETRY campaign={campaign_id}", clock_prompt)
            clock_retry = await self._completion.complete(
                system_prompt,
                clock_prompt,
                temperature=max(0.1, self._temperature - 0.1),
                max_tokens=self._max_tokens,
            )
            self._zork_log(f"CLOCK DRIFT RETRY RESPONSE campaign={campaign_id}", clock_retry or "(empty)")
            clock_payload = self._parse_model_payload(clock_retry)
            if clock_payload is not None and not emulator._is_tool_call(clock_payload):  # noqa: SLF001
                payload = clock_payload

        # Planning enforcement disabled: the keyword heuristic in
        # _looks_like_major_narrative_beat was far too broad (matching
        # "results", "truth", "escape", any calendar update, any scene
        # change) and fired an extra LLM round-trip on nearly every
        # substantive turn.  The LLM already has plot_plan /
        # consequence_log / chapter_plan tools available during the
        # normal tool loop and uses them when appropriate.

        # Attach research-phase SMS writes so complete_turn can dedup
        # against final inline tool_calls.
        if research_sms_writes:
            payload["_research_sms_writes"] = research_sms_writes
        return payload

    async def complete_turn(self, context, *, progress: ProgressCallback | None = None) -> LLMTurnOutput:
        emulator = self._emulator
        if emulator is None:
            logger.warning("complete_turn: emulator is None → DeterministicLLM fallback (campaign=%s)", context.campaign_id)
            return await self._fallback.complete_turn(context)

        with self._session_factory() as session:
            campaign = session.get(Campaign, context.campaign_id)
            if campaign is None:
                logger.warning("complete_turn: campaign %s not found → DeterministicLLM fallback", context.campaign_id)
                return await self._fallback.complete_turn(context)
            player = (
                session.query(Player)
                .filter(Player.campaign_id == context.campaign_id)
                .filter(Player.actor_id == context.actor_id)
                .first()
            )
            if player is None:
                logger.warning("complete_turn: player not found (campaign=%s actor=%s) → DeterministicLLM fallback", context.campaign_id, context.actor_id)
                return await self._fallback.complete_turn(context)
            turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == context.campaign_id)
                .order_by(Turn.id.desc())
                .limit(emulator.MAX_RECENT_TURNS)
                .all()
            )
            turns.reverse()

        try:
            turn_visibility_default = self._resolve_turn_visibility_default(
                context.session_id,
                campaign_id=context.campaign_id,
            )
            system_prompt, user_prompt = emulator.build_prompt(
                campaign,
                player,
                context.action,
                turns,
                turn_visibility_default=turn_visibility_default,
                prompt_stage=emulator.PROMPT_STAGE_RESEARCH,
            )
            final_system_prompt, final_user_prompt = emulator.build_prompt(
                campaign,
                player,
                context.action,
                turns,
                turn_visibility_default=turn_visibility_default,
                prompt_stage=emulator.PROMPT_STAGE_FINAL,
            )
            logger.debug("complete_turn: resolving payload for campaign=%s actor=%s action=%s", context.campaign_id, context.actor_id, (context.action or "")[:80])
            payload = await self._resolve_payload(
                context.campaign_id,
                context.actor_id,
                context.action,
                system_prompt,
                user_prompt,
                final_system_prompt,
                final_user_prompt,
                progress=progress,
            )
            if payload is None:
                logger.warning("complete_turn: _resolve_payload returned None → DeterministicLLM fallback (campaign=%s)", context.campaign_id)
                return await self._fallback.complete_turn(context)
            logger.debug(
                "complete_turn: _resolve_payload succeeded, keys=%s narration_len=%d, calling _payload_to_output (campaign=%s)",
                list(payload.keys())[:10],
                len(str(payload.get("narration") or "")),
                context.campaign_id,
            )
            self._zork_log(
                f"RESOLVE_PAYLOAD RESULT campaign={context.campaign_id}",
                json.dumps(payload, default=str, ensure_ascii=False),
            )
            output = self._payload_to_output(
                payload,
                actor_id=context.actor_id,
                character_name=context.player_state.get("character_name"),
            )
            logger.debug(
                "complete_turn: _payload_to_output succeeded, narration_len=%d (campaign=%s)",
                len(output.narration or ""),
                context.campaign_id,
            )
            # Execute any inline tool_calls that the LLM included alongside
            # its final narration. These have already been validated against
            # the allowlist in _payload_to_output.
            # Dedup: skip sms_write calls that were already executed during
            # research (models often re-include the player's outgoing SMS).
            _research_sms = payload.pop("_research_sms_writes", [])
            _research_sms_set: set[tuple[str, str]] = set()
            if _research_sms:
                _research_sms_set = {(r, c) for r, c in _research_sms}
            _executed_sms = False
            for tc in output.tool_calls:
                try:
                    tc_name = str(tc.get("tool_call") or "").strip().lower()
                    if tc_name in ("sms_write", "sms_schedule") and _research_sms_set:
                        _tc_recip = " ".join(str(tc.get("recipient") or tc.get("thread") or tc.get("to") or "").strip().lower().split())
                        _tc_content = " ".join(str(tc.get("content") or tc.get("text") or tc.get("message") or "").strip().lower().split())
                        if (_tc_recip, _tc_content) in _research_sms_set:
                            logger.debug("complete_turn: skipping duplicate sms_write to=%s (already sent in research)", _tc_recip)
                            continue
                    await self._execute_tool_call(context.campaign_id, tc, actor_id=context.actor_id)
                    if tc_name in ("sms_write",):
                        _executed_sms = True
                except Exception:
                    pass  # best-effort; don't break the turn
            # Notify player that new SMS messages were stored so the UI
            # refreshes the inbox (narration already shows the text, but the
            # SMS tab won't update without a push event).
            if _executed_sms and context.actor_id and emulator and emulator.notification_port is not None:
                try:
                    await emulator.notification_port.send_dm(
                        actor_id=context.actor_id,
                        message="\U0001f4f1 New SMS received.",
                    )
                except Exception:
                    pass
            return output
        except Exception:
            logger.exception("complete_turn: unhandled exception → DeterministicLLM fallback (campaign=%s action=%s)", context.campaign_id, (context.action or "")[:100])
            return await self._fallback.complete_turn(context)


ZorkToolAwareLLM = ToolAwareZorkLLM
