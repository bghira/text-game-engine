from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import Any, Callable

from .core.types import GiveItemInstruction, LLMTurnOutput, TimerInstruction
from .persistence.sqlalchemy.models import Campaign, Player, Turn
from .zork_emulator import ZorkEmulator


class DeterministicLLM:
    """Simple fallback LLM for bring-up and failure paths."""

    @staticmethod
    def _advance_time(game_time: dict[str, Any]) -> dict[str, Any]:
        day = int(game_time.get("day", 1) or 1)
        hour = int(game_time.get("hour", 8) or 8)
        minute = int(game_time.get("minute", 0) or 0)

        minute += 10
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

    async def complete_turn(self, context) -> LLMTurnOutput:
        action = (context.action or "").strip()
        lowered = action.lower()
        current_time = (
            context.campaign_state.get("game_time", {})
            if isinstance(context.campaign_state, dict)
            else {}
        )
        next_time = self._advance_time(
            current_time if isinstance(current_time, dict) else {}
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

    def bind_emulator(self, emulator: ZorkEmulator) -> None:
        self._emulator = emulator

    @staticmethod
    def _parse_json(text: str | None, default: Any) -> Any:
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            return default

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
        summary_update = payload.get("summary_update")
        character_updates = payload.get("character_updates")
        calendar_update = payload.get("calendar_update")
        scene_image_prompt = payload.get("scene_image_prompt")
        xp_awarded = payload.get("xp_awarded", 0)
        has_signal = (
            bool(state_update)
            or bool(player_state_update)
            or bool(character_updates)
            or bool(calendar_update)
        )
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
            return None
        if not response:
            return None
        cleaned = emulator._clean_response(response)  # noqa: SLF001
        json_text = emulator._extract_json(cleaned)  # noqa: SLF001
        try:
            payload = emulator._parse_json_lenient(json_text or cleaned)  # noqa: SLF001
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _payload_to_output(self, payload: dict[str, Any]) -> LLMTurnOutput:
        emulator = self._emulator
        if emulator is None:
            return LLMTurnOutput(narration="")

        narration = payload.get("narration")
        if not isinstance(narration, str):
            narration = ""

        state_update = payload.get("state_update")
        if not isinstance(state_update, dict):
            state_update = {}

        player_state_update = payload.get("player_state_update")
        if not isinstance(player_state_update, dict):
            player_state_update = {}

        turn_visibility = payload.get("turn_visibility")
        if not isinstance(turn_visibility, dict):
            turn_visibility = None

        state_update, player_state_update = emulator._split_room_state(  # noqa: SLF001
            state_update, player_state_update
        )

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

        return LLMTurnOutput(
            narration=narration,
            reasoning=reasoning,
            state_update=state_update,
            summary_update=summary_update,
            xp_awarded=xp_awarded,
            player_state_update=player_state_update,
            turn_visibility=turn_visibility,
            scene_image_prompt=scene_image_prompt,
            timer_instruction=timer_instruction,
            character_updates=character_updates,
            give_item=give_item,
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
            for query in queries[:4]:
                curated_hits.extend(
                    self._emulator.search_curated_memories(
                        query=query,
                        campaign_id=campaign_id,
                        category=category,
                        top_k=5,
                    )
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
        with self._session_factory() as session:
            turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .filter(Turn.kind == "narrator")
                .order_by(Turn.id.desc())
                .limit(500)
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
                    narrator_hits[int(turn.id)] = {
                        "turn_id": int(turn.id),
                        "score": score,
                        "content": content,
                        "visibility_scope": str(
                            (visibility or {}).get("scope") or "public"
                        ),
                        "actor_player_slug": str(meta.get("actor_player_slug") or ""),
                        "location_key": str(meta.get("location_key") or ""),
                    }

        ordered_narrator = sorted(
            narrator_hits.values(),
            key=lambda row: (float(row.get("score", 0.0)), int(row.get("turn_id", 0))),
            reverse=True,
        )[:5]
        source_hits_flat: list[tuple[str, str, int, str, float]] = []
        if self._emulator is not None and has_source_material and (
            source_scope or not category_scope
        ):
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

        lines = ["MEMORY_RECALL (results from memory_search):"]
        for query in queries[:4]:
            lines.append(f"Results for '{query}':")
        lines.append("Narrator turn matches:")
        if ordered_narrator:
            for hit in ordered_narrator:
                snippet = str(hit.get("content") or "").strip().replace("\n", " ")
                if len(snippet) > 280:
                    snippet = snippet[:279].rstrip() + "..."
                lines.append(
                    "- [narrator turn "
                    f"{hit['turn_id']}, relevance {float(hit['score']):.2f}"
                    f"{', actor ' + str(hit.get('actor_player_slug')) if hit.get('actor_player_slug') else ''}"
                    f"{', visibility ' + str(hit.get('visibility_scope')) if str(hit.get('visibility_scope') or 'public') != 'public' else ''}"
                    f"{', location ' + str(hit.get('location_key')) if hit.get('location_key') else ''}"
                    f"]: {snippet}"
                )
        else:
            lines.append("- (no narrator turn matches)")

        if curated_hits:
            lines.append("Curated memory matches:")
            for term, memory, score in curated_hits[:5]:
                snippet = str(memory or "").strip().replace("\n", " ")
                if len(snippet) > 220:
                    snippet = snippet[:219].rstrip() + "..."
                lines.append(
                    f"- [term {term}, relevance {float(score):.2f}]: {snippet}"
                )
        if source_hits_unique:
            lines.append("Source material matches:")
            for (
                source_doc_key,
                source_doc_label,
                source_chunk_index,
                source_chunk_text,
                source_score,
            ) in source_hits_unique:
                if float(source_score) < 0.40:
                    continue
                source_format = source_doc_formats.get(source_doc_key, "generic")
                source_text_lines = [
                    line.strip()
                    for line in str(source_chunk_text or "").splitlines()
                    if line.strip()
                ]
                source_text = (
                    "\n    ".join(source_text_lines)
                    if source_text_lines
                    else str(source_chunk_text or "").strip()
                )
                if len(source_text) > 4000:
                    source_text = (
                        source_text[:4000].rsplit(" ", 1)[0].strip() + "..."
                    )
                lines.append(
                    "- [source "
                    f"{source_doc_label} ({source_doc_key}) snippet {int(source_chunk_index)}, "
                    f"format {source_format}, relevance {float(source_score):.2f}]:\n    {source_text}"
                )
        elif has_source_material and source_scope:
            scope_label = f"source:{source_scope_key}" if source_scope_key else "source"
            lines.append(f"Source material matches: (none in scope '{scope_label}')")
        if has_source_material:
            total_snippets = 0
            for row in source_docs:
                try:
                    total_snippets += int(row.get("chunk_count") or 0)
                except Exception:
                    continue
            lines.append(
                f"SOURCE_MATERIAL_INDEX: {len(source_docs)} document(s), {total_snippets} total snippet(s)."
            )
            for row in source_docs[:5]:
                row_format = str(row.get("format") or "").strip().lower()
                if not row_format:
                    row_format = source_doc_formats.get(
                        str(row.get("document_key") or ""), "generic"
                    )
                lines.append(
                    "- "
                    f"key='{row.get('document_key')}' "
                    f"label='{row.get('document_label')}' "
                    f"format='{row_format}' "
                    f"snippets={row.get('chunk_count')}"
                )

        lines.extend(
            [
                "MEMORY_RECALL_NEXT_ACTIONS:",
                "- To retrieve FULL text for a specific hit turn number:",
                '  {"tool_call": "memory_turn", "turn_id": 1234}',
                "- To discover curated memory categories/terms before narrowing search:",
                '  {"tool_call": "memory_terms", "wildcard": "char:*"}',
                "- To search inside one curated category after term discovery:",
                '  {"tool_call": "memory_search", "category": "char:character-slug", "queries": ["keyword1", "keyword2"]}',
                "- To search narrator memories for interactions involving a player slug:",
                '  {"tool_call": "memory_search", "category": "interaction:player-slug", "queries": ["argument", "deal", "kiss"]}',
                "- To search for turns noticed by a specific NPC slug:",
                '  {"tool_call": "memory_search", "category": "awareness:npc-slug", "queries": ["overheard", "promise", "secret"]}',
                "- To restrict narrator-memory recall by visibility scope:",
                '  {"tool_call": "memory_search", "category": "visibility:private", "queries": ["secret meeting"]}',
                '  {"tool_call": "memory_search", "category": "visibility:local", "queries": ["bar argument"]}',
                "- To inspect off-scene SMS communications:",
                '  {"tool_call": "sms_list", "wildcard": "*"}',
                '  {"tool_call": "sms_read", "thread": "contact-slug", "limit": 20}',
                "- To schedule a delayed incoming SMS (hidden until delivery):",
                '  {"tool_call": "sms_schedule", "thread": "contact-slug", "from": "NPC", "to": "Player", "message": "...", "delay_seconds": 120}',
            ]
        )
        if has_source_material:
            source_formats = sorted(set(source_doc_formats.values()) or {"generic"})
            source_formats_set = set(source_formats)
            has_rulebook = "rulebook" in source_formats_set
            has_only_generic = source_formats_set == {"generic"}
            format_descriptions = {
                "story": "scripted scenes / prose",
                "rulebook": "line facts (`KEY: value`)",
                "generic": "notes/dumps (usually not indexed)",
            }
            lines.extend(
                [
                    "SOURCE_MATERIAL_FORMAT_GUIDE:",
                    f"- Active formats: {', '.join(source_formats)}",
                ]
            )
            for fmt in source_formats:
                lines.append(f"- {fmt}: {format_descriptions.get(fmt, fmt)}")
            lines.extend(
                [
                    "To inspect source text:",
                    '  {"tool_call": "memory_search", "category": "source", "queries": ["keyword"]}',
                    '  {"tool_call": "memory_search", "category": "source:<document-key>", "queries": ["keyword"]}',
                ]
            )
            if has_only_generic:
                lines.append(
                    "- Generic docs are usually summarized in setup prompts; use source search only for "
                    "exact wording when needed."
                )
            if has_rulebook:
                lines.extend(
                    [
                        "- Rulebook docs expose keyed snippets. First pass (no filter) to discover keys:",
                        '  {"tool_call": "source_browse"}',
                        '  {"tool_call": "source_browse", "document_key": "document-key"}',
                        "Then narrow with wildcard keys:",
                        '  {"tool_call": "source_browse", "wildcard": "keyword*"}',
                    ]
                )
        if roster_hints:
            lines.append("MEMORY_RECALL_ROSTER_RECOMMENDATIONS:")
            for hint in roster_hints[:6]:
                term = str(hint.get("term") or hint.get("slug") or "").strip() or "unknown-term"
                slug = str(hint.get("slug") or "").strip() or "character-slug"
                try:
                    count = int(hint.get("count") or 0)
                except Exception:
                    count = 0
                lines.append(
                    "- You have looked for "
                    f"'{term}' {count} times and it is not present in WORLD_CHARACTERS. "
                    "If this is stable/non-stale information and you can confirm it, "
                    f"store it with character_updates using slug '{slug}'."
                )
        return "\n".join(lines)

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

        lines = ["MEMORY_TERMS:"]
        if terms:
            for row in terms[:40]:
                if isinstance(row, dict):
                    category = str(row.get("category") or "").strip()
                    term = str(row.get("term") or "").strip()
                    label = f"{category} :: {term}" if category else term
                    lines.append(f"- {label}")
                else:
                    lines.append(f"- {row}")
        else:
            lines.append("- (none)")

        lines.extend(
            [
                "NEXT_ACTIONS:",
                '- {"tool_call": "memory_search", "category": "char:character-slug", "queries": ["keyword"]}',
                '- {"tool_call": "memory_store", "category": "char:character-slug", "term": "keyword", "memory": "fact"}',
            ]
        )
        return "\n".join(lines)

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
            return "MEMORY_TURN: invalid turn_id"

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
            return f"MEMORY_TURN: no turn found for turn_id={turn_id}"
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
                return "MEMORY_TURN: that turn exists, but it is not visible to this player."

        content = str(turn.content or "")
        if len(content) > 12000:
            content = content[:11999].rstrip() + "..."
        return (
            "MEMORY_TURN_FULLTEXT:\n"
            f"turn_id={int(turn.id)}\n"
            f"kind={turn.kind}\n"
            f"actor_id={turn.actor_id}\n"
            f"created_at={turn.created_at.isoformat() if turn.created_at else None}\n"
            "content:\n"
            f"{content}\n"
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

    def _tool_sms_list(self, campaign_id: str, payload: dict[str, Any]) -> str:
        wildcard_raw = payload.get("wildcard")
        wildcard = (
            wildcard_raw.strip()
            if isinstance(wildcard_raw, str) and wildcard_raw.strip()
            else "*"
        )

        rows = (
            self._emulator.list_sms_threads(campaign_id, wildcard=wildcard, limit=20)
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

    def _tool_sms_write(self, campaign_id: str, payload: dict[str, Any]) -> str:
        thread_raw = payload.get("thread")
        sender_raw = payload.get("from", payload.get("sender"))
        recipient_raw = payload.get("to", payload.get("recipient"))
        message_raw = payload.get("message")

        thread = thread_raw.strip() if isinstance(thread_raw, str) and thread_raw.strip() else ""
        sender = sender_raw.strip() if isinstance(sender_raw, str) and sender_raw.strip() else ""
        recipient = (
            recipient_raw.strip()
            if isinstance(recipient_raw, str) and recipient_raw.strip()
            else ""
        )
        message = message_raw.strip() if isinstance(message_raw, str) and message_raw.strip() else ""

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
        )
        return f"SMS_WRITE_RESULT: stored={bool(ok)} reason={reason} thread={thread}"

    def _tool_sms_schedule(self, campaign_id: str, payload: dict[str, Any]) -> str:
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
        speed = self._emulator.get_speed_multiplier(campaign) if campaign is not None else 1.0
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

    def _execute_tool_call(
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
        if name == "name_generate":
            return self._tool_name_generate(campaign_id, payload)
        if name == "memory_turn":
            return self._tool_memory_turn(campaign_id, payload, actor_id=actor_id)
        if name == "memory_store":
            return self._tool_memory_store(campaign_id, payload)
        if name == "sms_list":
            return self._tool_sms_list(campaign_id, payload)
        if name == "sms_read":
            return self._tool_sms_read(campaign_id, payload, actor_id=actor_id)
        if name == "sms_write":
            return self._tool_sms_write(campaign_id, payload)
        if name == "sms_schedule":
            return self._tool_sms_schedule(campaign_id, payload)
        if name == "story_outline":
            return self._tool_story_outline(campaign_id, payload)
        if name == "plot_plan":
            return self._tool_plot_plan(campaign_id, payload)
        if name == "chapter_plan":
            return self._tool_chapter_plan(campaign_id, payload)
        if name == "consequence_log":
            return self._tool_consequence_log(campaign_id, payload)
        if name == "ready_to_write":
            return "READY_TO_WRITE_ACK"
        if name == "recent_turns":
            return "RECENT_TURNS_TOOL_NOT_IMPLEMENTED"
        return f"TOOL_ERROR: unsupported tool_call '{name}'"

    async def _resolve_payload(
        self,
        campaign_id: str,
        actor_id: str,
        action_text: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any] | None:
        first = await self._completion.complete(
            system_prompt,
            user_prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        payload = self._parse_model_payload(first)
        if payload is None:
            return None

        tool_history = ""
        used_tool_names: set[str] = set()
        seen_tool_signatures: set[str] = set()
        emulator = self._emulator
        if emulator is None:
            return None
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
                tool_result = self._execute_tool_call(
                    campaign_id, tool_payload, actor_id=actor_id
                )
                tool_history += f"\n\n{tool_result}"
                augmented_prompt = (
                    f"{user_prompt}\n"
                    f"{tool_history}\n\n"
                    "Use the memory results above. Return ONLY the final turn JSON object."
                )
                nxt = await self._completion.complete(
                    system_prompt,
                    augmented_prompt,
                    temperature=max(0.1, self._temperature - 0.2),
                    max_tokens=self._max_tokens,
                )
                payload = self._parse_model_payload(nxt)
                self._bump_auto_fix_counter(campaign_id, "forced_memory_search")
                if payload is None:
                    return None

        for _ in range(max(0, self._max_tool_rounds)):
            if not emulator._is_tool_call(payload):  # noqa: SLF001
                break
            tool_name = str(payload.get("tool_call") or "").strip().lower()
            if tool_name:
                used_tool_names.add(tool_name)
            if tool_name == "ready_to_write":
                break
            if not memory_lookup_enabled and tool_name.startswith("memory_"):
                tool_history += (
                    "\n\nMEMORY_TOOLS_DISABLED: Long-term memory lookup is disabled for this turn "
                    "(early campaign context still fits prompt budget). "
                    "Do NOT call memory_* tools; continue with direct context or non-memory tools."
                )
                augmented_prompt = (
                    f"{user_prompt}\n"
                    f"{tool_history}\n\n"
                    "Return ONLY the final turn JSON object."
                )
                nxt = await self._completion.complete(
                    system_prompt,
                    augmented_prompt,
                    temperature=max(0.1, self._temperature - 0.2),
                    max_tokens=self._max_tokens,
                )
                payload = self._parse_model_payload(nxt)
                if payload is None:
                    return None
                continue
            tool_signature = self._tool_call_signature(payload)
            if tool_signature and tool_signature in seen_tool_signatures:
                tool_history += (
                    "\n\nTOOL_DEDUP_RESULT: duplicate tool_call payload already executed this turn; skipped. "
                    "Do NOT repeat identical tool calls. Use a distinct tool/payload or return final JSON (no tool_call)."
                )
                augmented_prompt = (
                    f"{user_prompt}\n"
                    f"{tool_history}\n\n"
                    "Return ONLY the final turn JSON object."
                )
                nxt = await self._completion.complete(
                    system_prompt,
                    augmented_prompt,
                    temperature=max(0.1, self._temperature - 0.2),
                    max_tokens=self._max_tokens,
                )
                payload = self._parse_model_payload(nxt)
                if payload is None:
                    return None
                continue
            if tool_signature:
                seen_tool_signatures.add(tool_signature)
            tool_result = self._execute_tool_call(
                campaign_id,
                payload,
                actor_id=actor_id,
            )
            tool_history += f"\n\n{tool_result}"
            augmented_prompt = (
                f"{user_prompt}\n"
                f"{tool_history}\n\n"
                "Use the tool results above. Return ONLY the final turn JSON object."
            )
            nxt = await self._completion.complete(
                system_prompt,
                augmented_prompt,
                temperature=max(0.1, self._temperature - 0.2),
                max_tokens=self._max_tokens,
            )
            payload = self._parse_model_payload(nxt)
            if payload is None:
                return None

        if emulator._is_tool_call(payload) and str(payload.get("tool_call") or "").strip().lower() != "ready_to_write":  # noqa: E501, SLF001
            return None
        if self._is_emptyish_payload(payload):
            self._bump_auto_fix_counter(campaign_id, "empty_response_repair_retry")
            repair_prompt = (
                f"{user_prompt}\n"
                f"{tool_history}\n\n"
                "OUTPUT_VALIDATION_FAILED: previous response was too empty.\n"
                "Return ONLY final JSON (no tool_call) with:\n"
                "- reasoning string grounded in evidence/context used\n"
                "- narration containing one concrete scene development\n"
                "- state_update object with game_time advanced\n"
                "- summary_update with durable consequence when applicable.\n"
            )
            repaired = await self._completion.complete(
                system_prompt,
                repair_prompt,
                temperature=max(0.1, self._temperature - 0.1),
                max_tokens=self._max_tokens,
            )
            repaired_payload = self._parse_model_payload(repaired)
            if repaired_payload is not None and not emulator._is_tool_call(repaired_payload):  # noqa: SLF001
                payload = repaired_payload

        narration = str(payload.get("narration") or "")
        if self._narration_has_explicit_clock_time(narration) and not self._action_requests_clock_time(action_text):
            self._bump_auto_fix_counter(campaign_id, "clock_drift_retry")
            clock_prompt = (
                f"{user_prompt}\n"
                f"{tool_history}\n\n"
                "OUTPUT_VALIDATION_FAILED: Do not invent explicit HH:MM clock timestamps unless asked.\n"
                "Use canonical CURRENT_GAME_TIME or omit exact times.\n"
                "Return ONLY final JSON (no tool_call) with reasoning.\n"
            )
            clock_retry = await self._completion.complete(
                system_prompt,
                clock_prompt,
                temperature=max(0.1, self._temperature - 0.1),
                max_tokens=self._max_tokens,
            )
            clock_payload = self._parse_model_payload(clock_retry)
            if clock_payload is not None and not emulator._is_tool_call(clock_payload):  # noqa: SLF001
                payload = clock_payload

        planning_used = bool(
            {"plot_plan", "chapter_plan", "consequence_log"} & used_tool_names
        )
        if not planning_used and self._looks_like_major_narrative_beat(payload):
            planning_prompt = (
                f"{user_prompt}\n"
                f"{tool_history}\n\n"
                "PLANNING_ENFORCEMENT: A major beat occurred.\n"
                "Return ONLY one planning tool call JSON now: plot_plan OR consequence_log "
                "(chapter_plan optional off-rails).\n"
                "No narration.\n"
            )
            planning_resp = await self._completion.complete(
                system_prompt,
                planning_prompt,
                temperature=max(0.1, self._temperature - 0.2),
                max_tokens=700,
            )
            planning_payload = self._parse_model_payload(planning_resp)
            if planning_payload is not None and emulator._is_tool_call(planning_payload):  # noqa: SLF001
                planning_name = str(planning_payload.get("tool_call") or "").strip().lower()
                if planning_name in {"plot_plan", "chapter_plan", "consequence_log"}:
                    _ = self._execute_tool_call(
                        campaign_id,
                        planning_payload,
                        actor_id=actor_id,
                    )
                    self._bump_auto_fix_counter(campaign_id, "forced_planning_tool")
        return payload

    async def complete_turn(self, context) -> LLMTurnOutput:
        emulator = self._emulator
        if emulator is None:
            return await self._fallback.complete_turn(context)

        with self._session_factory() as session:
            campaign = session.get(Campaign, context.campaign_id)
            if campaign is None:
                return await self._fallback.complete_turn(context)
            player = (
                session.query(Player)
                .filter(Player.campaign_id == context.campaign_id)
                .filter(Player.actor_id == context.actor_id)
                .first()
            )
            if player is None:
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
            )
            payload = await self._resolve_payload(
                context.campaign_id,
                context.actor_id,
                context.action,
                system_prompt,
                user_prompt,
            )
            if payload is None:
                return await self._fallback.complete_turn(context)
            return self._payload_to_output(payload)
        except Exception:
            return await self._fallback.complete_turn(context)


ZorkToolAwareLLM = ToolAwareZorkLLM

