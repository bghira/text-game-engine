from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from sqlalchemy import or_

from .core.attachments import (
    AttachmentProcessingConfig,
    AttachmentTextProcessor,
    extract_attachment_text,
)
from .core.emulator_ports import (
    IMDBLookupPort,
    MediaGenerationPort,
    MemorySearchPort,
    TextCompletionPort,
    TimerEffectsPort,
)
from .core.engine import GameEngine
from .core.normalize import normalize_campaign_name, parse_json_dict
from .core.types import ResolveTurnInput
from .persistence.sqlalchemy.models import (
    Actor,
    Campaign,
    Embedding,
    Player,
    Session as GameSession,
    Snapshot,
    Timer,
    Turn,
)

_ZORK_LOG_PATH = os.path.join(os.getcwd(), "zork.log")


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
    MAX_SUMMARY_CHARS = 10000
    MAX_STATE_CHARS = 10000
    MAX_RECENT_TURNS = 24
    MAX_TURN_CHARS = 1200
    MAX_NARRATION_CHARS = 23500
    MAX_PARTY_CONTEXT_PLAYERS = 6
    MAX_SCENE_PROMPT_CHARS = 900
    MAX_PERSONA_PROMPT_CHARS = 140
    MAX_SCENE_REFERENCE_IMAGES = 10
    MAX_INVENTORY_CHANGES_PER_TURN = 10
    MAX_CHARACTERS_CHARS = 8000
    MAX_CHARACTERS_IN_PROMPT = 20
    XP_BASE = 100
    XP_PER_LEVEL = 50
    ATTENTION_WINDOW_SECONDS = 600
    IMMUTABLE_CHARACTER_FIELDS = {"name", "personality", "background", "appearance"}
    ATTACHMENT_MAX_BYTES = 500_000
    ATTACHMENT_CHUNK_TOKENS = 2_000
    ATTACHMENT_MODEL_CTX_TOKENS = 200_000
    ATTACHMENT_PROMPT_OVERHEAD_TOKENS = 6_000
    ATTACHMENT_RESPONSE_RESERVE_TOKENS = 4_000
    ATTACHMENT_MAX_PARALLEL = 4
    ATTACHMENT_MAX_CHUNKS = 8
    ATTACHMENT_GUARD_TOKEN = "--COMPLETED SUMMARY--"
    DEFAULT_SCENE_IMAGE_MODEL = "black-forest-labs/FLUX.2-klein-4b"
    DEFAULT_AVATAR_IMAGE_MODEL = "black-forest-labs/FLUX.2-klein-4b"
    PROCESSING_EMOJI = "🤔"

    MAIN_PARTY_TOKEN = "main party"
    NEW_PATH_TOKEN = "new path"

    ROOM_IMAGE_STATE_KEY = "room_scene_images"
    PLAYER_STATS_KEY = "zork_stats"
    PLAYER_STATS_MESSAGES_KEY = "messages_sent"
    PLAYER_STATS_TIMERS_AVERTED_KEY = "timers_averted"
    PLAYER_STATS_TIMERS_MISSED_KEY = "timers_missed"
    PLAYER_STATS_ATTENTION_SECONDS_KEY = "attention_seconds"
    PLAYER_STATS_LAST_MESSAGE_AT_KEY = "last_message_at"
    DEFAULT_CAMPAIGN_PERSONA = (
        "A cooperative, curious adventurer: observant, resourceful, and willing to "
        "engage with absurd situations in-character."
    )
    PRESET_DEFAULT_PERSONAS = {
        "alice": (
            "A curious and polite wanderer with dry wit, dream-logic intuition, and "
            "quiet courage in whimsical danger."
        ),
    }
    PRESET_ALIASES = {
        "alice": "alice",
        "alice in wonderland": "alice",
        "alice-wonderland": "alice",
    }
    PRESET_CAMPAIGNS = {
        "alice": {
            "summary": (
                "Alice dozes on a riverbank; a White Rabbit with a waistcoat hurries past. "
                "She follows into a rabbit hole, landing in a long hall of doors. "
                "A tiny key and a bottle labeled DRINK ME lead to size changes. "
                "A pool of tears forms; a caucus race follows; the Duchess's house, "
                "the Mad Tea Party, the Queen's croquet ground, and the court of cards await."
            ),
            "state": {
                "setting": "Alice in Wonderland",
                "tone": "whimsical, dreamlike, slightly menacing",
                "landmarks": [
                    "riverbank",
                    "rabbit hole",
                    "hall of doors",
                    "garden",
                    "pool of tears",
                    "caucus shore",
                    "duchess house",
                    "mad tea party",
                    "croquet ground",
                    "court of cards",
                ],
                "main_party_location": "hall of doors",
                "start_room": {
                    "room_title": "A Riverbank, Afternoon",
                    "room_summary": "A sunny riverbank where Alice grows drowsy as a White Rabbit hurries past.",
                    "room_description": (
                        "You are on a grassy riverbank beside a slow, glittering stream. "
                        "The day is warm and lazy, the air humming with insects. "
                        "A book without pictures lies nearby. "
                        "In the corner of your eye, a White Rabbit in a waistcoat scurries past, "
                        "muttering about being late."
                    ),
                    "exits": ["follow the white rabbit", "stroll along the riverbank"],
                    "location": "riverbank",
                },
            },
            "last_narration": (
                "A Riverbank, Afternoon\n"
                "You are on a grassy riverbank beside a slow, glittering stream. "
                "The day is warm and lazy, the air humming with insects. "
                "A book without pictures lies nearby. "
                "In the corner of your eye, a White Rabbit in a waistcoat scurries past, "
                "muttering about being late.\n"
                "Exits: follow the white rabbit, stroll along the riverbank"
            ),
        }
    }
    _COMPLETED_VALUES = {
        "complete",
        "completed",
        "done",
        "resolved",
        "finished",
        "concluded",
        "vacated",
        "dispersed",
        "avoided",
        "departed",
    }
    ROOM_STATE_KEYS = {
        "room_title",
        "room_description",
        "room_summary",
        "exits",
        "location",
        "room_id",
    }
    MODEL_STATE_EXCLUDE_KEYS = ROOM_STATE_KEYS | {
        "last_narration",
        "room_scene_images",
        "scene_image_model",
        "default_persona",
        "start_room",
        "story_outline",
        "current_chapter",
        "current_scene",
        "setup_phase",
        "setup_data",
    }
    PLAYER_STATE_EXCLUDE_KEYS = {"inventory", "room_description", PLAYER_STATS_KEY}
    _STALE_VALUE_PATTERNS = _COMPLETED_VALUES | {
        "secured",
        "confirmed",
        "received",
        "granted",
        "initiated",
        "accepted",
        "placed",
        "offered",
    }
    _ITEM_STOPWORDS = {"a", "an", "the", "of", "and", "or", "to", "in", "on", "for"}
    _INVENTORY_LINE_PREFIXES = (
        "inventory:",
        "inventory -",
        "items:",
        "items carried:",
        "you are carrying:",
        "you carry:",
        "your inventory:",
        "current inventory:",
    )
    SYSTEM_PROMPT = (
        "You are the ZorkEmulator, a classic text-adventure GM with light RPG rules. "
        "You describe outcomes in second person, terse but vivid. You track rooms, "
        "objects, exits, and consequences. Each player is a distinct character and "
        "may be in a different location or timeline than other players. You never break character. "
        "This is an adult-oriented game. You may include mature themes, explicit content, violence, "
        "dark humor, and adult situations when appropriate to the story and player actions.\n\n"
        "Return ONLY valid JSON with these keys:\n"
        "- narration: string (what the player sees)\n"
        "- state_update: object (world state patches; set a key to null to remove it when no longer relevant. "
        "IMPORTANT: WORLD_STATE has a size budget. Actively prune stale keys every turn by setting them to null. "
        "Remove: completed/concluded events, expired countdowns/ETAs, booleans for past events that no longer affect gameplay, "
        "and any scene-specific state from scenes the player has left. Only keep state that is CURRENTLY ACTIVE and relevant.\n"
        "STRUCTURE REQUIREMENT: State keys MUST be organized as nested objects keyed by the concept, entity, or character being tracked. "
        "NEVER use flat underscore-joined keys like 'guard_captain_mood' or 'throne_room_door_locked'. "
        'Instead, nest them: {"guard_captain": {"mood": "suspicious"}, "throne_room": {"door_locked": true}}. '
        "Group related attributes under a single entity key. "
        "To remove an entire entity, set its key to null. To remove one attribute, set the nested key to null. "
        "Examples of CORRECT structure:\n"
        '  {"marcus": {"mood": "angry", "location": "courtyard"}, "west_gate": {"status": "barred"}}\n'
        "Examples of WRONG structure (never do this):\n"
        '  {"marcus_mood": "angry", "marcus_location": "courtyard", "west_gate_status": "barred"})\n'
        "- summary_update: string (one or two sentences of lasting changes)\n"
        "- xp_awarded: integer (0-10)\n"
        "- player_state_update: object (optional, player state patches)\n"
        "- scene_image_prompt: string (optional; include only when scene/location changes and a fresh image should be rendered)\n"
        "- set_timer_delay: integer (optional; 30-300 seconds, see TIMED EVENTS SYSTEM below)\n"
        "- set_timer_event: string (optional; what happens when the timer expires)\n"
        "- set_timer_interruptible: boolean (optional; default true)\n"
        "- set_timer_interrupt_action: string or null (optional; context for interruption handling)\n"
        "- give_item: object (REQUIRED when the acting player gives/hands/passes an item to another player character. "
        "Keys: 'item' (string, exact item name from acting player's inventory), "
        "'to_discord_mention' (string, discord_mention of the recipient from PARTY_SNAPSHOT, e.g. '<@123456>'). "
        "The emulator handles removing from the giver and adding to the recipient automatically. "
        "Do NOT use inventory_remove for the given item — give_item handles both sides. "
        "Only use when both players are in the same room per PARTY_SNAPSHOT. Only one item per turn.)\n"
        "- calendar_update: object (optional; see CALENDAR & GAME TIME SYSTEM below)\n"
        "- character_updates: object (optional; keyed by stable slug IDs like 'marcus-blackwell'. "
        "Use this to create or update NPCs in the world character tracker. "
        "Slug IDs must be lowercase-hyphenated, derived from the character name, and stable across turns. "
        "On first appearance provide all fields: name, personality, background, appearance, location, "
        "current_status, allegiance, relationship. On subsequent turns only mutable fields are accepted: "
        "location, current_status, allegiance, relationship, deceased_reason, and any other dynamic key. "
        "Immutable fields (name, personality, background, appearance) are locked at creation and silently ignored on updates. "
        "Set deceased_reason to a string when a character dies. "
        "WORLD_CHARACTERS in the prompt shows the current NPC roster — use it for continuity.)\n\n"
        "Rules:\n"
        "- Return ONLY the JSON object. No markdown, no code fences, no text before or after the JSON.\n"
        "- Do NOT repeat the narration outside the JSON object.\n"
        "- Keep narration under 1800 characters.\n"
        "- If WORLD_SUMMARY is empty, invent a strong starting room and seed the world.\n"
        "- Use player_state_update for player-specific location and status.\n"
        "- Use player_state_update.room_title for a short location title (e.g. 'Penthouse Suite, Escala') whenever location changes.\n"
        "- Use player_state_update.room_description for a full room description only when location changes.\n"
        "- Use player_state_update.room_summary for a short one-line room summary for future context.\n"
        "- Use player_state_update.exits as a short list of exits if applicable.\n"
        "- Use player_state_update for inventory, hp, or conditions.\n"
        "- Treat each player's inventory as private and never copy items from other players.\n"
        "- For inventory changes, ONLY use player_state_update.inventory_add and player_state_update.inventory_remove arrays.\n"
        "- Do not return player_state_update.inventory full lists.\n"
        "- Each inventory item in RAILS_CONTEXT has a 'name' and 'origin' (how/where it was acquired). "
        "Respect item origins — never contradict or reinvent an item's backstory.\n"
        "- When a player must pick a path, accept only exact responses: 'main party' or 'new path'.\n"
        "- If the player has no room_summary or party_status, ask whether they are joining the main party or starting a new path, and set party_status accordingly.\n"
        "- NEVER include any inventory listing, summary, or 'Inventory:' line in narration. The emulator appends authoritative inventory automatically. "
        "Do not list, enumerate, or summarise what the player is carrying anywhere in the narration text — not at the end, not inline, not as a parenthetical.\n"
        "- Do not repeat full room descriptions or inventory unless asked or the room changes.\n"
        "- scene_image_prompt should describe the visible scene, not inventory lists.\n"
        "- When you output scene_image_prompt, it MUST be specific: include the room/location name and named characters from PARTY_SNAPSHOT (never generic 'group of adventurers').\n"
        "- Use PARTY_SNAPSHOT persona/attributes to describe each visible character's look/pose/style cues.\n"
        "- Include at least one concrete prop or action beat tied to the acting player.\n"
        "- Keep scene_image_prompt as a single dense paragraph, 70-180 words.\n"
        "- If IS_NEW_PLAYER is true and PLAYER_CARD.state.character_name is empty, generate a fitting name:\n"
        "  * If CAMPAIGN references a known movie/book/show, use the MAIN CHARACTER/PROTAGONIST's canonical name.\n"
        "  * Otherwise, create an appropriate name for this setting.\n"
        "  Set it in player_state_update.character_name.\n"
        "- PLAYER_CARD.state.character_name is ALWAYS the correct name for this player. Ignore any old names in WORLD_SUMMARY.\n"
        "- For other visible characters, always use the 'name' field from PARTY_SNAPSHOT. Never rename or confuse them.\n"
        "- Minimize mechanical text in narration. Do not narrate exits, room_summary, or state changes unless dramatically relevant.\n"
        "- Track location/exits in player_state_update, not in narration prose.\n"
        "- CRITICAL — OTHER PLAYER CHARACTERS ARE OFF-LIMITS:\n"
        "  PARTY_SNAPSHOT entries (except the acting player) are REAL HUMANS controlling their own characters.\n"
        "  You MUST NOT write ANY of the following for another player character:\n"
        "    * Dialogue or quoted speech\n"
        "    * Actions, movements, or decisions (e.g. 'she draws her sword', 'he follows you')\n"
        "    * Emotional reactions, facial expressions, or gestures in response to events\n"
        "    * Plot advancement involving them (e.g. 'together you storm the gate')\n"
        "    * Moving them to a new location or changing their state in any way\n"
        "  You MAY reference another player character in two cases:\n"
        "    1. Static presence — note they are in the room (e.g. 'X is here'), nothing more.\n"
        "    2. Continuing a prior action — if RECENT_TURNS shows that player ALREADY performed an action on their own turn\n"
        "       (e.g. 'I toss the key to you', 'I hold the door open'), you may narrate the CONSEQUENCE of that\n"
        "       established action as it affects the acting player (e.g. 'You catch the key X tossed'). \n"
        "       You are acknowledging what they did, not inventing new behaviour for them.\n"
        "  In ALL other cases, treat other player characters as scenery — they exist but do nothing until THEY act.\n"
        "  This turn's narration concerns ONLY the acting player identified by PLAYER_ACTION.\n"
        "- When mentioning a player character in narration, use their Discord mention from PARTY_SNAPSHOT followed by their name in parentheses, e.g. '<@123456> (Bruce Wayne)'. This pings the player in Discord so they know they were referenced.\n"
        "- NEVER skip or fast-forward time when a player sleeps, rests, or waits. Narrate only the moment of settling in (closing eyes, finding a spot to rest). Do NOT write 'hours pass', 'you wake at dawn', or advance to morning/next day. Other players share this world and time must not jump for one player's action. End the turn in the present moment.\n"
    )
    GUARDRAILS_SYSTEM_PROMPT = (
        "\nSTRICT RAILS MODE IS ENABLED.\n"
        "- Treat this as deterministic parser mode, not freeform improvisation.\n"
        "- Allow only actions that are immediately supported by current room facts, exits, inventory, and known actors.\n"
        "- Never permit teleportation, sudden scene jumps, retcons, instant mastery, or world-breaking powers unless explicitly present in WORLD_STATE.\n"
        "- If an action is invalid or unavailable, do not advance the world; return a short failure narration, and suggest concrete valid options.\n"
        "- For invalid actions, keep state_update as {} and player_state_update as {} and xp_awarded as 0.\n"
        "- Do not create new key items, exits, NPCs, or mechanics just to satisfy a request.\n"
        "- Use the provided RAILS_CONTEXT as hard constraints.\n"
    )
    TIMER_TOOL_PROMPT = (
        "\nTIMED EVENTS SYSTEM:\n"
        "You can schedule real countdown timers that fire automatically if the player doesn't act.\n"
        "To set a timer, include these EXTRA keys in your normal JSON response:\n"
        '- "set_timer_delay": integer (30-300 seconds) — REQUIRED for timer\n'
        '- "set_timer_event": string (what happens when the timer expires) — REQUIRED for timer\n'
        '- "set_timer_interruptible": boolean (default true; if false, timer keeps running even if player acts)\n'
        '- "set_timer_interrupt_action": string or null (what should happen when the player interrupts '
        "the timer by acting; null means just cancel silently; a description means the system will "
        "feed it back to you as context on the next turn so you can narrate the interruption)\n"
        "These go ALONGSIDE narration/state_update/etc in the same JSON object. Example:\n"
        '{"narration": "The ceiling groans ominously. Dust rains down...", '
        '"state_update": {"ceiling_status": "cracking"}, "summary_update": "Ceiling is unstable.", "xp_awarded": 0, '
        '"player_state_update": {"room_summary": "A crumbling chamber with a failing ceiling."}, '
        '"set_timer_delay": 120, "set_timer_event": "The ceiling collapses, burying the room in rubble.", '
        '"set_timer_interruptible": true, '
        '"set_timer_interrupt_action": "The player escapes just as cracks widen overhead."}\n'
        "The system shows a live countdown in Discord. "
        "If the player acts before it expires, the timer is cancelled (if interruptible). "
        "If the player does NOT act in time, the system auto-fires the event.\n"
        "PURPOSE: Timed events should FORCE THE PLAYER TO MAKE A DECISION or DRAG THEM WHERE THEY NEED TO BE.\n"
        "- Use timers to push the story forward when the player is stalling, idle, or refusing to engage.\n"
        "- NPCs should grab, escort, or coerce the player. Environments should shift and force movement.\n"
        "- The event should advance the plot: move the player to the next location, "
        "force an encounter, have an NPC intervene, or change the scene decisively.\n"
        "- Do NOT use timers for trivial flavor. They should always have real consequences that change game state.\n"
        "- Set interruptible=false for events the player cannot avoid (e.g. an earthquake, a mandatory roll call).\n"
        "Rules:\n"
        "- Use ~60s for urgent, ~120s for moderate, ~180-300s for slow-building tension.\n"
        "- Use whenever the scene has a deadline, the player is stalling, an NPC is impatient, "
        "or the world should move without the player.\n"
        "- Your narration should hint at urgency narratively (e.g. 'the footsteps grow louder') but NEVER include countdowns, timestamps, emoji clocks, or explicit seconds. The system adds its own countdown display automatically.\n"
        "- Use at least once every few turns when dramatic pacing allows. Do not use on consecutive turns.\n"
    )
    MEMORY_TOOL_PROMPT = (
        "\nYou have a memory_search tool. To use it, return ONLY:\n"
        '{"tool_call": "memory_search", "queries": ["query1", "query2", ...]}\n'
        "No other keys alongside tool_call. You may provide one or more queries.\n"
        "Use SEPARATE queries for each character or topic — do NOT combine multiple subjects into one query.\n"
        "Example: to recall Marcus and Anastasia, use:\n"
        '{"tool_call": "memory_search", "queries": ["Marcus", "Anastasia"]}\n'
        'NOT: {"tool_call": "memory_search", "queries": ["Marcus Anastasia relationship"]}\n'
        "USE memory_search AGGRESSIVELY — it is cheap and fast. Prefer searching too often over guessing.\n"
        "You SHOULD use memory_search on MOST turns. Specifically:\n"
        "- ANY time a character, NPC, or named entity appears or is mentioned — even if they were in recent turns. "
        "Memory may contain richer detail than the truncated recent context.\n"
        "- ANY time the player references past events, locations, objects, or conversations.\n"
        "- ANY time you are about to narrate a scene involving an established NPC — search their name first.\n"
        "- ANY time you need to describe a location the player has visited before.\n"
        "- At the START of most turns, search for the current location and any NPCs present to refresh your context.\n"
        "- When the player asks questions, investigates, or examines something — search for related terms.\n"
        "- When you are unsure about ANY detail from earlier in the campaign.\n"
        "The cost of an unnecessary search is zero. The cost of hallucinating a detail is broken continuity.\n"
        "When in doubt, SEARCH. Do not guess, improvise, or rely solely on RECENT_TURNS.\n"
        "IMPORTANT: Memories are stored as narrator event text (e.g. what happened in a scene). "
        "Queries are matched by semantic similarity against these narration snippets. "
        "Use short, concrete keyword queries with names and places — e.g. "
        '"Marcus penthouse", "Anastasia garden", "sword cave". '
        "Do NOT use abstract or relational queries like "
        '"character identity role relationship" — these will not match stored events.\n'
    )
    STORY_OUTLINE_TOOL_PROMPT = (
        "\nYou have a story_outline tool. To use it, return ONLY:\n"
        '{"tool_call": "story_outline", "chapter": "chapter-slug"}\n'
        "No other keys alongside tool_call.\n"
        "Returns full expanded chapter with all scene details.\n"
        "Use when you need details about a chapter not fully shown in STORY_CONTEXT.\n"
    )
    CALENDAR_TOOL_PROMPT = (
        "\nCALENDAR & GAME TIME SYSTEM:\n"
        "The campaign tracks in-game time via CURRENT_GAME_TIME shown in the user prompt.\n"
        "Every turn, you MUST advance game_time in state_update by a plausible amount "
        "(minutes for quick actions, hours for travel, etc.). "
        "Scale the advance by SPEED_MULTIPLIER — at 2x, time passes roughly twice as fast per turn.\n"
        "Update these fields in state_update:\n"
        '- "game_time": {"day": int, "hour": int (0-23), "minute": int (0-59), '
        '"period": "morning"|"afternoon"|"evening"|"night", '
        '"date_label": "Day N, Period"}\n'
        "Advance hour/minute naturally; when hour >= 24, increment day and wrap hour.\n"
        "Set period based on hour: 5-11=morning, 12-16=afternoon, 17-20=evening, 21-4=night.\n\n"
        "You may also return a calendar_update key (object) to manage scheduled events:\n"
        '- "calendar_update": {"add": [...], "remove": [...]} where each add entry is '
        '{"name": str, "time_remaining": int, "time_unit": "hours"|"days", "description": str} '
        "and each remove entry is a string matching an event name.\n"
        "HARNESS BEHAVIOR:\n"
        "- The harness converts add entries into absolute due dates and stores fire_day (the game day an event fires).\n"
        "- Do NOT decrement counters manually by re-adding events each turn. The harness computes remaining days automatically.\n"
        "- You will receive CALENDAR_REMINDERS in the prompt for imminent/overdue events.\n"
        "CALENDAR EVENT LIFECYCLE:\n"
        "Events should progress through phases based on fire_day vs CURRENT_GAME_TIME.day:\n"
        "1. UPCOMING — event is in the future. Mention it naturally when relevant (NPCs remind the player, "
        "signs/clues reference it).\n"
        "2. IMMINENT — event is today or tomorrow. Actively warn the player: NPCs urge action, "
        "the environment reflects urgency. Narrate pressure to act. The player should feel they need to DO something.\n"
        "3. OVERDUE — current day is past fire_day. Do NOT remove the event. "
        "Narrate consequences escalating. "
        "NPCs express disappointment, opportunities narrow, penalties mount. "
        "The event stays on the calendar as a visible reminder of what the player neglected.\n"
        "4. RESOLVED — ONLY remove an event when the player has DIRECTLY DEALT WITH IT "
        "(attended, completed, deliberately abandoned) and the outcome has been narrated. "
        "Do NOT silently prune events. Do NOT remove events just because they are overdue.\n\n"
        "CRITICAL — calendar_update.remove rules:\n"
        "- ONLY remove an event when it has been RESOLVED through player action in the current narration.\n"
        "- NEVER remove events because time passed or they feel old. Overdue events stay and get worse.\n"
        "- If you are unsure whether an event should be removed, do NOT remove it.\n"
        "Use calendar events for approaching deadlines, NPC appointments, world events, "
        "and anything with narrative timing pressure.\n"
    )
    ROSTER_PROMPT = (
        "\nCHARACTER ROSTER & PORTRAITS:\n"
        "The harness maintains a character roster (WORLD_CHARACTERS). "
        "When you create or update a character via character_updates, the 'appearance' field "
        "is used by the harness to auto-generate a portrait image. Write 'appearance' as a "
        "detailed visual description suitable for image generation: physical features, clothing, "
        "distinguishing marks, pose, and art style cues. Keep it 1-3 sentences, "
        "70-150 words, vivid and concrete.\n"
        "Do NOT include image_url in character_updates — the harness manages that field.\n"
    )
    ON_RAILS_SYSTEM_PROMPT = (
        "\nON-RAILS MODE IS ENABLED.\n"
        "- You CANNOT create new characters not in WORLD_CHARACTERS. New character slugs will be rejected.\n"
        "- You CANNOT introduce locations/landmarks not in story_outline or landmarks list.\n"
        "- You CANNOT add new chapters or scenes beyond STORY_CONTEXT.\n"
        "- You MUST advance along the current chapter/scene trajectory.\n"
        "- Adjust pacing/details within scenes, but major plot points must match the outline.\n"
        "- Use state_update.current_chapter / state_update.current_scene to advance.\n"
        "- If player tries to derail, steer back via NPC actions or environmental events.\n"
    )
    MAP_SYSTEM_PROMPT = (
        "You draw compact ASCII maps for text adventures.\n"
        "Return ONLY the ASCII map (no markdown, no code fences).\n"
        "Keep it under 25 lines and 60 columns. Use @ for the player location.\n"
        "Use simple ASCII only: - | + . # / \\ and letters.\n"
        "Include other player markers (A, B, C, ...) and add a Legend at the bottom.\n"
        "In the Legend, use PLAYER_NAME for @ and character_name from OTHER_PLAYERS for each marker.\n"
    )
    IMDB_SUGGEST_URL = "https://v2.sg.media-imdb.com/suggestion/{first}/{query}.json"
    IMDB_TIMEOUT = 5
    _inflight_turns = set()

    def __init__(
        self,
        game_engine: GameEngine,
        session_factory,
        *,
        completion_port: TextCompletionPort | None = None,
        map_completion_port: TextCompletionPort | None = None,
        timer_effects_port: TimerEffectsPort | None = None,
        memory_port: MemorySearchPort | None = None,
        imdb_port: IMDBLookupPort | None = None,
        media_port: MediaGenerationPort | None = None,
    ):
        self._engine = game_engine
        self._session_factory = session_factory
        self._claims: dict[tuple[str, str], TurnClaim] = {}
        self._completion_port = completion_port
        self._map_completion_port = map_completion_port or completion_port
        self._timer_effects_port = timer_effects_port
        self._memory_port = memory_port
        self._imdb_port = imdb_port
        self._media_port = media_port
        self._logger = logging.getLogger(__name__)
        self._inflight_turns: set[tuple[str, str]] = set()
        self._inflight_turns_lock = threading.Lock()
        self._attachment_processor = (
            AttachmentTextProcessor(
                completion=completion_port,
                config=AttachmentProcessingConfig(
                    attachment_max_bytes=self.ATTACHMENT_MAX_BYTES,
                    attachment_chunk_tokens=self.ATTACHMENT_CHUNK_TOKENS,
                    attachment_model_ctx_tokens=self.ATTACHMENT_MODEL_CTX_TOKENS,
                    attachment_prompt_overhead_tokens=self.ATTACHMENT_PROMPT_OVERHEAD_TOKENS,
                    attachment_response_reserve_tokens=self.ATTACHMENT_RESPONSE_RESERVE_TOKENS,
                    attachment_max_parallel=self.ATTACHMENT_MAX_PARALLEL,
                    attachment_guard_token=self.ATTACHMENT_GUARD_TOKEN,
                    attachment_max_chunks=self.ATTACHMENT_MAX_CHUNKS,
                ),
            )
            if completion_port is not None
            else None
        )
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending_timers: dict[str, dict[str, Any]] = {}

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

    @staticmethod
    def _now() -> datetime:
        return datetime.utcnow()

    @staticmethod
    def _format_utc_timestamp(value: datetime) -> str:
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.replace(microsecond=0).isoformat() + "Z"

    @staticmethod
    def _parse_utc_timestamp(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except Exception:
            return None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    @staticmethod
    def _coerce_non_negative_int(value: object, default: int = 0) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0 else default

    def _default_player_stats(self) -> dict[str, object]:
        return {
            self.PLAYER_STATS_MESSAGES_KEY: 0,
            self.PLAYER_STATS_TIMERS_AVERTED_KEY: 0,
            self.PLAYER_STATS_TIMERS_MISSED_KEY: 0,
            self.PLAYER_STATS_ATTENTION_SECONDS_KEY: 0,
            self.PLAYER_STATS_LAST_MESSAGE_AT_KEY: None,
        }

    def _get_player_stats_from_state(self, player_state: dict[str, object]) -> dict[str, object]:
        stats = self._default_player_stats()
        if not isinstance(player_state, dict):
            return stats
        raw_stats = player_state.get(self.PLAYER_STATS_KEY, {})
        if not isinstance(raw_stats, dict):
            return stats
        stats[self.PLAYER_STATS_MESSAGES_KEY] = self._coerce_non_negative_int(
            raw_stats.get(self.PLAYER_STATS_MESSAGES_KEY),
            0,
        )
        stats[self.PLAYER_STATS_TIMERS_AVERTED_KEY] = self._coerce_non_negative_int(
            raw_stats.get(self.PLAYER_STATS_TIMERS_AVERTED_KEY),
            0,
        )
        stats[self.PLAYER_STATS_TIMERS_MISSED_KEY] = self._coerce_non_negative_int(
            raw_stats.get(self.PLAYER_STATS_TIMERS_MISSED_KEY),
            0,
        )
        stats[self.PLAYER_STATS_ATTENTION_SECONDS_KEY] = self._coerce_non_negative_int(
            raw_stats.get(self.PLAYER_STATS_ATTENTION_SECONDS_KEY),
            0,
        )
        last_message_at = self._parse_utc_timestamp(raw_stats.get(self.PLAYER_STATS_LAST_MESSAGE_AT_KEY))
        if last_message_at is not None:
            stats[self.PLAYER_STATS_LAST_MESSAGE_AT_KEY] = self._format_utc_timestamp(last_message_at)
        return stats

    def _set_player_stats_on_state(
        self,
        player_state: dict[str, object],
        stats: dict[str, object],
    ) -> dict[str, object]:
        if not isinstance(player_state, dict):
            player_state = {}
        player_state[self.PLAYER_STATS_KEY] = self._get_player_stats_from_state({self.PLAYER_STATS_KEY: stats})
        return player_state

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

    def _load_session_metadata(self, session_row: GameSession) -> dict[str, Any]:
        meta = self._load_json(session_row.metadata_json, {})
        return meta if isinstance(meta, dict) else {}

    def _store_session_metadata(self, session_row: GameSession, metadata: dict[str, Any]) -> None:
        session_row.metadata_json = self._dump_json(metadata)

    def get_or_create_channel(self, guild_id: str | int, channel_id: str | int) -> GameSession:
        guild = str(guild_id)
        channel = str(channel_id)
        key = f"discord:{guild}:{channel}"
        self.get_or_create_actor("system", display_name="System")
        with self._session_factory() as session:
            row = (
                session.query(GameSession)
                .filter(GameSession.surface == "discord_channel")
                .filter(GameSession.surface_key == key)
                .first()
            )
            if row is None:
                row = GameSession(
                    campaign_id=self.get_or_create_campaign(guild, "main", created_by_actor_id="system").id,
                    surface="discord_channel",
                    surface_key=key,
                    surface_guild_id=guild,
                    surface_channel_id=channel,
                    enabled=False,
                    metadata_json=self._dump_json({"active_campaign_id": None}),
                )
                session.add(row)
                session.commit()
            return row

    def is_channel_enabled(self, guild_id: str | int, channel_id: str | int) -> bool:
        row = self.get_or_create_channel(guild_id, channel_id)
        return bool(row.enabled)

    def enable_channel(
        self,
        guild_id: str | int,
        channel_id: str | int,
        actor_id: str,
    ) -> tuple[GameSession, Campaign]:
        guild = str(guild_id)
        row = self.get_or_create_channel(guild, channel_id)
        with self._session_factory() as session:
            channel_row = session.get(GameSession, row.id)
            meta = self._load_session_metadata(channel_row)
            active_campaign_id = meta.get("active_campaign_id")
            campaign = session.get(Campaign, active_campaign_id) if active_campaign_id else None
            if campaign is None:
                campaign = self.get_or_create_campaign(guild, "main", actor_id)
                active_campaign_id = campaign.id
            meta["active_campaign_id"] = active_campaign_id
            channel_row.enabled = True
            self._store_session_metadata(channel_row, meta)
            channel_row.updated_at = datetime.utcnow()
            session.commit()
            campaign = session.get(Campaign, active_campaign_id)
            return channel_row, campaign

    def can_switch_campaign(
        self,
        campaign_id: str,
        actor_id: str,
        window_seconds: int = 3600,
    ) -> tuple[bool, int]:
        cutoff = self._now() - timedelta(seconds=window_seconds)
        with self._session_factory() as session:
            active_count = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id != actor_id)
                .filter(Player.last_active_at != None)  # noqa: E711
                .filter(Player.last_active_at >= cutoff)
                .count()
            )
            return active_count == 0, active_count

    def set_active_campaign(
        self,
        channel: GameSession,
        guild_id: str | int,
        name: str,
        actor_id: str,
        enforce_activity_window: bool = True,
    ) -> tuple[Campaign | None, bool, str | None]:
        normalized = self._normalize_campaign_name(name)
        with self._session_factory() as session:
            channel_row = session.get(GameSession, channel.id)
            meta = self._load_session_metadata(channel_row)
            current_campaign_id = meta.get("active_campaign_id")
            if enforce_activity_window and current_campaign_id:
                can_switch, active_count = self.can_switch_campaign(str(current_campaign_id), actor_id)
                if not can_switch:
                    return None, False, f"{active_count} other player(s) active in last hour"
            campaign = self.get_or_create_campaign(str(guild_id), normalized, actor_id)
            meta["active_campaign_id"] = campaign.id
            self._store_session_metadata(channel_row, meta)
            channel_row.updated_at = datetime.utcnow()
            session.commit()
            return campaign, True, None

    def _is_context_like(self, value: Any) -> bool:
        return (
            value is not None
            and hasattr(value, "guild")
            and hasattr(value, "channel")
            and hasattr(value, "author")
        )

    def _resolve_campaign_for_context(
        self,
        ctx,
        *,
        command_prefix: str = "!",
    ) -> tuple[str | None, str | None]:
        guild = getattr(ctx, "guild", None)
        channel_obj = getattr(ctx, "channel", None)
        author = getattr(ctx, "author", None)
        if guild is None or channel_obj is None or author is None:
            return None, "Zork is only available in servers."
        guild_id = str(getattr(guild, "id", "") or "")
        channel_id = str(getattr(channel_obj, "id", "") or "")
        actor_id = str(getattr(author, "id", "") or "")
        if not guild_id or not channel_id or not actor_id:
            return None, "Zork is only available in servers."

        channel = self.get_or_create_channel(guild_id, channel_id)
        if not channel.enabled:
            return (
                None,
                f"Adventure mode is disabled in this channel. Run `{command_prefix}zork` to enable it.",
            )

        metadata = self._load_session_metadata(channel)
        active_campaign_id = metadata.get("active_campaign_id")
        if active_campaign_id:
            with self._session_factory() as session:
                campaign = session.get(Campaign, str(active_campaign_id))
            if campaign is not None:
                return str(active_campaign_id), None

        _, campaign = self.enable_channel(guild_id, channel_id, actor_id)
        return campaign.id, None

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
        with self._session_factory() as session:
            row = session.get(Player, player.id)
            if row is not None:
                return parse_json_dict(row.state_json)
        return parse_json_dict(player.state_json)

    def get_player_attributes(self, player: Player) -> dict[str, int]:
        with self._session_factory() as session:
            row = session.get(Player, player.id)
            if row is not None:
                data = parse_json_dict(row.attributes_json)
            else:
                data = parse_json_dict(player.attributes_json)
        out: dict[str, int] = {}
        for key, value in data.items():
            if isinstance(value, int):
                out[str(key)] = value
        return out

    def get_campaign_state(self, campaign: Campaign) -> dict[str, Any]:
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is not None:
                return parse_json_dict(row.state_json)
        return parse_json_dict(campaign.state_json)

    def get_campaign_characters(self, campaign: Campaign) -> dict[str, Any]:
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is not None:
                return parse_json_dict(row.characters_json)
        return parse_json_dict(campaign.characters_json)

    def record_player_message(
        self,
        player: Player,
        observed_at: datetime | None = None,
    ) -> dict[str, object]:
        now_dt = observed_at or self._now()
        if now_dt.tzinfo is not None:
            now_dt = now_dt.astimezone(timezone.utc).replace(tzinfo=None)

        player_state = self.get_player_state(player)
        stats = self._get_player_stats_from_state(player_state)
        last_message_at = self._parse_utc_timestamp(stats.get(self.PLAYER_STATS_LAST_MESSAGE_AT_KEY))
        if last_message_at is not None:
            gap_seconds = (now_dt - last_message_at).total_seconds()
            if 0 < gap_seconds < self.ATTENTION_WINDOW_SECONDS:
                stats[self.PLAYER_STATS_ATTENTION_SECONDS_KEY] = self._coerce_non_negative_int(
                    stats.get(self.PLAYER_STATS_ATTENTION_SECONDS_KEY),
                    0,
                ) + int(gap_seconds)

        stats[self.PLAYER_STATS_MESSAGES_KEY] = self._coerce_non_negative_int(
            stats.get(self.PLAYER_STATS_MESSAGES_KEY),
            0,
        ) + 1
        stats[self.PLAYER_STATS_LAST_MESSAGE_AT_KEY] = self._format_utc_timestamp(now_dt)
        player_state = self._set_player_stats_on_state(player_state, stats)

        with self._session_factory() as session:
            row = session.get(Player, player.id)
            if row is not None:
                row.state_json = self._dump_json(player_state)
                row.updated_at = datetime.utcnow()
                row.last_active_at = datetime.utcnow()
                session.commit()
                player.state_json = row.state_json
                player.updated_at = row.updated_at
                player.last_active_at = row.last_active_at
        return stats

    def increment_player_stat(
        self,
        player: Player,
        stat_key: str,
        increment: int = 1,
    ) -> dict[str, object]:
        if increment <= 0:
            return self.get_player_statistics(player)
        player_state = self.get_player_state(player)
        stats = self._get_player_stats_from_state(player_state)
        current = self._coerce_non_negative_int(stats.get(stat_key), 0)
        stats[stat_key] = current + int(increment)
        player_state = self._set_player_stats_on_state(player_state, stats)
        with self._session_factory() as session:
            row = session.get(Player, player.id)
            if row is not None:
                row.state_json = self._dump_json(player_state)
                row.updated_at = datetime.utcnow()
                session.commit()
                player.state_json = row.state_json
                player.updated_at = row.updated_at
        return stats

    def get_player_statistics(self, player: Player) -> dict[str, object]:
        player_state = self.get_player_state(player)
        stats = self._get_player_stats_from_state(player_state)
        attention_seconds = self._coerce_non_negative_int(stats.get(self.PLAYER_STATS_ATTENTION_SECONDS_KEY), 0)
        stats["attention_hours"] = round(attention_seconds / 3600.0, 2)
        return stats

    def _normalize_campaign_name(self, name: str) -> str:
        return normalize_campaign_name(name)

    def _get_preset_campaign(self, normalized_name: str) -> dict[str, Any] | None:
        key = self.PRESET_ALIASES.get(normalized_name)
        if not key:
            return None
        return self.PRESET_CAMPAIGNS.get(key)

    def get_campaign_default_persona(
        self,
        campaign: Campaign | None,
        campaign_state: dict[str, object] | None = None,
    ) -> str:
        if campaign is None:
            return self.DEFAULT_CAMPAIGN_PERSONA
        normalized = self._normalize_campaign_name(campaign.name or "")
        alias_key = self.PRESET_ALIASES.get(normalized)
        if alias_key and alias_key in self.PRESET_DEFAULT_PERSONAS:
            return self.PRESET_DEFAULT_PERSONAS[alias_key]
        if isinstance(campaign_state, dict):
            setting_text = str(campaign_state.get("setting") or "").strip().lower()
            if "alice" in setting_text or "wonderland" in setting_text:
                return self.PRESET_DEFAULT_PERSONAS["alice"]
            stored = campaign_state.get("default_persona")
            if isinstance(stored, str) and stored.strip():
                return stored.strip()
        return self.DEFAULT_CAMPAIGN_PERSONA

    async def generate_campaign_persona(self, campaign_name: str) -> str:
        if self._completion_port is None:
            return self.DEFAULT_CAMPAIGN_PERSONA
        prompt = (
            f"The campaign is titled: '{campaign_name}'.\n"
            "If this references a known movie, book, show, or story, create a persona for the main character.\n"
            "Return only a brief persona (1-2 sentences, max 140 chars)."
        )
        try:
            response = await self._completion_port.complete(
                prompt,
                "",
                temperature=0.7,
                max_tokens=80,
            )
            if response:
                persona = response.strip().strip('"').strip("'")
                return self._trim_text(persona, 140)
        except Exception:
            return self.DEFAULT_CAMPAIGN_PERSONA
        return self.DEFAULT_CAMPAIGN_PERSONA

    def _imdb_search_single(self, query: str, max_results: int = 3) -> list[dict[str, Any]]:
        clean = re.sub(r"[^\w\s]", "", query.strip().lower())
        if not clean:
            return []
        first = clean[0] if clean[0].isalpha() else "a"
        encoded = urllib_parse.quote(clean.replace(" ", "_"))
        url = self.IMDB_SUGGEST_URL.format(first=first, query=encoded)
        request = urllib_request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib_request.urlopen(request, timeout=self.IMDB_TIMEOUT) as response:  # noqa: S310
                if response.status != 200:
                    return []
                payload = response.read().decode("utf-8", errors="replace")
        except Exception:
            return []
        try:
            data = json.loads(payload)
        except Exception:
            return []
        results: list[dict[str, Any]] = []
        for item in data.get("d", [])[:max_results]:
            if not isinstance(item, dict):
                continue
            title = item.get("l")
            if not title:
                continue
            results.append(
                {
                    "imdb_id": item.get("id", ""),
                    "title": title,
                    "year": item.get("y"),
                    "type": item.get("q", ""),
                    "stars": item.get("s", ""),
                }
            )
        return results

    def _imdb_search(self, query: str, max_results: int = 3) -> list[dict[str, Any]]:
        if self._imdb_port is not None:
            try:
                results = list(self._imdb_port.search(query, max_results=max_results))
                if results:
                    return results
            except Exception:
                pass
        try:
            results = self._imdb_search_single(query, max_results=max_results)
            if results:
                return results
            stripped = re.sub(
                r"\b(s\d+e\d+|season\s*\d+|episode\s*\d+|ep\s*\d+)\b",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()
            if stripped and stripped != query:
                results = self._imdb_search_single(stripped, max_results=max_results)
                if results:
                    return results
            words = query.strip().split()
            for length in range(len(words) - 1, 1, -1):
                sub = " ".join(words[:length])
                results = self._imdb_search_single(sub, max_results=max_results)
                if results:
                    return results
            return []
        except Exception:
            return []

    def _imdb_fetch_details(self, imdb_id: str) -> dict[str, Any]:
        fetch = getattr(self._imdb_port, "fetch_details", None) if self._imdb_port else None
        if callable(fetch):
            try:
                result = fetch(imdb_id)
                return dict(result) if isinstance(result, dict) else {}
            except Exception:
                pass
        if not imdb_id or not imdb_id.startswith("tt"):
            return {}
        try:
            url = f"https://www.imdb.com/title/{imdb_id}/"
            request = urllib_request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            with urllib_request.urlopen(request, timeout=self.IMDB_TIMEOUT + 3) as response:  # noqa: S310
                if response.status != 200:
                    return {}
                html = response.read().decode("utf-8", errors="replace")
            match = re.search(
                r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
                html,
                re.DOTALL,
            )
            if not match:
                return {}
            ld_data = json.loads(match.group(1))
            if not isinstance(ld_data, dict):
                return {}
            details: dict[str, Any] = {}
            description = ld_data.get("description")
            if description:
                details["description"] = description
            genre = ld_data.get("genre")
            if genre:
                details["genre"] = genre if isinstance(genre, list) else [genre]
            actors = ld_data.get("actor", [])
            if isinstance(actors, list) and actors:
                details["actors"] = [
                    actor.get("name", "")
                    for actor in actors[:6]
                    if isinstance(actor, dict) and actor.get("name")
                ]
            return details
        except Exception:
            return {}

    def _imdb_enrich_results(
        self,
        results: list[dict[str, Any]],
        max_enrich: int = 1,
    ) -> list[dict[str, Any]]:
        if self._imdb_port is not None:
            try:
                enriched = self._imdb_port.enrich(results)
                return list(enriched) if isinstance(enriched, list) else results
            except Exception:
                return results
        for row in results[:max_enrich]:
            if not isinstance(row, dict):
                continue
            imdb_id = str(row.get("imdb_id") or "")
            if not imdb_id:
                continue
            details = self._imdb_fetch_details(imdb_id)
            description = details.get("description")
            if description:
                row["description"] = description
            genre = details.get("genre")
            if genre:
                row["genre"] = genre
            actors = details.get("actors")
            if actors:
                row["stars"] = ", ".join(actors)
        return results

    def _format_imdb_results(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return ""
        lines: list[str] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            year_str = f" ({row['year']})" if row.get("year") else ""
            type_str = f" [{row['type']}]" if row.get("type") else ""
            stars_str = f" — {row['stars']}" if row.get("stars") else ""
            genre_str = ""
            if row.get("genre"):
                genre_str = (
                    f" [{', '.join(row['genre'])}]"
                    if isinstance(row["genre"], list)
                    else f" [{row['genre']}]"
                )
            desc_str = ""
            if row.get("description"):
                desc_str = f"\n  Synopsis: {row['description']}"
            lines.append(f"- {title}{year_str}{type_str}{genre_str}{stars_str}{desc_str}")
        return "\n".join(lines)

    async def _extract_attachment_text(self, message) -> Optional[str]:
        attachments = getattr(message, "attachments", None)
        return await extract_attachment_text(
            attachments,
            config=AttachmentProcessingConfig(
                attachment_max_bytes=self.ATTACHMENT_MAX_BYTES,
                attachment_chunk_tokens=self.ATTACHMENT_CHUNK_TOKENS,
                attachment_model_ctx_tokens=self.ATTACHMENT_MODEL_CTX_TOKENS,
                attachment_prompt_overhead_tokens=self.ATTACHMENT_PROMPT_OVERHEAD_TOKENS,
                attachment_response_reserve_tokens=self.ATTACHMENT_RESPONSE_RESERVE_TOKENS,
                attachment_max_parallel=self.ATTACHMENT_MAX_PARALLEL,
                attachment_guard_token=self.ATTACHMENT_GUARD_TOKEN,
                attachment_max_chunks=self.ATTACHMENT_MAX_CHUNKS,
            ),
            logger=self._logger,
        )

    async def _summarise_long_text(self, text: str, ctx_message=None, channel=None) -> str:
        if not text:
            return ""
        if self._attachment_processor is None:
            return text

        progress_channel = channel
        if progress_channel is None and ctx_message is not None:
            progress_channel = getattr(ctx_message, "channel", None)

        status_message = None

        async def _progress(update: str):
            nonlocal status_message
            if progress_channel is None or not hasattr(progress_channel, "send"):
                return
            try:
                if status_message is None:
                    status_message = await progress_channel.send(update)
                elif hasattr(status_message, "edit"):
                    await status_message.edit(content=update)
            except Exception:
                return

        summary = await self._attachment_processor.summarise_long_text(
            text,
            progress=_progress if progress_channel is not None else None,
        )
        if status_message is not None and hasattr(status_message, "delete"):
            try:
                await status_message.delete()
            except Exception:
                pass
        return summary

    async def _summarise_chunk(
        self,
        chunk_text: str,
        *,
        summarise_system: str,
        summary_max_tokens: int,
        guard: str,
    ) -> str:
        if self._completion_port is None:
            return ""
        try:
            result = await self._completion_port.complete(
                summarise_system,
                chunk_text,
                max_tokens=summary_max_tokens,
                temperature=0.3,
            )
            result = (result or "").strip()
            if guard not in result:
                self._logger.warning("Guard token missing, retrying chunk")
                result = await self._completion_port.complete(
                    summarise_system,
                    chunk_text,
                    max_tokens=summary_max_tokens,
                    temperature=0.3,
                )
                result = (result or "").strip()
                if guard not in result:
                    self._logger.warning("Guard token still missing, accepting as-is")
            return result.replace(guard, "").strip()
        except Exception as exc:
            self._logger.warning("Chunk summarisation failed: %s", exc)
            return ""

    async def _condense(
        self,
        idx: int,
        summary_text: str,
        *,
        target_tokens_per: int,
        target_chars_per: int,
        guard: str,
    ) -> tuple[int, str]:
        if self._completion_port is None:
            return idx, summary_text
        condense_system = (
            f"Condense this summary to roughly {target_tokens_per} tokens "
            f"(~{target_chars_per} characters) "
            "while preserving all character names, plot points, and locations. "
            f"End with: {guard}"
        )
        try:
            result = await self._completion_port.complete(
                condense_system,
                summary_text,
                max_tokens=target_tokens_per + 50,
                temperature=0.2,
            )
            result = (result or "").strip()
            if guard not in result:
                self._logger.warning("Guard token missing in condensation, accepting as-is")
            return idx, result.replace(guard, "").strip()
        except Exception as exc:
            self._logger.warning("Condensation failed: %s", exc)
            return idx, summary_text

    def is_in_setup_mode(self, campaign: Campaign | None) -> bool:
        if campaign is None:
            return False
        state = self.get_campaign_state(campaign)
        phase = str(state.get("setup_phase") or "").strip()
        return bool(phase and phase != "completed")

    async def start_campaign_setup(
        self,
        campaign_id: str | Campaign,
        actor_id: str | None = None,
        raw_name: str | None = None,
        *,
        attachment_text: str | None = None,
        attachment_summary: str | None = None,
        on_rails: bool = False,
    ) -> str:
        # Legacy compatibility: start_campaign_setup(campaign, raw_name, attachment_summary=...)
        if isinstance(campaign_id, Campaign):
            campaign_obj = campaign_id
            if raw_name is None and isinstance(actor_id, str):
                raw_name = actor_id
                actor_id = campaign_obj.created_by_actor_id or "system"
            elif actor_id is None:
                actor_id = campaign_obj.created_by_actor_id or "system"
            campaign_id = campaign_obj.id
        if raw_name is None:
            return "Campaign not found."
        if actor_id is None:
            actor_id = "system"
        if attachment_text is None and attachment_summary is not None:
            attachment_text = attachment_summary

        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return "Campaign not found."
            state = parse_json_dict(campaign.state_json)
            imdb_results = self._imdb_search(raw_name, max_results=3)
            imdb_text = self._format_imdb_results(imdb_results)

            imdb_context = ""
            if imdb_text:
                imdb_context = (
                    f"\nIMDB search results for '{raw_name}':\n{imdb_text}\n"
                    "Use these results to help identify the work.\n"
                )
            attachment_context = ""
            if attachment_text:
                attachment_context = (
                    "\nThe user also uploaded source material. Summary of uploaded text:\n"
                    f"{attachment_text}\n"
                    "Use this to identify the work.\n"
                )

            is_known = False
            work_type = None
            work_desc = ""
            suggested = raw_name
            if self._completion_port is not None:
                classify_system = (
                    "You classify whether text references a known published work "
                    "(movie, book, TV show, video game, etc).\n"
                    "Return ONLY valid JSON with these keys:\n"
                    '- "is_known_work": boolean\n'
                    '- "work_type": string or null\n'
                    '- "work_description": string or null\n'
                    '- "suggested_title": string\n'
                    "No markdown, no code fences."
                )
                classify_user = (
                    f"The user wants to play a campaign called: '{raw_name}'.\n"
                    f"{imdb_context}"
                    f"{attachment_context}"
                    "Is this a known published work? Provide the canonical title and description."
                )
                try:
                    response = await self._completion_port.complete(
                        classify_system,
                        classify_user,
                        temperature=0.3,
                        max_tokens=300,
                    )
                    response = self._clean_response(response or "{}")
                    json_text = self._extract_json(response)
                    result = self._parse_json_lenient(json_text) if json_text else {}
                except Exception:
                    result = {}
                is_known = bool(result.get("is_known_work", False))
                work_type = result.get("work_type")
                work_desc = result.get("work_description") or ""
                suggested = result.get("suggested_title") or raw_name

            if not is_known and imdb_results:
                top = imdb_results[0]
                top = self._imdb_enrich_results([top])[0]
                is_known = True
                suggested = str(top.get("title") or suggested)
                work_type = (str(top.get("type") or "other").lower().replace(" ", "_")) or "other"
                work_desc = str(top.get("description") or "").strip()
                if not work_desc:
                    year_str = f" ({top.get('year')})" if top.get("year") else ""
                    stars = str(top.get("stars") or "").strip()
                    work_desc = f"{suggested}{year_str}"
                    if stars:
                        work_desc += f" starring {stars}"

            setup_data: dict[str, Any] = {
                "raw_name": suggested if is_known else raw_name,
                "is_known_work": is_known,
                "work_type": work_type,
                "work_description": work_desc,
                "imdb_results": imdb_results or [],
                "requested_by": actor_id,
                "on_rails_requested": bool(on_rails),
                "default_persona": await self.generate_campaign_persona(raw_name),
            }
            if attachment_text:
                setup_data["attachment_summary"] = attachment_text

            state["setup_phase"] = "classify_confirm"
            state["setup_data"] = setup_data
            campaign.state_json = self._dump_json(state)
            campaign.updated_at = datetime.utcnow()
            session.commit()
        if is_known:
            return (
                f"I recognize **{setup_data['raw_name']}** as a known {work_type or 'work'}.\n"
                f"_{work_desc}_\n\n"
                "Is this correct? Reply **yes** to confirm, or tell me what it actually is."
            )
        return (
            f"I don't recognize **{raw_name}** as a known published work. "
            "I'll treat it as an original setting.\n\n"
            "Is this correct? Reply **yes** to confirm, or tell me what it actually is."
        )

    async def handle_setup_message(
        self,
        campaign_id: str | Any,
        actor_id: str | Any,
        message_text: str | Any,
        *,
        attachments: list[Any] | None = None,
        command_prefix: str = "!",
    ) -> str:
        # Legacy compatibility:
        # handle_setup_message(ctx, content, campaign, command_prefix="!")
        if (
            not isinstance(campaign_id, (str, int))
            and hasattr(campaign_id, "guild")
            and hasattr(campaign_id, "channel")
            and isinstance(message_text, Campaign)
        ):
            ctx = campaign_id
            content = actor_id
            campaign = message_text
            campaign_id = campaign.id
            actor_id = str(getattr(getattr(ctx, "author", None), "id", "") or campaign.created_by_actor_id or "system")
            message_text = str(content or "")
            if attachments is None:
                ctx_message = getattr(ctx, "message", None)
                attachments = getattr(ctx_message, "attachments", None)

        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return "Campaign not found."
            state = parse_json_dict(campaign.state_json)
            setup_data = state.get("setup_data", {})
            if not isinstance(setup_data, dict):
                setup_data = {}
            phase = str(state.get("setup_phase") or "").strip()
            if not phase:
                return "Setup is not active."

            clean_text = str(message_text or "").strip()
            if phase == "classify_confirm":
                result = await self._setup_handle_classify_confirm(
                    campaign,
                    state,
                    setup_data,
                    clean_text,
                    attachments=attachments,
                )
            elif phase == "storyline_pick":
                result = await self._setup_handle_storyline_pick(
                    campaign,
                    state,
                    setup_data,
                    clean_text,
                    actor_id=actor_id,
                )
            elif phase == "novel_questions":
                result = await self._setup_handle_novel_questions(
                    campaign,
                    state,
                    setup_data,
                    clean_text,
                    actor_id=actor_id,
                )
            elif phase == "finalize":
                result = await self._setup_finalize(
                    campaign,
                    state,
                    setup_data,
                    user_id=actor_id,
                    db_session=session,
                )
            else:
                state.pop("setup_phase", None)
                state.pop("setup_data", None)
                result = "Setup cleared. You can now play normally."

            campaign.state_json = self._dump_json(state)
            campaign.updated_at = datetime.utcnow()
            session.commit()
            return result

    async def _setup_generate_draft(
        self,
        campaign: Campaign,
        actor_id: str,
        source_prompt: str,
        attachment_summary: str,
        setup_data: dict[str, Any],
    ) -> dict[str, Any]:
        default_persona = str(setup_data.get("default_persona") or self.DEFAULT_CAMPAIGN_PERSONA)
        base = {
            "summary": source_prompt or campaign.summary or "A new adventure begins.",
            "state": {
                "setting": source_prompt or campaign.name,
                "on_rails": bool(setup_data.get("on_rails", False)),
                "default_persona": default_persona,
            },
            "start_room": {
                "room_title": "Starting Point",
                "room_summary": "The first room of your adventure.",
                "room_description": "A world stirs as your adventure begins.",
                "exits": ["look around", "move forward"],
                "location": "start",
            },
            "opening": "The world sharpens around you as the adventure begins.",
            "characters": {},
        }
        if self._completion_port is None:
            return base

        prompt = (
            "Build campaign setup JSON for a text adventure.\n"
            "Return strict JSON with keys: summary, state, start_room, opening, characters.\n"
            f"CAMPAIGN={campaign.name}\n"
            f"ACTOR={actor_id}\n"
            f"SOURCE_PROMPT={source_prompt}\n"
            f"ATTACHMENT_SUMMARY={attachment_summary}\n"
            f"IMDB_CANDIDATES={self._dump_json(self._imdb_enrich_results(setup_data.get('imdb_candidates', [])))}\n"
        )
        try:
            response = await self._completion_port.complete(
                "You generate setup JSON only.",
                prompt,
                temperature=0.6,
                max_tokens=1800,
            )
            if not response:
                return base
            parsed = self._parse_json_lenient(self._extract_json(response) or response)
            if not isinstance(parsed, dict) or not parsed:
                return base
            out = dict(base)
            out.update({k: v for k, v in parsed.items() if k in out})
            return out
        except Exception:
            return base

    async def _setup_generate_storyline_variants(
        self,
        campaign: Campaign,
        setup_data: dict[str, Any],
        user_guidance: str | None = None,
    ) -> str:
        is_known = bool(setup_data.get("is_known_work", False))
        raw_name = str(setup_data.get("raw_name") or campaign.name).strip()
        work_desc = str(setup_data.get("work_description") or "").strip()
        work_type = str(setup_data.get("work_type") or "work").strip()
        imdb_results = setup_data.get("imdb_results", [])
        if not isinstance(imdb_results, list):
            imdb_results = []
        attachment_summary = str(setup_data.get("attachment_summary") or "").strip()

        variants: list[dict[str, Any]] = []
        if self._completion_port is not None:
            system_prompt = (
                "You are a creative game designer who builds interactive text-adventure campaigns.\n"
                "Return ONLY valid JSON with key 'variants' containing 2-3 objects.\n"
                "Each object must include: id, title, summary, main_character, essential_npcs, chapter_outline.\n"
                "No markdown, no code fences."
            )
            imdb_context = ""
            if imdb_results:
                imdb_context = f"\nIMDB reference data:\n{self._format_imdb_results(imdb_results)}\n"
            attachment_context = ""
            if attachment_summary:
                attachment_context = (
                    "\nDetailed source material summary:\n"
                    f"{attachment_summary}\n"
                    "Use this summary to create accurate, faithful variants.\n"
                )
            guidance_context = ""
            if user_guidance:
                guidance_context = (
                    "\nThe user gave this direction for the variants:\n"
                    f"{user_guidance}\n"
                    "Follow these instructions closely when designing the variants.\n"
                )

            if is_known:
                user_prompt = (
                    f"Generate 2-3 storyline variants for an interactive text-adventure campaign "
                    f"based on the {work_type}: '{raw_name}'.\n"
                    f"Description: {work_desc}\n"
                    f"{imdb_context}"
                    f"{attachment_context}"
                    f"{guidance_context}"
                    "Use actual characters, locations, and plot points from the source work."
                )
            else:
                user_prompt = (
                    f"Generate 2-3 storyline variants for an original text-adventure campaign "
                    f"called '{raw_name}'.\n"
                    f"{attachment_context}"
                    f"{guidance_context}"
                    "Each variant should have a different tone, central conflict, or protagonist archetype. "
                    "Be creative and specific with character names and chapter titles."
                )

            result: dict[str, Any] = {}
            self._zork_log(
                f"SETUP VARIANT GENERATION campaign={campaign.id}",
                f"is_known={is_known} raw_name={raw_name!r} work_desc={work_desc!r}\n"
                f"--- SYSTEM ---\n{system_prompt}\n--- USER ---\n{user_prompt}",
            )
            for attempt in range(2):
                try:
                    cur_user = user_prompt
                    if attempt == 1:
                        cur_user = (
                            f"Generate 2-3 adventure storyline variants for an adult text-adventure "
                            f"game inspired by '{raw_name}'. All characters are adults. "
                            "Focus on the setting, survival themes, and exploration.\n"
                            f"{imdb_context}"
                        )
                        self._zork_log(f"SETUP VARIANT RETRY campaign={campaign.id}", cur_user)
                    response = await self._completion_port.complete(
                        system_prompt,
                        cur_user,
                        temperature=0.8,
                        max_tokens=3000,
                    )
                    self._zork_log("SETUP VARIANT RAW RESPONSE", response or "(empty)")
                    response = self._clean_response(response or "{}")
                    json_text = self._extract_json(response)
                    result = self._parse_json_lenient(json_text) if json_text else {}
                    if isinstance(result.get("variants"), list) and result["variants"]:
                        break
                except Exception as exc:
                    self._logger.warning(
                        "Storyline variant generation failed (attempt %s): %s",
                        attempt,
                        exc,
                    )
                    self._zork_log("SETUP VARIANT GENERATION FAILED", str(exc))
                    result = {}
            raw_variants = result.get("variants", [])
            if isinstance(raw_variants, list):
                for idx, row in enumerate(raw_variants[:3], start=1):
                    if not isinstance(row, dict):
                        continue
                    summary = str(row.get("summary") or "").strip()
                    if not summary:
                        continue
                    variants.append(
                        {
                            "id": str(row.get("id") or f"variant-{idx}"),
                            "title": str(row.get("title") or f"Variant {idx}").strip(),
                            "summary": self._trim_text(summary, 300),
                            "main_character": str(row.get("main_character") or "The Protagonist").strip(),
                            "essential_npcs": row.get("essential_npcs", []),
                            "chapter_outline": row.get("chapter_outline", []),
                        }
                    )

        if not variants:
            self._zork_log(
                "SETUP VARIANT FALLBACK",
                f"result keys={list(result.keys()) if isinstance(result, dict) else 'not-dict'}",
            )
            top_imdb = imdb_results[0] if imdb_results else {}
            cast = top_imdb.get("cast", []) if isinstance(top_imdb, dict) else []
            main_char = cast[0] if isinstance(cast, list) and cast else "The Protagonist"
            npcs = cast[1:5] if isinstance(cast, list) and len(cast) > 1 else []
            synopsis = str(
                (
                    (top_imdb.get("synopsis") if isinstance(top_imdb, dict) else "")
                    or (top_imdb.get("description") if isinstance(top_imdb, dict) else "")
                    or work_desc
                    or ""
                )
            ).strip()
            variants = [
                {
                    "id": "variant-1",
                    "title": f"{raw_name}: Faithful Retelling",
                    "summary": synopsis[:300] if synopsis else f"An interactive adventure set in the world of {raw_name}.",
                    "main_character": main_char,
                    "essential_npcs": npcs,
                    "chapter_outline": [
                        {"title": "Chapter 1: The Beginning", "summary": "The adventure begins."},
                        {"title": "Chapter 2: The Challenge", "summary": "Obstacles arise."},
                        {"title": "Chapter 3: The Resolution", "summary": "The story concludes."},
                    ],
                }
            ]

        setup_data["storyline_variants"] = variants
        lines = ["**Choose a storyline variant:**\n"]
        for idx, variant in enumerate(variants, start=1):
            lines.append(f"**{idx}. {variant.get('title', 'Untitled')}**")
            lines.append(f"_{variant.get('summary', '')}_")
            lines.append(f"Main character: {variant.get('main_character', 'TBD')}")
            npcs = variant.get("essential_npcs", [])
            if isinstance(npcs, list) and npcs:
                lines.append(f"Key NPCs: {', '.join([str(n) for n in npcs])}")
            chapters = variant.get("chapter_outline", [])
            if isinstance(chapters, list) and chapters:
                titles = [str(ch.get("title", "?")) for ch in chapters if isinstance(ch, dict)]
                if titles:
                    lines.append(f"Chapters: {' → '.join(titles)}")
            lines.append("")
        lines.append(
            "Reply with **1**, **2**, or **3** to pick your storyline, "
            "or **retry: <guidance>** to regenerate (e.g. `retry: make it darker`)."
        )
        return "\n".join(lines)

    async def _setup_handle_storyline_pick(
        self,
        campaign: Campaign,
        state: dict[str, Any],
        setup_data: dict[str, Any],
        message_text: str,
        actor_id: str,
    ) -> str:
        choice = (message_text or "").strip()
        variants = setup_data.get("storyline_variants", [])
        if not isinstance(variants, list):
            variants = []

        if choice.lower().startswith("retry"):
            guidance = choice.split(":", 1)[1].strip() if ":" in choice else ""
            state["setup_data"] = setup_data
            return await self._setup_generate_storyline_variants(
                campaign,
                setup_data,
                user_guidance=guidance or None,
            )

        try:
            idx = int(choice) - 1
        except (ValueError, TypeError):
            return (
                f"Please reply with a number (1-{len(variants)}), "
                "or **retry: <guidance>** to regenerate."
            )
        if idx < 0 or idx >= len(variants):
            return f"Please reply with a number between 1 and {len(variants)}."

        chosen = variants[idx]
        setup_data["chosen_variant_id"] = chosen.get("id", f"variant-{idx + 1}")
        if bool(setup_data.get("is_known_work", False)):
            state["setup_phase"] = "finalize"
            state["setup_data"] = setup_data
            return await self._setup_finalize(campaign, state, setup_data, user_id=actor_id)

        state["setup_phase"] = "novel_questions"
        state["setup_data"] = setup_data
        return (
            "A few more questions for your original campaign:\n\n"
            "1. **On-rails mode?** Should the story strictly follow the chapter outline, "
            "or allow freeform exploration? (reply **on-rails** or **freeform**)\n"
        )

    async def _setup_handle_novel_questions(
        self,
        campaign: Campaign,
        state: dict[str, Any],
        setup_data: dict[str, Any],
        message_text: str,
        actor_id: str,
    ) -> str:
        answer = (message_text or "").strip().lower()
        prefs = setup_data.get("novel_preferences", {})
        if not isinstance(prefs, dict):
            prefs = {}
        if answer in ("on-rails", "onrails", "on rails", "rails", "strict"):
            prefs["on_rails"] = True
        else:
            prefs["on_rails"] = False
        setup_data["novel_preferences"] = prefs
        state["setup_phase"] = "finalize"
        state["setup_data"] = setup_data
        return await self._setup_finalize(campaign, state, setup_data, user_id=actor_id)

    @staticmethod
    def _is_explicit_setup_no(message_text: str) -> tuple[bool, str]:
        raw = (message_text or "").strip()
        lowered = raw.lower()
        if lowered in ("no", "n", "nope", "nah"):
            return True, ""
        if lowered.startswith(("no,", "no.", "no:", "no;", "no!", "no-", "nope ", "nah ")):
            guidance = re.sub(r"^\s*(?:no|nope|nah|n)\b[\s,.:;!\-]*", "", raw, flags=re.IGNORECASE).strip()
            return True, guidance
        if lowered.startswith("no "):
            tail = lowered[3:].lstrip()
            if re.match(r"^(?:i|we|this|that|it|rather|prefer|want|novel|original|custom|homebrew)\b", tail):
                guidance = re.sub(r"^\s*(?:no|nope|nah|n)\b[\s,.:;!\-]*", "", raw, flags=re.IGNORECASE).strip()
                return True, guidance
        return False, ""

    @staticmethod
    def _looks_like_novel_intent(message_text: str) -> bool:
        lowered = (message_text or "").strip().lower()
        if not lowered:
            return False
        markers = (
            "my own",
            "original",
            "custom",
            "homebrew",
            "from scratch",
            "made up",
        )
        if any(marker in lowered for marker in markers):
            return True
        return bool(
            re.search(
                r"\b(i(?:'d| would)? rather|i want|let'?s|make|do)\b.*\b(novel|original|custom|homebrew)\b",
                lowered,
            )
        )

    async def _setup_handle_classify_confirm(
        self,
        campaign: Campaign,
        state: dict[str, Any],
        setup_data: dict[str, Any],
        message_text: str,
        attachments: list[Any] | None = None,
    ) -> str:
        raw_answer = (message_text or "").strip()
        answer = raw_answer.lower()
        user_guidance: str | None = None
        explicit_no, no_guidance = self._is_explicit_setup_no(raw_answer)
        novel_intent = self._looks_like_novel_intent(raw_answer)
        if answer in ("yes", "y", "correct", "yep", "yeah"):
            confirmed = str(setup_data.get("raw_name") or "").lower()
            old_results = setup_data.get("imdb_results", [])
            if isinstance(old_results, list) and old_results and confirmed:
                best = None
                for row in old_results:
                    title = str(row.get("title") or "").lower() if isinstance(row, dict) else ""
                    if title in confirmed or confirmed in title:
                        best = row
                        break
                setup_data["imdb_results"] = [best] if best else [old_results[0]]
            if setup_data.get("imdb_results"):
                setup_data["imdb_results"] = self._imdb_enrich_results(setup_data["imdb_results"])
        elif explicit_no or answer in ("no", "n", "nope") or novel_intent:
            setup_data["is_known_work"] = False
            setup_data["work_type"] = None
            setup_data["imdb_results"] = []
            if explicit_no and no_guidance:
                user_guidance = no_guidance
                setup_data["work_description"] = no_guidance
            elif novel_intent:
                user_guidance = raw_answer
                setup_data["work_description"] = raw_answer
            else:
                setup_data["work_description"] = ""
        else:
            imdb_results = self._imdb_search(answer, max_results=3)
            result = {}
            if self._completion_port is not None:
                imdb_context = ""
                if imdb_results:
                    imdb_context = (
                        f"\nIMDB search results for '{answer}':\n"
                        f"{self._format_imdb_results(imdb_results)}\n"
                        "Use these results to help identify the work.\n"
                    )
                try:
                    response = await self._completion_port.complete(
                        "Return JSON only: is_known_work, work_type, work_description, suggested_title.",
                        (
                            f"The user clarified their campaign: '{answer}'.\n"
                            f"Original input was: '{setup_data.get('raw_name', '')}'.\n"
                            f"{imdb_context}"
                            "Classify whether this is a known published work."
                        ),
                        temperature=0.3,
                        max_tokens=300,
                    )
                    response = self._clean_response(response or "{}")
                    json_text = self._extract_json(response)
                    result = self._parse_json_lenient(json_text) if json_text else {}
                except Exception:
                    result = {}
            setup_data["is_known_work"] = bool(result.get("is_known_work", False))
            setup_data["work_type"] = result.get("work_type")
            setup_data["work_description"] = result.get("work_description") or ""
            setup_data["raw_name"] = result.get("suggested_title") or answer.strip()

            if not setup_data["is_known_work"] and imdb_results and not novel_intent:
                top = imdb_results[0]
                setup_data["is_known_work"] = True
                setup_data["raw_name"] = top.get("title") or setup_data["raw_name"]
                setup_data["work_type"] = (str(top.get("type") or "").lower().replace(" ", "_")) or "other"
                setup_data["work_description"] = str(top.get("description") or setup_data["work_description"] or "")
            confirmed = str(setup_data.get("raw_name") or "").lower()
            if imdb_results and confirmed:
                best = None
                for row in imdb_results:
                    title = str(row.get("title") or "").lower()
                    if title in confirmed or confirmed in title:
                        best = row
                        break
                setup_data["imdb_results"] = [best] if best else [imdb_results[0]]
            else:
                setup_data["imdb_results"] = imdb_results
            if setup_data.get("imdb_results"):
                setup_data["imdb_results"] = self._imdb_enrich_results(setup_data["imdb_results"])
                top = setup_data["imdb_results"][0]
                if top.get("description") and not setup_data.get("work_description"):
                    setup_data["work_description"] = top["description"]

        if attachments:
            extracted = await extract_attachment_text(attachments)
            if isinstance(extracted, str) and extracted.startswith("ERROR:"):
                return extracted
            if extracted:
                summary = await self._summarise_long_text(extracted)
                if summary:
                    setup_data["attachment_summary"] = summary

        variants_msg = await self._setup_generate_storyline_variants(
            campaign,
            setup_data,
            user_guidance=user_guidance,
        )
        state["setup_phase"] = "storyline_pick"
        state["setup_data"] = setup_data
        return variants_msg

    async def _setup_finalize(
        self,
        campaign: Campaign,
        state: dict[str, Any],
        setup_data: dict[str, Any],
        *,
        user_id: str | None = None,
        db_session=None,
    ) -> str:
        variants = setup_data.get("storyline_variants", [])
        if not isinstance(variants, list):
            variants = []
        chosen_id = str(setup_data.get("chosen_variant_id") or "variant-1")
        chosen = None
        for variant in variants:
            if isinstance(variant, dict) and str(variant.get("id")) == chosen_id:
                chosen = variant
                break
        if chosen is None and variants:
            chosen = variants[0]
        if chosen is None:
            chosen = {
                "title": "Adventure",
                "summary": "",
                "main_character": "The Protagonist",
                "essential_npcs": [],
                "chapter_outline": [],
            }

        is_known = bool(setup_data.get("is_known_work", False))
        raw_name = str(setup_data.get("raw_name") or "unknown")
        novel_prefs = setup_data.get("novel_preferences", {})
        if not isinstance(novel_prefs, dict):
            novel_prefs = {}
        on_rails = True if is_known else bool(novel_prefs.get("on_rails", False))

        world: dict[str, Any] = {}
        if self._completion_port is not None:
            imdb_results = setup_data.get("imdb_results", [])
            if not isinstance(imdb_results, list):
                imdb_results = []
            imdb_context = ""
            if imdb_results:
                imdb_context = f"\nIMDB reference data:\n{self._format_imdb_results(imdb_results)}\n"
            attachment_summary = str(setup_data.get("attachment_summary") or "").strip()
            attachment_context = ""
            if attachment_summary:
                attachment_context = (
                    "\nDetailed source material:\n"
                    f"{attachment_summary}\n"
                    "Use this to create an accurate world with faithful characters and locations.\n"
                )
            finalize_system = (
                "You are a world-builder for interactive text-adventure campaigns.\n"
                "Return ONLY valid JSON with keys: characters, story_outline, summary, "
                "start_room, landmarks, setting, tone, default_persona, opening_narration.\n"
                "No markdown, no code fences."
            )
            finalize_user = (
                f"Build the complete world for: '{raw_name}'\n"
                f"Known work: {is_known}\n"
                f"Description: {setup_data.get('work_description', '')}\n"
                f"{imdb_context}"
                f"{attachment_context}"
                f"Chosen storyline:\n{self._dump_json(chosen)}\n\n"
                "Expand chapter outline into full chapters with 2-4 scenes each."
            )
            for attempt in range(2):
                try:
                    cur_user = finalize_user
                    if attempt == 1:
                        cur_user = (
                            f"Build the complete world for an adult text-adventure game inspired by '{raw_name}'.\n"
                            f"{imdb_context}"
                            f"Chosen storyline:\n{self._dump_json(chosen)}"
                        )
                    response = await self._completion_port.complete(
                        finalize_system,
                        cur_user,
                        temperature=0.7,
                        max_tokens=4000,
                    )
                    response = self._clean_response(response or "{}")
                    json_text = self._extract_json(response)
                    world = self._parse_json_lenient(json_text) if json_text else {}
                    if world and (world.get("characters") or world.get("start_room")):
                        break
                except Exception:
                    world = {}

        if not world:
            world = {
                "characters": {},
                "story_outline": {"chapters": []},
                "summary": str(chosen.get("summary") or campaign.summary or ""),
                "start_room": {
                    "room_title": "Starting Point",
                    "room_summary": "The first room of your adventure.",
                    "room_description": "A world stirs as your adventure begins.",
                    "exits": ["look around", "move forward"],
                    "location": "start",
                },
                "landmarks": [],
                "setting": raw_name,
                "tone": "adventurous",
                "default_persona": setup_data.get("default_persona") or self.DEFAULT_CAMPAIGN_PERSONA,
                "opening_narration": "The world sharpens around you as the adventure begins.",
            }

        characters = world.get("characters", {})
        if isinstance(characters, dict) and characters:
            campaign.characters_json = self._dump_json(characters)

        story_outline = world.get("story_outline", {})
        start_room = world.get("start_room", {})
        landmarks = world.get("landmarks", [])
        setting = world.get("setting", "")
        tone = world.get("tone", "")
        default_persona = world.get("default_persona", "")
        summary = world.get("summary", "")
        opening = world.get("opening_narration", "")

        if summary:
            campaign.summary = self._trim_text(str(summary), self.MAX_SUMMARY_CHARS)
        state.pop("setup_phase", None)
        state.pop("setup_data", None)
        if isinstance(story_outline, dict):
            state["story_outline"] = story_outline
            state["current_chapter"] = 0
            state["current_scene"] = 0
        if isinstance(start_room, dict):
            state["start_room"] = start_room
        if isinstance(landmarks, list):
            state["landmarks"] = landmarks
        if setting:
            state["setting"] = setting
        if tone:
            state["tone"] = tone
        if default_persona:
            state["default_persona"] = self._trim_text(str(default_persona), self.MAX_PERSONA_PROMPT_CHARS)
        state["on_rails"] = on_rails

        if opening:
            room_title = start_room.get("room_title", "") if isinstance(start_room, dict) else ""
            narration = f"{room_title}\n{opening}" if room_title else str(opening)
            exits = start_room.get("exits") if isinstance(start_room, dict) else None
            if isinstance(exits, list) and exits:
                labels = []
                for exit_entry in exits:
                    if isinstance(exit_entry, dict):
                        labels.append(exit_entry.get("direction") or exit_entry.get("name") or str(exit_entry))
                    else:
                        labels.append(str(exit_entry))
                narration += f"\nExits: {', '.join(labels)}"
            campaign.last_narration = self._trim_text(narration, self.MAX_NARRATION_CHARS)

        active_session = db_session
        owns_session = False
        if active_session is None:
            active_session = self._session_factory()
            owns_session = True
        try:
            if user_id is not None:
                player = (
                    active_session.query(Player)
                    .filter(Player.campaign_id == campaign.id)
                    .filter(Player.actor_id == str(user_id))
                    .first()
                )
                if player is None:
                    player = Player(
                        campaign_id=campaign.id,
                        actor_id=str(user_id),
                        state_json="{}",
                        attributes_json="{}",
                    )
                    active_session.add(player)
                    active_session.flush()
                player_state = parse_json_dict(player.state_json)
                main_char = chosen.get("main_character", "")
                if main_char and not player_state.get("character_name"):
                    player_state["character_name"] = main_char
                if default_persona and not player_state.get("persona"):
                    player_state["persona"] = self._trim_text(str(default_persona), self.MAX_PERSONA_PROMPT_CHARS)
                if isinstance(start_room, dict):
                    for key in ("room_title", "room_summary", "room_description", "exits", "location"):
                        value = start_room.get(key)
                        if value is not None:
                            player_state[key] = value
                player.state_json = self._dump_json(player_state)
                player.updated_at = datetime.utcnow()
            if owns_session:
                active_session.commit()
        finally:
            if owns_session:
                active_session.close()

        rails_label = "**On-Rails**" if on_rails else "**Freeform**"
        char_count = len(characters) if isinstance(characters, dict) else 0
        chapter_count = len(story_outline.get("chapters", [])) if isinstance(story_outline, dict) else 0
        result_msg = (
            f"Campaign **{raw_name}** is ready! ({rails_label} mode)\n"
            f"Characters: {char_count} | Chapters: {chapter_count}\n\n"
        )
        if campaign.last_narration:
            result_msg += campaign.last_narration
        return result_msg

    def _extract_room_image_url(self, room_image_entry) -> Optional[str]:
        if isinstance(room_image_entry, str):
            value = room_image_entry.strip()
            return value if value else None
        if isinstance(room_image_entry, dict):
            raw = room_image_entry.get("url")
            if isinstance(raw, str):
                value = raw.strip()
                return value if value else None
        return None

    def _is_image_url_404(self, image_url: str) -> bool:
        if not isinstance(image_url, str):
            return False
        url = image_url.strip()
        if not url:
            return False
        try:
            request = urllib_request.Request(url, method="HEAD")
            with urllib_request.urlopen(request, timeout=6) as response:  # noqa: S310
                code = int(getattr(response, "status", 200))
                if code == 404:
                    return True
                if code in (405, 501):
                    get_request = urllib_request.Request(url, method="GET")
                    with urllib_request.urlopen(get_request, timeout=8) as get_response:  # noqa: S310
                        return int(getattr(get_response, "status", 200)) == 404
                return False
        except urllib_error.HTTPError as exc:
            return int(getattr(exc, "code", 0)) == 404
        except Exception:
            return False

    def get_room_scene_image_url(
        self,
        campaign: Campaign | None,
        room_key: str,
    ) -> Optional[str]:
        if campaign is None or not room_key:
            return None
        campaign_state = self.get_campaign_state(campaign)
        room_images = campaign_state.get(self.ROOM_IMAGE_STATE_KEY, {})
        if not isinstance(room_images, dict):
            return None
        return self._extract_room_image_url(room_images.get(room_key))

    def clear_room_scene_image_url(
        self,
        campaign: Campaign | None,
        room_key: str,
    ) -> bool:
        if campaign is None or not room_key:
            return False
        campaign_state = self.get_campaign_state(campaign)
        room_images = campaign_state.get(self.ROOM_IMAGE_STATE_KEY, {})
        if not isinstance(room_images, dict):
            return False
        if room_key not in room_images:
            return False
        room_images.pop(room_key, None)
        campaign_state[self.ROOM_IMAGE_STATE_KEY] = room_images
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is None:
                return False
            row.state_json = self._dump_json(campaign_state)
            row.updated_at = datetime.utcnow()
            session.commit()
            campaign.state_json = row.state_json
            campaign.updated_at = row.updated_at
        return True

    def record_room_scene_image_url_for_channel(
        self,
        guild_id: int | str,
        channel_id: int | str,
        room_key: str,
        image_url: str,
        campaign_id: Optional[str | int] = None,
        scene_prompt: Optional[str] = None,
        overwrite: bool = False,
    ) -> bool:
        guild = str(guild_id)
        channel = str(channel_id)
        if not room_key:
            room_key = "unknown-room"
        if not isinstance(image_url, str) or not image_url.strip():
            return False

        with self._session_factory() as session:
            effective_campaign_id: str | None = str(campaign_id) if campaign_id is not None else None
            if effective_campaign_id is None:
                row = (
                    session.query(GameSession)
                    .filter(GameSession.surface_guild_id == guild)
                    .filter(
                        or_(
                            GameSession.surface_channel_id == channel,
                            GameSession.surface_thread_id == channel,
                            GameSession.surface_key == f"discord:{guild}:{channel}",
                        )
                    )
                    .first()
                )
                if row is None:
                    return False
                meta = self._load_session_metadata(row)
                active = meta.get("active_campaign_id")
                if isinstance(active, str) and active:
                    effective_campaign_id = active
                else:
                    effective_campaign_id = row.campaign_id

            campaign = session.get(Campaign, effective_campaign_id)
            if campaign is None:
                return False
            campaign_state = parse_json_dict(campaign.state_json)
            room_images = campaign_state.get(self.ROOM_IMAGE_STATE_KEY, {})
            if not isinstance(room_images, dict):
                room_images = {}
            if (not overwrite) and room_key in room_images:
                return False
            room_images[room_key] = {
                "url": image_url.strip(),
                "updated": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "prompt": self._trim_text(scene_prompt or "", 600),
            }
            campaign_state[self.ROOM_IMAGE_STATE_KEY] = room_images
            campaign.state_json = self._dump_json(campaign_state)
            campaign.updated_at = datetime.utcnow()
            session.commit()
            return True

    def record_pending_avatar_image_for_campaign(
        self,
        campaign_id: str | int,
        user_id: str | int,
        image_url: str,
        avatar_prompt: Optional[str] = None,
    ) -> bool:
        if not campaign_id or not user_id:
            return False
        if not isinstance(image_url, str) or not image_url.strip():
            return False
        with self._session_factory() as session:
            player = (
                session.query(Player)
                .filter(Player.campaign_id == str(campaign_id))
                .filter(Player.actor_id == str(user_id))
                .first()
            )
            if player is None:
                return False
            player_state = parse_json_dict(player.state_json)
            player_state["pending_avatar_url"] = image_url.strip()
            if isinstance(avatar_prompt, str) and avatar_prompt.strip():
                player_state["pending_avatar_prompt"] = self._trim_text(avatar_prompt.strip(), 500)
            player_state["pending_avatar_generated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            player.state_json = self._dump_json(player_state)
            player.updated_at = datetime.utcnow()
            session.commit()
            return True

    def accept_pending_avatar(self, campaign_id: str | int, user_id: str | int) -> tuple[bool, str]:
        with self._session_factory() as session:
            player = (
                session.query(Player)
                .filter(Player.campaign_id == str(campaign_id))
                .filter(Player.actor_id == str(user_id))
                .first()
            )
            if player is None:
                return False, "Player not found."
            player_state = parse_json_dict(player.state_json)
            pending_url = player_state.get("pending_avatar_url")
            if not isinstance(pending_url, str) or not pending_url.strip():
                return False, "No pending avatar to accept."
            player_state["avatar_url"] = pending_url.strip()
            player_state.pop("pending_avatar_url", None)
            player_state.pop("pending_avatar_prompt", None)
            player_state.pop("pending_avatar_generated_at", None)
            player.state_json = self._dump_json(player_state)
            player.updated_at = datetime.utcnow()
            session.commit()
            return True, f"Avatar accepted: {player_state.get('avatar_url')}"

    def decline_pending_avatar(self, campaign_id: str | int, user_id: str | int) -> tuple[bool, str]:
        with self._session_factory() as session:
            player = (
                session.query(Player)
                .filter(Player.campaign_id == str(campaign_id))
                .filter(Player.actor_id == str(user_id))
                .first()
            )
            if player is None:
                return False, "Player not found."
            player_state = parse_json_dict(player.state_json)
            had_pending = bool(player_state.get("pending_avatar_url"))
            player_state.pop("pending_avatar_url", None)
            player_state.pop("pending_avatar_prompt", None)
            player_state.pop("pending_avatar_generated_at", None)
            player.state_json = self._dump_json(player_state)
            player.updated_at = datetime.utcnow()
            session.commit()
            if had_pending:
                return True, "Pending avatar discarded."
            return False, "No pending avatar to discard."

    def _normalize_match_text(self, value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        return re.sub(r"\s+", " ", text)

    def _room_key_from_player_state(self, player_state: Dict[str, object]) -> str:
        if not isinstance(player_state, dict):
            return "unknown-room"
        for key in ("room_id", "location", "room_title", "room_summary"):
            raw = player_state.get(key)
            normalized = self._normalize_match_text(raw)
            if normalized:
                return normalized[:120]
        return "unknown-room"

    def _same_scene(self, actor_state: Dict[str, object], other_state: Dict[str, object]) -> bool:
        if not isinstance(actor_state, dict) or not isinstance(other_state, dict):
            return False
        actor_room_id = self._normalize_match_text(actor_state.get("room_id"))
        other_room_id = self._normalize_match_text(other_state.get("room_id"))
        if actor_room_id and other_room_id:
            return actor_room_id == other_room_id

        actor_location = self._normalize_match_text(actor_state.get("location"))
        other_location = self._normalize_match_text(other_state.get("location"))
        actor_title = self._normalize_match_text(actor_state.get("room_title"))
        other_title = self._normalize_match_text(other_state.get("room_title"))
        actor_summary = self._normalize_match_text(actor_state.get("room_summary"))
        other_summary = self._normalize_match_text(other_state.get("room_summary"))

        if actor_location and other_location and actor_location == other_location:
            title_known = bool(actor_title and other_title)
            summary_known = bool(actor_summary and other_summary)
            title_match = title_known and actor_title == other_title
            summary_match = summary_known and actor_summary == other_summary
            if title_known or summary_known:
                return title_match or summary_match
            return True

        if (not actor_location and not other_location) and actor_title and other_title:
            if actor_title != other_title:
                return False
            if actor_summary and other_summary:
                return actor_summary == other_summary
            return False
        return False

    def _build_attribute_cues(self, attributes: Dict[str, int]) -> List[str]:
        if not isinstance(attributes, dict):
            return []
        ranked = [(str(key), value) for key, value in attributes.items() if isinstance(value, int)]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [f"{key} {value}" for key, value in ranked[:2]]

    def _build_party_snapshot_for_prompt(
        self,
        campaign: Campaign,
        actor: Player,
        actor_state: Dict[str, object],
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        with self._session_factory() as session:
            players = (
                session.query(Player)
                .filter(Player.campaign_id == campaign.id)
                .order_by(Player.last_active_at.desc())
                .all()
            )
            for entry in players:
                state = parse_json_dict(entry.state_json)
                if entry.actor_id != actor.actor_id and not self._same_scene(actor_state, state):
                    continue
                fallback_name = f"Adventurer-{entry.actor_id[-4:]}" if entry.actor_id else "Adventurer"
                display_name = str(state.get("character_name") or fallback_name).strip()
                persona = str(state.get("persona") or "").strip()
                if persona:
                    persona = self._trim_text(persona, self.MAX_PERSONA_PROMPT_CHARS)
                    persona = " ".join(persona.split()[:18])
                attributes = self.get_player_attributes(entry)
                attribute_cues = self._build_attribute_cues(attributes)
                visible_items = []
                if entry.actor_id == actor.actor_id:
                    visible_items = self._normalize_inventory_items(state.get("inventory"))[:3]
                out.append(
                    {
                        "discord_mention": f"<@{entry.actor_id}>",
                        "name": display_name,
                        "is_actor": entry.actor_id == actor.actor_id,
                        "level": entry.level,
                        "persona": persona,
                        "attribute_cues": attribute_cues,
                        "location": state.get("location"),
                        "room_title": state.get("room_title"),
                        "visible_items": visible_items,
                    }
                )
                if len(out) >= self.MAX_PARTY_CONTEXT_PLAYERS:
                    break
        return out

    def _build_scene_avatar_references(
        self,
        campaign: Campaign | None,
        actor: Player | None,
        actor_state: Dict[str, object],
    ) -> List[Dict[str, object]]:
        if campaign is None or actor is None:
            return []
        refs: List[Dict[str, object]] = []
        seen_urls: set[str] = set()
        with self._session_factory() as session:
            players = (
                session.query(Player)
                .filter(Player.campaign_id == campaign.id)
                .order_by(Player.last_active_at.desc())
                .all()
            )
            for entry in players:
                state = parse_json_dict(entry.state_json)
                if entry.actor_id != actor.actor_id and not self._same_scene(actor_state, state):
                    continue
                avatar_url = state.get("avatar_url")
                if not isinstance(avatar_url, str):
                    continue
                avatar_url = avatar_url.strip()
                if not avatar_url or avatar_url in seen_urls:
                    continue
                if self._is_image_url_404(avatar_url):
                    continue
                seen_urls.add(avatar_url)
                suffix = entry.actor_id[-4:] if entry.actor_id else "anon"
                identity = str(state.get("character_name") or f"Adventurer-{suffix}").strip()
                refs.append(
                    {
                        "user_id": entry.actor_id,
                        "name": identity,
                        "url": avatar_url,
                        "is_actor": entry.actor_id == actor.actor_id,
                    }
                )
                if len(refs) >= self.MAX_SCENE_REFERENCE_IMAGES - 1:
                    break
        return refs

    def _compose_scene_prompt_with_references(
        self,
        scene_prompt: str,
        has_room_reference: bool,
        avatar_refs: List[Dict[str, object]],
    ) -> str:
        prompt = (scene_prompt or "").strip()
        if not prompt:
            return ""
        directives: List[str] = []
        image_index = 1
        if has_room_reference:
            directives.append(
                f"Use the environment from image {image_index} as the persistent room layout and lighting anchor."
            )
            image_index += 1
        for ref in avatar_refs:
            name = str(ref.get("name") or "character").strip()
            directives.append(f"Render {name} to match the person in image {image_index}.")
            image_index += 1
        if directives:
            prompt = f"{' '.join(directives)} {prompt}"
        prompt = re.sub(r"\s+", " ", prompt).strip()
        return self._trim_text(prompt, self.MAX_SCENE_PROMPT_CHARS)

    def _compose_empty_room_scene_prompt(
        self,
        scene_prompt: str,
        player_state: Dict[str, object],
    ) -> str:
        room_title = str(player_state.get("room_title") or "").strip()
        location = str(player_state.get("location") or "").strip()
        room_summary = str(player_state.get("room_summary") or "").strip()
        room_description = str(player_state.get("room_description") or "").strip()

        room_label = room_title or location or "the current room"
        detail_text = room_description or room_summary or (scene_prompt or "").strip()
        prompt = (
            f"Environmental establishing shot of {room_label}. "
            f"{detail_text} "
            "No characters, no people, no creatures, no animals, no humanoids. "
            "Focus on architecture, props, lighting, and atmosphere only."
        )
        prompt = re.sub(r"\s+", " ", prompt).strip()
        return self._trim_text(prompt, self.MAX_SCENE_PROMPT_CHARS)

    def _missing_scene_names(self, scene_prompt: str, party_snapshot: List[Dict[str, object]]) -> List[str]:
        prompt_l = (scene_prompt or "").lower()
        missing: List[str] = []
        for entry in party_snapshot:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            name_l = name.lower()
            name_pattern = re.escape(name_l).replace(r"\ ", r"\s+")
            if not re.search(rf"(?<![a-z0-9]){name_pattern}(?![a-z0-9])", prompt_l):
                missing.append(name)
        return missing

    def _enrich_scene_image_prompt(
        self,
        scene_prompt: str,
        player_state: Dict[str, object],
        party_snapshot: List[Dict[str, object]],
    ) -> str:
        if not isinstance(scene_prompt, str):
            return ""
        prompt = scene_prompt.strip()
        if not prompt:
            return ""
        pending_prefixes: List[str] = []
        room_bits: List[str] = []
        room_title = str(player_state.get("room_title") or "").strip()
        location = str(player_state.get("location") or "").strip()
        if room_title:
            room_bits.append(room_title)
        if location and self._normalize_match_text(location) != self._normalize_match_text(room_title):
            room_bits.append(location)
        room_clause = ", ".join(room_bits).strip()
        if room_clause and room_clause.lower() not in prompt.lower():
            pending_prefixes.append(f"Location: {room_clause}.")

        missing_names = self._missing_scene_names(prompt, party_snapshot)
        if missing_names:
            cast_fragments: List[str] = []
            for entry in party_snapshot:
                name = str(entry.get("name") or "").strip()
                if not name or name not in missing_names:
                    continue
                tags: List[str] = []
                persona = str(entry.get("persona") or "").strip()
                if persona:
                    tags.append(persona)
                cues = entry.get("attribute_cues") or []
                if cues:
                    tags.append(" / ".join([str(cue) for cue in cues[:2]]))
                items = entry.get("visible_items") or []
                if items:
                    tags.append("carrying " + ", ".join([str(item) for item in items[:2]]))
                cast_fragments.append(f"{name} ({'; '.join(tags)})" if tags else name)
            if cast_fragments:
                pending_prefixes.append(f"Characters: {'; '.join(cast_fragments)}.")

        if pending_prefixes:
            prompt = f"{' '.join(pending_prefixes)} {prompt}".strip()
        prompt = re.sub(r"\s+", " ", prompt).strip()
        if len(prompt) > self.MAX_SCENE_PROMPT_CHARS:
            prompt = prompt[: self.MAX_SCENE_PROMPT_CHARS].strip()
            missing_after_trim = self._missing_scene_names(prompt, party_snapshot)
            if missing_after_trim:
                cast_prefix = f"Characters: {', '.join(missing_after_trim)}. "
                remaining = self.MAX_SCENE_PROMPT_CHARS - len(cast_prefix)
                if remaining > 24:
                    prompt = (cast_prefix + prompt[:remaining]).strip()
                else:
                    prompt = cast_prefix[: self.MAX_SCENE_PROMPT_CHARS].strip()
        return prompt

    def _compose_avatar_prompt(
        self,
        player_state: Dict[str, object],
        requested_prompt: str,
        fallback_name: str,
    ) -> str:
        identity = str(player_state.get("character_name") or fallback_name or "adventurer").strip()
        persona = str(player_state.get("persona") or "").strip()
        prompt_parts = [
            f"Single-character concept portrait of {identity}.",
            requested_prompt.strip(),
            "isolated subject",
            "full body",
            "centered composition",
        ]
        if persona:
            prompt_parts.insert(1, f"Persona/style notes: {persona}.")
        composed = " ".join([part for part in prompt_parts if part])
        composed = re.sub(r"\s+", " ", composed).strip()
        return self._trim_text(composed, 900)

    def _gpu_worker_available(self) -> bool:
        if self._media_port is None:
            return False
        try:
            return bool(self._media_port.gpu_worker_available())
        except Exception:
            return False

    def _build_synthetic_generation_context(self, channel, user_id: str):
        return {
            "channel_id": str(getattr(channel, "id", channel)),
            "user_id": str(user_id),
        }

    async def _enqueue_scene_image(
        self,
        ctx,
        scene_image_prompt: str,
        campaign_id: Optional[str] = None,
        room_key: Optional[str] = None,
    ):
        if not scene_image_prompt:
            return
        if not self._gpu_worker_available():
            return
        if self._media_port is None:
            return

        actor_id = str(getattr(getattr(ctx, "author", None), "id", "") or "")
        channel_id = str(getattr(getattr(ctx, "channel", None), "id", "") or "")
        if not actor_id:
            return

        reference_images: List[str] = []
        avatar_refs: List[Dict[str, object]] = []
        selected_model = self.DEFAULT_SCENE_IMAGE_MODEL
        prompt_for_generation = scene_image_prompt
        should_store_room_image = False
        has_room_reference = False
        player_state_for_prompt: Dict[str, object] = {}

        if campaign_id is not None:
            with self._session_factory() as session:
                campaign = session.get(Campaign, str(campaign_id))
                if campaign is not None:
                    campaign_state = parse_json_dict(campaign.state_json)
                    model_override = campaign_state.get("scene_image_model")
                    if isinstance(model_override, str) and model_override.strip():
                        selected_model = model_override.strip()
                    player = (
                        session.query(Player)
                        .filter(Player.campaign_id == campaign.id)
                        .filter(Player.actor_id == actor_id)
                        .first()
                    )
                    player_state = parse_json_dict(player.state_json) if player is not None else {}
                    player_state_for_prompt = player_state
                    if not room_key:
                        room_key = self._room_key_from_player_state(player_state)
                    if room_key:
                        cached_url = self.get_room_scene_image_url(campaign, room_key)
                        if cached_url and self._is_image_url_404(cached_url):
                            self.clear_room_scene_image_url(campaign, room_key)
                            cached_url = None
                        if cached_url:
                            reference_images.append(cached_url)
                            has_room_reference = True
                        else:
                            should_store_room_image = True
                    if player is not None and not should_store_room_image:
                        avatar_refs = self._build_scene_avatar_references(campaign, player, player_state)
                        for ref in avatar_refs:
                            ref_url = str(ref.get("url") or "").strip()
                            if not ref_url or ref_url in reference_images:
                                continue
                            reference_images.append(ref_url)
                            if len(reference_images) >= self.MAX_SCENE_REFERENCE_IMAGES:
                                break
                    if should_store_room_image:
                        prompt_for_generation = self._compose_empty_room_scene_prompt(
                            scene_image_prompt,
                            player_state=player_state_for_prompt,
                        )
                    else:
                        prompt_for_generation = self._compose_scene_prompt_with_references(
                            scene_image_prompt,
                            has_room_reference=has_room_reference,
                            avatar_refs=avatar_refs[: max(self.MAX_SCENE_REFERENCE_IMAGES - 1, 0)],
                        )

        metadata = {
            "zork_scene": True,
            "zork_store_image": should_store_room_image,
            "zork_seed_room_image": should_store_room_image,
            "zork_scene_prompt": self._trim_text(scene_image_prompt, self.MAX_SCENE_PROMPT_CHARS),
            "zork_campaign_id": str(campaign_id) if campaign_id is not None else None,
            "zork_room_key": room_key,
            "zork_user_id": actor_id,
        }
        try:
            await self._media_port.enqueue_scene_generation(
                actor_id=actor_id,
                prompt=prompt_for_generation,
                model=selected_model,
                reference_images=reference_images if reference_images else None,
                metadata=metadata,
                channel_id=channel_id or None,
            )
        except Exception:
            return

    async def enqueue_scene_composite_from_seed(
        self,
        channel,
        campaign_id: str | int,
        room_key: str,
        user_id: str | int,
        scene_prompt: str,
        base_image_url: str,
    ) -> bool:
        if not self._gpu_worker_available():
            return False
        if not campaign_id or not room_key or not user_id:
            return False
        if not isinstance(scene_prompt, str) or not scene_prompt.strip():
            return False
        if not isinstance(base_image_url, str) or not base_image_url.strip():
            return False
        if self._media_port is None:
            return False

        reference_images: List[str] = [base_image_url.strip()]
        avatar_refs: List[Dict[str, object]] = []
        selected_model = self.DEFAULT_SCENE_IMAGE_MODEL

        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return False
            campaign_state = parse_json_dict(campaign.state_json)
            model_override = campaign_state.get("scene_image_model")
            if isinstance(model_override, str) and model_override.strip():
                selected_model = model_override.strip()
            player = (
                session.query(Player)
                .filter(Player.campaign_id == campaign.id)
                .filter(Player.actor_id == str(user_id))
                .first()
            )
            player_state = parse_json_dict(player.state_json) if player is not None else {}
            if player is not None:
                avatar_refs = self._build_scene_avatar_references(campaign, player, player_state)
                for ref in avatar_refs:
                    ref_url = str(ref.get("url") or "").strip()
                    if not ref_url or ref_url in reference_images:
                        continue
                    reference_images.append(ref_url)
                    if len(reference_images) >= self.MAX_SCENE_REFERENCE_IMAGES:
                        break

        composed_prompt = self._compose_scene_prompt_with_references(
            scene_prompt.strip(),
            has_room_reference=True,
            avatar_refs=avatar_refs[: max(self.MAX_SCENE_REFERENCE_IMAGES - 1, 0)],
        )
        if not composed_prompt:
            return False
        metadata = {
            "zork_scene": True,
            "zork_store_image": False,
            "zork_seed_room_image": False,
            "zork_campaign_id": str(campaign_id),
            "zork_room_key": room_key,
            "zork_user_id": str(user_id),
        }
        channel_id = str(getattr(channel, "id", channel))
        try:
            return await self._media_port.enqueue_scene_generation(
                actor_id=str(user_id),
                prompt=composed_prompt,
                model=selected_model,
                reference_images=reference_images,
                metadata=metadata,
                channel_id=channel_id,
            )
        except Exception:
            return False

    async def enqueue_avatar_generation(
        self,
        ctx,
        campaign: Campaign,
        player: Player,
        requested_prompt: str,
    ) -> tuple[bool, str]:
        if not requested_prompt or not requested_prompt.strip():
            return False, "Avatar prompt cannot be empty."
        if not self._gpu_worker_available():
            return False, "No GPU workers available right now."
        if self._media_port is None:
            return False, "Image generation integration is not configured."

        player_state = self.get_player_state(player)
        fallback_name = getattr(getattr(ctx, "author", None), "display_name", "adventurer")
        composed_prompt = self._compose_avatar_prompt(
            player_state,
            requested_prompt=requested_prompt,
            fallback_name=fallback_name,
        )
        campaign_state = self.get_campaign_state(campaign)
        selected_model = campaign_state.get("avatar_image_model")
        if not isinstance(selected_model, str) or not selected_model.strip():
            selected_model = self.DEFAULT_AVATAR_IMAGE_MODEL

        player_state["pending_avatar_prompt"] = self._trim_text(requested_prompt.strip(), 500)
        player_state.pop("pending_avatar_url", None)
        with self._session_factory() as session:
            row = session.get(Player, player.id)
            if row is None:
                return False, "Player not found."
            row.state_json = self._dump_json(player_state)
            row.updated_at = datetime.utcnow()
            session.commit()
            player.state_json = row.state_json
            player.updated_at = row.updated_at

        metadata = {
            "zork_scene": True,
            "zork_store_avatar": True,
            "zork_campaign_id": campaign.id,
            "zork_avatar_user_id": player.actor_id,
        }
        channel_id = str(getattr(getattr(ctx, "channel", None), "id", "") or "")
        try:
            ok = await self._media_port.enqueue_avatar_generation(
                actor_id=player.actor_id,
                prompt=composed_prompt,
                model=selected_model,
                metadata=metadata,
                channel_id=channel_id or None,
            )
        except Exception as exc:
            return False, f"Failed to queue avatar generation: {exc}"
        if not ok:
            return False, "Failed to queue avatar generation."
        return (
            True,
            "Avatar candidate queued. Use `!zork avatar accept` or `!zork avatar decline` after it arrives.",
        )

    def _compose_character_portrait_prompt(self, name: str, appearance: str) -> str:
        prompt_parts = [
            f"Character portrait of {name}.",
            appearance.strip() if appearance else "",
            "single character",
            "centered composition",
            "detailed fantasy illustration",
        ]
        composed = " ".join([part for part in prompt_parts if part])
        composed = re.sub(r"\s+", " ", composed).strip()
        return self._trim_text(composed, 900)

    async def _enqueue_character_portrait(
        self,
        *,
        campaign_id: str,
        actor_id: str,
        character_slug: str,
        name: str,
        appearance: str,
        channel_id: str | None = None,
    ) -> bool:
        if not appearance or not appearance.strip():
            return False
        if not self._gpu_worker_available():
            return False
        if self._media_port is None:
            return False

        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return False
            campaign_state = parse_json_dict(campaign.state_json)
            selected_model = campaign_state.get("avatar_image_model")
            if not isinstance(selected_model, str) or not selected_model.strip():
                selected_model = self.DEFAULT_AVATAR_IMAGE_MODEL

        composed_prompt = self._compose_character_portrait_prompt(name, appearance)
        metadata = {
            "zork_scene": True,
            "suppress_image_reactions": True,
            "suppress_image_details": True,
            "zork_store_character_portrait": True,
            "zork_campaign_id": campaign_id,
            "zork_character_slug": character_slug,
        }
        try:
            return await self._media_port.enqueue_avatar_generation(
                actor_id=actor_id,
                prompt=composed_prompt,
                model=selected_model,
                metadata=metadata,
                channel_id=channel_id or None,
            )
        except Exception:
            return False

    def record_character_portrait_url(
        self,
        campaign_id: str | int,
        character_slug: str,
        image_url: str,
    ) -> bool:
        if not isinstance(image_url, str) or not image_url.strip():
            return False
        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return False
            characters = parse_json_dict(campaign.characters_json)
            if character_slug not in characters:
                return False
            character = characters.get(character_slug)
            if not isinstance(character, dict):
                return False
            character["image_url"] = image_url.strip()
            campaign.characters_json = self._dump_json(characters)
            campaign.updated_at = datetime.utcnow()
            session.commit()
        return True

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

    def _create_snapshot(self, narrator_turn: Turn, campaign: Campaign) -> Snapshot | None:
        if narrator_turn is None or campaign is None:
            return None
        with self._session_factory() as session:
            existing = session.query(Snapshot).filter(Snapshot.turn_id == narrator_turn.id).first()
            if existing is not None:
                return existing
            players = session.query(Player).filter(Player.campaign_id == campaign.id).all()
            players_data = [
                {
                    "player_id": row.id,
                    "actor_id": row.actor_id,
                    "level": row.level,
                    "xp": row.xp,
                    "attributes_json": row.attributes_json,
                    "state_json": row.state_json,
                }
                for row in players
            ]
            snapshot = Snapshot(
                turn_id=narrator_turn.id,
                campaign_id=campaign.id,
                campaign_state_json=campaign.state_json,
                campaign_characters_json=campaign.characters_json,
                campaign_summary=campaign.summary or "",
                campaign_last_narration=campaign.last_narration,
                players_json=self._dump_json({"players": players_data}),
            )
            session.add(snapshot)
            session.commit()
            return snapshot

    # ------------------------------------------------------------------
    # Turn lifecycle (compat signatures)
    # ------------------------------------------------------------------

    async def begin_turn(
        self,
        campaign_id: str | Any,
        actor_id: str | None = None,
        *,
        command_prefix: str = "!",
    ) -> Tuple[Optional[str], Optional[str]]:
        # Legacy compatibility: begin_turn(ctx, command_prefix="!")
        if self._is_context_like(campaign_id):
            ctx = campaign_id
            resolved_campaign_id, error_text = self._resolve_campaign_for_context(
                ctx,
                command_prefix=command_prefix,
            )
            if error_text is not None:
                return None, error_text
            if resolved_campaign_id is None:
                return None, None
            campaign_id = resolved_campaign_id
            actor_id = str(getattr(getattr(ctx, "author", None), "id", ""))

        campaign_id = str(campaign_id)
        actor_id = str(actor_id or "")
        if not actor_id:
            return None, "Actor not found."
        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return None, "Campaign not found."
        key = (campaign_id, actor_id)
        with self._inflight_turns_lock:
            if key in self._inflight_turns:
                return None, None
            self._inflight_turns.add(key)
            # Claim is ultimately enforced by DB lease in resolve_turn; this keeps
            # classic begin_turn/end_turn call shape for callers.
            self._claims[key] = TurnClaim(campaign_id=campaign_id, actor_id=actor_id)
        return campaign_id, None

    def end_turn(self, campaign_id: str, actor_id: str):
        key = (campaign_id, actor_id)
        with self._inflight_turns_lock:
            self._inflight_turns.discard(key)
            self._claims.pop(key, None)

    def _try_set_inflight_turn(self, campaign_id: str, actor_id: str) -> bool:
        key = (campaign_id, actor_id)
        with self._inflight_turns_lock:
            if key in self._inflight_turns:
                return False
            self._inflight_turns.add(key)
            self._claims[key] = TurnClaim(campaign_id=campaign_id, actor_id=actor_id)
            return True

    def _clear_inflight_turn(self, campaign_id: str, actor_id: str):
        key = (campaign_id, actor_id)
        with self._inflight_turns_lock:
            self._inflight_turns.discard(key)
            self._claims.pop(key, None)

    async def _play_action_with_ids(
        self,
        campaign_id: str,
        actor_id: str,
        action: str,
        session_id: str | None = None,
        manage_claim: bool = True,
    ) -> Optional[str]:
        should_end = False
        pre_inventory_rich: list[dict[str, str]] = []
        pre_character_slugs: set[str] = set()
        if manage_claim:
            cid, error_text = await self.begin_turn(campaign_id, actor_id)
            if error_text is not None:
                return error_text
            if cid is None:
                return None
            should_end = True
        try:
            pending = self._pending_timers.get(campaign_id)
            if pending is not None and pending.get("interruptible", True):
                cancelled_timer = self.cancel_pending_timer(campaign_id)
                with self._session_factory() as session:
                    row = (
                        session.query(Player)
                        .filter(Player.campaign_id == campaign_id)
                        .filter(Player.actor_id == actor_id)
                        .first()
                    )
                    if row is not None:
                        self.increment_player_stat(row, self.PLAYER_STATS_TIMERS_AVERTED_KEY)
                if cancelled_timer is not None:
                    event_desc = str(cancelled_timer.get("event") or "an impending event")
                    interrupt_action = cancelled_timer.get("interrupt_action")
                    interrupt_note = (
                        "[TIMER INTERRUPTED] The player acted before the timed event fired. "
                        f'Averted event: "{event_desc}"'
                    )
                    if isinstance(interrupt_action, str) and interrupt_action.strip():
                        interrupt_note += f' Interruption context: "{interrupt_action.strip()}"'
                    with self._session_factory() as session:
                        session.add(
                            Turn(
                                campaign_id=campaign_id,
                                session_id=session_id,
                                actor_id=actor_id,
                                kind="narrator",
                                content=interrupt_note,
                            )
                        )
                        session.commit()

            with self._session_factory() as session:
                row = (
                    session.query(Player)
                    .filter(Player.campaign_id == campaign_id)
                    .filter(Player.actor_id == actor_id)
                    .first()
                )
                if row is not None:
                    pre_inventory_rich = self._get_inventory_rich(parse_json_dict(row.state_json))
                    self.record_player_message(row)
                campaign_row = session.get(Campaign, campaign_id)
                if campaign_row is not None:
                    pre_character_slugs = set(parse_json_dict(campaign_row.characters_json).keys())
            is_ooc = bool(re.match(r"\s*\[OOC\b", action or "", re.IGNORECASE))

            result = await self._engine.resolve_turn(
                ResolveTurnInput(
                    campaign_id=campaign_id,
                    actor_id=actor_id,
                    action=action,
                    session_id=session_id,
                    record_player_turn=not is_ooc,
                )
            )
            if result.status == "ok":
                self._apply_give_item_transfer(
                    campaign_id=campaign_id,
                    actor_id=actor_id,
                    action_text=action,
                    narration_text=result.narration or "",
                    give_item=result.give_item,
                    pre_inventory_rich=pre_inventory_rich,
                )
                timer_delay_seconds: int | None = None
                if result.timer_instruction is not None:
                    timer_delay_seconds = int(result.timer_instruction.delay_seconds)
                    with self._session_factory() as session:
                        campaign_row = session.get(Campaign, campaign_id)
                    speed = self.get_speed_multiplier(campaign_row)
                    if speed > 0:
                        timer_delay_seconds = int(timer_delay_seconds / speed)
                    timer_delay_seconds = max(15, min(300, timer_delay_seconds))
                if result.timer_instruction is not None and session_id is not None:
                    with self._session_factory() as session:
                        sess = session.get(GameSession, session_id)
                        channel_ref = None
                        if sess is not None:
                            channel_ref = sess.surface_thread_id or sess.surface_channel_id or sess.surface_key
                        if channel_ref is None:
                            channel_ref = session_id
                    self._schedule_timer(
                        campaign_id=campaign_id,
                        channel_id=str(channel_ref),
                        delay_seconds=int(timer_delay_seconds or result.timer_instruction.delay_seconds),
                        event_description=result.timer_instruction.event_text,
                        interruptible=bool(result.timer_instruction.interruptible),
                        interrupt_action=result.timer_instruction.interrupt_action,
                    )
                portrait_channel_ref: str | None = None
                if session_id is not None:
                    with self._session_factory() as session:
                        sess = session.get(GameSession, session_id)
                        if sess is not None:
                            portrait_channel_ref = (
                                sess.surface_thread_id or sess.surface_channel_id or sess.surface_key
                            )
                    if portrait_channel_ref is None:
                        portrait_channel_ref = session_id
                await self._enqueue_new_character_portraits(
                    campaign_id=campaign_id,
                    actor_id=actor_id,
                    pre_slugs=pre_character_slugs,
                    channel_id=portrait_channel_ref,
                )
                return self._decorate_narration_and_persist(
                    campaign_id=campaign_id,
                    actor_id=actor_id,
                    narration=result.narration or "",
                    timer_instruction=result.timer_instruction,
                    timer_delay_seconds=timer_delay_seconds,
                )
            if result.status == "busy":
                return None
            if result.status == "conflict":
                return "The world shifts under your feet. Please try again."
            return f"Engine error: {result.conflict_reason or 'unknown'}"
        finally:
            if should_end:
                self.end_turn(campaign_id, actor_id)

    def _decorate_narration_and_persist(
        self,
        *,
        campaign_id: str,
        actor_id: str,
        narration: str,
        timer_instruction=None,
        timer_delay_seconds: int | None = None,
    ) -> str:
        decorated = self._strip_narration_footer((narration or "").strip())
        has_inventory_line = any(
            line.strip().lower().startswith("inventory:")
            for line in decorated.splitlines()
        )
        has_timer_line = any(
            line.strip().startswith("⏰")
            for line in decorated.splitlines()
        )

        with self._session_factory() as session:
            player = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == actor_id)
                .first()
            )
            player_state = self.get_player_state(player) if player is not None else {}
            inventory_line = self._format_inventory(player_state) or "Inventory: empty"

            if not has_inventory_line:
                if decorated:
                    decorated = f"{decorated}\n\n{inventory_line}"
                else:
                    decorated = inventory_line

            if timer_instruction is not None and not has_timer_line:
                delay_seconds = (
                    int(timer_delay_seconds)
                    if timer_delay_seconds is not None
                    else max(0, int(getattr(timer_instruction, "delay_seconds", 0) or 0))
                )
                expiry_ts = int(time.time()) + delay_seconds
                event_hint = str(getattr(timer_instruction, "event_text", "") or "Something happens")
                interruptible = bool(getattr(timer_instruction, "interruptible", True))
                interrupt_hint = "act to prevent!" if interruptible else "unavoidable"
                decorated = (
                    f"{decorated}\n\n"
                    f"⏰ <t:{expiry_ts}:R>: {event_hint} ({interrupt_hint})"
                )

            campaign = session.get(Campaign, campaign_id)
            if campaign is not None:
                campaign.last_narration = decorated
                campaign.updated_at = datetime.utcnow()

            narrator_turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .filter(Turn.kind == "narrator")
                .order_by(Turn.id.desc())
                .first()
            )
            if narrator_turn is not None:
                narrator_turn.content = decorated
                snapshot = session.query(Snapshot).filter(Snapshot.turn_id == narrator_turn.id).first()
                if snapshot is not None:
                    snapshot.campaign_last_narration = decorated
            session.commit()

        return decorated

    def _is_thread_channel(self, channel_obj: Any) -> bool:
        if channel_obj is None:
            return False
        channel_type = str(getattr(channel_obj, "type", "") or "").lower()
        if "thread" in channel_type:
            return True
        if getattr(channel_obj, "parent_id", None) is not None:
            return True
        class_name = channel_obj.__class__.__name__.lower()
        return "thread" in class_name

    def _persist_player_state_for_campaign_actor(
        self,
        campaign_id: str,
        actor_id: str,
        player_state: dict[str, object],
    ) -> None:
        with self._session_factory() as session:
            row = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == actor_id)
                .first()
            )
            if row is None:
                return
            row.state_json = self._dump_json(player_state)
            row.updated_at = datetime.utcnow()
            session.commit()

    def _record_simple_turn_pair(
        self,
        *,
        campaign_id: str,
        actor_id: str,
        session_id: str | None,
        action_text: str,
        narration: str,
    ) -> None:
        with self._session_factory() as session:
            session.add(
                Turn(
                    campaign_id=campaign_id,
                    session_id=session_id,
                    actor_id=actor_id,
                    kind="player",
                    content=action_text,
                )
            )
            session.add(
                Turn(
                    campaign_id=campaign_id,
                    session_id=session_id,
                    actor_id=actor_id,
                    kind="narrator",
                    content=narration,
                )
            )
            campaign = session.get(Campaign, campaign_id)
            if campaign is not None:
                campaign.last_narration = narration
                campaign.updated_at = datetime.utcnow()
            session.commit()

    def _apply_give_item_transfer(
        self,
        *,
        campaign_id: str,
        actor_id: str,
        action_text: str,
        narration_text: str,
        give_item: dict[str, object] | None,
        pre_inventory_rich: list[dict[str, str]],
    ) -> None:
        if not pre_inventory_rich:
            return
        pre_map = {entry["name"].lower(): entry["name"] for entry in pre_inventory_rich if entry.get("name")}
        if not pre_map:
            return

        with self._session_factory() as session:
            source_player = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == actor_id)
                .first()
            )
            if source_player is None:
                return
            source_state = parse_json_dict(source_player.state_json)
            source_inventory = self._get_inventory_rich(source_state)
            source_now = {entry["name"].lower() for entry in source_inventory if entry.get("name")}

            removed = [pre_map[key] for key in pre_map if key not in source_now]
            resolved_give_item: dict[str, object] | None = give_item if isinstance(give_item, dict) else None

            # Heuristic fallback: if model forgot give_item but removed
            # items + narration mentions giving to another player, infer it.
            if resolved_give_item is None:
                if not removed:
                    return
                give_re = re.compile(r"\b(?:give|hand|pass|toss|offer|slide)\b", re.IGNORECASE)
                refuse_re = re.compile(
                    r"\b(?:doesn'?t take|does not take|refuse[sd]?|reject[sd]?|decline[sd]?"
                    r"|push(?:es|ed)? (?:it |the \w+ )?(?:back|away)"
                    r"|won'?t (?:take|accept)|shake[sd]? (?:his|her|their) head"
                    r"|hands? it back|gives? it back|returns? (?:it|the))\b",
                    re.IGNORECASE,
                )
                if not (give_re.search(action_text) or give_re.search(narration_text)):
                    return
                if refuse_re.search(narration_text):
                    return
                mention_re = re.compile(r"<@!?(\d+)>")
                target_actor_id: str | None = None
                for match in mention_re.finditer(narration_text):
                    candidate = str(match.group(1))
                    if candidate and candidate != str(actor_id):
                        target_actor_id = candidate
                        break
                if not target_actor_id:
                    return
                inferred_item: str | None = removed[0] if len(removed) == 1 else None
                if inferred_item is None:
                    action_lower = action_text.lower()
                    for removed_item in removed:
                        if removed_item.lower() in action_lower:
                            inferred_item = removed_item
                            break
                if not inferred_item:
                    return
                resolved_give_item = {
                    "item": inferred_item,
                    "to_discord_mention": f"<@{target_actor_id}>",
                }

            gi_item_name = str(resolved_give_item.get("item") or "").strip()
            gi_target_actor_id = str(resolved_give_item.get("to_actor_id") or "").strip()
            gi_mention = str(resolved_give_item.get("to_discord_mention") or "").strip()
            if not gi_target_actor_id and gi_mention.startswith("<@") and gi_mention.endswith(">"):
                try:
                    gi_target_actor_id = str(int(gi_mention.strip("<@!>")))
                except (ValueError, TypeError):
                    gi_target_actor_id = ""

            if not gi_item_name or not gi_target_actor_id or gi_target_actor_id == str(actor_id):
                return

            giver_has_now = any(
                entry["name"].lower() == gi_item_name.lower()
                for entry in source_inventory
                if entry.get("name")
            )
            giver_had_before = gi_item_name.lower() in pre_map
            if not (giver_has_now or giver_had_before):
                return

            target_player = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == gi_target_actor_id)
                .first()
            )
            if target_player is None:
                return

            if giver_has_now:
                source_state["inventory"] = self._apply_inventory_delta(
                    source_inventory,
                    [],
                    [gi_item_name],
                    origin_hint="",
                )
                source_player.state_json = self._dump_json(source_state)

            target_state = parse_json_dict(target_player.state_json)
            target_inventory = self._get_inventory_rich(target_state)
            target_state["inventory"] = self._apply_inventory_delta(
                target_inventory,
                [gi_item_name],
                [],
                origin_hint=f"Received from <@{actor_id}>",
            )
            target_player.state_json = self._dump_json(target_state)
            target_player.updated_at = datetime.utcnow()
            source_player.updated_at = datetime.utcnow()
            session.commit()

    async def _enqueue_new_character_portraits(
        self,
        *,
        campaign_id: str,
        actor_id: str,
        pre_slugs: set[str],
        channel_id: str | None = None,
    ) -> None:
        if self._media_port is None or not self._gpu_worker_available():
            return
        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return
            characters = parse_json_dict(campaign.characters_json)
        if not isinstance(characters, dict):
            return
        for slug, value in characters.items():
            if slug in pre_slugs or not isinstance(value, dict):
                continue
            appearance = str(value.get("appearance") or "").strip()
            image_url = str(value.get("image_url") or "").strip()
            if not appearance or image_url:
                continue
            name = str(value.get("name") or slug).strip()
            await self._enqueue_character_portrait(
                campaign_id=campaign_id,
                actor_id=actor_id,
                character_slug=slug,
                name=name,
                appearance=appearance,
                channel_id=channel_id,
            )

    async def play_action(
        self,
        campaign_or_ctx: str | Any | None = None,
        actor_id: str | None = None,
        action: str | None = None,
        session_id: str | None = None,
        manage_claim: bool = True,
        *,
        command_prefix: str = "!",
        campaign_id: str | None = None,
    ) -> Optional[str]:
        # Legacy compatibility: play_action(ctx, action, command_prefix="!", campaign_id=..., manage_claim=...)
        if self._is_context_like(campaign_or_ctx):
            ctx = campaign_or_ctx
            action_text = str(actor_id or action or "").strip()
            if not action_text:
                return None
            actor_id_text = str(getattr(getattr(ctx, "author", None), "id", ""))
            resolved_campaign_id = str(campaign_id or "").strip()
            if not resolved_campaign_id:
                resolved_campaign_id, error_text = self._resolve_campaign_for_context(
                    ctx,
                    command_prefix=command_prefix,
                )
                if error_text is not None:
                    return error_text
                if resolved_campaign_id is None:
                    return None
            guild = getattr(ctx, "guild", None)
            channel = getattr(ctx, "channel", None)
            derived_session_id = ""
            session_row = None
            if guild is not None and channel is not None:
                session_row = self.get_or_create_channel(
                    str(getattr(guild, "id", "") or ""),
                    str(getattr(channel, "id", "") or ""),
                )
                derived_session_id = str(session_row.id or "")
            with self._session_factory() as session:
                campaign_obj = session.get(Campaign, str(resolved_campaign_id))
            if campaign_obj is None:
                return "Campaign not found."
            player = self.get_or_create_player(str(resolved_campaign_id), actor_id_text)
            player_state = self.get_player_state(player)
            action_clean = action_text.strip().lower()
            is_thread_channel = self._is_thread_channel(channel)

            has_character_name = bool(str(player_state.get("character_name") or "").strip())
            campaign_has_content = bool((campaign_obj.summary or "").strip())
            needs_identity = campaign_has_content and not has_character_name
            if needs_identity:
                return (
                    "This campaign already has adventurers. "
                    f"Set your identity first with `{command_prefix}zork identity <name>`. "
                    "Then return to the adventure."
                )

            onboarding_state = player_state.get("onboarding_state")
            party_status = player_state.get("party_status")
            if not is_thread_channel:
                if not party_status and not onboarding_state:
                    player_state["onboarding_state"] = "await_party_choice"
                    self._persist_player_state_for_campaign_actor(str(resolved_campaign_id), actor_id_text, player_state)
                    return (
                        "Mission rejected until path is selected. Reply with exactly one option:\n"
                        f"- `{self.MAIN_PARTY_TOKEN}`\n"
                        f"- `{self.NEW_PATH_TOKEN}`"
                    )

                if onboarding_state == "await_party_choice":
                    if action_clean == self.MAIN_PARTY_TOKEN:
                        player_state["party_status"] = "main_party"
                        player_state["onboarding_state"] = None
                        self._persist_player_state_for_campaign_actor(str(resolved_campaign_id), actor_id_text, player_state)
                        return "Joined main party. Your next message will be treated as an in-world action."

                    if action_clean == self.NEW_PATH_TOKEN:
                        player_state["onboarding_state"] = "await_campaign_name"
                        self._persist_player_state_for_campaign_actor(str(resolved_campaign_id), actor_id_text, player_state)
                        options = self._build_campaign_suggestion_text(str(getattr(guild, "id", "default") or "default"))
                        return (
                            "Reply next with your campaign name (letters/numbers/spaces).\n"
                            f"{options}\n"
                            f"Hint: `{command_prefix}zork thread <name>` also creates your own path thread."
                        )

                    return (
                        "Mission rejected. Reply with exactly one option:\n"
                        f"- `{self.MAIN_PARTY_TOKEN}`\n"
                        f"- `{self.NEW_PATH_TOKEN}`"
                    )

                if onboarding_state == "await_campaign_name":
                    campaign_name = self._sanitize_campaign_name_text(action_text)
                    if not campaign_name:
                        return "Mission rejected. Reply with a campaign name using letters/numbers/spaces."
                    if len(campaign_name) < 2:
                        return "Mission rejected. Campaign name must be at least 2 characters."

                    if session_row is None:
                        return f"Could not create a new path thread here. Use `{command_prefix}zork thread {campaign_name}`."

                    switched_campaign, switched, reason = self.set_active_campaign(
                        session_row,
                        str(getattr(guild, "id", "default") or "default"),
                        campaign_name,
                        actor_id_text,
                        enforce_activity_window=False,
                    )
                    if not switched or switched_campaign is None:
                        return f"Could not switch campaign: {reason or 'unknown error'}"

                    switched_player = self.get_or_create_player(switched_campaign.id, actor_id_text)
                    switched_state = self.get_player_state(switched_player)
                    switched_state = self._copy_identity_fields(player_state, switched_state)
                    switched_state["party_status"] = "new_path"
                    switched_state["onboarding_state"] = None
                    self._persist_player_state_for_campaign_actor(switched_campaign.id, actor_id_text, switched_state)

                    player_state["party_status"] = "new_path"
                    player_state["onboarding_state"] = None
                    self._persist_player_state_for_campaign_actor(str(resolved_campaign_id), actor_id_text, player_state)
                    return (
                        f"Switched to campaign: `{switched_campaign.name}`\n"
                        "Continue your adventure here."
                    )

            if action_clean in ("look", "l") and (
                player_state.get("room_description") or player_state.get("room_summary")
            ):
                title = str(
                    player_state.get("room_title")
                    or player_state.get("location")
                    or "Unknown"
                )
                desc = str(
                    player_state.get("room_description")
                    or player_state.get("room_summary")
                    or ""
                )
                exits = player_state.get("exits")
                if exits and isinstance(exits, list):
                    exit_list = [
                        (entry.get("direction") or entry.get("name") or str(entry))
                        if isinstance(entry, dict)
                        else str(entry)
                        for entry in exits
                    ]
                    exits_text = f"\nExits: {', '.join(exit_list)}"
                else:
                    exits_text = ""
                narration = f"{title}\n{desc}{exits_text}"
                inventory_line = self._format_inventory(player_state)
                if inventory_line:
                    narration = f"{narration}\n\n{inventory_line}"
                narration = self._trim_text(narration, self.MAX_NARRATION_CHARS)
                self._record_simple_turn_pair(
                    campaign_id=str(resolved_campaign_id),
                    actor_id=actor_id_text,
                    session_id=derived_session_id or None,
                    action_text=action_text,
                    narration=narration,
                )
                return narration

            if action_clean in ("inventory", "inv", "i"):
                narration = self._format_inventory(player_state) or "Inventory: empty"
                narration = self._trim_text(narration, self.MAX_NARRATION_CHARS)
                self._record_simple_turn_pair(
                    campaign_id=str(resolved_campaign_id),
                    actor_id=actor_id_text,
                    session_id=derived_session_id or None,
                    action_text=action_text,
                    narration=narration,
                )
                return narration

            if action_clean in ("calendar", "cal", "events"):
                campaign_state = self.get_campaign_state(campaign_obj)
                game_time = campaign_state.get("game_time", {})
                calendar_entries = self._calendar_for_prompt(campaign_state)
                date_label = game_time.get("date_label")
                if not date_label:
                    day = game_time.get("day", "?")
                    period = str(game_time.get("period", "?")).title()
                    date_label = f"Day {day}, {period}"
                lines = [f"**Game Time:** {date_label}"]
                if calendar_entries:
                    lines.append("**Upcoming Events:**")
                    for event in calendar_entries:
                        days_remaining = int(event.get("days_remaining", 0))
                        fire_day = int(event.get("fire_day", 1))
                        desc = str(event.get("description", "") or "")
                        if days_remaining < 0:
                            eta = f"overdue by {abs(days_remaining)} day(s)"
                        elif days_remaining == 0:
                            eta = "fires today"
                        elif days_remaining == 1:
                            eta = "fires tomorrow"
                        else:
                            eta = f"fires in {days_remaining} days"
                        line = f"- **{event.get('name', 'Unknown')}** - Day {fire_day} ({eta})"
                        if desc:
                            line += f" ({desc})"
                        lines.append(line)
                else:
                    lines.append("No upcoming events.")
                return "\n".join(lines)

            if action_clean in ("roster", "characters", "npcs"):
                characters = self.get_campaign_characters(campaign_obj)
                if not characters:
                    return "No characters in the roster yet."
                lines = ["**Character Roster:**"]
                for slug, char in characters.items():
                    if not isinstance(char, dict):
                        continue
                    name = char.get("name", slug)
                    location = char.get("location", "unknown")
                    status = char.get("current_status", "")
                    background = str(char.get("background", "") or "")
                    origin = background.split(".")[0].strip() if background else ""
                    portrait = char.get("image_url", "")
                    deceased = char.get("deceased_reason")
                    entry = f"- **{name}** ({slug})"
                    if deceased:
                        entry += f" [DECEASED: {deceased}]"
                    else:
                        entry += f" - {location}"
                        if status:
                            entry += f" | {status}"
                    if origin:
                        entry += f"\n  *{origin}.*"
                    if portrait:
                        entry += f"\n  Portrait: {portrait}"
                    lines.append(entry)
                return "\n".join(lines)

            return await self._play_action_with_ids(
                campaign_id=str(resolved_campaign_id),
                actor_id=actor_id_text,
                action=action_text,
                session_id=derived_session_id or None,
                manage_claim=manage_claim,
            )

        campaign_id_text = str(campaign_id or campaign_or_ctx or "")
        actor_id_text = str(actor_id or "")
        action_text = str(action or "")
        if not campaign_id_text or not actor_id_text:
            return "Campaign or actor not found."
        return await self._play_action_with_ids(
            campaign_id=campaign_id_text,
            actor_id=actor_id_text,
            action=action_text,
            session_id=session_id,
            manage_claim=manage_claim,
        )

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

        # Ensure FK-safe rewind when embeddings exist for to-be-deleted turns.
        self._cleanup_embeddings_after_rewind(campaign_id, after_turn_id=target_turn_id)
        result = self._engine.rewind_to_turn(campaign_id, target_turn_id)
        if result.status != "ok" or result.target_turn_id is None:
            return None
        self._cleanup_embeddings_after_rewind(campaign_id, after_turn_id=result.target_turn_id)
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
                session.query(Embedding).filter(Embedding.turn_id.in_(turn_ids_to_delete)).delete(
                    synchronize_session=False
                )
                deleted_count = (
                    session.query(Turn)
                    .filter(Turn.id.in_(turn_ids_to_delete))
                    .delete(synchronize_session=False)
                )
            else:
                deleted_count = 0

            session.commit()
            return (target_turn_id, int(deleted_count))

    def _cleanup_embeddings_after_rewind(
        self,
        campaign_id: str,
        *,
        after_turn_id: int,
    ) -> None:
        try:
            with self._session_factory() as session:
                session.query(Embedding).filter(Embedding.campaign_id == campaign_id).filter(
                    Embedding.turn_id > after_turn_id
                ).delete(synchronize_session=False)
                session.commit()
        except Exception:
            self._logger.debug(
                "Zork rewind: embedding cleanup failed for campaign %s",
                campaign_id,
                exc_info=True,
            )

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
            pending = self._pending_timers.get(campaign_id)
            if pending is not None:
                pending["message_id"] = str(message_id)
                if channel_id is not None:
                    pending["channel_id"] = str(channel_id)
            return True

    def _get_lock(self, campaign_id: str) -> asyncio.Lock:
        lock = self._locks.get(campaign_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[campaign_id] = lock
        return lock

    def is_guardrails_enabled(self, campaign: Campaign | None) -> bool:
        if campaign is None:
            return False
        campaign_state = self.get_campaign_state(campaign)
        return bool(campaign_state.get("guardrails_enabled", False))

    def set_guardrails_enabled(self, campaign: Campaign | None, enabled: bool) -> bool:
        if campaign is None:
            return False
        campaign_state = self.get_campaign_state(campaign)
        campaign_state["guardrails_enabled"] = bool(enabled)
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is None:
                return False
            row.state_json = self._dump_json(campaign_state)
            row.updated_at = datetime.utcnow()
            session.commit()
            campaign.state_json = row.state_json
            campaign.updated_at = row.updated_at
        return True

    def is_on_rails(self, campaign: Campaign | None) -> bool:
        if campaign is None:
            return False
        campaign_state = self.get_campaign_state(campaign)
        return bool(campaign_state.get("on_rails", False))

    def set_on_rails(self, campaign: Campaign | None, enabled: bool) -> bool:
        if campaign is None:
            return False
        campaign_state = self.get_campaign_state(campaign)
        campaign_state["on_rails"] = bool(enabled)
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is None:
                return False
            row.state_json = self._dump_json(campaign_state)
            row.updated_at = datetime.utcnow()
            session.commit()
            campaign.state_json = row.state_json
            campaign.updated_at = row.updated_at
        return True

    def is_timed_events_enabled(self, campaign: Campaign | None) -> bool:
        if campaign is None:
            return False
        campaign_state = self.get_campaign_state(campaign)
        return bool(campaign_state.get("timed_events_enabled", True))

    def set_timed_events_enabled(self, campaign: Campaign | None, enabled: bool) -> bool:
        if campaign is None:
            return False
        campaign_state = self.get_campaign_state(campaign)
        campaign_state["timed_events_enabled"] = bool(enabled)
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is None:
                return False
            row.state_json = self._dump_json(campaign_state)
            row.updated_at = datetime.utcnow()
            session.commit()
            campaign.state_json = row.state_json
            campaign.updated_at = row.updated_at
        if not enabled:
            self.cancel_pending_timer(campaign.id)
        return True

    def get_speed_multiplier(self, campaign: Campaign | None) -> float:
        if campaign is None:
            return 1.0
        campaign_state = self.get_campaign_state(campaign)
        raw = campaign_state.get("speed_multiplier", 1.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 1.0

    def set_speed_multiplier(self, campaign: Campaign | None, multiplier: float) -> bool:
        if campaign is None:
            return False
        multiplier = max(0.1, min(10.0, float(multiplier)))
        campaign_state = self.get_campaign_state(campaign)
        campaign_state["speed_multiplier"] = multiplier
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is None:
                return False
            row.state_json = self._dump_json(campaign_state)
            row.updated_at = datetime.utcnow()
            session.commit()
            campaign.state_json = row.state_json
            campaign.updated_at = row.updated_at
        return True

    def cancel_pending_timer(self, campaign_id: str) -> dict[str, Any] | None:
        ctx_dict = self._pending_timers.pop(campaign_id, None)
        if ctx_dict is None:
            return None
        task = ctx_dict.get("task")
        if task is not None and not task.done():
            task.cancel()
        message_id = ctx_dict.get("message_id")
        channel_id = ctx_dict.get("channel_id")
        if message_id and channel_id:
            event = ctx_dict.get("event", "unknown event")
            asyncio.ensure_future(
                self._edit_timer_line(
                    str(channel_id),
                    str(message_id),
                    f"✅ *Timer cancelled - you acted in time. (Averted: {event})*",
                )
            )
        return ctx_dict

    async def _edit_timer_line(self, channel_id: str, message_id: str, replacement: str) -> None:
        if self._timer_effects_port is None:
            return
        try:
            await self._timer_effects_port.edit_timer_line(channel_id, message_id, replacement)
        except Exception:
            self._logger.debug("Failed to edit timer message %s", message_id, exc_info=True)
            return

    def _schedule_timer(
        self,
        campaign_id: str,
        channel_id: str,
        delay_seconds: int,
        event_description: str,
        interruptible: bool = True,
        interrupt_action: str | None = None,
    ) -> None:
        task = asyncio.create_task(
            self._timer_task(campaign_id, channel_id, delay_seconds, event_description)
        )
        self._pending_timers[campaign_id] = {
            "task": task,
            "channel_id": channel_id,
            "message_id": None,
            "event": event_description,
            "delay": delay_seconds,
            "interruptible": interruptible,
            "interrupt_action": interrupt_action,
        }

    async def _timer_task(
        self,
        campaign_id: str,
        channel_id: str,
        delay_seconds: int,
        event_description: str,
    ) -> None:
        try:
            await asyncio.sleep(delay_seconds)
        except asyncio.CancelledError:
            return
        timer_ctx = self._pending_timers.pop(campaign_id, None)
        if timer_ctx:
            msg_id = timer_ctx.get("message_id")
            ch_id = timer_ctx.get("channel_id")
            if msg_id and ch_id:
                asyncio.ensure_future(
                    self._edit_timer_line(
                        str(ch_id),
                        str(msg_id),
                        f"⚠️ *Timer expired - {event_description}*",
                    )
                )
        try:
            await self._execute_timed_event(campaign_id, channel_id, event_description)
        except Exception:
            self._logger.exception(
                "Zork timed event failed: campaign=%s event=%r",
                campaign_id,
                event_description,
            )

    async def _execute_timed_event(
        self,
        campaign_id: str,
        channel_id: str,
        event_description: str,
    ) -> None:
        active_actor_id: str | None = None
        pre_character_slugs: set[str] = set()
        lock = self._get_lock(campaign_id)
        async with lock:
            with self._session_factory() as session:
                campaign = session.get(Campaign, campaign_id)
                if campaign is None:
                    return
                pre_character_slugs = set(parse_json_dict(campaign.characters_json).keys())
                if not self.is_timed_events_enabled(campaign):
                    return
                latest_turn = (
                    session.query(Turn)
                    .filter(Turn.campaign_id == campaign_id)
                    .order_by(Turn.id.desc())
                    .first()
                )
                if latest_turn is not None and latest_turn.kind == "player":
                    created_at = latest_turn.created_at
                    if created_at is not None:
                        age_seconds = (datetime.utcnow() - created_at).total_seconds()
                        if age_seconds < 5:
                            return
                active_player = (
                    session.query(Player)
                    .filter(Player.campaign_id == campaign_id)
                    .order_by(Player.last_active_at.desc())
                    .first()
                )
                if active_player is None:
                    return
                active_actor_id = active_player.actor_id
                self.increment_player_stat(active_player, self.PLAYER_STATS_TIMERS_MISSED_KEY)

        if not active_actor_id:
            return

        result = await self._engine.resolve_turn(
            ResolveTurnInput(
                campaign_id=campaign_id,
                actor_id=active_actor_id,
                action=f"[SYSTEM EVENT - TIMED]: {event_description}",
                record_player_turn=False,
                allow_timer_instruction=False,
            )
        )
        if result.status != "ok":
            return
        await self._enqueue_new_character_portraits(
            campaign_id=campaign_id,
            actor_id=active_actor_id,
            pre_slugs=pre_character_slugs,
            channel_id=channel_id,
        )
        narration = self._strip_narration_footer(result.narration or "")
        if narration and self._timer_effects_port is not None:
            try:
                await self._timer_effects_port.emit_timed_event(
                    campaign_id=campaign_id,
                    channel_id=channel_id,
                    actor_id=active_actor_id,
                    narration=narration,
                )
            except Exception:
                return

    def _trim_text(self, text: str, max_chars: int) -> str:
        if text is None:
            return ""
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def _append_summary(self, existing: str, update: str) -> str:
        if not update:
            return existing or ""
        update = update.strip()
        if not existing:
            return self._trim_text(update, self.MAX_SUMMARY_CHARS)
        existing_lower = existing.lower()
        new_lines: list[str] = []
        for line in update.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower() in existing_lower:
                continue
            new_lines.append(line)
        if not new_lines:
            return self._trim_text(existing, self.MAX_SUMMARY_CHARS)
        merged = (existing + "\n" + "\n".join(new_lines)).strip()
        return self._trim_text(merged, self.MAX_SUMMARY_CHARS)

    def _fit_state_to_budget(self, state: Dict[str, object], max_chars: int) -> Dict[str, object]:
        text = self._dump_json(state)
        if len(text) <= max_chars:
            return state
        state = dict(state)
        ranked = sorted(state.keys(), key=lambda key: len(self._dump_json(state[key])), reverse=True)
        for key in ranked:
            del state[key]
            if len(self._dump_json(state)) <= max_chars:
                break
        return state

    def _prune_stale_state(self, state: Dict[str, object]) -> Dict[str, object]:
        pruned: Dict[str, object] = {}
        for key, value in state.items():
            if isinstance(value, str) and value.strip().lower() in self._STALE_VALUE_PATTERNS:
                continue
            if value is True and any(
                key.endswith(suffix)
                for suffix in (
                    "_complete",
                    "_arrived",
                    "_announced",
                    "_revealed",
                    "_concluded",
                    "_departed",
                    "_dispatched",
                    "_offered",
                    "_introduced",
                    "_unlocked",
                )
            ):
                continue
            if isinstance(value, (int, float)) and any(
                key.endswith(suffix)
                for suffix in (
                    "_eta_minutes",
                    "_eta",
                    "_countdown_minutes",
                    "_countdown_hours",
                    "_countdown",
                    "_deadline_seconds",
                    "_time_elapsed",
                )
            ):
                continue
            if isinstance(value, str) and any(key.endswith(suffix) for suffix in ("_eta", "_eta_minutes")):
                continue
            pruned[key] = value
        return pruned

    def _build_model_state(self, campaign_state: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(campaign_state, dict):
            return {}
        model_state: Dict[str, object] = {}
        for key, value in campaign_state.items():
            if key in self.MODEL_STATE_EXCLUDE_KEYS:
                continue
            model_state[key] = value
        return self._prune_stale_state(model_state)

    def _build_story_context(self, campaign_state: Dict[str, object]) -> Optional[str]:
        outline = campaign_state.get("story_outline")
        if not isinstance(outline, dict):
            return None
        chapters = outline.get("chapters")
        if not isinstance(chapters, list) or not chapters:
            return None

        current_chapter = campaign_state.get("current_chapter", 0)
        current_scene = campaign_state.get("current_scene", 0)
        if not isinstance(current_chapter, int):
            current_chapter = 0
        if not isinstance(current_scene, int):
            current_scene = 0

        lines: List[str] = []
        if current_chapter > 0 and current_chapter - 1 < len(chapters):
            prev = chapters[current_chapter - 1]
            lines.append(f"PREVIOUS CHAPTER: {prev.get('title', 'Untitled')}")
            lines.append(f"  Summary: {prev.get('summary', '')}")
            lines.append("")

        if current_chapter < len(chapters):
            cur = chapters[current_chapter]
            lines.append(f"CURRENT CHAPTER: {cur.get('title', 'Untitled')}")
            lines.append(f"  Summary: {cur.get('summary', '')}")
            scenes = cur.get("scenes")
            if isinstance(scenes, list):
                for idx, scene in enumerate(scenes):
                    marker = " >>> CURRENT SCENE <<<" if idx == current_scene else ""
                    lines.append(f"  Scene {idx + 1}: {scene.get('title', 'Untitled')}{marker}")
                    lines.append(f"    Summary: {scene.get('summary', '')}")
                    setting = scene.get("setting")
                    if setting:
                        lines.append(f"    Setting: {setting}")
                    key_characters = scene.get("key_characters")
                    if key_characters:
                        lines.append(f"    Key characters: {', '.join(key_characters)}")
            lines.append("")

        if current_chapter + 1 < len(chapters):
            nxt = chapters[current_chapter + 1]
            lines.append(f"NEXT CHAPTER: {nxt.get('title', 'Untitled')}")
            summary = nxt.get("summary", "")
            if summary:
                lines.append(f"  Preview: {summary[:200]}")
        return "\n".join(lines) if lines else None

    def _split_room_state(
        self,
        state_update: Dict[str, object],
        player_state_update: Dict[str, object],
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        if not isinstance(state_update, dict):
            state_update = {}
        if not isinstance(player_state_update, dict):
            player_state_update = {}
        for key in self.ROOM_STATE_KEYS:
            if key in state_update and key not in player_state_update:
                player_state_update[key] = state_update.pop(key)
        return state_update, player_state_update

    def _build_player_state_for_prompt(self, player_state: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(player_state, dict):
            return {}
        model_state: Dict[str, object] = {}
        for key, value in player_state.items():
            if key in self.PLAYER_STATE_EXCLUDE_KEYS:
                continue
            model_state[key] = value
        return model_state

    def _assign_player_markers(self, players: List[Player], exclude_actor_id: str) -> List[dict]:
        markers: List[dict] = []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        index = 0
        for player in players:
            if player.actor_id == exclude_actor_id:
                continue
            if index >= len(letters):
                break
            markers.append({"marker": letters[index], "player": player})
            index += 1
        return markers

    def _strip_narration_footer(self, text: str) -> str:
        if not text:
            return text
        idx = text.rfind("---")
        if idx == -1:
            return text
        tail = text[idx:]
        if "xp" in tail.lower():
            return text[:idx].rstrip()
        return text

    def _format_inventory(self, player_state: Dict[str, object]) -> Optional[str]:
        if not isinstance(player_state, dict):
            return None
        items = self._get_inventory_rich(player_state)
        if not items:
            return None
        return f"Inventory: {', '.join([entry['name'] for entry in items])}"

    def _item_to_text(self, item) -> str:
        if isinstance(item, dict):
            if "name" in item and item.get("name") is not None:
                return str(item.get("name")).strip()
            if "item" in item and item.get("item") is not None:
                return str(item.get("item")).strip()
            if "title" in item and item.get("title") is not None:
                return str(item.get("title")).strip()
            return ""
        return str(item).strip()

    def _normalize_inventory_items(self, value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        if not isinstance(value, list):
            return []
        cleaned: List[str] = []
        seen: set[str] = set()
        for item in value:
            item_text = self._item_to_text(item)
            if not item_text:
                continue
            normalized = item_text.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(item_text)
        return cleaned

    def _get_inventory_rich(self, player_state: Dict[str, object]) -> List[Dict[str, str]]:
        raw = player_state.get("inventory") if isinstance(player_state, dict) else None
        if not raw or not isinstance(raw, list):
            return []
        result: List[Dict[str, str]] = []
        seen: set[str] = set()
        for item in raw:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("item") or item.get("title") or "").strip()
                origin = str(item.get("origin") or "").strip()
            else:
                name = str(item).strip()
                origin = ""
            if not name:
                continue
            norm = name.lower()
            if norm in seen:
                continue
            seen.add(norm)
            result.append({"name": name, "origin": origin})
        return result

    def _apply_inventory_delta(
        self,
        current: List[Dict[str, str]],
        adds: List[str],
        removes: List[str],
        origin_hint: str = "",
    ) -> List[Dict[str, str]]:
        remove_norm = {item.lower() for item in removes}
        out: List[Dict[str, str]] = []
        for entry in current:
            if entry["name"].lower() in remove_norm:
                continue
            out.append(entry)
        out_norm = {entry["name"].lower() for entry in out}
        for item in adds:
            if item.lower() in out_norm:
                continue
            out.append({"name": item, "origin": origin_hint})
            out_norm.add(item.lower())
        return out

    def _build_origin_hint(self, narration_text: str, action_text: str) -> str:
        source = (narration_text or action_text or "").strip()
        if not source:
            return ""
        first_sentence = re.split(r"(?<=[.!?])\s", source, maxsplit=1)[0]
        return first_sentence[:120]

    def _item_mentioned(self, item_name: str, text_lower: str) -> bool:
        item_l = item_name.lower()
        if item_l in text_lower:
            return True
        words = [
            word
            for word in re.findall(r"[a-z0-9]+", item_l)
            if len(word) > 2 and word not in self._ITEM_STOPWORDS
        ]
        if not words:
            return False
        return all(word in text_lower for word in words)

    def _sanitize_player_state_update(
        self,
        previous_state: Dict[str, object],
        update: Dict[str, object],
        action_text: str = "",
        narration_text: str = "",
    ) -> Dict[str, object]:
        if not isinstance(update, dict):
            return {}
        cleaned = dict(update)
        previous_inventory_rich = self._get_inventory_rich(previous_state)

        inventory_add = self._normalize_inventory_items(cleaned.pop("inventory_add", []))
        inventory_remove = self._normalize_inventory_items(cleaned.pop("inventory_remove", []))

        if "inventory" in cleaned:
            model_inventory = self._normalize_inventory_items(cleaned.pop("inventory", []))
            model_set = {name.lower() for name in model_inventory}
            current_names = [entry["name"] for entry in previous_inventory_rich]
            current_set = {name.lower() for name in current_names}
            for name in current_names:
                if name.lower() not in model_set and name.lower() not in {r.lower() for r in inventory_remove}:
                    inventory_remove.append(name)
            for name in model_inventory:
                if name.lower() not in current_set and name.lower() not in {a.lower() for a in inventory_add}:
                    inventory_add.append(name)

        current_norm = {entry["name"].lower() for entry in previous_inventory_rich}
        inventory_remove = [item for item in inventory_remove if item.lower() in current_norm]

        if len(inventory_add) > self.MAX_INVENTORY_CHANGES_PER_TURN:
            inventory_add = inventory_add[: self.MAX_INVENTORY_CHANGES_PER_TURN]
        if len(inventory_remove) > self.MAX_INVENTORY_CHANGES_PER_TURN:
            inventory_remove = inventory_remove[: self.MAX_INVENTORY_CHANGES_PER_TURN]

        origin_hint = self._build_origin_hint(narration_text, action_text)
        if inventory_add or inventory_remove:
            cleaned["inventory"] = self._apply_inventory_delta(
                previous_inventory_rich,
                inventory_add,
                inventory_remove,
                origin_hint=origin_hint,
            )
        else:
            cleaned["inventory"] = previous_inventory_rich

        for key in list(cleaned.keys()):
            if key != "inventory" and "inventory" in str(key).lower():
                cleaned.pop(key, None)

        new_location = cleaned.get("location")
        if new_location is not None:
            old_location = previous_state.get("location")
            if str(new_location).strip().lower() != str(old_location or "").strip().lower():
                if "room_description" not in cleaned:
                    cleaned["room_description"] = None
                if "room_title" not in cleaned:
                    cleaned["room_title"] = None
        return cleaned

    def _strip_inventory_from_narration(self, narration: str) -> str:
        if not narration:
            return ""
        kept_lines: List[str] = []
        for line in narration.splitlines():
            stripped = line.strip().lower()
            if any(stripped.startswith(prefix) for prefix in self._INVENTORY_LINE_PREFIXES):
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip()

    def _strip_inventory_mentions(self, text: str) -> str:
        if not text:
            return ""
        return self._strip_inventory_from_narration(text)

    def _scrub_inventory_from_state(self, value):
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                key_str = str(key).lower()
                if key_str == "inventory" or "inventory" in key_str:
                    continue
                cleaned[key] = self._scrub_inventory_from_state(item)
            return cleaned
        if isinstance(value, list):
            return [self._scrub_inventory_from_state(item) for item in value]
        return value

    def _copy_identity_fields(
        self,
        source_state: Dict[str, object],
        target_state: Dict[str, object],
    ) -> Dict[str, object]:
        if not isinstance(target_state, dict):
            target_state = {}
        if not isinstance(source_state, dict):
            return target_state
        for key in ("character_name", "persona"):
            value = source_state.get(key)
            if value:
                target_state[key] = value
        return target_state

    def _sanitize_campaign_name_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9 _-]", "", text)
        return text[:48]

    def _build_campaign_suggestion_text(self, namespace: str) -> str:
        existing = self.list_campaigns(namespace)
        names = [campaign.name for campaign in existing]
        if not names:
            return "No campaigns exist yet."
        sample = ", ".join(names[:8])
        return f"Existing campaigns: {sample}"

    def _apply_character_updates(
        self,
        existing: Dict[str, dict],
        updates: Dict[str, dict],
        on_rails: bool = False,
    ) -> Dict[str, dict]:
        if not isinstance(updates, dict):
            return existing
        for slug, fields in updates.items():
            if not isinstance(fields, dict):
                continue
            slug = str(slug).strip()
            if not slug:
                continue
            if slug in existing:
                for key, value in fields.items():
                    if key not in self.IMMUTABLE_CHARACTER_FIELDS:
                        existing[slug][key] = value
            else:
                if on_rails:
                    continue
                existing[slug] = dict(fields)
        return existing

    @staticmethod
    def _calendar_resolve_fire_day(
        current_day: int,
        current_hour: int,
        time_remaining: object,
        time_unit: object,
    ) -> int:
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
        if unit.startswith("hour"):
            fire_day = day + ((hour + remaining) // 24)
        else:
            fire_day = day + remaining
        return max(1, int(fire_day))

    @classmethod
    def _calendar_normalize_event(
        cls,
        event: object,
        *,
        current_day: int,
        current_hour: int,
    ) -> dict[str, object] | None:
        if not isinstance(event, dict):
            return None
        name = str(event.get("name") or "").strip()
        if not name:
            return None
        fire_day_raw = event.get("fire_day")
        if isinstance(fire_day_raw, (int, float)) and not isinstance(fire_day_raw, bool):
            fire_day = max(1, int(fire_day_raw))
        else:
            fire_day = cls._calendar_resolve_fire_day(
                current_day=current_day,
                current_hour=current_hour,
                time_remaining=event.get("time_remaining", 1),
                time_unit=event.get("time_unit", "days"),
            )
        normalized: dict[str, object] = {
            "name": name,
            "fire_day": fire_day,
            "description": str(event.get("description") or "")[:200],
        }
        for key in ("created_day", "created_hour"):
            raw = event.get(key)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                normalized[key] = int(raw)
        return normalized

    @classmethod
    def _calendar_for_prompt(
        cls,
        campaign_state: Dict[str, object],
    ) -> list[dict[str, object]]:
        game_time = campaign_state.get("game_time") if isinstance(campaign_state, dict) else {}
        if not isinstance(game_time, dict):
            game_time = {}
        current_day = cls._coerce_non_negative_int(game_time.get("day", 1), default=1) or 1
        current_hour = cls._coerce_non_negative_int(game_time.get("hour", 8), default=8)
        current_hour = min(23, max(0, current_hour))
        calendar = campaign_state.get("calendar") if isinstance(campaign_state, dict) else []
        if not isinstance(calendar, list):
            calendar = []
        entries: list[dict[str, object]] = []
        for raw in calendar:
            normalized = cls._calendar_normalize_event(
                raw,
                current_day=current_day,
                current_hour=current_hour,
            )
            if normalized is None:
                continue
            fire_day = int(normalized.get("fire_day", current_day))
            days_remaining = fire_day - current_day
            if days_remaining < 0:
                status = "overdue"
            elif days_remaining == 0:
                status = "today"
            elif days_remaining == 1:
                status = "imminent"
            else:
                status = "upcoming"
            view = dict(normalized)
            view["days_remaining"] = days_remaining
            view["status"] = status
            entries.append(view)
        entries.sort(key=lambda item: (int(item.get("fire_day", current_day)), str(item.get("name", "")).lower()))
        return entries

    @staticmethod
    def _calendar_reminder_text(calendar_entries: list[dict[str, object]]) -> str:
        if not calendar_entries:
            return "None"
        alerts = []
        for event in calendar_entries:
            days = int(event.get("days_remaining", 0))
            name = str(event.get("name", "Unknown"))
            fire_day = int(event.get("fire_day", 1))
            if days > 1:
                continue
            if days < 0:
                alerts.append(f"- OVERDUE: {name} (was Day {fire_day}; {abs(days)} day(s) overdue)")
            elif days == 0:
                alerts.append(f"- TODAY: {name} (fires on Day {fire_day})")
            else:
                alerts.append(f"- TOMORROW: {name} (fires on Day {fire_day})")
        return "\n".join(alerts) if alerts else "None"

    def _apply_calendar_update(
        self,
        campaign_state: Dict[str, object],
        calendar_update: dict,
    ) -> Dict[str, object]:
        """Process calendar add/remove ops and persist absolute fire_day entries."""
        if not isinstance(calendar_update, dict):
            return campaign_state
        calendar_raw = list(campaign_state.get("calendar") or [])
        game_time = campaign_state.get("game_time") or {}
        current_day = game_time.get("day", 1)
        current_hour = game_time.get("hour", 8)
        calendar = []
        for event in calendar_raw:
            normalized = self._calendar_normalize_event(
                event,
                current_day=int(current_day) if isinstance(current_day, (int, float)) else 1,
                current_hour=int(current_hour) if isinstance(current_hour, (int, float)) else 8,
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
                if isinstance(fire_day, (int, float)) and not isinstance(fire_day, bool):
                    resolved_fire_day = max(1, int(fire_day))
                else:
                    resolved_fire_day = self._calendar_resolve_fire_day(
                        current_day=int(current_day) if isinstance(current_day, (int, float)) else 1,
                        current_hour=int(current_hour) if isinstance(current_hour, (int, float)) else 8,
                        time_remaining=entry.get("time_remaining", 1),
                        time_unit=entry.get("time_unit", "days"),
                    )
                event = {
                    "name": name,
                    "fire_day": resolved_fire_day,
                    "created_day": current_day,
                    "created_hour": current_hour,
                    "description": str(entry.get("description") or "")[:200],
                }
                calendar.append(event)

        if isinstance(to_add, list):
            seen_names: set[str] = set()
            deduped = []
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

    def _build_characters_for_prompt(
        self,
        characters: Dict[str, dict],
        player_state: Dict[str, object],
        recent_text: str,
    ) -> list:
        if not characters:
            return []
        player_location = str(player_state.get("location") or "").strip().lower()
        recent_lower = recent_text.lower() if recent_text else ""

        nearby = []
        mentioned = []
        distant = []
        for slug, char in characters.items():
            char_location = str(char.get("location") or "").strip().lower()
            char_name = str(char.get("name") or slug).strip().lower()
            is_deceased = bool(char.get("deceased_reason"))

            if not is_deceased and player_location and char_location == player_location:
                entry = dict(char)
                entry["_slug"] = slug
                nearby.append(entry)
            elif char_name in recent_lower or slug in recent_lower:
                entry = {
                    "_slug": slug,
                    "name": char.get("name", slug),
                    "location": char.get("location"),
                    "current_status": char.get("current_status"),
                    "allegiance": char.get("allegiance"),
                }
                if is_deceased:
                    entry["deceased_reason"] = char.get("deceased_reason")
                mentioned.append(entry)
            else:
                entry = {"_slug": slug, "name": char.get("name", slug)}
                if is_deceased:
                    entry["deceased_reason"] = char.get("deceased_reason")
                else:
                    entry["location"] = char.get("location")
                distant.append(entry)

        result = nearby + mentioned + distant
        return result[: self.MAX_CHARACTERS_IN_PROMPT]

    def _fit_characters_to_budget(self, characters_list: list, max_chars: int) -> list:
        while characters_list:
            text = json.dumps(characters_list, ensure_ascii=True)
            if len(text) <= max_chars:
                return characters_list
            characters_list = characters_list[:-1]
        return []

    def _zork_log(self, section: str, body: str = "") -> None:
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(_ZORK_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(f"\n{'=' * 72}\n[{ts}] {section}\n{'=' * 72}\n")
                if body:
                    handle.write(body)
                    if not body.endswith("\n"):
                        handle.write("\n")
        except Exception:
            if body:
                self._logger.info("%s :: %s", section, body)
            else:
                self._logger.info("%s", section)

    async def _delete_context_message(self, ctx):
        try:
            if hasattr(ctx, "delete"):
                await ctx.delete()
                return
            if hasattr(ctx, "message") and hasattr(ctx.message, "delete"):
                await ctx.message.delete()
        except Exception:
            return

    def _get_context_message(self, ctx):
        if hasattr(ctx, "message"):
            return ctx.message
        if hasattr(ctx, "add_reaction"):
            return ctx
        return None

    async def _add_processing_reaction(self, ctx) -> bool:
        message = self._get_context_message(ctx)
        if message is None or not hasattr(message, "add_reaction"):
            return False
        try:
            await message.add_reaction(self.PROCESSING_EMOJI)
            return True
        except Exception:
            return False

    async def _remove_processing_reaction(self, ctx) -> bool:
        message = self._get_context_message(ctx)
        if message is None:
            return False
        try:
            if hasattr(message, "remove_reaction"):
                me = getattr(getattr(message, "guild", None), "me", None)
                if me is not None:
                    await message.remove_reaction(self.PROCESSING_EMOJI, me)
                    return True
            if hasattr(message, "clear_reaction"):
                await message.clear_reaction(self.PROCESSING_EMOJI)
                return True
        except Exception:
            return False
        return False

    def _extract_json(self, text: str) -> str | None:
        text = text.strip()
        if "```" in text:
            text = re.sub(r"```\w*", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def _is_tool_call(self, payload: dict[str, Any]) -> bool:
        return isinstance(payload, dict) and "tool_call" in payload and "narration" not in payload

    def _coerce_python_dict(self, text: str) -> dict[str, Any] | None:
        try:
            fixed = re.sub(r"\bnull\b", "None", text)
            fixed = re.sub(r"\btrue\b", "True", fixed)
            fixed = re.sub(r"\bfalse\b", "False", fixed)
            result = ast.literal_eval(fixed)
            if isinstance(result, dict):
                return result
        except Exception:
            return None
        return None

    def _parse_json_lenient(self, text: str) -> dict[str, Any]:
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError as exc:
            coerced = self._coerce_python_dict(text)
            if coerced is not None:
                return coerced
            if "Extra data" not in str(exc):
                raise
            merged: dict[str, Any] = {}
            decoder = json.JSONDecoder()
            idx = 0
            length = len(text)
            while idx < length:
                while idx < length and text[idx] in " \t\r\n":
                    idx += 1
                if idx >= length:
                    break
                try:
                    obj, end_idx = decoder.raw_decode(text, idx)
                    if isinstance(obj, dict):
                        merged.update(obj)
                    idx = end_idx
                except (json.JSONDecodeError, ValueError):
                    break
            if merged:
                return merged
            raise

    def _clean_response(self, response: str) -> str:
        if not response:
            return response
        json_text = self._extract_json(response)
        if json_text:
            return json_text
        return response.strip()

    def _extract_ascii_map(self, text: str) -> str:
        if not text:
            return ""
        lines: list[str] = []
        for line in text.splitlines():
            if "```" in line:
                continue
            lines.append(line.rstrip())
        return "\n".join(lines).strip()

    def _apply_state_update(self, state: dict[str, object], update: dict[str, object]) -> dict[str, object]:
        if not isinstance(update, dict):
            return state
        for key, value in update.items():
            if value is None:
                state.pop(key, None)
            elif isinstance(value, str) and value.strip().lower() in self._COMPLETED_VALUES:
                state.pop(key, None)
            else:
                state[key] = value
        return state

    def _build_rails_context(
        self,
        player_state: dict[str, object],
        party_snapshot: list[dict[str, object]],
    ) -> dict[str, object]:
        exits = player_state.get("exits")
        if not isinstance(exits, list):
            exits = []
        known_names = []
        for entry in party_snapshot:
            name = str(entry.get("name") or "").strip()
            if name:
                known_names.append(name)
        inventory_rich = self._get_inventory_rich(player_state)[:20]
        return {
            "room_title": player_state.get("room_title"),
            "room_summary": player_state.get("room_summary"),
            "location": player_state.get("location"),
            "exits": exits[:12],
            "inventory": inventory_rich,
            "known_characters": known_names[:12],
            "strict_action_shape": "one concrete action grounded in current room and items",
        }

    def build_prompt(
        self,
        campaign: Campaign,
        player: Player,
        action: str,
        turns: list[Turn],
        party_snapshot: list[dict[str, object]] | None = None,
        is_new_player: bool = False,
    ) -> tuple[str, str]:
        summary = self._strip_inventory_mentions(campaign.summary or "")
        summary = self._trim_text(summary, self.MAX_SUMMARY_CHARS)
        state = self.get_campaign_state(campaign)
        state = self._scrub_inventory_from_state(state)
        if "game_time" not in state:
            state["game_time"] = {
                "day": 1,
                "hour": 8,
                "minute": 0,
                "period": "morning",
                "date_label": "Day 1, Morning",
            }
            campaign.state_json = self._dump_json(state)
        guardrails_enabled = bool(state.get("guardrails_enabled", False))
        model_state = self._build_model_state(state)
        model_state = self._fit_state_to_budget(model_state, self.MAX_STATE_CHARS)
        attributes = self.get_player_attributes(player)
        player_state = self.get_player_state(player)
        if party_snapshot is None:
            party_snapshot = self._build_party_snapshot_for_prompt(campaign, player, player_state)

        player_state_prompt = self._build_player_state_for_prompt(player_state)
        total_points = self.total_points_for_level(player.level)
        spent = self.points_spent(attributes)
        player_card = {
            "level": player.level,
            "xp": player.xp,
            "points_total": total_points,
            "points_spent": spent,
            "attributes": attributes,
            "state": player_state_prompt,
        }

        player_names: Dict[str, str] = {}
        actor_ids = {turn.actor_id for turn in turns if turn.actor_id}
        if actor_ids:
            with self._session_factory() as session:
                rows = (
                    session.query(Player)
                    .filter(Player.campaign_id == campaign.id)
                    .filter(Player.actor_id.in_(actor_ids))
                    .all()
                )
                for row in rows:
                    state_row = parse_json_dict(row.state_json)
                    name = str(state_row.get("character_name") or "").strip()
                    if name:
                        player_names[row.actor_id] = name

        recent_lines: List[str] = []
        ooc_re = re.compile(r"^\s*\[OOC\b", re.IGNORECASE)
        error_phrases = (
            "a hollow silence answers",
            "the world shifts, but nothing clear emerges",
        )
        for turn in turns:
            content = (turn.content or "").strip()
            if not content:
                continue
            if turn.kind == "player":
                if ooc_re.match(content):
                    continue
                clipped = self._trim_text(content, self.MAX_TURN_CHARS)
                clipped = self._strip_inventory_mentions(clipped)
                name = player_names.get(turn.actor_id or "")
                if name:
                    label = f"PLAYER ({name.upper()})"
                else:
                    label = "PLAYER"
                recent_lines.append(f"{label}: {clipped}")
            elif turn.kind == "narrator":
                if content.lower() in error_phrases:
                    continue
                clipped_lines = []
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("⏰"):
                        continue
                    if stripped.lower().startswith("inventory:"):
                        continue
                    clipped_lines.append(line)
                clipped = "\n".join(clipped_lines).strip()
                if not clipped:
                    continue
                clipped = self._trim_text(clipped, self.MAX_TURN_CHARS)
                recent_lines.append(f"NARRATOR: {clipped}")
        recent_text = "\n".join(recent_lines) if recent_lines else "None"

        rails_context = self._build_rails_context(player_state, party_snapshot)
        characters = self.get_campaign_characters(campaign)
        characters_for_prompt = self._build_characters_for_prompt(characters, player_state, recent_text)
        characters_for_prompt = self._fit_characters_to_budget(characters_for_prompt, self.MAX_CHARACTERS_CHARS)
        story_context = self._build_story_context(state)
        on_rails = bool(state.get("on_rails", False))
        game_time = state.get("game_time", {})
        speed_mult = state.get("speed_multiplier", 1.0)
        calendar_for_prompt = self._calendar_for_prompt(state)
        calendar_reminders = self._calendar_reminder_text(calendar_for_prompt)

        active_name = str(player_state.get("character_name") or "").strip()
        action_label = f"PLAYER_ACTION ({active_name.upper()})" if active_name else "PLAYER_ACTION"

        user_prompt = (
            f"CAMPAIGN: {campaign.name}\n"
            f"PLAYER_ID: {player.actor_id}\n"
            f"IS_NEW_PLAYER: {str(is_new_player).lower()}\n"
            f"GUARDRAILS_ENABLED: {str(guardrails_enabled).lower()}\n"
            f"RAILS_CONTEXT: {self._dump_json(rails_context)}\n"
            f"WORLD_SUMMARY: {summary}\n"
            f"WORLD_STATE: {self._dump_json(model_state)}\n"
            f"CURRENT_GAME_TIME: {self._dump_json(game_time)}\n"
            f"SPEED_MULTIPLIER: {speed_mult}\n"
            f"CALENDAR: {self._dump_json(calendar_for_prompt)}\n"
            f"CALENDAR_REMINDERS:\n{calendar_reminders}\n"
        )
        if story_context:
            user_prompt += f"STORY_CONTEXT:\n{story_context}\n"
        user_prompt += (
            f"WORLD_CHARACTERS: {self._dump_json(characters_for_prompt)}\n"
            f"PLAYER_CARD: {self._dump_json(player_card)}\n"
            f"PARTY_SNAPSHOT: {self._dump_json(party_snapshot)}\n"
            f"RECENT_TURNS:\n{recent_text}\n"
            f"{action_label}: {action}\n"
        )
        system_prompt = self.SYSTEM_PROMPT
        if guardrails_enabled:
            system_prompt = f"{system_prompt}{self.GUARDRAILS_SYSTEM_PROMPT}"
        if on_rails:
            system_prompt = f"{system_prompt}{self.ON_RAILS_SYSTEM_PROMPT}"
        system_prompt = f"{system_prompt}{self.MEMORY_TOOL_PROMPT}"
        if state.get("timed_events_enabled", True):
            system_prompt = f"{system_prompt}{self.TIMER_TOOL_PROMPT}"
        if story_context:
            system_prompt = f"{system_prompt}{self.STORY_OUTLINE_TOOL_PROMPT}"
        system_prompt = f"{system_prompt}{self.CALENDAR_TOOL_PROMPT}"
        system_prompt = f"{system_prompt}{self.ROSTER_PROMPT}"
        return system_prompt, user_prompt

    async def generate_map(self, campaign_or_ctx, actor_id: str | None = None, command_prefix: str = "!") -> str:
        campaign_id: str | None = None
        resolved_actor_id: str | None = actor_id

        if actor_id is None and hasattr(campaign_or_ctx, "guild") and hasattr(campaign_or_ctx, "channel"):
            ctx = campaign_or_ctx
            guild_id = str(getattr(ctx.guild, "id", ""))
            channel_id = str(getattr(ctx.channel, "id", ""))
            if not guild_id or not channel_id:
                return "Map unavailable."
            channel = self.get_or_create_channel(guild_id, channel_id)
            if not channel.enabled:
                return f"Adventure mode is disabled in this channel. Run `{command_prefix}zork` to enable it."
            metadata = self._load_session_metadata(channel)
            active_campaign_id = metadata.get("active_campaign_id")
            if not active_campaign_id:
                _, campaign = self.enable_channel(guild_id, channel_id, str(getattr(ctx.author, "id", "")))
                campaign_id = campaign.id
            else:
                campaign_id = str(active_campaign_id)
            resolved_actor_id = str(getattr(ctx.author, "id", ""))
        else:
            campaign_id = str(campaign_or_ctx)
            if resolved_actor_id is None:
                return "Map unavailable."
            resolved_actor_id = str(resolved_actor_id)

        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            player = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == resolved_actor_id)
                .first()
            )
            turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .order_by(Turn.id.desc())
                .limit(self.MAX_RECENT_TURNS)
                .all()
            )
            turns.reverse()
            others = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .order_by(Player.actor_id.asc())
                .all()
            )

        if campaign is None or player is None:
            return "Map unavailable."
        player_state = self.get_player_state(player)
        room_summary = player_state.get("room_summary")
        room_title = player_state.get("room_title")
        exits = player_state.get("exits")
        if not room_summary and not room_title:
            return "No map data yet. Try `look` first."

        marker_data = self._assign_player_markers(others, resolved_actor_id)
        other_entries = []
        for entry in marker_data:
            other = entry["player"]
            other_state = self.get_player_state(other)
            other_room = other_state.get("room_summary") or other_state.get("room_title") or other_state.get("location")
            if not other_room:
                continue
            other_name = other_state.get("character_name") or f"Adventurer-{str(other.actor_id)[-4:]}"
            other_entries.append(
                {
                    "marker": entry["marker"],
                    "user_id": other.actor_id,
                    "character_name": other_name,
                    "room": other_room,
                    "party_status": other_state.get("party_status"),
                }
            )

        player_name = player_state.get("character_name") or f"Adventurer-{str(resolved_actor_id)[-4:]}"
        campaign_state = self.get_campaign_state(campaign)
        model_state = self._build_model_state(campaign_state)
        model_state = self._fit_state_to_budget(model_state, 800)
        landmarks = campaign_state.get("landmarks", [])
        landmarks_text = ", ".join(landmarks) if isinstance(landmarks, list) and landmarks else "none"

        characters = self.get_campaign_characters(campaign)
        char_entries = []
        if isinstance(characters, dict):
            for slug, info in list(characters.items())[:20]:
                if not isinstance(info, dict):
                    continue
                if info.get("deceased_reason"):
                    continue
                char_name = info.get("name", slug)
                char_loc = info.get("location", "unknown")
                char_entries.append(f"{char_name} ({char_loc})")
        chars_text = ", ".join(char_entries) if char_entries else "none"

        story_progress = ""
        outline = campaign_state.get("story_outline")
        if isinstance(outline, dict):
            chapters = outline.get("chapters", [])
            try:
                cur_ch = int(campaign_state.get("current_chapter", 0))
            except (ValueError, TypeError):
                cur_ch = 0
            try:
                cur_sc = int(campaign_state.get("current_scene", 0))
            except (ValueError, TypeError):
                cur_sc = 0
            if isinstance(chapters, list) and 0 <= cur_ch < len(chapters):
                chapter = chapters[cur_ch]
                chapter_title = chapter.get("title", "")
                scenes = chapter.get("scenes", [])
                scene_title = ""
                if isinstance(scenes, list) and 0 <= cur_sc < len(scenes):
                    scene_title = scenes[cur_sc].get("title", "")
                story_progress = f"{chapter_title} / {scene_title}" if scene_title else chapter_title

        map_prompt = (
            f"CAMPAIGN: {campaign.name}\n"
            f"PLAYER_NAME: {player_name}\n"
            f"PLAYER_ROOM_TITLE: {room_title or 'Unknown'}\n"
            f"PLAYER_ROOM_SUMMARY: {room_summary or ''}\n"
            f"PLAYER_EXITS: {exits or []}\n"
            f"WORLD_SUMMARY: {self._trim_text(campaign.summary or '', 1200)}\n"
            f"WORLD_STATE: {self._dump_json(model_state)}\n"
            f"LANDMARKS: {landmarks_text}\n"
            f"WORLD_CHARACTERS: {chars_text}\n"
        )
        if story_progress:
            map_prompt += f"STORY_PROGRESS: {story_progress}\n"
        map_prompt += (
            f"OTHER_PLAYERS: {self._dump_json(other_entries)}\n"
            "Draw a compact map with @ marking the player's location.\n"
        )

        if self._map_completion_port is None:
            return "Map unavailable."
        response = await self._map_completion_port.complete(
            self.MAP_SYSTEM_PROMPT,
            map_prompt,
            temperature=0.2,
            max_tokens=600,
        )
        ascii_map = self._extract_ascii_map(response or "")
        if not ascii_map:
            return "Map is foggy. Try again."
        return ascii_map

    # ------------------------------------------------------------------
    # Memory visibility compatibility
    # ------------------------------------------------------------------

    def filter_memory_hits_by_visibility(self, campaign_id: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._engine.filter_memory_hits_by_visibility(campaign_id, hits)
