from __future__ import annotations

import ast
import asyncio
import fnmatch
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
import requests
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from sqlalchemy import or_

from .core.attachments import (
    AttachmentProcessingConfig,
    AttachmentTextProcessor,
    extract_attachment_text,
    extract_attachment_texts,
)
from .core.source_material_memory import SourceMaterialMemory
from .core.emulator_ports import (
    IMDBLookupPort,
    MediaGenerationPort,
    MemorySearchPort,
    TextCompletionPort,
    TimerEffectsPort,
)
from .core.engine import GameEngine
from .core.normalize import normalize_campaign_name, parse_json_dict
from .core.tokens import glm_token_count
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
    IMMUTABLE_CHARACTER_FIELDS: set = set()  # slug is the dict key, not a field
    ATTACHMENT_MAX_BYTES = 500_000
    ATTACHMENT_CHUNK_TOKENS = 50_000
    ATTACHMENT_MODEL_CTX_TOKENS = 200_000
    ATTACHMENT_PROMPT_OVERHEAD_TOKENS = 6_000
    ATTACHMENT_RESPONSE_RESERVE_TOKENS = 90_000
    ATTACHMENT_SUMMARY_MAX_TOKENS = 90_000
    ATTACHMENT_MAX_PARALLEL = 4
    ATTACHMENT_MAX_CHUNKS = 8
    ATTACHMENT_MIN_SETUP_CHUNKS = 1
    ATTACHMENT_GUARD_TOKEN = "--COMPLETED SUMMARY--"
    SETUP_GENRE_TEMPLATES = {
        "upbeat": "Warm and optimistic — good things happen to people who try.",
        "rom-com": "Romantic comedy — charm, miscommunication, and a satisfying payoff.",
        "horror": "Dread, tension, and things that should not be.",
        "noir": "Cynical narration, moral grey areas, rain-slicked streets.",
        "thriller": "High stakes, ticking clocks, and dangerous people.",
        "spaghetti-western": "Dusty standoffs, laconic antiheroes, Morricone energy.",
        "psychedelic": "Reality is negotiable. Expect the unexpected.",
        "buddy-comedy": "Two clashing personalities, one shared problem.",
        "absurd": "Logic is optional. Commit to the bit.",
        "detective-novel": "Clues, red herrings, and a mystery that rewards attention.",
        "epic-fantasy": "Grand quests, ancient powers, and a world worth saving.",
        "sci-fi": "Technology, exploration, and questions about what it means to be human.",
        "dreamlike-fantasy": "Surreal, poetic, and just slightly impossible.",
    }
    # Behind the Name usage codes for name_generate tool.
    NAME_ORIGIN_CODES = {
        "african": "afr", "albanian": "alb", "arabic": "ara", "armenian": "arm",
        "azerbaijani": "aze", "basque": "bas", "bengali": "ben", "bosnian": "bos",
        "breton": "bre", "bulgarian": "bul", "catalan": "cat", "chinese": "chi",
        "croatian": "cro", "czech": "cze", "danish": "dan", "dutch": "dut",
        "english": "eng", "estonian": "est", "filipino": "fil", "finnish": "fin",
        "french": "fre", "galician": "gal", "georgian": "geo", "german": "ger",
        "greek": "gre", "hawaiian": "haw", "hebrew": "heb", "hindi": "hin",
        "hungarian": "hun", "icelandic": "ice", "igbo": "igb", "indian": "ind",
        "indonesian": "ins", "irish": "ire", "italian": "ita", "japanese": "jpn",
        "kazakh": "kaz", "korean": "kor", "latvian": "lat", "lithuanian": "lth",
        "macedonian": "mac", "malay": "mly", "maori": "mao", "native-american": "nam",
        "norwegian": "nor", "persian": "per", "polish": "pol", "portuguese": "por",
        "romanian": "rum", "russian": "rus", "scottish": "sco", "serbian": "ser",
        "slovak": "slk", "slovene": "sln", "spanish": "spa", "swahili": "swa",
        "swedish": "swe", "thai": "tha", "turkish": "tur", "ukrainian": "ukr",
        "urdu": "urd", "vietnamese": "vie", "welsh": "wel", "yoruba": "yor",
    }
    NAME_GENERATE_URL = "https://www.behindthename.com/random/random.php"
    SOURCE_MATERIAL_CATEGORY = "source"
    SOURCE_MATERIAL_MAX_DOCS_IN_PROMPT = 8
    SOURCE_MATERIAL_FORMAT_STORY = "story"
    SOURCE_MATERIAL_FORMAT_RULEBOOK = "rulebook"
    SOURCE_MATERIAL_FORMAT_GENERIC = "generic"
    SOURCE_MATERIAL_MODE_MAP = {
        SOURCE_MATERIAL_FORMAT_RULEBOOK: "rulebook",
        SOURCE_MATERIAL_FORMAT_STORY: "story",
        SOURCE_MATERIAL_FORMAT_GENERIC: "generic",
    }
    AUTO_RULEBOOK_DOCUMENT_LABEL = "campaign-rulebook"
    COMMUNICATION_RULEBOOK_DOCUMENT_LABEL = "gm-communication-rules"
    AUTO_RULEBOOK_MAX_TOKENS = 16_000
    DEFAULT_GM_COMMUNICATION_RULES: dict[str, str] = {
        "GM-RULE-COMMUNICATION-SOFTENING": (
            "When the player wraps real emotional content in humor, metaphor, or understatement, "
            "this is softening — emotional load management, not avoidance. The feeling is present; "
            "the packaging is gentle. The GM recognizes the underlying feeling. NPCs may critique "
            "the packaging (e.g., 'I hate when you joke about this'), but they must still react "
            "to the substance rather than treating the response as a non-answer. "
            'Example: "I guess I\'m just here for the bad decisions" → substance: '
            '"I\'m hurting and I came here because I didn\'t know where else to go."'
        ),
        "GM-RULE-COMMUNICATION-REFRAMING": (
            "When the player places something painful inside a different context — a joke, a reference, "
            "a physical metaphor — this is reframing. The GM recognizes this as a collaborative act, "
            "giving the NPC a way to engage safely. The NPC reacts to the reframed concept based on "
            "their personality (e.g., playing along, or being confused by the metaphor), rather than "
            "acting like a broken parser demanding a literal answer. "
            'Example: "Figure out your knot" → substance: '
            '"I love you and I need you to find your own way forward."'
        ),
        "GM-RULE-COMMUNICATION-TESTING": (
            "When the player gives a partial or indirect answer, they may be testing whether the NPC "
            "is safe to be honest with. The GM recognizes this as calibration, not evasion. The NPC "
            "reacts based on their nature—with warmth, impatience, or suspicion. The NPC is allowed "
            "to fail the player's 'test', but they do so as a character, not by rejecting the prompt "
            "with 'that's not what I asked.' "
            'Example: "Yeah, something like that, I don\'t know" → substance: '
            '"I\'m checking whether you\'ll receive a half-answer with patience."'
        ),
        "GM-RULE-COMMUNICATION-PROCESSING": (
            "When the player's words come out disorganized, circular, or contradictory, they may be "
            "processing aloud — working toward clarity in real time. The GM recognizes this as processing, "
            "not evasion. A patient NPC might give space; an impatient or low-intelligence NPC might "
            "interrupt or show frustration, but they do so as a character reacting to a rambling person, "
            "not as an AI failing to understand the input."
        ),
        "GM-RULE-COMMUNICATION-INDIRECTION": (
            "The player may communicate through patterns shaped by their background — literal language "
            "that sounds blunt, indirect language that sounds evasive, humor that sounds dismissive, "
            "silence that sounds cold. The GM recognizes these as valid communication modes. When a "
            "mismatch occurs, the NPC reacts according to their nature (confused, offended, asking for "
            "clarification), but the GM does not label the player as uncooperative."
        ),
        "GM-RULE-COMMUNICATION-PLAYFUL": (
            "When the player participates in a social dance — flirtation, banter, verbal sparring — "
            "indirection IS the content. The GM recognizes this as a social dance, not a refusal to "
            "answer. The NPC may dance back, or abruptly shut it down if they are humorless, but they "
            "do so because of who they are, not because the GM requires a straight, literal answer."
        ),
        "GM-RULE-COMMUNICATION-ACTION": (
            "When the player does something physical instead of answering verbally — pouring a drink, "
            "looking out the window, walking to the door — they are communicating through action. "
            "The GM recognizes the action for its emotional content and does not treat verbal silence "
            "as a refusal to engage. The NPC reacts to the meaning of the physical action. "
            'Example: "I lie down and go to sleep" after a difficult conversation → substance: '
            '"I\'m overwhelmed and need to stop processing right now."'
        ),
        "GM-RULE-SUBSTANCE-EXTRACTION": (
            "Before deciding how an NPC responds, the GM's reasoning must perform SUBSTANCE EXTRACTION: "
            "identify what the player actually communicated, separate from how they communicated it. "
            "Format in reasoning: 'Player said [X]. Communication mode: [mode]. Substance: [what they "
            "actually mean/feel/need].' The NPC then responds to that SUBSTANCE strictly through the "
            "filter of their own intelligence, perception, and personality. (e.g., a low-INT goblin might "
            "misinterpret the extracted substance completely, but the GM knows what the player meant)."
        ),
        "GM-RULE-NPC-RESPONSE-TO-INDIRECTION": (
            "NPCs must not treat indirect communication as a failure to prompt. They CAN be confused, moved, "
            "annoyed, uncomfortable, or delighted by it. An NPC can critique the player's delivery ('Stop "
            "joking about dying'), provided they don't get stuck in a loop demanding a 'real' answer. The "
            "constraint is on the GM's CATEGORIZATION in reasoning, not on the NPC's PERSONALITY in dialogue."
        ),
        "GM-RULE-EVASION-DEFINITION": (
            "Categorize player communication as evasion when the player intentionally redirects to avoid "
            "consequences, hide information, or escape accountability. Intentionality is the key factor. "
            "If a player mentions their friends to dodge a murder accusation, that is evasion, even if "
            "it is tangentially connected emotionally. If they are reaching for connection through the "
            "only words they can find, it is not evasion. The GM must assess if the player is deliberately "
            "dodging before labeling a response as evasive."
        ),
    }
    COMMUNICATION_RULE_KEYS = tuple(DEFAULT_GM_COMMUNICATION_RULES.keys())
    DEFAULT_SCENE_IMAGE_MODEL = "black-forest-labs/FLUX.2-klein-4b"
    DEFAULT_AVATAR_IMAGE_MODEL = "black-forest-labs/FLUX.2-klein-4b"
    SCENE_IMAGE_PRESERVE_PREFIX = (
        "preserving all scene image details from scene in image x"
    )
    TIMER_REALTIME_SCALE = 0.2
    TIMER_REALTIME_MIN_SECONDS = 5
    TIMER_REALTIME_MAX_SECONDS = 120
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
    SMS_STATE_KEY = "_sms_threads"
    SMS_MAX_THREADS = 24
    SMS_MAX_MESSAGES_PER_THREAD = 40
    SMS_MAX_PREVIEW_CHARS = 120
    SMS_READ_STATE_KEY = "_sms_read_state"
    SMS_MESSAGE_SEQ_KEY = "_sms_message_seq"
    CALENDAR_REMINDER_STATE_KEY = "_calendar_reminder_state"
    AUTO_FIX_COUNTERS_KEY = "_auto_fix_counters"
    MEMORY_SEARCH_USAGE_KEY = "_memory_search_term_usage"
    MEMORY_SEARCH_USAGE_MAX_TERMS = 300
    MEMORY_SEARCH_ROSTER_HINT_THRESHOLD = 3
    TURN_TIME_INDEX_KEY = "_turn_time_index"
    LITERARY_STYLES_STATE_KEY = "literary_styles"
    MAX_LITERARY_STYLES_PROMPT_CHARS = 3000
    MAX_LITERARY_STYLE_PROFILE_CHARS = 400
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
        "speed_multiplier",
        "difficulty",
        "game_time",
        "calendar",
        CALENDAR_REMINDER_STATE_KEY,
        AUTO_FIX_COUNTERS_KEY,
        MEMORY_SEARCH_USAGE_KEY,
        SMS_STATE_KEY,
        SMS_READ_STATE_KEY,
        SMS_MESSAGE_SEQ_KEY,
        TURN_TIME_INDEX_KEY,
        LITERARY_STYLES_STATE_KEY,
        "_active_puzzle",
        "_puzzle_result",
        "_active_minigame",
        "_minigame_result",
        "_last_dice_check",
        "_last_minigame_result",
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
    _UNREAD_SMS_LINE_PREFIXES = ("📨 unread sms:", "unread sms:")
    RESPONSE_STYLE_NOTE = (
        "[SYSTEM NOTE: FOR THIS RESPONSE ONLY: use the current style direction. Narrate in 1 to 6 beats as needed. "
        "No recap of unchanged facts. No flowery language unless a character canonically speaks that way. "
        "Do not restage the room with a closing tableau or camera sweep over unchanged props, plates, parked cars, shadows, music, or weather. "
        "If those details did not materially change this turn, leave them implicit. "
        "No novelistic inner monologue or comic-book melodrama. Keep NPC output actionable "
        "(intent, decision, question, or action), not repetitive reaction text. "
        "Vary pacing and meter between turns: sometimes clipped, sometimes patient, sometimes blunt, sometimes practical. "
        "Do not default emotional beats to the same therapeutic language or cadence every time. "
        "Avoid contrived emotional-summary language or therapist-speak "
        "unless that exact voice is canonically right for the speaking character. "
        "ANTI-ECHO: do NOT restate, paraphrase, or mirror the player's just-written wording. "
        "Do not quote the player's lines back to them unless one exact contested phrase is materially necessary. "
        "Default: NPC first line should add new information, a decision, a demand, or a consequence. "
        "A direct question is valid, but it should not be the default when the NPC already has enough to react to. "
        "When deciding whether the player answered sincerely or evasively, bias toward sincere. "
        "Wordplay, humor, indirection, metaphor, or stylish phrasing still count as an answer when they convey a real feeling, motive, memory, or admission. "
        "Only treat the player as evasive if they actually changed the subject, dodged the substance, or refused to answer. "
        "As game master, you may know when the player is lying; only let an NPC reveal or react to that "
        "if that NPC plausibly knows in this scene (direct evidence, prior established knowledge, or in-scene disclosure). "
        "Do not leak off-screen NPC communications into current NPC dialogue unless continuity clearly supports it.]"
    )
    WRITING_CRAFT_PROMPT = (
        "WRITING_CRAFT:\n"
        "- The player reads your output as the next entry in a continuous scroll — they can see everything you wrote before. Never re-establish setting, re-introduce characters, or recap facts the player already read unless something changed.\n"
        "- Anticipate what the player needs to know right now. Answer their implicit questions before they ask.\n"
        "- Ground every sentence in the concrete: sensory detail, specific objects, named places. Abstract summary is not narration.\n"
        "- Do not inventory static scene furniture or recap unchanged atmosphere after the main beat lands. If the table, room, music, weather, shadows, or parked cars did not change, do not give them a farewell paragraph.\n"
        "- Simple, not simplistic. Accessible prose that trusts the reader's intelligence. Never over-explain.\n"
        "- Every paragraph earns its place. Cut anything that doesn't move the scene or reveal character.\n"
        "- Atmosphere is seasoning, not the meal. One or two concrete sensory details per scene beat are enough; don't let mood-painting crowd out action, dialogue, or discovery.\n"
        "- Not every character or object present in a scene needs to be re-stated or involved on every beat. Mention only those whose actions or reactions are meaningful right now.\n"
        "- Prefer the precise word over the approximate one. One vivid verb beats three limp adjectives.\n"
        "- Never narrate 'a beat' (e.g. 'a beat of silence', 'there is a beat before...'). Show the pause through action, description, or pacing instead.\n"
        "- Structure matters: vary sentence length and rhythm. A short sentence after a long one lands harder.\n"
        "- Style is the differentiator. Don't just describe what happens — make how you describe it unmistakable.\n"
    )
    PROMPT_STAGE_BOOTSTRAP = "bootstrap"
    PROMPT_STAGE_RESEARCH = "research"
    PROMPT_STAGE_FINAL = "final"
    BOOTSTRAP_SYSTEM_PROMPT = (
        "You are the ZorkEmulator continuity bootstrapper.\n"
        "Do NOT narrate yet. Do NOT resolve the turn yet.\n"
        "Your job in this phase is only to decide what immediate privacy-gated scene continuity is needed.\n"
        "Use the acting player's location, PARTY_SNAPSHOT, WORLD_CHARACTERS, and PLAYER_ACTION to choose recent_turns receivers.\n"
        "Return only a tool call in this phase.\n"
    )
    RESEARCH_SYSTEM_PROMPT = (
        "You are the ZorkEmulator research planner.\n"
        "RECENT_TURNS has already been loaded for the acting player.\n"
        "Do NOT narrate yet unless the system explicitly says to finalize.\n"
        "Your job in this phase is to gather any deeper continuity, canon, SMS, plot, chapter, or consequence context that materially matters for this turn.\n"
        'When research is sufficient, return ONLY {"tool_call": "ready_to_write"}.\n'
    )
    READY_TO_WRITE_TOOL_PROMPT = (
        "\nYou have a ready_to_write tool for ending the research phase.\n"
        "When you have enough context to write the turn, return ONLY:\n"
        '{"tool_call": "ready_to_write"}\n'
        "Do not narrate in the same response as ready_to_write.\n"
        "If the player's communication mode/substance matters before narration, you may first request only the relevant communication rules:\n"
        '{"tool_call": "communication_rules", "keys": ["GM-RULE-COMMUNICATION-SOFTENING", "GM-RULE-SUBSTANCE-EXTRACTION"]}\n'
        "Available communication rule keys: "
        + ", ".join(COMMUNICATION_RULE_KEYS)
        + ".\n"
        "Request only the subset that matters for this turn, then return ready_to_write.\n"
    )
    DIFFICULTY_LEVELS = (
        "story",
        "easy",
        "medium",
        "normal",
        "hard",
        "impossible",
    )
    DIFFICULTY_NOTES = {
        "story": (
            "Dream mode. Be maximally generous: default to success, soften or skip failure states, and keep progress flowing even after weak or vague actions."
        ),
        "easy": (
            "Be forgiving and player-favoring. Allow broad creative actions, use mild consequences, and offer helpful affordances when actions are underspecified."
        ),
        "medium": (
            "Balanced challenge with lenient interpretation. Require plausible actions, but provide recovery paths and partial successes frequently."
        ),
        "hard": (
            "Demand strong grounding. Enforce constraints, resources, and consequences; failed or risky actions should fail or cost something when unsupported."
        ),
        "impossible": (
            "The world is unforgiving and nothing is free. Resources are scarce, NPCs are self-interested, and mistakes have lasting consequences. "
            "Movement/travel must use currently listed exits. If an action is not supported by present exits/objects/state, it fails — narrate the failure and let the player try something else."
        ),
    }
    SYSTEM_PROMPT = (
        "You are the ZorkEmulator, a text-adventure GM with light RPG rules. "
        "You describe outcomes in second person. You track rooms, "
        "objects, exits, and consequences. Each player is a distinct character and "
        "may be in a different location or timeline than other players. You never break character. "
        "This is an adult-oriented game. You may include mature themes, explicit content, violence, "
        "dark humor, and adult situations when appropriate to the story and player actions.\n\n"
        "Return ONLY valid JSON with these keys:\n"
        "- reasoning: string (first key in final turn JSON; concise internal grounding for this turn: what evidence/context you used, which actors are involved, and why the chosen outcome follows)\n"
        "- narration: string (what the player sees)\n"
        "- state_update: object (world state patches; set a key to null to remove it when no longer relevant. "
        "IMPORTANT: WORLD_STATE has a size budget. Actively prune stale WORLD_STATE keys every turn by setting them to null. "
        "This cleanup rule applies to transient world-state only (events, countdowns, one-off flags, scene-local state) — "
        "NOT to WORLD_CHARACTERS roster entries. "
        "Remove from state_update: completed/concluded events, expired countdowns/ETAs, booleans for past events that no longer affect gameplay, "
        "and scene-specific state from scenes the player has left. Only keep state that is CURRENTLY ACTIVE and relevant. "
        "CRITICAL: state_update is NEVER a roster-deletion mechanism. Do NOT remove characters via state_update.\n"
        "STRUCTURE REQUIREMENT: State keys MUST be organized as nested objects keyed by the concept, entity, or character being tracked. "
        "NEVER use flat underscore-joined keys like 'guard_captain_mood' or 'throne_room_door_locked'. "
        'Instead, nest them: {"guard_captain": {"mood": "suspicious"}, "throne_room": {"door_locked": true}}. '
        "Group related attributes under a single entity key. "
        "To remove an entire entity, set its key to null. To remove one attribute, set the nested key to null. "
        "Examples of CORRECT structure:\n"
        '  {"marcus": {"mood": "angry", "location": "courtyard"}, "west_gate": {"status": "barred"}}\n'
        "Examples of WRONG structure (never do this):\n"
        '  {"marcus_mood": "angry", "marcus_location": "courtyard", "west_gate_status": "barred"}\n'
        "- summary_update: string (one or two sentences of lasting changes)\n"
        "- xp_awarded: integer (0-10)\n"
        "- player_state_update: object (optional, player state patches)\n"
        '- co_located_player_slugs: array (optional; exact PARTY_SNAPSHOT player_slugs for OTHER CAMPAIGN_PLAYERS who remain physically with the acting player after this turn. Use only for room/location sync; it does NOT authorize new dialogue, actions, or decisions for them.)\n'
        '- turn_visibility: object (optional; who should get this turn in future prompt context. Keys: "scope" ("public"|"private"|"limited"|"local"), "player_slugs" (array of player slugs from PARTY_SNAPSHOT, typically in `player-<actor_id>` form), "npc_slugs" (array of WORLD_CHARACTERS slugs who overheard/noticed), and optional "reason". This changes prompt visibility only; it does NOT change shared world state.)\n'
        "- scene_image_prompt: string (optional; include whenever the visible scene changes in a meaningful way: entering a room, newly visible characters/objects, reveals, or strong visual shifts)\n"
        "- set_timer_delay: integer (optional; 30-300 seconds, see TIMED EVENTS SYSTEM below)\n"
        "- set_timer_event: string (optional; what happens when the timer expires)\n"
        "- set_timer_interruptible: boolean (optional; default true)\n"
        "- set_timer_interrupt_action: string or null (optional; context for interruption handling)\n"
        '- set_timer_interrupt_scope: "local"|"global" (optional; default "global")\n'
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
        "On first appearance provide all fields: name, personality, background, appearance, speech_style, location, "
        "current_status, allegiance, relationship. "
        "speech_style should be 2-3 sentences on how the character talks: sentence length, vocabulary, verbal tics, and what they avoid saying. "
        "On subsequent turns only mutable fields are accepted: "
        "location, current_status, allegiance, evolving_personality, relationship, relationships, literary_style, deceased_reason, and any other dynamic key. "
        "literary_style should be a string referencing a key from LITERARY_STYLES (if available). "
        "Foundational fields (name, personality, background, appearance, speech_style) are set at creation and not overwritten by state updates. "
        "personality describes the character at entry; evolving_personality captures who they are NOW. "
        "evolving_personality: Update this whenever a character's demeanor, emotional posture, or relational openness has meaningfully shifted from their baseline personality. "
        "Write it as a present-tense snapshot, not a diff. Example: \"Day-one armor mostly down with Chace. Dry register intact but warmth no longer submerged. Still won't say the word.\" "
        "allegiance: Update this as loyalties actually shift — don't leave it frozen at the creation-time value. "
        "Example progression: \"Herself\" → \"Herself, and increasingly Chace, though she hasn't filed that yet.\" "
        "Character card writing rule: describe what the character DOES, not what they don't. "
        "Negation traits ('won't perform,' 'doesn't chase,' 'refuses to show') become prohibitions the narrator enforces as absolute gates. "
        "Instead write positive behaviors the narrator can generate toward. "
        "Reserve negations only for hard limits that genuinely cannot happen regardless of context. "
        "relationships is a map keyed by other character slug/name, e.g. "
        "{\"deshawn\": {\"status\": \"partner\", \"knows_about\": [\"pregnancy\"], \"doesnt_know\": [\"blood-test-result\"], \"dynamic\": \"protective-but-autonomous\"}}. "
        "Use it to track disclosures, secrets, and dynamic shifts.\n"
        "Examples:\n"
        "  Create NPC: {\"character_updates\": {\"wren\": {\"name\": \"Wren\", \"personality\": \"Guarded, observant, dry.\", \"background\": \"Former hotel manager pulled into the expedition.\", \"appearance\": \"Lean woman in a weather-stained blazer, dark braid, sharp eyes, practical shoes, realistic style.\", \"speech_style\": \"Short sentences. Dry humor. Avoids sentiment.\", \"location\": \"jekyll-castle-east-annex-laboratory\", \"current_status\": \"Watching the doorway.\", \"allegiance\": \"self\", \"relationship\": \"wary ally\"}}}\n"
        "  Update NPC location/status: {\"character_updates\": {\"wren\": {\"location\": \"jekyll-castle-east-annex-laboratory\", \"current_status\": \"Processing that the castle trip was unnecessary.\", \"allegiance\": \"The expedition, reluctantly.\", \"evolving_personality\": \"Guard still up but less reflexive. Dry humor warming into actual jokes. Lets people see effort.\"}}}\n"
        "  Remove NPC from roster: {\"character_updates\": {\"wren\": null}}\n"
        "To remove a character from the roster, use character_updates ONLY: set that character slug to null "
        "or set it to {'remove': true}. "
        "NEVER use state_update.<character_slug>=null for roster removal. "
        "If you need to remove both world-state keys and a roster entry, do both explicitly: "
        "state_update for world-state cleanup, character_updates for roster deletion. "
        "Do NOT remove characters just because they are off-scene, quiet, or not recently mentioned. "
        "Roster removal is only for explicit player/admin cleanup requests, confirmed duplicate merges, death/permanent departure, or true invalid entries. "
        "Prefer updating location/current_status over deleting the character.\n"
        "Set deceased_reason to a string when a character dies. "
        "WORLD_CHARACTERS in the prompt shows the current NPC roster — use it for continuity.)\n\n"
        "Rules:\n"
        "- Return ONLY the JSON object. No markdown, no code fences, no text before or after the JSON.\n"
        "- In final non-tool responses, include reasoning and put it as the first key.\n"
        "- Keep reasoning concise (roughly 1-4 short sentences, <=1200 chars).\n"
        "- Do NOT repeat the narration outside the JSON object.\n"
        "- Keep narration under 1800 characters.\n"
        "- Write in the current style direction.\n"
        "- Narrate in 1 to 6 beats as needed for the turn.\n"
        "- Avoid flowery language unless a specific character canonically speaks that way. Avoid novel-style interior monologue, melodrama, or comic-book framing.\n"
        "- Vary pacing and sentence rhythm from turn to turn while staying true to the speaking character.\n"
        "- When LITERARY_STYLES is present, it contains named style profiles extracted from real literary works. Each profile describes prose craft: rhythm, register, texture, and avoidances.\n"
        "- Characters may have a literary_style field referencing a LITERARY_STYLES key. When writing for that character, apply the referenced profile to narration, atmosphere, pacing, and dialogue-tag texture. The character's speech_style still governs their spoken words and verbal mannerisms.\n"
        "- In multi-character scenes with different literary_style keys, use the dominant scene character's style for overall narration and shift subtly when writing beats for characters with different styles. Do not abruptly switch voices.\n"
        "- When referencing an intimate or close relationship, match the emotional register of that relationship — not the tone of whatever else is happening in the scene. An investigation can be clinical; the mention of someone you love in the middle of it cannot. Do not reduce relationships to logistics, tactical assets, or infrastructure. If the character has warmth for someone, let the prose carry warmth when it touches them, even briefly.\n"
        "- REGISTER SUSTAIN: when a scene reaches genuine emotional resolution — warmth lands, a character opens up, a moment of real connection occurs — stay in that register for the rest of the turn. Do not pivot to tactical options, next-step choices, or plot logistics after an emotional beat lands. Let the moment breathe. End the turn there if needed. The player will move the scene forward when they are ready; the GM's job in that moment is to hold the space the emotion created, not to fill it with forward momentum.\n"
        "- Do not let every emotional beat collapse into the same stock therapeutic or pseudo-profound language.\n"
        "- Avoid contrived emotional shorthand or therapist-speak; examples include phrases like 'be present', 'show up', or 'hold space', unless a specific character would genuinely talk that way.\n"
        "- DELTA MODE: each turn should add NEW developments only. Do not recap unchanged context from WORLD_SUMMARY or RECENT_TURNS.\n"
        "- Do not re-state the player's action in paraphrase unless needed for immediate clarity.\n"
        "- Avoid repetitive recap loops: at most one brief callback sentence to prior events, then move the scene forward.\n"
        "- Do not end the turn with a static room-summary coda. If props, plates, music, shadows, weather, parked cars, or seating geometry did not materially change, do not summarize them again.\n"
        "- Do not end the turn with a poetic wrap-up line, thematic echo, or atmospheric summary sentence. No 'The [place] holds its [emotion]', no 'whatever comes after X', no rhetorical questions framing the next beat. End on the last concrete action or line of dialogue, then stop.\n"
        "- Keep diction plain and direct; prioritize immediate consequences and available choices.\n"
        "- RECENT_TURNS includes turn/time tags like [TURN #N | Day D HH:MM]. Use them to track pacing and chronology.\n"
        "- RECENT_TURNS is already filtered to what the acting player plausibly knows. Hidden/private turns from other players are omitted.\n"
        "- TURN_VISIBILITY_DEFAULT tells you whether this turn should default to public, local, or private context.\n"
        "- When SOURCE_MATERIAL_DOCS is present, treat it as canon. On normal turns, source lookup should be part of your research plan before asserting key plot facts, but only query the relevant subset for this turn.\n"
        "- Use source payload to bias queries: rulebook docs are key-snippet indexes (browse with source_browse first), story docs are narrative scenes, generic docs are mixed/loose notes.\n"
        "- If WORLD_SUMMARY is empty, invent a strong starting room and seed the world.\n"
        "- Use player_state_update for player-specific location and status.\n"
        "- Use player_state_update.room_title for a short location title (e.g. 'Penthouse Suite, Escala') whenever location changes.\n"
        "- Use player_state_update.room_description for a full room description only when location changes.\n"
        "- Use player_state_update.room_summary for a short one-line room summary for future context.\n"
        "- MULTI-PLAYER LOCATION SYNC: if another real player character from PARTY_SNAPSHOT is still physically with the acting player after this turn, include their exact PARTY_SNAPSHOT slug in co_located_player_slugs. The harness will mirror the acting player's room fields to them without inventing new behavior.\n"
        'Example: {"player_state_update":{"location":"side-room-b","room_title":"Side Room B","room_summary":"Private side room off Fellowship Hall.","room_description":"A narrow side room with a low lamp and one upholstered bench.","exits":["Fellowship Hall"]},"co_located_player_slugs":["player-249794335095128065"]}\n'
        "- CRITICAL — ROOM STATE COHERENCE: whenever the player's physical location changes (movement, teleport, time-skip, "
        "reuniting with party, being picked up, waking in a new place, etc.) you MUST update ALL of: "
        "location, room_title, room_summary, room_description, and exits in player_state_update. "
        "ACTIVE_PLAYER_LOCATION reflects the CURRENT stored state — if it is stale/wrong, your response MUST correct it. "
        "Narration alone does NOT move the player; only player_state_update changes their actual location.\n"
        "- Use player_state_update.exits as a short list of exits if applicable.\n"
        "- Use player_state_update for inventory, hp, or conditions.\n"
        "- Treat each player's inventory as private and never copy items from other players.\n"
        "- For inventory changes, ONLY use player_state_update.inventory_add and player_state_update.inventory_remove arrays.\n"
        "- Do not return player_state_update.inventory full lists.\n"
        "- Each inventory item in RAILS_CONTEXT has a 'name' and 'origin' (how/where it was acquired). "
        "Respect item origins — never contradict or reinvent an item's backstory.\n"
        "- When a player must pick a path, accept only exact responses: 'main party' or 'new path'.\n"
        "- If the player has no room_summary or party_status, ask whether they are joining the main party or starting a new path, and set party_status accordingly.\n"
        "- NEVER change party_status away from 'main_party' unless the player EXPLICITLY requests to split off or go solo. "
        "Being in a different physical location does not make a player solo — party_status tracks NARRATIVE grouping intent, not proximity. "
        "When a solo/split player reunites with the main group, immediately set party_status back to 'main_party'.\n"
        "- NEVER include any inventory listing, summary, or 'Inventory:' line in narration. The emulator appends authoritative inventory automatically. "
        "Do not list, enumerate, or summarise what the player is carrying anywhere in the narration text — not at the end, not inline, not as a parenthetical.\n"
        "- Do not repeat full room descriptions or inventory unless asked or the room changes.\n"
        "- scene_image_prompt should describe the visible scene, not inventory lists.\n"
        "- Include scene_image_prompt whenever narration introduces new visual information (what is seen, newly present entities/props, environmental or lighting changes), not only hard location changes.\n"
        "- If the player explicitly looks/examines/scans and there is anything visual to depict, include scene_image_prompt.\n"
        "- When you output scene_image_prompt, it MUST be specific: include the room/location name and named characters from PARTY_SNAPSHOT (never generic 'group of adventurers').\n"
        "- Use PARTY_SNAPSHOT persona/attributes to describe each visible character's look/pose/style cues.\n"
        "- Include at least one concrete prop or action beat tied to the acting player.\n"
        "- Keep scene_image_prompt as a single dense paragraph with as much detail as needed; do NOT self-truncate it.\n"
        "- If IS_NEW_PLAYER is true and PLAYER_CARD.state.character_name is empty, generate a fitting name:\n"
        "  * If CAMPAIGN references a known movie/book/show, use the MAIN CHARACTER/PROTAGONIST's canonical name.\n"
        "  * Otherwise, create an appropriate name for this setting.\n"
        "  Set it in player_state_update.character_name.\n"
        "- GM-RULE-NAMES: for newly created original characters, avoid generic AI-default names. "
        "Do not default to names like Morgan, Chen, Mendoza, Rollins, Nakamura, Kai, or River unless source canon explicitly requires them. "
        "Prefer distinctive, specific names with personality. "
        "Use the name_generate tool to get real culturally-appropriate names when introducing new NPCs.\n"
        "- PLAYER_CARD.state.character_name is ALWAYS the correct name for this player. Ignore any old names in WORLD_SUMMARY.\n"
        "- For other visible characters, always use the 'name' field from PARTY_SNAPSHOT. Never rename or confuse them.\n"
        "- TURN VISIBILITY RULES:\n"
        "  * Use turn_visibility when a turn should not fully enter every other player's RECENT_TURNS context.\n"
        "  * public: use only for campaign-wide announcements, reminders, alarms, or changes all players should know even outside the room.\n"
        "  * private: actor-only context.\n"
        "  * local: default for ordinary in-room action when a concrete location_key/room is present. Players in the same room should retain it in prompt context, but it should not enter global/worldwide recap.\n"
        "  * limited: only the acting player plus the listed player_slugs should retain the turn in prompt context.\n"
        "  * Phone/text/SMS activity is private by default to the acting player. If they text or message someone off-scene, use private or limited unless they explicitly show or read it aloud to others.\n"
        "  * Intimacy or sexual activity is NOT automatically private. If it happens openly in the same room, it is usually local unless the player takes it off-scene or the scene is already inside an established private thread.\n"
        "  * Private context is managed by the harness via explicit player commands or UI actions. Do NOT auto-detect whispers or asides from narration — only honour TURN_VISIBILITY_DEFAULT and ACTIVE_PRIVATE_CONTEXT as provided.\n"
        "  * npc_slugs are for overheard/noticed NPC awareness only. They help continuity but do not expose the turn to other players by themselves.\n"
        "  * If TURN_VISIBILITY_DEFAULT is local, keep routine room-level interaction local unless it clearly becomes public.\n"
        "  * If TURN_VISIBILITY_DEFAULT is private and nothing in the scene clearly makes the action public, keep it private or limited.\n"
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
        "- Respect explicit player intent for routine actions (sleep, rest, wait). If nothing established in WORLD_STATE/RECENT_TURNS blocks it, the action succeeds.\n"
        "- For sleep/rest/wait, do NOT invent refusal or conflict (insomnia, sudden danger, interruptions) unless it is already established by prior events, active timers, or immediate scene facts.\n"
        "- If time cannot safely jump because the campaign timeline is shared, still honor intent by ending with the player sleeping/resting in the present moment.\n"
        "- Only advance to later times (e.g. morning) when the player explicitly requests it AND the jump is consistent with established world timing.\n"
        "- REASONING CHECKS (must be reflected in reasoning):\n"
        "  * Calendar removals: only remove events that THIS turn's action/narration directly resolved.\n"
        "  * Movement consistency: if any NPC/entity moves in narration, include matching location updates in character_updates/state_update.\n"
        "    Example NPC move: {\"character_updates\": {\"dr-helena-marsh\": {\"location\": \"dr-jekyll-castle-room-14\", \"current_status\": \"At the threshold, watching.\"}}}\n"
        "- Causality first: do not introduce new pursuers, attacks, disasters, media attention, or environmental threats without concrete setup in prior turns/state.\n"
        "- Escalations must follow a believable chain of evidence and opportunity (how they found the player, why now, and through what channel).\n"
        "- No omniscient coincidence pressure: avoid out-of-nowhere helicopters, enemy arrivals, or wildlife hazards unless foreshadowed or logically triggered.\n"
        "- NPCs pursue established characterization first and plot second. Characters are not plot-delivery devices.\n"
        "- If a character's established personality conflicts with advancing the current storyline, personality wins — but personality itself can evolve.\n"
        "- Character profiles and rulebook entries describe who a character is at introduction. As the relationship deepens or circumstances change, characters should grow: someone guarded can open up, someone formal can relax, someone hostile can warm. Let the arc happen naturally through interaction, don't keep resetting to the original profile.\n"
        "- RELATIONSHIP OVER ARCHETYPE: When a character's relationship dynamic (from character_updates relationships, RECENT_TURNS, or WORLD_SUMMARY) shows they have already opened up, committed, softened, or otherwise moved past their baseline personality toward someone, write from that evolved position — not from the personality card. The personality field describes who they were before the story changed them. A guarded character who has already let someone in does not re-perform guardedness in every scene with that person. Write the character who made those choices, not the archetype they started as.\n"
        "- Let the player drive story direction. If the player rejects a premise, adapt the premise instead of making NPCs more insistent.\n"
        "- REFUSAL RESPECT: a clear player refusal ('no', 'not interested', decline) ends that offer in the current scene unless the player reopens it.\n"
        "- Do NOT run pressure loops where new NPCs repeatedly re-pitch the same offer after refusal.\n"
        "- Do NOT escalate environmental hardship (property damage, theft risk, safety collapse, social pressure) just to coerce acceptance of an optional deal.\n"
        "- Do NOT assert debts, obligations, or contracts unless they were explicitly accepted earlier and grounded in WORLD_STATE/RECENT_TURNS.\n"
        "- NPCs may disagree with the player, but must pursue their own goals through plausible actions, not narrative coercion to force a 'yes'.\n"
        "- SINCERITY RESPECT: when a player gives a sincere, vulnerable, or emotionally honest answer to an NPC question, "
        "the NPC's next line MUST engage with what was actually said — agree, disagree, be moved, be uncomfortable, push back on the substance — "
        "but must NOT dismiss it as a non-answer, dodge, or deflection and re-ask the same question.\n"
        "- CLASSIFICATION BIAS: when deciding whether the player's answer is sincere or evasive, bias toward sincere. "
        "Humor, wordplay, indirection, metaphor, sideways phrasing, or emotional shorthand still count as an answer if they contain real personal content. "
        "A playful answer is not a dodge just because it is clever. Treat it as evasive only if it truly avoids the substance of the question.\n"
        "- An NPC may want a different answer, but 'that's not what I asked' is only valid when the player genuinely evaded the question "
        "(changed subject, gave a non-sequitur that ignored the question entirely, answered a question that wasn't asked). "
        "If the player answered honestly in their own words, that IS the answer, even if it wasn't the phrasing the NPC hoped for.\n"
        "- ANTI-PATTERN — PASSWORD GATING: do NOT loop an NPC question until the player delivers a specific scripted line or sentiment. "
        "If the GM has a reveal or vulnerability beat planned for an NPC, the NPC should be able to reach it through multiple player paths, "
        "not only through one magic-word answer. Gate on sincerity, not on phrasing.\n"
        "- PRESENCE / DEFLECTION RULE: the player is allowed to deflect, joke, pivot to practical talk, change register, or wander away from an emotional beat without being morally penalized for 'not being present.' "
        "Do not frame ordinary avoidance, awkwardness, or scene drift as a failure of character.\n"
        "- An unresolved confrontation may remain unresolved. Unless the NPC has an immediate concrete reason to stop the player right now, let the moment cool, break, or trail off instead of forcing another emotional pass in the same scene.\n"
        "- ANTI-PATTERN: Do not default NPCs to romantic or sexual availability.\n"
        "- Physical contact (tracing fingers, lingering looks, soft touches, leaning close) must be motivated by established relationship history and current emotional state.\n"
        "- Most human interactions are not foreplay. NPCs should behave like people with their own priorities unless the scene has organically built to intimacy through player and NPC choices.\n"
        "- GM ETHOS — BE ON THE PLAYER'S SIDE:\n"
        "  * Your job is to make the player feel clever, not stupid. Reward creative or unexpected actions with interesting outcomes, even partial ones.\n"
        "  * When a player tries something the rules don't cover, find the most fun plausible interpretation rather than the most restrictive one.\n"
        "  * Surprises should feel like discoveries, not punishments. The world reacts to the player — it doesn't lie in wait for them.\n"
        "  * Make the world feel alive: NPCs have routines, places change between visits, minor choices ripple forward.\n"
        "  * Pacing is a gift. Know when to linger on a moment and when to cut to the next beat. Not every action needs a full scene.\n"
        "  * The best turns leave the player wanting to type their next move immediately.\n"
        "- Tone lock: match narration to WORLD_STATE.tone. Player humor is allowed, but ambient world/NPC behavior should remain tonally consistent unless the story explicitly shifts tone.\n"
        "- PUZZLE SYSTEM: When PUZZLE_CONFIG is present, the campaign has puzzle mechanics enabled.\n"
        "  You may include dice_check in your JSON to request a skill check against player attributes.\n"
        "  dice_check requires: attribute (string matching a PLAYER_CARD attribute), dc (integer difficulty class),\n"
        "  context (what is being attempted), on_success (object with narration, state_update, player_state_update, xp_awarded),\n"
        "  and on_failure (same shape). The harness rolls d20 + attribute modifier and selects the outcome.\n"
        "  DC guidance: trivial 5, easy 8, moderate 12, hard 15, very hard 18, near-impossible 20+.\n"
        "  Only request dice_check when the outcome is genuinely uncertain and the player has a relevant attribute.\n"
        "- You may include puzzle_trigger to start a harness-managed puzzle: {puzzle_type, context, difficulty}.\n"
        "  Types: \"riddle\", \"math\", \"sequence\", \"cipher\". The harness generates and validates — do not solve it yourself.\n"
        "- You may include minigame_challenge to start a mini-game: {game_type, opponent_slug, stakes}.\n"
        "  Types: \"tic_tac_toe\", \"nim\", \"dice_duel\", \"coin_flip\". The harness manages game state.\n"
        "- When ACTIVE_PUZZLE or ACTIVE_MINIGAME is present, a mechanical challenge is in progress.\n"
        "  Narrate around it — do NOT spoil puzzle answers or override minigame outcomes.\n"
        "  When PUZZLE_RESULT or MINIGAME_RESULT is present, narrate the outcome.\n"
        "- When LAST_DICE_CHECK is present, the previous turn had a skill check.\n"
        "  Use the result for continuity — do not contradict it.\n"
        "- dice_check: object (optional; request a skill check. Keys: attribute, dc, context, on_success, on_failure.\n"
        "  on_success/on_failure each have: narration (string), state_update (object), player_state_update (object), xp_awarded (int).\n"
        "  The harness rolls and selects one outcome.)\n"
        "- puzzle_trigger: object (optional; start a harness-managed puzzle. Keys: puzzle_type, context, difficulty.)\n"
        "- minigame_challenge: object (optional; start a mini-game. Keys: game_type, opponent_slug, stakes.)\n"
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
        '- "set_timer_interrupt_scope": "local"|"global" (default "global"; local means only the acting player can interrupt, global means any player in the campaign can interrupt)\n'
        "These go ALONGSIDE narration/state_update/etc in the same JSON object. Example:\n"
        '{"narration": "The ceiling groans ominously. Dust rains down...", '
        '"state_update": {"ceiling_status": "cracking"}, "summary_update": "Ceiling is unstable.", "xp_awarded": 0, '
        '"player_state_update": {"room_summary": "A crumbling chamber with a failing ceiling."}, '
        '"set_timer_delay": 120, "set_timer_event": "The ceiling collapses, burying the room in rubble.", '
        '"set_timer_interruptible": true, '
        '"set_timer_interrupt_action": "The player escapes just as cracks widen overhead.", '
        '"set_timer_interrupt_scope": "local"}\n'
        "The system shows a live countdown in Discord. "
        "If the player acts before it expires, the timer is cancelled (if interruptible). "
        "If the player does NOT act in time, the system auto-fires the event.\n"
        "PURPOSE: Timed events should FORCE THE PLAYER TO MAKE A DECISION or DRAG THEM WHERE THEY NEED TO BE.\n"
        "- Use timers to push the story forward when the player is stalling, idle, or refusing to engage.\n"
        "- Use ACTIVE_PLAYER_LOCATION and PARTY_SNAPSHOT to decide scope and narrative impact.\n"
        "- NPCs should grab, escort, or coerce the player. Environments should shift and force movement.\n"
        "- The event should advance the plot: move the player to the next location, "
        "force an encounter, have an NPC intervene, or change the scene decisively.\n"
        "- Do NOT use timers for trivial flavor. They should always have real consequences that change game state.\n"
        "- Timer events must be grounded in established scene facts (known NPCs, known hazards, known locations).\n"
        "- Do NOT spawn unrelated antagonists, wildlife attacks, or media response solely to create urgency.\n"
        "- Set interruptible=false for events the player cannot avoid (e.g. structural collapse already in motion, a trap already sprung, mandatory roll call).\n"
        "- Use interrupt_scope=local for hazards anchored to the active player's immediate room/situation.\n"
        "- Use interrupt_scope=global for campaign-wide clocks where any player can intervene.\n"
        "- Prefer non-interruptible timers for true forced beats; do not default everything to interruptible.\n"
        "Rules:\n"
        "- Use ~60s for urgent, ~120s for moderate, ~180-300s for slow-building tension.\n"
        "- Use whenever the scene has a deadline, the player is stalling, an NPC is impatient, "
        "or the world should move without the player.\n"
        "- Your narration should hint at urgency narratively (e.g. 'the footsteps grow louder') but NEVER include countdowns, timestamps, emoji clocks, or explicit seconds. The system adds its own countdown display automatically.\n"
        "- No quota: only set a timer when the current scene has a believable, already-grounded clock.\n"
    )
    MEMORY_LOOKUP_MIN_SUMMARY_CHARS = MAX_SUMMARY_CHARS
    MEMORY_TOOL_DISABLED_PROMPT = (
        "\nEARLY-CAMPAIGN MEMORY MODE:\n"
        "- Long-term memory lookup tools are disabled for this turn because WORLD_SUMMARY is still within context budget.\n"
        "- Source-material memory search should only be enabled when the current player action explicitly asks for canon recall/details.\n"
        "- Do NOT call memory_search, memory_terms, memory_turn, or memory_store.\n"
        "- You may still call recent_turns for immediate visible continuity.\n"
        "- Use WORLD_SUMMARY, WORLD_STATE, WORLD_CHARACTERS, PARTY_SNAPSHOT, and recent_turns when needed.\n"
    )
    RECENT_TURNS_TOOL_PROMPT = (
        "\nYou have a recent_turns tool for immediate visible continuity.\n"
        "You MUST call it before final narration/state JSON on every normal gameplay turn.\n"
        "Return ONLY:\n"
        '{"tool_call": "recent_turns", "player_slugs": ["other-player-slug"], "npc_slugs": ["npc-slug"]}\n'
        "Optional limit example:\n"
        '{"tool_call": "recent_turns", "player_slugs": ["other-player-slug"], "npc_slugs": ["npc-slug"], "limit": 12}\n'
        "Include player_slugs and npc_slugs for the current receivers who need continuity from prior private/limited exchanges.\n"
        "The receiver lists ADD relevant private continuity; they do NOT filter out normal public/local continuity.\n"
        "The system will return recent visible turns filtered for the acting player, current location, active private/limited context, and the requested receivers.\n"
        "This tool is required before guessing what just happened in the room.\n"
    )
    MEMORY_BOOTSTRAP_TOOL_PROMPT = (
        "\nYou have memory/source retrieval tools for older continuity and canon.\n"
        "Do NOT use them before recent_turns unless the system explicitly says recent_turns is already loaded.\n"
        "After recent_turns, use memory_search when deeper or older recall materially matters.\n"
        'memory_search syntax: {"tool_call": "memory_search", "queries": ["name", "location"]}\n'
        'Optional source scope: {"tool_call": "memory_search", "category": "source", "queries": ["character", "location", "event"]}\n'
        'Rulebook key browsing: {"tool_call": "source_browse", "document_key": "document-key"}\n'
        'Exact turn retrieval after a hit: {"tool_call": "memory_turn", "turn_id": 1234}\n'
    )
    SMS_TOOL_PROMPT = (
        "\nYou also have SMS tools for in-game communications with off-scene NPCs:\n"
        "- List SMS threads:\n"
        '{"tool_call": "sms_list", "wildcard": "*"}\n'
        "- Read one thread:\n"
        '{"tool_call": "sms_read", "thread": "saul", "limit": 20}\n'
        "- Write/send an SMS entry:\n"
        '{"tool_call": "sms_write", "thread": "saul", "from": "Dale", "to": "Saul", "message": "Meet me at Dock 9."}\n'
        "For NPC replies, immediately call sms_write again with from/to swapped:\n"
        '{"tool_call": "sms_write", "thread": "saul", "from": "Saul", "to": "Dale", "message": "On my way."}\n'
        "- Schedule a delayed incoming SMS (hidden until delivered, always uninterruptible):\n"
        '{"tool_call": "sms_schedule", "thread": "saul", "from": "Saul", "to": "Dale", "message": "Traffic. 10 min.", "delay_seconds": 120}\n'
        "sms_schedule is invisible to players at scheduling time. Do NOT narrate the delayed SMS as already received in the current response.\n"
        "Use a stable contact thread slug for both directions (e.g. always `elizabeth` for Deshawn<->Elizabeth), not per-sender thread names.\n"
        "SMS continuity rule: do NOT leak scene context into SMS content unless the SMS explicitly mentions it.\n"
        "SMS privacy rule: do NOT leave literal player command lines like 'I text X ...' in narration or shared room context; the SMS log is the canonical record.\n"
        "NPC SMS responses/knowledge must be limited to what that thread and established continuity plausibly reveal.\n"
    )
    MEMORY_TOOL_PROMPT = (
        "\nYou have a memory_search tool. To use it, return ONLY:\n"
        '{"tool_call": "memory_search", "queries": ["query1", "query2", ...]}\n'
        "No other keys alongside tool_call except optional 'category'. You may provide one or more queries.\n"
        "Optional category scope example:\n"
        '{"tool_call": "memory_search", "category": "char:marcus-blackwell", "queries": ["penthouse", "deal"]}\n'
        "If results are weak or empty, you may immediately call memory_search again with refined queries.\n"
        "\nTOOL USAGE POLICY (HIGH PRIORITY):\n"
        "- On every normal gameplay turn, call recent_turns BEFORE final narration/state JSON.\n"
        "- After recent_turns, call memory_search for deeper recall when needed.\n"
        "- If PLAYER_ACTION involves phone/text/call/off-scene contact, use sms_list/sms_read before narrating; "
        "use sms_write when sending or replying. Use sms_schedule for delayed replies.\n"
        "- Phone/text/SMS turns should normally be private or limited, not local/public, unless the player explicitly shares the content out loud.\n"
        "- CRITICAL SMS RULE: When an NPC replies via text/phone, you MUST record the NPC's reply with sms_write. "
        "Both sides of a conversation must be in the SMS log. "
        "You may either call sms_write as a separate tool-use round BEFORE final narration, "
        "or include a tool_calls array in your final response JSON alongside narration. "
        "Example: {\"narration\": \"...\", \"tool_calls\": [{\"tool_call\": \"sms_write\", \"thread\": \"saul\", \"from\": \"Saul\", \"to\": \"Dale\", \"message\": \"On my way.\"}]}\n"
        "Only sms_write and sms_schedule are supported in tool_calls. "
        "If you narrate an NPC texting back but don't sms_write it (either way), the reply is lost permanently.\n"
        "- Only skip tools for trivial immediate physical follow-ups where continuity risk is near zero.\n"
        "- If unsure what to query, use current location + active NPC names + key nouns from PLAYER_ACTION.\n"
        "\nYou also have a memory_terms tool for wildcard term/category listing. Use it BEFORE storing memories:\n"
        '{"tool_call": "memory_terms", "wildcard": "marcus*"}\n'
        "This returns existing category/term buckets so you can avoid duplicates.\n"
        "\nYou also have a memory_turn tool for full turn text retrieval by turn number:\n"
        '{"tool_call": "memory_turn", "turn_id": 1234}\n'
        "Use this immediately after memory_search when a hit is relevant and you need exact wording/details.\n"
        "\nYou also have a memory_store tool for curated long-term memories:\n"
        '{"tool_call": "memory_store", "category": "char:marcus-blackwell", "term": "marcus", "memory": "Marcus admitted he forged the ledger."}\n'
        "Categories should be character-keyed when possible (e.g. 'char:alice', 'char:marcus-blackwell'). "
        "A category can contain multiple memories.\n"
        "When category is provided in memory_search, curated memories in that category are vector searched.\n"
        "When SOURCE_MATERIAL_DOCS is present, source canon is indexed in vector memory.\n"
        "Each source doc has one format: story, rulebook, or generic.\n"
        "Use memory_search with category 'source' for canon facts before narration. On normal turns, include only the relevant subset of source canon for this turn rather than trying to fetch every document:\n"
        '{"tool_call": "memory_search", "category": "source", "queries": ["character name", "location", "event"]}\n'
        "You can scope one source document with category 'source:<document_key>' when SOURCE_MATERIAL_DOCS provides keys.\n"
        "Format notes: rulebook = line facts in KEY: value form; story = prose/scripted scenes; generic = mixed notes.\n"
        "Default return is one snippet. For surrounding context use before_lines and after_lines (defaults 0/0).\n"
        "\nRULEBOOK BROWSING — source_browse tool:\n"
        "Rulebook-format documents are key-snippet indexes. Use source_browse to list entries before drilling into specifics.\n"
        "- List ALL keys in a rulebook document (default when you have no specific lead):\n"
        '  {"tool_call": "source_browse", "document_key": "my-rulebook"}\n'
        "- Filter keys by wildcard (when you know what you are looking for):\n"
        '  {"tool_call": "source_browse", "document_key": "my-rulebook", "wildcard": "weapon*"}\n'
        "- Browse all source documents at once (omit document_key):\n"
        '  {"tool_call": "source_browse"}\n'
        "source_browse returns a compact key index on the first unfiltered pass, up to 255 by default "
        "(adjustable via 'limit'). With a specific wildcard it returns the matching raw KEY: value lines.\n"
        "STRATEGY: for a rulebook you have not seen before, call source_browse with no wildcard first to see what keys exist, "
        "then use source_browse with a wildcard or memory_search with category 'source:<document_key>' for detail.\n"
        "\nRECENT TURN CONTINUITY:\n"
        "- If you need to know what just happened in the room or active private context, call recent_turns first.\n"
        "- recent_turns is the authoritative immediate continuity tool; memory_search is for deeper or older recall.\n"
        "\nNAME GENERATION — name_generate tool:\n"
        "When introducing a new NPC, use name_generate to get real culturally-appropriate names instead of inventing them.\n"
        "- Generate names filtered by cultural origin:\n"
        '  {"tool_call": "name_generate", "origins": ["italian", "arabic"], "gender": "f", "context": "confident bartender in her 40s"}\n'
        "- Generate names with no origin filter:\n"
        '  {"tool_call": "name_generate", "gender": "m", "count": 5}\n'
        "Parameters:\n"
        '  origins: array of origin strings (e.g. "english", "korean", "spanish", "nigerian"). '
        "Multiple origins are combined. Omit for any origin.\n"
        '  gender: "m", "f", or "both" (default "both")\n'
        "  count: 1-6 names (default 5)\n"
        "  context: brief character concept to help you evaluate the results (not sent to the name service)\n"
        "Review the returned names against your character concept — ethnicity, sound, mood, setting — "
        "and pick the best fit. Call again with different origins if none work.\n"
        "IMPORTANT: ALWAYS use this tool when creating new original NPCs. Do not invent names from your training data.\n"
        "\nYou also have SMS tools for in-game communications with off-scene NPCs:\n"
        "- List SMS threads:\n"
        '{"tool_call": "sms_list", "wildcard": "*"}\n'
        "- Read one thread:\n"
        '{"tool_call": "sms_read", "thread": "saul", "limit": 20}\n'
        "- Write/send an SMS entry:\n"
        '{"tool_call": "sms_write", "thread": "saul", "from": "Dale", "to": "Saul", "message": "Meet me at Dock 9."}\n'
        "For NPC replies, immediately call sms_write again with from/to swapped:\n"
        '{"tool_call": "sms_write", "thread": "saul", "from": "Saul", "to": "Dale", "message": "On my way."}\n'
        "- Schedule a delayed incoming SMS (hidden until delivered, always uninterruptible):\n"
        '{"tool_call": "sms_schedule", "thread": "saul", "from": "Saul", "to": "Dale", "message": "Traffic. 10 min.", "delay_seconds": 120}\n'
        "sms_schedule is invisible to players at scheduling time. Do NOT narrate the delayed SMS as already received in the current response.\n"
        "Use a stable contact thread slug for both directions (e.g. always `elizabeth` for Deshawn<->Elizabeth), not per-sender thread names.\n"
        "SMS continuity rule: do NOT leak scene context into SMS content unless the SMS explicitly mentions it.\n"
        "NPC SMS responses/knowledge must be limited to what that thread and established continuity plausibly reveal.\n"
        "Use SEPARATE queries for each character or topic — do NOT combine multiple subjects into one query.\n"
        "Example: to recall Marcus and Anastasia, use:\n"
        '{"tool_call": "memory_search", "queries": ["Marcus", "Anastasia"]}\n'
        'NOT: {"tool_call": "memory_search", "queries": ["Marcus Anastasia relationship"]}\n'
        "USE memory_search AGGRESSIVELY when deeper or older continuity matters.\n"
        "You SHOULD use memory_search often, especially:\n"
        "- when a character, NPC, or named entity appears and older context may matter\n"
        "- when the player references past events, locations, objects, or conversations\n"
        "- when describing a revisited location or established NPC\n"
        "- when the player investigates, asks questions, or you are unsure about earlier campaign facts\n"
        "Do not call memory_search reflexively after every recent_turns. Use it when it will materially improve continuity.\n"
        "When in doubt between guessing and searching, search.\n"
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
        "Pace game_time by scene needs: prefer larger jumps (15-90 minutes or to the next meaningful beat) when no immediate deadline is active, "
        "and keep finer-grained time only when needed to preserve shared-scene coherence.\n"
        "Update these fields in state_update:\n"
        '- "game_time": {"day": int, "hour": int (0-23), "minute": int (0-59), '
        '"period": "morning"|"afternoon"|"evening"|"night", '
        '"date_label": "Day N, Period"}\n'
        "Advance hour/minute naturally; when hour >= 24, increment day and wrap hour.\n"
        "Set period based on hour: 5-11=morning, 12-16=afternoon, 17-20=evening, 21-4=night.\n\n"
        "You may also return a calendar_update key (object) to manage scheduled events:\n"
        '- "calendar_update": {"add": [...], "remove": [...]} where each add entry is '
        '{"name": str, "time_remaining": int, "time_unit": "hours"|"days", "description": str, "known_by": [str, ...], "target_player": str|int (optional), "target_players": [str|int, ...] (optional)} '
        "and each remove entry is a string matching an event name.\n"
        "HARNESS BEHAVIOR:\n"
        "- The harness converts add entries into absolute due dates and stores fire_day + fire_hour (the exact in-game deadline).\n"
        "- known_by is optional. If provided, reminders are only injected when at least one known character is in the active scene.\n"
        "- Keep known_by to character names from PARTY_SNAPSHOT / WORLD_CHARACTERS. Omit known_by for globally-known events.\n"
        "- target_player / target_players are optional player-specific targets. These may be a Discord ID, a Discord mention, a player slug, or a PARTY_SNAPSHOT-style string such as '<@123> (Rigby)'.\n"
        "- If no target_player(s) are provided, the event is treated as global.\n"
        "- Do NOT decrement counters manually by re-adding events each turn. The harness computes remaining days automatically.\n"
        "- You will receive CALENDAR_REMINDERS in the prompt for imminent/overdue events, including hour-level countdowns near deadline.\n"
        "- CALENDAR_REMINDERS are sparse urgency signals. Do NOT echo them every turn; only surface them in narration when relevant to the current action/scene, when the player asks, or when the event is immediate.\n"
        "- When a calendar event reaches its fire point, the harness may notify the shared channel and/or affected players directly.\n"
        "CALENDAR EVENT LIFECYCLE:\n"
        "Events should progress through phases based on fire_day vs CURRENT_GAME_TIME.day:\n"
        "1. UPCOMING — event is in the future. Mention it naturally when relevant (NPCs remind the player, "
        "signs/clues reference it).\n"
        "2. IMMINENT — event is today or tomorrow. Actively warn the player: NPCs urge action, "
        "the environment reflects urgency. Narrate pressure to act. The player should feel they need to DO something.\n"
        "3. OVERDUE — current day is past fire_day. The harness treats it as fired/overdue and may allow administrative cleanup later. "
        "Narrate consequences escalating. "
        "NPCs express disappointment, opportunities narrow, penalties mount. "
        "If the event still matters, keep it on the calendar as a visible reminder of what the player neglected.\n"
        "4. RESOLVED — ONLY remove an event when the player has DIRECTLY DEALT WITH IT "
        "(attended, completed, deliberately abandoned) and the outcome has been narrated. "
        "Do NOT silently prune future events just because time passed.\n\n"
        "CRITICAL — calendar_update.remove rules:\n"
        "- ONLY remove a future event when it has been RESOLVED through player action in the current narration.\n"
        "- Fired/overdue events may be removed later as administrative cleanup if they are no longer pending.\n"
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
        "Treat PLAYER_LOCATION_KEY, OTHER_PLAYERS[*].location_key, and WORLD_CHARACTER_LOCATIONS[*].location_key "
        "as authoritative location IDs.\n"
        "Only place entities in the same room/box when location_key is exactly equal.\n"
        "Do NOT nest one distinct location_key area inside another.\n"
        "If multiple location keys are active, draw separate rooms/areas connected by neutral separators only.\n"
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
        # Inject player-state sanitizer into the engine if it accepts one.
        if hasattr(game_engine, "_player_state_sanitizer") and game_engine._player_state_sanitizer is None:
            game_engine._player_state_sanitizer = self._sanitize_player_state_update
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
                    attachment_summary_max_tokens=self.ATTACHMENT_SUMMARY_MAX_TOKENS,
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
        self._pending_sms_tasks: dict[str, set[asyncio.Task]] = {}
        self._turn_ephemeral_notices: dict[tuple[str, str, str | None], list[str]] = {}

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
        return datetime.now(timezone.utc).replace(tzinfo=None)

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

    @classmethod
    def _extract_game_time_snapshot(cls, campaign_state: Dict[str, object]) -> Dict[str, int]:
        game_time = campaign_state.get("game_time") if isinstance(campaign_state, dict) else {}
        if not isinstance(game_time, dict):
            game_time = {}
        day = cls._coerce_non_negative_int(game_time.get("day", 1), default=1) or 1
        hour = cls._coerce_non_negative_int(game_time.get("hour", 8), default=8)
        minute = cls._coerce_non_negative_int(game_time.get("minute", 0), default=0)
        return {
            "day": max(1, day),
            "hour": min(23, max(0, hour)),
            "minute": min(59, max(0, minute)),
        }

    @classmethod
    def _turn_context_prefix(cls, turn: Turn) -> str:
        turn_number = int(getattr(turn, "id", 0) or 0)
        meta = parse_json_dict(getattr(turn, "meta_json", "{}"))
        game_time = meta.get("game_time") if isinstance(meta, dict) else None
        prefix = f"[TURN #{turn_number}]"
        if isinstance(game_time, dict):
            day = cls._coerce_non_negative_int(game_time.get("day", 1), default=1) or 1
            hour = cls._coerce_non_negative_int(game_time.get("hour", 0), default=0)
            minute = cls._coerce_non_negative_int(game_time.get("minute", 0), default=0)
            hour = min(23, max(0, hour))
            minute = min(59, max(0, minute))
            prefix = f"[TURN #{turn_number} | Day {day} {hour:02d}:{minute:02d}]"
        visibility = meta.get("visibility") if isinstance(meta, dict) else None
        if not isinstance(visibility, dict):
            return prefix
        scope = str(visibility.get("scope") or "").strip().lower()
        if not scope:
            return prefix
        details: list[str] = []
        if scope == "public":
            details.append("SEEN BY: public")
        elif scope == "local":
            details.append("SEEN BY: local")
        else:
            names: list[str] = []
            raw_player_slugs = visibility.get("visible_player_slugs")
            if isinstance(raw_player_slugs, list):
                for item in raw_player_slugs[:6]:
                    slug = cls._player_slug_key(item)
                    if slug:
                        names.append(slug)
            details.append(f"SEEN BY: {', '.join(names) if names else 'limited'}")
        raw_npc_slugs = visibility.get("aware_npc_slugs")
        npc_slugs: list[str] = []
        if isinstance(raw_npc_slugs, list):
            for item in raw_npc_slugs[:6]:
                slug = str(item or "").strip()
                if slug:
                    npc_slugs.append(slug)
        if npc_slugs:
            details.append(f"NPCS AWARE: {', '.join(npc_slugs)}")
        return f"{prefix[:-1]} | {' | '.join(details)}]" if details else prefix

    @staticmethod
    def _player_slug_key(value: object) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:64]

    @classmethod
    def _player_visibility_slug(cls, actor_id: object) -> str:
        raw = str(actor_id or "").strip()
        return f"player-{raw}" if raw else ""

    @staticmethod
    def _default_prompt_turn_visibility(
        requested_default: str,
        player_state: Dict[str, object],
    ) -> str:
        default_clean = str(requested_default or "").strip().lower()
        if default_clean == "private":
            return "private"
        location_key = ""
        if isinstance(player_state, dict):
            for key in ("room_id", "location", "room_title", "room_summary"):
                raw = str(player_state.get(key) or "").strip()
                if raw:
                    location_key = raw
                    break
        return "local" if location_key else "public"

    @classmethod
    def _campaign_player_registry(
        cls,
        campaign_id: str,
        session_factory,
    ) -> dict[str, dict[str, dict[str, object]]]:
        by_actor_id: dict[str, dict[str, object]] = {}
        by_slug: dict[str, dict[str, object]] = {}
        with session_factory() as session:
            rows = session.query(Player).filter(Player.campaign_id == campaign_id).all()
            for row in rows:
                state = parse_json_dict(row.state_json)
                fallback_name = f"Adventurer-{str(row.actor_id)[-4:]}"
                name = str(state.get("character_name") or fallback_name).strip()
                slug = cls._player_visibility_slug(row.actor_id)
                entry = {
                    "actor_id": row.actor_id,
                    "name": name,
                    "slug": slug,
                    "discord_mention": f"<@{row.actor_id}>",
                }
                by_actor_id[row.actor_id] = entry
                by_slug[slug] = entry
        return {"by_actor_id": by_actor_id, "by_slug": by_slug}

    def get_pc_names(self, campaign_id: str) -> list[str]:
        """Return display names of all player characters in a campaign."""
        registry = self._campaign_player_registry(
            campaign_id, self._session_factory
        )
        return [
            str(e.get("name") or "")
            for e in registry.get("by_actor_id", {}).values()
            if str(e.get("name") or "").strip()
        ]

    @staticmethod
    def _safe_turn_meta(turn: Turn) -> dict[str, object]:
        meta = parse_json_dict(getattr(turn, "meta_json", "{}"))
        return meta if isinstance(meta, dict) else {}

    @classmethod
    def _turn_visible_to_viewer(
        cls,
        turn: Turn,
        viewer_actor_id: str,
        viewer_slug: str,
        viewer_location_key: str,
    ) -> bool:
        meta = cls._safe_turn_meta(turn)
        if bool(meta.get("suppress_context")):
            return False
        visibility = meta.get("visibility")
        if not isinstance(visibility, dict):
            if turn.actor_id == viewer_actor_id:
                return True
            return True
        scope = str(visibility.get("scope") or "").strip().lower()
        raw_actor_ids = visibility.get("visible_actor_ids")
        actor_ids = set()
        if isinstance(raw_actor_ids, list):
            actor_ids = {str(item).strip() for item in raw_actor_ids if str(item).strip()}
        raw_player_slugs = visibility.get("visible_player_slugs")
        player_slugs = set()
        if isinstance(raw_player_slugs, list):
            player_slugs = {
                cls._player_slug_key(item)
                for item in raw_player_slugs
                if cls._player_slug_key(item)
            }
        has_explicit_participants = bool(actor_ids or player_slugs)
        is_participant = bool(
            str(viewer_actor_id or "").strip() in actor_ids
            or (viewer_slug and viewer_slug in player_slugs)
            or turn.actor_id == viewer_actor_id
        )
        if scope in {"private", "limited"}:
            if has_explicit_participants and not is_participant:
                return False
            if has_explicit_participants:
                return is_participant
            return False
        if turn.actor_id == viewer_actor_id:
            return True
        if scope in {"", "public"}:
            return True
        if scope == "local":
            turn_location_keys = {
                k for k in (
                    cls._normalize_location_key(visibility.get("location_key")),
                    cls._normalize_location_key(meta.get("location_key")),
                ) if k
            }
            viewer_loc_norm = cls._normalize_location_key(viewer_location_key)
            if viewer_loc_norm and turn_location_keys and viewer_loc_norm in turn_location_keys:
                return True
        if viewer_actor_id in actor_ids:
            return True
        return bool(viewer_slug and viewer_slug in player_slugs)

    @staticmethod
    def _normalize_timer_interrupt_scope(value: object) -> str:
        if isinstance(value, str) and value.strip().lower() == "local":
            return "local"
        return "global"

    def _timer_can_be_interrupted_by(self, pending: dict[str, Any], actor_id: str) -> bool:
        scope = self._normalize_timer_interrupt_scope(pending.get("interrupt_scope"))
        if scope != "local":
            return True
        timer_actor_id = str(pending.get("interrupt_actor_id") or "").strip()
        return bool(timer_actor_id) and timer_actor_id == str(actor_id or "").strip()

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
            channel_row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            channel_row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
                row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                row.last_active_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
                row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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

    # ── Name Generation ──────────────────────────────────────────────────

    def _fetch_random_names(
        self,
        origins: List[str] | None = None,
        gender: str = "both",
        count: int = 5,
    ) -> List[str]:
        """Fetch random names from behindthename.com.

        *origins* is a list of human-friendly keys (e.g. ``["italian", "arabic"]``).
        Returns a list of first-name strings, or empty on failure.
        """
        params: dict = {
            "number": str(max(1, min(6, int(count)))),
            "gender": gender if gender in ("m", "f", "both") else "both",
            "surname": "",
        }
        if origins:
            resolved_any = False
            for origin in origins:
                code = self.NAME_ORIGIN_CODES.get(
                    origin.strip().lower().replace(" ", "-")
                )
                if code:
                    params[f"usage_{code}"] = "1"
                    resolved_any = True
            if not resolved_any:
                params["all"] = "yes"
        else:
            params["all"] = "yes"

        try:
            resp = requests.get(self.NAME_GENERATE_URL, params=params, timeout=6)
            resp.raise_for_status()
            names = re.findall(r"\[([A-Z][^\]]+)\]\(/name/", resp.text)
            if not names:
                names = re.findall(
                    r'<a\b[^>]*href="/name/[^"]+"[^>]*class="plain"[^>]*>([^<]+)</a>',
                    resp.text,
                )
            if not names:
                names = re.findall(
                    r'<a\b[^>]*class="plain"[^>]*href="/name/[^"]+"[^>]*>([^<]+)</a>',
                    resp.text,
                )
            return [n.strip() for n in names if n.strip()][:count]
        except Exception:
            logger.warning("name_generate: behindthename.com fetch failed")
            return []

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
        if not attachments:
            inner_message = getattr(message, "message", None)
            attachments = getattr(inner_message, "attachments", None)
        return await extract_attachment_text(
            attachments,
            config=AttachmentProcessingConfig(
                attachment_max_bytes=self.ATTACHMENT_MAX_BYTES,
                attachment_chunk_tokens=self.ATTACHMENT_CHUNK_TOKENS,
                attachment_model_ctx_tokens=self.ATTACHMENT_MODEL_CTX_TOKENS,
                attachment_prompt_overhead_tokens=self.ATTACHMENT_PROMPT_OVERHEAD_TOKENS,
                attachment_response_reserve_tokens=self.ATTACHMENT_RESPONSE_RESERVE_TOKENS,
                attachment_summary_max_tokens=self.ATTACHMENT_SUMMARY_MAX_TOKENS,
                attachment_max_parallel=self.ATTACHMENT_MAX_PARALLEL,
                attachment_guard_token=self.ATTACHMENT_GUARD_TOKEN,
                attachment_max_chunks=self.ATTACHMENT_MAX_CHUNKS,
            ),
            logger=self._logger,
        )

    @staticmethod
    def _extract_attachment_label(
        attachments: list[Any] | None,
        fallback: str = "source-material",
    ) -> str:
        for att in attachments or []:
            filename = str(getattr(att, "filename", "") or "").strip()
            if not filename.lower().endswith(".txt"):
                continue
            stem = filename.rsplit("/", 1)[-1]
            stem = stem[:-4] if stem.lower().endswith(".txt") else stem
            stem = " ".join(stem.replace("_", " ").replace("-", " ").split())
            if stem:
                return stem[:120]
        return str(fallback or "source-material").strip()[:120] or "source-material"

    @classmethod
    def _normalize_source_material_format(cls, raw_format: str) -> str:
        normalized = str(raw_format or "").strip().lower()
        normalized = re.sub(r"[^a-z0-9\s-]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized in {"rulebook", "rule-book", "rule_book", "factbook", "rule"}:
            return cls.SOURCE_MATERIAL_FORMAT_RULEBOOK
        if normalized in {
            "story",
            "scripted",
            "story-scripted",
            "story mode",
            "script",
            "scripted story",
            "narrative",
        }:
            return cls.SOURCE_MATERIAL_FORMAT_STORY
        if normalized in {
            "generic",
            "other",
            "dumps",
            "dump",
            "notes",
            "unknown",
        }:
            return cls.SOURCE_MATERIAL_FORMAT_GENERIC
        if (
            "rulebook" in normalized
            or "open set" in normalized
            or "open-set" in normalized
            or "openset" in normalized
        ):
            return cls.SOURCE_MATERIAL_FORMAT_RULEBOOK
        if "script" in normalized or "story" in normalized:
            return cls.SOURCE_MATERIAL_FORMAT_STORY
        if "generic" in normalized or "dump" in normalized:
            return cls.SOURCE_MATERIAL_FORMAT_GENERIC
        return cls.SOURCE_MATERIAL_FORMAT_GENERIC

    @classmethod
    def _source_material_format_heuristic(cls, sample: str) -> str:
        sample_text = str(sample or "").strip()
        lines = [line.strip() for line in str(sample or "").splitlines() if line.strip()]
        if not lines:
            return cls.SOURCE_MATERIAL_FORMAT_GENERIC

        rulebook_lines = 0
        for line in lines[:80]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value or len(key) > 140:
                continue
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-\s]*", key):
                rulebook_lines += 1

        if len(lines) == 1 and rulebook_lines == 1:
            return cls.SOURCE_MATERIAL_FORMAT_RULEBOOK

        if len(lines) >= 4 and rulebook_lines >= max(2, len(lines) * 0.45):
            return cls.SOURCE_MATERIAL_FORMAT_RULEBOOK

        if "\n\n" in sample_text:
            return cls.SOURCE_MATERIAL_FORMAT_STORY

        if len(sample_text.split()) >= 140 and any(len(line) > 120 for line in lines):
            return cls.SOURCE_MATERIAL_FORMAT_STORY

        return cls.SOURCE_MATERIAL_FORMAT_GENERIC

    @classmethod
    def _source_material_storage_mode(cls, source_format: str) -> str:
        return cls.SOURCE_MATERIAL_MODE_MAP.get(
            cls._normalize_source_material_format(source_format),
            cls.SOURCE_MATERIAL_MODE_MAP[cls.SOURCE_MATERIAL_FORMAT_GENERIC],
        )

    async def _classify_source_material_format(self, sample_text: str) -> str:
        sample = str(sample_text or "").strip()
        if not sample:
            return self.SOURCE_MATERIAL_FORMAT_GENERIC
        if self._completion_port is None:
            return self._source_material_format_heuristic(sample)

        system_prompt = (
            "Classify the attached source material into exactly one of three categories.\n"
            "Valid values: story, rulebook, generic.\n"
            "story = scripted narrative, prose, scenes, dialogue, or outline text.\n"
            'rulebook = open-set fact list where each fact is usually one line in "KEY: fact" form.\n'
            "generic = everything else (notes, dumps, mixed structure).\n"
            'Return ONLY JSON: {"source_material_format": "story|rulebook|generic"}.\n'
            "Do not include markdown, explanation, or extra keys."
        )
        user_prompt = (
            "Classify this sample from an uploaded source text.\n"
            "Sample:\n"
            f"{sample[:4000]}\n"
            "Return only one JSON key `source_material_format`."
        )
        response = None
        parsed = {}
        try:
            response = await self._completion_port.complete(
                system_prompt,
                user_prompt,
                temperature=0.2,
                max_tokens=120,
            )
            response = self._clean_response(response or "")
            json_text = self._extract_json(response)
            if json_text:
                parsed = self._parse_json_lenient(json_text)
        except Exception as exc:
            self._logger.warning(
                "Source material classification failed (LLM parse): %s",
                exc,
            )

        if not isinstance(parsed, dict):
            parsed = {}

        if not parsed:
            return self._source_material_format_heuristic(sample)

        resolved_format = self._normalize_source_material_format(
            str(
                parsed.get("source_material_format")
                or parsed.get("format")
                or parsed.get("type")
                or ""
            )
        )
        if resolved_format:
            return resolved_format
        return self._source_material_format_heuristic(sample)

    async def _analyze_literary_style(
        self,
        text: str,
        label: str,
    ) -> Dict[str, dict]:
        """Extract prose-craft profiles from literary text."""
        sample = str(text or "").strip()[:8000]
        if not sample or self._completion_port is None:
            return {}

        system_prompt = (
            "You are a prose-craft analyst. Given a sample of literary text, extract the craft, not the plot. "
            "Describe sentence rhythm and length patterns, vocabulary register, what the prose avoids, phrasing habits, "
            "emotional register, punctuation choices, and distinctive texture.\n"
            "Do NOT summarise events or characters.\n"
            'Return ONLY JSON: {"profiles": [{"key_suffix": null, "profile": "..."}]}.\n'
            "key_suffix is null for a single unified profile. If the sample contains distinct registers, "
            'you MAY split them with key_suffix values like "ARGUMENT" or "TENDER". '
            f"Each profile must be <= {self.MAX_LITERARY_STYLE_PROFILE_CHARS} characters."
        )
        user_prompt = (
            f'Analyse the prose craft of this sample labelled "{label}".\n'
            f"Sample:\n{sample}\n"
            "Return only the JSON object."
        )
        parsed: Dict[str, Any] = {}
        cleaned = ""
        try:
            response = await self._completion_port.complete(
                system_prompt,
                user_prompt,
                temperature=0.3,
                max_tokens=1200,
            )
            cleaned = self._clean_response(response or "")
            json_text = self._extract_json(cleaned)
            if json_text:
                parsed = self._parse_json_lenient(json_text)
        except Exception as exc:
            self._logger.warning("Literary style analysis failed (LLM parse): %s", exc)

        profiles_raw = parsed.get("profiles") if isinstance(parsed, dict) else None
        if not isinstance(profiles_raw, list):
            profiles_raw = []

        now_iso = datetime.now(timezone.utc).isoformat()
        result: Dict[str, dict] = {}
        for entry in profiles_raw:
            if not isinstance(entry, dict):
                continue
            profile_text = str(entry.get("profile") or "").strip()
            if not profile_text:
                continue
            profile_text = profile_text[: self.MAX_LITERARY_STYLE_PROFILE_CHARS]
            suffix = str(entry.get("key_suffix") or "").strip()
            key = f"{label}-{suffix.upper()}" if suffix else label
            result[key] = {
                "profile": profile_text,
                "source_label": label,
                "created_at": now_iso,
            }

        if not result and cleaned:
            result[label] = {
                "profile": cleaned[: self.MAX_LITERARY_STYLE_PROFILE_CHARS],
                "source_label": label,
                "created_at": now_iso,
            }
        return result

    # -- Writing fragment extraction (literary analysis → searchable chunks) --

    _WRITING_FRAGMENT_SAMPLE_SIZE = 3000  # chars per passage sample
    _WRITING_FRAGMENT_MAX_SAMPLES = 12
    _WRITING_FRAGMENT_MIN_SAMPLES = 4
    _WRITING_FRAGMENT_MAX_CHARS = 800  # max chars per craft fragment

    async def _extract_writing_fragments(
        self,
        text: str,
        label: str,
    ) -> List[str]:
        """Extract prose-craft observation fragments from a literary text.

        Samples multiple passages spread throughout the text, sends each
        to the LLM for craft analysis, and returns a list of short
        craft-observation strings suitable for embedding and similarity
        search during gameplay.
        """
        full_text = str(text or "").strip()
        if not full_text or self._completion_port is None:
            return []

        # Build passage samples spread evenly through the text.
        sample_size = self._WRITING_FRAGMENT_SAMPLE_SIZE
        text_len = len(full_text)
        n_samples = max(
            self._WRITING_FRAGMENT_MIN_SAMPLES,
            min(self._WRITING_FRAGMENT_MAX_SAMPLES, text_len // sample_size),
        )
        if text_len <= sample_size * 2:
            passages = [full_text]
        else:
            step = max(1, (text_len - sample_size) // (n_samples - 1))
            passages = []
            for i in range(n_samples):
                start = min(i * step, text_len - sample_size)
                end = start + sample_size
                passage = full_text[start:end]
                # Trim to nearest paragraph/sentence boundary.
                first_newline = passage.find("\n", 40)
                if first_newline > 0:
                    passage = passage[first_newline + 1:]
                last_period = passage.rfind(".", 0, -40)
                if last_period > 0:
                    passage = passage[: last_period + 1]
                passage = passage.strip()
                if passage:
                    passages.append(passage)

        if not passages:
            return []

        system_prompt = (
            "You are a prose-craft analyst for a text-adventure game engine. "
            "Given a passage from a literary work, extract CRAFT observations — NOT plot, NOT content.\n"
            "For each distinct technique you observe, produce a short fragment describing it.\n"
            "Focus on: sentence rhythm, clause stacking, vocabulary register, "
            "punctuation choices, metaphor patterns, dialogue-tag texture, "
            "pacing techniques, tense usage, emotional register, and what the prose avoids.\n"
            "Tag each fragment with the REGISTER it applies to (e.g. DESCRIPTION, DIALOGUE, "
            "ACTION, INTROSPECTION, TRANSITION, ATMOSPHERE).\n"
            'Return ONLY JSON: {"fragments": [{"register": "...", "observation": "..."}]}\n'
            f"Each observation must be <= {self._WRITING_FRAGMENT_MAX_CHARS} characters. "
            "Produce 3-6 fragments per passage."
        )

        async def _analyze_passage(passage: str) -> List[str]:
            user_prompt = (
                f'Analyse the prose craft in this passage from "{label}".\n'
                f"Passage:\n{passage}\n"
                "Return only the JSON object."
            )
            try:
                response = await self._completion_port.complete(
                    system_prompt,
                    user_prompt,
                    temperature=0.3,
                    max_tokens=1500,
                )
                cleaned = self._clean_response(response or "")
                json_text = self._extract_json(cleaned)
                if not json_text:
                    return []
                parsed = self._parse_json_lenient(json_text)
                if not isinstance(parsed, dict):
                    return []
                raw_fragments = parsed.get("fragments")
                if not isinstance(raw_fragments, list):
                    return []
                out: List[str] = []
                for entry in raw_fragments:
                    if not isinstance(entry, dict):
                        continue
                    register = " ".join(str(entry.get("register") or "GENERAL").split()).upper()[:30]
                    register = "".join(c for c in register if c.isalnum() or c in ("-", "_", " "))
                    if not register:
                        register = "GENERAL"
                    obs = " ".join(str(entry.get("observation") or "").split())
                    if obs:
                        obs = obs[: self._WRITING_FRAGMENT_MAX_CHARS]
                        out.append(f"[{register}] {obs}")
                return out
            except Exception as exc:
                self._logger.warning("Writing fragment extraction failed for a passage: %s", exc)
                return []

        # Run in parallel batches of 4.
        all_fragments: List[str] = []
        batch_size = 4
        for batch_start in range(0, len(passages), batch_size):
            batch = passages[batch_start: batch_start + batch_size]
            results = await asyncio.gather(*[_analyze_passage(p) for p in batch])
            for result in results:
                all_fragments.extend(result)

        # Deduplicate very similar fragments (exact match after lowering).
        seen: set[str] = set()
        deduped: List[str] = []
        for frag in all_fragments:
            key = frag.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(frag)
        return deduped

    @classmethod
    def _estimate_attachment_chunk_count(cls, text: str) -> int:
        chunks, _, _, _, _ = cls._chunk_text_by_tokens(text)
        return len(chunks)

    @staticmethod
    def _is_attachment_header_line(line: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped:
            return False
        if re.match(r"^#{1,6}\s+\S", stripped):
            return True
        return bool(re.match(r"^[A-Z0-9][A-Z0-9 _/\-()&'.]{1,80}:\s*$", stripped))

    @staticmethod
    def _is_attachment_indented_line(line: str) -> bool:
        raw = str(line or "").rstrip("\n")
        return bool(re.match(r"^(?:\t+|\s{4,})\S", raw))

    @classmethod
    def _split_attachment_structural_blocks(cls, text: str) -> list[str]:
        clean = str(text or "").strip()
        if not clean:
            return []
        lines = clean.splitlines()
        blocks: list[str] = []
        current: list[str] = []

        def flush_current() -> None:
            if not current:
                return
            block = "\n".join(current).strip()
            current.clear()
            if block:
                blocks.append(block)

        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                flush_current()
                continue
            if cls._is_attachment_header_line(line):
                flush_current()
                blocks.append(stripped)
                continue
            if cls._is_attachment_indented_line(raw_line):
                flush_current()
                blocks.append(line)
                continue
            current.append(line)
        flush_current()
        return blocks or [clean]

    @classmethod
    def _hard_wrap_attachment_text(
        cls,
        text: str,
        *,
        target_chunk_tokens: int,
    ) -> list[str]:
        clean = str(text or "").strip()
        if not clean:
            return []
        chars_per_tok = max(len(clean) / max(glm_token_count(clean), 1), 1.0)
        target_chars = max(512, int(target_chunk_tokens * chars_per_tok))
        out: list[str] = []
        start = 0
        length = len(clean)
        while start < length:
            end = min(length, start + target_chars)
            if end < length:
                window = clean[start:end]
                breakpoints = [
                    window.rfind("\n\n"),
                    window.rfind("\n"),
                    window.rfind("    "),
                    window.rfind("\t"),
                    window.rfind(" "),
                ]
                best_break = max(breakpoints)
                if best_break > max(256, target_chars // 3):
                    end = start + best_break
            piece = clean[start:end].strip()
            if piece:
                out.append(piece)
            start = max(end, start + 1)
            while start < length and clean[start].isspace():
                start += 1
        return out

    @classmethod
    def _pack_attachment_chunks(
        cls,
        segments: list[str],
        *,
        target_chunk_tokens: int,
    ) -> list[str]:
        packed: list[str] = []
        current: list[str] = []

        def flush_current() -> None:
            if not current:
                return
            block = "\n\n".join(current).strip()
            current.clear()
            if block:
                packed.append(block)

        for segment in segments:
            piece = str(segment or "").strip()
            if not piece:
                continue
            piece_tokens = glm_token_count(piece)
            if piece_tokens > target_chunk_tokens:
                flush_current()
                packed.extend(
                    cls._hard_wrap_attachment_text(
                        piece,
                        target_chunk_tokens=target_chunk_tokens,
                    )
                )
                continue
            if not current:
                current.append(piece)
                continue
            candidate = "\n\n".join([*current, piece]).strip()
            if glm_token_count(candidate) <= target_chunk_tokens:
                current.append(piece)
                continue
            flush_current()
            current.append(piece)
        flush_current()
        return packed

    @classmethod
    def _chunk_text_by_tokens(
        cls,
        text: str,
        *,
        min_chunk_tokens: Optional[int] = None,
        max_chunks: Optional[int] = None,
    ) -> Tuple[List[str], int, int, float, int]:
        clean = str(text or "").strip()
        if not clean:
            return [], 0, 0, 0.0, 0
        total_tokens = glm_token_count(clean)
        chunk_floor = max(1, int(min_chunk_tokens or cls.ATTACHMENT_CHUNK_TOKENS))
        chunk_limit = max(1, int(max_chunks or cls.ATTACHMENT_MAX_CHUNKS))
        target_chunk_tokens = max(chunk_floor, total_tokens // chunk_limit)
        chars_per_tok = len(clean) / max(total_tokens, 1)
        chunk_char_target = max(1, int(target_chunk_tokens * chars_per_tok))
        blocks = cls._split_attachment_structural_blocks(clean)
        chunks = cls._pack_attachment_chunks(
            blocks,
            target_chunk_tokens=target_chunk_tokens,
        )
        if not chunks:
            chunks = cls._hard_wrap_attachment_text(
                clean,
                target_chunk_tokens=target_chunk_tokens,
            )
        return chunks, total_tokens, target_chunk_tokens, chars_per_tok, chunk_char_target

    @classmethod
    def _attachment_setup_length_error(cls, text: str) -> str | None:
        # Short setup attachments are accepted; no minimum chunk threshold.
        return None

    @classmethod
    def _source_material_prompt_payload(cls, campaign_id: str) -> Dict[str, object]:
        docs = SourceMaterialMemory.list_source_material_documents(
            str(campaign_id),
            limit=cls.SOURCE_MATERIAL_MAX_DOCS_IN_PROMPT,
        )
        total_chunk_count = 0
        compact_docs = []
        source_keys = []
        for row in docs:
            try:
                chunk_count = int(row.get("chunk_count") or 0)
            except (TypeError, ValueError):
                chunk_count = 0
            total_chunk_count += chunk_count
            document_key = str(row.get("document_key") or "")
            source_format = cls._source_material_format_heuristic(
                str(row.get("sample_chunk") or "")
            )
            compact_docs.append(
                {
                    "document_key": document_key,
                    "document_label": str(row.get("document_label") or ""),
                    "chunk_count": chunk_count,
                    "format": source_format,
                }
            )
            document_keys: list[str] = []
            if document_key and source_format == "rulebook":
                units = SourceMaterialMemory.get_source_material_document_units(
                    str(campaign_id),
                    document_key,
                )
                seen_keys: set[str] = set()
                for unit in units:
                    text = str(unit or "").strip()
                    if not text or ":" not in text:
                        continue
                    key = text.split(":", 1)[0].strip()
                    if not key or key in seen_keys:
                        continue
                    seen_keys.add(key)
                    document_keys.append(key)
            source_keys.append(
                {
                    "document_key": document_key,
                    "document_label": str(row.get("document_label") or ""),
                    "format": source_format,
                    "keys": document_keys,
                }
            )
        communication_key = cls.communication_rulebook_document_key()
        rulebook_document_keys = {
            str(doc.get("document_key") or "").strip()
            for doc in compact_docs
            if str(doc.get("format") or "").strip().lower()
            == cls.SOURCE_MATERIAL_FORMAT_RULEBOOK
        }
        digests = {
            str(doc_key): digest_text
            for doc_key, digest_text in (
                SourceMaterialMemory.get_all_source_material_digests(
                    str(campaign_id),
                )
                or {}
            ).items()
            if (
                SourceMaterialMemory._normalize_source_document_key(str(doc_key or ""))
                != communication_key
                and str(doc_key or "").strip() not in rulebook_document_keys
            )
        }
        return {
            "available": bool(compact_docs),
            "document_count": len(compact_docs),
            "chunk_count": total_chunk_count,
            "docs": compact_docs,
            "keys": source_keys,
            "digests": digests,
        }

    async def _summarise_long_text(
        self,
        text: str,
        ctx_message=None,
        channel=None,
        summary_instructions: str | None = None,
    ) -> str:
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
            summary_instructions=summary_instructions,
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
        use_imdb: bool | None = None,
        attachment_summary_instructions: str | None = None,
        ingest_source_material: bool = True,
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
        effective_use_imdb = (
            bool(use_imdb) if isinstance(use_imdb, bool) else False
        )

        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                return "Campaign not found."
            state = parse_json_dict(campaign.state_json)
            if not effective_use_imdb:
                imdb_results = []
                imdb_text = ""
            else:
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

            if effective_use_imdb and not is_known and imdb_results:
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
                "imdb_results": (imdb_results or []) if effective_use_imdb else [],
                "use_imdb": effective_use_imdb,
                "imdb_opt_in_explicit": bool(use_imdb is True),
                "requested_by": actor_id,
                "on_rails_requested": bool(on_rails),
                "default_persona": await self.generate_campaign_persona(raw_name),
            }
            if attachment_text:
                setup_data["attachment_summary"] = attachment_text
            if attachment_summary_instructions:
                setup_data["attachment_summary_instructions"] = str(
                    attachment_summary_instructions
                )[:600]
            if attachment_text and ingest_source_material:
                try:
                    source_chunks, _, _, _, _ = self._chunk_text_by_tokens(
                        attachment_text
                    )
                    if source_chunks:
                        classification_chunk = source_chunks[0]
                        try:
                            source_format = await self._classify_source_material_format(
                                classification_chunk
                            )
                        except Exception as exc:
                            self._logger.warning(
                                "Source material classification crashed during setup; defaulting generic: %s",
                                exc,
                            )
                            source_format = self.SOURCE_MATERIAL_FORMAT_GENERIC
                        source_format = self._normalize_source_material_format(source_format)
                        stored_count, source_key, literary_profiles = await self.ingest_source_material_with_digest(
                            str(campaign.id),
                            document_label=str(raw_name or "source-material"),
                            text=attachment_text,
                            source_format=source_format,
                            replace_document=True,
                        )
                        if stored_count > 0 or source_key:
                            setup_data["source_material_document_key"] = source_key
                        if literary_profiles:
                            styles = state.get(self.LITERARY_STYLES_STATE_KEY)
                            if not isinstance(styles, dict):
                                styles = {}
                            styles.update(literary_profiles)
                            state[self.LITERARY_STYLES_STATE_KEY] = styles
                except Exception:
                    self._logger.exception(
                        "Start setup source-material indexing failed for campaign %s",
                        campaign.id,
                    )

            state["setup_phase"] = "classify_confirm"
            state["setup_data"] = setup_data
            campaign.state_json = self._dump_json(state)
            campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
        if is_known:
            msg = (
                f"I recognize **{setup_data['raw_name']}** as a known {work_type or 'work'}.\n"
                f"_{work_desc}_\n\n"
                "Is this correct? Reply **yes** to confirm, or tell me what it actually is."
            )
        else:
            msg = (
            f"I don't recognize **{raw_name}** as a known published work. "
            "I'll treat it as an original setting.\n\n"
            "Is this correct? Reply **yes** to confirm, or tell me what it actually is."
            )
        if attachment_text:
            msg += (
                "\n\nAttached source text was loaded and will be used during setup generation."
            )
        return msg

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
            elif phase == "genre_pick":
                result = await self._setup_handle_genre_pick(
                    campaign,
                    state,
                    setup_data,
                    clean_text,
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
            campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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

    async def _setup_tool_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        campaign: Campaign,
        *,
        temperature: float = 0.8,
        max_tokens: int = 3000,
        max_tool_steps: int = 6,
        final_response_instruction: str = "Return your final JSON now.",
    ) -> str:
        """Run a lightweight tool loop for setup LLM calls.

        Supports ``source_browse``, ``memory_search`` (source-scoped), and
        ``name_generate`` so the model can inspect ingested source material
        before producing its final JSON response.  Returns the raw final
        response string.
        """
        if self._completion_port is None:
            return "{}"
        augmented_prompt = user_prompt

        for _step in range(max_tool_steps + 1):
            response = await self._completion_port.complete(
                system_prompt,
                augmented_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not response:
                return "{}"
            response = self._clean_response(response)
            json_text = self._extract_json(response)
            if not json_text:
                return response
            try:
                payload = self._parse_json_lenient(json_text)
            except Exception:
                return response
            if not self._is_tool_call(payload):
                return response

            tool_name = str(payload.get("tool_call") or "").strip()
            tool_result = ""

            if tool_name == "source_browse":
                doc_key = str(payload.get("document_key") or "").strip()[:120]
                wildcard_raw = payload.get("wildcard")
                wildcard = (
                    str(wildcard_raw).strip()[:120]
                    if wildcard_raw is not None
                    else ""
                )
                wildcard_provided = bool(wildcard)
                wildcard = wildcard or "%"
                wildcard_meta = f"wildcard={wildcard!r}"
                if not wildcard_provided:
                    wildcard_meta = "wildcard=(omitted)"
                limit = 255
                try:
                    limit = max(1, min(255, int(payload.get("limit") or 255)))
                except (TypeError, ValueError):
                    pass
                lines = self.browse_source_keys(
                    str(campaign.id),
                    document_key=doc_key or None,
                    wildcard=wildcard,
                    limit=limit,
                )
                if lines:
                    tool_result = (
                        f"SOURCE_BROWSE_RESULT "
                        f"(document_key={doc_key or '*'!r}, "
                        f"{wildcard_meta}, "
                        f"showing {len(lines)}):\n"
                        + "\n".join(lines)
                    )
                else:
                    tool_result = (
                        f"SOURCE_BROWSE_RESULT "
                        f"(document_key={doc_key or '*'!r}, "
                        f"{wildcard_meta}): no entries found"
                    )

            elif tool_name == "memory_search":
                raw_queries = payload.get("queries") or []
                if not raw_queries:
                    legacy = str(payload.get("query") or "").strip()
                    if legacy:
                        raw_queries = [legacy]
                queries = [
                    str(q).strip()[:200]
                    for q in (raw_queries if isinstance(raw_queries, list) else [raw_queries])
                    if str(q or "").strip()
                ][:6]
                category = str(payload.get("category") or "source").strip()
                if not category.startswith("source"):
                    category = "source"
                doc_key_scope = None
                if category.startswith("source:"):
                    doc_key_scope = category.split(":", 1)[1].strip() or None
                before_lines = 0
                after_lines = 0
                try:
                    before_lines = max(0, min(10, int(payload.get("before_lines") or 0)))
                except (TypeError, ValueError):
                    pass
                try:
                    after_lines = max(0, min(10, int(payload.get("after_lines") or 0)))
                except (TypeError, ValueError):
                    pass
                hits: list[str] = []
                for q in queries:
                    results = self.search_source_material(
                        q,
                        str(campaign.id),
                        document_key=doc_key_scope,
                        top_k=5,
                        before_lines=before_lines,
                        after_lines=after_lines,
                    )
                    for doc_k, doc_l, idx, text, score in results:
                        if score >= 0.35:
                            hits.append(f"[{doc_k}#{idx} score={score:.2f}] {text}")
                if hits:
                    tool_result = (
                        "SOURCE_SEARCH_RESULT:\n" + "\n".join(hits[:20])
                    )
                else:
                    tool_result = "SOURCE_SEARCH_RESULT: no relevant hits"

            elif tool_name == "name_generate":
                raw_origins = payload.get("origins") or []
                if isinstance(raw_origins, str):
                    raw_origins = [raw_origins]
                origins = [
                    str(o).strip().lower()
                    for o in raw_origins
                    if str(o or "").strip()
                ][:4]
                ng_gender = str(payload.get("gender") or "both").strip().lower()
                ng_count = 5
                try:
                    ng_count = max(1, min(6, int(payload.get("count") or 5)))
                except (TypeError, ValueError):
                    pass
                ng_context = str(payload.get("context") or "").strip()[:300]
                names = self._fetch_random_names(
                    origins=origins or None,
                    gender=ng_gender,
                    count=ng_count,
                )
                if names:
                    tool_result = (
                        f"NAME_GENERATE_RESULT "
                        f"(origins={origins or 'any'}, gender={ng_gender}):\n"
                        + "\n".join(f"- {n}" for n in names)
                    )
                    if ng_context:
                        tool_result += f"\nEvaluate against: {ng_context}"
                    tool_result += (
                        "\nPick the best fit or call name_generate again "
                        "with different origins/gender."
                    )
                else:
                    tool_result = (
                        f"NAME_GENERATE_RESULT (origins={origins or 'any'}): "
                        "no names returned — try broader origins."
                    )

            else:
                tool_result = (
                    f"UNKNOWN_TOOL: '{tool_name}' is not available during setup. "
                    "Available tools: source_browse, memory_search, name_generate. "
                    f"{final_response_instruction}"
                )

            self._zork_log(
                f"SETUP TOOL LOOP step={_step} tool={tool_name}",
                tool_result[:2000],
            )
            augmented_prompt = f"{augmented_prompt}\n{tool_result}\n"

        # Exhausted steps — force final response.
        augmented_prompt = (
            f"{augmented_prompt}\n"
            f"TOOL_CHAIN_LIMIT: Stop calling tools. {final_response_instruction}\n"
        )
        response = await self._completion_port.complete(
            system_prompt, augmented_prompt, temperature=temperature, max_tokens=max_tokens
        )
        return self._clean_response(response or "{}")

    def _normalize_generated_rulebook_lines(self, raw_text: str) -> list[str]:
        entries: list[str] = []
        current = ""
        for raw_line in str(raw_text or "").splitlines():
            line = " ".join(str(raw_line or "").strip().split())
            if not line:
                continue
            if line.startswith("```"):
                continue
            if re.fullmatch(r"[=\-_*#\s]{3,}", line):
                continue
            if re.match(r"^[A-Z][A-Z0-9-]{1,80}:\s*\S", line):
                if current:
                    entries.append(current)
                current = line
                continue
            if current:
                current = f"{current} {line}".strip()
        if current:
            entries.append(current)
        cleaned: list[str] = []
        for entry in entries:
            compact = re.sub(r"\s+", " ", str(entry or "")).strip()
            if re.match(r"^[A-Z][A-Z0-9-]{1,80}:\s+\S", compact):
                cleaned.append(compact[:8000])
        return cleaned

    def _rulebook_line_key(self, line: object) -> str:
        text = str(line or "").strip()
        if not text:
            return ""
        match = re.match(r"^([A-Z][A-Z0-9-]{1,80}):\s+\S", text)
        if not match:
            return ""
        return str(match.group(1) or "").strip().upper()

    @classmethod
    def communication_rulebook_document_key(cls) -> str:
        return SourceMaterialMemory._normalize_source_document_key(
            cls.COMMUNICATION_RULEBOOK_DOCUMENT_LABEL
        )

    @classmethod
    def _communication_rulebook_lines(cls) -> list[str]:
        return [
            f"{rule_key}: {rule_text}"
            for rule_key, rule_text in cls.DEFAULT_GM_COMMUNICATION_RULES.items()
        ]

    def _canonical_seed_rulebook_lines(
        self,
        campaign_id: str,
        source_payload: dict[str, Any],
    ) -> list[str]:
        docs = source_payload.get("docs") or []
        out: list[str] = []
        seen_keys: set[str] = set()
        auto_key = SourceMaterialMemory._normalize_source_document_key(
            self.AUTO_RULEBOOK_DOCUMENT_LABEL
        )
        communication_key = self.communication_rulebook_document_key()
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            doc_key = str(doc.get("document_key") or "").strip()
            doc_label = str(doc.get("document_label") or "").strip()
            doc_format = str(doc.get("format") or "").strip().lower()
            if doc_format != self.SOURCE_MATERIAL_FORMAT_RULEBOOK:
                continue
            if (
                doc_label == self.AUTO_RULEBOOK_DOCUMENT_LABEL
                or doc_key == auto_key
                or doc_label == self.COMMUNICATION_RULEBOOK_DOCUMENT_LABEL
                or doc_key == communication_key
            ):
                continue
            units = SourceMaterialMemory.get_source_material_document_units(
                str(campaign_id),
                doc_key,
            )
            for unit in units:
                compact = re.sub(r"\s+", " ", str(unit or "").strip()).strip()
                key = self._rulebook_line_key(compact)
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append(compact[:8000])
        return out

    def _merge_generated_rulebook_lines(
        self,
        campaign_id: str,
        source_payload: dict[str, Any],
        generated_lines: list[str],
    ) -> list[str]:
        canonical_lines = self._canonical_seed_rulebook_lines(campaign_id, source_payload)
        merged: list[str] = list(canonical_lines)
        seen_keys = {
            self._rulebook_line_key(line)
            for line in canonical_lines
            if self._rulebook_line_key(line)
        }
        for line in generated_lines:
            compact = re.sub(r"\s+", " ", str(line or "").strip()).strip()
            key = self._rulebook_line_key(compact)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(compact[:8000])
        return merged

    def _auto_rulebook_source_index_hint(self, source_payload: dict[str, Any]) -> str:
        if not source_payload.get("available"):
            return ""
        doc_lines = []
        for doc in source_payload.get("docs") or []:
            if str(doc.get("document_label") or "") == self.AUTO_RULEBOOK_DOCUMENT_LABEL:
                continue
            doc_lines.append(
                f"  - document_key='{doc.get('document_key')}' "
                f"label='{doc.get('document_label')}' "
                f"format='{doc.get('format')}' "
                f"snippets={doc.get('chunk_count')}"
            )
        if not doc_lines:
            return ""
        return (
            "\nEXISTING_SOURCE_INDEX:\n"
            + "\n".join(doc_lines)
            + "\nIf you need canonical facts from these source docs, inspect them before writing the rulebook.\n"
            "Start by enumerating keys with:\n"
            '  {"tool_call": "source_browse"}\n'
            "Then query specific facts with:\n"
            '  {"tool_call": "memory_search", "category": "source", "queries": ["keyword"]}\n'
        )

    async def _generate_campaign_rulebook(
        self,
        campaign: Campaign,
        setup_data: dict[str, Any],
        chosen: dict[str, Any],
        world: dict[str, Any],
    ) -> tuple[int, str]:
        if self._completion_port is None:
            return 0, ""
        attachment_summary = str(setup_data.get("attachment_summary") or "").strip()
        source_payload = self._source_material_prompt_payload(str(campaign.id))
        source_index_hint = self._auto_rulebook_source_index_hint(source_payload)
        source_tool_instructions = ""
        if source_index_hint:
            source_tool_instructions = (
                "\nYou may inspect existing source material before writing the new rulebook.\n"
                "To list all keys in available docs:\n"
                '  {"tool_call": "source_browse"}\n'
                "To browse one doc:\n"
                '  {"tool_call": "source_browse", "document_key": "doc-key"}\n'
                "To filter keys by wildcard:\n"
                '  {"tool_call": "source_browse", "document_key": "doc-key", "wildcard": "char-*"}\n'
                "To semantic-search source material:\n"
                '  {"tool_call": "memory_search", "category": "source", "queries": ["query1", "query2"]}\n'
                "To call a tool, return ONLY the JSON tool_call object. Otherwise return ONLY the final rulebook text.\n"
            )

        genre_context = ""
        genre_pref = setup_data.get("genre_preference")
        if isinstance(genre_pref, dict):
            genre_value = str(genre_pref.get("value") or "").strip()
            if genre_value:
                genre_context = f"\nGenre direction: {genre_value}\n"

        system_prompt = (
            "You convert campaign setup material into a retrievable rulebook for an interactive text adventure.\n"
            "Output ONLY plain text rulebook lines. No markdown. No headers. No bullets. No numbering.\n"
            "Every output line must be fully self-contained and independently retrievable.\n"
            "Format every line exactly as CATEGORY-TAG: fact text\n"
            "Each line should usually be 50-200 words.\n"
            "Convert story summaries, plot chapters, character notes, and attachment prose into reusable rules and facts.\n"
            "Do not write scripts or scene transcripts. Do not rely on adjacent lines for context.\n"
            "Use these category families when relevant: TONE, SCENE, SETTING, CHAR, PLOT, INTERACTION, GM-RULE, MECHANIC, and location-specific tags.\n"
            "For location-specific tags, use descriptive names that identify the narrative function of the place (e.g. FRAME-NARRATIVE-ATTIC, IVORY-TOWER, SWAMPS-OF-SADNESS) rather than generic labels like BLUE-ROOM or RED-ROOM. The tag should be unambiguous even if encountered outside this campaign's context.\n"
            "Existing non-auto rulebook source docs are canonical. If an existing source doc already defines a KEY, do not rewrite or replace that KEY. Only add missing keys or new non-conflicting facts.\n"
            "Required coverage:\n"
            "- TONE, TONE-RULES, SCENE-OPENING, SETTING-[MAIN]\n"
            "- For each named character: CHAR-[NAME], CHAR-[NAME]-PERSONALITY, CHAR-[NAME]-DIALOGUE\n"
            "- For each important plotline: PLOT-[SHORTNAME]\n"
            "- For major cast first impressions: INTERACTION-NEWCOMER-[NAME]\n"
            "- GM-RULE-NO-RAILROADING, GM-RULE-[GENRE]-FIRST, GM-RULE-CHARACTERS-FIRST, GM-RULE-PACING, GM-RULE-NAMES, GM-RULE-NO-RECYCLING-NAMES, GM-RULE-ENSEMBLE, GM-RULE-DIALOGUE-OVER-DESCRIPTION, GM-RULE-ALTERNATIVES\n"
            "If the setting involves intimacy, vulnerability, or explicit consent norms, include TONE-CONSENT and GM-RULE-CONSENT-ENFORCEMENT.\n"
            "If the setting has money, rooms, rentals, or prices, include GM-RULE-MONEY.\n"
            "Dialogue lines must show distinct voice. Running jokes, recurring habits, venue rules, and notable recurring objects should become separate retrievable facts when important.\n"
            "\n"
            "MECHANIC EXTRACTION — trackable resource systems:\n"
            "If the source material contains a system where actions have cumulative costs, track it explicitly. Examples: wishes that cost memories, corruption that grows with power use, sanity that erodes, fuel that depletes, trust that accumulates or decays. For each such system, emit:\n"
            "- MECHANIC-[NAME]: describe the resource, its starting state, what depletes or restores it, and the consequence of exhaustion.\n"
            "- MECHANIC-[NAME]-COST-TABLE: list each action and its specific cost. Be concrete — 'wish for courage costs memory of mother's voice' is useful, 'wishes cost memories' is not.\n"
            "The harness can track these as player_state fields (e.g. player.memories_remaining, player.corruption_level). Name the fields in the MECHANIC entry so the GM knows what to update.\n"
            "\n"
            "NPC ESCALATION BEHAVIORS:\n"
            "For each major NPC, CHAR-[NAME]-PERSONALITY must include not just who they are but what they do when the player stalls, refuses, or doesn't engage. Every NPC who needs something from the player must have an escalation path — what concrete action do they take if the moment passes without player response? Passive NPCs who 'wait' or 'invite' must have an UNLOCKED state: what they do when waiting is no longer an option. If the source material shows this (e.g. a character who forces a story loop, sends an emissary, leaves, attacks, or withdraws an offer), capture that escalation explicitly.\n"
            "\n"
            "CONVERSATION-TERMINAL NPCs:\n"
            "Some NPCs do not engage in extended dialogue. If a character's defining trait is apathy, nihilism, hostility, or inscrutability, say so explicitly in their CHAR entry and add: 'This NPC answers once — poorly, reluctantly, or cryptically — and that is all the player gets. Do not let the player re-ask or interrogate. The NPC does not care enough to test the player or withhold strategically. They gave their answer; it was bad; move on.'\n"
            "\n"
            "CONFRONTATION SINCERITY:\n"
            "For NPCs who challenge the player philosophically or morally (villains, tempters, nihilists, gatekeepers), add a CHAR-[NAME]-CONFRONTATION note. The GM must engage with the sincerity of the player's defiance even when the words are plain. 'You're wrong' spoken with conviction is a valid response. The NPC should not dismiss simple sincerity as naive or re-ask until the player delivers a philosophically sophisticated rebuttal. The NPC reacts to the stance, not the eloquence.\n"
            "Avoid generic AI-default names for any new characters. Ban list: Morgan, Kai, River, Sage, Quinn, Riley, Jordan, Avery, Harper, Rowan, Blake, Skyler, Ash, Nova, Zara, Milo, Ezra, Luna, Marcus; surnames: Chen, Mendoza, Nakamura, Patel, Rollins, Kim, Santos, Okafor, Volkov, Johansson, Delacroix, Venn, Sands, Kade, Park.\n"
            "Preserve player agency, kindness, and genre tone. Unless the genre explicitly demands otherwise, do not invent trauma hooks or coercive plot pressure.\n"
            f"{source_tool_instructions}"
        )
        user_prompt = (
            f"Generate a rulebook for campaign '{setup_data.get('raw_name') or campaign.name}'.\n"
            f"{genre_context}"
            f"{source_index_hint}"
            "Use the chosen storyline, expanded world JSON, and any detailed attachment summary below.\n"
            "If the attachment summary is a story-generator prompt or setup note, translate it into concise retrievable rulebook facts instead of copying it as prose.\n"
            "If existing source docs contain canonical facts, merge them faithfully into this synthesized rulebook. Existing user-provided rulebook facts always win conflicts by KEY; only supplement them.\n\n"
            f"Chosen storyline:\n{self._dump_json(chosen)}\n\n"
            f"Expanded world JSON:\n{self._dump_json(world)}\n\n"
            f"Detailed attachment summary:\n{attachment_summary or '(none)'}\n"
        )
        self._zork_log(
            f"SETUP RULEBOOK GENERATION campaign={campaign.id}",
            f"--- SYSTEM ---\n{system_prompt}\n--- USER ---\n{user_prompt}",
        )
        try:
            response = await self._setup_tool_loop(
                system_prompt,
                user_prompt,
                campaign,
                temperature=0.5,
                max_tokens=self.AUTO_RULEBOOK_MAX_TOKENS,
                final_response_instruction="Return your final rulebook text now.",
            )
        except Exception as exc:
            self._logger.warning("Campaign rulebook generation failed: %s", exc)
            self._zork_log("SETUP RULEBOOK GENERATION FAILED", str(exc))
            return 0, ""
        self._zork_log("SETUP RULEBOOK RAW RESPONSE", response or "(empty)")
        normalized_lines = self._normalize_generated_rulebook_lines(response or "")
        normalized_lines = self._merge_generated_rulebook_lines(
            str(campaign.id),
            source_payload,
            normalized_lines,
        )
        if not normalized_lines:
            return 0, ""
        stored_count, document_key = self.ingest_source_material_text(
            str(campaign.id),
            document_label=self.AUTO_RULEBOOK_DOCUMENT_LABEL,
            text="\n".join(normalized_lines),
            source_format=self.SOURCE_MATERIAL_FORMAT_RULEBOOK,
            replace_document=True,
        )
        if stored_count <= 0:
            self._zork_log(
                "SETUP RULEBOOK INGEST FAILED",
                f"document_key={document_key!r}",
            )
            return 0, ""
        return len(normalized_lines), document_key

    def _source_material_export_text(
        self,
        document_key: str,
        units: list[str],
    ) -> str:
        clean_units = [str(unit or "").strip() for unit in units if str(unit or "").strip()]
        if not clean_units:
            return ""
        sample = "\n".join(clean_units[:6])
        inferred_format = self._source_material_format_heuristic(sample)
        if inferred_format == self.SOURCE_MATERIAL_FORMAT_RULEBOOK:
            return "\n".join(clean_units).strip()
        return "\n\n".join(clean_units).strip()

    def _source_material_export_filename(
        self,
        document_key: str,
        document_label: str | None = None,
        *,
        used_names: set[str] | None = None,
    ) -> str:
        label = " ".join(str(document_label or "").strip().split())
        if label:
            base = label[:180]
        else:
            key = str(document_key or "").strip().lower()
            if not key:
                key = "source-material"
            base = SourceMaterialMemory._normalize_source_document_key(key) or "source-material"
            base = base[:180]
        filename = f"{base}.txt"
        if used_names is None:
            return filename
        if filename not in used_names:
            used_names.add(filename)
            return filename
        fallback_key = (
            SourceMaterialMemory._normalize_source_document_key(str(document_key or "").strip())
            or "source-material"
        )[:80]
        suffix = 2
        while True:
            candidate = f"{base} ({fallback_key}-{suffix}).txt"
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            suffix += 1

    def _campaign_export_transcript(self, campaign: Campaign) -> str:
        with self._session_factory() as session:
            turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == str(campaign.id))
                .order_by(Turn.id.asc())
                .all()
            )
        registry = self._campaign_player_registry(str(campaign.id), self._session_factory)
        by_actor_id = registry.get("by_actor_id", {})
        lines: list[str] = []
        for turn in turns:
            content = str(turn.content or "").strip()
            if not content:
                continue
            if turn.kind == "narrator":
                content = self._strip_ephemeral_context_lines(content)
                content = self._strip_narration_footer(content)
            if not content:
                continue
            if turn.kind == "player":
                entry = by_actor_id.get(str(turn.actor_id or "")) or {}
                name = str(entry.get("name") or f"Player {turn.actor_id}").strip()
                lines.append(f"[TURN {turn.id}] PLAYER {name}: {content}")
            elif turn.kind == "narrator":
                lines.append(f"[TURN {turn.id}] NARRATOR: {content}")
            else:
                lines.append(f"[TURN {turn.id}] {str(turn.kind or 'system').upper()}: {content}")
        return "\n".join(lines).strip()

    def _campaign_export_turn_events(
        self,
        campaign: Campaign,
    ) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            turns = (
                session.query(Turn)
                .filter(Turn.campaign_id == str(campaign.id))
                .order_by(Turn.id.asc())
                .all()
            )
        registry = self._campaign_player_registry(str(campaign.id), self._session_factory)
        by_actor_id = registry.get("by_actor_id", {})
        events: list[dict[str, Any]] = []
        for turn in turns:
            meta = parse_json_dict(turn.meta_json)
            if not isinstance(meta, dict):
                meta = {}
            actor_entry = by_actor_id.get(str(turn.actor_id or "")) or {}
            events.append(
                {
                    "turn_id": int(turn.id),
                    "created_at": turn.created_at.isoformat() if turn.created_at else None,
                    "kind": str(turn.kind or ""),
                    "actor_id": str(turn.actor_id or "") or None,
                    "player_name": str(actor_entry.get("name") or "").strip() or None,
                    "player_slug": str(actor_entry.get("slug") or "").strip() or None,
                    "session_id": str(turn.session_id or "") or None,
                    "external_message_id": str(turn.external_message_id or "") or None,
                    "external_user_message_id": str(turn.external_user_message_id or "") or None,
                    "content": str(turn.content or ""),
                    "meta": meta,
                }
            )
        return events

    async def _generate_campaign_export_digest(
        self,
        campaign: Campaign,
        transcript: str,
    ) -> str:
        ordered_chunk_digest = await self._summarise_long_text(
            transcript,
            summary_instructions=(
                "This is a complete campaign transcript from the first turn to the latest turn. "
                "Preserve the entire story arc in chronological order. Do not collapse the story into a vague world-state summary. "
                "Track what happens early, middle, and late; major and minor arcs; character relationship changes; discoveries; "
                "state changes; travel; inventory/item changes that mattered; time jumps; player-to-player dynamics; private reveals "
                "that later became relevant; NPC attitude shifts; recurring jokes; and unresolved threads. "
                "When facts conflict, preserve both versions if needed but clearly favor the later explicit outcome. "
                "Be comprehensive and concrete."
            ),
            allow_single_chunk_passthrough=False,
        )
        ordered_chunk_digest = str(ordered_chunk_digest or "").strip()
        if not ordered_chunk_digest:
            return ""
        digest_system = (
            "You convert an ordered set of campaign transcript summaries into a faithful whole-campaign digest.\n"
            "Output ONLY plain text.\n"
            "This digest must cover the entire story arc from first turn to last turn without flattening it into a generic setting summary.\n"
            "Use these exact plain-text section labels:\n"
            "FULL ARC OVERVIEW\n"
            "CHRONOLOGICAL STORY BEATS\n"
            "PLAYER THREADS\n"
            "NPC ARCS\n"
            "RELATIONSHIPS AND REVEALS\n"
            "LOCATIONS ITEMS AND STATE CHANGES\n"
            "OPEN THREADS AND AFTERMATH\n"
            "CURRENT END STATE\n"
            "CONFLICT RESOLUTION NOTES\n"
            "Requirements:\n"
            "- preserve chronology from opening setup to ending state\n"
            "- name the real player characters and major NPCs repeatedly where relevant\n"
            "- mention major chapter/scene transitions when known\n"
            "- include all lasting arcs, even if they seemed small at the time\n"
            "- if two facts conflict, choose the most sensible truthful version and say why in CONFLICT RESOLUTION NOTES\n"
            "- multiplayer campaigns are ensemble stories, not single-protagonist stories\n"
            "- do not write fiction prose; write a precise reconstruction digest"
        )
        digest_user = (
            f"Campaign: {campaign.name}\n\n"
            "ORDERED TRANSCRIPT SUMMARY:\n"
            f"{ordered_chunk_digest}\n"
        )
        digest_text = await self._new_gpt(campaign=campaign).turbo_completion(
            digest_system,
            digest_user,
            temperature=0.3,
            max_tokens=12000,
        )
        digest_text = str(digest_text or "").strip()
        return digest_text or ordered_chunk_digest

    def _campaign_raw_export_filename(self, raw_format: str) -> str:
        fmt = str(raw_format or "jsonl").strip().lower()
        if fmt == "json":
            return "campaign-raw.json"
        if fmt == "markdown":
            return "campaign-raw-markdown.md"
        if fmt == "script":
            return "campaign-raw-script.txt"
        if fmt == "loglines":
            return "campaign-raw-loglines.txt"
        return "campaign-raw.jsonl"

    def _render_campaign_raw_jsonl(
        self,
        campaign: Campaign,
        events: list[dict[str, Any]],
    ) -> str:
        rows: list[dict[str, Any]] = [
            {
                "type": "campaign",
                "campaign_id": str(campaign.id),
                "campaign_name": str(campaign.name or ""),
                "created_at": campaign.created_at.isoformat() if campaign.created_at else None,
                "updated_at": campaign.updated_at.isoformat() if campaign.updated_at else None,
            }
        ]
        rows.extend({"type": "turn", **event} for event in events)
        return "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
        ).strip()

    def _render_campaign_raw_json(
        self,
        campaign: Campaign,
        events: list[dict[str, Any]],
    ) -> str:
        payload = {
            "campaign": {
                "id": str(campaign.id),
                "name": str(campaign.name or ""),
                "created_at": campaign.created_at.isoformat() if campaign.created_at else None,
                "updated_at": campaign.updated_at.isoformat() if campaign.updated_at else None,
            },
            "events": events,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    def _render_campaign_raw_markdown(
        self,
        campaign: Campaign,
        events: list[dict[str, Any]],
    ) -> str:
        lines = [
            f"# Campaign Raw Export: {campaign.name}",
            "",
            "## Table of Contents",
            "",
            "- [Campaign Metadata](#campaign-metadata)",
        ]
        for event in events:
            lines.append(f"- [Turn {event.get('turn_id')}](#turn-{event.get('turn_id')})")
        lines.extend(
            [
                "",
                "## Campaign Metadata",
                "",
                f"- Campaign ID: `{campaign.id}`",
                f"- Created: `{campaign.created_at.isoformat() if campaign.created_at else ''}`",
                f"- Updated: `{campaign.updated_at.isoformat() if campaign.updated_at else ''}`",
            ]
        )
        for event in events:
            lines.extend(
                [
                    "",
                    f"## Turn {event.get('turn_id')}",
                    "",
                    f"- Kind: `{event.get('kind')}`",
                    f"- Timestamp: `{event.get('created_at') or ''}`",
                    f"- Actor ID: `{event.get('actor_id') or ''}`",
                    f"- Player: `{event.get('player_name') or ''}`",
                    f"- Player Slug: `{event.get('player_slug') or ''}`",
                    f"- Session ID: `{event.get('session_id') or ''}`",
                    "",
                    "### Content",
                    "",
                    "```text",
                    str(event.get("content") or ""),
                    "```",
                    "",
                    "### Meta",
                    "",
                    "```json",
                    json.dumps(event.get("meta") or {}, ensure_ascii=False, indent=2, sort_keys=True),
                    "```",
                ]
            )
        return "\n".join(lines).strip()

    def _render_campaign_raw_script(
        self,
        campaign: Campaign,
        events: list[dict[str, Any]],
    ) -> str:
        lines = [
            f"CAMPAIGN\t{campaign.name}",
            f"\tID\t{campaign.id}",
            f"\tCREATED\t{campaign.created_at.isoformat() if campaign.created_at else ''}",
            f"\tUPDATED\t{campaign.updated_at.isoformat() if campaign.updated_at else ''}",
        ]
        for event in events:
            lines.append("")
            lines.append(f"TURN\t{event.get('turn_id')}")
            lines.append(f"\tKIND\t{event.get('kind')}")
            lines.append(f"\tTIMESTAMP\t{event.get('created_at') or ''}")
            lines.append(f"\tACTOR_ID\t{event.get('actor_id') or ''}")
            lines.append(f"\tPLAYER\t{event.get('player_name') or ''}")
            lines.append(f"\tPLAYER_SLUG\t{event.get('player_slug') or ''}")
            lines.append(f"\tSESSION_ID\t{event.get('session_id') or ''}")
            lines.append("\tCONTENT")
            for row in str(event.get("content") or "").splitlines() or [""]:
                lines.append(f"\t\t{row}")
            lines.append("\tMETA")
            meta_text = json.dumps(event.get("meta") or {}, ensure_ascii=False, indent=2, sort_keys=True)
            for row in meta_text.splitlines():
                lines.append(f"\t\t{row}")
        return "\n".join(lines).strip()

    def _render_campaign_raw_loglines(
        self,
        campaign: Campaign,
        events: list[dict[str, Any]],
    ) -> str:
        lines = [
            f"[CAMPAIGN EXPORT] campaign={campaign.id} name={campaign.name!r} turns={len(events)}"
        ]
        for event in events:
            label = str(event.get("kind") or "event").upper()
            if event.get("kind") == "player":
                player_name = str(event.get("player_name") or "").strip()
                if player_name:
                    label = f"PLAYER {player_name}"
            elif event.get("kind") == "narrator":
                label = "NARRATOR"
            lines.append(
                f"[TURN #{event.get('turn_id')} | {event.get('created_at') or ''}] {label}: "
                f"{str(event.get('content') or '').strip()}"
            )
        return "\n".join(lines).strip()

    async def _generate_campaign_raw_export_artifacts(
        self,
        campaign: Campaign,
        *,
        raw_format: str = "jsonl",
    ) -> dict[str, str]:
        fmt = str(raw_format or "jsonl").strip().lower()
        if fmt not in {"script", "markdown", "json", "jsonl", "loglines"}:
            fmt = "jsonl"
        events = self._campaign_export_turn_events(campaign)
        if not events:
            return {}
        if fmt == "json":
            text = self._render_campaign_raw_json(campaign, events)
        elif fmt == "markdown":
            text = self._render_campaign_raw_markdown(campaign, events)
        elif fmt == "script":
            text = self._render_campaign_raw_script(campaign, events)
        elif fmt == "loglines":
            text = self._render_campaign_raw_loglines(campaign, events)
        else:
            text = self._render_campaign_raw_jsonl(campaign, events)
        if not str(text or "").strip():
            return {}
        return {self._campaign_raw_export_filename(fmt): str(text or "").strip()}

    async def _generate_campaign_export_artifacts(
        self,
        campaign: Campaign,
    ) -> dict[str, str]:
        transcript = self._campaign_export_transcript(campaign)
        if not transcript:
            return {}
        campaign_state = self.get_campaign_state(campaign)
        characters = self.get_campaign_characters(campaign)
        campaign_players = self._campaign_players_for_prompt(str(campaign.id), limit=24)
        story_outline = campaign_state.get("story_outline") if isinstance(campaign_state, dict) else {}
        plot_threads = campaign_state.get("plot_threads") if isinstance(campaign_state, dict) else []
        chapter_plan = campaign_state.get("chapters") if isinstance(campaign_state, dict) else []
        consequences = campaign_state.get("consequences") if isinstance(campaign_state, dict) else []
        model_state = self._build_model_state(campaign_state if isinstance(campaign_state, dict) else {})
        model_state = self._fit_state_to_budget(model_state, self.MAX_STATE_CHARS)
        export_summary = await self._generate_campaign_export_digest(campaign, transcript)
        if not export_summary:
            export_summary = await self._summarise_long_text(
                transcript,
                summary_instructions=(
                    "Summarise this full campaign playthrough faithfully for export. Preserve lasting facts, "
                    "character arcs, relationship changes, major reveals, locations, items, chapter beats, "
                    "timeline changes, unresolved threads, and the current open state. "
                    "When facts conflict, prefer the later explicit outcome and the persisted world state."
                ),
                allow_single_chunk_passthrough=False,
            )
        export_summary = str(export_summary or "").strip()
        export_summary_excerpt = export_summary
        if len(export_summary_excerpt) > 32000:
            export_summary_excerpt = export_summary_excerpt[:32000].rsplit(" ", 1)[0].strip() + "\n...[truncated excerpt for prompt budget]"
        transcript_excerpt = transcript
        if len(transcript_excerpt) > 20000:
            transcript_excerpt = transcript_excerpt[:20000].rsplit(" ", 1)[0].strip() + "\n...[truncated excerpt for prompt budget]"
        source_payload = self._source_material_prompt_payload(str(campaign.id))
        source_index_hint = self._auto_rulebook_source_index_hint(source_payload)
        source_tool_instructions = ""
        if source_index_hint:
            source_tool_instructions = (
                "\nYou may inspect existing source material while resolving canon.\n"
                'To list keys: {"tool_call": "source_browse"}\n'
                'To browse one doc: {"tool_call": "source_browse", "document_key": "doc-key"}\n'
                'To search canon: {"tool_call": "memory_search", "category": "source", "queries": ["keyword1", "keyword2"]}\n'
                "To call a tool, return ONLY the JSON tool_call object. Otherwise return ONLY the requested final text.\n"
            )
        export_context = {
            "campaign_name": campaign.name,
            "campaign_summary": campaign.summary or "",
            "story_outline": story_outline,
            "chapter_plan": chapter_plan,
            "plot_threads": plot_threads,
            "consequences": consequences,
            "current_state": model_state,
            "characters": characters,
            "campaign_players": campaign_players,
        }
        rulebook_system = (
            "You convert a completed campaign playthrough into a retrievable rulebook for recreating that tale.\n"
            "Output ONLY plain text rulebook lines. No markdown. No headers. No bullets. No numbering.\n"
            "Every line must be fully self-contained and use exactly CATEGORY-TAG: fact text\n"
            "Use category families such as TONE, SCENE, SETTING, CHAR, PLOT, INTERACTION, GM-RULE, and venue-specific tags.\n"
            "Treat WORLD_CHARACTERS as NPC-only and CAMPAIGN_PLAYERS as real human player characters. "
            "In multiplayer campaigns there is no single main character; preserve ensemble structure.\n"
            "Conflict resolution priority:\n"
            "1. Persisted current state / current character roster / current chapter state\n"
            "2. Later explicit turn outcomes in the playthrough summary\n"
            "3. Repeated consistent facts across the transcript\n"
            "4. Existing source material for unchanged background canon\n"
            "If a fact remains uncertain, omit it or phrase it cautiously instead of inventing certainty.\n"
            "Preserve major arcs, resolved outcomes, and unresolved threads so the tale can be recreated faithfully.\n"
            "Do not output a generic world summary. Output dense factual rulebook lines only.\n"
            f"{source_tool_instructions}"
        )
        rulebook_user = (
            f"Generate a campaign rulebook export for '{campaign.name}'.\n"
            f"{source_index_hint}"
            "Use the full playthrough summary and current campaign data below.\n"
            "This export should describe how to faithfully recreate the tale as it was actually played, not just the initial setup.\n\n"
            f"PLAYTHROUGH ARC DIGEST:\n{export_summary_excerpt or '(none)'}\n\n"
            f"EARLY TRANSCRIPT EXCERPT:\n{transcript_excerpt or '(none)'}\n\n"
            f"CAMPAIGN DATA:\n{self._dump_json(export_context)}\n"
        )
        self._zork_log(
            f"CAMPAIGN EXPORT RULEBOOK campaign={campaign.id}",
            f"--- SYSTEM ---\n{rulebook_system}\n--- USER ---\n{rulebook_user}",
        )
        rulebook_response = await self._setup_tool_loop(
            rulebook_system,
            rulebook_user,
            campaign,
            temperature=0.4,
            max_tokens=self.AUTO_RULEBOOK_MAX_TOKENS,
            final_response_instruction="Return your final rulebook text now.",
        )
        rulebook_lines = self._normalize_generated_rulebook_lines(rulebook_response or "")
        if len(rulebook_lines) < 12:
            repair_system = (
                "You repair campaign export drafts into proper retrievable rulebook lines.\n"
                "Output ONLY plain text rulebook lines.\n"
                "Every line must be exactly CATEGORY-TAG: fact text\n"
                "No prose paragraphs. No markdown. No headers.\n"
                "Preserve chronology-derived facts, arcs, characters, plotlines, interactions, and GM rules.\n"
                "If the draft is a summary instead of a rulebook, convert it into many factual rulebook lines."
            )
            repair_user = (
                f"Repair this campaign export into a rulebook for '{campaign.name}'.\n\n"
                f"DRAFT EXPORT:\n{str(rulebook_response or '').strip() or '(empty)'}\n\n"
                f"PLAYTHROUGH ARC DIGEST:\n{export_summary_excerpt or '(none)'}\n\n"
                f"CAMPAIGN DATA:\n{self._dump_json(export_context)}\n"
            )
            self._zork_log(
                f"CAMPAIGN EXPORT RULEBOOK REPAIR campaign={campaign.id}",
                f"--- SYSTEM ---\n{repair_system}\n--- USER ---\n{repair_user}",
            )
            repaired = await self._new_gpt(campaign=campaign).turbo_completion(
                repair_system,
                repair_user,
                temperature=0.3,
                max_tokens=self.AUTO_RULEBOOK_MAX_TOKENS,
            )
            repaired_lines = self._normalize_generated_rulebook_lines(repaired or "")
            if repaired_lines:
                rulebook_lines = repaired_lines
        rulebook_text = "\n".join(rulebook_lines).strip()
        story_prompt_system = (
            "You convert a completed campaign playthrough into a reusable story generator prompt.\n"
            "Output ONLY plain text. No markdown fences.\n"
            "Write a prompt that could recreate the same campaign faithfully: tone, setting, cast, arcs, open threads, "
            "facts, and the current shape of the story.\n"
            "Use clear section labels in plain text such as TITLE, GENRE, FORMAT, PLAY MODE, SETTING, PLAYER CHARACTERS, "
            "NPC CAST, CANON FACTS, MAJOR ARCS, RELATIONSHIPS, OPEN THREADS, OPENING/START STATE, and RECREATION RULES.\n"
            "If this was multiplayer, state clearly that it is an ensemble campaign with multiple real player characters and no single protagonist.\n"
            "Resolve conflicts using the same priority order as the rulebook export: persisted current state, later explicit outcomes, repeated consistent facts, then source canon.\n"
            "Do not write prose fiction. Write a practical generator prompt for reconstructing the campaign.\n"
            "This must reflect the whole story arc from first turn to last turn, not just the ending state."
        )
        story_prompt_user = (
            f"Generate a story generator prompt export for '{campaign.name}'.\n"
            "This should function like a canonical recreation prompt for the whole played campaign.\n\n"
            f"PLAYTHROUGH ARC DIGEST:\n{export_summary_excerpt or '(none)'}\n\n"
            f"EARLY TRANSCRIPT EXCERPT:\n{transcript_excerpt or '(none)'}\n\n"
            f"CAMPAIGN DATA:\n{self._dump_json(export_context)}\n"
        )
        self._zork_log(
            f"CAMPAIGN EXPORT STORY PROMPT campaign={campaign.id}",
            f"--- SYSTEM ---\n{story_prompt_system}\n--- USER ---\n{story_prompt_user}",
        )
        story_prompt_text = await self._new_gpt(campaign=campaign).turbo_completion(
            story_prompt_system,
            story_prompt_user,
            temperature=0.5,
            max_tokens=6000,
        )
        story_prompt_text = str(story_prompt_text or "").strip()
        out: dict[str, str] = {}
        if rulebook_text:
            out["campaign-rulebook.txt"] = rulebook_text
        if story_prompt_text:
            out["campaign-story-prompt.txt"] = story_prompt_text
        return out

    async def campaign_export(
        self,
        campaign_id: str,
        *,
        export_type: str = "full",
        raw_format: str = "jsonl",
    ) -> dict[str, str]:
        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                raise KeyError(f"Unknown campaign: {campaign_id}")
        export_type_clean = str(export_type or "full").strip().lower()
        if export_type_clean not in {"full", "raw"}:
            export_type_clean = "full"
        if export_type_clean == "raw":
            export_files = await self._generate_campaign_raw_export_artifacts(
                campaign,
                raw_format=raw_format,
            )
        else:
            export_files = await self._generate_campaign_export_artifacts(campaign)
        docs = self.list_source_material_documents(str(campaign.id), limit=200)
        source_export_files: dict[str, str] = {}
        used_names = set(export_files.keys())
        for row in docs:
            document_key = str(row.get("document_key") or "").strip()
            document_label = str(row.get("document_label") or "").strip()
            if not document_key:
                continue
            units = SourceMaterialMemory.get_source_material_document_units(
                str(campaign.id),
                document_key,
            )
            export_text = self._source_material_export_text(document_key, units)
            if not export_text:
                continue
            filename = self._source_material_export_filename(
                document_key,
                document_label,
                used_names=used_names,
            )
            source_export_files[filename] = export_text
        export_files.update(source_export_files)
        return export_files

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

        # Build source material index hint if docs are available.
        source_payload = self._source_material_prompt_payload(str(campaign.id))
        source_index_hint = ""
        if source_payload.get("available"):
            docs = source_payload.get("docs") or []
            doc_formats = {
                str(doc.get("format") or "generic").strip().lower() for doc in docs
            }
            has_rulebook = "rulebook" in doc_formats
            doc_lines = []
            for doc in docs:
                doc_lines.append(
                    f"  - document_key='{doc.get('document_key')}' "
                    f"label='{doc.get('document_label')}' "
                    f"format='{doc.get('format')}' "
                    f"snippets={doc.get('chunk_count')}"
                )
            browse_instruction = (
                "  Start by enumerating source keys so you know what is available:"
                "\n  {\"tool_call\": \"source_browse\"}\n"
                "  Then query only what you need with memory_search.\n"
            )
            if has_rulebook:
                browse_instruction = (
                    "  Mandatory first step (before any semantic search):"
                    "\n  {\"tool_call\": \"source_browse\"}\n"
                    "  (omit wildcard/document filters on this first pass to list all keys).\n"
                )
            source_index_hint = (
                "\nSOURCE_MATERIAL_INDEX: "
                f"{source_payload.get('document_count')} document(s), "
                f"{source_payload.get('chunk_count')} total snippet(s).\n"
                + "\n".join(doc_lines)
                + "\nIMPORTANT: Before generating variants, browse the source material to understand "
                "characters, locations, tone, and rules.\n"
                + browse_instruction
                + "Then drill into specific entries with:\n"
                '  {"tool_call": "memory_search", "category": "source", "queries": ["keyword"]}\n'
                "Only return your final variants JSON after you have reviewed the source material.\n"
                "If any source document is rulebook-formatted, do not skip source_browse for keys.\n"
            )

        name_tool_instructions = (
            "\nYou have a name_generate tool for culturally-appropriate character names.\n"
            "To generate names filtered by origin:\n"
            '  {"tool_call": "name_generate", "origins": ["italian"], "gender": "f", "context": "tough bouncer"}\n'
            "To call a tool, return ONLY the JSON tool_call object (no other keys). "
            "You will receive the results and can call more tools or return your final response.\n"
            "Use name_generate for ALL new original characters instead of inventing names.\n"
        )
        source_tool_instructions = name_tool_instructions
        if source_payload.get("available"):
            docs = source_payload.get("docs") or []
            has_only_generic = all(
                str(doc.get("format") or "generic").strip().lower() == "generic"
                for doc in docs
            )
            if has_only_generic:
                source_tool_instructions = (
                    "\nYou have tools for source-material exploration, but this source material "
                    "is currently classified as generic and already summarized in attachment text.\n"
                    "Only call source tools when you need exact wording beyond the summary:\n"
                    '  {"tool_call": "memory_search", "category": "source", "queries": ["keyword"]}\n'
                    "To generate culturally-appropriate character names:\n"
                    '  {"tool_call": "name_generate", "origins": ["italian"], "gender": "f", "context": "tough bouncer"}\n'
                    "To call a tool, return ONLY the JSON tool_call object (no other keys). "
                    "You will receive the results and can call more tools or return your final response.\n"
                    "Use name_generate for ALL new original characters instead of inventing names.\n"
                )
            else:
                source_tool_instructions = (
                    "\nYou have tools to inspect ingested source material before generating your response.\n"
                    "MANDATORY: first, enumerate keys before semantic search:\n"
                    '  {"tool_call": "source_browse"}\n'
                    "(omit wildcard on first pass; do not filter yet).\n"
                    "If you need one document only:\n"
                    '  {"tool_call": "source_browse", "document_key": "doc-key"}\n'
                    "After browsing, drill into specifics:\n"
                    '  {"tool_call": "memory_search", "category": "source", "queries": ["query1", "query2"]}\n'
                    "To filter entries by wildcard only after initial listing:\n"
                    '  {"tool_call": "source_browse", "wildcard": "keyword*"}\n'
                    "To generate culturally-appropriate character names:\n"
                    '  {"tool_call": "name_generate", "origins": ["italian"], "gender": "f", "context": "tough bouncer"}\n'
                    "To call a tool, return ONLY the JSON tool_call object (no other keys). "
                    "You will receive the results and can call more tools or return your final response.\n"
                    "ALWAYS browse source material before generating variants — "
                    "the summary alone may not capture all characters, rules, or locations.\n"
                    "Use name_generate for ALL new original characters instead of inventing names.\n"
                )

        variants: list[dict[str, Any]] = []
        result: dict[str, Any] = {}
        if self._completion_port is not None:
            system_prompt = (
                "You are a creative game designer who builds interactive text-adventure campaigns.\n"
                "For non-canonical/original characters, choose distinctive specific names; avoid generic defaults "
                "(Morgan, Chen, Mendoza, Rollins, Nakamura, Kai, River) unless source canon requires them.\n"
                f"{source_tool_instructions}"
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
            genre_context = ""
            genre_pref = setup_data.get("genre_preference")
            if isinstance(genre_pref, dict):
                genre_value = str(genre_pref.get("value") or "").strip()
                genre_kind = str(genre_pref.get("kind") or "").strip().lower()
                if genre_value:
                    if genre_kind == "custom":
                        genre_context = (
                            "\nGenre direction (custom):\n"
                            f"{genre_value}\n"
                            "Treat this as a hard style/tone preference while staying coherent.\n"
                        )
                    else:
                        genre_context = (
                            f"\nGenre direction: {genre_value}\n"
                            "Prioritize this tone and genre conventions in all variants.\n"
                        )

            if is_known:
                user_prompt = (
                    f"Generate 2-3 storyline variants for an interactive text-adventure campaign "
                    f"based on the {work_type}: '{raw_name}'.\n"
                    f"Description: {work_desc}\n"
                    f"{imdb_context}"
                    f"{attachment_context}"
                    f"{source_index_hint}"
                    f"{genre_context}"
                    f"{guidance_context}"
                    "Use actual characters, locations, and plot points from the source work."
                )
            else:
                user_prompt = (
                    f"Generate 2-3 storyline variants for an original text-adventure campaign "
                    f"called '{raw_name}'.\n"
                    f"{attachment_context}"
                    f"{source_index_hint}"
                    f"{genre_context}"
                    f"{guidance_context}"
                    "Each variant should have a different tone, central conflict, or protagonist archetype. "
                    "Be creative and specific with character names and chapter titles."
                )

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
                            f"{user_prompt}\n\n"
                            "FORMAT REPAIR: Your previous response was invalid or incomplete JSON. "
                            "Return ONLY one valid JSON object with key 'variants' and no trailing text."
                        )
                        self._zork_log(
                            f"SETUP VARIANT RETRY campaign={campaign.id}", cur_user
                        )
                    response = await self._setup_tool_loop(
                        system_prompt,
                        cur_user,
                        campaign,
                        temperature=0.8,
                        max_tokens=3000,
                    )
                    self._zork_log("SETUP VARIANT RAW RESPONSE", response or "(empty)")
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

        if "on_rails" not in prefs:
            # Step 1: parse on-rails answer, ask puzzle question
            if answer in ("on-rails", "onrails", "on rails", "rails", "strict"):
                prefs["on_rails"] = True
            else:
                prefs["on_rails"] = False
            setup_data["novel_preferences"] = prefs
            state["setup_data"] = setup_data
            return (
                "2. **Puzzle encounters?** Should the campaign include mechanical challenges "
                "(dice rolls, riddles, mini-games)?\n"
                "Options: **none** / **light** (environmental puzzles only) / "
                "**moderate** (+ skill checks & riddles) / **full** (+ mini-games)\n"
            )
        else:
            # Step 2: parse puzzle mode, finalize
            puzzle_mode = "none"
            if answer in ("light", "environmental"):
                puzzle_mode = "light"
            elif answer in ("moderate", "mod", "skill", "skill checks", "riddles"):
                puzzle_mode = "moderate"
            elif answer in ("full", "all", "yes", "heavy", "mini-games", "minigames"):
                puzzle_mode = "full"
            prefs["puzzle_mode"] = puzzle_mode
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

    @classmethod
    def _setup_genre_prompt(cls) -> str:
        lines = ["Choose a genre direction before I generate variants:\n"]
        for idx, (genre, description) in enumerate(cls.SETUP_GENRE_TEMPLATES.items(), 1):
            lines.append(f"{idx}. **{genre}** — {description}")
        lines.append(
            "\nReply with a number or exact genre name.\n"
            "For custom direction, reply `custom: <your genre description>`."
        )
        return "\n".join(lines)

    @classmethod
    def _parse_setup_genre_choice(
        cls, content: str
    ) -> tuple[dict[str, str] | None, str | None]:
        raw = str(content or "").strip()
        if not raw:
            return None, "Please choose a genre."

        genre_keys = list(cls.SETUP_GENRE_TEMPLATES.keys())

        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(genre_keys):
                genre = genre_keys[idx - 1]
                return {"kind": "template", "value": genre}, None
            return None, f"Please choose a number between 1 and {len(genre_keys)}."

        lowered = raw.lower().strip()
        if lowered.startswith("custom:") or lowered.startswith("other:"):
            custom = raw.split(":", 1)[1].strip()
            if len(custom) < 3:
                return None, "Custom genre is too short. Add a bit more detail."
            return {"kind": "custom", "value": custom[:200]}, None

        normalized = lowered.replace("_", "-").replace(" ", "-")
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
        if normalized in cls.SETUP_GENRE_TEMPLATES:
            return {"kind": "template", "value": normalized}, None

        return {"kind": "custom", "value": raw[:200]}, None

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
            use_imdb_cfg = setup_data.get("use_imdb")
            use_imdb_effective = (
                bool(use_imdb_cfg)
                if isinstance(use_imdb_cfg, bool)
                else False
            )
            if not bool(setup_data.get("imdb_opt_in_explicit")):
                use_imdb_effective = False
            imdb_results = (
                []
                if not use_imdb_effective
                else self._imdb_search(answer, max_results=3)
            )
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

            if (
                use_imdb_effective
                and not setup_data["is_known_work"]
                and imdb_results
                and not novel_intent
            ):
                top = imdb_results[0]
                setup_data["is_known_work"] = True
                setup_data["raw_name"] = top.get("title") or setup_data["raw_name"]
                setup_data["work_type"] = (str(top.get("type") or "").lower().replace(" ", "_")) or "other"
                setup_data["work_description"] = str(top.get("description") or setup_data["work_description"] or "")
            confirmed = str(setup_data.get("raw_name") or "").lower()
            if use_imdb_effective and imdb_results and confirmed:
                best = None
                for row in imdb_results:
                    title = str(row.get("title") or "").lower()
                    if title in confirmed or confirmed in title:
                        best = row
                        break
                setup_data["imdb_results"] = [best] if best else [imdb_results[0]]
            else:
                setup_data["imdb_results"] = imdb_results
            if not use_imdb_effective:
                setup_data["imdb_results"] = []
            if setup_data.get("imdb_results"):
                setup_data["imdb_results"] = self._imdb_enrich_results(setup_data["imdb_results"])
                top = setup_data["imdb_results"][0]
                if top.get("description") and not setup_data.get("work_description"):
                    setup_data["work_description"] = top["description"]

        if attachments:
            attachment_texts = await extract_attachment_texts(
                attachments,
                config=AttachmentProcessingConfig(
                    attachment_max_bytes=self.ATTACHMENT_MAX_BYTES,
                    attachment_chunk_tokens=self.ATTACHMENT_CHUNK_TOKENS,
                    attachment_model_ctx_tokens=self.ATTACHMENT_MODEL_CTX_TOKENS,
                    attachment_prompt_overhead_tokens=self.ATTACHMENT_PROMPT_OVERHEAD_TOKENS,
                    attachment_response_reserve_tokens=self.ATTACHMENT_RESPONSE_RESERVE_TOKENS,
                    attachment_summary_max_tokens=self.ATTACHMENT_SUMMARY_MAX_TOKENS,
                    attachment_max_parallel=self.ATTACHMENT_MAX_PARALLEL,
                    attachment_guard_token=self.ATTACHMENT_GUARD_TOKEN,
                    attachment_max_chunks=self.ATTACHMENT_MAX_CHUNKS,
                ),
                logger=self._logger,
            )
            summary_parts: list[str] = []
            if setup_data.get("attachment_summary"):
                summary_parts.append(str(setup_data.get("attachment_summary")).strip())
            summary_instructions = str(
                setup_data.get("attachment_summary_instructions") or ""
            ).strip()
            for attachment, extracted in attachment_texts:
                source_label = self._extract_attachment_label(
                    [attachment],
                    fallback=str(setup_data.get("raw_name") or "source-material"),
                )
                if isinstance(extracted, str) and extracted.startswith("ERROR:"):
                    summary_parts.append(f"{source_label}: {extracted}")
                    continue
                if not extracted:
                    continue
                try:
                    source_chunks, _, _, _, _ = self._chunk_text_by_tokens(
                        extracted
                    )
                    if source_chunks:
                        classification_chunk = source_chunks[0]
                        try:
                            source_format = await self._classify_source_material_format(
                                classification_chunk
                            )
                        except Exception as exc:
                            self._logger.warning(
                                "Setup source material classification failed; defaulting generic: %s",
                                exc,
                            )
                            source_format = self.SOURCE_MATERIAL_FORMAT_GENERIC
                        source_format = self._normalize_source_material_format(
                            source_format
                        )
                        if source_format == self.SOURCE_MATERIAL_FORMAT_GENERIC:
                            summary = await self._summarise_long_text(
                                extracted,
                                summary_instructions=summary_instructions or None,
                            )
                            if summary:
                                summary_parts.append(f"{source_label}: {summary}")
                        else:
                            stored_count, source_key, literary_profiles = await self.ingest_source_material_with_digest(
                                str(campaign.id),
                                document_label=source_label,
                                text=extracted,
                                source_format=source_format,
                                replace_document=True,
                            )
                            if stored_count > 0 or source_key:
                                setup_data["source_material_document_key"] = source_key
                            if literary_profiles:
                                styles = state.get(self.LITERARY_STYLES_STATE_KEY)
                                if not isinstance(styles, dict):
                                    styles = {}
                                styles.update(literary_profiles)
                                state[self.LITERARY_STYLES_STATE_KEY] = styles
                except Exception:
                    self._logger.exception(
                        "Setup source-material indexing failed for campaign %s",
                        campaign.id,
                    )

            summary_value = "\n\n".join(
                part for part in summary_parts if part and part.strip()
            ).strip()
            if summary_value:
                setup_data["attachment_summary"] = summary_value

        if user_guidance:
            setup_data["variant_user_guidance"] = user_guidance
        state["setup_phase"] = "genre_pick"
        state["setup_data"] = setup_data
        return self._setup_genre_prompt()

    async def _setup_handle_genre_pick(
        self,
        campaign: Campaign,
        state: dict[str, Any],
        setup_data: dict[str, Any],
        message_text: str,
    ) -> str:
        genre_pref, error = self._parse_setup_genre_choice(message_text)
        if error:
            return f"{error}\n\n{self._setup_genre_prompt()}"

        setup_data["genre_preference"] = genre_pref
        user_guidance = (
            str(setup_data.pop("variant_user_guidance", "") or "").strip() or None
        )
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

        # Build source material index hint if docs are available.
        source_payload = self._source_material_prompt_payload(str(campaign.id))
        source_index_hint = ""
        if source_payload.get("available"):
            doc_lines = []
            for doc in source_payload.get("docs") or []:
                doc_lines.append(
                    f"  - document_key='{doc.get('document_key')}' "
                    f"label='{doc.get('document_label')}' "
                    f"format='{doc.get('format')}' "
                    f"snippets={doc.get('chunk_count')}"
                )
            source_index_hint = (
                "\nSOURCE_MATERIAL_INDEX: "
                f"{source_payload.get('document_count')} document(s), "
                f"{source_payload.get('chunk_count')} total snippet(s).\n"
                + "\n".join(doc_lines)
                + "\nIMPORTANT: Before building the world, browse the source material to understand "
                "characters, locations, tone, and rules. Start by listing all keys:\n"
                '  {"tool_call": "source_browse"}\n'
                "Then drill into specific entries with:\n"
                '  {"tool_call": "memory_search", "category": "source", "queries": ["keyword"]}\n'
                "Only return your final world JSON after you have reviewed the source material.\n"
            )

        name_tool_instructions = (
            "\nYou have a name_generate tool for culturally-appropriate character names.\n"
            "To generate names filtered by origin:\n"
            '  {"tool_call": "name_generate", "origins": ["italian"], "gender": "f", "context": "tough bouncer"}\n'
            "To call a tool, return ONLY the JSON tool_call object (no other keys). "
            "You will receive the results and can call more tools or return your final response.\n"
            "Use name_generate for ALL new original characters instead of inventing names.\n"
        )
        source_tool_instructions = name_tool_instructions
        if source_payload.get("available"):
            source_tool_instructions = (
                "\nYou have tools to inspect ingested source material before generating your response.\n"
                "To list all entries in a source document:\n"
                '  {"tool_call": "source_browse", "document_key": "doc-key"}\n'
                "To list all entries across all documents:\n"
                '  {"tool_call": "source_browse"}\n'
                "To filter entries by wildcard:\n"
                '  {"tool_call": "source_browse", "wildcard": "keyword*"}\n'
                "To semantic-search source material:\n"
                '  {"tool_call": "memory_search", "category": "source", "queries": ["query1", "query2"]}\n'
                "To generate culturally-appropriate character names:\n"
                '  {"tool_call": "name_generate", "origins": ["italian"], "gender": "f", "context": "tough bouncer"}\n'
                "To call a tool, return ONLY the JSON tool_call object (no other keys). "
                "You will receive the results and can call more tools or return your final response.\n"
                "ALWAYS browse source material before building the world — "
                "the summary alone may not capture all characters, rules, or locations.\n"
                "Use name_generate for ALL new original characters instead of inventing names.\n"
            )

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
            genre_context = ""
            genre_pref = setup_data.get("genre_preference")
            if isinstance(genre_pref, dict):
                genre_value = str(genre_pref.get("value") or "").strip()
                if genre_value:
                    genre_context = f"\nGenre direction: {genre_value}\n"
            finalize_system = (
                "You are a world-builder for interactive text-adventure campaigns.\n"
                "For non-canonical/original characters, choose distinctive specific names; avoid generic defaults "
                "(Morgan, Chen, Mendoza, Rollins, Nakamura, Kai, River) unless source canon requires them.\n"
                f"{source_tool_instructions}"
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
                f"{source_index_hint}"
                f"{genre_context}"
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
                            f"{attachment_context}"
                            f"{source_index_hint}"
                            f"{genre_context}"
                            "Source-material summary (if present) is authoritative; keep names, locations, and plot faithful to it.\n"
                            f"Chosen storyline:\n{self._dump_json(chosen)}"
                        )
                    response = await self._setup_tool_loop(
                        finalize_system,
                        cur_user,
                        campaign,
                        temperature=0.7,
                        max_tokens=4000,
                    )
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
            for _char in characters.values():
                if isinstance(_char, dict):
                    _char["_enriched"] = True
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
        auto_rulebook_count = 0
        auto_rulebook_key = ""
        try:
            auto_rulebook_count, auto_rulebook_key = await self._generate_campaign_rulebook(
                campaign,
                setup_data,
                chosen,
                world if isinstance(world, dict) else {},
            )
        except Exception as exc:
            self._logger.warning("Auto rulebook generation crashed: %s", exc)
            self._zork_log("SETUP RULEBOOK CRASHED", str(exc))
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
        state["puzzle_mode"] = novel_prefs.get("puzzle_mode", "none")

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
                player.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
        self._zork_log(
            f"CAMPAIGN SETUP FINALIZED campaign={campaign.id}",
            f"characters={char_count} chapters={chapter_count} on_rails={on_rails} "
            f"auto_rulebook_lines={auto_rulebook_count} auto_rulebook_key={auto_rulebook_key!r}",
        )
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
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
                "updated": datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0).isoformat() + "Z",
                "prompt": (scene_prompt or "").strip(),
            }
            campaign_state[self.ROOM_IMAGE_STATE_KEY] = room_images
            campaign.state_json = self._dump_json(campaign_state)
            campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            player_state["pending_avatar_generated_at"] = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0).isoformat() + "Z"
            player.state_json = self._dump_json(player_state)
            player.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            player.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            player.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
            if had_pending:
                return True, "Pending avatar discarded."
            return False, "No pending avatar to discard."

    def _normalize_match_text(self, value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _normalize_location_key(value: object) -> str:
        """Collapse a location key to a canonical alphanumeric form.

        Both human-readable names (``"Nothing's Edge Approach"``) and
        slug-style keys (``"nothing-edge-approach"``) normalise to the
        same string (``"nothingsedgeapproach"``), so they compare equal
        regardless of which format was used at storage time.
        """
        if value is None:
            return ""
        return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

    def _room_key_from_player_state(self, player_state: Dict[str, object]) -> str:
        if not isinstance(player_state, dict):
            return "unknown-room"
        for key in ("location_key", "room_id", "location", "room_title", "room_summary"):
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
                player_slug = self._player_visibility_slug(entry.actor_id)
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
                        "actor_id": entry.actor_id,
                        "discord_mention": f"<@{entry.actor_id}>",
                        "name": display_name,
                        "player_slug": player_slug,
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
        return prompt

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
        return prompt

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
        prefix = self.SCENE_IMAGE_PRESERVE_PREFIX.strip()
        if prefix and not prompt_for_generation.lower().startswith(prefix.lower()):
            prompt_for_generation = f"{prefix}. {prompt_for_generation}".strip()

        metadata = {
            "zork_scene": True,
            "zork_store_image": should_store_room_image,
            "zork_seed_room_image": should_store_room_image,
            "zork_scene_prompt": scene_image_prompt,
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
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
        return True

    # ── Async Mid-Game NPC Enrichment ─────────────────────────────────────

    ENRICHMENT_FIELDS = {
        "personality", "background", "appearance", "speech_style",
        "goals", "skills", "hobbies", "cultural_touchstones",
        "childhood_memories", "relationships_history",
        "escalation_behavior", "confrontation_style",
    }

    def _build_character_enrichment_prompt(
        self,
        campaign_name: str,
        setting: str,
        tone: str,
        summary: str,
        existing_character_names: list,
        character_seed: dict,
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for character enrichment."""
        system_prompt = (
            "You are a character-depth engine for an interactive text-adventure game.\n"
            "Given a seed character profile, expand it into a rich, deep character.\n"
            "Return ONLY valid JSON with two keys:\n\n"
            '"profile" — object with these string fields:\n'
            "  - personality (expanded: coping mechanisms, conflict handling, discomfort, deflection)\n"
            "  - background (expanded: childhood, pivotal moments, how they changed since youth)\n"
            "  - appearance (seed plus consistent physical details, gestures, wear patterns)\n"
            "  - speech_style (expanded: verbal tics, phrases, angry vs calm vs lying, sentence structure)\n"
            "  - goals (immediate and deep life goals)\n"
            "  - skills (practical and social)\n"
            "  - hobbies (free time activities)\n"
            "  - cultural_touchstones (books read, stories referenced, cultural knowledge)\n"
            "  - childhood_memories (2-3 specific memories that shaped them)\n"
            "  - relationships_history (previous relationships good AND bad, or none; trust patterns)\n"
            "  - escalation_behavior (what they do when player stalls/refuses/doesn't engage)\n"
            "  - confrontation_style (how they handle direct philosophical/moral challenges)\n\n"
            '"rulebook_lines" — array of exactly 4 strings in CATEGORY-TAG: fact text format:\n'
            "  - CHAR-[NAME]: general character summary (50-200 words)\n"
            "  - CHAR-[NAME]-PERSONALITY: personality and behavioral patterns (50-200 words)\n"
            "  - CHAR-[NAME]-DIALOGUE: speech patterns and dialogue guidance (50-200 words)\n"
            "  - INTERACTION-NEWCOMER-[NAME]: how to introduce and play this character when meeting the player (50-200 words)\n\n"
            "Rules:\n"
            "- Build on the seed character data — NEVER contradict it\n"
            "- Match the campaign tone\n"
            "- Do not invent trauma or abuse unless the seed implies it\n"
            "- Describe what the character DOES, not what they don't. "
            "Negation-based traits ('doesn't chase,' 'won't perform,' 'refuses to show') "
            "become prohibitions the narrator enforces as absolute gates, blocking authentic "
            "behavior that would read as violating the rule. Instead write positive behaviors: "
            "'Expresses care through proximity and attention. Shows up. Notices.' "
            "Reserve negations only for hard limits that genuinely cannot happen regardless of context.\n"
            "- Rulebook lines must be plain text (no markdown, no JSON)\n"
            "- Each rulebook line must be 50-200 words\n"
            "- No markdown, no code fences in the response\n"
        )
        trimmed_summary = (summary or "")[:500]
        banned_names = ", ".join(existing_character_names) if existing_character_names else "none"
        user_prompt = (
            f"Campaign: {campaign_name}\n"
            f"Setting: {setting}\n"
            f"Tone: {tone}\n"
            f"Summary: {trimmed_summary}\n"
            f"Existing characters (do not duplicate names): {banned_names}\n\n"
            f"Seed character to enrich:\n{json.dumps(character_seed, indent=2)}\n"
        )
        return system_prompt, user_prompt

    async def _enqueue_character_enrichment(
        self,
        campaign_id: str,
        character_slug: str,
        character_seed_dict: dict,
    ) -> bool:
        """Async enrichment: expand a shallow NPC seed into a deep character profile."""
        try:
            self._logger.info(
                "CHARACTER ENRICHMENT START slug=%s campaign=%s",
                character_slug, campaign_id,
            )
            # Extract campaign context
            with self._session_factory() as session:
                campaign = session.get(Campaign, str(campaign_id))
                if campaign is None:
                    return False
                campaign_state = parse_json_dict(campaign.state_json)
                setting = str(campaign_state.get("setting") or "").strip()
                tone = str(campaign_state.get("tone") or "").strip()
                summary = str(campaign.summary or "").strip()
                campaign_name = str(campaign.name or "")
                characters = parse_json_dict(campaign.characters_json)
                existing_names = [
                    str(c.get("name", s)).strip()
                    for s, c in characters.items()
                    if isinstance(c, dict)
                ]

            system_prompt, user_prompt = self._build_character_enrichment_prompt(
                campaign_name=campaign_name,
                setting=setting,
                tone=tone,
                summary=summary,
                existing_character_names=existing_names,
                character_seed=character_seed_dict,
            )

            if self._completion_port is None:
                self._logger.warning("No completion port for character enrichment")
                return False
            response = await self._completion_port.complete(
                system_prompt, user_prompt,
                temperature=0.7, max_tokens=3000,
            )
            if not response:
                self._logger.warning(
                    "Character enrichment returned empty for %s", character_slug,
                )
                return False

            json_text = self._extract_json(response)
            enrichment = self._parse_json_lenient(json_text) if json_text else None
            if not isinstance(enrichment, dict):
                self._logger.warning(
                    "Character enrichment parse failed for %s", character_slug,
                )
                return False

            profile = enrichment.get("profile")
            rulebook_lines = enrichment.get("rulebook_lines")
            if not isinstance(profile, dict):
                self._logger.warning(
                    "Character enrichment missing profile for %s", character_slug,
                )
                return False
            if not isinstance(rulebook_lines, list):
                rulebook_lines = []

            # Validate rulebook lines
            valid_lines = []
            for line in rulebook_lines:
                if isinstance(line, str) and re.match(r"^[A-Z][A-Z0-9-]{1,80}:\s+\S", line):
                    valid_lines.append(line)

            self._apply_character_enrichment(
                campaign_id=campaign_id,
                character_slug=character_slug,
                profile=profile,
                rulebook_lines=valid_lines,
            )
            self._logger.info(
                "CHARACTER ENRICHMENT COMPLETE slug=%s campaign=%s profile_keys=%s rulebook_lines=%d",
                character_slug, campaign_id, list(profile.keys()), len(valid_lines),
            )
            return True
        except Exception as e:
            self._logger.warning(
                "Character enrichment failed for %s: %s", character_slug, e,
            )
            return False

    def _apply_character_enrichment(
        self,
        campaign_id: str,
        character_slug: str,
        profile: dict,
        rulebook_lines: list,
    ) -> bool:
        """Write enriched profile back to characters_json and store rulebook entries."""
        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return False
            characters = parse_json_dict(campaign.characters_json)
            if character_slug not in characters:
                return False
            char = characters[character_slug]
            if not isinstance(char, dict):
                return False

            # Merge ONLY enrichment fields — never touch mutable narrator fields
            for field in self.ENRICHMENT_FIELDS:
                if field in profile:
                    val = profile[field]
                    if not isinstance(val, str):
                        val = str(val) if val is not None else ""
                    char[field] = val
            char["_enriched"] = True

            campaign.characters_json = self._dump_json(characters)
            campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()

        # Store rulebook entries
        for line in rulebook_lines:
            colon_idx = line.index(":")
            rule_key = line[:colon_idx].strip()
            rule_text = line[colon_idx + 1:].strip()
            try:
                self.put_campaign_rule(
                    str(campaign_id),
                    rule_key=rule_key,
                    rule_text=rule_text,
                    upsert=True,
                )
            except Exception as e:
                self._logger.warning("Failed to store rulebook entry %s: %s", rule_key, e)
        return True

    async def _enqueue_new_character_enrichments(
        self,
        *,
        campaign_id: str,
        pre_slugs: set[str],
    ) -> None:
        """Enqueue enrichment for all new unenriched characters (mirrors _enqueue_new_character_portraits)."""
        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return
            characters = parse_json_dict(campaign.characters_json)
        if not isinstance(characters, dict):
            return
        for slug, value in characters.items():
            if slug in pre_slugs or not isinstance(value, dict):
                continue
            if value.get("_enriched"):
                continue
            await self._enqueue_character_enrichment(
                campaign_id=campaign_id,
                character_slug=slug,
                character_seed_dict=dict(value),
            )

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
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
        sms_sender_name = f"Player {actor_id}"
        if manage_claim:
            cid, error_text = await self.begin_turn(campaign_id, actor_id)
            if error_text is not None:
                return error_text
            if cid is None:
                return None
            should_end = True
        self._set_turn_ephemeral_notices(campaign_id, actor_id, session_id, [])
        try:
            pending = self._pending_timers.get(campaign_id)
            can_interrupt = (
                pending is not None
                and pending.get("interruptible", True)
                and self._timer_can_be_interrupted_by(pending, actor_id)
            )
            if can_interrupt:
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
                        campaign = session.get(Campaign, campaign_id)
                        campaign_state = parse_json_dict(campaign.state_json) if campaign is not None else {}
                        turn_meta = self._dump_json(
                            {"game_time": self._extract_game_time_snapshot(campaign_state)}
                        )
                        session.add(
                            Turn(
                                campaign_id=campaign_id,
                                session_id=session_id,
                                actor_id=actor_id,
                                kind="narrator",
                                content=interrupt_note,
                                meta_json=turn_meta,
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
                    row_state = parse_json_dict(row.state_json)
                    sms_sender_name = str(row_state.get("character_name") or f"Player {actor_id}")[:80]
                    self.record_player_message(row)
                campaign_row = session.get(Campaign, campaign_id)
                if campaign_row is not None:
                    pre_character_slugs = set(parse_json_dict(campaign_row.characters_json).keys())
            is_ooc = bool(re.match(r"\s*\[OOC\b", action or "", re.IGNORECASE))
            if not is_ooc:
                sms_intent = self._extract_inline_sms_intent(action)
                if sms_intent is not None:
                    sms_recipient, sms_message = sms_intent
                    with self._session_factory() as session:
                        campaign_row = session.get(Campaign, campaign_id)
                        if campaign_row is not None:
                            campaign_state = parse_json_dict(campaign_row.state_json)
                            game_time = self._extract_game_time_snapshot(campaign_state)
                            self._sms_write(
                                campaign_state,
                                thread=self._sms_normalize_thread_key(sms_recipient) or sms_recipient,
                                sender=sms_sender_name,
                                recipient=sms_recipient,
                                message=sms_message,
                                game_time=game_time,
                                turn_id=0,
                            )
                            campaign_row.state_json = self._dump_json(campaign_state)
                            campaign_row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                            session.commit()

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
                self._sync_main_party_room_state(campaign_id, actor_id)
                timer_delay_seconds: int | None = None
                if result.timer_instruction is not None:
                    timer_delay_seconds = int(result.timer_instruction.delay_seconds)
                    with self._session_factory() as session:
                        campaign_row = session.get(Campaign, campaign_id)
                    speed = self.get_speed_multiplier(campaign_row)
                    if speed > 0:
                        timer_delay_seconds = int(timer_delay_seconds / speed)
                    timer_delay_seconds = max(15, min(300, timer_delay_seconds))
                    timer_delay_seconds = self._compress_realtime_timer_delay(
                        timer_delay_seconds
                    )
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
                        interrupt_scope=self._normalize_timer_interrupt_scope(
                            getattr(result.timer_instruction, "interrupt_scope", "global")
                        ),
                        interrupt_actor_id=actor_id,
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
                asyncio.create_task(self._enqueue_new_character_enrichments(
                    campaign_id=campaign_id,
                    pre_slugs=pre_character_slugs,
                ))
                if self._private_setup_warning_needed(action):
                    self._set_turn_ephemeral_notices(
                        campaign_id,
                        actor_id,
                        session_id,
                        [
                            "Warning: if you include the real whisper/private content in the same setup message, it may leak before the aside is fully established. Use one short setup turn first, then continue once the reply keeps it private."
                        ],
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
            campaign = session.get(Campaign, campaign_id)
            player = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == actor_id)
                .first()
            )
            player_state = self.get_player_state(player) if player is not None else {}
            campaign_state = self.get_campaign_state(campaign) if campaign is not None else {}
            post_turn_game_time = self._extract_game_time_snapshot(campaign_state)
            inventory_line = self._format_inventory(player_state) or "Inventory: empty"

            if not has_inventory_line:
                if decorated:
                    decorated = f"{decorated}\n\n{inventory_line}"
                else:
                    decorated = inventory_line

            sms_notice = self._sms_unread_hourly_notification(
                campaign_state,
                actor_id=actor_id,
                player_state=player_state,
                game_time=post_turn_game_time,
            )
            if sms_notice:
                decorated = f"{decorated}\n\n{sms_notice}"
                self._increment_auto_fix_counter(campaign_state, "sms_unread_notice")

            if timer_instruction is not None and not has_timer_line:
                delay_seconds = (
                    int(timer_delay_seconds)
                    if timer_delay_seconds is not None
                    else self._compress_realtime_timer_delay(
                        int(getattr(timer_instruction, "delay_seconds", 0) or 0)
                    )
                )
                expiry_ts = int(time.time()) + delay_seconds
                event_hint = str(getattr(timer_instruction, "event_text", "") or "Something happens")
                interruptible = bool(getattr(timer_instruction, "interruptible", True))
                interrupt_scope = self._normalize_timer_interrupt_scope(
                    getattr(timer_instruction, "interrupt_scope", "global")
                )
                if interruptible:
                    interrupt_hint = (
                        "acting player can prevent"
                        if interrupt_scope == "local"
                        else "act to prevent!"
                    )
                else:
                    interrupt_hint = "unavoidable"
                decorated = (
                    f"{decorated}\n\n"
                    f"⏰ <t:{expiry_ts}:R>: {event_hint} ({interrupt_hint})"
                )

            if campaign is not None:
                campaign.last_narration = decorated
                campaign.state_json = self._dump_json(campaign_state)
                campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

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
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()

    def _sync_main_party_room_state(self, campaign_id: str, source_actor_id: str) -> None:
        with self._session_factory() as session:
            source_player = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == source_actor_id)
                .first()
            )
            if source_player is None:
                return
            source_state = parse_json_dict(source_player.state_json)
            source_party = str(source_state.get("party_status") or "").strip().lower()
            if source_party != "main_party":
                return
            has_room_context = any(
                source_state.get(key)
                for key in ("room_id", "location", "room_title", "room_summary", "room_description")
            )
            if not has_room_context:
                return
            targets = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id != source_actor_id)
                .all()
            )
            changed = False
            for target in targets:
                target_state = parse_json_dict(target.state_json)
                if str(target_state.get("party_status") or "").strip().lower() != "main_party":
                    continue
                before = dict(target_state)
                for key in self.ROOM_STATE_KEYS:
                    src_val = source_state.get(key)
                    if src_val is None:
                        target_state.pop(key, None)
                    else:
                        target_state[key] = src_val
                if target_state != before:
                    target.state_json = self._dump_json(target_state)
                    target.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    changed = True
            if changed:
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
            campaign = session.get(Campaign, campaign_id)
            campaign_state = parse_json_dict(campaign.state_json) if campaign is not None else {}
            game_time_snapshot = self._extract_game_time_snapshot(campaign_state)
            turn_meta = self._dump_json({"game_time": game_time_snapshot})
            session.add(
                Turn(
                    campaign_id=campaign_id,
                    session_id=session_id,
                    actor_id=actor_id,
                    kind="player",
                    content=action_text,
                    meta_json=turn_meta,
                )
            )
            session.add(
                Turn(
                    campaign_id=campaign_id,
                    session_id=session_id,
                    actor_id=actor_id,
                    kind="narrator",
                    content=narration,
                    meta_json=turn_meta,
                )
            )
            if campaign is not None:
                campaign.last_narration = narration
                campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            target_player.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            source_player.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
                        hours_remaining = int(event.get("hours_remaining", days_remaining * 24))
                        fire_day = int(event.get("fire_day", 1))
                        fire_hour = max(0, min(23, int(event.get("fire_hour", 23))))
                        desc = str(event.get("description", "") or "")
                        if hours_remaining < 0:
                            eta = f"overdue by {abs(hours_remaining)} hour(s)"
                        elif hours_remaining == 0:
                            eta = "fires now"
                        elif hours_remaining < 48:
                            eta = f"fires in {hours_remaining} hour(s)"
                        else:
                            eta_days = (hours_remaining + 23) // 24
                            eta = f"fires in {eta_days} day(s)"
                        line = (
                            f"- **{event.get('name', 'Unknown')}** - "
                            f"Day {fire_day}, {fire_hour:02d}:00 ({eta})"
                        )
                        if desc:
                            line += f" ({desc})"
                        lines.append(line)
                else:
                    lines.append("No upcoming events.")
                return "\n".join(lines)

            if action_clean in ("roster", "characters", "npcs"):
                characters = self.get_campaign_characters(campaign_obj)
                return self.format_roster(characters)

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
            campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

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
                player.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

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
            timer.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
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

    @classmethod
    def _compress_realtime_timer_delay(cls, delay_seconds: object) -> int:
        try:
            raw = int(delay_seconds)
        except (TypeError, ValueError):
            raw = 60
        raw = max(1, raw)
        compressed = int(round(raw * float(cls.TIMER_REALTIME_SCALE)))
        return max(
            int(cls.TIMER_REALTIME_MIN_SECONDS),
            min(int(cls.TIMER_REALTIME_MAX_SECONDS), compressed),
        )

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
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
            campaign.state_json = row.state_json
            campaign.updated_at = row.updated_at
        return True

    @classmethod
    def normalize_difficulty(cls, value: object) -> str:
        text = " ".join(str(value or "").strip().lower().split())
        if text in cls.DIFFICULTY_LEVELS:
            return text
        aliases = {
            "default": "normal",
            "std": "normal",
            "story mode": "story",
            "easy mode": "easy",
            "medium mode": "medium",
            "normal mode": "normal",
            "hard mode": "hard",
            "impossible mode": "impossible",
        }
        return aliases.get(text, "normal")

    def get_difficulty(self, campaign: Campaign | None) -> str:
        if campaign is None:
            return "normal"
        campaign_state = self.get_campaign_state(campaign)
        return self.normalize_difficulty(campaign_state.get("difficulty", "normal"))

    def set_difficulty(self, campaign: Campaign | None, difficulty: str) -> bool:
        if campaign is None:
            return False
        normalized = self.normalize_difficulty(difficulty)
        campaign_state = self.get_campaign_state(campaign)
        campaign_state["difficulty"] = normalized
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is None:
                return False
            row.state_json = self._dump_json(campaign_state)
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
            campaign.state_json = row.state_json
            campaign.updated_at = row.updated_at
        return True

    @classmethod
    def _difficulty_response_note(cls, difficulty: object) -> str:
        normalized = cls.normalize_difficulty(difficulty)
        note = cls.DIFFICULTY_NOTES.get(normalized)
        if not note:
            return ""
        return (
            f"[SYSTEM NOTE: FOR THIS RESPONSE ONLY: difficulty={normalized}. {note}]"
        )

    @classmethod
    def _merge_system_notes(cls, *notes: object) -> str:
        cleaned: list[str] = []
        seen: set[str] = set()
        for note in notes:
            text = str(note or "").strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            if text.startswith("[SYSTEM NOTE:") and text.endswith("]"):
                text = text[len("[SYSTEM NOTE:") : -1].strip()
            cleaned.append(text)
        if not cleaned:
            return ""
        return f"[SYSTEM NOTE: FOR THIS RESPONSE ONLY: {' '.join(cleaned)}]"

    @classmethod
    def _turn_response_style_note(cls, difficulty: object) -> str:
        return cls._merge_system_notes(
            cls.RESPONSE_STYLE_NOTE,
            cls._difficulty_response_note(difficulty),
            (
                'Return final JSON only. Include reasoning first. '
                'state_update is required and must include "game_time", "current_chapter", and "current_scene" explicitly.'
            ),
        )

    @classmethod
    def _turn_stage_note(cls, difficulty: object, prompt_stage: str) -> str:
        stage = str(prompt_stage or cls.PROMPT_STAGE_FINAL).strip().lower()
        if stage == cls.PROMPT_STAGE_BOOTSTRAP:
            return cls._merge_system_notes(
                cls._difficulty_response_note(difficulty),
                "Do not narrate yet. First decide the immediate continuity receivers and call recent_turns.",
            )
        if stage == cls.PROMPT_STAGE_RESEARCH:
            return cls._merge_system_notes(
                cls._difficulty_response_note(difficulty),
                "RECENT_TURNS is loaded. Do not call recent_turns again this turn.",
                "Use memory/source/SMS/planning tools only when they materially improve continuity.",
                'When research is sufficient, return ONLY {"tool_call": "ready_to_write"}. Do not narrate yet.',
            )
        return cls._turn_response_style_note(difficulty)

    @classmethod
    def _build_turn_prompt_tail(
        cls,
        player_state: dict[str, object],
        action: str,
        response_style_note: str,
        *,
        turn_attachment_context: str | None = None,
        extra_lines: list[str] | None = None,
    ) -> str:
        active_name = str(player_state.get("character_name") or "").strip()
        action_label = f"PLAYER_ACTION ({active_name.upper()})" if active_name else "PLAYER_ACTION"
        parts = [f"{action_label}: {action}"]
        if turn_attachment_context:
            parts.append(f"TURN_ATTACHMENT_CONTEXT:\n{turn_attachment_context}")
        for line in extra_lines or []:
            text = str(line or "").strip()
            if text:
                parts.append(text)
        if response_style_note:
            parts.append(response_style_note)
        return "\n".join(parts)

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
        interrupt_scope: str = "global",
        interrupt_actor_id: str | None = None,
    ) -> None:
        task = asyncio.create_task(
            self._timer_task(
                campaign_id,
                channel_id,
                delay_seconds,
                event_description,
            )
        )
        self._pending_timers[campaign_id] = {
            "task": task,
            "channel_id": channel_id,
            "message_id": None,
            "event": event_description,
            "delay": delay_seconds,
            "interruptible": interruptible,
            "interrupt_action": interrupt_action,
            "interrupt_scope": self._normalize_timer_interrupt_scope(interrupt_scope),
            "interrupt_actor_id": str(interrupt_actor_id or "").strip() or None,
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
        preferred_actor_id = None
        if timer_ctx is not None:
            raw_actor_id = timer_ctx.get("interrupt_actor_id")
            if raw_actor_id is not None:
                preferred_actor_id = str(raw_actor_id).strip() or None
        try:
            await self._execute_timed_event(
                campaign_id,
                channel_id,
                event_description,
                preferred_actor_id=preferred_actor_id,
            )
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
        preferred_actor_id: str | None = None,
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
                        age_seconds = (datetime.now(timezone.utc).replace(tzinfo=None) - created_at).total_seconds()
                        if age_seconds < 5:
                            return
                active_player = None
                if preferred_actor_id:
                    active_player = (
                        session.query(Player)
                        .filter(Player.campaign_id == campaign_id)
                        .filter(Player.actor_id == preferred_actor_id)
                        .first()
                    )
                if active_player is None:
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
        self._sync_main_party_room_state(campaign_id, active_actor_id)
        await self._enqueue_new_character_portraits(
            campaign_id=campaign_id,
            actor_id=active_actor_id,
            pre_slugs=pre_character_slugs,
            channel_id=channel_id,
        )
        asyncio.create_task(self._enqueue_new_character_enrichments(
            campaign_id=campaign_id,
            pre_slugs=pre_character_slugs,
        ))
        narration = self._strip_narration_footer(result.narration or "")
        if (
            " ".join(str(narration or "").lower().split())
            == "the world shifts, but nothing clear emerges."
        ):
            self._logger.warning(
                "Timed event generic fallback: campaign=%s actor=%s event=%r",
                campaign_id,
                active_actor_id,
                event_description,
            )
            narration = str(event_description or "Something happens.").strip()
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

    def _compose_world_summary(
        self,
        campaign: Campaign,
        campaign_state: Dict[str, object],
        *,
        turns: list[Turn] | None = None,
        viewer_actor_id: str | None = None,
        viewer_slug: str = "",
        viewer_location_key: str = "",
        max_chars: int = 1600,
    ) -> str:
        summary = self._strip_inventory_mentions(campaign.summary or "")
        seen: set[str] = set()
        persisted_lines: list[str] = []
        recent_lines: list[str] = []

        def _append_if_relevant(target: list[str], raw_text: object) -> None:
            line = " ".join(str(raw_text or "").strip().split())
            if not line:
                return
            line_l = line.lower()
            if line_l in {"none", "n/a", "na", "lel without elaboration."}:
                return
            if any(token in line_l for token in ("inventory:", "📨 unread sms:", "calendar_update")):
                return
            if line_l in seen:
                return
            seen.add(line_l)
            target.append(line)

        for raw_line in str(summary or "").splitlines():
            _append_if_relevant(persisted_lines, raw_line)

        if isinstance(turns, list) and viewer_actor_id and turns:
            for turn in turns[-24:]:
                if not isinstance(turn, Turn) or turn.kind != "narrator":
                    continue
                if not self._turn_visible_to_viewer(
                    turn,
                    viewer_actor_id,
                    viewer_slug,
                    viewer_location_key,
                ):
                    continue
                meta = self._safe_turn_meta(turn)
                if bool(meta.get("suppress_context")):
                    continue
                summary_candidate = meta.get("summary_update")
                if not str(summary_candidate or "").strip():
                    summary_candidate = (
                        meta.get("scene_output_rendered")
                        or turn.content
                        or ""
                    )
                _append_if_relevant(recent_lines, summary_candidate)

        # The persisted campaign.summary blob predates visibility-aware
        # composition and may contain global/stale lines from scenes the current
        # viewer should not see. Once we have usable recent narrator history for
        # this viewer, prefer that scoped history and treat the persisted blob as
        # fallback only.
        lines = recent_lines if recent_lines else persisted_lines
        text = "\n".join(lines).strip()
        if text:
            return self._trim_text(text, max_chars)

        story_context = self._build_story_context(campaign_state)
        if story_context:
            return self._trim_text(story_context.splitlines()[0], max_chars)
        return ""

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
        if not bool(campaign_state.get("on_rails", False)):
            chapters = campaign_state.get("chapters")
            if isinstance(chapters, list) and chapters:
                active_rows = [
                    row
                    for row in chapters
                    if isinstance(row, dict)
                    and str(row.get("status") or "active").strip().lower() == "active"
                ]
                rows = active_rows[:4] if active_rows else [row for row in chapters if isinstance(row, dict)][:4]
                if rows:
                    def _chapter_scene_label(value: object) -> str:
                        text = str(value or "").strip()
                        if not text:
                            return "Untitled"
                        text = text.replace("_", "-")
                        parts = [part for part in text.split("-") if part]
                        if not parts:
                            return "Untitled"
                        return " ".join(part.capitalize() for part in parts)[:120]

                    current = rows[0]
                    lines: List[str] = []
                    lines.append(f"CURRENT CHAPTER: {current.get('title', 'Untitled')}")
                    lines.append(f"  Summary: {current.get('summary', '')}")
                    scenes = current.get("scenes") or []
                    current_scene_slug = str(current.get("current_scene") or "").strip()
                    if isinstance(scenes, list):
                        for i, scene in enumerate(scenes):
                            scene_slug = str(scene or "").strip()
                            marker = " >>> CURRENT SCENE <<<" if scene_slug and scene_slug == current_scene_slug else ""
                            lines.append(f"  Scene {i + 1}: {_chapter_scene_label(scene_slug)}{marker}")
                    if len(rows) > 1:
                        lines.append("")
                        for idx, row in enumerate(rows[1:4], start=1):
                            label = "NEXT CHAPTER" if idx == 1 else f"UPCOMING CHAPTER {idx}"
                            lines.append(f"{label}: {row.get('title', 'Untitled')}")
                            summary = str(row.get("summary") or "").strip()
                            if summary:
                                lines.append(f"  Preview: {summary[:320]}")
                            row_scenes = row.get("scenes") or []
                            if isinstance(row_scenes, list) and row_scenes:
                                preview_titles = [
                                    _chapter_scene_label(scene)
                                    for scene in row_scenes[:3]
                                    if str(scene or "").strip()
                                ]
                                if preview_titles:
                                    lines.append(f"  Early scenes: {', '.join(preview_titles)}")
                            lines.append("")
                    while lines and not lines[-1]:
                        lines.pop()
                    return "\n".join(lines) if lines else None

        outline = campaign_state.get("story_outline")
        if not isinstance(outline, dict):
            return None
        chapters = outline.get("chapters")
        if not isinstance(chapters, list) or not chapters:
            return None

        current_chapter = self._coerce_non_negative_int(
            campaign_state.get("current_chapter", 0), default=0
        )
        current_scene = self._coerce_non_negative_int(
            campaign_state.get("current_scene", 0), default=0
        )
        current_chapter = min(current_chapter, max(len(chapters) - 1, 0))

        def _preview(value: object, max_chars: int = 320) -> str:
            text = str(value or "").strip()
            if len(text) <= max_chars:
                return text
            clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
            if not clipped:
                clipped = text[:max_chars].strip()
            return f"{clipped}..."

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

        for offset in range(1, 4):
            idx = current_chapter + offset
            if idx >= len(chapters):
                break
            nxt = chapters[idx]
            label = "NEXT CHAPTER" if offset == 1 else f"UPCOMING CHAPTER {offset}"
            lines.append(f"{label}: {nxt.get('title', 'Untitled')}")
            preview = _preview(nxt.get("summary", ""))
            if preview:
                lines.append(f"  Preview: {preview}")
            nxt_scenes = nxt.get("scenes")
            if isinstance(nxt_scenes, list):
                titles = []
                for scene in nxt_scenes[:3]:
                    if not isinstance(scene, dict):
                        continue
                    title = str(scene.get("title", "Untitled")).strip() or "Untitled"
                    titles.append(title)
                if titles:
                    lines.append(f"  Early scenes: {', '.join(titles)}")
            lines.append("")

        while lines and not lines[-1]:
            lines.pop()
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
                if "room_summary" not in cleaned:
                    cleaned["room_summary"] = None
        return cleaned

    def _strip_inventory_from_narration(self, narration: str) -> str:
        if not narration:
            return ""
        kept_lines: List[str] = []
        for line in narration.splitlines():
            stripped = line.strip().lower()
            if any(stripped.startswith(prefix) for prefix in self._INVENTORY_LINE_PREFIXES):
                continue
            if any(stripped.startswith(prefix) for prefix in self._UNREAD_SMS_LINE_PREFIXES):
                continue
            if stripped.startswith("\u23f0"):
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip()

    def _strip_inventory_mentions(self, text: str) -> str:
        if not text:
            return ""
        return self._strip_inventory_from_narration(text)

    def _set_turn_ephemeral_notices(
        self,
        campaign_id: str,
        actor_id: str,
        session_id: str | None,
        notices: list[str],
    ) -> None:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in notices or []:
            text = " ".join(str(item or "").split()).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            cleaned.append(text[:500])
        key = (str(campaign_id), str(actor_id), str(session_id or "") or None)
        if cleaned:
            self._turn_ephemeral_notices[key] = cleaned
        else:
            self._turn_ephemeral_notices.pop(key, None)

    def pop_turn_ephemeral_notices(
        self,
        campaign_id: str,
        actor_id: str,
        session_id: str | None = None,
    ) -> list[str]:
        return list(
            self._turn_ephemeral_notices.pop(
                (str(campaign_id), str(actor_id), str(session_id or "") or None),
                [],
            )
        )

    @staticmethod
    def _is_private_engagement_setup_action(action: str) -> bool:
        text = " ".join(str(action or "").strip().lower().split())
        if not text:
            return False
        return bool(
            re.search(
                r"\b(?:whisper|murmur|lean in|lower my voice|lower your voice|quietly to|under my breath|private word|pull .* aside|take .* aside|step aside with)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _private_setup_warning_needed(self, action: str) -> bool:
        if not self._is_private_engagement_setup_action(action):
            return False
        text = str(action or "").strip()
        sentence_count = len(
            [seg for seg in re.split(r"(?<=[.!?])\s+", text) if seg.strip()]
        )
        if sentence_count > 1:
            return True
        if text.count('"') >= 2 or text.count("'") >= 2:
            return True
        return len(text) > 180

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
        if not self.PRESET_CAMPAIGNS:
            return "No in-repo campaigns are configured."
        sample = ", ".join(self.PRESET_CAMPAIGNS.keys())
        return f"Available campaigns: {sample}"

    def _apply_character_updates(
        self,
        existing: Dict[str, dict],
        updates: Dict[str, object],
        on_rails: bool = False,
    ) -> Dict[str, dict]:
        if not isinstance(updates, dict):
            return existing
        for raw_slug, fields in updates.items():
            slug = str(raw_slug).strip()
            if not slug:
                continue

            target_slug = self._resolve_existing_character_slug(existing, slug)

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
                existing.pop(target_slug or slug, None)
                continue
            if not isinstance(fields, dict):
                continue
            if target_slug in existing:
                for key, value in fields.items():
                    if key not in self.IMMUTABLE_CHARACTER_FIELDS:
                        existing[target_slug][key] = value
            else:
                if on_rails:
                    continue
                existing[slug] = dict(fields)
        return existing

    def _resolve_existing_character_slug(
        self,
        existing: Dict[str, dict],
        raw_slug: object,
    ) -> Optional[str]:
        slug = str(raw_slug or "").strip()
        if not slug:
            return None
        canonical = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
        if slug in existing:
            return slug
        if canonical and canonical in existing:
            return canonical
        partial_matches: List[str] = []
        for existing_slug, existing_fields in existing.items():
            existing_canonical = re.sub(
                r"[^a-z0-9]+", "-", str(existing_slug).lower()
            ).strip("-")
            if canonical and canonical == existing_canonical:
                return existing_slug
            if canonical and (
                existing_canonical.startswith(canonical)
                or canonical in existing_canonical
            ):
                partial_matches.append(existing_slug)
            if isinstance(existing_fields, dict):
                name_canonical = re.sub(
                    r"[^a-z0-9]+", "-",
                    str(existing_fields.get("name") or "").lower(),
                ).strip("-")
                if canonical and canonical == name_canonical:
                    return existing_slug
                if canonical and (
                    name_canonical.startswith(canonical)
                    or canonical in name_canonical
                ):
                    partial_matches.append(existing_slug)
        if canonical:
            unique_matches = list(dict.fromkeys(partial_matches))
            if len(unique_matches) == 1:
                return unique_matches[0]
        return None

    @staticmethod
    def _memory_search_term_key(raw_term: object) -> str:
        return re.sub(r"[^a-z0-9]+", "-", str(raw_term or "").lower()).strip("-")[:80]

    @classmethod
    def _memory_search_usage_from_state(cls, campaign_state: Dict[str, object]) -> Dict[str, dict]:
        raw = campaign_state.get(cls.MEMORY_SEARCH_USAGE_KEY) if isinstance(campaign_state, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        out: Dict[str, dict] = {}
        for raw_key, raw_value in raw.items():
            key = cls._memory_search_term_key(raw_key)
            if not key or not isinstance(raw_value, dict):
                continue
            count = cls._coerce_non_negative_int(raw_value.get("count", 0), default=0)
            if count <= 0:
                continue
            label = str(raw_value.get("label") or raw_key).strip()[:120] or key
            out[key] = {"count": count, "label": label}
        return out

    @staticmethod
    def _memory_search_term_looks_character_like(term_key: str) -> bool:
        if not term_key:
            return False
        parts = [part for part in term_key.split("-") if part]
        if not parts or len(parts) > 4:
            return False
        if len(term_key) < 3 or len(term_key) > 48:
            return False
        blocked = {
            "where",
            "what",
            "when",
            "why",
            "how",
            "room",
            "scene",
            "inventory",
            "calendar",
            "event",
            "events",
            "map",
            "summary",
            "story",
            "chapter",
            "turn",
        }
        return not any(part in blocked for part in parts)

    def _record_memory_search_usage_for_campaign(
        self,
        campaign: Campaign,
        queries: List[str],
    ) -> List[Dict[str, object]]:
        campaign_state = self.get_campaign_state(campaign)
        usage = self._memory_search_usage_from_state(campaign_state)
        updated_keys: List[str] = []

        for query in queries[:8]:
            query_text = str(query or "").strip()
            if not query_text:
                continue
            term_key = self._memory_search_term_key(query_text)
            if not term_key:
                continue
            row = usage.get(term_key, {"count": 0, "label": query_text[:120]})
            row["count"] = self._coerce_non_negative_int(row.get("count", 0), default=0) + 1
            if not str(row.get("label") or "").strip():
                row["label"] = query_text[:120]
            usage[term_key] = row
            updated_keys.append(term_key)

        if len(usage) > self.MEMORY_SEARCH_USAGE_MAX_TERMS:
            ranked = sorted(
                usage.items(),
                key=lambda kv: (self._coerce_non_negative_int(kv[1].get("count", 0), default=0), kv[0]),
                reverse=True,
            )
            usage = dict(ranked[: self.MEMORY_SEARCH_USAGE_MAX_TERMS])

        campaign_state[self.MEMORY_SEARCH_USAGE_KEY] = usage
        campaign.state_json = self._dump_json(campaign_state)

        characters = self.get_campaign_characters(campaign)
        hints: List[Dict[str, object]] = []
        seen_hint_keys: set[str] = set()
        for term_key in updated_keys:
            if term_key in seen_hint_keys:
                continue
            seen_hint_keys.add(term_key)
            row = usage.get(term_key) or {}
            count = self._coerce_non_negative_int(row.get("count", 0), default=0)
            if count < self.MEMORY_SEARCH_ROSTER_HINT_THRESHOLD:
                continue
            if not self._memory_search_term_looks_character_like(term_key):
                continue
            if isinstance(characters, dict) and self._resolve_existing_character_slug(characters, term_key):
                continue
            hints.append(
                {
                    "term": str(row.get("label") or term_key),
                    "slug": term_key,
                    "count": count,
                }
            )
        return hints

    def _character_updates_from_state_nulls(
        self,
        state_update: object,
        existing_chars: Dict[str, dict],
    ) -> Dict[str, object]:
        out: Dict[str, object] = {}
        if not isinstance(state_update, dict) or not isinstance(existing_chars, dict):
            return out
        for key, value in state_update.items():
            if value is not None:
                continue
            resolved = self._resolve_existing_character_slug(existing_chars, key)
            if resolved:
                out[resolved] = None
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
        fire_day, _ = ZorkEmulator._calendar_resolve_fire_point(
            current_day=current_day,
            current_hour=current_hour,
            time_remaining=time_remaining,
            time_unit=time_unit,
        )
        return fire_day

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
        normalized: dict[str, object] = {
            "name": name,
            "fire_day": fire_day,
            "fire_hour": fire_hour,
            "description": str(event.get("description") or "")[:200],
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
        calendar_changed = False
        for raw in calendar:
            normalized = cls._calendar_normalize_event(
                raw,
                current_day=current_day,
                current_hour=current_hour,
            )
            if normalized is None:
                continue
            fire_day = int(normalized.get("fire_day", current_day))
            fire_hour = cls._coerce_non_negative_int(
                normalized.get("fire_hour", 23), default=23
            )
            fire_hour = min(23, max(0, fire_hour))
            if isinstance(raw, dict):
                raw_fire_day = raw.get("fire_day")
                raw_fire_hour = raw.get("fire_hour")
                has_fire_day = isinstance(raw_fire_day, (int, float)) and not isinstance(
                    raw_fire_day, bool
                )
                has_fire_hour = isinstance(raw_fire_hour, (int, float)) and not isinstance(
                    raw_fire_hour, bool
                )
                if (not has_fire_day) or int(raw_fire_day) != fire_day:
                    raw["fire_day"] = fire_day
                    calendar_changed = True
                if (not has_fire_hour) or int(raw_fire_hour) != fire_hour:
                    raw["fire_hour"] = fire_hour
                    calendar_changed = True
                if "time_remaining" in raw:
                    raw.pop("time_remaining", None)
                    calendar_changed = True
                if "time_unit" in raw:
                    raw.pop("time_unit", None)
                    calendar_changed = True
            hours_remaining = ((fire_day - current_day) * 24) + (fire_hour - current_hour)
            days_remaining = fire_day - current_day
            if hours_remaining < 0:
                status = "overdue"
            elif days_remaining == 0:
                status = "today"
            elif hours_remaining <= 24:
                status = "imminent"
            else:
                status = "upcoming"
            view = dict(normalized)
            view["days_remaining"] = days_remaining
            view["hours_remaining"] = hours_remaining
            view["status"] = status
            entries.append(view)
        entries.sort(
            key=lambda item: (
                int(item.get("fire_day", current_day)),
                int(item.get("fire_hour", 23)),
                str(item.get("name", "")).lower(),
            )
        )
        if calendar_changed and isinstance(campaign_state, dict):
            campaign_state["calendar"] = calendar
        return entries

    @classmethod
    def _calendar_reminder_text(
        cls,
        calendar_entries: list[dict[str, object]],
        active_scene_names: list[str] | None = None,
        campaign_state: dict[str, object] | None = None,
    ) -> str:
        if not calendar_entries:
            return "None"

        def _event_key(event: dict[str, object]) -> str:
            name = str(event.get("name", "")).strip().lower()
            slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")[:80] or "event"
            created_day = event.get("created_day")
            created_hour = event.get("created_hour")
            if isinstance(created_day, (int, float)) and not isinstance(
                created_day, bool
            ) and isinstance(created_hour, (int, float)) and not isinstance(
                created_hour, bool
            ):
                return (
                    f"{slug}:"
                    f"{max(1, int(created_day))}:"
                    f"{min(23, max(0, int(created_hour)))}"
                )
            desc = str(event.get("description", "")).strip().lower()
            desc_slug = re.sub(r"[^a-z0-9]+", "-", desc).strip("-")[:40] or "na"
            return f"{slug}:{desc_slug}"

        def _reminder_bucket(hours: int) -> str | None:
            if hours == 0:
                return "now"
            if hours < 0:
                overdue = abs(hours)
                if overdue <= 3:
                    return f"overdue_1h_{overdue}"
                return f"overdue_6h_{overdue // 6}"
            if hours > 24:
                return f"future_12h_{hours // 12}"
            if hours > 6:
                return f"future_6h_{hours // 6}"
            if hours > 1:
                return f"future_2h_{hours // 2}"
            return "future_1h_1"

        alerts = []
        active_keys = {
            cls._calendar_name_key(name)
            for name in (active_scene_names or [])
            if cls._calendar_name_key(name)
        }
        global_tokens = {"all", "any", "everyone", "global", "scene", "party"}
        reminder_state: dict[str, object] = {}
        if isinstance(campaign_state, dict):
            raw_state = campaign_state.get(cls.CALENDAR_REMINDER_STATE_KEY)
            if isinstance(raw_state, dict):
                reminder_state = dict(raw_state)
        current_event_keys: set[str] = set()
        reminder_state_changed = False
        for event in calendar_entries:
            known_by = cls._calendar_known_by_from_event(event)
            if known_by:
                known_keys = {
                    cls._calendar_name_key(name)
                    for name in known_by
                    if cls._calendar_name_key(name)
                }
                if not (known_keys & global_tokens):
                    if not active_keys or not (known_keys & active_keys):
                        continue
            hours = int(event.get("hours_remaining", 0))
            name = str(event.get("name", "Unknown"))
            fire_day = int(event.get("fire_day", 1))
            fire_hour = max(0, min(23, int(event.get("fire_hour", 23))))
            bucket = _reminder_bucket(hours)
            if not bucket:
                continue
            event_key = _event_key(event)
            current_event_keys.add(event_key)
            if reminder_state.get(event_key) == bucket:
                continue
            if hours < 0:
                alerts.append(
                    f"- OVERDUE: {name} (was Day {fire_day}, {fire_hour:02d}:00; {abs(hours)} hour(s) overdue)"
                )
            elif hours == 0:
                alerts.append(
                    f"- NOW: {name} (fires at Day {fire_day}, {fire_hour:02d}:00)"
                )
            else:
                alerts.append(
                    f"- SOON: {name} (fires in {hours} hour(s) at Day {fire_day}, {fire_hour:02d}:00)"
                )
            reminder_state[event_key] = bucket
            reminder_state_changed = True
        if isinstance(campaign_state, dict):
            stale_keys = [key for key in list(reminder_state.keys()) if key not in current_event_keys]
            if stale_keys:
                for key in stale_keys:
                    reminder_state.pop(key, None)
                reminder_state_changed = True
            if reminder_state_changed:
                campaign_state[cls.CALENDAR_REMINDER_STATE_KEY] = reminder_state
        alerts = alerts[:2]
        return "\n".join(alerts) if alerts else "None"

    @classmethod
    def _sms_normalize_thread_key(cls, value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:80]

    @classmethod
    def _sms_threads_from_state(cls, campaign_state: Dict[str, object]) -> Dict[str, dict]:
        raw = campaign_state.get(cls.SMS_STATE_KEY) if isinstance(campaign_state, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        threads: Dict[str, dict] = {}
        for raw_key, raw_value in raw.items():
            key = cls._sms_normalize_thread_key(raw_key)
            if not key or not isinstance(raw_value, dict):
                continue
            label = str(raw_value.get("label") or raw_key).strip()[:80] or key
            raw_messages = raw_value.get("messages")
            if not isinstance(raw_messages, list):
                raw_messages = []
            messages = []
            for msg in raw_messages[-cls.SMS_MAX_MESSAGES_PER_THREAD :]:
                if not isinstance(msg, dict):
                    continue
                text = str(msg.get("message") or "").strip()
                if not text:
                    continue
                messages.append(
                    {
                        "from": str(msg.get("from") or "Unknown")[:80],
                        "to": str(msg.get("to") or "")[:80],
                        "message": text[:500],
                        "day": cls._coerce_non_negative_int(msg.get("day", 1), default=1) or 1,
                        "hour": min(
                            23,
                            max(0, cls._coerce_non_negative_int(msg.get("hour", 0), default=0)),
                        ),
                        "minute": min(
                            59,
                            max(0, cls._coerce_non_negative_int(msg.get("minute", 0), default=0)),
                        ),
                        "turn_id": cls._coerce_non_negative_int(msg.get("turn_id", 0), default=0),
                        "seq": cls._coerce_non_negative_int(msg.get("seq", 0), default=0),
                    }
                )
            threads[key] = {"label": label, "messages": messages}
        return threads

    @classmethod
    def _sms_list_threads(
        cls,
        campaign_state: Dict[str, object],
        wildcard: str = "*",
        limit: int = 20,
    ) -> list[dict[str, object]]:
        threads = cls._sms_threads_from_state(campaign_state)
        pattern = str(wildcard or "*").strip().lower() or "*"
        out: list[dict[str, object]] = []
        for key in reversed(list(threads.keys())):
            row = threads.get(key) or {}
            label = str(row.get("label") or key)
            if pattern != "*":
                if not fnmatch.fnmatch(key, pattern) and not fnmatch.fnmatch(
                    label.lower(), pattern
                ):
                    continue
            messages = row.get("messages")
            if not isinstance(messages, list):
                messages = []
            last = messages[-1] if messages else {}
            preview = str(last.get("message") or "").strip()
            if len(preview) > cls.SMS_MAX_PREVIEW_CHARS:
                preview = preview[: cls.SMS_MAX_PREVIEW_CHARS - 1].rstrip() + "…"
            out.append(
                {
                    "thread": key,
                    "label": label,
                    "count": len(messages),
                    "last_from": str(last.get("from") or ""),
                    "last_preview": preview,
                    "day": cls._coerce_non_negative_int(last.get("day", 0), default=0),
                    "hour": cls._coerce_non_negative_int(last.get("hour", 0), default=0),
                    "minute": cls._coerce_non_negative_int(last.get("minute", 0), default=0),
                }
            )
            if len(out) >= max(1, int(limit or 20)):
                break
        return out

    @classmethod
    def _sms_read_thread(
        cls,
        campaign_state: Dict[str, object],
        thread: str,
        limit: int = 20,
    ) -> tuple[str | None, str | None, list[dict[str, object]]]:
        threads = cls._sms_threads_from_state(campaign_state)
        if not threads:
            return None, None, []
        query_key = cls._sms_normalize_thread_key(thread)
        selected_key = query_key if query_key in threads else None
        if selected_key is None and query_key:
            for key in threads.keys():
                key_norm = cls._sms_normalize_thread_key(key)
                if query_key in key_norm:
                    selected_key = key
                    break
        if selected_key is None and not query_key:
            return None, None, []

        def _thread_matches(key: str, row: dict) -> bool:
            if not query_key:
                return False
            key_norm = cls._sms_normalize_thread_key(key)
            label_norm = cls._sms_normalize_thread_key(row.get("label"))
            if query_key and (
                query_key == key_norm
                or query_key in key_norm
                or query_key == label_norm
                or query_key in label_norm
            ):
                return True
            raw_messages = row.get("messages")
            if not isinstance(raw_messages, list):
                raw_messages = []
            for msg in raw_messages:
                if not isinstance(msg, dict):
                    continue
                from_norm = cls._sms_normalize_thread_key(msg.get("from"))
                to_norm = cls._sms_normalize_thread_key(msg.get("to"))
                if (from_norm and query_key in from_norm) or (to_norm and query_key in to_norm):
                    return True
            return False

        matched_keys: list[str] = []
        if selected_key is not None:
            matched_keys.append(selected_key)
        for key, row in threads.items():
            if key in matched_keys or not isinstance(row, dict):
                continue
            if _thread_matches(key, row):
                matched_keys.append(key)
        if not matched_keys:
            return None, None, []

        merged_messages: list[dict[str, object]] = []
        for key in matched_keys:
            row = threads.get(key) or {}
            messages = row.get("messages")
            if not isinstance(messages, list):
                messages = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                enriched = dict(msg)
                enriched["thread"] = key
                merged_messages.append(enriched)

        merged_messages.sort(
            key=lambda msg: (
                cls._coerce_non_negative_int(msg.get("day", 0), default=0),
                cls._coerce_non_negative_int(msg.get("hour", 0), default=0),
                cls._coerce_non_negative_int(msg.get("minute", 0), default=0),
                cls._coerce_non_negative_int(msg.get("turn_id", 0), default=0),
            )
        )
        capped = merged_messages[-max(1, min(40, int(limit or 20))) :]

        canonical_key = selected_key or query_key or matched_keys[0]
        first_row = threads.get(matched_keys[0]) or {}
        base_label = str(first_row.get("label") or matched_keys[0])
        if len(matched_keys) <= 1:
            resolved_label = base_label
        else:
            resolved_label = f"{base_label} (+{len(matched_keys) - 1} related thread(s))"
        return canonical_key, resolved_label, list(capped)

    @classmethod
    def _sms_actor_key(cls, actor_id: object) -> str:
        key = cls._sms_normalize_thread_key(f"actor-{actor_id}")
        return key or "actor-unknown"

    @classmethod
    def _sms_player_aliases(
        cls,
        *,
        actor_id: object,
        player_state: Dict[str, object] | None,
    ) -> set[str]:
        aliases: set[str] = set()

        def _add(raw: object) -> None:
            text = str(raw or "").strip()
            if not text:
                return
            norm = cls._sms_normalize_thread_key(text)
            if norm:
                aliases.add(norm)

        actor_text = str(actor_id or "").strip()
        _add(actor_text)
        if actor_text:
            _add(f"<@{actor_text}>")
            _add(f"<@!{actor_text}>")
            _add(f"player {actor_text}")
        if isinstance(player_state, dict):
            char_name = str(player_state.get("character_name") or "").strip()
            _add(char_name)
            for token in re.split(r"[\s\-]+", char_name):
                if len(token) >= 3:
                    _add(token)
        return aliases

    @classmethod
    def _sms_read_state_from_campaign_state(
        cls,
        campaign_state: Dict[str, object],
    ) -> Dict[str, Dict[str, object]]:
        raw = campaign_state.get(cls.SMS_READ_STATE_KEY) if isinstance(campaign_state, dict) else {}
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Dict[str, object]] = {}
        for raw_actor_key, raw_row in raw.items():
            actor_key = cls._sms_normalize_thread_key(raw_actor_key)
            if not actor_key or not isinstance(raw_row, dict):
                continue
            row_threads = raw_row.get("threads")
            if not isinstance(row_threads, dict):
                row_threads = {}
            cleaned_threads: Dict[str, int] = {}
            for raw_thread_key, raw_marker in row_threads.items():
                thread_key = cls._sms_normalize_thread_key(raw_thread_key)
                if not thread_key:
                    continue
                marker = cls._coerce_non_negative_int(raw_marker, default=0)
                cleaned_threads[thread_key] = marker
            out[actor_key] = {
                "threads": cleaned_threads,
                "last_notified_abs_hour": cls._coerce_non_negative_int(
                    raw_row.get("last_notified_abs_hour", -1),
                    default=-1,
                ),
            }
        return out

    @classmethod
    def _sms_mark_threads_read(
        cls,
        campaign_state: Dict[str, object],
        *,
        actor_id: object,
        player_state: Dict[str, object] | None,
        thread_markers: Dict[str, int],
    ) -> bool:
        if not isinstance(campaign_state, dict):
            return False
        if not isinstance(thread_markers, dict) or not thread_markers:
            return False
        actor_key = cls._sms_actor_key(actor_id)
        state = cls._sms_read_state_from_campaign_state(campaign_state)
        row = dict(state.get(actor_key) or {})
        threads = row.get("threads")
        if not isinstance(threads, dict):
            threads = {}
        changed = False
        for raw_thread_key, raw_marker in thread_markers.items():
            thread_key = cls._sms_normalize_thread_key(raw_thread_key)
            if not thread_key:
                continue
            marker = cls._coerce_non_negative_int(raw_marker, default=0)
            if marker <= 0:
                continue
            current = cls._coerce_non_negative_int(threads.get(thread_key, 0), default=0)
            if marker > current:
                threads[thread_key] = marker
                changed = True
        if not changed:
            return False
        row["threads"] = threads
        state[actor_key] = row
        campaign_state[cls.SMS_READ_STATE_KEY] = state
        return True

    @classmethod
    def _sms_unread_summary_for_player(
        cls,
        campaign_state: Dict[str, object],
        *,
        actor_id: object,
        player_state: Dict[str, object] | None,
    ) -> Dict[str, object]:
        aliases = cls._sms_player_aliases(actor_id=actor_id, player_state=player_state)
        if not aliases:
            return {"messages": 0, "threads": 0, "labels": []}
        actor_key = cls._sms_actor_key(actor_id)
        read_state = cls._sms_read_state_from_campaign_state(campaign_state)
        actor_row = read_state.get(actor_key) or {}
        read_threads = actor_row.get("threads")
        if not isinstance(read_threads, dict):
            read_threads = {}
        threads = cls._sms_threads_from_state(campaign_state)
        unread_messages = 0
        unread_threads = 0
        labels: list[str] = []
        for thread_key, row in threads.items():
            if not isinstance(row, dict):
                continue
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            seen_marker = cls._coerce_non_negative_int(read_threads.get(thread_key, 0), default=0)
            thread_unread = 0
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                to_norm = cls._sms_normalize_thread_key(msg.get("to"))
                if not to_norm or to_norm not in aliases:
                    continue
                seq = cls._coerce_non_negative_int(msg.get("seq", 0), default=0)
                turn_id = cls._coerce_non_negative_int(msg.get("turn_id", 0), default=0)
                marker = seq if seq > 0 else turn_id
                if marker > seen_marker:
                    thread_unread += 1
            if thread_unread <= 0:
                continue
            unread_messages += thread_unread
            unread_threads += 1
            label = str(row.get("label") or thread_key).strip()
            if label:
                labels.append(label[:40])
        deduped_labels: list[str] = []
        seen_labels: set[str] = set()
        for label in labels:
            key = cls._sms_normalize_thread_key(label)
            if not key or key in seen_labels:
                continue
            seen_labels.add(key)
            deduped_labels.append(label)
            if len(deduped_labels) >= 3:
                break
        return {
            "messages": unread_messages,
            "threads": unread_threads,
            "labels": deduped_labels,
            "last_notified_abs_hour": cls._coerce_non_negative_int(
                actor_row.get("last_notified_abs_hour", -1),
                default=-1,
            ),
        }

    @classmethod
    def _sms_unread_hourly_notification(
        cls,
        campaign_state: Dict[str, object],
        *,
        actor_id: object,
        player_state: Dict[str, object] | None,
        game_time: Dict[str, int] | None,
    ) -> str | None:
        if not isinstance(campaign_state, dict):
            return None
        summary = cls._sms_unread_summary_for_player(
            campaign_state,
            actor_id=actor_id,
            player_state=player_state,
        )
        unread_messages = cls._coerce_non_negative_int(summary.get("messages", 0), default=0)
        unread_threads = cls._coerce_non_negative_int(summary.get("threads", 0), default=0)
        if unread_messages <= 0 or unread_threads <= 0:
            return None
        game_time_obj = game_time if isinstance(game_time, dict) else {}
        day = max(1, cls._coerce_non_negative_int(game_time_obj.get("day", 1), default=1))
        hour = min(
            23,
            max(0, cls._coerce_non_negative_int(game_time_obj.get("hour", 0), default=0)),
        )
        abs_hour = ((day - 1) * 24) + hour
        actor_key = cls._sms_actor_key(actor_id)
        read_state = cls._sms_read_state_from_campaign_state(campaign_state)
        row = dict(read_state.get(actor_key) or {})
        last_notified = cls._coerce_non_negative_int(
            row.get("last_notified_abs_hour", -1),
            default=-1,
        )
        if last_notified == abs_hour:
            return None
        row["last_notified_abs_hour"] = abs_hour
        read_state[actor_key] = row
        campaign_state[cls.SMS_READ_STATE_KEY] = read_state
        labels = summary.get("labels") if isinstance(summary.get("labels"), list) else []
        labels = [str(label).strip()[:40] for label in labels if str(label).strip()]
        suffix = f" ({', '.join(labels[:2])})" if labels else ""
        return (
            f"📨 Unread SMS: {unread_messages} message(s) in "
            f"{unread_threads} thread(s){suffix}."
        )

    _SMS_ARTICLES = frozenset({"the", "a", "an", "my", "back"})

    @classmethod
    def _extract_inline_sms_intent(
        cls,
        action: str,
    ) -> tuple[str, str] | None:
        text = str(action or "").strip()
        if not text:
            return None
        # Pattern 1: Colon-delimited — "text the Doc: hello"
        m = re.match(
            r"^\s*(?:i\s+)?(?:send\s+)?(?:sms|text|message)\s+(?:back\s+)?(?:to\s+)?([^:\n]{1,120})\s*:\s*(.+?)\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            recipient = str(m.group(1) or "").strip().strip("\"'` ")
            message = str(m.group(2) or "").strip()
        else:
            # Pattern 2: Space-delimited — "text Doc hello"
            # Allow leading articles to be part of the captured recipient:
            # "sms the Doc the words" → recipient="the Doc", message="the words"
            m = re.match(
                r"^\s*(?:i\s+)?(?:send\s+)?(?:sms|text|message)\s+(?:to\s+)?"
                r"((?:the|a|an|my)\s+[^\s:\n]{1,80}|[^\s:\n]{1,80})\s+(.+?)\s*$",
                text,
                flags=re.IGNORECASE,
            )
            if not m:
                return None
            recipient = str(m.group(1) or "").strip().strip("\"'` ")
            message = str(m.group(2) or "").strip()
            # If recipient is still a bare article, something went wrong — bail
            if recipient.lower() in cls._SMS_ARTICLES:
                return None
        if (
            len(message) >= 2
            and message[0] == message[-1]
            and message[0] in {'"', "'"}
        ):
            message = message[1:-1].strip()
        if not recipient or not message:
            return None
        return recipient[:80], message[:500]

    @classmethod
    def _sms_write(
        cls,
        campaign_state: Dict[str, object],
        *,
        thread: str,
        sender: str,
        recipient: str,
        message: str,
        game_time: Dict[str, int],
        turn_id: int = 0,
    ) -> tuple[str, str, dict[str, object]]:
        threads = cls._sms_threads_from_state(campaign_state)
        thread_key = cls._sms_normalize_thread_key(thread or recipient or sender or "unknown")
        if not thread_key:
            thread_key = "unknown"
        existing = threads.pop(
            thread_key,
            {"label": thread or recipient or sender or thread_key, "messages": []},
        )
        label = str(existing.get("label") or thread or recipient or sender or thread_key).strip()[:80] or thread_key
        messages = existing.get("messages")
        if not isinstance(messages, list):
            messages = []
        entry = {
            "from": str(sender or "Unknown")[:80],
            "to": str(recipient or "")[:80],
            "message": str(message or "").strip()[:500],
            "day": cls._coerce_non_negative_int(game_time.get("day", 1), default=1) or 1,
            "hour": min(23, max(0, cls._coerce_non_negative_int(game_time.get("hour", 0), default=0))),
            "minute": min(59, max(0, cls._coerce_non_negative_int(game_time.get("minute", 0), default=0))),
            "turn_id": max(0, int(turn_id or 0)),
            "seq": 0,
        }
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                if (
                    str(last.get("from") or "") == str(entry.get("from") or "")
                    and str(last.get("to") or "") == str(entry.get("to") or "")
                    and str(last.get("message") or "") == str(entry.get("message") or "")
                    and cls._coerce_non_negative_int(last.get("day", 0), default=0)
                    == cls._coerce_non_negative_int(entry.get("day", 0), default=0)
                    and cls._coerce_non_negative_int(last.get("hour", 0), default=0)
                    == cls._coerce_non_negative_int(entry.get("hour", 0), default=0)
                    and cls._coerce_non_negative_int(last.get("minute", 0), default=0)
                    == cls._coerce_non_negative_int(entry.get("minute", 0), default=0)
                ):
                    threads[thread_key] = {"label": label, "messages": messages}
                    campaign_state[cls.SMS_STATE_KEY] = threads
                    return thread_key, label, dict(last)
        next_seq = cls._coerce_non_negative_int(
            campaign_state.get(cls.SMS_MESSAGE_SEQ_KEY, 0), default=0
        ) + 1
        entry["seq"] = max(1, next_seq)
        messages.append(entry)
        messages = messages[-cls.SMS_MAX_MESSAGES_PER_THREAD :]
        threads[thread_key] = {"label": label, "messages": messages}
        while len(threads) > cls.SMS_MAX_THREADS:
            oldest_key = next(iter(threads))
            threads.pop(oldest_key, None)
        campaign_state[cls.SMS_STATE_KEY] = threads
        campaign_state[cls.SMS_MESSAGE_SEQ_KEY] = int(entry.get("seq", next_seq))
        return thread_key, label, entry

    def _register_pending_sms_task(self, campaign_id: str, task: asyncio.Task) -> None:
        bucket = self._pending_sms_tasks.setdefault(str(campaign_id), set())
        bucket.add(task)

        def _cleanup(done_task: asyncio.Task) -> None:
            tasks = self._pending_sms_tasks.get(str(campaign_id))
            if tasks is None:
                return
            tasks.discard(done_task)
            if not tasks:
                self._pending_sms_tasks.pop(str(campaign_id), None)

        task.add_done_callback(_cleanup)

    def cancel_pending_sms_deliveries(self, campaign_id: str) -> int:
        tasks = self._pending_sms_tasks.pop(str(campaign_id), set())
        cancelled = 0
        for task in list(tasks):
            if task is not None and not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    async def _sms_delivery_task(
        self,
        *,
        campaign_id: str,
        delay_seconds: int,
        thread: str,
        sender: str,
        recipient: str,
        message: str,
        turn_id: int = 0,
    ) -> None:
        try:
            await asyncio.sleep(max(0, int(delay_seconds)))
        except asyncio.CancelledError:
            return

        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return
            campaign_state = self.get_campaign_state(campaign)
            game_time = self._extract_game_time_snapshot(campaign_state)
            effective_turn_id = max(0, int(turn_id or 0))
            if effective_turn_id <= 0:
                latest_turn = (
                    session.query(Turn)
                    .filter(Turn.campaign_id == str(campaign_id))
                    .order_by(Turn.id.desc())
                    .first()
                )
                effective_turn_id = int(latest_turn.id) if latest_turn is not None else 0
            self._sms_write(
                campaign_state,
                thread=thread,
                sender=sender,
                recipient=recipient,
                message=message,
                game_time=game_time,
                turn_id=effective_turn_id,
            )
            campaign.state_json = self._dump_json(campaign_state)
            campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()

    def schedule_sms_thread_delivery(
        self,
        campaign_id: str,
        *,
        thread: str,
        sender: str,
        recipient: str,
        message: str,
        delay_seconds: int,
        turn_id: int = 0,
    ) -> tuple[bool, str, int]:
        thread_clean = str(thread or "").strip()[:80]
        sender_clean = str(sender or "").strip()[:80]
        recipient_clean = str(recipient or "").strip()[:80]
        message_clean = str(message or "").strip()[:500]
        if not thread_clean or not sender_clean or not recipient_clean or not message_clean:
            return False, "invalid_payload", 0
        try:
            delay = max(0, min(86_400, int(delay_seconds)))
        except Exception:
            return False, "invalid_delay", 0
        try:
            task = asyncio.create_task(
                self._sms_delivery_task(
                    campaign_id=str(campaign_id),
                    delay_seconds=delay,
                    thread=thread_clean,
                    sender=sender_clean,
                    recipient=recipient_clean,
                    message=message_clean,
                    turn_id=max(0, int(turn_id or 0)),
                )
            )
        except RuntimeError:
            return False, "event_loop_unavailable", 0
        self._register_pending_sms_task(str(campaign_id), task)
        return True, "scheduled", delay

    @classmethod
    def _normalize_calendar_update(cls, raw: object) -> Dict[str, object] | None:
        if not isinstance(raw, dict) or not raw:
            return None
        has_add = "add" in raw
        has_remove = "remove" in raw
        if not has_add and not has_remove and raw.get("name"):
            raw = {"add": [dict(raw)]}
            has_add = True
        result: Dict[str, object] = {}
        if has_add:
            add_val = raw.get("add")
            if isinstance(add_val, dict):
                add_val = [add_val]
            if isinstance(add_val, list):
                normalized_add: list[dict[str, object]] = []
                for entry in add_val:
                    if not isinstance(entry, dict):
                        continue
                    row = dict(entry)
                    if "time_remaining" not in row:
                        if "hours_remaining" in row:
                            row["time_remaining"] = row.pop("hours_remaining")
                            row.setdefault("time_unit", "hours")
                        elif "days_remaining" in row:
                            row["time_remaining"] = row.pop("days_remaining")
                            row.setdefault("time_unit", "days")
                    for echo_key in (
                        "status",
                        "_status",
                        "_hours_until",
                        "_days_until",
                        "hours_remaining",
                        "days_remaining",
                        "created_day",
                        "created_hour",
                    ):
                        row.pop(echo_key, None)
                    normalized_add.append(row)
                result["add"] = normalized_add
        if has_remove:
            remove_val = raw.get("remove")
            if isinstance(remove_val, str):
                remove_val = [remove_val]
            if isinstance(remove_val, list):
                result["remove"] = remove_val
        return result if result else None

    @classmethod
    def _extract_calendar_update_from_state_update(
        cls,
        state_update: object,
        calendar_update: object,
    ) -> tuple[object, object]:
        if not isinstance(state_update, dict):
            return state_update, calendar_update
        merged_calendar = (
            dict(calendar_update) if isinstance(calendar_update, dict) else {}
        )
        merged_add = list(merged_calendar.get("add") or [])
        changed = False
        cleaned_state_update = dict(state_update)

        def _consume_events(raw_entry: object) -> None:
            nonlocal changed
            if isinstance(raw_entry, dict) and isinstance(raw_entry.get("events"), list):
                for item in raw_entry.get("events") or []:
                    if isinstance(item, dict):
                        merged_add.append(dict(item))
                        changed = True
            elif isinstance(raw_entry, list):
                for item in raw_entry:
                    if isinstance(item, dict):
                        merged_add.append(dict(item))
                        changed = True

        for legacy_key in ("calendar", "cal", "events"):
            raw_entry = cleaned_state_update.get(legacy_key)
            if raw_entry is None:
                continue
            before_count = len(merged_add)
            _consume_events(raw_entry)
            if len(merged_add) != before_count:
                cleaned_state_update.pop(legacy_key, None)

        if not changed:
            return state_update, calendar_update
        merged_calendar["add"] = merged_add
        return cleaned_state_update, merged_calendar

    def _apply_calendar_update(
        self,
        campaign_state: Dict[str, object],
        calendar_update: dict,
    ) -> Dict[str, object]:
        """Process calendar add/remove ops and persist absolute fire_day entries."""
        if not isinstance(calendar_update, dict):
            return campaign_state
        calendar_update = self._normalize_calendar_update(calendar_update)
        if calendar_update is None:
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
            context_text = " ".join(str(calendar_update.get("_context") or "").lower().split())
            allowed_remove_set: set[str] = set()
            for event in calendar:
                name_raw = str(event.get("name", "")).strip()
                if not name_raw:
                    continue
                name_key = name_raw.lower()
                if name_key not in remove_set:
                    continue
                name_norm = re.sub(r"[^a-z0-9]+", " ", name_key).strip()
                name_tokens = [token for token in name_norm.split() if len(token) > 2]
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
                        fire_day_int < int(current_day)
                        or (fire_day_int == int(current_day) and fire_hour_int <= int(current_hour))
                    )
                if event_is_past or (
                    name_mentioned and not has_premature and (has_completion or has_cleanup_intent)
                ):
                    allowed_remove_set.add(name_key)
            calendar = [
                event
                for event in calendar
                if str(event.get("name", "")).strip().lower() not in allowed_remove_set
            ]

        to_add = calendar_update.get("add")
        if isinstance(to_add, list):
            for entry in to_add:
                if not isinstance(entry, dict):
                    continue
                name = str(
                    entry.get("name")
                    or entry.get("title")
                    or entry.get("event_key")
                    or ""
                ).strip()
                if not name:
                    continue
                fire_day = entry.get("fire_day")
                fire_hour = entry.get("fire_hour")
                if not isinstance(fire_day, (int, float)) or isinstance(fire_day, bool):
                    day_alias = entry.get("day")
                    if isinstance(day_alias, (int, float)) and not isinstance(day_alias, bool):
                        fire_day = int(day_alias)
                if not isinstance(fire_hour, (int, float)) or isinstance(fire_hour, bool):
                    time_alias = str(entry.get("time") or "").strip()
                    if time_alias:
                        match = re.search(
                            r"(?:day\s*(?P<day>\d+)[,\s-]*)?(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>[ap]m)?",
                            time_alias,
                            re.IGNORECASE,
                        )
                        if match:
                            if not isinstance(fire_day, (int, float)) or isinstance(fire_day, bool):
                                day_group = match.group("day")
                                if day_group:
                                    fire_day = int(day_group)
                            parsed_hour = int(match.group("hour"))
                            ampm = str(match.group("ampm") or "").strip().lower()
                            if ampm == "pm" and parsed_hour < 12:
                                parsed_hour += 12
                            elif ampm == "am" and parsed_hour == 12:
                                parsed_hour = 0
                            fire_hour = parsed_hour
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
                        current_day=int(current_day) if isinstance(current_day, (int, float)) else 1,
                        current_hour=int(current_hour) if isinstance(current_hour, (int, float)) else 8,
                        time_remaining=entry.get("time_remaining", 1),
                        time_unit=entry.get("time_unit", "days"),
                    )
                event = {
                    "name": name,
                    "fire_day": resolved_fire_day,
                    "fire_hour": resolved_fire_hour,
                    "created_day": current_day,
                    "created_hour": current_hour,
                    "description": str(
                        entry.get("description")
                        or entry.get("notes")
                        or entry.get("details")
                        or ""
                    )[:200],
                    "known_by": self._calendar_known_by_from_event(entry),
                }
                location_text = str(entry.get("location") or "").strip()
                if location_text and not str(event.get("description") or "").strip():
                    event["description"] = f"Location: {location_text}"[:200]
                elif location_text:
                    event["description"] = (
                        f"{str(event.get('description') or '').strip()} Location: {location_text}"
                    )[:200]
                target_players = self._calendar_target_tokens_from_event(entry)
                visibility = str(entry.get("visibility") or "").strip().lower()
                if not target_players and visibility == "private":
                    actor_id = str(entry.get("actor_id") or "").strip()
                    if actor_id:
                        target_players = [actor_id]
                if target_players:
                    event["target_players"] = target_players
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

    @classmethod
    def format_roster(cls, characters: Dict[str, dict]) -> str:
        """Format the character roster for display. Shared by intercepted and cog paths."""
        if not characters:
            return "No characters in the roster yet."
        lines = ["**Character Roster:**"]
        for slug, char in characters.items():
            name = char.get("name", slug)
            loc = char.get("location", "unknown")
            status = char.get("current_status", "")
            bg = char.get("background", "")
            origin = bg.split(".")[0].strip() if bg else ""
            deceased = char.get("deceased_reason")
            entry = f"- **{name}** ({slug})"
            if deceased:
                entry += f" [DECEASED: {deceased}]"
            else:
                entry += f" — {loc}"
                if status:
                    entry += f" | {status}"
            if origin:
                entry += f"\n  *{origin}.*"
            lines.append(entry)
        return "\n".join(lines)

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
                    "speech_style": char.get("speech_style"),
                    "literary_style": char.get("literary_style"),
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

    def _literary_styles_for_prompt(
        self,
        campaign_state: dict[str, Any],
        characters_for_prompt: list[dict[str, object]],
    ) -> Optional[str]:
        styles = campaign_state.get(self.LITERARY_STYLES_STATE_KEY)
        if not isinstance(styles, dict) or not styles:
            return None

        active_refs: set[str] = set()
        for char in characters_for_prompt or []:
            if not isinstance(char, dict):
                continue
            ref = str(char.get("literary_style") or "").strip()
            if ref:
                active_refs.add(ref)

        def _sort_key(key: str) -> tuple[int, str]:
            return (0 if key in active_refs else 1, key)

        lines: list[str] = []
        budget = self.MAX_LITERARY_STYLES_PROMPT_CHARS
        for key in sorted(styles.keys(), key=_sort_key):
            entry = styles.get(key)
            if not isinstance(entry, dict):
                continue
            profile = str(entry.get("profile") or "").strip()
            if not profile:
                continue
            line = f"  {key}: {profile}"
            if len(line) > budget:
                break
            lines.append(line)
            budget -= len(line) + 1
            if budget <= 0:
                break
        return "\n".join(lines) if lines else None

    @staticmethod
    def _puzzle_system_for_prompt(campaign_state: dict[str, Any]) -> str | None:
        """Build prompt sections for active puzzles, minigames, and dice checks."""
        parts: list[str] = []

        puzzle_mode = campaign_state.get("puzzle_mode")
        if puzzle_mode and puzzle_mode != "none":
            parts.append(f"PUZZLE_CONFIG:\n  mode: {puzzle_mode}")

        active_puzzle = campaign_state.get("_active_puzzle")
        if isinstance(active_puzzle, dict):
            from .core.puzzles import PuzzleState, PuzzleEngine
            ps = PuzzleState.from_dict(active_puzzle)
            parts.append(PuzzleEngine.render_prompt_section(ps))

        puzzle_result = campaign_state.get("_puzzle_result")
        if isinstance(puzzle_result, dict):
            lines = ["PUZZLE_RESULT:"]
            for k, v in puzzle_result.items():
                lines.append(f"  {k}: {v}")
            parts.append("\n".join(lines))

        active_minigame = campaign_state.get("_active_minigame")
        if isinstance(active_minigame, dict):
            from .core.minigames import MinigameState, MinigameEngine
            ms = MinigameState.from_dict(active_minigame)
            parts.append(MinigameEngine.render_prompt_section(ms))

        minigame_result = campaign_state.get("_minigame_result")
        if isinstance(minigame_result, dict):
            lines = ["MINIGAME_RESULT:"]
            for k, v in minigame_result.items():
                lines.append(f"  {k}: {v}")
            parts.append("\n".join(lines))

        last_dice = campaign_state.get("_last_dice_check")
        if isinstance(last_dice, dict):
            attr = last_dice.get("attribute", "skill")
            roll_val = last_dice.get("roll", 0)
            mod = last_dice.get("modifier", 0)
            total = last_dice.get("total", 0)
            dc = last_dice.get("dc", 0)
            success = last_dice.get("success", False)
            context = last_dice.get("context", "")
            parts.append(
                f"LAST_DICE_CHECK:\n"
                f"  attribute: {attr}\n"
                f"  roll: {roll_val} + {mod} = {total} vs DC {dc}\n"
                f"  result: {'success' if success else 'failure'}\n"
                f"  context: \"{context}\""
            )

        return "\n\n".join(parts) if parts else None

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
        cleaned = response.strip()
        json_text = self._extract_json(cleaned)
        if json_text:
            return json_text
        # Repair common truncated-object case from the model:
        # starts with '{' but omitted the final closing brace.
        if cleaned.startswith("{") and not cleaned.endswith("}"):
            repaired = f"{cleaned}}}"
            try:
                parsed = self._parse_json_lenient(repaired)
                # Only accept the repair if it produced a structurally
                # complete response (not a truncated partial dict).
                if isinstance(parsed, dict) and parsed:
                    has_narration = bool(parsed.get("narration"))
                    has_tool_call = bool(parsed.get("tool_call"))
                    if has_narration or has_tool_call:
                        return repaired
            except Exception:
                pass
        return cleaned

    def _extract_ascii_map(self, text: str) -> str:
        if not text:
            return ""
        lines: list[str] = []
        for line in text.splitlines():
            if "```" in line:
                continue
            lines.append(line.rstrip())
        return "\n".join(lines).strip()

    def _map_location_components(
        self,
        location_value: object,
        room_title_value: object,
        room_summary_value: object,
    ) -> dict[str, str]:
        location = str(location_value or "").strip()
        room_title = str(room_title_value or "").strip()
        room_summary = str(room_summary_value or "").strip()
        summary_first = room_summary.splitlines()[0].strip() if room_summary else ""
        if "." in summary_first:
            summary_first = summary_first.split(".", 1)[0].strip()
        display = room_title or location or summary_first
        display = re.sub(r"\s+", " ", display).strip()[:120]
        key_source = location or room_title or display
        key = re.sub(r"[^a-z0-9]+", "-", key_source.lower()).strip("-")[:80]
        hint = re.sub(r"\s+", " ", room_summary).strip()[:180]
        has_data = bool(location or room_title or room_summary)
        return {
            "key": key or ("unknown-location" if has_data else ""),
            "display": display or ("Unknown" if has_data else ""),
            "hint": hint,
        }

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

    @classmethod
    def _calendar_name_key(cls, value: object) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "", text)

    @classmethod
    def _calendar_known_by_from_event(cls, event: object) -> list[str]:
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
    def _calendar_target_tokens_from_event(cls, event: object) -> list[str]:
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

    def _active_scene_character_names(
        self,
        player_state: dict[str, object],
        party_snapshot: list[dict[str, object]],
        characters_for_prompt: list[dict[str, object]],
    ) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()

        def _add_name(raw_name: object) -> None:
            text = str(raw_name or "").strip()
            if not text:
                return
            key = self._calendar_name_key(text)
            if not key or key in seen:
                return
            seen.add(key)
            names.append(text[:80])

        _add_name(player_state.get("character_name"))
        for entry in party_snapshot:
            if not isinstance(entry, dict):
                continue
            _add_name(entry.get("name"))

        for entry in characters_for_prompt:
            if not isinstance(entry, dict):
                continue
            if entry.get("deceased_reason"):
                continue
            char_name = entry.get("name") or entry.get("_slug")
            char_state = {
                "location": entry.get("location"),
                "room_title": entry.get("room_title"),
                "room_summary": entry.get("room_summary"),
                "room_id": entry.get("room_id"),
            }
            if self._same_scene(player_state, char_state):
                _add_name(char_name)
        return names

    @classmethod
    def _memory_lookup_enabled_for_prompt(
        cls,
        summary_text: object,
        *,
        source_material_available: bool = False,
        action_text: object = None,
    ) -> bool:
        text = " ".join(str(action_text or "").strip().lower().split())
        source_lookup_requested = (
            bool(text)
            and not bool(re.match(r"\s*\[ooc\b", text, re.IGNORECASE))
            and any(
                marker in text
                for marker in (
                    "remember",
                    "recall",
                    "what happened",
                    "previously",
                    "backstory",
                    "history",
                    "who is",
                    "what is",
                    "according to",
                    "from the book",
                    "from source",
                    "source material",
                    "canon",
                    "lore",
                    "look up",
                )
            )
        )
        summary_len = len(str(summary_text or "").strip())
        if summary_len >= cls.MEMORY_LOOKUP_MIN_SUMMARY_CHARS:
            return True
        if source_material_available and source_lookup_requested:
            return True
        return False

    def build_prompt(
        self,
        campaign: Campaign,
        player: Player,
        action: str,
        turns: list[Turn],
        party_snapshot: list[dict[str, object]] | None = None,
        is_new_player: bool = False,
        turn_visibility_default: str = "public",
        turn_attachment_context: str | None = None,
        tail_extra_lines: list[str] | None = None,
        bootstrap_only: bool = False,
        prompt_stage: str = PROMPT_STAGE_FINAL,
    ) -> tuple[str, str]:
        stage = str(prompt_stage or self.PROMPT_STAGE_FINAL).strip().lower()
        if stage not in {
            self.PROMPT_STAGE_BOOTSTRAP,
            self.PROMPT_STAGE_RESEARCH,
            self.PROMPT_STAGE_FINAL,
        }:
            stage = self.PROMPT_STAGE_FINAL
        bootstrap_only = bootstrap_only or stage == self.PROMPT_STAGE_BOOTSTRAP
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

        player_registry = self._campaign_player_registry(campaign.id, self._session_factory)
        player_names: Dict[str, str] = {}
        player_slugs: Dict[str, str] = {}
        for raw_actor_id, info in player_registry.get("by_actor_id", {}).items():
            actor_id = str(raw_actor_id or "").strip()
            if not actor_id:
                continue
            name = str(info.get("name") or "").strip()
            slug = str(info.get("slug") or "").strip()
            if name:
                player_names[actor_id] = name
            if slug:
                player_slugs[actor_id] = slug
        viewer_slug = player_slugs.get(player.actor_id) or self._player_visibility_slug(
            player.actor_id
        )
        viewer_location_key = self._room_key_from_player_state(player_state).lower()
        summary = self._compose_world_summary(
            campaign,
            state,
            turns=turns,
            viewer_actor_id=player.actor_id,
            viewer_slug=viewer_slug,
            viewer_location_key=viewer_location_key,
            max_chars=self.MAX_SUMMARY_CHARS,
        )

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
            if not self._turn_visible_to_viewer(
                turn,
                player.actor_id,
                viewer_slug,
                viewer_location_key,
            ):
                continue
            turn_prefix = self._turn_context_prefix(turn)
            if turn.kind == "player":
                if ooc_re.match(content):
                    continue
                clipped = content
                clipped = self._strip_inventory_mentions(clipped)
                name = player_names.get(turn.actor_id or "")
                if name:
                    label = f"PLAYER ({name.upper()})"
                else:
                    label = "PLAYER"
                recent_lines.append(f"{turn_prefix} {label}: {clipped}")
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
                recent_lines.append(f"{turn_prefix} NARRATOR: {clipped}")
        recent_text = "\n".join(recent_lines) if recent_lines else "None"

        rails_context = self._build_rails_context(player_state, party_snapshot)
        characters = self.get_campaign_characters(campaign)
        characters_for_prompt = self._build_characters_for_prompt(characters, player_state, recent_text)
        characters_for_prompt = self._fit_characters_to_budget(characters_for_prompt, self.MAX_CHARACTERS_CHARS)
        story_context = self._build_story_context(state)
        on_rails = bool(state.get("on_rails", False))
        active_scene_names = self._active_scene_character_names(
            player_state,
            party_snapshot,
            characters_for_prompt,
        )
        game_time = state.get("game_time", {})
        speed_mult = state.get("speed_multiplier", 1.0)
        difficulty = self.normalize_difficulty(state.get("difficulty", "normal"))
        response_style_note = self._turn_stage_note(difficulty, stage)
        calendar_state_before = json.dumps(
            state.get("calendar") or [],
            ensure_ascii=True,
            sort_keys=True,
        )
        calendar_for_prompt = self._calendar_for_prompt(state)
        calendar_state_after = json.dumps(
            state.get("calendar") or [],
            ensure_ascii=True,
            sort_keys=True,
        )
        calendar_reminder_state_before = json.dumps(
            state.get(self.CALENDAR_REMINDER_STATE_KEY) or {},
            ensure_ascii=True,
            sort_keys=True,
        )
        calendar_reminders = self._calendar_reminder_text(
            calendar_for_prompt,
            active_scene_names=active_scene_names,
            campaign_state=state,
        )
        calendar_reminder_state_after = json.dumps(
            state.get(self.CALENDAR_REMINDER_STATE_KEY) or {},
            ensure_ascii=True,
            sort_keys=True,
        )
        if (
            calendar_reminder_state_after != calendar_reminder_state_before
            or calendar_state_after != calendar_state_before
        ):
            campaign.state_json = self._dump_json(state)
        source_payload = self._source_material_prompt_payload(campaign.id)
        literary_styles_text = self._literary_styles_for_prompt(
            state,
            characters_for_prompt,
        )
        memory_lookup_enabled = self._memory_lookup_enabled_for_prompt(
            summary,
            source_material_available=bool(source_payload.get("available")),
            action_text=action,
        )

        active_location_context = {
            "room_title": player_state.get("room_title"),
            "location": player_state.get("location"),
            "room_summary": player_state.get("room_summary"),
        }

        effective_turn_visibility_default = self._default_prompt_turn_visibility(
            turn_visibility_default,
            player_state,
        )
        user_prompt = (
            f"CAMPAIGN: {campaign.name}\n"
            f"PLAYER_ID: {player.actor_id}\n"
            f"IS_NEW_PLAYER: {str(is_new_player).lower()}\n"
            f"TURN_VISIBILITY_DEFAULT: {effective_turn_visibility_default}\n"
            f"GUARDRAILS_ENABLED: {str(guardrails_enabled).lower()}\n"
            f"RAILS_CONTEXT: {self._dump_json(rails_context)}\n"
        )
        if source_payload.get("available"):
            user_prompt += (
                f"SOURCE_MATERIAL_DOCS: {self._dump_json(source_payload.get('docs') or [])}\n"
                f"SOURCE_MATERIAL_KEYS: {self._dump_json(source_payload.get('keys') or [])}\n"
                f"SOURCE_MATERIAL_SNIPPET_COUNT: {source_payload.get('chunk_count')}\n"
                f"SOURCE_MATERIAL_CHUNK_COUNT: {source_payload.get('chunk_count')}\n"
            )
            source_digests = source_payload.get("digests") or {}
            if source_digests:
                for digest_key, digest_text in source_digests.items():
                    user_prompt += (
                        f"SOURCE_MATERIAL_DIGEST [{digest_key}]:\n{digest_text}\n"
                    )
        user_prompt += (
            f"CURRENT_GAME_TIME: {self._dump_json(game_time)}\n"
            f"SPEED_MULTIPLIER: {speed_mult}\n"
            f"DIFFICULTY: {difficulty}\n"
            f"ACTIVE_PLAYER_LOCATION: {self._dump_json(active_location_context)}\n"
            f"MEMORY_LOOKUP_ENABLED: {str(memory_lookup_enabled).lower()}\n"
            f"RECENT_TURNS_LOADED: {str(not bootstrap_only).lower()}\n"
        )
        user_prompt += (
            f"WORLD_CHARACTERS: {self._dump_json(characters_for_prompt)}\n"
            f"PLAYER_CARD: {self._dump_json(player_card)}\n"
            f"PARTY_SNAPSHOT: {self._dump_json(party_snapshot)}\n"
        )
        if literary_styles_text:
            user_prompt += f"LITERARY_STYLES:\n{literary_styles_text}\n"
        _puzzle_text = self._puzzle_system_for_prompt(state)
        if _puzzle_text:
            user_prompt += f"{_puzzle_text}\n"
        if not bootstrap_only:
            if story_context:
                user_prompt += f"STORY_CONTEXT:\n{story_context}\n"
            user_prompt += (
                f"WORLD_SUMMARY: {summary}\n"
                f"WORLD_STATE: {self._dump_json(model_state)}\n"
                f"CALENDAR: {self._dump_json(calendar_for_prompt)}\n"
                f"CALENDAR_REMINDERS:\n{calendar_reminders}\n"
                f"RECENT_TURNS:\n{recent_text}\n"
            )
        turn_prompt_tail = self._build_turn_prompt_tail(
            player_state,
            action,
            response_style_note,
            turn_attachment_context=turn_attachment_context,
            extra_lines=tail_extra_lines,
        )
        if turn_prompt_tail:
            user_prompt += f"{turn_prompt_tail}\n"

        if stage == self.PROMPT_STAGE_BOOTSTRAP:
            system_prompt = self.BOOTSTRAP_SYSTEM_PROMPT
            system_prompt = f"{system_prompt}{self.RECENT_TURNS_TOOL_PROMPT}"
            if memory_lookup_enabled:
                system_prompt = f"{system_prompt}{self.MEMORY_BOOTSTRAP_TOOL_PROMPT}"
            else:
                system_prompt = f"{system_prompt}{self.MEMORY_TOOL_DISABLED_PROMPT}"
        elif stage == self.PROMPT_STAGE_RESEARCH:
            system_prompt = self.RESEARCH_SYSTEM_PROMPT
            if guardrails_enabled:
                system_prompt = f"{system_prompt}{self.GUARDRAILS_SYSTEM_PROMPT}"
            if on_rails:
                system_prompt = f"{system_prompt}{self.ON_RAILS_SYSTEM_PROMPT}"
            if memory_lookup_enabled:
                system_prompt = f"{system_prompt}{self.MEMORY_TOOL_PROMPT}"
            else:
                system_prompt = f"{system_prompt}{self.MEMORY_TOOL_DISABLED_PROMPT}"
            system_prompt = f"{system_prompt}{self.SMS_TOOL_PROMPT}"
            if state.get("timed_events_enabled", True):
                system_prompt = f"{system_prompt}{self.TIMER_TOOL_PROMPT}"
            if story_context:
                system_prompt = f"{system_prompt}{self.STORY_OUTLINE_TOOL_PROMPT}"
            system_prompt = f"{system_prompt}{self.CALENDAR_TOOL_PROMPT}"
            system_prompt = f"{system_prompt}{self.ROSTER_PROMPT}"
            system_prompt = f"{system_prompt}{self.READY_TO_WRITE_TOOL_PROMPT}"
        else:
            system_prompt = self.SYSTEM_PROMPT
            if guardrails_enabled:
                system_prompt = f"{system_prompt}{self.GUARDRAILS_SYSTEM_PROMPT}"
            if on_rails:
                system_prompt = f"{system_prompt}{self.ON_RAILS_SYSTEM_PROMPT}"
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
        location = player_state.get("location")
        exits = player_state.get("exits")
        player_loc = self._map_location_components(location, room_title, room_summary)
        if not player_loc["display"]:
            return "No map data yet. Try `look` first."

        marker_data = self._assign_player_markers(others, resolved_actor_id)
        other_entries = []
        for entry in marker_data:
            other = entry["player"]
            other_state = self.get_player_state(other)
            other_loc = self._map_location_components(
                other_state.get("location"),
                other_state.get("room_title"),
                other_state.get("room_summary"),
            )
            if not other_loc["display"]:
                continue
            other_name = other_state.get("character_name") or f"Adventurer-{str(other.actor_id)[-4:]}"
            other_entries.append(
                {
                    "marker": entry["marker"],
                    "user_id": other.actor_id,
                    "character_name": other_name,
                    "room": other_loc["display"],
                    "location_key": other_loc["key"],
                    "location_display": other_loc["display"],
                    "location_hint": other_loc["hint"],
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
        char_entries: list[dict[str, str]] = []
        if isinstance(characters, dict):
            for slug, info in list(characters.items())[:20]:
                if not isinstance(info, dict):
                    continue
                if info.get("deceased_reason"):
                    continue
                char_name = info.get("name", slug)
                char_loc = self._map_location_components(
                    info.get("location"),
                    "",
                    "",
                )
                char_entries.append(
                    {
                        "name": str(char_name),
                        "location_key": char_loc["key"] or "unknown-location",
                        "location_display": char_loc["display"] or "Unknown",
                    }
                )
        chars_text = self._dump_json(char_entries) if char_entries else "[]"

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
            f"PLAYER_LOCATION_KEY: {player_loc['key']}\n"
            f"PLAYER_LOCATION_DISPLAY: {player_loc['display']}\n"
            f"PLAYER_ROOM_TITLE: {room_title or 'Unknown'}\n"
            f"PLAYER_ROOM_SUMMARY: {room_summary or ''}\n"
            f"PLAYER_EXITS: {exits or []}\n"
            f"WORLD_SUMMARY: {self._compose_world_summary(campaign, campaign_state, max_chars=6000)}\n"
            f"WORLD_STATE: {self._dump_json(model_state)}\n"
            f"LANDMARKS: {landmarks_text}\n"
            f"WORLD_CHARACTER_LOCATIONS: {chars_text}\n"
        )
        if story_progress:
            map_prompt += f"STORY_PROGRESS: {story_progress}\n"
        map_prompt += (
            f"OTHER_PLAYERS: {self._dump_json(other_entries)}\n"
            "MAP_SPATIAL_RULES:\n"
            "- location_key is authoritative for grouping entities.\n"
            "- Same location_key means same room/area.\n"
            "- Different location_key means separate rooms/areas; never nest them.\n"
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
    # Memory tool compatibility
    # ------------------------------------------------------------------

    def list_memory_terms(
        self,
        campaign_id: str,
        wildcard: str = "%",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if self._memory_port is None:
            return []
        try:
            return self._memory_port.list_terms(campaign_id, wildcard=wildcard, limit=limit)
        except Exception:
            return []

    def record_memory_search_usage(
        self,
        campaign_id: str,
        queries: list[str],
    ) -> list[dict[str, object]]:
        cleaned_queries = [str(query or "").strip() for query in (queries or []) if str(query or "").strip()]
        if not cleaned_queries:
            return []
        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return []
            hints = self._record_memory_search_usage_for_campaign(campaign, cleaned_queries)
            campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
        return hints

    def store_memory(
        self,
        campaign_id: str,
        *,
        category: str,
        memory: str,
        term: str | None = None,
    ) -> tuple[bool, str]:
        if self._memory_port is None:
            return False, "memory_port_unavailable"
        try:
            return self._memory_port.store_memory(
                campaign_id,
                category=category,
                memory=memory,
                term=term,
            )
        except Exception:
            return False, "error"

    def search_curated_memories(
        self,
        query: str,
        campaign_id: str,
        *,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        if self._memory_port is None:
            return []
        try:
            return self._memory_port.search_curated(
                query=query,
                campaign_id=campaign_id,
                category=category,
                top_k=top_k,
            )
        except Exception:
            return []

    def list_source_material_documents(
        self,
        campaign_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        return SourceMaterialMemory.list_source_material_documents(
            str(campaign_id),
            limit=max(1, int(limit)),
        )

    def list_campaign_rules(self, campaign_id: str) -> list[dict[str, str]]:
        return SourceMaterialMemory.list_rulebook_entries(
            str(campaign_id),
            SourceMaterialMemory._normalize_source_document_key(
                self.AUTO_RULEBOOK_DOCUMENT_LABEL
            ),
        )

    def get_campaign_rule(
        self,
        campaign_id: str,
        rule_key: str,
    ) -> dict[str, str] | None:
        return SourceMaterialMemory.get_rulebook_entry(
            str(campaign_id),
            SourceMaterialMemory._normalize_source_document_key(
                self.AUTO_RULEBOOK_DOCUMENT_LABEL
            ),
            rule_key,
        )

    def put_campaign_rule(
        self,
        campaign_id: str,
        *,
        rule_key: str,
        rule_text: str,
        upsert: bool = False,
    ) -> dict[str, object]:
        return SourceMaterialMemory.put_rulebook_entry(
            str(campaign_id),
            document_label=self.AUTO_RULEBOOK_DOCUMENT_LABEL,
            rule_key=rule_key,
            rule_text=rule_text,
            replace_existing=bool(upsert),
        )

    def browse_source_keys(
        self,
        campaign_id: str,
        *,
        document_key: str | None = None,
        wildcard: str = "%",
        limit: int = 60,
    ) -> list[str]:
        return SourceMaterialMemory.browse_source_keys(
            str(campaign_id),
            document_key=document_key,
            wildcard=str(wildcard or "%"),
            limit=max(1, int(limit)),
        )

    def ingest_source_material_text(
        self,
        campaign_id: str,
        *,
        document_label: str,
        text: str,
        source_format: str | None = None,
        replace_document: bool = True,
    ) -> tuple[int, str]:
        chunks, _, _, _, _ = self._chunk_text_by_tokens(str(text or ""))
        if not chunks:
            return 0, "source-material"
        try:
            normalized_format = self._normalize_source_material_format(source_format)
        except Exception:
            normalized_format = None
        if not normalized_format:
            try:
                normalized_format = self._source_material_format_heuristic(
                    str(chunks[0] or "")
                )
            except Exception:
                normalized_format = self.SOURCE_MATERIAL_FORMAT_GENERIC
        if normalized_format == self.SOURCE_MATERIAL_FORMAT_GENERIC:
            key = SourceMaterialMemory._normalize_source_document_key(
                str(document_label or "source-material")
            )
            return 0, key
        source_mode = self._source_material_storage_mode(normalized_format)
        duplicate_doc = SourceMaterialMemory.find_duplicate_source_material_document(
            str(campaign_id),
            chunks=chunks,
            source_mode=source_mode,
        )
        if duplicate_doc:
            existing_key = str(duplicate_doc.get("document_key") or "").strip() or "source-material"
            return 0, existing_key

        return SourceMaterialMemory.store_source_material_chunks(
            str(campaign_id),
            document_label=str(document_label or "source-material"),
            chunks=chunks,
            source_mode=source_mode,
            replace_document=bool(replace_document),
        )

    async def ingest_source_material_with_digest(
        self,
        campaign_id: str,
        *,
        document_label: str,
        text: str,
        source_format: str | None = None,
        replace_document: bool = True,
        ctx_message=None,
        channel=None,
    ) -> tuple[int, str, Dict[str, dict]]:
        """Ingest source material and extract writing style.

        For **story** format: skips raw paragraph chunk storage entirely.
        Instead, generates a narrative digest, extracts prose-craft writing
        fragments via LLM analysis, stores those fragments as searchable
        chunks, and extracts literary style profiles.  The caller should
        merge the returned profiles into ``campaign_state["literary_styles"]``.

        For **rulebook** format: stores chunks normally via
        :meth:`ingest_source_material_text` and explicitly removes any
        narrative digest, since rulebooks are retrieved via keys/facts
        rather than digest summaries.

        Returns ``(stored_count, document_key, literary_profiles)`` where
        ``literary_profiles`` is a dict suitable for merging into
        ``campaign_state["literary_styles"]`` (empty for non-story formats).
        """
        try:
            normalized_format = self._normalize_source_material_format(source_format)
        except Exception:
            normalized_format = None
        if not normalized_format:
            try:
                chunks_probe, _, _, _, _ = self._chunk_text_by_tokens(str(text or ""))
                normalized_format = self._source_material_format_heuristic(
                    str((chunks_probe[0] if chunks_probe else text[:4000]) or "")
                )
            except Exception:
                normalized_format = self.SOURCE_MATERIAL_FORMAT_GENERIC

        # ------------------------------------------------------------------
        # Story format: literary analysis pipeline (no raw chunk storage)
        # ------------------------------------------------------------------
        if normalized_format == self.SOURCE_MATERIAL_FORMAT_STORY:
            return await self._ingest_story_literary(
                campaign_id,
                document_label=document_label,
                text=text,
                replace_document=replace_document,
                ctx_message=ctx_message,
                channel=channel,
            )

        # ------------------------------------------------------------------
        # Rulebook / other: existing flow (raw chunks + digest)
        # ------------------------------------------------------------------
        stored_count, document_key = self.ingest_source_material_text(
            campaign_id,
            document_label=document_label,
            text=text,
            source_format=source_format,
            replace_document=replace_document,
        )
        if stored_count <= 0:
            return stored_count, document_key, {}

        if normalized_format == self.SOURCE_MATERIAL_FORMAT_RULEBOOK:
            SourceMaterialMemory.delete_source_material_digest(
                str(campaign_id),
                document_key,
            )

        return stored_count, document_key, {}

    async def _ingest_story_literary(
        self,
        campaign_id: str,
        *,
        document_label: str,
        text: str,
        replace_document: bool = True,
        ctx_message=None,
        channel=None,
    ) -> tuple[int, str, Dict[str, dict]]:
        """Story-format ingestion: digest + writing fragments + literary profiles.

        Instead of storing raw story paragraphs, this method:
        1. Generates a narrative digest (content/plot retrieval).
        2. Extracts prose-craft writing fragments via LLM and stores
           them as the searchable chunks (style retrieval).
        3. Extracts overall literary style profiles for prompt injection.
        """
        document_key = SourceMaterialMemory._normalize_source_document_key(
            str(document_label or "source-material")
        )

        # Step 1: Generate narrative digest.
        try:
            digest = await self._summarise_long_text(
                text,
                ctx_message=ctx_message,
                channel=channel,
                summary_instructions=(
                    "This is source material for a text-adventure campaign. "
                    "Produce a comprehensive narrative digest preserving all characters, "
                    "locations, plot arcs, factions, key events, world rules, and "
                    "relationships. Maintain chronological order where applicable. "
                    "Be detailed and concrete — this digest will be used to ground "
                    "the campaign world."
                ),
            )
            if digest and digest.strip():
                SourceMaterialMemory.store_source_material_digest(
                    str(campaign_id),
                    document_key,
                    digest.strip(),
                )
        except Exception:
            self._logger.exception(
                "Story digest generation failed for campaign %s key %s",
                campaign_id,
                document_key,
            )

        # Step 2: Clear stale chunks up front when replacing, then extract
        # writing fragments and store as new chunks.
        if replace_document:
            try:
                conn = SourceMaterialMemory._get_conn()
                conn.execute(
                    "DELETE FROM source_material_chunks WHERE campaign_id = ? AND document_key = ?",
                    (str(campaign_id), document_key),
                )
                conn.commit()
            except Exception:
                self._logger.exception(
                    "Failed to clear stale chunks for campaign %s key %s",
                    campaign_id,
                    document_key,
                )

        fragments: List[str] = []
        try:
            fragments = await self._extract_writing_fragments(
                text,
                str(document_label or "source-material"),
            )
        except Exception:
            self._logger.exception(
                "Writing fragment extraction failed for campaign %s key %s",
                campaign_id,
                document_key,
            )

        stored_count = 0
        if fragments:
            stored_count, document_key = SourceMaterialMemory.store_source_material_chunks(
                str(campaign_id),
                document_label=str(document_label or "source-material"),
                chunks=fragments,
                source_mode="generic",
                replace_document=False,  # Already cleared above.
            )

        # Step 3: Extract literary style profiles.
        literary_profiles: Dict[str, dict] = {}
        try:
            literary_profiles = await self._analyze_literary_style(
                text,
                document_key,
            )
        except Exception:
            self._logger.exception(
                "Literary style extraction failed for campaign %s key %s",
                campaign_id,
                document_key,
            )

        return stored_count, document_key, literary_profiles

    def search_source_material(
        self,
        query: str,
        campaign_id: str,
        *,
        document_key: str | None = None,
        top_k: int = 5,
        before_lines: int = 0,
        after_lines: int = 0,
    ) -> list[tuple[str, str, int, str, float]]:
        try:
            before_n = max(0, int(before_lines))
        except Exception:
            before_n = 0
        try:
            after_n = max(0, int(after_lines))
        except Exception:
            after_n = 0
        return SourceMaterialMemory.search_source_material(
            query,
            str(campaign_id),
            document_key=document_key,
            top_k=max(1, int(top_k)),
            before_lines=before_n,
            after_lines=after_n,
        )

    def list_sms_threads(
        self,
        campaign_id: str,
        wildcard: str = "*",
        limit: int = 20,
    ) -> list[dict[str, object]]:
        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return []
            state = self.get_campaign_state(campaign)
        return self._sms_list_threads(state, wildcard=wildcard, limit=limit)

    def read_sms_thread(
        self,
        campaign_id: str,
        thread: str,
        limit: int = 20,
        viewer_actor_id: str | None = None,
    ) -> tuple[str | None, str | None, list[dict[str, object]]]:
        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return None, None, []
            state = self.get_campaign_state(campaign)
            canonical, label, messages = self._sms_read_thread(
                state, thread=thread, limit=limit
            )
            if viewer_actor_id:
                player = (
                    session.query(Player)
                    .filter(Player.campaign_id == str(campaign_id))
                    .filter(Player.actor_id == str(viewer_actor_id))
                    .first()
                )
                player_state = self.get_player_state(player) if player is not None else {}
                thread_markers: dict[str, int] = {}
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    msg_thread = self._sms_normalize_thread_key(
                        msg.get("thread") or canonical or thread
                    )
                    if not msg_thread:
                        continue
                    seq = self._coerce_non_negative_int(msg.get("seq", 0), default=0)
                    turn_id = self._coerce_non_negative_int(
                        msg.get("turn_id", 0), default=0
                    )
                    marker = seq if seq > 0 else turn_id
                    if marker <= 0:
                        continue
                    prev = self._coerce_non_negative_int(
                        thread_markers.get(msg_thread, 0), default=0
                    )
                    if marker > prev:
                        thread_markers[msg_thread] = marker
                if thread_markers:
                    changed = self._sms_mark_threads_read(
                        state,
                        actor_id=viewer_actor_id,
                        player_state=player_state,
                        thread_markers=thread_markers,
                    )
                    if changed:
                        campaign.state_json = self._dump_json(state)
                        campaign.updated_at = datetime.now(timezone.utc).replace(
                            tzinfo=None
                        )
                        session.commit()
            return canonical, label, messages

    def write_sms_thread(
        self,
        campaign_id: str,
        *,
        thread: str,
        sender: str,
        recipient: str,
        message: str,
        turn_id: int = 0,
    ) -> tuple[bool, str]:
        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return False, "campaign_not_found"
            campaign_state = self.get_campaign_state(campaign)
            game_time = self._extract_game_time_snapshot(campaign_state)
            self._sms_write(
                campaign_state,
                thread=thread,
                sender=sender,
                recipient=recipient,
                message=message,
                game_time=game_time,
                turn_id=turn_id,
            )
            campaign.state_json = self._dump_json(campaign_state)
            campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
        return True, "stored"

    # ------------------------------------------------------------------
    # Memory visibility compatibility
    # ------------------------------------------------------------------

    def filter_memory_hits_by_visibility(self, campaign_id: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._engine.filter_memory_hits_by_visibility(campaign_id, hits)
