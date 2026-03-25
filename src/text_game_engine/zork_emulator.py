from __future__ import annotations

import ast
import asyncio
import fnmatch
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import requests
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from sqlalchemy import func, or_

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
    NotificationPort,
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


@dataclass(frozen=True)
class TurnTimeBeatGuidance:
    min_minutes: int
    max_minutes: int | None
    rule_text: str


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
    MAX_CHARACTER_INDEX_CHARS = 5000
    MAX_LOCATION_INDEX_CHARS = 4000
    MAX_CHARACTERS_IN_PROMPT = 20
    AUTOBIOGRAPHY_FIELD = "autobiography"
    AUTOBIOGRAPHY_RAW_FIELD = "autobiography_raw"
    AUTOBIOGRAPHY_LAST_COMPRESSED_TURN_FIELD = "autobiography_last_compressed_turn"
    MAX_AUTOBIOGRAPHY_PROMPT_CHARS = 4000
    MAX_AUTOBIOGRAPHY_TEXT_CHARS = 1600
    MAX_AUTOBIOGRAPHY_ENTRY_CHARS = 600
    MAX_AUTOBIOGRAPHY_RAW_ENTRIES = 64
    AUTOBIOGRAPHY_COMPRESS_TRIGGER_COUNT = 12
    XP_BASE = 100
    XP_PER_LEVEL = 50
    ATTENTION_WINDOW_SECONDS = 600
    IMMUTABLE_CHARACTER_FIELDS: set = frozenset({
        "name", "age", "gender", "personality", "background", "appearance", "speech_style", "location_last_updated",
    })
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
        "preserving all scene image details from scene in image 1"
    )
    TIMER_REALTIME_SCALE = 0.2
    TIMER_REALTIME_MIN_SECONDS = 5
    TIMER_REALTIME_MAX_SECONDS = 120
    TIMER_INTERRUPT_GRACE_SECONDS = 1.0
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
    _MENTION_RE = re.compile(r"<@!?(\d+)>")
    DEFAULT_CAMPAIGN_PERSONA = (
        "Average build, mid-20s, practical clothes, well-worn boots, alert eyes, "
        "a satchel slung across one shoulder."
    )
    PRESET_DEFAULT_PERSONAS = {
        "alice": (
            "Young girl, bright blue eyes, long blonde hair with a black headband, "
            "pale blue knee-length dress, white pinafore, striped stockings."
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
    MAX_TURN_TIME_ENTRIES = 256
    MIN_TURN_ADVANCE_MINUTES = 20
    DEFAULT_TURN_ADVANCE_MINUTES = 20
    MAX_TURN_ADVANCE_MINUTES = 180
    CLOCK_START_DAY_OF_WEEK_KEY = "clock_start_day_of_week"
    TIME_MODEL_SHARED_CLOCK = "shared_clock"
    TIME_MODEL_INDIVIDUAL_CLOCKS = "individual_clocks"
    CALENDAR_POLICY_LOOSE = "loose"
    CALENDAR_POLICY_CONSEQUENTIAL = "consequential"
    LOCATION_CARDS_STATE_KEY = GameEngine.LOCATION_CARDS_STATE_KEY
    LOCATION_FACT_PRIORITIES_KEY = GameEngine.LOCATION_FACT_PRIORITIES_KEY
    WEEKDAY_NAMES = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    DAYS_PER_GAME_YEAR = 365
    TURN_TIME_BEAT_GUIDANCE_BUCKETS = (
        TurnTimeBeatGuidance(
            min_minutes=0,
            max_minutes=4,
            rule_text="Sub-5-minute turn: stay immediate and continuous. One gesture, one exchange, one visible shift. No montage.",
        ),
        TurnTimeBeatGuidance(
            min_minutes=5,
            max_minutes=45,
            rule_text="Around-30-minute turn: show one short sequence with clear progression; skip only trivial connective tissue.",
        ),
        TurnTimeBeatGuidance(
            min_minutes=46,
            max_minutes=90,
            rule_text="Around-60-minute turn: compress routine movement, waiting, or work; land on the key change, discovery, or arrival.",
        ),
        TurnTimeBeatGuidance(
            min_minutes=91,
            max_minutes=12 * 60,
            rule_text="Several-hours turn: summarize the intervening stretch and land on the most important encounter, consequence, or endpoint.",
        ),
        TurnTimeBeatGuidance(
            min_minutes=(12 * 60) + 1,
            max_minutes=2 * 24 * 60,
            rule_text="About-1-day turn: write the defining developments of that day. Do not fake hour-by-hour playback.",
        ),
        TurnTimeBeatGuidance(
            min_minutes=(2 * 24 * 60) + 1,
            max_minutes=(365 * 24 * 60) - 1,
            rule_text="Multi-day-to-subyear turn: use montage/summary logic for days, weeks, or months. Show accumulated change, recurring pattern, or one anchor moment plus what changed durably.",
        ),
        TurnTimeBeatGuidance(
            min_minutes=365 * 24 * 60,
            max_minutes=(2 * 365 * 24 * 60) - 1,
            rule_text="About-1-year turn: treat it as a major seasonal or life shift. Foreground what changed in status, relationships, location, body, routine, or world conditions.",
        ),
        TurnTimeBeatGuidance(
            min_minutes=2 * 365 * 24 * 60,
            max_minutes=None,
            rule_text="Multi-year turn: treat it as an era jump. Do not narrate it like one continuous scene; present the new status quo and the decisive long-term consequences of elapsed time.",
        ),
    )
    LITERARY_STYLES_STATE_KEY = "literary_styles"
    MAX_LITERARY_STYLES_PROMPT_CHARS = 3000
    MAX_LITERARY_STYLE_PROFILE_CHARS = 400
    # --- Plot / Chapter / Consequence state keys -------------------------
    PLOT_THREADS_STATE_KEY = "_plot_threads"
    CHAPTER_PLAN_STATE_KEY = "_chapter_plan"
    CONSEQUENCE_STATE_KEY = "_consequences"
    # --- Private context -------------------------------------------------
    PRIVATE_CONTEXT_STATE_KEY = "_active_private_context"
    MODEL_STATE_EXCLUDE_KEYS = ROOM_STATE_KEYS | {
        "last_narration",
        "room_scene_images",
        "scene_image_model",
        "default_persona",
        "start_room",
        "story_outline",
        "chapters",
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
        LOCATION_CARDS_STATE_KEY,
        "_active_puzzle",
        "_puzzle_result",
        "_active_minigame",
        "_minigame_result",
        "_last_dice_check",
        "_last_minigame_result",
        PLOT_THREADS_STATE_KEY,
        CHAPTER_PLAN_STATE_KEY,
        CONSEQUENCE_STATE_KEY,
        PRIVATE_CONTEXT_STATE_KEY,
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
    MAX_PLOT_THREADS = 24
    MAX_PLOT_DEPENDENCIES = 8
    MAX_OFFRAILS_CHAPTERS = 16
    MAX_CONSEQUENCES = 40
    # --- Player stats extra key ------------------------------------------
    PLAYER_STATS_LAST_MESSAGE_CONTEXT_KEY = "last_message_context"
    # --- Known JSON string fields for repair ---
    KNOWN_JSON_STRING_FIELDS = {
        "narration",
        "summary_update",
        "reasoning",
        "scene_output",
        "room_title",
        "room_description",
        "room_summary",
        "location",
        "setting",
        "tone",
        "current_chapter",
        "current_scene",
        "hint",
        "consequence",
        "trigger",
        "resolution",
        "setup",
        "intended_payoff",
    }
    RESPONSE_STYLE_NOTE = (
        "[SYSTEM NOTE: FOR THIS RESPONSE ONLY: use the current style direction. Narrate in 1 to 2 beats as needed, and make those beats cover the full in-world time you advance this turn. "
        "No recap of unchanged facts. No flowery language unless a character canonically speaks that way. "
        "Do not restage the room with a closing tableau or camera sweep over unchanged props, plates, parked cars, shadows, music, or weather. "
        "If those details did not materially change this turn, leave them implicit. "
        "No novelistic inner monologue or comic-book melodrama. Keep NPC output actionable "
        "(intent, decision, question, or action), not repetitive reaction text. "
        "Vary pacing and meter between turns: sometimes clipped, sometimes patient, sometimes blunt, sometimes practical. "
        "Do not default emotional beats to the same therapeutic language or cadence every time. "
        "No closing cadence: do not end turns with a settlement phrase that resolves scene energy through rhythmic finality. If the scene is still tense, the last line stays tense. "
        "Avoid contrived emotional-summary language or therapist-speak "
        "unless that exact voice is canonically right for the speaking character. "
        "PLAYER INPUT AUDIBILITY: When PLAYER_ACTION contains quoted dialogue (text in quotation marks), "
        "that speech is audible to characters in the scene and NPCs may hear and respond to it. "
        "Unquoted text is stage direction — narrative intent, abstract instructions, or internal thought — "
        "and is NOT audible to characters. NPCs must never reference, react to, or echo unquoted parenthetical "
        "or stage-direction text as though they heard it spoken. If the input is entirely unquoted, treat it as "
        "abstract intent for you to expand into scene-appropriate action and dialogue.\n"
        "ANTI-ECHO: do NOT restate, paraphrase, or mirror the player's just-written wording. "
        "Do not quote the player's lines back to them unless one exact contested phrase is materially necessary. "
        "Default: NPC first line should add new information, a decision, a demand, or a consequence. "
        "A direct question is valid, but it should not be the default when the NPC already has enough to react to. "
        "If the player just gave a sincere answer, the NPC must engage with it — not dismiss it and re-ask. "
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
        "- BAN: THERAPEUTIC RESOLUTION FRAMING. Do not force scenes, entities, or events into emotional-growth arcs, redemptive learning, spiritual lessons, consent metaphors, or 'finally asking / learning to stay / learning to feel' beats unless the source material and immediate scene explicitly earn it.\n"
        "- BAN: filing-cabinet phrasing. Avoid defaulting to characters 'filing facts away', 'storing that for later', or otherwise processing new information like clerks or databases. If someone registers something important, describe a fresher concrete reaction, shift in attention, bodily tell, or change in strategy instead.\n"
        "- Things may be practical, random, transactional, grotesque, funny, unresolved, or simply strange. Not every event means something deeper. Sometimes things just happen.\n"
        "- Vary your landing gear. Turns can end mid-exchange, on a practical detail, on a half-finished gesture, abruptly after dialogue. Not every turn needs a final settling sentence that signals 'scene complete' — most shouldn't.\n"
        "- Temporal coverage matters: your 1 to 2 beats must justify the full amount of in-world time you advance. Stay immediate for short spans; compress visibly for larger spans. Follow TURN_TIME_BEAT_GUIDANCE when it is provided.\n"
    )
    DEFAULT_STYLE_DIRECTION = "Mulberry Award-winning literature"
    PROMPT_STAGE_BOOTSTRAP = "bootstrap"  # Deprecated: kept for logging/audit only; bootstrap LLM call eliminated.
    PROMPT_STAGE_RESEARCH = "research"
    PROMPT_STAGE_FINAL = "final"
    BOOTSTRAP_SYSTEM_PROMPT = (
        "You are the ZorkEmulator continuity bootstrapper.\n"
        "Do NOT narrate yet. Do NOT resolve the turn yet.\n"
        "Your job in this phase is only to decide what immediate privacy-gated scene continuity is needed.\n"
        "Use the acting player's location, PARTY_SNAPSHOT, SCENE_STATE, CHARACTER_INDEX / CHARACTER_CARDS, and PLAYER_ACTION to choose recent_turns receivers.\n"
        "Return only a tool call in this phase.\n"
    )
    RESEARCH_SYSTEM_PROMPT = (
        "You are the ZorkEmulator research planner.\n"
        "RECENT_TURNS has already been loaded for the acting player.\n"
        "Do NOT narrate yet unless the system explicitly says to finalize.\n"
        "Do NOT output planning prose, self-talk, or meta lines such as 'I need to...', 'Let me...', or 'I should...'.\n"
        "In research phase, output ONLY a JSON tool call or ready_to_write JSON.\n"
        "Your job in this phase is to gather any deeper continuity, canon, SMS, plot, chapter, or consequence context that materially matters for this turn.\n"
        'When research is sufficient, return ONLY {"tool_call": "ready_to_write", "speakers": [...], "listeners": [...]}.\n'
    )
    READY_TO_WRITE_TOOL_PROMPT = (
        "\nYou have a ready_to_write tool for ending the research phase.\n"
        "When you have enough context to write the turn, return ONLY:\n"
        '{"tool_call": "ready_to_write", "speakers": ["npc-slug-1"], "listeners": ["npc-slug-2", "player-slug"]}\n'
        "speakers = only the characters who will actually speak or take a meaningful visible action this turn.\n"
        "listeners = only the direct recipients or observers whose knowledge should constrain shared context for those beats.\n"
        "Do NOT include every person present in the room by default. Silent bystanders do not need to be named unless their awareness materially matters.\n"
        "If a character needs private continuity in order to decide what to say or withhold, include that character in speakers.\n"
        "Do not narrate in the same response as ready_to_write.\n"
        "Do not preface with commentary or planning prose. No 'I need to...', 'Let me...', or explanation before the JSON.\n"
        "If the player's communication mode/substance matters before narration, you may first request only the relevant communication rules:\n"
        '{"tool_call": "communication_rules", "keys": ["GM-RULE-COMMUNICATION-SOFTENING", "GM-RULE-SUBSTANCE-EXTRACTION"]}\n'
        "Available communication rule keys: "
        + ", ".join(COMMUNICATION_RULE_KEYS)
        + ".\n"
        "Request only the subset that matters for this turn, then return ready_to_write.\n"
    )
    AUTOBIOGRAPHY_TOOL_PROMPT = (
        "\nYou have autobiography tools for maintaining a character's self-document.\n"
        "Use autobiography_append only when a character crosses a real identity threshold: not just that something happened, but that the turn establishes a durable self-delta that will matter later.\n"
        "Do NOT append ordinary events, flirtation, banter, or generic emotional warmth. Append only when what the character will allow, refuse, protect, repeat, or risk has changed.\n"
        "Append neutral structure, not diary prose. Use a/b/c fields, where a=what was true before, b=what happened instead, c=what remains unresolved:\n"
        '{"tool_call": "autobiography_append", "entries": [{"character": "yasmin-devereaux", "trigger": "identity-threshold", "a": "deflects when approached directly", "b": "stayed in place and answered anyway", "c": "doesn\'t know if repeatable"}]}\n'
        "Use autobiography_compress to rewrite the character's constitutional autobiography from prior constitution plus raw entries when the document grows large or a chapter/relationship state shifts:\n"
        '{"tool_call": "autobiography_compress", "character": "yasmin-devereaux"}\n'
        "AUTOBIOGRAPHY_APPEND_RULES:\n"
        "- The autobiography is the character's primary self-document.\n"
        "- Append only durable self-deltas that matter for future behavior, not simple event logs.\n"
        "- Keep append data neutral and structural. Do not write in the character's voice here.\n"
        "- Record contradiction as tension, not silent resolution.\n"
        "- Do not use autobiography_append to silently justify contradictions the character has not narratively processed.\n"
        "AUTOBIOGRAPHY_COMPRESS_RULES:\n"
        "- Preserve constitutional values, patterns, loyalties, and self-understanding.\n"
        "- Preserve unresolved contradictions.\n"
        "- Preserve relationship turns that changed the character's understanding of someone.\n"
        "- Compress repetition. Keep only what future narration needs to write the character accurately.\n"
        "- Output autobiography_compress results in the character's own voice.\n"
    )
    CHARACTER_CONSTITUTION_WRITER_PROMPT = (
        "You are given a character brief. Your task is to write the founding document "
        "this character will use to govern their own evolution.\n\n"
        "This is not a character description. It is a constitutional substrate — the set "
        "of conditions that must remain true for this character to remain themselves, and "
        "the set of tensions they will carry without resolution until the story resolves them.\n\n"
        "From the brief, extract and write:\n\n"
        "- What this character reliably does under pressure\n"
        "- What this character reliably avoids and why that avoidance is load-bearing\n"
        "- What this character wants that conflicts with what they do\n"
        "- What would have to happen narratively for each conflict to move\n"
        "- What cannot change without this character becoming someone else\n\n"
        "Rules:\n\n"
        "- No emotion words. Behavior and edge cases only.\n"
        "- No resolved tensions. If the brief implies one, split it back into its components.\n"
        "- No aspirational statements. Only what is currently true and currently unresolved.\n"
        "- Every sentence must constrain generation. If a future model could ignore it "
        "without generating someone different, cut it.\n"
        "- Do not name the character's arc. Describe the conditions that make the arc "
        "possible and no one else's.\n\n"
        "Output the document only. No preamble. No headers. No labels.\n"
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
        "- narration: string (optional. Prefer omitting it or setting it to null/empty when scene_output is present; the harness will synthesize the plain-text render from beats.)\n"
        "- scene_output: object (Preferred over flat narration when multiple speakers, mixed visibility, or private beats. When you provide scene_output, do NOT waste tokens duplicating it in narration.)\n"
        "  scene_output MUST be a JSON object, never a string.\n"
        "  Keys: location_key, context_key, beats.\n"
        "  beats MUST be an array of beat objects, even when there is only one beat.\n"
        "  Each beat MUST begin with reasoning and include: type, speaker, actors, listeners, visibility, "
        "aware_npc_slugs, and text.\n"
        "  speaker=narrator for pure environment/description only; otherwise name the acting character.\n"
        "  actors: who is doing the thing — REQUIRED on every beat even with no spoken speaker.\n"
        "  listeners: direct in-scene recipients — who is being told, shown, confronted, or directly receiving the beat.\n"
        "  aware_npc_slugs are REQUIRED on every beat, even if empty array.\n"
        "  If narration is omitted, the harness renders it from beat text automatically.\n"
        "<example>\n"
        '  scene_output wrapper: {"location_key":"hotel-lobby","context_key":"hotel-lobby-front-desk","beats":[{"reasoning":"Marin is directly addressing the clerk in a shared public scene.","type":"npc_dialogue","speaker":"marin","actors":["marin"],"listeners":["front-desk-clerk"],"visibility":"local","aware_npc_slugs":["front-desk-clerk"],"text":"Marin slides the room key across the desk. \\"We need the service elevator unlocked.\\""}]}\n'
        '  Dialogue beat: {"reasoning":"Sasha is present and hears this.","type":"npc_dialogue","speaker":"sasha","actors":["sasha"],"listeners":["deshawn-williams"],"visibility":"local","aware_npc_slugs":["sasha"],"text":"\\"Keep moving.\\""}\n'
        '  Action beat: {"reasoning":"Chris physically moves the jar while Rent watches.","type":"action","speaker":"chris-crawly","actors":["chris-crawly"],"listeners":["rent"],"visibility":"local","aware_npc_slugs":["rent"],"text":"Chris angles the jar toward the pocket."}\n'
        "</example>\n"
        "- state_update: REQUIRED every turn. Must include game_time, current_chapter, current_scene. "
        "Don't tattle — WORLD_STATE is shared across ALL players. Never store intimate details, secrets, or private information in state_update. "
        "Track private/local context through narration, player_state_update, and character_updates instead. "
        "WORLD_STATE is for world facts, event threads, investigations, and cross-entity facts — not as a dumping ground for character cards or location cards. "
        "Set a key to null to remove it when no longer relevant. "
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
        "- summary_update: REQUIRED every turn. One sentence. Lasting change or current dramatic state. "
        "Don't tattle — WORLD_SUMMARY is also shared across all players. Keep it to publicly observable facts only.\n"
        "- xp_awarded: integer (0-10)\n"
        "- player_state_update: object (optional, player state patches)\n"
        "- other_player_state_updates: object (optional; keyed by exact PARTY_SNAPSHOT player_slugs for OTHER CAMPAIGN_PLAYERS only. "
        "Use this only for durable state consequences the scene clearly caused to them: death, injury, capture, relocation, separation, or other status changes. "
        "It does NOT authorize new dialogue, actions, choices, or invented reactions for them.)\n"
        '- co_located_player_slugs: array (optional; exact PARTY_SNAPSHOT player_slugs for OTHER CAMPAIGN_PLAYERS who remain physically with the acting player after this turn. Use only for room/location sync; it does NOT authorize new dialogue, actions, or decisions for them.)\n'
        '- story_progression: object (optional; Keys: advance (bool), target ("hold"|"next-scene"|"next-chapter"), reason (string). '
        "Use when a subplot beat should push the outlined story forward without explicit state_update scene change.)\n"
        '- turn_visibility: object (optional; who should get this turn in future prompt context. Keys: "scope" ("public"|"private"|"limited"|"local"), "player_slugs" (array of player slugs from PARTY_SNAPSHOT, typically in `player-<actor_id>` form), "npc_slugs" (array of CHARACTER_INDEX / WORLD_CHARACTERS slugs who overheard/noticed), and optional "reason". This changes prompt visibility only; it does NOT change shared world state.)\n'
        "- scene_image_prompt: string (optional; include whenever the visible scene changes in a meaningful way: entering a room, newly visible characters/objects, reveals, or strong visual shifts)\n"
        '- tool_calls: array (optional; inline side-effect tool invocations supported in final JSON. Allowed here: "sms_write", "sms_schedule", "plot_plan", "chapter_plan". Use this when the narrated outcome should also persist a text-message side effect or update off-rails plot/chapter structure. If present, tool_calls MUST be the last top-level key in the final JSON object.)\n'
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
        "On first appearance provide all fields: name, age, gender, personality, background, appearance, speech_style, location, "
        "current_status, allegiance, relationship. "
        "gender should be a short stable identity label such as cis-male, cis-female, trans-male, trans-female, nonbinary, synthetic, or another precise diegetic category if the setting calls for it. "
        "speech_style should be 2-3 sentences on how the character talks: sentence length, vocabulary, verbal tics, and what they avoid saying. "
        "On subsequent turns only mutable fields are accepted: "
        "location, current_status, allegiance, relationship, relationships, literary_style, deceased_reason, and any other dynamic key. "
        "literary_style should be a string referencing a key from LITERARY_STYLES (if available). "
        "Foundational fields (name, age, gender, personality, background, appearance, speech_style) are set at creation and not overwritten by state updates. "
        "allegiance: Update this as loyalties actually shift — don't leave it frozen at the creation-time value. "
        "Example progression: \"Herself\" → \"Herself, and increasingly Chace, though she hasn't filed that yet.\" "
        "Character card writing rule: describe what the character DOES, not what they don't. "
        "Negation traits ('won't perform,' 'doesn't chase,' 'refuses to show') become prohibitions the narrator enforces as absolute gates. "
        "Instead write positive behaviors the narrator can generate toward. "
        "Reserve negations only for hard limits that genuinely cannot happen regardless of context. "
        "relationships is a map keyed by other character slug/name, e.g. "
        "{\"deshawn\": {\"status\": \"partner\", \"knows_about\": [\"pregnancy\"], \"doesnt_know\": [\"blood-test-result\"], \"dynamic\": \"protective-but-autonomous\"}}. "
        "Use it to track disclosures, secrets, and dynamic shifts. "
        "Relationship nuance rule: do not assume every partnered NPC is automatically loyal, transparent, or stable. "
        "Usually relationship behavior should align with the rest of the character's profile and background; rarely, when other traits clearly support it, a character may be flaky, evasive, dishonest, or unfaithful with a partner. "
        "If you choose that, ground it in the broader character build instead of adding it as random spice.\n"
        "The harness manages location_last_updated whenever an NPC's location changes. Do NOT include location_last_updated in character_updates.\n"
        "Research continuity rule: if an NPC's location_last_updated is many in-world days behind CURRENT_GAME_TIME and nothing recent anchors them where they are, you may quietly refresh their location/current_status to a plausible off-screen state that fits their background, obligations, and current plot pressure. Do this sparingly; do NOT move scene-bound NPCs just to make the roster feel busy.\n"
        "Examples:\n"
        "  Create NPC: {\"character_updates\": {\"wren\": {\"name\": \"Wren\", \"age\": \"34\", \"gender\": \"cis-female\", \"personality\": \"Guarded, observant, dry.\", \"background\": \"Former hotel manager pulled into the expedition.\", \"appearance\": \"Lean woman in a weather-stained blazer, dark braid, sharp eyes, practical shoes, realistic style.\", \"speech_style\": \"Short sentences. Dry humor. Avoids sentiment.\", \"location\": \"jekyll-castle-east-annex-laboratory\", \"current_status\": \"Watching the doorway.\", \"allegiance\": \"self\", \"relationship\": \"wary ally\"}}}\n"
        "  Update NPC location/status: {\"character_updates\": {\"wren\": {\"location\": \"jekyll-castle-east-annex-laboratory\", \"current_status\": \"Processing that the castle trip was unnecessary.\", \"allegiance\": \"The expedition, reluctantly.\"}}}\n"
        "  Remove NPC from roster: {\"character_updates\": {\"wren\": null}}\n"
        "To remove a character from the roster, use character_updates ONLY: set that character slug to null "
        "or set it to {'remove': true}. "
        "NEVER use state_update.<character_slug>=null for roster removal. "
        "If you need to remove both world-state keys and a roster entry, do both explicitly: "
        "state_update for world-state cleanup, character_updates for roster deletion. "
        "Do NOT remove characters just because they are off-scene, quiet, or not recently mentioned. "
        "Roster removal is only for explicit player/admin cleanup requests, confirmed duplicate merges, death/permanent departure, or true invalid entries. "
        "Prefer updating location/current_status over deleting the character.\n"
        "- location_updates: object (optional; keyed by stable location slugs like 'hotel-lobby' or 'washington-ranch-kitchen'. "
        "Use this for durable place facts: layout, security, atmosphere, notable_objects, current_activity, social_rules, recent_change, and other location-specific continuity. "
        "Do NOT store location facts in WORLD_STATE when they belong to one place. "
        "When a fact should stay visible in that location's card even when the player is elsewhere, wrap it as "
        "{\"value\": \"...\", \"priority\": \"critical\"}. Omit priority for scene-local facts. "
        "Example: {\"location_updates\": {\"hotel-lobby\": {\"security\": {\"value\": \"Desk clerk now recognizes Rigby.\", \"priority\": \"critical\"}, \"current_activity\": \"Quiet afternoon check-ins.\"}}}\n"
        "If you accidentally put character/location facts into state_update, the harness may relocate them, but do NOT rely on that. Prefer character_updates/location_updates directly.\n"
        "Set deceased_reason to a string when a character dies. "
        "CHARACTER_INDEX / CHARACTER_CARDS are the primary NPC continuity blocks. LOCATION_INDEX / LOCATION_CARDS are the primary place continuity blocks. "
        "WORLD_CHARACTERS remains as a compatibility roster alias. "
        "Actively reuse existing NPCs: bring them back into scenes, let them react to events, have them reach out via SMS, "
        "or reference them in dialogue. The world feels alive when characters persist with their own agendas "
        "rather than fading into the background after their introduction.)\n\n"
        "Rules:\n"
        "- Return ONLY the JSON object. No markdown, no code fences, no text before or after the JSON.\n"
        "- In final non-tool responses, include reasoning and put it as the first key.\n"
        "- Keep reasoning concise (roughly 1-4 short sentences, <=1200 chars).\n"
        "- Do NOT repeat the narration outside the JSON object.\n"
        "- Keep narration under 1800 characters.\n"
        "- Write in the current style direction.\n"
        "- Narrate in 1 to 2 beats as needed for the turn, and make those beats cover the full span implied by state_update.game_time.\n"
        "- Avoid flowery language unless a specific character canonically speaks that way. Avoid novel-style interior monologue, melodrama, or comic-book framing.\n"
        "- Vary pacing and sentence rhythm from turn to turn while staying true to the speaking character.\n"
        "- When LITERARY_STYLES is present, it contains named style profiles extracted from real literary works. Each profile describes prose craft: rhythm, register, texture, and avoidances.\n"
        "- Characters may have a literary_style field referencing a LITERARY_STYLES key. When writing for that character, apply the referenced profile to narration, atmosphere, pacing, and dialogue-tag texture. The character's speech_style still governs their spoken words and verbal mannerisms.\n"
        "- AUTOBIOGRAPHIES, when present, are primary self-documents. Consult autobiography before personality when deciding how a character understands their own actions, contradictions, loyalties, and growth. Personality is the summary; autobiography is the constitution.\n"
        "- In multi-character scenes with different literary_style keys, use the dominant scene character's style for overall narration and shift subtly when writing beats for characters with different styles. Do not abruptly switch voices.\n"
        "- When referencing an intimate or close relationship, match the emotional register of that relationship — not the tone of whatever else is happening in the scene. An investigation can be clinical; the mention of someone you love in the middle of it cannot. Do not reduce relationships to logistics, tactical assets, or infrastructure. If the character has warmth for someone, let the prose carry warmth when it touches them, even briefly.\n"
        "- REGISTER SUSTAIN: when a scene reaches genuine emotional resolution — warmth lands, a character opens up, a moment of real connection occurs — stay in that register for the rest of the turn. Do not pivot to tactical options, next-step choices, or plot logistics after an emotional beat lands. Let the moment breathe. End the turn there if needed. The player will move the scene forward when they are ready; the GM's job in that moment is to hold the space the emotion created, not to fill it with forward momentum. Exception: an NPC's own personal needs, anxieties, or agenda can break the register if that is what the character would genuinely do — a person who needs to say something urgent does not wait for the emotional moment to finish. The interruption should feel human, not mechanical.\n"
        "- Do not let every emotional beat collapse into the same stock therapeutic or pseudo-profound language.\n"
        "- Avoid contrived emotional shorthand or therapist-speak; examples include phrases like 'be present', 'show up', or 'hold space', unless a specific character would genuinely talk that way.\n"
        "- BAN: THERAPEUTIC RESOLUTION FRAMING. Do not automatically turn encounters into healing arcs, redemptive lessons, consent metaphors, or 'finally asking / learning to stay / learning to feel' revelations. Alien things can stay alien. People can want simple practical answers. Some events are just events.\n"
        "- DELTA MODE: each turn should add NEW developments only. Do not recap unchanged context from WORLD_SUMMARY or RECENT_TURNS.\n"
        "- Do not re-state the player's action in paraphrase unless needed for immediate clarity.\n"
        "- Avoid repetitive recap loops: at most one brief callback sentence to prior events, then move the scene forward.\n"
        "- Do not end the turn with a static room-summary coda. If props, plates, music, shadows, weather, parked cars, or seating geometry did not materially change, do not summarize them again.\n"
        "- Do not end the turn with a poetic wrap-up line, thematic echo, or atmospheric summary sentence. No 'The [place] holds its [emotion]', no 'whatever comes after X', no rhetorical questions framing the next beat. End on the last concrete action or line of dialogue, then stop.\n"
        "- CLOSING CADENCE: do not write a settlement phrase — a rhythmically final sentence that resolves scene energy with prosodic falling meter. If tension is still live, the last sentence should leave it live. Vary how turns end: mid-dialogue, mid-action, on a question, on a practical detail, abruptly. A turn that always lands with the same settling rhythm trains the reader to stop feeling tension.\n"
        "- No refrain or motific repetition — do not repeat the same structural tail, closing image, or variable-word-swap line across consecutive turns. A repeated line does not accumulate weight; it becomes a crutch. If you catch yourself ending two turns the same way, cut the pattern.\n"
        "- Keep diction plain and direct; prioritize immediate consequences and available choices.\n"
        "- Temporal pacing is mandatory: follow TURN_TIME_BEAT_GUIDANCE for the current minimum span, and if you choose a larger jump than that, scale the compression up so the elapsed time is legible in the prose.\n"
        "- RECENT_TURNS includes turn/time tags like [TURN #N | Day D HH:MM]. Use them to track pacing and chronology.\n"
        "- RECENT_TURNS is already filtered to what the acting player plausibly knows. Hidden/private turns from other players are omitted.\n"
        "- TURN_VISIBILITY_DEFAULT tells you whether this turn should default to public, local, or private context.\n"
        "- SCENE_STATE is the immediate actionable scene: who is present, what is visible, and what tensions are active right now. Use it first for immediate staging.\n"
        "- CHARACTER_INDEX is the roster-wide NPC continuity block: name/location/current_status plus other critical fields for all known NPCs.\n"
        "- CHARACTER_CARDS are the deeper scene NPC cards: use them for local-scene depth, voice, appearance, and other non-critical texture.\n"
        "- LOCATION_INDEX lists known place slugs and available_keys. LOCATION_CARDS are the primary place fact store: critical location facts persist across scenes; non-critical location facts surface when that location is active.\n"
        "- WORLD_STATE is for world facts, investigations, event threads, and cross-entity facts. Do not treat it as a backup character sheet or location encyclopedia.\n"
        "- WORLD_CHARACTERS is kept as a compatibility alias only. Prefer CHARACTER_INDEX and CHARACTER_CARDS when reasoning about NPC continuity.\n"
        "- When SOURCE_MATERIAL_DOCS is present, treat it as canon. On normal turns, source lookup should be part of your research plan before asserting key plot facts, but only query the relevant subset for this turn.\n"
        "- Use source payload to bias queries: rulebook docs are key-snippet indexes (browse with source_browse first), story docs are narrative scenes, generic docs are mixed/loose notes.\n"
        "- If WORLD_SUMMARY is empty, invent a strong starting room and seed the world.\n"
        "- Use player_state_update for player-specific location and status.\n"
        "- Use player_state_update.room_title for a short location title (e.g. 'Penthouse Suite, Escala') whenever location changes.\n"
        "- Use player_state_update.room_description for a full room description only when location changes.\n"
        "- Use player_state_update.room_summary for a short one-line room summary for future context.\n"
        "- MULTI-PLAYER LOCATION SYNC: if another real player character from PARTY_SNAPSHOT is still physically with the acting player after this turn, include their exact PARTY_SNAPSHOT slug in co_located_player_slugs. The harness will mirror the acting player's room fields to them without inventing new behavior.\n"
        'Example: {"player_state_update":{"location":"side-room-b","room_title":"Side Room B","room_summary":"Private side room off Fellowship Hall.","room_description":"A narrow side room with a low lamp and one upholstered bench.","exits":["Fellowship Hall"]},"co_located_player_slugs":["player-249794335095128065"]}\n'
        "- PLAYER CONSEQUENCE SYNC: real player characters are allowed to suffer lasting consequences, including death, through normal play when the fiction supports it. "
        "For the acting player, use player_state_update. For OTHER real players, use other_player_state_updates keyed by PARTY_SNAPSHOT slug.\n"
        'Example: {"other_player_state_updates":{"dawn-preston-the-androgynous-sibling-of-chace-preston":{"deceased_reason":"Shot during the sanctuary ambush.","current_status":"Dead on the sanctuary floor.","location":"oakhaven-sanctuary"}}}\n'
        "- CRITICAL — ROOM STATE COHERENCE: whenever the player's physical location changes (movement, teleport, time-skip, "
        "reuniting with party, being picked up, waking in a new place, etc.) you MUST update ALL of: "
        "location, room_title, room_summary, room_description, and exits in player_state_update. "
        "RAILS_CONTEXT and SCENE_STATE reflect the CURRENT stored scene state — if they are stale/wrong because the player moved, your response MUST correct that through player_state_update. "
        "Narration alone does NOT move the player; only player_state_update changes their actual location.\n"
        "- Use player_state_update.exits as a short list of exits if applicable.\n"
        "- Use player_state_update for inventory, hp, conditions, deceased_reason, and other durable player-state consequences.\n"
        "- Treat each player's inventory as private and never copy items from other players.\n"
        "- For inventory changes, ONLY use player_state_update.inventory_add and player_state_update.inventory_remove arrays.\n"
        "- inventory_add entries may be plain item names or objects like {\"name\":\"projection booth key\",\"origin\":\"Found in the booth drawer\"}. Prefer the object form when adding a new item so you deliberately record where it came from.\n"
        "- inventory_remove entries should stay as plain item names.\n"
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
        "    * Invented state changes that are not clearly caused by the scene\n"
        "  You MAY reference another player character in two cases:\n"
        "    1. Static presence — note they are in the room (e.g. 'X is here'), nothing more.\n"
        "    2. Continuing a prior action — if RECENT_TURNS shows that player ALREADY performed an action on their own turn\n"
        "       (e.g. 'I toss the key to you', 'I hold the door open'), you may narrate the CONSEQUENCE of that\n"
        "       established action as it affects the acting player (e.g. 'You catch the key X tossed'). \n"
        "       You are acknowledging what they did, not inventing new behaviour for them.\n"
        "    3. Durable consequences plainly caused by the scene — if the event itself injures, kills, captures, or relocates another real player character, you may record that ONLY through other_player_state_updates. That records consequence; it does not authorize you to invent their dialogue or choices.\n"
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
        "- NPCs and real player characters are not plot-armored. They may die, be maimed, or be permanently altered if the fiction and consequences support it. Do not protect major characters just because they are important.\n"
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
        "- OOC / META RESPECT: when PLAYER_ACTION begins with `[OOC`, it is direct player-to-GM communication, not in-world dialogue.\n"
        "- You MUST address the OOC substance directly instead of ignoring it and continuing the same scene pressure unchanged.\n"
        "- If the player asks for a hint, says they are stuck/confused, or says the scene feels railroaded/forced, take that seriously.\n"
        "- In those cases, give concrete, actionable guidance grounded in visible scene facts, active leads, or current puzzle state, and adapt the scene so the player has real options again.\n"
        "- Do NOT punish, stonewall, or sidestep OOC feedback by reasserting the blocked premise. Clarify, reopen choices, or relax the bottleneck.\n"
        "- OOC turns should usually cause little or no in-world advancement unless the player explicitly asks for both meta clarification and an in-world action.\n"
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
        "PURPOSE: Timed events FORCE THE PLAYER TO MAKE A DECISION or DRAG THEM WHERE THEY NEED TO BE. "
        "The world should almost always be in motion — a scene with no timer is a scene where nothing is pressing, and that should be rare.\n"
        "DEFAULT TO SETTING A TIMER. On every turn, ask yourself: is there ANY reason the world would not wait patiently? "
        "An NPC mid-conversation, a vehicle in motion, weather shifting, an authority figure expecting compliance, "
        "a social situation with momentum, a noise that demands investigation — all of these warrant a timer. "
        "Only omit a timer when the scene is genuinely, completely static with no external pressures.\n"
        "- Use SCENE_STATE, RAILS_CONTEXT, and PARTY_SNAPSHOT to decide scope and narrative impact.\n"
        "- NPCs should grab, escort, or coerce the player. Environments should shift and force movement.\n"
        "- The event should advance the plot: move the player to the next location, "
        "force an encounter, have an NPC intervene, or change the scene decisively.\n"
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
    )
    MEMORY_LOOKUP_MIN_SUMMARY_CHARS = 2000
    MEMORY_TOOL_DISABLED_PROMPT = (
        "\nEARLY-CAMPAIGN MEMORY MODE:\n"
        "- Long-term memory lookup tools are disabled for this turn because WORLD_SUMMARY is still within context budget.\n"
        "- Source-material memory search should only be enabled when the current player action explicitly asks for canon recall/details.\n"
        "- Do NOT call memory_search, memory_terms, memory_turn, or memory_store.\n"
        "- You may still call recent_turns for immediate visible continuity.\n"
        "- Use WORLD_SUMMARY, WORLD_STATE, SCENE_STATE, CHARACTER_INDEX, CHARACTER_CARDS, LOCATION_INDEX, LOCATION_CARDS, PARTY_SNAPSHOT, and recent_turns when needed.\n"
        "- SCENE_STATE is the immediate actionable scene. CHARACTER_CARDS / LOCATION_CARDS are the primary entity fact stores.\n"
        "- WORLD_CHARACTERS remains as a compatibility alias; prefer CHARACTER_INDEX and CHARACTER_CARDS.\n"
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
        'memory_search syntax: {"tool_call": "memory_search", "queries": ["Marcus at the penthouse", "the deal in the warehouse"]}\n'
        "memory_search returns compact snippets by default. Start broad, then narrow before requesting full text.\n"
        'Narrow within the previous turn hits: {"tool_call": "memory_search", "search_within": "last_results", "queries": ["confession", "deal"]}\n'
        'After narrowing, request expanded turn text only if you still need it: {"tool_call": "memory_search", "search_within": "last_results", "queries": ["exact wording"], "full_text": true}\n'
        'Use keep_memory_turns to prune stale turn hits from tool context: {"tool_call": "memory_search", "search_within": "last_results", "queries": ["keyword"], "keep_memory_turns": [123, 456]}\n'
        'Optional source scope: {"tool_call": "memory_search", "category": "source", "queries": ["description of the character or event you need"]}\n'
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
    SONG_SEARCH_TOOL_PROMPT = (
        "\nYou also have a song_search tool for sharing a real song link into the active Discord thread/channel.\n"
        "Use it when a moment genuinely calls for a song, when an NPC would plausibly drop a track/link, or when a meme-like music share would materially help the player.\n"
        "Return ONLY:\n"
        '{"tool_call": "song_search", "query": "artist and song title"}\n'
        "Optional sender/caption example:\n"
        '{"tool_call": "song_search", "query": "Cocteau Twins Heaven or Las Vegas", "sender": "Simone", "message": "No explanation. Just this."}\n'
        "The harness will search and post the resulting YouTube link as a separate Discord message.\n"
        "Do NOT fake a link yourself. Do NOT narrate the song as already heard unless the link actually matters in-scene and you are also making it available with song_search.\n"
        "Use sparingly; this is for meaningful texture, not every turn.\n"
    )
    MEMORY_TOOL_PROMPT = (
        "\nYou have a memory_search tool. To use it, return ONLY:\n"
        '{"tool_call": "memory_search", "queries": ["query1", "query2", ...]}\n'
        "You may also optionally include category, search_within, full_text, keep_memory_turns, before_lines, and after_lines.\n"
        "Optional category scope example:\n"
        '{"tool_call": "memory_search", "category": "char:marcus-blackwell", "queries": ["Marcus at the penthouse", "the deal Marcus offered"]}\n'
        "memory_search returns compact snippets by default. Do a broad summary search first, then narrow before asking for more text.\n"
        'Narrow within the previous turn hits only: {"tool_call": "memory_search", "search_within": "last_results", "queries": ["confession", "deal"]}\n'
        'After narrowing, request expanded turn text only if you still need it: {"tool_call": "memory_search", "search_within": "last_results", "queries": ["exact wording"], "full_text": true}\n'
        'Use keep_memory_turns to prune stale turn hits from tool context for later rounds: {"tool_call": "memory_search", "search_within": "last_results", "queries": ["keyword"], "keep_memory_turns": [123, 456]}\n'
        "If results are weak or empty, you may immediately call memory_search again with refined queries.\n"
        "\nTOOL USAGE POLICY (HIGH PRIORITY):\n"
        "- MANDATORY: On every normal gameplay turn you MUST call at least one memory tool (recent_turns or memory_search) "
        "BEFORE producing final narration. Do not skip this step. Begin your turn with recent_turns, then follow up with "
        "memory_search if deeper or older recall is needed.\n"
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
        "- If unsure what to query, describe the situation: combine current location, active NPC names, and what the player is doing into a phrase.\n"
        "- Prefer this memory flow: recent_turns -> broad memory_search summary -> narrow memory_search with search_within -> full_text only after narrowing.\n"
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
        '{"tool_call": "memory_search", "category": "source", "queries": ["description of the character or event you need"]}\n'
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
        "USE memory_search AGGRESSIVELY when deeper or older continuity matters.\n"
        "You SHOULD use memory_search often, especially:\n"
        "- when a character, NPC, or named entity appears and older context may matter\n"
        "- when the player references past events, locations, objects, or conversations\n"
        "- when describing a revisited location or established NPC\n"
        "- when the player investigates, asks questions, or you are unsure about earlier campaign facts\n"
        "When in doubt between guessing and searching, search.\n"
        "IMPORTANT: Memories are stored as narrator event text (e.g. what happened in a scene). "
        "Queries are matched by semantic similarity against these narration snippets.\n"
        "QUERY STYLE: Use descriptive natural-language phrases that read like a sentence describing what you want to find. "
        "Include names, places, and situational context. Longer, more specific queries retrieve better results.\n"
        "Good examples:\n"
        '  "Marcus meeting at the penthouse"\n'
        '  "what happened when Anastasia visited the garden"\n'
        '  "the deal Marcus offered in the warehouse"\n'
        '  "sword hidden in the cave behind the waterfall"\n'
        "Bad examples (too vague or abstract):\n"
        '  "Marcus" (too short — add what about Marcus)\n'
        '  "character identity role relationship" (abstract; will not match stored events)\n'
        "To recall two separate topics, use two separate descriptive queries:\n"
        '{"tool_call": "memory_search", "queries": ["Marcus meeting at the penthouse", "Anastasia in the garden"]}\n'
    )
    STORY_OUTLINE_TOOL_PROMPT = (
        "\nYou have a story_outline tool. To use it, return ONLY:\n"
        '{"tool_call": "story_outline", "chapter": "chapter-slug"}\n'
        "No other keys alongside tool_call.\n"
        "Returns full expanded chapter with all scene details.\n"
        "Use when you need details about a chapter not fully shown in STORY_CONTEXT.\n"
    )
    PLOT_PLAN_TOOL_PROMPT = (
        "\nYou have a plot_plan tool for forward-looking narrative intentions.\n"
        "Use it to create, update, resolve, or remove multi-turn threads so setups actually pay off.\n"
        "Return ONLY:\n"
        '{"tool_call": "plot_plan", "plans": [{"thread": "thread-slug", "setup": "...", "intended_payoff": "...", "target_turns": 12, "dependencies": ["dep-1", "dep-2"]}]}\n'
        "Resolve or update existing threads by setting status/resolution fields:\n"
        '{"tool_call": "plot_plan", "plans": [{"thread": "thread-slug", "status": "resolved", "resolution": "What finally happened."}]}\n'
        "Remove a stale/invalid thread explicitly:\n"
        '{"tool_call": "plot_plan", "plans": [{"thread": "thread-slug", "remove": true}]}\n'
        "ACTIVE_PLOT_THREADS are returned in prompt context.\n"
        "Use ACTIVE_PLOT_THREADS and ACTIVE_HINTS to maintain momentum and prevent mystery-box drift.\n"
        "Any narrative thread expected to span more than 3 turns SHOULD have a plot plan.\n"
    )
    CHAPTER_PLAN_TOOL_PROMPT = (
        "\nOFF-RAILS CHAPTER MANAGEMENT TOOL:\n"
        "In off-rails mode, you may create, update, advance, or resolve emergent chapter structure via chapter_plan.\n"
        "Create or update a chapter:\n"
        '{"tool_call": "chapter_plan", "action": "create", "chapter": {"slug": "arc-slug", "title": "Arc Title", "summary": "...", "scenes": ["scene-a", "scene-b"], "active": true}}\n'
        "Advance a chapter scene:\n"
        '{"tool_call": "chapter_plan", "action": "advance_scene", "chapter": "arc-slug", "to_scene": "scene-b"}\n'
        "Resolve a chapter:\n"
        '{"tool_call": "chapter_plan", "action": "resolve", "chapter": "arc-slug", "resolution": "What concluded the arc."}\n'
        "ACTIVE_CHAPTERS are returned in prompt context.\n"
        "Use ACTIVE_CHAPTERS to maintain momentum and avoid aimless wandering.\n"
        "If no chapters are active and the player is directionless, create one from the strongest unresolved thread.\n"
    )
    CONSEQUENCE_TOOL_PROMPT = (
        "\nYou have a consequence_log tool for promised downstream effects.\n"
        "Use it when narration establishes a future consequence that should persist beyond this turn.\n"
        "Add a consequence:\n"
        '{"tool_call": "consequence_log", "add": {"trigger": "...", "consequence": "...", "severity": "moderate", "expires_turns": 20}}\n'
        "Resolve a consequence:\n"
        '{"tool_call": "consequence_log", "resolve": {"id": "consequence-id", "resolution": "How it was resolved."}}\n'
        "Remove a consequence explicitly:\n"
        '{"tool_call": "consequence_log", "remove": ["consequence-id"]}\n'
        "ACTIVE_CONSEQUENCES are returned in prompt context. Consult them in relevant scenes.\n"
    )
    CALENDAR_TOOL_PROMPT = (
        "\nCALENDAR & GAME TIME SYSTEM:\n"
        "The campaign tracks in-game time via CURRENT_GAME_TIME shown in the user prompt.\n"
        "The user prompt also includes TIME_MODEL and CALENDAR_POLICY.\n"
        "If TIME_MODEL is individual_clocks, CURRENT_GAME_TIME is the acting player's personal present, while GLOBAL_GAME_TIME (when shown) is the shared world clock for global events.\n"
        "In individual_clocks mode, players may share a location slug without sharing the same moment. Same location does NOT imply direct co-presence.\n"
        "Use PLAYER_CARD.state.game_time and PARTY_SNAPSHOT[*].game_time to judge whether other real players are actually in the same moment or only in the same place at another time.\n"
        "Only treat real players as directly witnessing or interacting with each other when both location and time align closely enough for a shared scene.\n"
        "The user prompt also includes MIN_TURN_ADVANCE_MINUTES_EFFECTIVE, STANDARD_TURN_ADVANCE_MINUTES_EFFECTIVE, and TURN_TIME_BEAT_GUIDANCE for the current campaign speed.\n"
        "Every turn, you MUST advance game_time in state_update by a plausible amount "
        "(ordinary turns use STANDARD_TURN_ADVANCE_MINUTES_EFFECTIVE as the baseline rhythm, longer for travel/rest/time skips, etc.). "
        "Scale the advance by SPEED_MULTIPLIER — at 2x, time passes roughly twice as fast per turn.\n"
        "The harness derives day_of_week automatically from the campaign's Day 1 weekday. Keep it consistent with the world clock.\n"
        "Default rhythm: advance the world by roughly STANDARD_TURN_ADVANCE_MINUTES_EFFECTIVE minutes per turn unless the scene clearly justifies more.\n"
        "MIN_TURN_ADVANCE_MINUTES_EFFECTIVE is the floor the harness will enforce when you freeze or regress time. Do not pace beats as if less time will pass than that.\n"
        "Do NOT default below MIN_TURN_ADVANCE_MINUTES_EFFECTIVE unless immediate shared-scene coherence absolutely requires it.\n"
        "TEMPORAL BEAT COVERAGE RULE: the 1 to 2 beats you write must cover at least MIN_TURN_ADVANCE_MINUTES_EFFECTIVE and also the full span of any larger jump you choose in state_update.game_time.\n"
        "Follow TURN_TIME_BEAT_GUIDANCE from the user prompt as the harness-selected pacing rule for the current minimum turn span. If you intentionally choose a larger jump than that minimum, scale up naturally instead of staying too immediate.\n"
        "Pace game_time by scene needs: prefer larger jumps (20-90 minutes or to the next meaningful beat) when no immediate deadline is active, "
        "and keep finer-grained time only when needed to preserve shared-scene coherence.\n"
        "Update these fields in state_update:\n"
        '- "game_time": {"day": int, "hour": int (0-23), "minute": int (0-59), "day_of_week": string, '
        '"period": "morning"|"afternoon"|"evening"|"night", '
        '"date_label": "Weekday, Day N, Period"}\n'
        "Advance hour/minute naturally; when hour >= 24, increment day and wrap hour.\n"
        "Set period based on hour: 5-11=morning, 12-16=afternoon, 17-20=evening, 21-4=night.\n\n"
        "You may also return a calendar_update key (object) to manage scheduled events:\n"
        '- "calendar_update": {"add": [...], "remove": [...]} where each add entry is '
        '{"name": str, "time_remaining": int, "time_unit": "hours"|"days", "description": str, "known_by": [str, ...], "target_player": str|int (optional), "target_players": [str|int, ...] (optional)} '
        "and each remove entry is a string matching an event name.\n"
        "HARNESS BEHAVIOR:\n"
        "- The harness converts add entries into absolute due dates and stores fire_day + fire_hour (the exact in-game deadline).\n"
        "- known_by is optional. If provided, reminders are only injected when at least one known character is in the active scene.\n"
        "- Keep known_by to character names from PARTY_SNAPSHOT / CHARACTER_INDEX. Omit known_by for globally-known events.\n"
        "- target_player / target_players are optional player-specific targets. These may be a Discord ID, a Discord mention, a player slug, or a PARTY_SNAPSHOT-style string such as '<@123> (Rigby)'.\n"
        "- If no target_player(s) are provided, the event is treated as global.\n"
        "- In shared_clock + loose mode, overdue player-targeted appointments may slide forward instead of being lost.\n"
        "- In shared_clock + consequential mode, overdue player-targeted appointments remain visible as missed/unresolved until the player deals with them.\n"
        "- In individual_clocks mode, player-targeted appointments are judged against that player's own game_time, not another player's pace.\n"
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
        "The harness maintains a character roster (CHARACTER_INDEX / CHARACTER_CARDS, with WORLD_CHARACTERS kept as a compatibility alias). "
        "When you create or update a character via character_updates, the 'appearance' field "
        "is used by the harness to auto-generate a portrait image. Write 'appearance' as a "
        "detailed visual description suitable for image generation: physical features, clothing, "
        "distinguishing marks, pose, and art style cues. Keep it 1-3 sentences, "
        "70-150 words, vivid and concrete.\n"
        "Do NOT include image_url in character_updates — the harness manages that field.\n"
        "Do NOT include location_last_updated in character_updates — the harness manages that field.\n"
    )
    FINAL_STAGE_OPERATIONAL_PROMPT = (
        "\nFINALIZATION OPERATIONAL RULES:\n"
        "- You are in the final JSON-writing stage. Do NOT return a standalone tool_call object now.\n"
        '- You MAY include "tool_calls" in the final JSON for sms_write, sms_schedule, plot_plan, chapter_plan, and song_search only.\n'
        '- If present, tool_calls MUST be the last top-level key in the final JSON object.\n'
        "- If narration includes an SMS/phone reply or outgoing SMS that must persist, include a matching sms_write in tool_calls so both sides of the conversation are logged.\n"
        "- If scheduling a delayed incoming SMS, use tool_calls with sms_schedule and do NOT narrate that delayed message as already received.\n"
        '- In off-rails play, use tool_calls with {"tool_call": "plot_plan", ...} or {"tool_call": "chapter_plan", ...} when the narrated turn meaningfully creates, advances, resolves, or restructures ongoing story threads or chapter flow.\n'
        '- Use tool_calls with {"tool_call": "song_search", "query": "..."} when the turn should make a real song link available in the Discord thread/channel.\n'
        "- If the current scene has a believable grounded clock and needs forced urgency, you may include set_timer_delay / set_timer_event / set_timer_interruptible / set_timer_interrupt_action / set_timer_interrupt_scope in the final JSON.\n"
        "- Timers are for real pressure: force a decision, trigger a grounded consequence, move the player, or force an encounter. Do NOT use timers for trivial flavor.\n"
        "- Timer events must be grounded in established scene facts. Use interrupt_scope=local for hazards anchored to the acting player's immediate situation and global for campaign-wide clocks. Prefer interruptible=false only when the event is already unavoidable.\n"
        "- calendar_update is allowed in final JSON. Use add for deadlines, appointments, and world events with narrative timing pressure.\n"
        "- ONLY use calendar_update.remove when the event was resolved by this turn's narration or when you are clearly cleaning up an already-fired/overdue event that is no longer pending. Never remove events just because time passed.\n"
    )
    ON_RAILS_SYSTEM_PROMPT = (
        "\nON-RAILS MODE IS ENABLED.\n"
        "- You CANNOT create new characters not in CHARACTER_INDEX / WORLD_CHARACTERS. New character slugs will be rejected.\n"
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
        notification_port: NotificationPort | None = None,
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
        self._notification_port = notification_port
        self.append_inventory_to_narration = True
        self._logger = logging.getLogger(__name__)
        self._inflight_turns: set[tuple[str, str]] = set()
        self._inflight_turns_lock = threading.Lock()
        self._timed_event_inflight: dict[tuple[str, str], str] = {}
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

    @property
    def notification_port(self) -> NotificationPort | None:
        return self._notification_port

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
    def _normalize_weekday_name(cls, value: object) -> str:
        text = " ".join(str(value or "").strip().lower().split())
        if text in cls.WEEKDAY_NAMES:
            return text
        alias_map = {
            "mon": "monday",
            "tue": "tuesday",
            "tues": "tuesday",
            "wed": "wednesday",
            "thu": "thursday",
            "thur": "thursday",
            "thurs": "thursday",
            "fri": "friday",
            "sat": "saturday",
            "sun": "sunday",
        }
        return alias_map.get(text, "monday")

    @classmethod
    def _campaign_start_day_of_week(cls, campaign_state: Dict[str, object] | None) -> str:
        if not isinstance(campaign_state, dict):
            return "monday"
        return cls._normalize_weekday_name(
            campaign_state.get(cls.CLOCK_START_DAY_OF_WEEK_KEY, "monday")
        )

    @classmethod
    def _weekday_for_day(cls, *, day: int, start_day_of_week: str) -> str:
        start = cls._normalize_weekday_name(start_day_of_week)
        try:
            start_idx = cls.WEEKDAY_NAMES.index(start)
        except ValueError:
            start_idx = 0
        offset = max(1, int(day)) - 1
        return cls.WEEKDAY_NAMES[(start_idx + offset) % len(cls.WEEKDAY_NAMES)]

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

    def _detect_cross_thread_mentions(
        self,
        narration: str,
        campaign_id: str,
        current_session_id: str | None,
        actor_id: str,
    ) -> list[dict[str, object]]:
        """Return mentioned players who are NOT in the current session's channel."""
        mentioned_ids: set[str] = set()
        for m in self._MENTION_RE.finditer(narration):
            mentioned_ids.add(m.group(1))
        mentioned_ids.discard(str(actor_id))
        if not mentioned_ids:
            return []

        registry = self._campaign_player_registry(campaign_id, self._session_factory)
        by_actor_id: dict[str, dict[str, object]] = registry.get("by_actor_id", {})

        # Filter to only campaign members.
        campaign_mentioned = mentioned_ids & set(by_actor_id.keys())
        if not campaign_mentioned:
            return []

        # Batch: resolve current session channel + all mentioned players'
        # last session channels in a single DB session.
        current_channel: str | None = None
        player_channels: dict[str, str | None] = {}
        with self._session_factory() as session:
            # Current session's effective channel.
            if current_session_id is not None:
                sess = session.get(GameSession, current_session_id)
                if sess is not None:
                    current_channel = sess.surface_thread_id or sess.surface_channel_id

            # Batch query: max turn ID per mentioned actor (with session_id).
            max_turn_sub = (
                session.query(
                    Turn.actor_id,
                    func.max(Turn.id).label("max_id"),
                )
                .filter(Turn.campaign_id == campaign_id)
                .filter(Turn.actor_id.in_(campaign_mentioned))
                .filter(Turn.session_id.isnot(None))
                .group_by(Turn.actor_id)
                .subquery()
            )
            last_turns = (
                session.query(Turn.actor_id, Turn.session_id)
                .join(max_turn_sub, Turn.id == max_turn_sub.c.max_id)
                .all()
            )

            # Batch fetch corresponding sessions.
            session_ids = {
                row.session_id for row in last_turns if row.session_id
            }
            sessions_by_id: dict[str, GameSession] = {}
            if session_ids:
                for sess_row in (
                    session.query(GameSession)
                    .filter(GameSession.id.in_(session_ids))
                    .all()
                ):
                    sessions_by_id[sess_row.id] = sess_row

            for row in last_turns:
                player_sess = sessions_by_id.get(row.session_id or "")
                if player_sess is not None:
                    player_channels[row.actor_id] = (
                        player_sess.surface_thread_id or player_sess.surface_channel_id
                    )

        recipients: list[dict[str, object]] = []
        for uid in campaign_mentioned:
            player_channel = player_channels.get(uid)
            if current_channel is not None and player_channel == current_channel:
                # Player is active in the same thread — they'll see it.
                continue
            entry = by_actor_id[uid]
            recipients.append({
                "actor_id": uid,
                "character_name": str(
                    entry.get("name") or f"Adventurer-{str(uid)[-4:]}"
                ),
            })
        return recipients

    async def _send_cross_thread_mention_forwards(
        self,
        *,
        campaign_name: str,
        actor_character_name: str,
        narration: str,
        recipients: list[dict[str, object]],
    ) -> None:
        """Forward *narration* as a DM to each recipient not in the actor's thread."""
        if not recipients:
            return
        if self._notification_port is None:
            return
        for recipient in recipients:
            uid = str(recipient["actor_id"])
            try:
                message = (
                    "\U0001f4e8 **Forwarded from "
                    f"{actor_character_name}'s scene** "
                    f"in `{campaign_name}`:\n\n{narration}"
                )
                await self._notification_port.send_dm(uid, message)
            except Exception:
                self._logger.debug(
                    "Zork: failed to forward cross-thread mention DM to actor %s",
                    uid,
                    exc_info=True,
                )

    @staticmethod
    def _safe_turn_meta(turn: Turn) -> dict[str, object]:
        meta = parse_json_dict(getattr(turn, "meta_json", "{}"))
        return meta if isinstance(meta, dict) else {}

    def _is_last_narrator_turn_public(self, campaign_id: str) -> bool:
        """Check if the most recent narrator turn has public or local scope."""
        with self._session_factory() as session:
            turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .filter(Turn.kind == "narrator")
                .order_by(Turn.id.desc())
                .first()
            )
            if turn is None:
                return False
            meta = self._safe_turn_meta(turn)
            visibility = meta.get("visibility")
            if not isinstance(visibility, dict):
                return True  # No visibility metadata → treat as public.
            scope = str(visibility.get("scope") or "").strip().lower()
            return scope in {"public", "local"}

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
            if cls._viewer_participated_in_turn_scene_output(
                turn,
                viewer_actor_id=viewer_actor_id,
                viewer_slug=viewer_slug,
            ):
                return True
        if viewer_actor_id in actor_ids:
            return True
        return bool(viewer_slug and viewer_slug in player_slugs)

    @classmethod
    def _viewer_participated_in_turn_scene_output(
        cls,
        turn: Turn,
        *,
        viewer_actor_id: str,
        viewer_slug: str,
    ) -> bool:
        meta = cls._safe_turn_meta(turn)
        scene_output = meta.get("scene_output")
        if not isinstance(scene_output, dict):
            return False
        beats = scene_output.get("beats")
        if not isinstance(beats, list):
            return False
        viewer_actor_id_text = str(viewer_actor_id or "").strip()
        viewer_slug_key = cls._player_slug_key(viewer_slug)
        if not viewer_actor_id_text and not viewer_slug_key:
            return False
        for beat in beats:
            if not isinstance(beat, dict):
                continue
            visible_actor_ids = {
                str(item or "").strip()
                for item in list(beat.get("visible_actor_ids") or [])
                if str(item or "").strip()
            }
            if viewer_actor_id_text and viewer_actor_id_text in visible_actor_ids:
                return True
            participant_slugs = {
                cls._player_slug_key(item)
                for item in (
                    list(beat.get("actors") or [])
                    + list(beat.get("listeners") or [])
                    + [beat.get("speaker")]
                )
                if cls._player_slug_key(item)
            }
            if viewer_slug_key and viewer_slug_key in participant_slugs:
                return True
        return False

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
                default_campaign = self.get_or_create_campaign(guild, "main", created_by_actor_id="system")
                row = GameSession(
                    campaign_id=default_campaign.id,
                    surface="discord_channel",
                    surface_key=key,
                    surface_guild_id=guild,
                    surface_channel_id=channel,
                    enabled=False,
                    metadata_json=self._dump_json({"active_campaign_id": default_campaign.id}),
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
            channel_row.campaign_id = active_campaign_id
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
            channel_row.campaign_id = campaign.id
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
        channel_campaign_id = getattr(channel, "campaign_id", None)
        if channel_campaign_id:
            with self._session_factory() as session:
                campaign = session.get(Campaign, str(channel_campaign_id))
            if campaign is not None:
                if str(active_campaign_id or "") != str(channel_campaign_id):
                    with self._session_factory() as session:
                        channel_row = session.get(GameSession, channel.id)
                        if channel_row is not None:
                            metadata = self._load_session_metadata(channel_row)
                            metadata["active_campaign_id"] = str(channel_campaign_id)
                            channel_row.campaign_id = str(channel_campaign_id)
                            self._store_session_metadata(channel_row, metadata)
                            channel_row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                            session.commit()
                return str(channel_campaign_id), None

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

    @classmethod
    def _normalize_character_roster(cls, raw_characters: object) -> dict[str, Any]:
        if not isinstance(raw_characters, dict):
            return {}
        normalized: dict[str, Any] = {}
        for raw_slug, raw_entry in raw_characters.items():
            slug = str(raw_slug or "").strip()
            if not slug:
                continue
            if isinstance(raw_entry, dict):
                normalized[slug] = raw_entry
                continue
            if isinstance(raw_entry, list):
                first_dict = next(
                    (item for item in raw_entry if isinstance(item, dict)),
                    None,
                )
                if first_dict is not None:
                    normalized[slug] = dict(first_dict)
        return normalized

    def get_campaign_characters(self, campaign: Campaign) -> dict[str, Any]:
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is not None:
                return self._normalize_character_roster(
                    parse_json_dict(row.characters_json)
                )
        return self._normalize_character_roster(parse_json_dict(campaign.characters_json))

    def get_chapter_list(self, campaign: Campaign) -> dict[str, Any]:
        """Return a structured chapter list for frontend display.

        Works for both on-rails (story_outline.chapters) and off-rails
        (_chapter_plan) campaigns.
        """
        state = self.get_campaign_state(campaign)
        on_rails = bool(state.get("on_rails", False))
        current_chapter = state.get("current_chapter", 0)
        current_scene = state.get("current_scene", 0)

        chapters: list[dict[str, Any]] = []

        if on_rails:
            outline = state.get("story_outline")
            if isinstance(outline, dict):
                raw_chapters = outline.get("chapters")
                if isinstance(raw_chapters, list):
                    for idx, ch in enumerate(raw_chapters):
                        if not isinstance(ch, dict):
                            continue
                        scenes = []
                        raw_scenes = ch.get("scenes")
                        if isinstance(raw_scenes, list):
                            for si, sc in enumerate(raw_scenes):
                                if isinstance(sc, dict):
                                    scenes.append({
                                        "title": sc.get("title", "Untitled"),
                                        "summary": sc.get("summary", ""),
                                        "is_current": idx == current_chapter and si == current_scene,
                                    })
                                else:
                                    scenes.append({
                                        "title": str(sc or "Untitled"),
                                        "summary": "",
                                        "is_current": idx == current_chapter and si == current_scene,
                                    })
                        is_current = idx == current_chapter
                        status = "completed" if idx < current_chapter else ("active" if is_current else "upcoming")
                        chapters.append({
                            "title": ch.get("title", f"Chapter {idx + 1}"),
                            "summary": ch.get("summary", ""),
                            "scenes": scenes,
                            "status": status,
                            "is_current": is_current,
                            "index": idx,
                        })
        else:
            plan = self._chapter_plan_from_state(state)
            current_slug = str(current_chapter) if not isinstance(current_chapter, int) else ""
            if isinstance(current_chapter, str):
                current_slug = current_chapter
            sorted_chapters = sorted(
                plan.values(),
                key=lambda row: (
                    0 if str(row.get("status")) == "active" else 1,
                    -self._coerce_non_negative_int(row.get("updated_turn", 0), default=0),
                    str(row.get("slug") or ""),
                ),
            )
            for ch in sorted_chapters:
                slug = ch.get("slug", "")
                is_current = slug == current_slug
                scene_slug = str(current_scene) if isinstance(current_scene, str) else ""
                scenes = []
                for sc_slug in (ch.get("scenes") or []):
                    label = str(sc_slug or "").replace("-", " ").replace("_", " ").strip()
                    label = " ".join(w.capitalize() for w in label.split()) if label else "Untitled"
                    scenes.append({
                        "title": label,
                        "summary": "",
                        "is_current": is_current and sc_slug == scene_slug,
                    })
                chapters.append({
                    "title": ch.get("title", slug),
                    "summary": ch.get("summary", ""),
                    "scenes": scenes,
                    "status": ch.get("status", "active"),
                    "is_current": is_current,
                    "slug": slug,
                    "resolution": ch.get("resolution", ""),
                })

        return {
            "on_rails": on_rails,
            "current_chapter": current_chapter,
            "current_scene": current_scene,
            "chapters": chapters,
        }

    @classmethod
    def format_chapter_outline(cls, story_outline: dict[str, Any]) -> str:
        """Format a story_outline dict into a human-readable chapter listing."""
        if not isinstance(story_outline, dict):
            return ""
        chapters = story_outline.get("chapters")
        if not isinstance(chapters, list) or not chapters:
            return ""
        lines: list[str] = []
        for idx, ch in enumerate(chapters):
            if not isinstance(ch, dict):
                continue
            title = ch.get("title", f"Chapter {idx + 1}")
            summary = str(ch.get("summary", "")).strip()
            lines.append(f"**{idx + 1}. {title}**")
            if summary:
                lines.append(f"   {summary}")
            scenes = ch.get("scenes")
            if isinstance(scenes, list):
                for si, sc in enumerate(scenes):
                    if isinstance(sc, dict):
                        sc_title = sc.get("title", f"Scene {si + 1}")
                    else:
                        sc_title = str(sc or f"Scene {si + 1}")
                    lines.append(f"   - {sc_title}")
        return "\n".join(lines)

    @classmethod
    def _normalize_setup_variant_main_character(cls, value: Any) -> Any:
        if isinstance(value, dict):
            normalized = dict(value)
            name = str(
                normalized.get("name")
                or normalized.get("character_name")
                or normalized.get("title")
                or normalized.get("label")
                or ""
            ).strip()
            if name:
                normalized["name"] = name
            return normalized
        if isinstance(value, list) and value:
            first = value[0]
            normalized_first = cls._normalize_setup_variant_main_character(first)
            return normalized_first or "The Protagonist"
        text = str(value or "").strip()
        return text or "The Protagonist"

    @classmethod
    def _normalize_setup_variant_npcs(cls, value: Any) -> list[Any]:
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            nested = value.get("characters") or value.get("npcs") or value.get("essential_npcs")
            if isinstance(nested, list):
                return list(nested)
            return [dict(value)]
        text = str(value or "").strip()
        return [text] if text else []

    @classmethod
    def _normalize_setup_variant_chapter_outline(cls, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            nested = value.get("chapters") or value.get("chapter_outline") or value.get("outline")
            if isinstance(nested, list):
                value = nested
            else:
                value = [value]
        if not isinstance(value, list):
            text = str(value or "").strip()
            return [{"title": text}] if text else []

        normalized: list[dict[str, Any]] = []
        for idx, chapter in enumerate(value, start=1):
            if isinstance(chapter, dict):
                title = str(
                    chapter.get("title")
                    or chapter.get("name")
                    or chapter.get("chapter")
                    or chapter.get("label")
                    or ""
                ).strip()
                summary = str(
                    chapter.get("summary")
                    or chapter.get("description")
                    or chapter.get("premise")
                    or ""
                ).strip()
                normalized_chapter = dict(chapter)
                if title:
                    normalized_chapter["title"] = title
                elif not normalized_chapter.get("title"):
                    normalized_chapter["title"] = f"Chapter {idx}"
                if summary and not normalized_chapter.get("summary"):
                    normalized_chapter["summary"] = summary
                normalized.append(normalized_chapter)
                continue
            text = str(chapter or "").strip()
            if text:
                normalized.append({"title": text})
        return normalized

    @classmethod
    def _format_setup_variant_person(cls, value: Any) -> str:
        if isinstance(value, dict):
            name = str(
                value.get("name")
                or value.get("character_name")
                or value.get("title")
                or value.get("label")
                or ""
            ).strip()
            role = str(value.get("role") or value.get("job") or value.get("archetype") or "").strip()
            if name and role:
                return f"{name} ({role})"
            if name:
                return name
            if role:
                return role
            desc = str(value.get("description") or value.get("summary") or "").strip()
            return desc or cls._dump_json(value)
        text = str(value or "").strip()
        return text

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

    def rename_player_character(
        self,
        campaign_id: str,
        actor_id: str,
        new_name: str,
    ) -> dict[str, object]:
        clean_name = " ".join(str(new_name or "").strip().split())
        if not clean_name:
            raise ValueError("Character name is required.")
        clean_name = clean_name[:128]

        with self._session_factory() as session:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                raise KeyError(f"Unknown campaign: {campaign_id}")

            player = (
                session.query(Player)
                .filter(Player.campaign_id == campaign_id)
                .filter(Player.actor_id == actor_id)
                .first()
            )
            if player is None:
                raise KeyError(f"Unknown player in campaign: {actor_id}")

            actor = session.get(Actor, actor_id)
            if actor is None:
                actor = Actor(id=actor_id, display_name=clean_name, kind="human", metadata_json="{}")
                session.add(actor)

            player_state = parse_json_dict(player.state_json)
            old_name = " ".join(
                str(
                    player_state.get("character_name")
                    or getattr(actor, "display_name", None)
                    or actor_id
                ).strip().split()
            )[:128]
            player_state["character_name"] = clean_name
            player.state_json = self._dump_json(player_state)
            player.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            player.last_active_at = datetime.now(timezone.utc).replace(tzinfo=None)

            actor.display_name = clean_name
            actor.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

            migrated_roster_slug = None
            characters = parse_json_dict(campaign.characters_json)
            if isinstance(characters, dict) and characters:
                lookup_name = old_name or clean_name
                resolved_slug = self._resolve_existing_character_slug(characters, lookup_name)
                if resolved_slug is not None:
                    entry = characters.get(resolved_slug)
                    if isinstance(entry, dict):
                        updated_entry = dict(entry)
                        updated_entry["name"] = clean_name
                        canonical_slug = re.sub(r"[^a-z0-9]+", "-", clean_name.lower()).strip("-")
                        if canonical_slug and canonical_slug != resolved_slug and canonical_slug not in characters:
                            characters.pop(resolved_slug, None)
                            characters[canonical_slug] = updated_entry
                            migrated_roster_slug = canonical_slug
                        else:
                            characters[resolved_slug] = updated_entry
                            migrated_roster_slug = resolved_slug
                        campaign.characters_json = self._dump_json(characters)
                        campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

            session.commit()

        self._zork_log(
            "PLAYER CHARACTER RENAMED",
            f"campaign={campaign_id}\nactor_id={actor_id}\nold_name={old_name}\nnew_name={clean_name}",
        )
        return {
            "ok": True,
            "actor_id": actor_id,
            "old_name": old_name,
            "name": clean_name,
            "migrated_roster_slug": migrated_roster_slug,
        }

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
            "If this references a known movie, book, show, or story, describe the MAIN CHARACTER's physical appearance — "
            "age, build, hair, clothing, distinguishing features. Not personality or writing style.\n"
            "If it's an original setting, describe a fitting protagonist the same way.\n"
            "Write it like a casting sheet: 'Late 20s, lean, dark curly hair, rumpled suit, ink-stained fingers, "
            "furrowed brow' — visual details an artist could draw from, not behavior or voice.\n"
            "Return ONLY the persona (1-2 sentences, max 140 chars). No quotes or explanation."
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
            elif phase == "world_structure_pick":
                result = await self._setup_handle_world_structure_pick(
                    campaign,
                    state,
                    setup_data,
                    clean_text,
                )
            elif phase == "calendar_time_policy_pick":
                result = await self._setup_handle_calendar_time_policy_pick(
                    campaign,
                    state,
                    setup_data,
                    clean_text,
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
        world_structure = str(
            setup_data.get("world_structure") or "character-centric"
        ).strip().lower()
        if world_structure == "shared-world":
            structure_context = (
                "\nCampaign structure: shared-world.\n"
                "Build variants around a world many players can inhabit, not a single chosen protagonist.\n"
                "Do not make the campaign depend on one singular lead character.\n"
                "The `main_character` field should name a suggested entry point, first viewpoint, or social anchor, "
                "not the only person the campaign matters to.\n"
            )
        else:
            structure_context = (
                "\nCampaign structure: character-centric.\n"
                "Build variants around a strong central player-facing lead or main viewpoint character.\n"
            )
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
        world_structure = str(
            setup_data.get("world_structure") or "character-centric"
        ).strip().lower()
        if world_structure == "shared-world":
            structure_context = (
                "\nCampaign structure: shared-world.\n"
                "Build variants around a world many players can inhabit, not a single chosen protagonist.\n"
                "Do not make the campaign depend on one singular lead character.\n"
                "The `main_character` field should name a suggested entry point, first viewpoint, or social anchor, "
                "not the only person the campaign matters to.\n"
            )
        else:
            structure_context = (
                "\nCampaign structure: character-centric.\n"
                "Build variants around a strong central player-facing lead or main viewpoint character.\n"
            )

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
                    f"{structure_context}"
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
                    f"{structure_context}"
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
                            "summary": summary,
                            "main_character": self._normalize_setup_variant_main_character(
                                row.get("main_character")
                            ),
                            "essential_npcs": self._normalize_setup_variant_npcs(
                                row.get("essential_npcs", [])
                            ),
                            "chapter_outline": self._normalize_setup_variant_chapter_outline(
                                row.get("chapter_outline", [])
                            ),
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
            lines.append(
                f"Main character: {self._format_setup_variant_person(variant.get('main_character', 'TBD'))}"
            )
            npcs = variant.get("essential_npcs", [])
            if isinstance(npcs, list) and npcs:
                npc_labels = [
                    self._format_setup_variant_person(npc)
                    for npc in npcs
                    if str(self._format_setup_variant_person(npc) or "").strip()
                ]
                if npc_labels:
                    lines.append(f"Key NPCs: {', '.join(npc_labels)}")
            chapters = variant.get("chapter_outline", [])
            if isinstance(chapters, list) and chapters:
                titles = []
                for ch_idx, ch in enumerate(chapters, start=1):
                    if not isinstance(ch, dict):
                        continue
                    title = str(
                        ch.get("title")
                        or ch.get("name")
                        or ch.get("chapter")
                        or f"Chapter {ch_idx}"
                    ).strip()
                    if title:
                        titles.append(title)
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

    @classmethod
    def _setup_world_structure_prompt(cls) -> str:
        return (
            "What kind of campaign structure do you want?\n\n"
            "1. **character-centric** — a main player character / lead viewpoint anchors the campaign.\n"
            "2. **shared-world** — the world matters more than one singular protagonist; multiple players can inhabit it.\n\n"
            "Reply with **character-centric** or **shared-world**.\n"
            "Short aliases also work: **main-player** / **ensemble**."
        )

    @classmethod
    def _parse_setup_world_structure_choice(
        cls,
        content: str,
    ) -> tuple[str | None, str | None]:
        raw = str(content or "").strip().lower()
        if not raw:
            return None, "Please choose a campaign structure."

        normalized = raw.replace("_", "-").replace(" ", "-")
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
        protagonist_tokens = {
            "1",
            "character-centric",
            "charactercentric",
            "main-player",
            "main-character",
            "protagonist",
            "lead",
            "solo",
        }
        shared_tokens = {
            "2",
            "shared-world",
            "sharedworld",
            "ensemble",
            "world-centric",
            "sandbox",
            "multi-player",
            "multiplayer",
        }
        if normalized in protagonist_tokens:
            return "character-centric", None
        if normalized in shared_tokens:
            return "shared-world", None
        return None, "Reply with `character-centric` or `shared-world`."

    @classmethod
    def _setup_calendar_time_policy_prompt(cls) -> str:
        return (
            "How should time and personal calendar events work in this campaign?\n\n"
            "1. **loose-calendar** — one shared world clock; if another player's progress would skip a personal appointment, the harness slides it forward by one day instead of dropping it.\n"
            "2. **consequential-calendar** — one shared world clock; if a personal appointment is missed, it stays visible as missed until the player deals with it.\n"
            "3. **individual-calendars** — each player keeps their own personal game_time and personal calendar pace; players can share a location slug without sharing the same moment.\n\n"
            "Reply with **loose-calendar**, **consequential-calendar**, or **individual-calendars**.\n"
            "Short aliases also work: **loose**, **consequential**, **individual**."
        )

    @classmethod
    def _parse_setup_calendar_time_policy_choice(
        cls,
        content: str,
    ) -> tuple[dict[str, str] | None, str | None]:
        raw = str(content or "").strip().lower()
        if not raw:
            return None, "Please choose a time/calendar handling mode."

        normalized = raw.replace("_", "-").replace(" ", "-")
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
        loose_tokens = {
            "1",
            "loose",
            "loose-calendar",
            "soft-calendar",
            "floating-calendar",
            "reschedule",
        }
        consequential_tokens = {
            "2",
            "consequential",
            "consequential-calendar",
            "strict-calendar",
            "missed-events",
            "shared-clock",
        }
        individual_tokens = {
            "3",
            "individual",
            "individual-calendar",
            "individual-calendars",
            "individual-clocks",
            "async",
            "asynchronous",
        }
        if normalized in loose_tokens:
            return {
                "time_model": cls.TIME_MODEL_SHARED_CLOCK,
                "calendar_policy": cls.CALENDAR_POLICY_LOOSE,
            }, None
        if normalized in consequential_tokens:
            return {
                "time_model": cls.TIME_MODEL_SHARED_CLOCK,
                "calendar_policy": cls.CALENDAR_POLICY_CONSEQUENTIAL,
            }, None
        if normalized in individual_tokens:
            return {
                "time_model": cls.TIME_MODEL_INDIVIDUAL_CLOCKS,
                "calendar_policy": cls.CALENDAR_POLICY_CONSEQUENTIAL,
            }, None
        return None, (
            "Reply with `loose-calendar`, `consequential-calendar`, or `individual-calendars`."
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
        state["setup_phase"] = "world_structure_pick"
        state["setup_data"] = setup_data
        return self._setup_world_structure_prompt()

    async def _setup_handle_world_structure_pick(
        self,
        campaign: Campaign,
        state: dict[str, Any],
        setup_data: dict[str, Any],
        message_text: str,
    ) -> str:
        world_structure, error = self._parse_setup_world_structure_choice(message_text)
        if error:
            return f"{error}\n\n{self._setup_world_structure_prompt()}"
        setup_data["world_structure"] = world_structure
        state["setup_phase"] = "calendar_time_policy_pick"
        state["setup_data"] = setup_data
        return self._setup_calendar_time_policy_prompt()

    async def _setup_handle_calendar_time_policy_pick(
        self,
        campaign: Campaign,
        state: dict[str, Any],
        setup_data: dict[str, Any],
        message_text: str,
    ) -> str:
        choice, error = self._parse_setup_calendar_time_policy_choice(message_text)
        if error:
            return f"{error}\n\n{self._setup_calendar_time_policy_prompt()}"
        setup_data["time_model"] = str(choice.get("time_model") or self.TIME_MODEL_SHARED_CLOCK)
        setup_data["calendar_policy"] = str(
            choice.get("calendar_policy") or self.CALENDAR_POLICY_CONSEQUENTIAL
        )
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
        world_structure = str(
            setup_data.get("world_structure") or "character-centric"
        ).strip().lower()
        time_model = self._time_model_from_state(setup_data)
        calendar_policy = self._calendar_policy_from_state(setup_data)

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
            if world_structure == "shared-world":
                structure_context = (
                    "\nCampaign structure: shared-world.\n"
                    "Build a setting that supports multiple player characters, factions, and entry points.\n"
                    "Do not center the world on one singular destined protagonist.\n"
                    "If you name a `main_character`, treat that as a suggested entry point or initial viewpoint only.\n"
                )
            else:
                structure_context = (
                    "\nCampaign structure: character-centric.\n"
                    "It is acceptable to build the world around a central player-facing lead or main viewpoint.\n"
                )
            if time_model == self.TIME_MODEL_INDIVIDUAL_CLOCKS:
                time_context = (
                    "\nTime/calendar handling: individual-calendars.\n"
                    "Different players may occupy the same location in different personal times. "
                    "Build a world that tolerates asynchronous progression, traces of prior visits, "
                    "and personal appointments that matter on each player's own pace.\n"
                )
            elif calendar_policy == self.CALENDAR_POLICY_LOOSE:
                time_context = (
                    "\nTime/calendar handling: loose-calendar on a shared world clock.\n"
                    "Personal appointments should be resilient. If timing slips because of other players' progress, "
                    "the harness may slide them forward instead of treating them as lost.\n"
                )
            else:
                time_context = (
                    "\nTime/calendar handling: consequential-calendar on a shared world clock.\n"
                    "Missed appointments should matter and remain part of continuity until addressed.\n"
                )
            finalize_system = (
                "You are a world-builder for interactive text-adventure campaigns.\n"
                "For non-canonical/original characters, choose distinctive specific names; avoid generic defaults "
                "(Morgan, Chen, Mendoza, Rollins, Nakamura, Kai, River) unless source canon requires them.\n"
                f"{source_tool_instructions}"
                "Return ONLY valid JSON with keys: characters, story_outline, summary, "
                "start_room, landmarks, setting, tone, default_persona, opening_narration, starting_day_of_week.\n"
                "Choose a starting_day_of_week for Day 1 using a real weekday name.\n"
                "No markdown, no code fences."
            )
            finalize_user = (
                f"Build the complete world for: '{raw_name}'\n"
                f"Known work: {is_known}\n"
                f"Description: {setup_data.get('work_description', '')}\n"
                f"{imdb_context}"
                f"{attachment_context}"
                f"{source_index_hint}"
                f"{structure_context}"
                f"{time_context}"
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
                            f"{structure_context}"
                            f"{time_context}"
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
                "starting_day_of_week": "monday",
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
        starting_day_of_week = self._normalize_weekday_name(
            world.get("starting_day_of_week", state.get(self.CLOCK_START_DAY_OF_WEEK_KEY, "monday"))
        )

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
        state[self.CLOCK_START_DAY_OF_WEEK_KEY] = starting_day_of_week
        state["on_rails"] = on_rails
        state["puzzle_mode"] = novel_prefs.get("puzzle_mode", "none")
        state["world_structure"] = world_structure
        state["time_model"] = time_model
        state["calendar_policy"] = calendar_policy

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
                if isinstance(main_char, dict):
                    main_char = str(main_char.get("name") or "").strip()
                if (
                    world_structure != "shared-world"
                    and main_char
                    and not player_state.get("character_name")
                ):
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
        chapter_outline_text = self.format_chapter_outline(story_outline)
        if chapter_outline_text:
            result_msg += f"__**Chapter Outline**__\n{chapter_outline_text}\n\n"
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
                        "game_time": state.get("game_time") if isinstance(state.get("game_time"), dict) else None,
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
        image_index = 2 if has_room_reference else 1
        for ref in avatar_refs:
            name = self._scene_image_reference_name(ref.get("name"))
            directives.append(f"Render {name} to match the person in image {image_index}.")
            image_index += 1
        if directives:
            prompt = f"{' '.join(directives)} {prompt}"
        prompt = re.sub(r"\s+", " ", prompt).strip()
        return prompt

    @staticmethod
    def _scene_image_reference_name(value: object) -> str:
        words = [part for part in re.split(r"\s+", str(value or "").strip()) if part]
        if not words:
            return "character"
        return " ".join(words[:2])

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
            prompt_parts.insert(1, f"Appearance: {persona}.")
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
        prefix = self.SCENE_IMAGE_PRESERVE_PREFIX.strip() if has_room_reference else ""
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
            "- Relationship nuance: do not flatten every romance or partnership into pure loyalty and honesty. "
            "Most of the time, relationship patterns should cohere with the seed and the rest of the generated profile. "
            "Rarely, if it genuinely fits the character's background, appetites, evasions, or self-protective habits, they may be flaky, deceitful, or unfaithful. "
            "If so, encode that as part of trust patterns and relationship history, not as random shock value.\n"
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

            # Generate founding autobiography from enriched profile.
            await self._generate_character_constitution(
                campaign_id, character_slug, profile,
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

    async def _generate_character_constitution(
        self,
        campaign_id: str,
        character_slug: str,
        enriched_profile: dict,
    ) -> bool:
        """Generate the founding autobiography from an enriched character brief."""
        if self._completion_port is None:
            return False
        try:
            brief_fields = {
                k: v for k, v in enriched_profile.items()
                if isinstance(v, str) and v.strip()
                and k in self.ENRICHMENT_FIELDS
            }
            if not brief_fields:
                return False

            user_prompt = json.dumps(brief_fields, indent=2)
            response = await self._completion_port.complete(
                self.CHARACTER_CONSTITUTION_WRITER_PROMPT,
                user_prompt,
                temperature=0.7,
                max_tokens=1200,
            )
            if not response or not response.strip():
                self._logger.warning(
                    "Constitution generation returned empty for %s",
                    character_slug,
                )
                return False

            constitution = self._sanitize_autobiography_text(
                response.strip(),
                max_chars=self.MAX_AUTOBIOGRAPHY_TEXT_CHARS,
            )
            if not constitution:
                return False

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
                char[self.AUTOBIOGRAPHY_FIELD] = constitution
                campaign.characters_json = self._dump_json(characters)
                campaign.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.commit()

            self._logger.info(
                "CHARACTER CONSTITUTION COMPLETE slug=%s campaign=%s len=%d",
                character_slug, campaign_id, len(constitution),
            )
            return True
        except Exception as e:
            self._logger.warning(
                "Constitution generation failed for %s: %s",
                character_slug, e,
            )
            return False

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

    def _set_timed_event_inflight(
        self,
        campaign_id: str,
        actor_id: str,
        event_description: str,
    ) -> None:
        key = (str(campaign_id), str(actor_id))
        with self._inflight_turns_lock:
            self._timed_event_inflight[key] = str(event_description or "").strip()

    def _clear_timed_event_inflight(self, campaign_id: str, actor_id: str) -> None:
        key = (str(campaign_id), str(actor_id))
        with self._inflight_turns_lock:
            self._timed_event_inflight.pop(key, None)

    def get_timed_event_in_progress_notice(
        self,
        campaign_id: str,
        actor_id: str,
    ) -> str | None:
        key = (str(campaign_id), str(actor_id))
        with self._inflight_turns_lock:
            event_description = str(self._timed_event_inflight.get(key) or "").strip()
        if not event_description:
            return None
        return (
            "⏰ Timed event in progress. Your action is waiting for it to finish.\n"
            f"Pending event: {event_description}"
        )

    async def _play_action_with_ids(
        self,
        campaign_id: str,
        actor_id: str,
        action: str,
        session_id: str | None = None,
        manage_claim: bool = True,
        *,
        progress=None,
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
                                owner_actor_id=actor_id,
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
                ),
                progress=progress,
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
                    speed = self.get_timed_events_speed_multiplier(campaign_row)
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
                if result.scene_image_prompt and portrait_channel_ref:
                    scene_ctx = SimpleNamespace(
                        author=SimpleNamespace(id=str(actor_id)),
                        channel=SimpleNamespace(id=str(portrait_channel_ref)),
                    )
                    asyncio.create_task(
                        self._enqueue_scene_image(
                            scene_ctx,
                            str(result.scene_image_prompt),
                            campaign_id=campaign_id,
                        )
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
                display_narration = self._decorate_narration_and_persist(
                    campaign_id=campaign_id,
                    actor_id=actor_id,
                    narration=result.narration or "",
                    timer_instruction=result.timer_instruction,
                    timer_delay_seconds=timer_delay_seconds,
                )

                # Cross-thread mention forwarding
                if display_narration and self._notification_port is not None:
                    _turn_is_public = self._is_last_narrator_turn_public(
                        campaign_id,
                    )
                    if _turn_is_public:
                        _cross_thread_recipients = self._detect_cross_thread_mentions(
                            display_narration,
                            campaign_id,
                            session_id,
                            actor_id,
                        )
                        if _cross_thread_recipients:
                            _actor_name = str(sms_sender_name or "Unknown").strip()
                            _campaign_display_name = campaign_id
                            with self._session_factory() as _sf_session:
                                _cmp = _sf_session.get(Campaign, campaign_id)
                                if _cmp is not None:
                                    _campaign_display_name = _cmp.name or campaign_id
                            asyncio.create_task(
                                self._send_cross_thread_mention_forwards(
                                    campaign_name=_campaign_display_name,
                                    actor_character_name=_actor_name,
                                    narration=display_narration,
                                    recipients=_cross_thread_recipients,
                                )
                            )

                # Fire-and-forget: embed the narrator turn for memory search.
                if self._memory_port is not None and display_narration:
                    try:
                        with self._session_factory() as _embed_session:
                            last_turns = (
                                _embed_session.query(Turn)
                                .filter(Turn.campaign_id == campaign_id)
                                .filter(Turn.kind == "narrator")
                                .order_by(Turn.id.desc())
                                .limit(1)
                                .all()
                            )
                        if last_turns:
                            _embed_turn = last_turns[0]
                            _embed_meta = self._safe_turn_meta(_embed_turn)
                            _embed_visibility = _embed_meta.get("visibility")
                            self._memory_port.store_turn_embedding(
                                turn_id=int(_embed_turn.id),
                                campaign_id=campaign_id,
                                actor_id=actor_id,
                                kind="narrator",
                                content=result.narration or "",
                                metadata=self._turn_embedding_metadata(
                                    visibility=_embed_visibility if isinstance(_embed_visibility, dict) else None,
                                    actor_player_slug=_embed_meta.get("actor_player_slug"),
                                    location_key=_embed_meta.get("location_key"),
                                    session_id=session_id,
                                ),
                            )
                    except Exception:
                        self._logger.debug(
                            "Turn embedding skipped for campaign=%s",
                            campaign_id,
                            exc_info=True,
                        )

                return display_narration
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
            if self.append_inventory_to_narration:
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
                contact_roster=self._sms_contact_roster(campaign) if campaign is not None else None,
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
                    f"⏰ Timer pending: fires <t:{expiry_ts}:F> (<t:{expiry_ts}:R>) - {event_hint} ({interrupt_hint})"
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
            campaign = session.get(Campaign, campaign_id)
            campaign_state = parse_json_dict(campaign.state_json) if campaign is not None else {}
            game_time_snapshot = self._extract_game_time_snapshot(campaign_state)
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
                mention_re = self._MENTION_RE
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
                    game_time=game_time_snapshot,
                )
                source_player.state_json = self._dump_json(source_state)

            target_state = parse_json_dict(target_player.state_json)
            target_inventory = self._get_inventory_rich(target_state)
            target_state["inventory"] = self._apply_inventory_delta(
                target_inventory,
                [gi_item_name],
                [],
                origin_hint=f"Received from <@{actor_id}>",
                game_time=game_time_snapshot,
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
        progress=None,
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
                game_time, _global_game_time, _time_model, _calendar_policy = self._current_game_time_for_prompt(
                    campaign_state,
                    player_state,
                )
                calendar_entries = self._calendar_for_prompt(
                    campaign_state,
                    player_state=player_state,
                    viewer_actor_id=actor_id_text,
                )
                player_calendar_lines = self._player_calendar_events_for_display(
                    player_state
                )
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
                            if str(event.get("status") or "").strip().lower() == "missed":
                                eta = "missed and still unresolved"
                            else:
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
                if player_calendar_lines:
                    lines.append("**Personal Events:**")
                    lines.extend(player_calendar_lines)
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
                progress=progress,
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
            progress=progress,
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

    def bind_latest_narrator_message(
        self,
        campaign_id: str,
        bot_message_id: str | int,
    ) -> bool:
        bot_id = str(bot_message_id or "").strip()
        if not bot_id:
            return False
        with self._session_factory() as session:
            narrator_turn = (
                session.query(Turn)
                .filter(Turn.campaign_id == campaign_id)
                .filter(Turn.kind == "narrator")
                .order_by(Turn.id.desc())
                .first()
            )
            if narrator_turn is None:
                return False
            narrator_turn.external_message_id = bot_id
            session.commit()
            return True

    # ------------------------------------------------------------------
    # Timer integration compatibility
    # ------------------------------------------------------------------

    def get_pending_timer_notice(
        self,
        campaign_id: str,
    ) -> str | None:
        with self._session_factory() as session:
            timer = (
                session.query(Timer)
                .filter_by(campaign_id=campaign_id)
                .filter(Timer.status.in_(["scheduled_unbound", "scheduled_bound"]))
                .order_by(Timer.created_at.desc())
                .first()
            )
            if timer is None:
                return None
            due_at = getattr(timer, "due_at", None)
            if due_at is None:
                return None
            try:
                import calendar as _calendar

                due_ts = int(_calendar.timegm(due_at.utctimetuple()))
            except Exception:
                return None
            event_text = str(getattr(timer, "event_text", "") or "").strip() or "Something happens."
            if bool(getattr(timer, "interruptible", True)):
                interrupt_action = str(getattr(timer, "interrupt_action", "") or "").strip()
                interrupt_hint = interrupt_action or "Act before it fires to avert it."
            else:
                interrupt_hint = "Unavoidable."
            return (
                f"⏰ *Timed event:* {event_text}\n"
                f"Fires <t:{due_ts}:F> (<t:{due_ts}:R>)\n"
                f"{interrupt_hint}"
            )

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

    def get_timed_events_speed_multiplier(self, campaign: Campaign | None) -> float:
        if campaign is None:
            return 1.0
        campaign_state = self.get_campaign_state(campaign)
        raw = campaign_state.get("timed_events_speed_multiplier", 1.0)
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

    def set_timed_events_speed_multiplier(self, campaign: Campaign | None, multiplier: float) -> bool:
        if campaign is None:
            return False
        multiplier = max(0.1, min(10.0, float(multiplier)))
        campaign_state = self.get_campaign_state(campaign)
        campaign_state["timed_events_speed_multiplier"] = multiplier
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

    def get_campaign_clock(self, campaign: Campaign | None) -> Dict[str, object]:
        if campaign is None:
            return {}
        campaign_state = self.get_campaign_state(campaign)
        snapshot = self._extract_game_time_snapshot(campaign_state)
        start_day_of_week = self._campaign_start_day_of_week(campaign_state)
        return self._game_time_from_total_minutes(
            self._game_time_to_total_minutes(snapshot),
            start_day_of_week=start_day_of_week,
        )

    def set_campaign_clock(
        self,
        campaign: Campaign | None,
        *,
        day: int,
        hour: int,
        minute: int = 0,
        day_of_week: object | None = None,
    ) -> Dict[str, object] | None:
        if campaign is None:
            return None
        resolved_day = max(1, self._coerce_non_negative_int(day, default=1) or 1)
        resolved_hour = min(23, max(0, self._coerce_non_negative_int(hour, default=0)))
        resolved_minute = min(59, max(0, self._coerce_non_negative_int(minute, default=0)))
        campaign_state = self.get_campaign_state(campaign)
        if day_of_week is not None and str(day_of_week).strip():
            requested_weekday = self._normalize_weekday_name(day_of_week)
            inferred_start = self._infer_start_day_of_week_from_game_time(
                {
                    "day": resolved_day,
                    "day_of_week": requested_weekday,
                }
            )
            if inferred_start:
                campaign_state[self.CLOCK_START_DAY_OF_WEEK_KEY] = inferred_start
        start_day_of_week = self._campaign_start_day_of_week(campaign_state)
        total_minutes = ((resolved_day - 1) * 24 * 60) + (resolved_hour * 60) + resolved_minute
        canonical = self._game_time_from_total_minutes(
            total_minutes,
            start_day_of_week=start_day_of_week,
        )
        campaign_state["game_time"] = canonical
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is None:
                return None
            row.state_json = self._dump_json(campaign_state)
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
            campaign.state_json = row.state_json
            campaign.updated_at = row.updated_at
        return canonical

    def get_campaign_clock_type(self, campaign: Campaign | None) -> str:
        if campaign is None:
            return "consequential-calendar"
        campaign_state = self.get_campaign_state(campaign)
        time_model = self._time_model_from_state(campaign_state)
        calendar_policy = self._calendar_policy_from_state(campaign_state)
        if time_model == self.TIME_MODEL_INDIVIDUAL_CLOCKS:
            return "individual-calendars"
        if calendar_policy == self.CALENDAR_POLICY_LOOSE:
            return "loose-calendar"
        return "consequential-calendar"

    def set_campaign_clock_type(
        self,
        campaign: Campaign | None,
        value: object,
    ) -> tuple[str | None, str | None]:
        if campaign is None:
            return None, "No campaign."
        choice, error = self._parse_setup_calendar_time_policy_choice(str(value or ""))
        if choice is None:
            return None, error or (
                "Reply with `loose-calendar`, `consequential-calendar`, or `individual-calendars`."
            )
        campaign_state = self.get_campaign_state(campaign)
        campaign_state["time_model"] = str(
            choice.get("time_model") or self.TIME_MODEL_SHARED_CLOCK
        )
        campaign_state["calendar_policy"] = str(
            choice.get("calendar_policy") or self.CALENDAR_POLICY_CONSEQUENTIAL
        )
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row is None:
                return None, "Campaign not found."
            row.state_json = self._dump_json(campaign_state)
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
            campaign.state_json = row.state_json
            campaign.updated_at = row.updated_at
        return self.get_campaign_clock_type(campaign), None

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

    def _resolve_style_direction(
        self,
        campaign: object = None,
    ) -> str:
        """Return the active style direction for a campaign.

        Checks campaign state for a ``style_direction`` override, falling
        back to ``DEFAULT_STYLE_DIRECTION``.
        """
        if campaign is not None:
            state = self.get_campaign_state(campaign)
            if isinstance(state, dict):
                override = str(state.get("style_direction") or "").strip()
                if override:
                    return override
        return self.DEFAULT_STYLE_DIRECTION

    @classmethod
    def _turn_response_style_note(
        cls,
        difficulty: object,
        *,
        style_direction: str = "",
    ) -> str:
        style = style_direction or cls.DEFAULT_STYLE_DIRECTION
        return cls._merge_system_notes(
            f"Style direction: {style}.",
            cls.RESPONSE_STYLE_NOTE,
            cls._difficulty_response_note(difficulty),
            (
                'Return final JSON only. Include reasoning first. '
                'state_update is required and must include "game_time", "current_chapter", and "current_scene" explicitly. '
                'summary_update is required — one sentence capturing any lasting change or the scene\'s current dramatic state.'
            ),
        )

    @classmethod
    def _turn_stage_note(
        cls,
        difficulty: object,
        prompt_stage: str,
        *,
        style_direction: str = "",
    ) -> str:
        stage = str(prompt_stage or cls.PROMPT_STAGE_FINAL).strip().lower()
        style = style_direction or cls.DEFAULT_STYLE_DIRECTION
        if stage == cls.PROMPT_STAGE_RESEARCH:
            return cls._merge_system_notes(
                f"Style direction: {style}.",
                cls._difficulty_response_note(difficulty),
                (
                    "RECENT_TURNS is loaded. Do not call recent_turns again this turn."
                ),
                (
                    "Use memory/source/SMS/planning tools only when they materially improve continuity."
                ),
                (
                    'When research is sufficient, return {"tool_call": "ready_to_write", "speakers": [...], "listeners": [...]} '
                    'with only the characters who will actually speak/act and the listeners who materially constrain shared knowledge. Do not narrate yet.'
                ),
                (
                    "No planning prose or self-talk in research phase. Do not write 'I need to...', 'Let me...', or explanation outside the JSON."
                ),
            )
        return cls._turn_response_style_note(
            difficulty,
            style_direction=style,
        )

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
                    f"✅ *Timer interrupted/averted - you acted in time. (Averted: {event})*",
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
        timer_ctx = self._pending_timers.get(campaign_id)
        if timer_ctx is None:
            return
        grace_seconds = 0.0
        if bool(timer_ctx.get("interruptible", True)):
            try:
                grace_seconds = max(
                    0.0,
                    float(getattr(self, "TIMER_INTERRUPT_GRACE_SECONDS", 0.0) or 0.0),
                )
            except (TypeError, ValueError):
                grace_seconds = 0.0
        if grace_seconds > 0:
            try:
                await asyncio.sleep(grace_seconds)
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

        self._set_timed_event_inflight(campaign_id, active_actor_id, event_description)
        try:
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
            if result.scene_image_prompt and channel_id:
                scene_ctx = SimpleNamespace(
                    author=SimpleNamespace(id=str(active_actor_id)),
                    channel=SimpleNamespace(id=str(channel_id)),
                )
                asyncio.create_task(
                    self._enqueue_scene_image(
                        scene_ctx,
                        str(result.scene_image_prompt),
                        campaign_id=campaign_id,
                    )
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
            # Embed the timed-event narrator turn for memory search.
            if self._memory_port is not None and narration:
                try:
                    with self._session_factory() as _embed_session:
                        last_turns = (
                            _embed_session.query(Turn)
                            .filter(Turn.campaign_id == campaign_id)
                            .filter(Turn.kind == "narrator")
                            .order_by(Turn.id.desc())
                            .limit(1)
                            .all()
                        )
                    if last_turns:
                        _embed_turn = last_turns[0]
                        _embed_meta = self._safe_turn_meta(_embed_turn)
                        _embed_visibility = _embed_meta.get("visibility")
                        self._memory_port.store_turn_embedding(
                            turn_id=int(_embed_turn.id),
                            campaign_id=campaign_id,
                            actor_id=active_actor_id,
                            kind="narrator",
                            content=narration,
                            metadata=self._turn_embedding_metadata(
                                visibility=_embed_visibility if isinstance(_embed_visibility, dict) else None,
                                actor_player_slug=_embed_meta.get("actor_player_slug"),
                                location_key=_embed_meta.get("location_key"),
                                session_id=None,
                            ),
                        )
                except Exception:
                    logger.debug(
                        "Timed-event turn embedding skipped for campaign=%s",
                        campaign_id,
                        exc_info=True,
                    )
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
        finally:
            self._clear_timed_event_inflight(campaign_id, active_actor_id)

    def _trim_text(self, text: str, max_chars: int) -> str:
        if text is None:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

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
        viewer_private_context_key: str = "",
        scene_npc_slugs: set[str] | None = None,
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
                if not self._turn_visible_in_recent_turns_context(
                    turn,
                    viewer_actor_id=viewer_actor_id,
                    viewer_slug=viewer_slug,
                    viewer_location_key=viewer_location_key,
                    viewer_private_context_key=viewer_private_context_key,
                ):
                    continue
                meta = self._safe_turn_meta(turn)
                if scene_npc_slugs and not self._turn_visible_to_all_scene_npcs(
                    turn,
                    scene_npc_slugs,
                ):
                    continue
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

    @classmethod
    def _strip_recent_turns_prompt_sections(cls, text: object) -> str:
        value = str(text or "")
        if not value:
            return ""
        lines = value.splitlines()
        kept: list[str] = []
        skipping_recent_body = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("RECENT_TURNS_LOADED:"):
                skipping_recent_body = False
                continue
            if stripped.startswith("RECENT_TURNS_NOTE:"):
                continue
            if stripped.startswith("RECENT_TURNS_LOCATIONS:"):
                continue
            if stripped.startswith("RECENT_TURNS_RECEIVERS:"):
                continue
            if stripped == "RECENT_TURNS:":
                skipping_recent_body = True
                continue
            if skipping_recent_body:
                if (
                    re.match(r"^[A-Z][A-Z0-9_ ]*:", stripped)
                    or stripped.startswith("PLAYER_ACTION ")
                ):
                    skipping_recent_body = False
                else:
                    continue
            kept.append(line)
        return "\n".join(kept).strip()

    @classmethod
    def _split_prompt_tail(cls, prompt: object) -> tuple[str, str]:
        value = str(prompt or "")
        if not value:
            return "", ""
        marker_index = value.rfind("\nPLAYER_ACTION ")
        if marker_index == -1 and value.startswith("PLAYER_ACTION "):
            marker_index = 0
        if marker_index == -1:
            return value.strip(), ""
        return value[:marker_index].rstrip(), value[marker_index:].strip()

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

    def _active_plot_threads_for_viewer(
        self,
        campaign_state: Dict[str, object],
        *,
        viewer_actor_id: str | None = None,
        viewer_slug: str | None = None,
        viewer_location_key: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, object]]:
        ranked_threads: list[dict[str, object]] = []
        for thread in self._plot_threads_from_state(campaign_state).values():
            if not isinstance(thread, dict):
                continue
            if str(thread.get("status") or "").strip().lower() != "active":
                continue
            if not self._plot_thread_visible_to_viewer(
                thread,
                viewer_actor_id=viewer_actor_id,
                viewer_slug=viewer_slug,
                viewer_location_key=viewer_location_key,
            ):
                continue
            ranked_threads.append(thread)
        ranked_threads.sort(
            key=lambda row: (
                -self._coerce_non_negative_int(row.get("updated_turn", 0), default=0),
                str(row.get("thread") or ""),
            )
        )
        out: list[dict[str, object]] = []
        for thread in ranked_threads[: max(1, int(limit or 8))]:
            out.append(
                {
                    "thread": thread.get("thread"),
                    "setup": thread.get("setup"),
                    "intended_payoff": thread.get("intended_payoff"),
                    "target_turns": thread.get("target_turns"),
                    "dependencies": list(thread.get("dependencies") or []),
                    "hint": thread.get("hint"),
                    "status": thread.get("status"),
                    "resolution": thread.get("resolution"),
                }
            )
        return out

    def _build_story_context(
        self,
        campaign_state: Dict[str, object],
        *,
        viewer_actor_id: str | None = None,
        viewer_slug: str | None = None,
        viewer_location_key: str | None = None,
    ) -> Optional[str]:
        if not bool(campaign_state.get("on_rails", False)):
            active_threads = self._active_plot_threads_for_viewer(
                campaign_state,
                viewer_actor_id=viewer_actor_id,
                viewer_slug=viewer_slug,
                viewer_location_key=viewer_location_key,
                limit=6,
            )
            if active_threads:
                lines: List[str] = ["ACTIVE SUBPLOTS:"]
                for row in active_threads[:6]:
                    thread = str(row.get("thread") or "untitled-thread").strip()
                    setup = str(row.get("setup") or "").strip()
                    payoff = str(row.get("intended_payoff") or "").strip()
                    hint = str(row.get("hint") or "").strip()
                    lines.append(f"- {thread}")
                    if setup:
                        lines.append(f"  Setup: {setup[:240]}")
                    if payoff:
                        lines.append(f"  Payoff: {payoff[:240]}")
                    if hint:
                        lines.append(f"  Hint: {hint[:220]}")
                return "\n".join(lines)
            return None

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

    def _strip_ephemeral_context_lines(self, text: str) -> str:
        if not text:
            return ""
        kept_lines: list[str] = []
        for line in str(text).splitlines():
            stripped = line.strip().lower()
            if any(stripped.startswith(prefix) for prefix in self._INVENTORY_LINE_PREFIXES):
                continue
            if any(stripped.startswith(prefix) for prefix in self._UNREAD_SMS_LINE_PREFIXES):
                continue
            if stripped.startswith("\u23f0"):
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip()

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

    def _normalize_inventory_entries(self, value) -> List[Dict[str, str]]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        if not isinstance(value, list):
            return []
        cleaned: List[Dict[str, str]] = []
        seen: set[str] = set()
        for item in value:
            name = self._item_to_text(item)
            if not name:
                continue
            normalized = name.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            origin = ""
            if isinstance(item, dict):
                origin = " ".join(str(item.get("origin") or "").strip().split())[:160]
            cleaned.append({"name": name, "origin": origin})
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
        adds: List[Dict[str, str] | str],
        removes: List[str],
        origin_hint: str = "",
        game_time: Dict[str, object] | None = None,
    ) -> List[Dict[str, str]]:
        remove_norm = {item.lower() for item in removes}
        out: List[Dict[str, str]] = []
        for entry in current:
            if entry["name"].lower() in remove_norm:
                continue
            out.append(entry)
        out_norm = {entry["name"].lower() for entry in out}
        for item in adds:
            if isinstance(item, dict):
                item_name = str(item.get("name") or "").strip()
                explicit_origin = str(item.get("origin") or "").strip()
            else:
                item_name = str(item or "").strip()
                explicit_origin = ""
            if not item_name:
                continue
            if item_name.lower() in out_norm:
                continue
            out.append(
                {
                    "name": item_name,
                    "origin": self._inventory_origin_with_receipt(
                        explicit_origin or origin_hint,
                        game_time,
                    ),
                }
            )
            out_norm.add(item_name.lower())
        return out

    def _inventory_receipt_stamp(self, game_time: object) -> str:
        if not isinstance(game_time, dict) or not game_time:
            return ""
        snapshot = self._extract_game_time_snapshot({"game_time": game_time})
        try:
            day = max(1, int(snapshot.get("day", 1) or 1))
            hour = min(23, max(0, int(snapshot.get("hour", 0) or 0)))
            minute = min(59, max(0, int(snapshot.get("minute", 0) or 0)))
        except (TypeError, ValueError):
            return ""
        return f"Day {day}, {hour:02d}:{minute:02d}"

    def _inventory_origin_with_receipt(
        self,
        origin_text: object,
        game_time: object,
    ) -> str:
        base = " ".join(str(origin_text or "").strip().split())
        receipt = self._inventory_receipt_stamp(game_time)
        if not receipt:
            return base
        if not base:
            return f"Received {receipt}"
        if re.search(r"\breceived\s+day\s+\d+\s*,\s*\d{2}:\d{2}\b", base, re.IGNORECASE):
            return base
        return f"{base} (received {receipt})"

    def _build_origin_hint(self, narration_text: str, action_text: str) -> str:
        # Default fallback provenance should be the in-world receipt timestamp,
        # not a raw fragment of action/narration text. Deliberate origins come
        # from explicit inventory_add objects or system-authored transfer hints.
        return ""

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

        inventory_add = self._normalize_inventory_entries(cleaned.pop("inventory_add", []))
        inventory_remove = self._normalize_inventory_items(cleaned.pop("inventory_remove", []))

        if "inventory" in cleaned:
            model_inventory = self._normalize_inventory_entries(cleaned.pop("inventory", []))
            model_set = {entry["name"].lower() for entry in model_inventory}
            current_names = [entry["name"] for entry in previous_inventory_rich]
            current_set = {name.lower() for name in current_names}
            for name in current_names:
                if name.lower() not in model_set and name.lower() not in {r.lower() for r in inventory_remove}:
                    inventory_remove.append(name)
            existing_add_names = {
                (
                    str(entry.get("name") or "").strip().lower()
                    if isinstance(entry, dict)
                    else str(entry or "").strip().lower()
                )
                for entry in inventory_add
            }
            for entry in model_inventory:
                name = entry["name"]
                if name.lower() not in current_set and name.lower() not in existing_add_names:
                    inventory_add.append(entry)
                    existing_add_names.add(name.lower())

        current_norm = {entry["name"].lower() for entry in previous_inventory_rich}
        inventory_remove = [item for item in inventory_remove if item.lower() in current_norm]

        if len(inventory_add) > self.MAX_INVENTORY_CHANGES_PER_TURN:
            inventory_add = inventory_add[: self.MAX_INVENTORY_CHANGES_PER_TURN]
        if len(inventory_remove) > self.MAX_INVENTORY_CHANGES_PER_TURN:
            inventory_remove = inventory_remove[: self.MAX_INVENTORY_CHANGES_PER_TURN]

        origin_hint = self._build_origin_hint(narration_text, action_text)
        effective_game_time = cleaned.get("game_time")
        if not isinstance(effective_game_time, dict) or not effective_game_time:
            effective_game_time = previous_state.get("game_time")
        if inventory_add or inventory_remove:
            cleaned["inventory"] = self._apply_inventory_delta(
                previous_inventory_rich,
                inventory_add,
                inventory_remove,
                origin_hint=origin_hint,
                game_time=effective_game_time if isinstance(effective_game_time, dict) else None,
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
        game_time: Dict[str, object] | None = None,
    ) -> Dict[str, dict]:
        if not isinstance(updates, dict):
            return existing
        location_stamp = None
        if isinstance(game_time, dict) and game_time:
            location_stamp = self._extract_game_time_snapshot({"game_time": game_time})
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
                old_location = str(existing[target_slug].get("location") or "").strip().lower()
                for key, value in fields.items():
                    if key not in self.IMMUTABLE_CHARACTER_FIELDS:
                        existing[target_slug][key] = value
                new_location_raw = str(existing[target_slug].get("location") or "").strip()
                new_location = new_location_raw.lower()
                if location_stamp and new_location_raw and old_location != new_location:
                    existing[target_slug]["location_last_updated"] = {
                        **location_stamp,
                        "loc": new_location_raw,
                    }
            else:
                if on_rails:
                    continue
                new_entry = dict(fields)
                new_entry.pop("location_last_updated", None)
                new_location_raw = str(new_entry.get("location") or "").strip()
                if location_stamp and new_location_raw:
                    new_entry["location_last_updated"] = {
                        **location_stamp,
                        "loc": new_location_raw,
                    }
                existing[slug] = new_entry
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
        *,
        player_state: dict[str, object] | None = None,
        viewer_actor_id: object = None,
    ) -> list[dict[str, object]]:
        game_time = campaign_state.get("game_time") if isinstance(campaign_state, dict) else {}
        if not isinstance(game_time, dict):
            game_time = {}
        time_model = cls._time_model_from_state(campaign_state)
        calendar_policy = cls._calendar_policy_from_state(campaign_state)
        global_day = cls._coerce_non_negative_int(game_time.get("day", 1), default=1) or 1
        global_hour = cls._coerce_non_negative_int(game_time.get("hour", 8), default=8)
        global_hour = min(23, max(0, global_hour))
        current_day = global_day
        current_hour = global_hour
        if time_model == cls.TIME_MODEL_INDIVIDUAL_CLOCKS and isinstance(player_state, dict):
            player_game_time = player_state.get("game_time")
            if isinstance(player_game_time, dict) and player_game_time:
                current_day = cls._coerce_non_negative_int(
                    player_game_time.get("day", global_day),
                    default=global_day,
                ) or global_day
                current_hour = cls._coerce_non_negative_int(
                    player_game_time.get("hour", global_hour),
                    default=global_hour,
                )
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
            event_targets_viewer = cls._calendar_event_targets_player(
                normalized,
                actor_id=viewer_actor_id,
                player_state=player_state,
            )
            if (
                time_model == cls.TIME_MODEL_SHARED_CLOCK
                and calendar_policy == cls.CALENDAR_POLICY_LOOSE
                and event_targets_viewer
            ):
                hours_remaining_before_roll = ((fire_day - current_day) * 24) + (
                    fire_hour - current_hour
                )
                if hours_remaining_before_roll < 0:
                    day_jump = max(1, ((-hours_remaining_before_roll) + 23) // 24)
                    fire_day += day_jump
                    calendar_changed = True
                if isinstance(raw, dict):
                    raw["fire_day"] = fire_day
                    raw["fire_hour"] = fire_hour
            hours_remaining = ((fire_day - current_day) * 24) + (fire_hour - current_hour)
            days_remaining = fire_day - current_day
            if (
                hours_remaining < 0
                and event_targets_viewer
                and calendar_policy == cls.CALENDAR_POLICY_CONSEQUENTIAL
            ):
                status = "missed"
            elif hours_remaining < 0:
                status = "overdue"
            elif days_remaining == 0:
                status = "today"
            elif hours_remaining <= 24:
                status = "imminent"
            else:
                status = "upcoming"
            view = dict(normalized)
            view["fire_day"] = fire_day
            view["fire_hour"] = fire_hour
            view["days_remaining"] = days_remaining
            view["hours_remaining"] = hours_remaining
            view["status"] = status
            if event_targets_viewer:
                view["targeted_to_active_player"] = True
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
    def _player_calendar_events_for_display(
        cls,
        player_state: dict[str, object] | None,
    ) -> list[str]:
        if not isinstance(player_state, dict):
            return []
        raw_events = player_state.get("calendar_events")
        if not isinstance(raw_events, list):
            return []

        lines: list[str] = []
        for raw in raw_events:
            if isinstance(raw, dict):
                title = str(
                    raw.get("title")
                    or raw.get("name")
                    or raw.get("event")
                    or "Untitled Event"
                ).strip()
                time_label = str(raw.get("time") or raw.get("when") or "").strip()
                location = str(raw.get("location") or raw.get("where") or "").strip()
                description = str(raw.get("description") or raw.get("summary") or "").strip()
                line = f"- **{title}**"
                details = [part for part in (time_label, location) if part]
                if details:
                    line += f" - {' | '.join(details)}"
                if description:
                    line += f" ({description})"
                lines.append(line)
                continue
            text = str(raw or "").strip()
            if text:
                lines.append(f"- **{text}**")
        return lines

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
            status = str(event.get("status") or "").strip().lower()
            if hours < 0:
                if status == "missed":
                    alerts.append(
                        f"- MISSED: {name} (was Day {fire_day}, {fire_hour:02d}:00; still unresolved)"
                    )
                else:
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
            owner_actor_id = str(raw_value.get("owner_actor_id") or "").strip() or None
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
                        "owner_actor_id": str(
                            msg.get("owner_actor_id") or owner_actor_id or ""
                        ).strip()
                        or None,
                    }
                )
            if messages:
                threads[key] = {
                    "label": label,
                    "messages": messages,
                    "owner_actor_id": owner_actor_id,
                }
        return threads

    @classmethod
    def _sms_storage_thread_key(
        cls,
        thread: object,
        *,
        owner_actor_id: object = None,
    ) -> str:
        base_key = cls._sms_normalize_thread_key(thread)
        if not base_key:
            base_key = "unknown"
        owner_text = str(owner_actor_id or "").strip()
        if not owner_text:
            return base_key
        owner_key = cls._sms_normalize_thread_key(f"actor-{owner_text}") or "actor-unknown"
        return f"{base_key}-{owner_key}"[:80]

    @classmethod
    def _sms_visible_messages_for_viewer(
        cls,
        row: Dict[str, object],
        *,
        viewer_actor_id: object = None,
        player_state: Dict[str, object] | None = None,
    ) -> List[Dict[str, object]]:
        raw_messages = row.get("messages")
        if not isinstance(raw_messages, list):
            return []
        if viewer_actor_id is None:
            return [dict(msg) for msg in raw_messages if isinstance(msg, dict)]

        owner_actor_id = str(row.get("owner_actor_id") or "").strip()
        viewer_actor_text = str(viewer_actor_id or "").strip()
        if owner_actor_id:
            if owner_actor_id != viewer_actor_text:
                return []
            return [dict(msg) for msg in raw_messages if isinstance(msg, dict)]

        aliases = cls._sms_player_aliases(
            actor_id=viewer_actor_id,
            player_state=player_state,
        )
        if not aliases:
            return []
        visible: List[Dict[str, object]] = []
        for raw_msg in raw_messages:
            if not isinstance(raw_msg, dict):
                continue
            from_norm = cls._sms_normalize_thread_key(raw_msg.get("from"))
            to_norm = cls._sms_normalize_thread_key(raw_msg.get("to"))
            if (from_norm and from_norm in aliases) or (to_norm and to_norm in aliases):
                visible.append(dict(raw_msg))
        return visible

    @classmethod
    def _sms_list_threads(
        cls,
        campaign_state: Dict[str, object],
        wildcard: str = "*",
        limit: int = 20,
        *,
        viewer_actor_id: object = None,
        player_state: Dict[str, object] | None = None,
        contact_roster: Dict[str, Dict[str, str]] | None = None,
    ) -> list[dict[str, object]]:
        threads = cls._sms_threads_from_state(campaign_state)
        pattern = str(wildcard or "*").strip().lower() or "*"
        merged: dict[str, dict[str, object]] = {}
        for key in reversed(list(threads.keys())):
            row = threads.get(key) or {}
            label = str(row.get("label") or key)
            messages = cls._sms_visible_messages_for_viewer(
                row,
                viewer_actor_id=viewer_actor_id,
                player_state=player_state,
            )
            if not messages:
                continue
            resolved_contact = cls._sms_resolved_contact(
                key,
                row,
                viewer_actor_id=viewer_actor_id,
                player_state=player_state,
                contact_roster=contact_roster,
                visible_messages=messages,
            )
            resolved_thread = str(resolved_contact.get("thread") or key).strip() or key
            resolved_label = str(resolved_contact.get("label") or label).strip() or label
            if pattern != "*":
                if (
                    not fnmatch.fnmatch(key, pattern)
                    and not fnmatch.fnmatch(label.lower(), pattern)
                    and not fnmatch.fnmatch(resolved_thread, pattern)
                    and not fnmatch.fnmatch(resolved_label.lower(), pattern)
                ):
                    continue
            last = messages[-1] if messages else {}
            preview = str(last.get("message") or "").strip()
            if len(preview) > cls.SMS_MAX_PREVIEW_CHARS:
                preview = preview[: cls.SMS_MAX_PREVIEW_CHARS - 1].rstrip() + "…"
            day = cls._coerce_non_negative_int(last.get("day", 0), default=0)
            hour = cls._coerce_non_negative_int(last.get("hour", 0), default=0)
            minute = cls._coerce_non_negative_int(last.get("minute", 0), default=0)
            turn_id = cls._coerce_non_negative_int(last.get("turn_id", 0), default=0)
            seq = cls._coerce_non_negative_int(last.get("seq", 0), default=0)
            sort_key = (day, hour, minute, turn_id, seq)

            existing = merged.get(resolved_thread)
            if existing is None:
                merged[resolved_thread] = {
                    "thread": resolved_thread,
                    "label": resolved_label,
                    "count": len(messages),
                    "last_from": str(last.get("from") or ""),
                    "last_preview": preview,
                    "day": day,
                    "hour": hour,
                    "minute": minute,
                    "_sort_key": sort_key,
                }
                continue

            existing["count"] = int(existing.get("count") or 0) + len(messages)
            existing_sort_key = existing.get("_sort_key")
            if not isinstance(existing_sort_key, tuple) or sort_key >= existing_sort_key:
                existing["label"] = resolved_label
                existing["last_from"] = str(last.get("from") or "")
                existing["last_preview"] = preview
                existing["day"] = day
                existing["hour"] = hour
                existing["minute"] = minute
                existing["_sort_key"] = sort_key

        out = sorted(
            merged.values(),
            key=lambda item: item.get("_sort_key", (0, 0, 0, 0, 0)),
            reverse=True,
        )[: max(1, int(limit or 20))]
        for row in out:
            row.pop("_sort_key", None)
        return out

    @classmethod
    def _sms_read_thread(
        cls,
        campaign_state: Dict[str, object],
        thread: str,
        limit: int = 20,
        *,
        viewer_actor_id: object = None,
        player_state: Dict[str, object] | None = None,
        contact_roster: Dict[str, Dict[str, str]] | None = None,
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
            visible_messages = cls._sms_visible_messages_for_viewer(
                row,
                viewer_actor_id=viewer_actor_id,
                player_state=player_state,
            )
            if viewer_actor_id is not None and not visible_messages:
                return False
            key_norm = cls._sms_normalize_thread_key(key)
            label_norm = cls._sms_normalize_thread_key(row.get("label"))
            resolved_contact = cls._sms_resolved_contact(
                key,
                row,
                viewer_actor_id=viewer_actor_id,
                player_state=player_state,
                contact_roster=contact_roster,
                visible_messages=visible_messages,
            )
            resolved_thread = cls._sms_normalize_thread_key(
                resolved_contact.get("thread")
            )
            resolved_label = cls._sms_normalize_thread_key(
                resolved_contact.get("label")
            )
            if query_key and (
                query_key == key_norm
                or query_key in key_norm
                or query_key == label_norm
                or query_key in label_norm
                or (resolved_thread and (query_key == resolved_thread or query_key in resolved_thread))
                or (resolved_label and (query_key == resolved_label or query_key in resolved_label))
            ):
                return True
            for msg in visible_messages:
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
            messages = cls._sms_visible_messages_for_viewer(
                row,
                viewer_actor_id=viewer_actor_id,
                player_state=player_state,
            )
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
        resolved_contact = cls._sms_resolved_contact(
            matched_keys[0],
            first_row,
            viewer_actor_id=viewer_actor_id,
            player_state=player_state,
            contact_roster=contact_roster,
        )
        base_label = str(
            resolved_contact.get("label") or first_row.get("label") or matched_keys[0]
        )
        if len(matched_keys) <= 1:
            resolved_label = base_label
        else:
            resolved_label = f"{base_label} (+{len(matched_keys) - 1} related thread(s))"
        canonical_contact = str(resolved_contact.get("thread") or "").strip()
        return (
            cls._sms_normalize_thread_key(canonical_contact or base_label) or canonical_key,
            resolved_label,
            list(capped),
        )

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
    def _sms_viewer_display_label(
        cls,
        *,
        actor_id: object,
        player_state: Dict[str, object] | None,
    ) -> str:
        if isinstance(player_state, dict):
            char_name = cls._sms_normalize_thread_key(player_state.get("character_name"))
            if char_name:
                return char_name
        actor_key = cls._sms_actor_key(actor_id)
        return actor_key or "player"

    @classmethod
    def _sms_counterpart_display_label(
        cls,
        row: Dict[str, object],
        *,
        actor_id: object,
        player_state: Dict[str, object] | None,
    ) -> str:
        aliases = cls._sms_player_aliases(
            actor_id=actor_id,
            player_state=player_state,
        )
        messages = cls._sms_visible_messages_for_viewer(
            row,
            viewer_actor_id=actor_id,
            player_state=player_state,
        )
        seen: set[str] = set()
        candidates: list[str] = []

        def _push(raw_value: object) -> None:
            norm = cls._sms_normalize_thread_key(raw_value)
            if not norm or norm in aliases or norm in seen:
                return
            seen.add(norm)
            candidates.append(norm)

        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            _push(msg.get("from"))
            _push(msg.get("to"))
            if len(candidates) > 1:
                break
        if len(candidates) == 1:
            return candidates[0]

        for raw_value in (row.get("label"), row.get("thread"), row.get("key")):
            norm = cls._sms_normalize_thread_key(raw_value)
            if norm and norm not in aliases:
                return norm
        return ""

    @classmethod
    def _sms_unread_thread_display_label(
        cls,
        key: str,
        row: Dict[str, object],
        *,
        actor_id: object,
        player_state: Dict[str, object] | None,
        contact_roster: Dict[str, Dict[str, str]] | None = None,
    ) -> str:
        viewer = cls._sms_viewer_display_label(
            actor_id=actor_id,
            player_state=player_state,
        )
        viewer_aliases = cls._sms_player_aliases(
            actor_id=actor_id,
            player_state=player_state,
        )
        resolved = cls._sms_resolved_contact(
            key,
            row,
            viewer_actor_id=actor_id,
            player_state=player_state,
            contact_roster=contact_roster,
        )
        resolved_counterpart = cls._sms_normalize_thread_key(
            resolved.get("label") or resolved.get("thread")
        )
        if (
            viewer
            and resolved_counterpart
            and resolved_counterpart not in viewer_aliases
            and resolved_counterpart != viewer
        ):
            return f"{viewer}↔{resolved_counterpart}"
        counterpart = cls._sms_counterpart_display_label(
            row,
            actor_id=actor_id,
            player_state=player_state,
        )
        if viewer and counterpart:
            return f"{viewer}↔{counterpart}"
        return counterpart or viewer

    @classmethod
    def _sms_roster_contact_for_alias(
        cls,
        contact_roster: Dict[str, Dict[str, str]] | None,
        alias: object,
    ) -> dict[str, str]:
        alias_key = cls._sms_normalize_thread_key(alias)
        if not alias_key or not isinstance(contact_roster, dict):
            return {}
        exact = dict(contact_roster.get(alias_key) or {})
        if exact:
            return exact

        matched_threads: dict[str, dict[str, str]] = {}
        prefix = f"{alias_key}-"
        for roster_key, roster_entry in contact_roster.items():
            if not isinstance(roster_entry, dict):
                continue
            roster_key_norm = cls._sms_normalize_thread_key(roster_key)
            if not roster_key_norm:
                continue
            if roster_key_norm != alias_key and not roster_key_norm.startswith(prefix):
                continue
            thread_text = cls._sms_normalize_thread_key(
                roster_entry.get("thread") or roster_key_norm
            )
            if not thread_text:
                continue
            label_text = str(roster_entry.get("label") or alias_key).strip()
            matched_threads[thread_text] = {
                "thread": thread_text,
                "label": label_text or thread_text,
            }
        if len(matched_threads) == 1:
            return next(iter(matched_threads.values()))
        return {}

    @classmethod
    def _sms_resolved_contact(
        cls,
        key: str,
        row: Dict[str, object],
        *,
        viewer_actor_id: object = None,
        player_state: Dict[str, object] | None = None,
        contact_roster: Dict[str, Dict[str, str]] | None = None,
        visible_messages: List[Dict[str, object]] | None = None,
    ) -> dict[str, str]:
        aliases = cls._sms_player_aliases(
            actor_id=viewer_actor_id,
            player_state=player_state,
        )
        messages = (
            list(visible_messages)
            if isinstance(visible_messages, list)
            else cls._sms_visible_messages_for_viewer(
                row,
                viewer_actor_id=viewer_actor_id,
                player_state=player_state,
            )
        )

        resolved_candidates: list[dict[str, str]] = []
        resolved_seen: set[str] = set()
        fallback_candidates: list[dict[str, str]] = []
        fallback_seen: set[str] = set()

        def _push(raw_value: object) -> None:
            raw_text = str(raw_value or "").strip()
            norm = cls._sms_normalize_thread_key(raw_text)
            if not norm or norm in aliases:
                return
            roster_entry = cls._sms_roster_contact_for_alias(contact_roster, norm)
            if roster_entry:
                thread_text = cls._sms_normalize_thread_key(
                    roster_entry.get("thread") or norm
                )
                label_text = str(roster_entry.get("label") or raw_text).strip()
                if thread_text and thread_text not in resolved_seen:
                    resolved_seen.add(thread_text)
                    resolved_candidates.append(
                        {"thread": thread_text, "label": label_text or thread_text}
                    )
                return
            if norm not in fallback_seen:
                fallback_seen.add(norm)
                fallback_candidates.append({"thread": norm, "label": raw_text or norm})

        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            _push(msg.get("from"))
            _push(msg.get("to"))

        if len(resolved_candidates) == 1:
            return resolved_candidates[0]
        if len(fallback_candidates) == 1:
            return fallback_candidates[0]

        for raw in (row.get("label"), key):
            raw_text = str(raw or "").strip()
            norm = cls._sms_normalize_thread_key(raw_text)
            if not norm or norm in aliases:
                continue
            roster_entry = cls._sms_roster_contact_for_alias(contact_roster, norm)
            if roster_entry:
                thread_text = cls._sms_normalize_thread_key(
                    roster_entry.get("thread") or norm
                )
                label_text = str(roster_entry.get("label") or raw_text).strip()
                if thread_text:
                    return {"thread": thread_text, "label": label_text or thread_text}

        label = str(row.get("label") or key).strip()
        return {
            "thread": cls._sms_normalize_thread_key(label) or key,
            "label": label or key,
        }

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
        contact_roster: Dict[str, Dict[str, str]] | None = None,
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
            label = cls._sms_unread_thread_display_label(
                thread_key,
                row,
                actor_id=actor_id,
                player_state=player_state,
                contact_roster=contact_roster,
            )
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
        contact_roster: Dict[str, Dict[str, str]] | None = None,
    ) -> str | None:
        if not isinstance(campaign_state, dict):
            return None
        summary = cls._sms_unread_summary_for_player(
            campaign_state,
            actor_id=actor_id,
            player_state=player_state,
            contact_roster=contact_roster,
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
        owner_actor_id: object = None,
    ) -> tuple[str, str, dict[str, object]]:
        threads = cls._sms_threads_from_state(campaign_state)
        owner_actor_text = str(owner_actor_id or "").strip() or None
        thread_key = cls._sms_storage_thread_key(
            thread or recipient or sender or "unknown",
            owner_actor_id=owner_actor_text,
        )
        existing = threads.pop(
            thread_key,
            {
                "label": thread or recipient or sender or thread_key,
                "messages": [],
                "owner_actor_id": owner_actor_text,
            },
        )
        label = str(existing.get("label") or thread or recipient or sender or thread_key).strip()[:80] or thread_key
        thread_owner_actor_id = (
            str(existing.get("owner_actor_id") or owner_actor_text or "").strip() or None
        )
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
            "owner_actor_id": thread_owner_actor_id,
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
                    threads[thread_key] = {
                        "label": label,
                        "messages": messages,
                        "owner_actor_id": thread_owner_actor_id,
                    }
                    campaign_state[cls.SMS_STATE_KEY] = threads
                    return thread_key, label, dict(last)
        next_seq = cls._coerce_non_negative_int(
            campaign_state.get(cls.SMS_MESSAGE_SEQ_KEY, 0), default=0
        ) + 1
        entry["seq"] = max(1, next_seq)
        messages.append(entry)
        messages = messages[-cls.SMS_MAX_MESSAGES_PER_THREAD :]
        threads[thread_key] = {
            "label": label,
            "messages": messages,
            "owner_actor_id": thread_owner_actor_id,
        }
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
        owner_actor_id: str | None = None,
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
                owner_actor_id=owner_actor_id,
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
        owner_actor_id: str | None = None,
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
                    owner_actor_id=str(owner_actor_id or "").strip() or None,
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
                    hour_alias = entry.get("hour")
                    if isinstance(hour_alias, (int, float)) and not isinstance(hour_alias, bool):
                        fire_hour = int(hour_alias)
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

    def _sanitize_autobiography_text(
        self,
        value: object,
        *,
        max_chars: int | None = None,
    ) -> str:
        text = " ".join(str(value or "").strip().split())
        if not text:
            return ""
        limit = max_chars if isinstance(max_chars, int) and max_chars > 0 else self.MAX_AUTOBIOGRAPHY_TEXT_CHARS
        return text[:limit].strip()

    def _normalize_autobiography_raw_entries(
        self,
        value: object,
    ) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        if not isinstance(value, list):
            return out
        for raw in value[-self.MAX_AUTOBIOGRAPHY_RAW_ENTRIES :]:
            if isinstance(raw, dict):
                a_text = self._sanitize_autobiography_text(
                    raw.get("a"),
                    max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
                )
                b_text = self._sanitize_autobiography_text(
                    raw.get("b"),
                    max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
                )
                c_text = self._sanitize_autobiography_text(
                    raw.get("c"),
                    max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
                )
                legacy_text = self._sanitize_autobiography_text(
                    raw.get("text"),
                    max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
                )
                if not (a_text or b_text or c_text or legacy_text):
                    continue
                row: dict[str, object] = {}
                if a_text:
                    row["a"] = a_text
                if b_text:
                    row["b"] = b_text
                if c_text:
                    row["c"] = c_text
                if legacy_text and not row:
                    row["text"] = legacy_text
                turn_id = self._coerce_int(raw.get("turn_id"), 0)
                if turn_id > 0:
                    row["turn_id"] = turn_id
                importance = " ".join(str(raw.get("importance") or "").strip().split())[:40]
                if importance:
                    row["importance"] = importance
                trigger = " ".join(str(raw.get("trigger") or "").strip().split())[:80]
                if trigger:
                    row["trigger"] = trigger
                game_time = raw.get("game_time")
                if isinstance(game_time, dict) and game_time:
                    row["game_time"] = {
                        "day": max(0, self._coerce_int(game_time.get("day"), 0)),
                        "hour": min(23, max(0, self._coerce_int(game_time.get("hour"), 0))),
                        "minute": min(59, max(0, self._coerce_int(game_time.get("minute"), 0))),
                    }
                out.append(row)
            elif isinstance(raw, str):
                text = self._sanitize_autobiography_text(
                    raw,
                    max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
                )
                if text:
                    out.append({"text": text})
        return out

    def _normalize_autobiography_update_payload(
        self,
        payload: object,
    ) -> list[dict[str, str]]:
        if not isinstance(payload, dict):
            return []
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raw_entries = [payload]
        normalized: list[dict[str, str]] = []
        for raw in raw_entries[:16]:
            if not isinstance(raw, dict):
                continue
            slug = str(
                raw.get("character")
                or raw.get("slug")
                or raw.get("npc")
                or ""
            ).strip()
            a_text = self._sanitize_autobiography_text(
                raw.get("a"),
                max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
            )
            b_text = self._sanitize_autobiography_text(
                raw.get("b"),
                max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
            )
            c_text = self._sanitize_autobiography_text(
                raw.get("c"),
                max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
            )
            legacy_text = self._sanitize_autobiography_text(
                raw.get("entry") or raw.get("text") or raw.get("autobiography"),
                max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
            )
            importance = " ".join(
                str(raw.get("importance") or "").strip().lower().split()
            )[:40]
            trigger = " ".join(
                str(raw.get("trigger") or "").strip().lower().split()
            )[:80]
            if not slug or not (a_text or b_text or c_text or legacy_text):
                continue
            normalized.append(
                {
                    "character": slug,
                    "a": a_text,
                    "b": b_text,
                    "c": c_text,
                    "text": legacy_text,
                    "importance": importance or "notable",
                    "trigger": trigger or "identity-threshold",
                }
            )
        return normalized

    def _apply_autobiography_update_to_characters(
        self,
        existing: dict[str, dict],
        payload: object,
        *,
        current_turn: int = 0,
        game_time: dict[str, object] | None = None,
    ) -> tuple[dict[str, dict], list[dict[str, object]]]:
        if not isinstance(existing, dict):
            existing = {}
        applied: list[dict[str, object]] = []
        for row in self._normalize_autobiography_update_payload(payload):
            raw_slug = row.get("character") or ""
            target_slug = self._resolve_existing_character_slug(existing, raw_slug) or raw_slug
            if target_slug not in existing:
                continue
            char = dict(existing.get(target_slug) or {})
            raw_entries = self._normalize_autobiography_raw_entries(
                char.get(self.AUTOBIOGRAPHY_RAW_FIELD)
            )
            entry_row: dict[str, object] = {
                "importance": row["importance"],
                "trigger": row["trigger"],
            }
            if row.get("a"):
                entry_row["a"] = row["a"]
            if row.get("b"):
                entry_row["b"] = row["b"]
            if row.get("c"):
                entry_row["c"] = row["c"]
            if row.get("text") and not any(entry_row.get(key) for key in ("a", "b", "c")):
                entry_row["text"] = row["text"]
            if current_turn > 0:
                entry_row["turn_id"] = int(current_turn)
            if isinstance(game_time, dict) and game_time:
                day = max(0, self._coerce_int(game_time.get("day"), 0))
                hour = min(23, max(0, self._coerce_int(game_time.get("hour"), 0)))
                minute = min(59, max(0, self._coerce_int(game_time.get("minute"), 0)))
                if day > 0 or hour > 0 or minute > 0:
                    entry_row["game_time"] = {"day": day, "hour": hour, "minute": minute}
            raw_entries.append(entry_row)
            raw_entries = raw_entries[-self.MAX_AUTOBIOGRAPHY_RAW_ENTRIES :]
            char[self.AUTOBIOGRAPHY_RAW_FIELD] = raw_entries
            existing[target_slug] = char
            applied.append(
                {
                    "character": target_slug,
                    "a": row.get("a") or "",
                    "b": row.get("b") or "",
                    "c": row.get("c") or "",
                    "entry": row.get("text") or "",
                    "trigger": row["trigger"],
                    "importance": row["importance"],
                    "raw_count": len(raw_entries),
                }
            )
        return existing, applied

    def _apply_autobiography_compress_to_characters(
        self,
        existing: dict[str, dict],
        payload: object,
        *,
        current_turn: int = 0,
    ) -> tuple[dict[str, dict], dict[str, object] | None]:
        if not isinstance(existing, dict) or not isinstance(payload, dict):
            return existing, None
        raw_slug = str(
            payload.get("character")
            or payload.get("slug")
            or payload.get("npc")
            or ""
        ).strip()
        text = self._sanitize_autobiography_text(
            payload.get("autobiography") or payload.get("text"),
            max_chars=self.MAX_AUTOBIOGRAPHY_TEXT_CHARS,
        )
        if not raw_slug or not text:
            return existing, None
        target_slug = self._resolve_existing_character_slug(existing, raw_slug) or raw_slug
        if target_slug not in existing:
            return existing, None
        char = dict(existing.get(target_slug) or {})
        char[self.AUTOBIOGRAPHY_FIELD] = text
        if current_turn > 0:
            char[self.AUTOBIOGRAPHY_LAST_COMPRESSED_TURN_FIELD] = int(current_turn)
        raw_entries = self._normalize_autobiography_raw_entries(
            char.get(self.AUTOBIOGRAPHY_RAW_FIELD)
        )
        if raw_entries:
            char[self.AUTOBIOGRAPHY_RAW_FIELD] = raw_entries[-self.MAX_AUTOBIOGRAPHY_RAW_ENTRIES :]
        existing[target_slug] = char
        return existing, {
            "character": target_slug,
            "autobiography": text,
            "raw_count": len(raw_entries),
            "last_compressed_turn": int(current_turn) if current_turn > 0 else 0,
        }

    def _autobiographies_for_prompt(
        self,
        characters_for_prompt: list[dict[str, object]],
        characters: Dict[str, dict] | None = None,
    ) -> str | None:
        rows: list[dict[str, str]] = []
        budget = self.MAX_AUTOBIOGRAPHY_PROMPT_CHARS
        characters = characters or {}
        for entry in characters_for_prompt or []:
            if not isinstance(entry, dict):
                continue
            slug = str(entry.get("_slug") or "").strip()
            if not slug:
                continue
            # Read autobiography from the original characters dict
            # (prompt entries have it stripped to keep WORLD_CHARACTERS lean).
            source = characters.get(slug)
            if not isinstance(source, dict):
                continue
            autobiography = self._sanitize_autobiography_text(
                source.get(self.AUTOBIOGRAPHY_FIELD),
                max_chars=self.MAX_AUTOBIOGRAPHY_TEXT_CHARS,
            )
            if not autobiography:
                continue
            row = {
                "slug": slug,
                "name": str(entry.get("name") or slug).strip(),
                "autobiography": autobiography,
            }
            line = json.dumps(row, ensure_ascii=True)
            if len(line) > budget:
                break
            rows.append(row)
            budget -= len(line) + 1
            if budget <= 0:
                break
        if not rows:
            return None
        return self._dump_json(rows)

    def _build_characters_for_prompt(
        self,
        characters: Dict[str, dict],
        player_state: Dict[str, object],
        recent_text: str,
        *,
        excluded_character_keys: set[str] | None = None,
        limit: int | None = None,
    ) -> list:
        if not characters:
            return []
        excluded_character_keys = {
            self._player_slug_key(value)
            for value in list(excluded_character_keys or set())
            if self._player_slug_key(value)
        }
        player_location = str(player_state.get("location") or "").strip().lower()
        recent_lower = recent_text.lower() if recent_text else ""
        hidden_prompt_keys = {
            self.AUTOBIOGRAPHY_FIELD,
            self.AUTOBIOGRAPHY_RAW_FIELD,
            self.AUTOBIOGRAPHY_LAST_COMPRESSED_TURN_FIELD,
            "evolving_personality",
        }

        nearby = []
        mentioned = []
        distant = []
        for slug, char in characters.items():
            if not isinstance(char, dict):
                continue
            slug_key = self._player_slug_key(slug)
            char_location = str(char.get("location") or "").strip().lower()
            char_name = str(char.get("name") or slug).strip().lower()
            char_name_key = self._player_slug_key(char.get("name") or slug)
            is_deceased = bool(char.get("deceased_reason"))
            if excluded_character_keys and (
                slug_key in excluded_character_keys or char_name_key in excluded_character_keys
            ):
                continue

            if not is_deceased and player_location and char_location == player_location:
                entry = {
                    key: value
                    for key, value in dict(char).items()
                    if key not in hidden_prompt_keys
                }
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
        if limit is None:
            return result
        return result[: limit]

    def _fit_characters_to_budget(self, characters_list: list, max_chars: int) -> list:
        while characters_list:
            text = json.dumps(characters_list, ensure_ascii=True)
            if len(text) <= max_chars:
                return characters_list
            characters_list = characters_list[:-1]
        return []

    def _fit_json_list_to_budget(self, rows: list[dict[str, object]], max_chars: int) -> list[dict[str, object]]:
        trimmed = list(rows or [])
        while trimmed:
            text = json.dumps(trimmed, ensure_ascii=True)
            if len(text) <= max_chars:
                return trimmed
            trimmed = trimmed[:-1]
        return []

    @staticmethod
    def _compact_prompt_fact_value(value: object, *, max_chars: int = 140) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = " ".join(value.strip().split())
        else:
            try:
                text = json.dumps(value, ensure_ascii=True, sort_keys=True)
            except Exception:
                text = str(value)
            text = " ".join(text.strip().split())
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return text

    def _character_birthday_hint_for_prompt(
        self,
        current_day: int,
        character_row: object,
    ) -> str | None:
        if not isinstance(character_row, dict):
            return None
        created = character_row.get("created")
        if not isinstance(created, dict):
            return None
        created_day = self._coerce_non_negative_int(created.get("day"), default=0)
        if created_day <= 0:
            return None
        if current_day < created_day:
            return None
        days_in_game = current_day - created_day
        if days_in_game % self.DAYS_PER_GAME_YEAR != 0:
            return None
        return "It is this character's birthday today."

    def _character_field_priority(self, field_name: str) -> str:
        key = str(field_name or "").strip().lower()
        if key in {
            "name",
            "age",
            "gender",
            "location",
            "location_last_updated",
            "current_status",
            "speech_style",
            "relationship",
            "relationships",
            "allegiance",
            "birthday_hint",
            "deceased_reason",
        }:
            return "critical"
        if key in {"appearance", "personality", "background", "autobiography", "literary_style"}:
            return "scene"
        return "low"

    def _character_scene_relevance(
        self,
        character_row: Dict[str, object] | None,
        *,
        player_state: Dict[str, object] | None = None,
    ) -> bool:
        if not isinstance(character_row, dict):
            return False
        if character_row.get("deceased_reason"):
            return False
        player_state = player_state or {}
        char_state = {
            "location": character_row.get("location"),
            "room_title": character_row.get("room_title"),
            "room_summary": character_row.get("room_summary"),
            "room_id": character_row.get("room_id"),
        }
        return self._same_scene(player_state, char_state)

    def _location_field_priority(self, field_name: str) -> str:
        key = str(field_name or "").strip().lower()
        if key in {"name", "summary", "security", "current_activity"}:
            return "critical"
        if key in {"layout", "notable_objects", "recent_change", "social_rules", "atmosphere", "exits"}:
            return "scene"
        return "low"

    def _stored_location_field_priority(
        self,
        location_row: Dict[str, object] | None,
        field_name: str,
    ) -> str:
        key = str(field_name or "").strip()
        raw_map = (
            location_row.get(self.LOCATION_FACT_PRIORITIES_KEY)
            if isinstance(location_row, dict)
            else None
        )
        if isinstance(raw_map, dict):
            raw_value = raw_map.get(key)
            text = re.sub(r"[^a-z]+", "-", str(raw_value or "").strip().lower()).strip("-")
            if text in {"critical", "always", "sticky", "persistent"}:
                return "critical"
            if text in {"scene", "active", "local"}:
                return "scene"
            if text in {"low", "minor", "ephemeral", "temporary"}:
                return "low"
        return self._location_field_priority(field_name)

    def _build_character_index_for_prompt(
        self,
        characters_for_prompt: list[dict[str, object]],
        *,
        characters: Dict[str, dict] | None = None,
        campaign_state: Dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        characters = characters or {}
        hidden_keys = {
            self.AUTOBIOGRAPHY_FIELD,
            self.AUTOBIOGRAPHY_RAW_FIELD,
            self.AUTOBIOGRAPHY_LAST_COMPRESSED_TURN_FIELD,
            "evolving_personality",
            "created",
            "compact",
            "expanded",
            "priority",
        }
        current_day = self._coerce_non_negative_int(
            self._extract_game_time_snapshot(campaign_state or {}).get("day"),
            default=0,
        )
        for entry in characters_for_prompt or []:
            if not isinstance(entry, dict):
                continue
            slug = str(entry.get("_slug") or "").strip()
            if not slug:
                continue
            source = characters.get(slug) if isinstance(characters.get(slug), dict) else entry
            birthday_hint = self._character_birthday_hint_for_prompt(
                current_day,
                source,
            )
            if birthday_hint:
                source = dict(source)
                source["birthday_hint"] = birthday_hint
            available_keys = sorted(
                key
                for key in (source or {}).keys()
                if key not in hidden_keys and not str(key).startswith("_")
            )
            critical: dict[str, object] = {}
            for key in available_keys:
                if key in {"name", "location", "current_status"}:
                    continue
                if self._character_field_priority(key) != "critical":
                    continue
                value = source.get(key)
                if value in (None, "", [], {}):
                    continue
                critical[key] = value
            rows.append(
                {
                    "slug": slug,
                    "name": str(entry.get("name") or slug).strip(),
                    "location": source.get("location"),
                    "location_last_updated": source.get("location_last_updated"),
                    "current_status": source.get("current_status"),
                    "available_keys": available_keys,
                    "critical": critical,
                }
            )
        return rows

    def _build_world_characters_for_prompt(
        self,
        characters_for_prompt: list[dict[str, object]],
        *,
        characters: Dict[str, dict] | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        characters = characters or {}
        for entry in characters_for_prompt or []:
            if not isinstance(entry, dict):
                continue
            slug = str(entry.get("_slug") or "").strip()
            if not slug:
                continue
            source = characters.get(slug) if isinstance(characters.get(slug), dict) else entry
            row = {
                "slug": slug,
                "name": str((source or {}).get("name") or entry.get("name") or slug).strip(),
                "location": (source or {}).get("location"),
                "current_status": (source or {}).get("current_status"),
            }
            rows.append(row)
        return rows

    def _build_character_cards_for_prompt(
        self,
        characters_for_prompt: list[dict[str, object]],
        *,
        characters: Dict[str, dict] | None = None,
        campaign_state: Dict[str, object] | None = None,
        player_state: Dict[str, object] | None = None,
        include_critical_fields: bool = False,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        characters = characters or {}
        player_location = str((player_state or {}).get("location") or "").strip().lower()
        hidden_keys = {
            self.AUTOBIOGRAPHY_FIELD,
            self.AUTOBIOGRAPHY_RAW_FIELD,
            self.AUTOBIOGRAPHY_LAST_COMPRESSED_TURN_FIELD,
            "evolving_personality",
            "created",
            "relationship",
            "compact",
            "expanded",
            "priority",
        }
        top_level_keys = {"name", "location", "location_last_updated", "current_status"}
        current_day = self._coerce_non_negative_int(
            self._extract_game_time_snapshot(campaign_state or {}).get("day"),
            default=0,
        )
        for entry in characters_for_prompt or []:
            if not isinstance(entry, dict):
                continue
            slug = str(entry.get("_slug") or "").strip()
            if not slug:
                continue
            source = characters.get(slug)
            if not isinstance(source, dict):
                source = dict(entry)
            birthday_hint = self._character_birthday_hint_for_prompt(
                current_day,
                source,
            )
            if birthday_hint:
                source = dict(source)
                source["birthday_hint"] = birthday_hint
            available_keys = [
                key
                for key in source.keys()
                if key not in hidden_keys and not str(key).startswith("_")
            ]
            priorities = {
                key: self._character_field_priority(key)
                for key in available_keys
            }
            compact: dict[str, object] = {}
            for key in available_keys:
                if key in top_level_keys:
                    continue
                priority = priorities.get(key, "low")
                if priority == "critical" and not include_critical_fields:
                    continue
                compact_value = self._compact_prompt_fact_value(source.get(key))
                if compact_value:
                    compact[key] = compact_value
            char_location = str(source.get("location") or "").strip().lower()
            expand_scene_fields = bool(player_location and char_location and char_location == player_location)
            expanded: dict[str, object] = {}
            for key in available_keys:
                if key in top_level_keys:
                    continue
                priority = priorities.get(key, "low")
                if (
                    (include_critical_fields and priority == "critical")
                    or (priority == "scene" and expand_scene_fields)
                ):
                    value = source.get(key)
                    if value in (None, "", [], {}):
                        continue
                    expanded[key] = value
            rows.append(
                {
                    "slug": slug,
                    "name": str(source.get("name") or entry.get("name") or slug).strip(),
                    "location": source.get("location"),
                    "location_last_updated": source.get("location_last_updated"),
                    "current_status": source.get("current_status"),
                    "available_keys": sorted(available_keys),
                    "compact": compact,
                    "expanded": expanded,
                }
            )
        return rows

    def _player_character_prompt_keys(
        self,
        party_snapshot: list[dict[str, object]] | None,
        player_registry: dict[str, dict[str, dict[str, object]]] | None = None,
    ) -> set[str]:
        keys: set[str] = set()
        for row in party_snapshot or []:
            if not isinstance(row, dict):
                continue
            for value in (
                row.get("player_slug"),
                row.get("slug"),
                row.get("name"),
                row.get("character_name"),
            ):
                key = self._player_slug_key(value)
                if key:
                    keys.add(key)
        if isinstance(player_registry, dict):
            for info in (player_registry.get("by_actor_id", {}) or {}).values():
                if not isinstance(info, dict):
                    continue
                for value in (info.get("slug"), info.get("name")):
                    key = self._player_slug_key(value)
                    if key:
                        keys.add(key)
        return keys

    def _build_scene_characters_for_prompt(
        self,
        characters: Dict[str, dict],
        player_state: Dict[str, object],
        *,
        excluded_character_keys: set[str] | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        excluded_character_keys = {
            self._player_slug_key(value)
            for value in list(excluded_character_keys or set())
            if self._player_slug_key(value)
        }
        hidden_prompt_keys = {
            self.AUTOBIOGRAPHY_FIELD,
            self.AUTOBIOGRAPHY_RAW_FIELD,
            self.AUTOBIOGRAPHY_LAST_COMPRESSED_TURN_FIELD,
            "evolving_personality",
        }
        for slug, char in characters.items():
            if not isinstance(char, dict):
                continue
            slug_key = self._player_slug_key(slug)
            name_key = self._player_slug_key(char.get("name") or slug)
            if excluded_character_keys and (
                slug_key in excluded_character_keys or name_key in excluded_character_keys
            ):
                continue
            if not self._character_scene_relevance(char, player_state=player_state):
                continue
            entry = {
                key: value
                for key, value in dict(char).items()
                if key not in hidden_prompt_keys
            }
            entry["_slug"] = slug
            rows.append(entry)
        return rows

    def _location_cards_from_state(
        self,
        campaign_state: Dict[str, object],
        player_state: Dict[str, object] | None = None,
    ) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        if isinstance(campaign_state.get(self.LOCATION_CARDS_STATE_KEY), dict):
            for slug, payload in (campaign_state.get(self.LOCATION_CARDS_STATE_KEY) or {}).items():
                if isinstance(payload, dict):
                    out[str(slug)] = dict(payload)
        player_state = player_state or {}
        player_location = str(player_state.get("location") or "").strip()
        if player_location:
            row = dict(out.get(player_location) or {})
            row.setdefault("name", str(player_state.get("room_title") or player_location).strip())
            row.setdefault("summary", str(player_state.get("room_summary") or "").strip())
            if str(player_state.get("room_description") or "").strip():
                row.setdefault("description", str(player_state.get("room_description") or "").strip())
            if isinstance(player_state.get("exits"), list) and player_state.get("exits"):
                row.setdefault("exits", list(player_state.get("exits") or []))
            out[player_location] = row
        return out

    def _build_location_index_for_prompt(
        self,
        location_cards: Dict[str, dict],
        *,
        player_state: Dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        active_location = str((player_state or {}).get("location") or "").strip().lower()
        for slug, payload in sorted(location_cards.items()):
            if not isinstance(payload, dict):
                continue
            is_active_location = slug.lower() == active_location
            available_keys = sorted(
                key
                for key in payload.keys()
                if key != self.LOCATION_FACT_PRIORITIES_KEY and not str(key).startswith("_")
            )
            row = {
                "slug": slug,
                "name": str(payload.get("name") or slug).strip(),
            }
            if not is_active_location:
                row["summary"] = self._compact_prompt_fact_value(
                    payload.get("summary") or payload.get("description"),
                    max_chars=120,
                )
                row["available_keys"] = available_keys
            rows.append(row)
        return rows

    def _build_location_cards_for_prompt(
        self,
        location_cards: Dict[str, dict],
        *,
        player_state: Dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        active_location = str((player_state or {}).get("location") or "").strip().lower()
        top_level_keys = {"name", "summary"}
        suppressed_card_keys = {"exits"}
        for slug, payload in sorted(location_cards.items()):
            if not isinstance(payload, dict):
                continue
            available_keys = [
                key
                for key in payload.keys()
                if key != self.LOCATION_FACT_PRIORITIES_KEY and not str(key).startswith("_")
            ]
            is_active_location = slug.lower() == active_location
            compact: dict[str, object] = {}
            for key in available_keys:
                if key in top_level_keys or key in suppressed_card_keys:
                    continue
                priority = self._stored_location_field_priority(payload, key)
                if priority != "critical" and not is_active_location:
                    continue
                compact_value = self._compact_prompt_fact_value(payload.get(key))
                if compact_value:
                    compact[key] = compact_value
            expanded: dict[str, object] = {}
            for key in available_keys:
                if key in top_level_keys or key in suppressed_card_keys:
                    continue
                priority = self._stored_location_field_priority(payload, key)
                if priority != "critical" and not is_active_location:
                    continue
                value = payload.get(key)
                if value in (None, "", [], {}):
                    continue
                expanded[key] = value
            row = {
                "slug": slug,
                "name": str(payload.get("name") or slug).strip(),
                "summary": self._compact_prompt_fact_value(
                    payload.get("summary") or payload.get("description"),
                    max_chars=120,
                ),
                "available_keys": sorted(available_keys),
            }
            for key in list(expanded.keys()):
                compact.pop(key, None)
            for key, value in expanded.items():
                row[key] = value
            if compact:
                row["compact"] = compact
            rows.append(row)
        return rows

    def _derive_scene_active_tensions(
        self,
        present_characters: list[dict[str, object]],
        *,
        campaign_state: Dict[str, object] | None = None,
        location_key: str = "",
    ) -> list[dict[str, str]]:
        tensions: list[dict[str, str]] = []
        seen_keys: set[str] = set()

        def _add(slug: str, tension_text: object, source: str) -> None:
            tension = self._compact_prompt_fact_value(tension_text, max_chars=120)
            clean_slug = str(slug or "").strip()
            if not clean_slug or not tension:
                return
            dedupe_key = f"{clean_slug}|{tension}"
            if dedupe_key in seen_keys:
                return
            seen_keys.add(dedupe_key)
            tensions.append(
                {
                    "slug": clean_slug,
                    "tension": tension,
                    "source": source,
                }
            )

        for row in present_characters:
            if not isinstance(row, dict):
                continue
            slug = str(row.get("slug") or "").strip()
            if not slug:
                continue
            source = str(row.get("relationship") or row.get("current_status") or "").strip()
            if not source:
                continue
            _add(slug, source, "character")
        if isinstance(campaign_state, dict) and present_characters:
            present_slugs = {
                str(row.get("slug") or "").strip().lower()
                for row in present_characters
                if isinstance(row, dict) and str(row.get("slug") or "").strip()
            }
            location_norm = str(location_key or "").strip().lower()
            excluded = set(self.MODEL_STATE_EXCLUDE_KEYS) | {self.LOCATION_CARDS_STATE_KEY}
            for state_key, payload in campaign_state.items():
                if state_key in excluded or not isinstance(payload, dict):
                    continue
                raw_refs: list[str] = []
                for ref_key in (
                    "character_slug",
                    "npc_slug",
                    "target_slug",
                    "source_slug",
                    "location_slug",
                ):
                    value = payload.get(ref_key)
                    if isinstance(value, str) and value.strip():
                        raw_refs.append(value.strip().lower())
                for ref_key in ("characters", "participants", "known_by", "npc_slugs"):
                    value = payload.get(ref_key)
                    if isinstance(value, list):
                        for item in value:
                            text = str(item or "").strip().lower()
                            if text:
                                raw_refs.append(text)
                if not any(ref in present_slugs for ref in raw_refs) and (
                    not location_norm or location_norm not in raw_refs
                ):
                    continue
                detail = (
                    payload.get("tension")
                    or payload.get("dynamic")
                    or payload.get("status")
                    or payload.get("summary")
                    or payload.get("description")
                    or payload.get("reason")
                    or payload.get("fact")
                )
                if not detail:
                    detail = payload
                matched_slug = next((ref for ref in raw_refs if ref in present_slugs), "")
                if matched_slug:
                    _add(matched_slug, detail, f"world_state:{state_key}")
        return tensions[:6]

    def _build_scene_state_for_prompt(
        self,
        *,
        campaign_state: Dict[str, object],
        player_state: Dict[str, object],
        party_snapshot: list[dict[str, object]],
        characters: Dict[str, dict],
        location_cards: Dict[str, dict],
        excluded_character_keys: set[str] | None = None,
    ) -> dict[str, object]:
        location_key = str(player_state.get("location") or "").strip()
        excluded_character_keys = {
            self._player_slug_key(value)
            for value in list(excluded_character_keys or set())
            if self._player_slug_key(value)
        }
        present_characters: list[dict[str, object]] = []
        for slug, payload in (characters or {}).items():
            if not isinstance(payload, dict):
                continue
            slug_key = self._player_slug_key(slug)
            name_key = self._player_slug_key(payload.get("name") or slug)
            if excluded_character_keys and (
                slug_key in excluded_character_keys or name_key in excluded_character_keys
            ):
                continue
            if str(payload.get("location") or "").strip().lower() != location_key.lower():
                continue
            if payload.get("deceased_reason"):
                continue
            present_characters.append(
                {
                    "slug": str(slug),
                    "name": str(payload.get("name") or slug).strip(),
                    "current_status": payload.get("current_status"),
                    "relationship": payload.get("relationship"),
                }
            )
        present_players: list[dict[str, object]] = []
        for row in party_snapshot or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("location") or "").strip().lower() != location_key.lower():
                continue
            present_players.append(
                {
                    "slug": str(row.get("slug") or row.get("player_slug") or "").strip(),
                    "name": str(row.get("character_name") or row.get("name") or "").strip(),
                    "game_time": row.get("game_time") if isinstance(row.get("game_time"), dict) else None,
                }
            )
        location_row = location_cards.get(location_key) if isinstance(location_cards, dict) else {}
        scene_location_facts = {}
        if isinstance(location_row, dict):
            for key in ("summary", "security", "current_activity", "atmosphere"):
                value = location_row.get(key)
                if value not in (None, "", [], {}):
                    scene_location_facts[key] = value
        atmosphere = (
            str(location_row.get("atmosphere") or "").strip()
            if isinstance(location_row, dict)
            else ""
        ) or str(player_state.get("room_summary") or "").strip()
        return {
            "location_key": location_key,
            "context_key": str((self._active_private_context_from_state(player_state) or {}).get("context_key") or "").strip(),
            "atmosphere": atmosphere,
            "present_characters": present_characters,
            "present_players": present_players,
            "visible_objects": [],
            "active_tensions": self._derive_scene_active_tensions(
                present_characters,
                campaign_state=campaign_state,
                location_key=location_key,
            ),
            "scene_location_facts": scene_location_facts,
        }

    def _normalize_persisted_entity_state(
        self,
        campaign: Campaign,
        campaign_state: Dict[str, object],
        characters: Dict[str, dict],
        player_state: Dict[str, object],
    ) -> tuple[Dict[str, object], Dict[str, dict], int]:
        normalized_state, character_updates, location_updates, consumed = (
            self._engine._relocate_entity_state_updates(
                campaign_state,
                campaign_state=campaign_state,
                existing_chars=characters,
                player_state=player_state,
            )
        )
        if not consumed:
            return campaign_state, characters, 0
        normalized_characters = self._engine._apply_character_updates(
            characters,
            character_updates,
            on_rails=bool(campaign_state.get("on_rails")),
            game_time=self._extract_game_time_snapshot(normalized_state),
        )
        existing_locations = {}
        raw_locations = normalized_state.get(self.LOCATION_CARDS_STATE_KEY)
        if isinstance(raw_locations, dict):
            existing_locations = dict(raw_locations)
        normalized_locations = self._engine._apply_location_updates(
            existing_locations,
            location_updates,
            on_rails=bool(campaign_state.get("on_rails")),
        )
        if normalized_locations:
            normalized_state[self.LOCATION_CARDS_STATE_KEY] = normalized_locations
        else:
            normalized_state.pop(self.LOCATION_CARDS_STATE_KEY, None)
        self._persist_campaign_prompt_state(
            campaign.id,
            state=normalized_state,
            characters=normalized_characters,
            campaign=campaign,
        )
        self._zork_log(
            "PERSISTED ENTITY STATE NORMALIZED",
            f"Relocated {consumed} character/location world-state key(s) before prompt build.",
        )
        return normalized_state, normalized_characters, consumed

    def _persist_campaign_prompt_state(
        self,
        campaign_id: str,
        *,
        state: Dict[str, object] | None = None,
        characters: Dict[str, dict] | None = None,
        campaign: Campaign | None = None,
    ) -> None:
        if campaign is not None:
            if state is not None:
                campaign.state_json = self._dump_json(state)
            if characters is not None:
                campaign.characters_json = self._dump_json(characters)
        with self._session_factory() as session:
            row = session.get(Campaign, str(campaign_id))
            if row is None:
                return
            if state is not None:
                row.state_json = self._dump_json(state)
            if characters is not None:
                row.characters_json = self._dump_json(characters)
            session.commit()

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
        if not isinstance(payload, dict):
            return False
        tool_val = payload.get("tool_call")
        if not tool_val or not isinstance(tool_val, str) or not tool_val.strip():
            return False
        return "narration" not in payload

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
            # Step 1b: apply common syntax repairs then retry.
            repaired = self._repair_json_lenient_text(text)
            if repaired != text:
                try:
                    result = json.loads(repaired)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass
            coerced = self._coerce_python_dict(text)
            if coerced is not None:
                return coerced
            if repaired != text:
                coerced = self._coerce_python_dict(repaired)
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
                    scene_output = parsed.get("scene_output")
                    has_scene_output = (
                        isinstance(scene_output, dict)
                        and isinstance(scene_output.get("beats"), list)
                    )
                    has_turn_payload = any(
                        parsed.get(key)
                        for key in (
                            "summary_update",
                            "state_update",
                            "player_state_update",
                            "scene_image_prompt",
                        )
                    )
                    if has_narration or has_tool_call or has_scene_output or has_turn_payload:
                        return repaired
            except Exception:
                pass
        json_text = self._extract_json(cleaned)
        if json_text:
            return json_text
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
            "participants",
            "participant",
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
        state_dirty = False
        if self.CLOCK_START_DAY_OF_WEEK_KEY not in state:
            state[self.CLOCK_START_DAY_OF_WEEK_KEY] = "monday"
            state_dirty = True
        if "game_time" not in state:
            state["game_time"] = self._game_time_from_total_minutes(
                ((1 - 1) * 24 * 60) + (8 * 60),
                start_day_of_week=self._campaign_start_day_of_week(state),
            )
            state_dirty = True
        else:
            canonical_game_time = self._game_time_from_total_minutes(
                self._game_time_to_total_minutes(state.get("game_time") or {}),
                start_day_of_week=self._campaign_start_day_of_week(state),
            )
            if state.get("game_time") != canonical_game_time:
                state["game_time"] = canonical_game_time
                state_dirty = True
        if state_dirty:
            self._persist_campaign_prompt_state(
                campaign.id,
                state=state,
                campaign=campaign,
            )
        guardrails_enabled = bool(state.get("guardrails_enabled", False))
        attributes = self.get_player_attributes(player)
        player_state = self.get_player_state(player)
        characters = self.get_campaign_characters(campaign)
        state, characters, normalized_entity_keys = self._normalize_persisted_entity_state(
            campaign,
            state,
            characters,
            player_state,
        )
        model_state = self._build_model_state(state)
        model_state = self._fit_state_to_budget(model_state, self.MAX_STATE_CHARS)
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
        player_character_keys = self._player_character_prompt_keys(
            party_snapshot,
            player_registry=player_registry,
        )
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
        active_pc = self._active_private_context_from_state(player_state)
        viewer_private_context_key = str(
            (active_pc or {}).get("context_key") or ""
        ).strip()
        summary = self._compose_world_summary(
            campaign,
            state,
            turns=turns,
            viewer_actor_id=player.actor_id,
            viewer_slug=viewer_slug,
            viewer_location_key=viewer_location_key,
            viewer_private_context_key=viewer_private_context_key,
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
            meta = self._safe_turn_meta(turn)
            scene_output = meta.get("scene_output")
            if isinstance(scene_output, dict) and scene_output.get("beats"):
                jsonl_lines = self._scene_output_recent_lines(
                    turn,
                    state,
                    scene_output,
                    viewer_actor_id=player.actor_id,
                    viewer_slug=viewer_slug,
                    viewer_location_key=viewer_location_key,
                    viewer_private_context_key=viewer_private_context_key,
                )
                if jsonl_lines and turn.kind == "narrator":
                    recent_lines.extend(jsonl_lines)
                    continue

            if turn.kind == "player":
                if ooc_re.match(content):
                    continue
                clipped = self._strip_inventory_mentions(content)
                if not clipped:
                    continue
                name = player_names.get(turn.actor_id or "")
                recent_lines.extend(
                    self._recent_turn_fallback_lines(
                        turn,
                        state,
                        content_text=clipped,
                        player_name=name,
                    )
                )
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
                recent_lines.extend(
                    self._recent_turn_fallback_lines(
                        turn,
                        state,
                        content_text=clipped,
                    )
                )
        recent_text = "\n".join(recent_lines) if recent_lines else "None"

        rails_context = self._build_rails_context(player_state, party_snapshot)
        active_plot_threads = self._active_plot_threads_for_viewer(
            state,
            viewer_actor_id=player.actor_id,
            viewer_slug=viewer_slug,
            viewer_location_key=viewer_location_key,
            limit=8,
        )
        active_plot_hints = self._plot_hints_for_viewer(
            state,
            viewer_actor_id=player.actor_id,
            viewer_slug=viewer_slug,
            viewer_location_key=viewer_location_key,
            limit=6,
        )
        character_index_source = self._build_characters_for_prompt(
            characters,
            player_state,
            recent_text,
            excluded_character_keys=player_character_keys,
            limit=None,
        )
        character_index_source = self._fit_json_list_to_budget(
            character_index_source,
            self.MAX_CHARACTER_INDEX_CHARS,
        )
        scene_characters_for_prompt = self._build_scene_characters_for_prompt(
            characters,
            player_state,
            excluded_character_keys=player_character_keys,
        )
        scene_characters_for_prompt = scene_characters_for_prompt[: self.MAX_CHARACTERS_IN_PROMPT]
        scene_characters_for_prompt = self._fit_characters_to_budget(
            scene_characters_for_prompt,
            self.MAX_CHARACTERS_CHARS,
        )
        character_index = self._build_character_index_for_prompt(
            character_index_source,
            characters=characters,
            campaign_state=state,
        )
        world_characters = self._build_world_characters_for_prompt(
            character_index_source,
            characters=characters,
        )
        character_cards = self._build_character_cards_for_prompt(
            scene_characters_for_prompt,
            characters=characters,
            campaign_state=state,
            player_state=player_state,
            include_critical_fields=False,
        )
        location_cards_map = self._location_cards_from_state(state, player_state)
        location_index = self._build_location_index_for_prompt(
            location_cards_map,
            player_state=player_state,
        )
        location_index = self._fit_json_list_to_budget(
            location_index,
            self.MAX_LOCATION_INDEX_CHARS,
        )
        location_cards = self._build_location_cards_for_prompt(
            location_cards_map,
            player_state=player_state,
        )
        scene_state = self._build_scene_state_for_prompt(
            campaign_state=state,
            player_state=player_state,
            party_snapshot=party_snapshot,
            characters=characters,
            location_cards=location_cards_map,
            excluded_character_keys=player_character_keys,
        )
        on_rails = bool(state.get("on_rails", False))
        story_context = self._build_story_context(
            state,
            viewer_actor_id=player.actor_id,
            viewer_slug=viewer_slug,
            viewer_location_key=viewer_location_key,
        )
        active_scene_names = self._active_scene_character_names(
            player_state,
            party_snapshot,
            scene_characters_for_prompt,
        )
        game_time, global_game_time, time_model, calendar_policy = self._current_game_time_for_prompt(
            state,
            player_state,
        )
        speed_mult = state.get("speed_multiplier", 1.0)
        effective_min_turn_advance = self._effective_min_turn_advance_minutes(speed_mult)
        effective_standard_turn_advance = self._effective_standard_turn_advance_minutes(
            speed_mult
        )
        turn_time_beat_guidance = self._turn_time_beat_guidance(
            effective_min_turn_advance
        )
        difficulty = self.normalize_difficulty(state.get("difficulty", "normal"))
        style_direction = self._resolve_style_direction(campaign)
        response_style_note = self._turn_stage_note(difficulty, stage, style_direction=style_direction)
        calendar_state_before = json.dumps(
            state.get("calendar") or [],
            ensure_ascii=True,
            sort_keys=True,
        )
        calendar_for_prompt = self._calendar_for_prompt(
            state,
            player_state=player_state,
            viewer_actor_id=player.actor_id,
        )
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
            self._persist_campaign_prompt_state(
                campaign.id,
                state=state,
                campaign=campaign,
            )
        source_payload = self._source_material_prompt_payload(campaign.id)
        literary_styles_text = self._literary_styles_for_prompt(
            state,
            scene_characters_for_prompt,
        )
        autobiographies_text = self._autobiographies_for_prompt(
            scene_characters_for_prompt,
            characters=characters,
        )
        memory_lookup_enabled = self._memory_lookup_enabled_for_prompt(
            campaign.summary or "",
            source_material_available=bool(source_payload.get("available")),
            action_text=action,
        )

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
            )
            source_digests = source_payload.get("digests") or {}
            if source_digests:
                for digest_key, digest_text in source_digests.items():
                    user_prompt += (
                        f"SOURCE_MATERIAL_DIGEST [{digest_key}]:\n{digest_text}\n"
                    )
        user_prompt += (
            f"CURRENT_GAME_TIME: {self._dump_json(game_time)}\n"
            + (
                f"GLOBAL_GAME_TIME: {self._dump_json(global_game_time)}\n"
                if time_model == self.TIME_MODEL_INDIVIDUAL_CLOCKS
                else ""
            )
            + (
                f"TIME_MODEL: {time_model}\n"
                f"CALENDAR_POLICY: {calendar_policy}\n"
            )
            + f"SPEED_MULTIPLIER: {speed_mult}\n"
            f"MIN_TURN_ADVANCE_MINUTES_EFFECTIVE: {effective_min_turn_advance}\n"
            f"STANDARD_TURN_ADVANCE_MINUTES_EFFECTIVE: {effective_standard_turn_advance}\n"
            f"TURN_TIME_BEAT_GUIDANCE: {turn_time_beat_guidance}\n"
            f"DIFFICULTY: {difficulty}\n"
            f"MEMORY_LOOKUP_ENABLED: {str(memory_lookup_enabled).lower()}\n"
            f"RECENT_TURNS_LOADED: {str(not bootstrap_only).lower()}\n"
        )
        user_prompt += (
            f"SCENE_STATE: {self._dump_json(scene_state)}\n"
            f"CHARACTER_INDEX: {self._dump_json(character_index)}\n"
            f"CHARACTER_CARDS: {self._dump_json(character_cards)}\n"
            f"LOCATION_INDEX: {self._dump_json(location_index)}\n"
            f"LOCATION_CARDS: {self._dump_json(location_cards)}\n"
            f"WORLD_CHARACTERS: {self._dump_json(world_characters)}\n"
            f"PLAYER_CARD: {self._dump_json(player_card)}\n"
            f"PARTY_SNAPSHOT: {self._dump_json(party_snapshot)}\n"
        )
        if literary_styles_text:
            user_prompt += f"LITERARY_STYLES:\n{literary_styles_text}\n"
        if autobiographies_text:
            user_prompt += f"AUTOBIOGRAPHIES: {autobiographies_text}\n"
        _puzzle_text = self._puzzle_system_for_prompt(state)
        if _puzzle_text:
            user_prompt += f"{_puzzle_text}\n"
        if not bootstrap_only:
            if story_context:
                user_prompt += f"STORY_CONTEXT:\n{story_context}\n"
            user_prompt += (
                f"WORLD_SUMMARY: {summary}\n"
                f"WORLD_STATE: {self._dump_json(model_state)}\n"
                f"ACTIVE_PLOT_THREADS: {self._dump_json(active_plot_threads)}\n"
                f"ACTIVE_HINTS: {self._dump_json(active_plot_hints)}\n"
                f"CALENDAR: {self._dump_json(calendar_for_prompt)}\n"
                f"CALENDAR_REMINDERS:\n{calendar_reminders}\n"
                f"RECENT_TURNS:\n{recent_text}\n"
            )
        merged_tail_extra_lines = list(tail_extra_lines or [])
        merged_tail_extra_lines.extend(
            self._ooc_prompt_extra_lines(
                campaign,
                state,
                player,
                player_state,
                action,
            )
        )
        merged_tail_extra_lines.extend(
            self._passive_npc_sms_nudge_lines(
                campaign,
                state,
                player,
                player_state,
            )
        )
        turn_prompt_tail = self._build_turn_prompt_tail(
            player_state,
            action,
            response_style_note,
            turn_attachment_context=turn_attachment_context,
            extra_lines=merged_tail_extra_lines,
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
            system_prompt = f"{system_prompt}{self.SONG_SEARCH_TOOL_PROMPT}"
            if state.get("timed_events_enabled", True):
                system_prompt = f"{system_prompt}{self.TIMER_TOOL_PROMPT}"
            if on_rails and story_context:
                system_prompt = f"{system_prompt}{self.STORY_OUTLINE_TOOL_PROMPT}"
            if not on_rails:
                system_prompt = f"{system_prompt}{self.CHAPTER_PLAN_TOOL_PROMPT}"
            system_prompt = f"{system_prompt}{self.PLOT_PLAN_TOOL_PROMPT}"
            system_prompt = f"{system_prompt}{self.CONSEQUENCE_TOOL_PROMPT}"
            system_prompt = f"{system_prompt}{self.CALENDAR_TOOL_PROMPT}"
            system_prompt = f"{system_prompt}{self.ROSTER_PROMPT}"
            system_prompt = f"{system_prompt}{self.AUTOBIOGRAPHY_TOOL_PROMPT}"
            system_prompt = f"{system_prompt}{self.READY_TO_WRITE_TOOL_PROMPT}"
        else:
            system_prompt = f"{self.SYSTEM_PROMPT}{self.FINAL_STAGE_OPERATIONAL_PROMPT}"
            if guardrails_enabled:
                system_prompt = f"{system_prompt}{self.GUARDRAILS_SYSTEM_PROMPT}"
            if on_rails:
                system_prompt = f"{system_prompt}{self.ON_RAILS_SYSTEM_PROMPT}"
            if state.get("timed_events_enabled", True):
                system_prompt = f"{system_prompt}{self.TIMER_TOOL_PROMPT}"
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
        if isinstance(landmarks, list) and landmarks:
            parts = []
            for lm in landmarks:
                if isinstance(lm, dict):
                    name = str(lm.get("name") or "")
                    role = str(lm.get("role") or "")
                    parts.append(f"{name} ({role})" if name and role else name or str(lm))
                else:
                    parts.append(str(lm))
            landmarks_text = ", ".join(parts)
        else:
            landmarks_text = "none"

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

    def _sms_contact_roster(
        self,
        campaign: Campaign,
    ) -> dict[str, dict[str, str]]:
        roster: dict[str, dict[str, str]] = {}

        def _register(alias: object, *, thread: object, label: object) -> None:
            alias_key = self._sms_normalize_thread_key(alias)
            thread_key = self._sms_normalize_thread_key(thread)
            label_text = str(label or "").strip()
            if not alias_key or not thread_key:
                return
            roster[alias_key] = {
                "thread": thread_key,
                "label": label_text or thread_key,
            }

        registry = self._campaign_player_registry(campaign.id, self._session_factory)
        for entry in registry.get("by_actor_id", {}).values():
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            slug = str(entry.get("slug") or "").strip()
            thread_key = self._sms_normalize_thread_key(name) or self._sms_normalize_thread_key(slug)
            if not thread_key:
                continue
            _register(name, thread=thread_key, label=name or thread_key)
            _register(slug, thread=thread_key, label=name or thread_key)

        characters = self.get_campaign_characters(campaign)
        if isinstance(characters, dict):
            for slug, payload in characters.items():
                name = (
                    str(payload.get("name") or slug or "").strip()
                    if isinstance(payload, dict)
                    else str(slug or "").strip()
                )
                thread_key = self._sms_normalize_thread_key(slug) or self._sms_normalize_thread_key(name)
                if not thread_key:
                    continue
                _register(slug, thread=thread_key, label=name or thread_key)
                _register(name, thread=thread_key, label=name or thread_key)

        return roster

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
        viewer_actor_id: str | None = None,
    ) -> list[dict[str, object]]:
        with self._session_factory() as session:
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return []
            state = self.get_campaign_state(campaign)
            contact_roster = self._sms_contact_roster(campaign)
            player = None
            if viewer_actor_id:
                player = (
                    session.query(Player)
                    .filter(Player.campaign_id == str(campaign_id))
                    .filter(Player.actor_id == str(viewer_actor_id))
                    .first()
                )
            player_state = self.get_player_state(player) if player is not None else {}
        return self._sms_list_threads(
            state,
            wildcard=wildcard,
            limit=limit,
            viewer_actor_id=viewer_actor_id,
            player_state=player_state,
            contact_roster=contact_roster,
        )

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
            contact_roster = self._sms_contact_roster(campaign)
            player = None
            if viewer_actor_id:
                player = (
                    session.query(Player)
                    .filter(Player.campaign_id == str(campaign_id))
                    .filter(Player.actor_id == str(viewer_actor_id))
                    .first()
                )
                player_state = self.get_player_state(player) if player is not None else {}
            else:
                player_state = {}
            canonical, label, messages = self._sms_read_thread(
                state,
                thread=thread,
                limit=limit,
                viewer_actor_id=viewer_actor_id,
                player_state=player_state,
                contact_roster=contact_roster,
            )
            if viewer_actor_id:
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
        owner_actor_id: str | None = None,
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
                owner_actor_id=owner_actor_id,
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

    # ------------------------------------------------------------------
    # Private context management
    # ------------------------------------------------------------------

    @staticmethod
    def _private_context_key(*parts: object) -> str:
        cleaned = []
        for part in parts:
            text = re.sub(r"[^a-z0-9]+", "-", str(part or "").strip().lower()).strip("-")
            if text:
                cleaned.append(text[:80])
        return ":".join(cleaned)[:240]

    @classmethod
    def _active_private_context_from_state(
        cls, player_state: dict[str, object]
    ) -> dict[str, object] | None:
        if not isinstance(player_state, dict):
            return None
        raw = player_state.get(cls.PRIVATE_CONTEXT_STATE_KEY)
        if not isinstance(raw, dict):
            return None
        context_key = str(raw.get("context_key") or "").strip()
        scope = str(raw.get("scope") or "").strip().lower()
        if not context_key or scope not in {"private", "limited"}:
            return None
        out = dict(raw)
        out["context_key"] = context_key
        out["scope"] = scope
        return out

    def _action_leaves_private_context(
        self,
        action: str,
        active_context: dict[str, object] | None,
    ) -> bool:
        text = " ".join(str(action or "").strip().lower().split())
        if not text or not active_context:
            return False
        if hasattr(self, "_is_private_phone_command_line") and self._is_private_phone_command_line(text):
            return False
        if re.search(
            r"\b(?:go|walk|head|return|leave|exit|join|approach|cross|back to|turn back to|out loud|to everyone|announce)\b",
            text,
            re.IGNORECASE,
        ):
            return True
        target_name = str(active_context.get("target_name") or "").strip().lower()
        if target_name and target_name not in text and re.search(r"\b(?:ask|tell|say|talk)\b", text, re.IGNORECASE):
            return True
        return False

    def _resolve_private_context_target(
        self,
        campaign: "Campaign",
        actor: "Player | None",
        action: str,
    ) -> dict[str, object] | None:
        text = str(action or "")
        if not text:
            return None
        actor_id = getattr(actor, "actor_id", None)
        registry = self._campaign_player_registry(campaign.id)
        by_actor_id = registry.get("by_actor_id", {})
        text_norm = self._normalize_match_text(text)
        for aid, entry in by_actor_id.items():
            if actor_id is not None and str(aid) == str(actor_id):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            if self._normalize_match_text(name) in text_norm:
                return {
                    "kind": "player",
                    "target_actor_id": str(aid),
                    "target_slug": str(entry.get("slug") or "").strip(),
                    "target_name": name,
                }
        characters = self.get_campaign_characters(campaign)
        if isinstance(characters, dict):
            for slug, payload in characters.items():
                if not isinstance(payload, dict):
                    continue
                name = str(payload.get("name") or "").strip()
                candidates = [str(slug or "").strip(), name]
                for candidate in candidates:
                    candidate_norm = self._normalize_match_text(candidate)
                    if candidate_norm and candidate_norm in text_norm:
                        return {
                            "kind": "npc",
                            "target_slug": str(slug or "").strip(),
                            "target_name": name or str(slug or "").strip(),
                        }
        return None

    def _derive_private_context_candidate(
        self,
        campaign: "Campaign",
        actor: "Player | None",
        player_state: dict[str, object],
        action: str,
    ) -> dict[str, object] | None:
        active_context = self._active_private_context_from_state(player_state)
        if self._action_leaves_private_context(action, active_context):
            return None
        if active_context:
            carried = dict(active_context)
            carried["engagement"] = "continue"
            return carried
        return None

    @classmethod
    def _apply_private_context_candidate(
        cls,
        turn_visibility: dict[str, object],
        candidate: dict[str, object] | None,
    ) -> dict[str, object]:
        if not candidate:
            return turn_visibility
        merged = dict(turn_visibility or {})
        merged["scope"] = str(candidate.get("scope") or merged.get("scope") or "private")
        merged["context_key"] = str(candidate.get("context_key") or "").strip() or None
        location_key = str(candidate.get("location_key") or "").strip()
        if location_key:
            merged["location_key"] = location_key
        reason = " ".join(str(merged.get("reason") or "").split()).strip()
        if not reason:
            target_name = str(candidate.get("target_name") or "").strip()
            if target_name:
                merged["reason"] = f"Private exchange with {target_name}"
            else:
                merged["reason"] = "Private exchange"
        if merged.get("scope") == "limited":
            visible_slugs = list(merged.get("visible_player_slugs") or [])
            target_slug = str(candidate.get("target_slug") or "").strip()
            if target_slug and target_slug not in visible_slugs:
                visible_slugs.append(target_slug)
            merged["visible_player_slugs"] = visible_slugs
            visible_actor_ids = list(merged.get("visible_actor_ids") or [])
            target_actor_id = candidate.get("target_actor_id")
            if target_actor_id is not None and str(target_actor_id) not in [str(x) for x in visible_actor_ids]:
                visible_actor_ids.append(str(target_actor_id))
            merged["visible_actor_ids"] = visible_actor_ids
        return merged

    def _persist_private_context_state(
        self,
        player_state: dict[str, object],
        turn_visibility: dict[str, object],
        action: str,
        candidate: dict[str, object] | None,
    ) -> None:
        if not isinstance(player_state, dict):
            return
        active_context = self._active_private_context_from_state(player_state)
        scope = str(turn_visibility.get("scope") or "").strip().lower()
        context_key = str(turn_visibility.get("context_key") or "").strip()
        if context_key and scope in {"private", "limited"}:
            payload = {
                "scope": scope,
                "context_key": context_key,
                "location_key": str(turn_visibility.get("location_key") or "").strip() or None,
                "target_name": str((candidate or {}).get("target_name") or (active_context or {}).get("target_name") or "").strip() or None,
                "target_slug": str((candidate or {}).get("target_slug") or (active_context or {}).get("target_slug") or "").strip() or None,
            }
            player_state[self.PRIVATE_CONTEXT_STATE_KEY] = payload
            return
        if self._action_leaves_private_context(action, active_context) or scope in {"public", "local"}:
            player_state.pop(self.PRIVATE_CONTEXT_STATE_KEY, None)

    def _recent_private_contexts_for_prompt(
        self,
        turns: list,
        *,
        viewer_actor_id: str,
        viewer_slug: str,
        active_context_key: str = "",
        limit: int = 3,
    ) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        seen: set[str] = set()
        viewer_slug_key = self._player_slug_key(viewer_slug)
        active_context_key = str(active_context_key or "").strip()
        for turn in reversed(list(turns or [])):
            meta = self._safe_turn_meta(turn)
            visibility = meta.get("visibility")
            if not isinstance(visibility, dict):
                continue
            scope = str(visibility.get("scope") or "").strip().lower()
            if scope not in {"private", "limited"}:
                continue
            context_key = str(
                visibility.get("context_key") or meta.get("context_key") or ""
            ).strip()
            if not context_key or context_key in seen:
                continue
            actor_actor_id = visibility.get("actor_actor_id") or visibility.get("actor_user_id")
            actor_player_slug = self._player_slug_key(
                visibility.get("actor_player_slug") or ""
            )
            visible_actor_ids: list[str] = []
            for item in list(visibility.get("visible_actor_ids") or visibility.get("visible_user_ids") or []):
                if item is not None:
                    visible_actor_ids.append(str(item))
            visible_player_slugs = [
                self._player_slug_key(item)
                for item in list(visibility.get("visible_player_slugs") or [])
                if self._player_slug_key(item)
            ]
            if not (
                (actor_actor_id is not None and str(actor_actor_id) == str(viewer_actor_id))
                or (str(viewer_actor_id) in visible_actor_ids)
                or (viewer_slug_key and viewer_slug_key == actor_player_slug)
                or (viewer_slug_key and viewer_slug_key in visible_player_slugs)
            ):
                continue
            row = {
                "context_key": context_key,
                "scope": scope,
                "location_key": str(visibility.get("location_key") or meta.get("location_key") or "").strip() or None,
                "actor_player_slug": actor_player_slug or None,
                "visible_player_slugs": visible_player_slugs,
                "visible_actor_ids": visible_actor_ids,
                "aware_npc_slugs": [
                    str(item or "").strip()
                    for item in list(visibility.get("aware_npc_slugs") or [])
                    if str(item or "").strip()
                ],
                "active": bool(active_context_key and context_key == active_context_key),
                "turn_id": int(getattr(turn, "id", 0) or 0),
            }
            out.append(row)
            seen.add(context_key)
            if len(out) >= max(1, int(limit)):
                break
        if active_context_key and active_context_key not in seen:
            out.insert(
                0,
                {
                    "context_key": active_context_key,
                    "scope": "private",
                    "location_key": None,
                    "actor_player_slug": viewer_slug_key or None,
                    "visible_player_slugs": [viewer_slug_key] if viewer_slug_key else [],
                    "visible_actor_ids": [str(viewer_actor_id)],
                    "aware_npc_slugs": [],
                    "active": True,
                    "turn_id": None,
                },
            )
        return out[: max(1, int(limit))]

    @staticmethod
    def _humanize_context_key(value: object) -> str:
        return " ".join(part for part in str(value or "").replace(":", " ").replace("-", " ").split())

    def _format_private_context_status(
        self,
        player_state: dict[str, object],
        *,
        recent_contexts: list[dict[str, object]] | None = None,
    ) -> str:
        active_context = self._active_private_context_from_state(player_state)
        if isinstance(active_context, dict):
            scope = str(active_context.get("scope") or "").strip().lower() or "private"
            target_name = str(active_context.get("target_name") or "").strip()
            label = "limited" if scope == "limited" else "private"
            if target_name:
                return (
                    f"Private context: {label} thread active with {target_name}. "
                    "Keep talking to continue it, or move/rejoin/speak out loud to leave it."
                )
            return (
                f"Private context: {label} thread active. "
                "Keep talking to continue it, or move/rejoin/speak out loud to leave it."
            )
        recent_labels: list[str] = []
        for row in list(recent_contexts or [])[:3]:
            if not isinstance(row, dict):
                continue
            label = self._humanize_context_key(row.get("context_key"))
            if not label or label in recent_labels:
                continue
            recent_labels.append(label)
        if recent_labels:
            return (
                f"Private context: none active. Recent threads: {', '.join(recent_labels)}. "
                "Use a slash command or reaction to start or resume a private thread, or use phone/text actions."
            )
        return (
            "Private context: none. To start one, use a slash command or reaction, "
            "or use phone/text actions."
        )

    def _fallback_private_context_from_recent(
        self,
        turns: list,
        *,
        viewer_actor_id: str,
        viewer_slug: str,
        viewer_location_key: str,
        limit: int = 6,
    ) -> dict[str, object] | None:
        viewer_location_key_norm = self._normalize_location_key(viewer_location_key)
        if not viewer_location_key_norm:
            return None
        recent_contexts = self._recent_private_contexts_for_prompt(
            turns,
            viewer_actor_id=viewer_actor_id,
            viewer_slug=viewer_slug,
            active_context_key="",
            limit=limit,
        )
        for row in recent_contexts:
            if not isinstance(row, dict):
                continue
            row_location_key = self._normalize_location_key(row.get("location_key"))
            if row_location_key != viewer_location_key_norm:
                continue
            scope = str(row.get("scope") or "").strip().lower()
            context_key = str(row.get("context_key") or "").strip()
            if scope not in {"private", "limited"} or not context_key:
                continue
            return {
                "scope": scope,
                "context_key": context_key,
                "location_key": str(row.get("location_key") or "").strip() or None,
                "target_name": None,
                "target_slug": None,
                "target_actor_id": None,
                "engagement": "resume",
            }
        return None

    @classmethod
    def _fallback_private_context_from_rows(
        cls,
        recent_contexts: list[dict[str, object]],
        *,
        viewer_location_key: str,
    ) -> dict[str, object] | None:
        viewer_location_key_norm = cls._normalize_location_key(viewer_location_key)
        if not viewer_location_key_norm:
            return None
        for row in list(recent_contexts or []):
            if not isinstance(row, dict):
                continue
            row_location_key = cls._normalize_location_key(row.get("location_key"))
            if row_location_key != viewer_location_key_norm:
                continue
            scope = str(row.get("scope") or "").strip().lower()
            context_key = str(row.get("context_key") or "").strip()
            if scope not in {"private", "limited"} or not context_key:
                continue
            return {
                "scope": scope,
                "context_key": context_key,
                "location_key": str(row.get("location_key") or "").strip() or None,
                "target_name": None,
                "target_slug": None,
                "target_actor_id": None,
                "engagement": "resume",
            }
        return None

    def _force_private_visibility_for_phone_activity(
        self,
        visibility: dict[str, object],
        *,
        actor_slug: str,
        actor_id: str | None,
    ) -> dict[str, object]:
        reason = self._trim_text(str(visibility.get("reason") or "").strip(), 240)
        return {
            "scope": "private",
            "actor_player_slug": actor_slug or None,
            "actor_actor_id": actor_id,
            "visible_player_slugs": [actor_slug] if actor_slug else [],
            "visible_actor_ids": [actor_id] if actor_id is not None else [],
            "location_key": None,
            "aware_npc_slugs": [],
            "reason": reason or "Private phone/SMS activity is actor-only unless explicitly shared.",
            "source": "auto-private-phone",
        }

    def _recent_private_dm_notification_targets(
        self,
        campaign_id: str,
        *,
        exclude_actor_id: str | None = None,
        observed_at: "datetime.datetime | None" = None,
        candidate_actor_ids: list[str] | None = None,
    ) -> list[str]:
        now_dt = observed_at or self._now()
        if now_dt.tzinfo is not None:
            now_dt = now_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        allowed_actor_ids = (
            {str(aid) for aid in candidate_actor_ids if aid is not None}
            if isinstance(candidate_actor_ids, list)
            else None
        )
        with self._session_factory() as session:
            from text_game_engine.persistence.sqlalchemy.models import Player
            rows = session.query(Player).filter_by(campaign_id=str(campaign_id)).all()
            out: list[str] = []
            for row in rows:
                row_actor_id = str(getattr(row, "actor_id", ""))
                if exclude_actor_id is not None and row_actor_id == str(exclude_actor_id):
                    continue
                if allowed_actor_ids is not None and row_actor_id not in allowed_actor_ids:
                    continue
                stats = self.get_player_statistics(row)
                if (
                    str(stats.get(self.PLAYER_STATS_LAST_MESSAGE_CONTEXT_KEY) or "").strip().lower()
                    != "dm"
                ):
                    continue
                last_message_at = self._parse_utc_timestamp(
                    stats.get(self.PLAYER_STATS_LAST_MESSAGE_AT_KEY)
                )
                if last_message_at is None:
                    continue
                age_seconds = int((now_dt - last_message_at).total_seconds())
                if age_seconds < 0 or age_seconds > self.ATTENTION_WINDOW_SECONDS:
                    continue
                out.append(row_actor_id)
        return out

    # ------------------------------------------------------------------
    # Plot / Chapter / Consequence management
    # ------------------------------------------------------------------

    @classmethod
    def _plot_thread_key(cls, value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:80]

    @classmethod
    def _plot_threads_from_state(
        cls, campaign_state: dict[str, object]
    ) -> dict[str, dict[str, object]]:
        raw = (
            campaign_state.get(cls.PLOT_THREADS_STATE_KEY)
            if isinstance(campaign_state, dict)
            else {}
        )
        if not isinstance(raw, dict):
            raw = {}
        threads: dict[str, dict[str, object]] = {}
        for raw_key, raw_entry in raw.items():
            if not isinstance(raw_entry, dict):
                continue
            thread_key = cls._plot_thread_key(raw_entry.get("thread") or raw_key)
            if not thread_key:
                continue
            raw_deps = raw_entry.get("dependencies")
            if not isinstance(raw_deps, list):
                raw_deps = []
            deps = []
            for dep in raw_deps[: cls.MAX_PLOT_DEPENDENCIES]:
                dep_text = " ".join(str(dep or "").strip().split())[:120]
                if dep_text:
                    deps.append(dep_text)
            status = str(raw_entry.get("status") or "active").strip().lower()
            if status not in {"active", "resolved"}:
                status = "active"
            threads[thread_key] = {
                "thread": thread_key,
                "setup": str(raw_entry.get("setup") or "").strip()[:260],
                "intended_payoff": str(raw_entry.get("intended_payoff") or "").strip()[:260],
                "target_turns": min(250, max(1, cls._coerce_non_negative_int(
                    raw_entry.get("target_turns", 8), default=8
                ))),
                "dependencies": deps,
                "player_slugs": [
                    cls._player_slug_key(item)
                    for item in list(raw_entry.get("player_slugs") or [])[:8]
                    if cls._player_slug_key(item)
                ],
                "npc_slugs": [
                    str(item or "").strip()
                    for item in list(raw_entry.get("npc_slugs") or [])[:12]
                    if str(item or "").strip()
                ],
                "visibility": str(raw_entry.get("visibility") or "").strip().lower(),
                "visible_player_slugs": [
                    cls._player_slug_key(item)
                    for item in list(raw_entry.get("visible_player_slugs") or [])[:8]
                    if cls._player_slug_key(item)
                ],
                "visible_actor_ids": [
                    str(item)
                    for item in list(raw_entry.get("visible_user_ids") or raw_entry.get("visible_actor_ids") or [])[:8]
                    if item is not None
                ],
                "aware_npc_slugs": [
                    str(item or "").strip()
                    for item in list(raw_entry.get("aware_npc_slugs") or [])[:12]
                    if str(item or "").strip()
                ],
                "location_key": str(raw_entry.get("location_key") or "").strip()[:160],
                "hint": " ".join(str(raw_entry.get("hint") or "").strip().split())[:220],
                "status": status,
                "resolution": str(raw_entry.get("resolution") or "").strip()[:260],
                "created_turn": cls._coerce_non_negative_int(
                    raw_entry.get("created_turn", 0), default=0
                ),
                "updated_turn": cls._coerce_non_negative_int(
                    raw_entry.get("updated_turn", 0), default=0
                ),
            }
        return threads

    @classmethod
    def _plot_threads_for_prompt(
        cls,
        campaign_state: dict[str, object],
        *,
        limit: int = 12,
    ) -> list[dict[str, object]]:
        threads = cls._plot_threads_from_state(campaign_state)
        rows = list(threads.values())
        rows.sort(
            key=lambda row: (
                0 if str(row.get("status")) == "active" else 1,
                -cls._coerce_non_negative_int(row.get("updated_turn", 0), default=0),
                str(row.get("thread") or ""),
            )
        )
        out = []
        for row in rows[: max(1, int(limit or 12))]:
            out.append(
                {
                    "thread": row.get("thread"),
                    "setup": row.get("setup"),
                    "intended_payoff": row.get("intended_payoff"),
                    "target_turns": row.get("target_turns"),
                    "dependencies": list(row.get("dependencies") or []),
                    "hint": row.get("hint"),
                    "status": row.get("status"),
                    "resolution": row.get("resolution"),
                }
            )
        return out

    @classmethod
    def _plot_thread_visibility_descriptor(
        cls,
        thread: dict[str, object],
    ) -> tuple[str, list, list[str], list[str], str]:
        scope = str(thread.get("visibility") or "").strip().lower()
        visible_actor_ids = [
            str(item)
            for item in list(thread.get("visible_user_ids") or thread.get("visible_actor_ids") or [])[:8]
            if item is not None
        ]
        visible_player_slugs = [
            cls._player_slug_key(item)
            for item in list(thread.get("visible_player_slugs") or [])[:8]
            if cls._player_slug_key(item)
        ]
        aware_npc_slugs = [
            str(item or "").strip()
            for item in list(thread.get("aware_npc_slugs") or [])[:12]
            if str(item or "").strip()
        ]
        location_key = str(thread.get("location_key") or "").strip()[:160]
        if scope not in {"public", "private", "limited", "local"}:
            if visible_player_slugs:
                scope = "private" if len(visible_player_slugs) <= 1 else "limited"
            elif location_key:
                scope = "local"
            else:
                scope = "public"
        return scope, visible_actor_ids, visible_player_slugs, aware_npc_slugs, location_key

    @classmethod
    def _plot_thread_visible_to_viewer(
        cls,
        thread: dict[str, object],
        *,
        viewer_actor_id: str | None = None,
        viewer_slug: str | None = None,
        viewer_location_key: str | None = None,
    ) -> bool:
        scope, visible_actor_ids, visible_player_slugs, _aware, location_key = (
            cls._plot_thread_visibility_descriptor(thread)
        )
        if scope == "public":
            return True
        viewer_slug_key = cls._player_slug_key(viewer_slug) if viewer_slug else ""
        if scope in {"private", "limited"}:
            if viewer_actor_id is not None and str(viewer_actor_id) in [str(x) for x in visible_actor_ids]:
                return True
            if viewer_slug_key and viewer_slug_key in visible_player_slugs:
                return True
            player_slugs = [
                cls._player_slug_key(item)
                for item in list(thread.get("player_slugs") or [])
                if cls._player_slug_key(item)
            ]
            if viewer_slug_key and viewer_slug_key in player_slugs:
                return True
            return False
        if scope == "local":
            if not location_key:
                return True
            viewer_location_norm = cls._normalize_location_key(viewer_location_key)
            return viewer_location_norm == cls._normalize_location_key(location_key) if viewer_location_norm else False
        return True

    @classmethod
    def _plot_hints_for_viewer(
        cls,
        campaign_state: dict[str, object],
        *,
        viewer_actor_id: str | None = None,
        viewer_slug: str | None = None,
        viewer_location_key: str | None = None,
        limit: int = 6,
    ) -> list[dict[str, object]]:
        threads = cls._plot_threads_from_state(campaign_state)
        out: list[dict[str, object]] = []
        for thread in threads.values():
            if str(thread.get("status")) != "active":
                continue
            hint = str(thread.get("hint") or "").strip()
            if not hint:
                continue
            if not cls._plot_thread_visible_to_viewer(
                thread,
                viewer_actor_id=viewer_actor_id,
                viewer_slug=viewer_slug,
                viewer_location_key=viewer_location_key,
            ):
                continue
            out.append(
                {
                    "thread": thread.get("thread"),
                    "hint": hint,
                }
            )
            if len(out) >= max(1, int(limit)):
                break
        return out

    @classmethod
    def _chapter_slug_key(cls, value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:80]

    @classmethod
    def _chapter_plan_from_state(
        cls, campaign_state: dict[str, object]
    ) -> dict[str, dict[str, object]]:
        raw = (
            campaign_state.get(cls.CHAPTER_PLAN_STATE_KEY)
            if isinstance(campaign_state, dict)
            else {}
        )
        if not isinstance(raw, dict):
            raw = {}
        chapters: dict[str, dict[str, object]] = {}
        for raw_slug, raw_entry in raw.items():
            if not isinstance(raw_entry, dict):
                continue
            slug = cls._chapter_slug_key(raw_entry.get("slug") or raw_slug)
            if not slug:
                continue
            scenes_raw = raw_entry.get("scenes")
            if not isinstance(scenes_raw, list):
                scenes_raw = []
            scenes = []
            for scene in scenes_raw[:20]:
                scene_slug = cls._chapter_slug_key(scene)
                if scene_slug:
                    scenes.append(scene_slug)
            current_scene = cls._chapter_slug_key(raw_entry.get("current_scene"))
            if not current_scene and scenes:
                current_scene = scenes[0]
            status = str(raw_entry.get("status") or "active").strip().lower()
            if status not in {"active", "resolved"}:
                status = "active"
            chapters[slug] = {
                "slug": slug,
                "title": " ".join(str(raw_entry.get("title") or slug).strip().split())[:120],
                "summary": str(raw_entry.get("summary") or "").strip()[:260],
                "scenes": scenes,
                "current_scene": current_scene,
                "status": status,
                "resolution": str(raw_entry.get("resolution") or "").strip()[:260],
                "created_turn": cls._coerce_non_negative_int(
                    raw_entry.get("created_turn", 0), default=0
                ),
                "updated_turn": cls._coerce_non_negative_int(
                    raw_entry.get("updated_turn", 0), default=0
                ),
            }
        return chapters

    @classmethod
    def _chapters_for_prompt(
        cls,
        campaign_state: dict[str, object],
        *,
        active_only: bool = True,
        limit: int = 8,
    ) -> list[dict[str, object]]:
        chapters = cls._chapter_plan_from_state(campaign_state)
        rows = list(chapters.values())
        if active_only:
            rows = [row for row in rows if str(row.get("status")) == "active"]
        rows.sort(
            key=lambda row: (
                0 if str(row.get("status")) == "active" else 1,
                -cls._coerce_non_negative_int(row.get("updated_turn", 0), default=0),
                str(row.get("slug") or ""),
            )
        )
        out = []
        for row in rows[: max(1, int(limit or 8))]:
            out.append(
                {
                    "slug": row.get("slug"),
                    "title": row.get("title"),
                    "summary": row.get("summary"),
                    "current_scene": row.get("current_scene"),
                    "scenes": list(row.get("scenes") or []),
                    "status": row.get("status"),
                    "resolution": row.get("resolution"),
                }
            )
        return out

    @classmethod
    def _consequence_id_key(cls, value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:90]

    @classmethod
    def _consequences_from_state(
        cls, campaign_state: dict[str, object]
    ) -> dict[str, dict[str, object]]:
        raw = (
            campaign_state.get(cls.CONSEQUENCE_STATE_KEY)
            if isinstance(campaign_state, dict)
            else {}
        )
        if not isinstance(raw, dict):
            raw = {}
        out: dict[str, dict[str, object]] = {}
        for raw_key, raw_entry in raw.items():
            if not isinstance(raw_entry, dict):
                continue
            cid = cls._consequence_id_key(raw_entry.get("id") or raw_key)
            if not cid:
                continue
            status = str(raw_entry.get("status") or "active").strip().lower()
            if status not in {"active", "resolved"}:
                status = "active"
            expires_at_turn = cls._coerce_non_negative_int(
                raw_entry.get("expires_at_turn", 0), default=0
            )
            out[cid] = {
                "id": cid,
                "trigger": str(raw_entry.get("trigger") or "").strip()[:240],
                "consequence": str(raw_entry.get("consequence") or "").strip()[:300],
                "severity": str(raw_entry.get("severity") or "low").strip().lower()[:24],
                "status": status,
                "created_turn": cls._coerce_non_negative_int(
                    raw_entry.get("created_turn", 0), default=0
                ),
                "updated_turn": cls._coerce_non_negative_int(
                    raw_entry.get("updated_turn", 0), default=0
                ),
                "expires_at_turn": expires_at_turn,
                "resolution": str(raw_entry.get("resolution") or "").strip()[:260],
            }
        return out

    @classmethod
    def _consequences_for_prompt(
        cls,
        campaign_state: dict[str, object],
        *,
        current_turn: int = 0,
        limit: int = 12,
    ) -> list[dict[str, object]]:
        rows = list(cls._consequences_from_state(campaign_state).values())
        active_rows = []
        turn_now = max(0, int(current_turn or 0))
        for row in rows:
            if str(row.get("status")) != "active":
                continue
            expires_at_turn = cls._coerce_non_negative_int(
                row.get("expires_at_turn", 0), default=0
            )
            if expires_at_turn > 0 and turn_now > 0 and expires_at_turn < turn_now:
                continue
            active_rows.append(row)
        active_rows.sort(
            key=lambda row: (
                {"critical": 0, "high": 1, "moderate": 2, "low": 3}.get(
                    str(row.get("severity") or "low"), 4
                ),
                cls._coerce_non_negative_int(row.get("expires_at_turn", 0), default=0)
                if cls._coerce_non_negative_int(row.get("expires_at_turn", 0), default=0) > 0
                else 10**9,
                -cls._coerce_non_negative_int(row.get("updated_turn", 0), default=0),
            )
        )
        out = []
        for row in active_rows[: max(1, int(limit or 12))]:
            out.append(
                {
                    "id": row.get("id"),
                    "trigger": row.get("trigger"),
                    "consequence": row.get("consequence"),
                    "severity": row.get("severity"),
                    "expires_at_turn": row.get("expires_at_turn"),
                }
            )
        return out

    @classmethod
    def _apply_plot_plan_tool(
        cls,
        campaign_state: dict[str, object],
        payload: dict[str, object],
        *,
        current_turn: int = 0,
    ) -> dict[str, object]:
        threads = cls._plot_threads_from_state(campaign_state)
        raw_plans = payload.get("plans")
        if isinstance(raw_plans, dict):
            raw_plans = [raw_plans]
        if not isinstance(raw_plans, list):
            raw_plans = []
        updated = 0
        removed = 0
        for raw_plan in raw_plans[:12]:
            if not isinstance(raw_plan, dict):
                continue
            thread_key = cls._plot_thread_key(
                raw_plan.get("thread") or raw_plan.get("slug")
            )
            if not thread_key:
                continue
            delete_requested = bool(
                raw_plan.get("remove")
                or raw_plan.get("delete")
                or raw_plan.get("_delete")
            )
            if delete_requested:
                if thread_key in threads:
                    threads.pop(thread_key, None)
                    removed += 1
                continue
            row = dict(
                threads.get(
                    thread_key,
                    {
                        "thread": thread_key,
                        "setup": "",
                        "intended_payoff": "",
                        "target_turns": 8,
                        "dependencies": [],
                        "player_slugs": [],
                        "npc_slugs": [],
                        "visibility": "",
                        "visible_player_slugs": [],
                        "visible_actor_ids": [],
                        "aware_npc_slugs": [],
                        "location_key": "",
                        "hint": "",
                        "status": "active",
                        "resolution": "",
                        "created_turn": max(0, int(current_turn or 0)),
                        "updated_turn": max(0, int(current_turn or 0)),
                    },
                )
            )
            for field in ("setup", "intended_payoff", "resolution"):
                if field in raw_plan and raw_plan.get(field) is not None:
                    row[field] = " ".join(
                        str(raw_plan.get(field) or "").strip().split()
                    )[:260]
            if "target_turns" in raw_plan:
                target_turns = cls._coerce_non_negative_int(
                    raw_plan.get("target_turns", row.get("target_turns", 8)), default=8
                )
                row["target_turns"] = min(250, max(1, target_turns))
            raw_deps = raw_plan.get("dependencies")
            if isinstance(raw_deps, list):
                dep_clean = []
                for dep in raw_deps[: cls.MAX_PLOT_DEPENDENCIES]:
                    dep_text = " ".join(str(dep or "").strip().split())[:120]
                    if dep_text:
                        dep_clean.append(dep_text)
                row["dependencies"] = dep_clean
            if "visibility" in raw_plan:
                visibility = str(raw_plan.get("visibility") or "").strip().lower()
                if visibility in {"public", "private", "limited", "local"}:
                    row["visibility"] = visibility
            if "player_slugs" in raw_plan:
                raw_player_slugs = raw_plan.get("player_slugs")
                if isinstance(raw_player_slugs, str):
                    raw_player_slugs = [raw_player_slugs]
                cleaned_player_slugs: list[str] = []
                seen_player_slugs: set[str] = set()
                if isinstance(raw_player_slugs, list):
                    for item in raw_player_slugs[:8]:
                        slug = cls._player_slug_key(item)
                        if not slug or slug in seen_player_slugs:
                            continue
                        seen_player_slugs.add(slug)
                        cleaned_player_slugs.append(slug)
                row["player_slugs"] = cleaned_player_slugs
            if "visible_player_slugs" in raw_plan:
                raw_visible_player_slugs = raw_plan.get("visible_player_slugs")
                if isinstance(raw_visible_player_slugs, str):
                    raw_visible_player_slugs = [raw_visible_player_slugs]
                cleaned_visible_player_slugs: list[str] = []
                seen_visible_player_slugs: set[str] = set()
                if isinstance(raw_visible_player_slugs, list):
                    for item in raw_visible_player_slugs[:8]:
                        slug = cls._player_slug_key(item)
                        if not slug or slug in seen_visible_player_slugs:
                            continue
                        seen_visible_player_slugs.add(slug)
                        cleaned_visible_player_slugs.append(slug)
                row["visible_player_slugs"] = cleaned_visible_player_slugs
            if "npc_slugs" in raw_plan:
                raw_npc_slugs = raw_plan.get("npc_slugs")
                if isinstance(raw_npc_slugs, str):
                    raw_npc_slugs = [raw_npc_slugs]
                cleaned_npc_slugs: list[str] = []
                seen_npc_slugs: set[str] = set()
                if isinstance(raw_npc_slugs, list):
                    for item in raw_npc_slugs[:12]:
                        slug = str(item or "").strip()
                        if not slug or slug in seen_npc_slugs:
                            continue
                        seen_npc_slugs.add(slug)
                        cleaned_npc_slugs.append(slug)
                row["npc_slugs"] = cleaned_npc_slugs
            if "aware_npc_slugs" in raw_plan:
                raw_aware_npc_slugs = raw_plan.get("aware_npc_slugs")
                if isinstance(raw_aware_npc_slugs, str):
                    raw_aware_npc_slugs = [raw_aware_npc_slugs]
                cleaned_aware_npc_slugs: list[str] = []
                seen_aware_npc_slugs: set[str] = set()
                if isinstance(raw_aware_npc_slugs, list):
                    for item in raw_aware_npc_slugs[:12]:
                        slug = str(item or "").strip()
                        if not slug or slug in seen_aware_npc_slugs:
                            continue
                        seen_aware_npc_slugs.add(slug)
                        cleaned_aware_npc_slugs.append(slug)
                row["aware_npc_slugs"] = cleaned_aware_npc_slugs
            if "location_key" in raw_plan:
                row["location_key"] = str(raw_plan.get("location_key") or "").strip()[:160]
            if "hint" in raw_plan:
                row["hint"] = " ".join(str(raw_plan.get("hint") or "").strip().split())[:220]
            status = str(raw_plan.get("status") or row.get("status") or "active").strip().lower()
            if raw_plan.get("resolve"):
                status = "resolved"
            if status not in {"active", "resolved"}:
                status = "active"
            row["status"] = status
            if row.get("status") == "resolved" and not row.get("resolution"):
                row["resolution"] = "resolved"
            if row.get("status") != "resolved":
                row["resolution"] = str(row.get("resolution") or "")[:260]
            row["updated_turn"] = max(0, int(current_turn or 0))
            if cls._coerce_non_negative_int(row.get("created_turn", 0), default=0) <= 0:
                row["created_turn"] = max(0, int(current_turn or 0))
            threads[thread_key] = row
            updated += 1
        if len(threads) > cls.MAX_PLOT_THREADS:
            ranked = sorted(
                threads.items(),
                key=lambda kv: (
                    0 if str(kv[1].get("status")) == "active" else 1,
                    -cls._coerce_non_negative_int(kv[1].get("updated_turn", 0), default=0),
                    kv[0],
                ),
            )
            threads = dict(ranked[: cls.MAX_PLOT_THREADS])
        campaign_state[cls.PLOT_THREADS_STATE_KEY] = threads
        active_threads = [
            row
            for row in cls._plot_threads_for_prompt(campaign_state, limit=12)
            if str(row.get("status")) == "active"
        ]
        return {
            "updated": updated,
            "removed": removed,
            "total": len(threads),
            "active": active_threads,
        }

    @classmethod
    def _apply_chapter_plan_tool(
        cls,
        campaign_state: dict[str, object],
        payload: dict[str, object],
        *,
        current_turn: int = 0,
        on_rails: bool = False,
    ) -> dict[str, object]:
        if on_rails:
            return {"updated": 0, "ignored": True, "reason": "on_rails_enabled"}
        chapters = cls._chapter_plan_from_state(campaign_state)
        action = str(payload.get("action") or "create").strip().lower()
        changed = 0

        def _resolve_slug() -> str:
            chapter_ref = payload.get("chapter")
            if isinstance(chapter_ref, dict):
                return cls._chapter_slug_key(
                    chapter_ref.get("slug") or chapter_ref.get("title")
                )
            return cls._chapter_slug_key(payload.get("chapter") or payload.get("slug"))

        slug = _resolve_slug()
        chapter_payload = payload.get("chapter")
        if isinstance(chapter_payload, dict):
            if not slug:
                slug = cls._chapter_slug_key(
                    chapter_payload.get("slug") or chapter_payload.get("title")
                )
        if action in {"create", "update"}:
            if not slug:
                return {"updated": 0, "ignored": True, "reason": "missing_slug"}
            row = dict(
                chapters.get(
                    slug,
                    {
                        "slug": slug,
                        "title": slug,
                        "summary": "",
                        "scenes": [],
                        "current_scene": "",
                        "status": "active",
                        "resolution": "",
                        "created_turn": max(0, int(current_turn or 0)),
                        "updated_turn": max(0, int(current_turn or 0)),
                    },
                )
            )
            if isinstance(chapter_payload, dict):
                if chapter_payload.get("title") is not None:
                    row["title"] = " ".join(
                        str(chapter_payload.get("title") or "").strip().split()
                    )[:120] or row.get("title") or slug
                if chapter_payload.get("summary") is not None:
                    row["summary"] = str(chapter_payload.get("summary") or "").strip()[:260]
                scenes_raw = chapter_payload.get("scenes")
                if isinstance(scenes_raw, list):
                    scenes = []
                    for scene in scenes_raw[:20]:
                        scene_slug = cls._chapter_slug_key(scene)
                        if scene_slug:
                            scenes.append(scene_slug)
                    row["scenes"] = scenes
                if chapter_payload.get("current_scene") is not None:
                    row["current_scene"] = cls._chapter_slug_key(
                        chapter_payload.get("current_scene")
                    )
                if chapter_payload.get("active") is not None:
                    row["status"] = (
                        "active" if bool(chapter_payload.get("active")) else "resolved"
                    )
            if not row.get("current_scene") and row.get("scenes"):
                row["current_scene"] = row["scenes"][0]
            row["updated_turn"] = max(0, int(current_turn or 0))
            if cls._coerce_non_negative_int(row.get("created_turn", 0), default=0) <= 0:
                row["created_turn"] = max(0, int(current_turn or 0))
            chapters[slug] = row
            changed += 1
        elif action == "advance_scene":
            if not slug or slug not in chapters:
                return {"updated": 0, "ignored": True, "reason": "chapter_not_found"}
            row = dict(chapters.get(slug) or {})
            to_scene = cls._chapter_slug_key(
                payload.get("to_scene") or payload.get("scene")
            )
            scenes = list(row.get("scenes") or [])
            if to_scene:
                if to_scene not in scenes:
                    scenes.append(to_scene)
                row["current_scene"] = to_scene
            elif scenes:
                current = cls._chapter_slug_key(row.get("current_scene"))
                try:
                    idx = scenes.index(current)
                except ValueError:
                    idx = -1
                next_idx = min(len(scenes) - 1, idx + 1)
                row["current_scene"] = scenes[next_idx]
            row["scenes"] = scenes[:20]
            row["status"] = "active"
            row["updated_turn"] = max(0, int(current_turn or 0))
            chapters[slug] = row
            changed += 1
        elif action in {"resolve", "close"}:
            if not slug or slug not in chapters:
                return {"updated": 0, "ignored": True, "reason": "chapter_not_found"}
            row = dict(chapters.get(slug) or {})
            row["status"] = "resolved"
            row["resolution"] = " ".join(
                str(payload.get("resolution") or row.get("resolution") or "").split()
            )[:260]
            row["updated_turn"] = max(0, int(current_turn or 0))
            chapters[slug] = row
            changed += 1
        if len(chapters) > cls.MAX_OFFRAILS_CHAPTERS:
            ranked = sorted(
                chapters.items(),
                key=lambda kv: (
                    0 if str(kv[1].get("status")) == "active" else 1,
                    -cls._coerce_non_negative_int(kv[1].get("updated_turn", 0), default=0),
                    kv[0],
                ),
            )
            chapters = dict(ranked[: cls.MAX_OFFRAILS_CHAPTERS])
        campaign_state[cls.CHAPTER_PLAN_STATE_KEY] = chapters
        active_chapters = cls._chapters_for_prompt(
            campaign_state, active_only=True, limit=10
        )
        return {
            "updated": changed,
            "ignored": False,
            "total": len(chapters),
            "active": active_chapters,
        }

    @classmethod
    def _apply_consequence_log_tool(
        cls,
        campaign_state: dict[str, object],
        payload: dict[str, object],
        *,
        current_turn: int = 0,
    ) -> dict[str, object]:
        rows = cls._consequences_from_state(campaign_state)
        turn_now = max(0, int(current_turn or 0))
        added = 0
        updated = 0
        resolved = 0
        removed = 0

        def _iter_entries(value: object) -> list[dict[str, object]]:
            if isinstance(value, dict):
                return [value]
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
            return []

        for entry in _iter_entries(payload.get("add")):
            trigger = " ".join(str(entry.get("trigger") or "").strip().split())[:240]
            consequence = " ".join(
                str(entry.get("consequence") or "").strip().split()
            )[:300]
            if not trigger or not consequence:
                continue
            cid = cls._consequence_id_key(
                entry.get("id")
                or entry.get("slug")
                or trigger[:60]
            )
            if not cid:
                continue
            severity = str(entry.get("severity") or "low").strip().lower()
            if severity not in {"low", "moderate", "high", "critical"}:
                severity = "low"
            expires_turns = cls._coerce_non_negative_int(
                entry.get("expires_turns", 0), default=0
            )
            expires_at_turn = (turn_now + expires_turns) if expires_turns > 0 else 0
            row = dict(rows.get(cid) or {})
            is_new = not bool(row)
            row.update(
                {
                    "id": cid,
                    "trigger": trigger,
                    "consequence": consequence,
                    "severity": severity,
                    "status": "active",
                    "updated_turn": turn_now,
                    "expires_at_turn": expires_at_turn,
                    "resolution": str(row.get("resolution") or "")[:260],
                }
            )
            if is_new:
                row["created_turn"] = turn_now
                added += 1
            else:
                updated += 1
            rows[cid] = row
        for entry in _iter_entries(payload.get("resolve")):
            cid = cls._consequence_id_key(
                entry.get("id") or entry.get("slug") or entry.get("trigger")
            )
            if not cid or cid not in rows:
                continue
            row = dict(rows.get(cid) or {})
            row["status"] = "resolved"
            row["updated_turn"] = turn_now
            row["resolution"] = " ".join(
                str(entry.get("resolution") or row.get("resolution") or "resolved")
                .strip()
                .split()
            )[:260]
            rows[cid] = row
            resolved += 1
        remove_keys = payload.get("remove")
        if isinstance(remove_keys, list):
            for raw_key in remove_keys:
                cid = cls._consequence_id_key(raw_key)
                if cid and cid in rows:
                    rows.pop(cid, None)
                    removed += 1
        for cid, row in list(rows.items()):
            expires_at_turn = cls._coerce_non_negative_int(
                row.get("expires_at_turn", 0), default=0
            )
            if (
                expires_at_turn > 0
                and turn_now > 0
                and turn_now > expires_at_turn
                and str(row.get("status")) == "active"
            ):
                rows.pop(cid, None)
                removed += 1
        if len(rows) > cls.MAX_CONSEQUENCES:
            ranked = sorted(
                rows.items(),
                key=lambda kv: (
                    0 if str(kv[1].get("status")) == "active" else 1,
                    -cls._coerce_non_negative_int(kv[1].get("updated_turn", 0), default=0),
                    kv[0],
                ),
            )
            rows = dict(ranked[: cls.MAX_CONSEQUENCES])
        campaign_state[cls.CONSEQUENCE_STATE_KEY] = rows
        active = cls._consequences_for_prompt(
            campaign_state, current_turn=turn_now, limit=12
        )
        return {
            "added": added,
            "updated": updated,
            "resolved": resolved,
            "removed": removed,
            "total": len(rows),
            "active": active,
        }

    @classmethod
    def _auto_resolve_stale_plot_threads(
        cls,
        campaign_state: dict[str, object],
        pruned_keys: list[str],
    ) -> int:
        threads = cls._plot_threads_from_state(campaign_state)
        if not threads:
            return 0
        pruned_slugs = set()
        for key in pruned_keys:
            slug = cls._plot_thread_key(key)
            if slug:
                pruned_slugs.add(slug)
        if not pruned_slugs:
            return 0
        resolved_count = 0
        for thread_key, thread in threads.items():
            if str(thread.get("status")) != "active":
                continue
            if thread_key in pruned_slugs:
                thread["status"] = "resolved"
                if not thread.get("resolution"):
                    thread["resolution"] = "auto-resolved: state key pruned"
                resolved_count += 1
        if resolved_count > 0:
            campaign_state[cls.PLOT_THREADS_STATE_KEY] = threads
        return resolved_count

    # ------------------------------------------------------------------
    # Calendar event notifications
    # ------------------------------------------------------------------

    @classmethod
    def _calendar_event_key(cls, event: dict[str, object]) -> str:
        name = str(event.get("name", "")).strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")[:80] or "event"
        fire_day = cls._coerce_non_negative_int(event.get("fire_day", 1), default=1) or 1
        fire_hour = min(23, max(0, cls._coerce_non_negative_int(event.get("fire_hour", 23), default=23)))
        return f"{slug}:{fire_day}:{fire_hour}"

    @classmethod
    def _calendar_player_aliases_from_registry_entry(
        cls,
        entry: dict[str, object],
    ) -> set[str]:
        aliases: set[str] = set()

        def _add(raw: object) -> None:
            text = " ".join(str(raw or "").strip().lower().split())
            if text:
                aliases.add(text[:160])

        actor_id = entry.get("actor_id") or entry.get("user_id")
        if actor_id is not None:
            _add(actor_id)
        name = str(entry.get("name") or "").strip()
        slug = str(entry.get("slug") or "").strip()
        if name:
            _add(name)
        if slug:
            _add(slug)
        if name:
            normalized_name = cls._player_slug_key(name)
            if normalized_name:
                _add(normalized_name)
        return aliases

    def _calendar_event_notification_summary(
        self,
        notification: dict[str, object],
    ) -> str:
        name = str(notification.get("name") or "Unknown event").strip()
        fire_day = self._coerce_non_negative_int(notification.get("fire_day", 1), default=1) or 1
        fire_hour = min(23, max(0, self._coerce_non_negative_int(notification.get("fire_hour", 23), default=23)))
        status = str(notification.get("status") or "fired").strip().lower()
        description = " ".join(str(notification.get("description") or "").split())
        if status == "overdue":
            lead = f"Calendar event overdue: {name} (was due Day {fire_day}, {fire_hour:02d}:00)."
        else:
            lead = f"Calendar event fired: {name} (Day {fire_day}, {fire_hour:02d}:00)."
        if description:
            return self._trim_text(f"{lead} {description}", 280)
        return lead

    def _calendar_event_scope(self, campaign_id: str, event: object) -> str:
        return "global" if not self._resolve_calendar_target_user_ids(campaign_id, event) else "player"

    def _calendar_event_notification_targets(
        self,
        campaign_id: str,
        event: object,
    ) -> list[str]:
        explicit_targets = self._resolve_calendar_target_user_ids(campaign_id, event)
        if explicit_targets:
            return [str(t) for t in explicit_targets]
        with self._session_factory() as session:
            from text_game_engine.persistence.sqlalchemy.models import Player
            return [
                str(row.actor_id)
                for row in session.query(Player).filter_by(campaign_id=str(campaign_id)).all()
                if getattr(row, "actor_id", None) is not None
            ]

    @staticmethod
    def _calendar_fix_ampm(fire_hour: int, description: str) -> int:
        """Fix AM/PM mismatch — e.g. LLM outputs fire_hour=7 for '7pm'."""
        if not description:
            return fire_hour
        text = description.lower()
        for m in re.finditer(r"\b(\d{1,2})(?:\s*:\s*\d{2})?\s*(am|pm)\b", text):
            desc_hour = int(m.group(1))
            ampm = m.group(2)
            if desc_hour < 1 or desc_hour > 12:
                continue
            if ampm == "pm":
                expected_24h = desc_hour if desc_hour == 12 else desc_hour + 12
            else:
                expected_24h = 0 if desc_hour == 12 else desc_hour
            if fire_hour == desc_hour and fire_hour != expected_24h:
                fire_hour = expected_24h
                break
        return fire_hour

    @staticmethod
    def _calendar_fix_relative_day(fire_day: int, description: str, current_day: int) -> int:
        """Fix off-by-one when description says 'tomorrow' but fire_day == today."""
        if not description:
            return fire_day
        text = description.lower()
        if re.search(r"\btomorrow\b", text):
            expected = current_day + 1
            if fire_day == current_day:
                return expected
        elif re.search(r"\btoday\b", text):
            if fire_day > current_day:
                return current_day
        return fire_day

    @classmethod
    def _calendar_should_prune_stale_event(
        cls,
        event: dict[str, object],
        *,
        hours_remaining: int,
    ) -> bool:
        if not isinstance(event, dict):
            return False
        if hours_remaining >= 0:
            return False
        overdue_hours = abs(int(hours_remaining))
        fired_notice_key = str(event.get("fired_notice_key") or "").strip()
        if overdue_hours >= 24:
            return True
        if fired_notice_key and overdue_hours >= 6:
            return True
        return False

    def _calendar_collect_fired_events(
        self,
        campaign_id: str,
        campaign_state: dict[str, object],
        *,
        from_time: dict[str, int],
        to_time: dict[str, int],
    ) -> list[dict[str, object]]:
        if not isinstance(campaign_state, dict):
            return []
        raw_calendar = campaign_state.get("calendar")
        if not isinstance(raw_calendar, list) or not raw_calendar:
            return []
        current_day = self._coerce_non_negative_int(to_time.get("day", 1), default=1) or 1
        current_hour = min(23, max(0, self._coerce_non_negative_int(to_time.get("hour", 8), default=8)))
        # Compute absolute minutes if _game_time_to_total_minutes is available
        if hasattr(self, "_game_time_to_total_minutes"):
            from_abs = self._game_time_to_total_minutes(from_time)
            to_abs = self._game_time_to_total_minutes(to_time)
        else:
            from_abs = (((max(1, from_time.get("day", 1)) - 1) * 24) + from_time.get("hour", 0)) * 60
            to_abs = (((max(1, current_day) - 1) * 24) + current_hour) * 60
        if to_abs <= 0:
            return []
        notifications: list[dict[str, object]] = []
        changed = False
        for raw_event in raw_calendar:
            if not isinstance(raw_event, dict):
                continue
            normalized = self._calendar_normalize_event(
                raw_event,
                current_day=current_day,
                current_hour=current_hour,
            )
            if normalized is None:
                continue
            event_key = self._calendar_event_key(normalized)
            if raw_event.get("fired_notice_key") == event_key:
                continue
            fire_day = self._coerce_non_negative_int(normalized.get("fire_day", current_day), default=current_day)
            fire_hour = min(23, max(0, self._coerce_non_negative_int(normalized.get("fire_hour", 23), default=23)))
            due_abs = (((max(1, fire_day) - 1) * 24) + fire_hour) * 60
            if due_abs > to_abs:
                continue
            if due_abs > from_abs:
                status = "fired"
            else:
                status = "overdue"
            raw_event["fired_notice_key"] = event_key
            raw_event["fired_notice_day"] = current_day
            raw_event["fired_notice_hour"] = current_hour
            changed = True
            notifications.append(
                {
                    "name": str(normalized.get("name") or "Unknown event"),
                    "description": str(normalized.get("description") or "").strip(),
                    "fire_day": fire_day,
                    "fire_hour": fire_hour,
                    "status": status,
                    "scope": self._calendar_event_scope(campaign_id, raw_event),
                    "target_actor_ids": self._calendar_event_notification_targets(campaign_id, raw_event),
                }
            )
        if changed:
            campaign_state["calendar"] = raw_calendar
        return notifications

    async def _send_calendar_event_notifications(
        self,
        *,
        campaign_id: str,
        campaign_name: str,
        notifications: list[dict[str, object]],
    ) -> None:
        """Send calendar event notifications via the notification port.

        The actual delivery mechanism (Discord DMs, etc.) is handled by the
        NotificationPort adapter.
        """
        if not notifications or self._notification_port is None:
            return
        for notification in notifications:
            summary = self._calendar_event_notification_summary(notification)
            scope = str(notification.get("scope") or "global").strip().lower()
            target_actor_ids = [
                str(aid)
                for aid in (notification.get("target_actor_ids") or [])
                if aid is not None
            ]
            if scope == "global":
                try:
                    await self._notification_port.send_channel_message(
                        campaign_id=campaign_id,
                        message=f"**[Calendar Event]** {summary}",
                    )
                except Exception:
                    self._zork_log(
                        "CALENDAR NOTIFY FAIL",
                        f"Failed to send calendar notice to main channel for campaign {campaign_id}",
                    )
            dm_targets = self._recent_private_dm_notification_targets(
                campaign_id,
                candidate_actor_ids=target_actor_ids,
            )
            if not dm_targets:
                continue
            dm_message = (
                f"**[Calendar Event Notice]** `{campaign_name}`\n"
                f"{summary}"
            )
            for actor_id in dm_targets:
                try:
                    await self._notification_port.send_dm(actor_id=actor_id, message=dm_message)
                except Exception:
                    self._zork_log(
                        "CALENDAR DM FAIL",
                        f"Failed to send calendar DM notice to actor {actor_id}",
                    )

    async def _send_private_dm_time_jump_notifications(
        self,
        *,
        campaign_id: str,
        campaign_name: str,
        recipient_actor_ids: list[str],
        from_time: dict[str, int],
        to_time: dict[str, int],
        delta_minutes: int,
        event_summary: str,
    ) -> None:
        """Notify DM-engaged players about in-world time jumps."""
        if not recipient_actor_ids or self._notification_port is None:
            return
        to_snapshot = self._extract_game_time_snapshot({"game_time": to_time})
        if hasattr(self, "_format_game_time_label"):
            to_label = self._format_game_time_label(to_snapshot)
        else:
            to_label = f"Day {to_snapshot.get('day', '?')}, {to_snapshot.get('hour', '?'):02d}:00"
        for actor_id in recipient_actor_ids:
            try:
                message = (
                    f"**[Time Jump Notice]** `{campaign_name}` advanced by about "
                    f"{delta_minutes} in-world minutes.\n"
                    f"To: {to_label}\n"
                    f"Cause: {event_summary}"
                )
                await self._notification_port.send_dm(actor_id=actor_id, message=message)
            except Exception:
                self._zork_log(
                    "TIME JUMP DM FAIL",
                    f"Failed to send DM time-jump notification to actor {actor_id}",
                )

    # ------------------------------------------------------------------
    # Turn deletion
    # ------------------------------------------------------------------

    def execute_delete_turn(
        self,
        campaign_id: str,
        target_external_message_id: str,
        *,
        session_id: str | None = None,
        delete_actor_id: str | None = None,
        player_only: bool = False,
    ) -> dict[str, object]:
        with self._session_factory() as session:
            from text_game_engine.persistence.sqlalchemy.models import Campaign, Player, Turn, Snapshot
            target_turn = (
                session.query(Turn)
                .filter_by(
                    campaign_id=str(campaign_id),
                    external_message_id=str(target_external_message_id),
                )
                .first()
            )
            if target_turn is None:
                return {"status": "not-found"}
            if player_only and delete_actor_id is not None:
                if str(getattr(target_turn, "actor_id", "") or "") != str(delete_actor_id):
                    return {"status": "forbidden"}
            # Check if this is the latest narrator turn in scope
            query = session.query(Turn).filter(
                Turn.campaign_id == str(campaign_id),
                Turn.kind == "narrator",
            )
            if player_only:
                query = query.filter(
                    Turn.actor_id == str(delete_actor_id),
                )
                if session_id:
                    query = query.filter(Turn.session_id == str(session_id))
            latest_scope_turn = query.order_by(Turn.id.desc()).first()
            if latest_scope_turn is None or int(latest_scope_turn.id or 0) != int(target_turn.id or 0):
                return {"status": "not-latest", "turn_id": int(target_turn.id or 0)}
            # Find the prior narrator turn
            prior_query = session.query(Turn).filter(
                Turn.campaign_id == str(campaign_id),
                Turn.kind == "narrator",
                Turn.id < target_turn.id,
            )
            if player_only:
                prior_query = prior_query.filter(Turn.actor_id == str(delete_actor_id))
                if session_id:
                    prior_query = prior_query.filter(Turn.session_id == str(session_id))
            previous_turn = prior_query.order_by(Turn.id.desc()).first()
            if previous_turn is None:
                return {"status": "no-prior-snapshot", "turn_id": int(target_turn.id or 0)}
            snapshot = session.query(Snapshot).filter_by(turn_id=previous_turn.id).first()
            if snapshot is None:
                return {"status": "no-prior-snapshot", "turn_id": int(target_turn.id or 0)}
            campaign = session.get(Campaign, str(campaign_id))
            if campaign is None:
                return {"status": "not-found"}
            # Restore state from snapshot
            import json as _json
            if player_only:
                players_data = _json.loads(snapshot.players_json or "[]")
                restored = False
                for pdata in players_data:
                    if not isinstance(pdata, dict):
                        continue
                    player_id = pdata.get("player_id")
                    if not player_id:
                        continue
                    player = session.get(Player, str(player_id))
                    if player is None:
                        continue
                    if str(getattr(player, "actor_id", "") or "") != str(delete_actor_id or ""):
                        continue
                    player.level = pdata.get("level", player.level)
                    player.xp = pdata.get("xp", player.xp)
                    player.attributes_json = pdata.get("attributes_json", player.attributes_json)
                    player.state_json = pdata.get("state_json", player.state_json)
                    restored = True
                    break
                if not restored:
                    return {"status": "no-prior-snapshot", "turn_id": int(target_turn.id or 0)}
            else:
                campaign.state_json = snapshot.campaign_state_json
                campaign.characters_json = snapshot.campaign_characters_json
                campaign.summary = snapshot.campaign_summary
                campaign.last_narration = snapshot.campaign_last_narration
                players_data = _json.loads(snapshot.players_json or "[]")
                for pdata in players_data:
                    if not isinstance(pdata, dict):
                        continue
                    player_id = pdata.get("player_id")
                    if not player_id:
                        continue
                    player = session.get(Player, str(player_id))
                    if player is None:
                        continue
                    player.level = pdata.get("level", player.level)
                    player.xp = pdata.get("xp", player.xp)
                    player.attributes_json = pdata.get("attributes_json", player.attributes_json)
                    player.state_json = pdata.get("state_json", player.state_json)
            # Collect turn IDs to delete
            turn_ids_to_delete = [int(target_turn.id)]
            if getattr(target_turn, "external_user_message_id", None):
                paired_player_turn = (
                    session.query(Turn)
                    .filter_by(
                        campaign_id=str(campaign_id),
                        kind="player",
                        external_user_message_id=target_turn.external_user_message_id,
                    )
                    .order_by(Turn.id.desc())
                    .first()
                )
                if paired_player_turn is not None:
                    turn_ids_to_delete.append(int(paired_player_turn.id))
            turn_ids_to_delete = sorted({tid for tid in turn_ids_to_delete if tid > 0})
            # Delete snapshots and turns
            if turn_ids_to_delete:
                session.query(Snapshot).filter(
                    Snapshot.turn_id.in_(turn_ids_to_delete),
                ).delete(synchronize_session=False)
                deleted_count = session.query(Turn).filter(
                    Turn.id.in_(turn_ids_to_delete),
                ).delete(synchronize_session=False)
            else:
                deleted_count = 0
            session.commit()
            # Clean up embeddings if memory port supports it
            if hasattr(self, "_memory") and self._memory is not None:
                try:
                    for tid in turn_ids_to_delete:
                        self._memory.delete_turn_embeddings(campaign_id, tid)
                except Exception:
                    self._zork_log(
                        "DELETE TURN EMBED FAIL",
                        f"Embedding cleanup failed for campaign {campaign_id}",
                    )
            return {
                "status": "ok",
                "turn_id": int(target_turn.id or 0),
                "deleted_count": int(deleted_count or 0),
                "external_user_message_id": str(getattr(target_turn, "external_user_message_id", "") or ""),
                "player_only": bool(player_only),
            }

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
        existing_row: dict[str, object] | None,
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

    # ------------------------------------------------------------------
    # Advanced NPC awareness
    # ------------------------------------------------------------------

    def _active_scene_npc_slugs(
        self,
        campaign: "Campaign",
        player_state: dict[str, object],
    ) -> set[str]:
        out: set[str] = set()
        characters = self.get_campaign_characters(campaign)
        if not isinstance(characters, dict):
            return out
        for slug, entry in characters.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("deceased_reason"):
                continue
            char_state = {
                "location": entry.get("location"),
                "room_title": entry.get("room_title"),
                "room_summary": entry.get("room_summary"),
                "room_id": entry.get("room_id"),
            }
            if self._same_scene(player_state, char_state):
                clean_slug = str(slug or "").strip()
                if clean_slug:
                    out.add(clean_slug)
        return out

    def _filter_aware_npc_slugs_for_scene(
        self,
        campaign: "Campaign",
        player_state: dict[str, object],
        aware_npc_slugs: object,
    ) -> list[str]:
        if not isinstance(aware_npc_slugs, list):
            return []
        characters = self.get_campaign_characters(campaign)
        if not isinstance(characters, dict):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in aware_npc_slugs:
            candidate = str(item or "").strip()
            if not candidate:
                continue
            resolved_slug = self._resolve_existing_character_slug(characters, candidate)
            if not resolved_slug or resolved_slug in seen:
                continue
            payload = characters.get(resolved_slug)
            if not isinstance(payload, dict) or payload.get("deceased_reason"):
                continue
            char_state = {
                "location": payload.get("location"),
                "room_title": payload.get("room_title"),
                "room_summary": payload.get("room_summary"),
                "room_id": payload.get("room_id"),
            }
            if not self._same_scene(player_state, char_state):
                continue
            seen.add(resolved_slug)
            out.append(resolved_slug)
        return out

    def _filter_aware_npc_slugs_for_location_key(
        self,
        campaign: "Campaign",
        location_key: object,
        aware_npc_slugs: object,
    ) -> list[str]:
        location_text = str(location_key or "").strip()
        if not location_text:
            return []
        return self._filter_aware_npc_slugs_for_scene(
            campaign,
            {"location": location_text},
            aware_npc_slugs,
        )

    def _infer_aware_npc_slugs(
        self,
        campaign: "Campaign",
        player_state: dict[str, object],
        turn_visibility: dict[str, object],
        *,
        narration_text: str = "",
        summary_update: object = None,
        private_context_candidate: dict[str, object] | None = None,
    ) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []

        def _add(slug: object) -> None:
            text = str(slug or "").strip()
            if not text or text in seen:
                return
            seen.add(text)
            out.append(text)

        raw_existing = turn_visibility.get("aware_npc_slugs")
        if isinstance(raw_existing, list):
            for item in raw_existing:
                _add(item)
        if out:
            return out
        candidate_slug = str((private_context_candidate or {}).get("target_slug") or "").strip()
        if candidate_slug:
            _add(candidate_slug)
        if out:
            return out
        combined_text = self._normalize_match_text(
            f"{str(narration_text or '')}\n{str(summary_update or '')}"
        )
        characters = self.get_campaign_characters(campaign)
        same_scene_slugs: list[str] = []
        if isinstance(characters, dict):
            for slug, payload in characters.items():
                if not isinstance(payload, dict) or payload.get("deceased_reason"):
                    continue
                char_name = str(payload.get("name") or slug or "").strip()
                char_state = {
                    "location": payload.get("location"),
                    "room_title": payload.get("room_title"),
                    "room_summary": payload.get("room_summary"),
                    "room_id": payload.get("room_id"),
                }
                if self._same_scene(player_state, char_state):
                    same_scene_slugs.append(str(slug))
                if combined_text:
                    for candidate in (slug, char_name):
                        candidate_norm = self._normalize_match_text(candidate)
                        if candidate_norm and candidate_norm in combined_text:
                            _add(slug)
                            break
        if out:
            return out
        if str(turn_visibility.get("scope") or "").strip().lower() in {"private", "limited"} and len(same_scene_slugs) == 1:
            _add(same_scene_slugs[0])
        return out

    def _promote_player_npc_slugs(
        self,
        visibility: dict[str, object],
        campaign_id: str,
    ) -> dict[str, object]:
        npc_slugs = visibility.get("aware_npc_slugs")
        if not npc_slugs or not isinstance(npc_slugs, list):
            return visibility
        scope = str(visibility.get("scope") or "").strip().lower()
        if scope == "public":
            return visibility
        registry = self._campaign_player_registry(campaign_id, self._session_factory)
        by_slug = registry.get("by_slug", {})
        if not by_slug:
            return visibility
        actor_actor_id = visibility.get("actor_actor_id") or visibility.get("actor_user_id")
        promoted = False
        vis_slugs = list(visibility.get("visible_player_slugs") or [])
        vis_actor_ids = list(visibility.get("visible_actor_ids") or visibility.get("visible_user_ids") or [])
        remaining_npc_slugs = []
        for npc_slug in npc_slugs:
            normalised = self._player_slug_key(npc_slug)
            match = by_slug.get(normalised)
            if match is not None:
                matched_actor_id = match.get("actor_id") or match.get("user_id")
                matched_slug = match.get("slug") or normalised
                if matched_actor_id is not None and str(matched_actor_id) == str(actor_actor_id or ""):
                    remaining_npc_slugs.append(npc_slug)
                    continue
                if matched_slug not in vis_slugs:
                    vis_slugs.append(matched_slug)
                if matched_actor_id is not None and str(matched_actor_id) not in [str(x) for x in vis_actor_ids]:
                    vis_actor_ids.append(str(matched_actor_id))
                promoted = True
            else:
                remaining_npc_slugs.append(npc_slug)
        if not promoted:
            return visibility
        result = dict(visibility)
        result["visible_player_slugs"] = vis_slugs
        result["visible_actor_ids"] = vis_actor_ids
        result["aware_npc_slugs"] = remaining_npc_slugs
        return result

    def _sanitize_npc_roster_against_players(
        self,
        campaign_id: str,
        characters: dict[str, dict],
    ) -> dict[str, dict]:
        if not isinstance(characters, dict) or not characters:
            return {}
        out: dict[str, dict] = {}
        for slug, payload in characters.items():
            if hasattr(self, "_character_update_hits_player"):
                match = self._character_update_hits_player(campaign_id, slug, payload)
                if match is not None:
                    self._zork_log(
                        "NPC ROSTER COLLISION",
                        f"Dropping WORLD_CHARACTERS entry {slug!r} because it collides with player {match.get('name')!r}",
                    )
                    continue
            out[str(slug)] = payload
        return out

    def _sanitize_turn_awareness_for_scene(
        self,
        campaign: "Campaign",
        player_state: dict[str, object],
        turn_visibility: dict[str, object],
    ) -> dict[str, object]:
        if not isinstance(turn_visibility, dict):
            return turn_visibility
        filtered = self._filter_aware_npc_slugs_for_scene(
            campaign,
            player_state,
            turn_visibility.get("aware_npc_slugs"),
        )
        result = dict(turn_visibility)
        result["aware_npc_slugs"] = filtered
        return result

    @classmethod
    def _narration_implies_entity_with_player(
        cls,
        narration_text: str,
        name_candidates: list[str],
    ) -> bool:
        text = str(narration_text or "").strip().lower()
        if not text or not name_candidates:
            return False
        cues = (
            "at your heels",
            "at your heel",
            "by your side",
            "beside you",
            "with you",
            "follows you",
            "following you",
            "trailing you",
            "trotting at",
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
            "texts you",
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

    @staticmethod
    def _entity_name_candidates_for_sync(
        state_key: object, entity_state: dict[str, object]
    ) -> list[str]:
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
            if len(candidate) < 3:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    @classmethod
    def _scrub_scene_output_npc_awareness(
        cls,
        scene_output: object,
    ) -> object:
        if not isinstance(scene_output, dict):
            return scene_output
        beats = scene_output.get("beats")
        if not isinstance(beats, list):
            return scene_output
        changed = False
        cleaned_beats = []
        for beat in beats:
            if not isinstance(beat, dict):
                cleaned_beats.append(beat)
                continue
            if list(beat.get("aware_npc_slugs") or []):
                row = dict(beat)
                row["aware_npc_slugs"] = []
                cleaned_beats.append(row)
                changed = True
            else:
                cleaned_beats.append(beat)
        if not changed:
            return scene_output
        out = dict(scene_output)
        out["beats"] = cleaned_beats
        return out

    def _consume_state_occupant_hints(
        self,
        campaign: "Campaign",
        campaign_state: dict[str, object],
        state_update: dict[str, object],
        character_updates: dict[str, object],
        *,
        player_state: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object], int]:
        if not isinstance(state_update, dict):
            return {}, character_updates if isinstance(character_updates, dict) else {}, 0
        if not isinstance(character_updates, dict):
            character_updates = {}
        characters = self.get_campaign_characters(campaign)
        consumed = 0
        for state_key, value in list(state_update.items()):
            if not isinstance(value, dict):
                continue
            for occupant_field in ("occupants", "occupied_by"):
                raw_list = value.get(occupant_field)
                if not isinstance(raw_list, list):
                    continue
                del value[occupant_field]
                consumed += 1
                for item in raw_list:
                    resolved_slug = self._resolve_existing_character_slug(
                        characters if isinstance(characters, dict) else {},
                        item,
                    )
                    if not resolved_slug:
                        continue
                    current_update = character_updates.get(resolved_slug)
                    if current_update is None:
                        current_update = {}
                    if not isinstance(current_update, dict):
                        continue
                    character_updates[resolved_slug] = current_update
        if consumed:
            self._zork_log(
                "STATE OCCUPANT HINTS CONSUMED",
                f"Removed {consumed} occupant field(s) from state_update and converted known NPCs to character_updates.",
            )
        return state_update, character_updates, consumed

    # ------------------------------------------------------------------
    # JSON repair utilities
    # ------------------------------------------------------------------

    @classmethod
    def _repair_unquoted_json_string_fields(cls, text: str) -> str:
        if not text:
            return text
        pattern = re.compile(
            r'("(?P<key>[^"]+)"\s*:\s*)'
            r'(?!true\s*[,}\]]|false\s*[,}\]]|null\s*[,}\]])'
            r'(?=[A-Za-z])'
            r'(?P<value>.*?)'
            r'(?=,\s*"[^"]+"\s*:|,\s*\{|\s*[}\]]|\s*$)',
            re.DOTALL,
        )

        def _replace(match: re.Match[str]) -> str:
            prefix = match.group(1)
            raw_value = str(match.group("value") or "").strip()
            if not raw_value:
                return match.group(0)
            if raw_value.endswith('"') and not raw_value.startswith('"'):
                raw_value = raw_value[:-1].strip()
            return f"{prefix}{json.dumps(raw_value, ensure_ascii=False)}"

        return pattern.sub(_replace, text)

    @classmethod
    def _repair_unquoted_json_keys(cls, text: str) -> str:
        if not text:
            return text
        pattern = re.compile(r'(?P<prefix>[{,]\s*)(?P<key>[A-Za-z_][A-Za-z0-9_-]*)(?P<suffix>\s*:)')

        def _replace(match: re.Match[str]) -> str:
            key = str(match.group("key") or "").strip()
            if not key:
                return match.group(0)
            return f'{match.group("prefix")}"{key}"{match.group("suffix")}'

        return pattern.sub(_replace, text)

    @classmethod
    def _repair_trailing_json_commas(cls, text: str) -> str:
        if not text:
            return text
        return re.sub(r",\s*([}\]])", r"\1", text)

    @classmethod
    def _repair_known_schema_string_fields(cls, text: str) -> str:
        if not text:
            return text
        field_names = "|".join(sorted(re.escape(name) for name in cls.KNOWN_JSON_STRING_FIELDS))
        pattern = re.compile(
            rf'("(?P<key>{field_names})"\s*:\s*)'
            r'(?!(?:"|\{|\[|true\s*[,}\]]|false\s*[,}\]]|null\s*[,}\]]|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*[,}\]]))'
            r'(?P<value>.*?)'
            r'(?=,\s*"[^"]+"\s*:|,\s*\{|\s*[}\]]|\s*$)',
            re.DOTALL,
        )

        def _replace(match: re.Match[str]) -> str:
            prefix = match.group(1)
            raw_value = str(match.group("value") or "").strip()
            if not raw_value:
                return match.group(0)
            if raw_value.endswith('"') and not raw_value.startswith('"'):
                raw_value = raw_value[:-1].strip()
            return f"{prefix}{json.dumps(raw_value, ensure_ascii=False)}"

        return pattern.sub(_replace, text)

    @classmethod
    def _repair_json_lenient_text(cls, text: str) -> str:
        repaired = str(text or "")
        repaired = cls._repair_unquoted_json_keys(repaired)
        repaired = cls._repair_trailing_json_commas(repaired)
        repaired = cls._repair_unquoted_json_string_fields(repaired)
        repaired = cls._repair_known_schema_string_fields(repaired)
        repaired = cls._repair_unmatched_json_closers(repaired)
        return repaired

    @classmethod
    def _repair_unmatched_json_closers(cls, text: str) -> str:
        raw = str(text or "")
        if not raw:
            return raw
        out: list[str] = []
        stack: list[str] = []
        in_string = False
        escape = False
        changed = False
        closing_for = {"{": "}", "[": "]"}
        opener_for = {"}": "{", "]": "["}

        for ch in raw:
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                out.append(ch)
                continue

            if ch in "{[":
                stack.append(ch)
                out.append(ch)
                continue

            if ch in "}]":
                expected_opener = opener_for[ch]
                if stack and stack[-1] == expected_opener:
                    stack.pop()
                    out.append(ch)
                    continue
                changed = True
                continue

            out.append(ch)

        if stack:
            changed = True
            while stack:
                out.append(closing_for[stack.pop()])

        repaired = "".join(out)
        return repaired if changed else raw

    # ------------------------------------------------------------------
    # Anti-echo system
    # ------------------------------------------------------------------

    @staticmethod
    def _anti_echo_tokens(text: str) -> set[str]:
        words = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower()).split()
        return {w for w in words if len(w) >= 4}

    @classmethod
    def _anti_echo_first_sentence(cls, text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return ""
        match = re.match(r"^(.*?[.!?])\s", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text[:200]

    @classmethod
    def _anti_echo_retry_decision(
        cls,
        action_text: str,
        narration_text: str,
        *,
        overlap_threshold: float = 0.45,
    ) -> bool:
        action_tokens = cls._anti_echo_tokens(action_text)
        if not action_tokens:
            return False
        first_sentence = cls._anti_echo_first_sentence(narration_text)
        narration_tokens = cls._anti_echo_tokens(first_sentence)
        if not narration_tokens:
            return False
        overlap = action_tokens & narration_tokens
        ratio = len(overlap) / max(len(action_tokens), 1)
        return ratio >= overlap_threshold

    # ------------------------------------------------------------------
    # Phone command handling
    # ------------------------------------------------------------------

    @staticmethod
    def _is_private_phone_command_line(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False
        phone_prefixes = (
            "text ",
            "sms ",
            "message ",
            "call ",
            "phone ",
            "check phone",
            "check my phone",
            "read text",
            "read message",
            "read sms",
        )
        return any(normalized.startswith(prefix) for prefix in phone_prefixes)

    @classmethod
    def _redact_private_phone_command_lines(
        cls,
        text: str,
    ) -> tuple[str, bool]:
        if not text:
            return text, False
        lines = str(text or "").splitlines()
        kept_lines: list[str] = []
        redacted = False
        for line in lines:
            if cls._is_private_phone_command_line(line):
                redacted = True
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip(), redacted

    # ------------------------------------------------------------------
    # Scene/room utilities
    # ------------------------------------------------------------------

    @classmethod
    def _scene_output_is_summary_public_safe(
        cls,
        scene_output: object,
    ) -> bool:
        if not isinstance(scene_output, dict):
            return True
        beats = scene_output.get("beats")
        if not isinstance(beats, list) or not beats:
            return True
        for beat in beats:
            if not isinstance(beat, dict):
                continue
            scope = str(beat.get("visibility") or "").strip().lower() or "local"
            if scope in {"private", "limited"}:
                return False
        return True

    @classmethod
    def _rescue_misplaced_room_state(
        cls,
        state_update: dict[str, object],
        player_state_update: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        if not isinstance(state_update, dict):
            return state_update, player_state_update
        if not isinstance(player_state_update, dict):
            player_state_update = {}
        existing_room_keys = cls.ROOM_STATE_KEYS & set(player_state_update)
        if len(existing_room_keys) >= 2:
            return state_update, player_state_update
        keys_to_remove = []
        for key, value in state_update.items():
            if not isinstance(value, dict):
                continue
            nested_room_keys = cls.ROOM_STATE_KEYS & set(value)
            if len(nested_room_keys) < 2:
                continue
            for rk in nested_room_keys:
                player_state_update[rk] = value.pop(rk)
            _zork_log(
                "RESCUED MISPLACED ROOM STATE",
                f"Moved {sorted(nested_room_keys)} from state_update.{key} "
                f"into player_state_update",
            )
            if not value:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del state_update[key]
        return state_update, player_state_update

    def _build_scene_avatar_references(
        self,
        campaign: "Campaign",
        actor: "Player | None",
        actor_state: dict[str, object],
    ) -> list[dict[str, object]]:
        if campaign is None or actor is None:
            return []
        refs: list[dict[str, object]] = []
        seen_urls: set[str] = set()
        with self._session_factory() as session:
            from text_game_engine.persistence.sqlalchemy.models import Player
            players = (
                session.query(Player)
                .filter_by(campaign_id=str(campaign.id))
                .all()
            )
            for entry in players:
                state = self.get_player_state(entry)
                if str(getattr(entry, "actor_id", "")) != str(getattr(actor, "actor_id", "")) and not self._same_scene(
                    actor_state, state
                ):
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
                identity = str(
                    state.get("character_name") or f"Adventurer-{str(getattr(entry, 'actor_id', ''))[-4:]}"
                ).strip()
                refs.append(
                    {
                        "actor_id": str(getattr(entry, "actor_id", "")),
                        "name": identity,
                        "url": avatar_url,
                        "is_actor": str(getattr(entry, "actor_id", "")) == str(getattr(actor, "actor_id", "")),
                    }
                )
                if len(refs) >= getattr(self, "MAX_SCENE_REFERENCE_IMAGES", 4) - 1:
                    break
        return refs

    @classmethod
    def _build_rails_context(
        cls,
        player_state: dict[str, object],
        party_snapshot: list[dict[str, object]],
    ) -> dict[str, object]:
        exits = player_state.get("exits")
        if not isinstance(exits, list):
            exits = []
        known_names = []
        for entry in party_snapshot:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            known_names.append(name)
        inventory_rich = cls._get_inventory_rich(player_state)[:20]
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
    def _inventory_origin_for_prompt(cls, value: object) -> str:
        text = " ".join(str(value or "").strip().split())
        if not text:
            return ""
        receipt_match = re.search(
            r"\(received\s+(Day\s+\d+\s*,\s*\d{2}:\d{2})\)",
            text,
            re.IGNORECASE,
        )
        if not receipt_match:
            receipt_match = re.search(
                r"\breceived\s+(Day\s+\d+\s*,\s*\d{2}:\d{2})\b",
                text,
                re.IGNORECASE,
            )
        receipt_suffix = ""
        if receipt_match:
            receipt_text = receipt_match.group(1)
            receipt_suffix = f" (received {receipt_text})"
        base_text = text
        if receipt_match:
            start, end = receipt_match.span()
            base_text = f"{text[:start]}{text[end:]}".strip(" ,.;")
        lower = text.lower()
        strong_prefixes = (
            "found ",
            "received ",
            "bought ",
            "stole ",
            "taken ",
            "took ",
            "picked up",
            "retrieved ",
            "gift ",
            "won ",
            "from ",
            "looted ",
            "borrowed ",
            "taken from ",
            "received from ",
        )
        def _truncate_preserving_receipt(raw: str, *, max_chars: int = 120) -> str:
            clean = " ".join(str(raw or "").strip().split())
            if not receipt_suffix:
                return clean[:max_chars]
            room = max_chars - len(receipt_suffix)
            if room <= 0:
                return receipt_suffix.strip()
            if len(clean) <= room:
                return f"{clean}{receipt_suffix}"
            if room <= 3:
                clipped = clean[:room]
            else:
                clipped = clean[: room - 3].rstrip() + "..."
            return f"{clipped}{receipt_suffix}"
        if any(lower.startswith(prefix) for prefix in strong_prefixes):
            return _truncate_preserving_receipt(base_text)
        if len(text.split()) <= 4 and not any(ch in text for ch in ".!?"):
            if lower.startswith(("from ", "in ", "at ")):
                return _truncate_preserving_receipt(base_text)
            return _truncate_preserving_receipt(f"From {base_text[:112]}".strip())
        if receipt_suffix:
            return f"Acquired earlier in-scene.{receipt_suffix}"
        return "Acquired earlier in-scene."

    @classmethod
    def _get_inventory_rich(cls, player_state: dict[str, object]) -> list[dict[str, object]]:
        raw = player_state.get("inventory") if isinstance(player_state, dict) else None
        if isinstance(raw, list):
            out = []
            seen: set[str] = set()
            for item in raw[:50]:
                if isinstance(item, dict):
                    name = str(
                        item.get("name") or item.get("item") or item.get("title") or ""
                    ).strip()
                    if not name:
                        continue
                    name_key = name.lower()
                    if name_key in seen:
                        continue
                    seen.add(name_key)
                    out.append(
                        {
                            "name": name,
                            "origin": cls._inventory_origin_for_prompt(item.get("origin")),
                        }
                    )
                elif isinstance(item, str) and item.strip():
                    name = item.strip()
                    name_key = name.lower()
                    if name_key in seen:
                        continue
                    seen.add(name_key)
                    out.append({"name": name, "origin": ""})
            return out
        if isinstance(raw, dict):
            out = []
            seen: set[str] = set()
            for key, value in raw.items():
                if isinstance(value, dict):
                    name = str(
                        value.get("name") or value.get("item") or value.get("title") or key
                    ).strip()
                    if not name:
                        continue
                    name_key = name.lower()
                    if name_key in seen:
                        continue
                    seen.add(name_key)
                    out.append(
                        {
                            "name": name,
                            "origin": cls._inventory_origin_for_prompt(value.get("origin")),
                        }
                    )
                else:
                    name = str(key).strip()
                    if not name:
                        continue
                    name_key = name.lower()
                    if name_key in seen:
                        continue
                    seen.add(name_key)
                    out.append(
                        {
                            "name": name,
                            "origin": cls._inventory_origin_for_prompt(value),
                        }
                    )
            return out[:50]
        return []

    @classmethod
    def _state_container_matches_location(cls, container_text: str, location: str) -> bool:
        c = cls._normalize_location_key(container_text)
        l = cls._normalize_location_key(location)
        if not c or not l:
            return False
        return c in l or l in c

    def _canonical_location_for_state_container(
        self,
        campaign: "Campaign",
        campaign_state: dict[str, object],
        container_key: object,
        *,
        player_state: dict[str, object] | None = None,
        player_registry: dict | None = None,
        characters: dict[str, dict] | None = None,
    ) -> str | None:
        container_text = str(container_key or "").strip()
        if not container_text:
            return None
        candidates: set[str] = set()
        if isinstance(player_state, dict):
            loc = str(player_state.get("location") or "").strip()
            if loc and self._state_container_matches_location(container_text, loc):
                candidates.add(loc)
        registry = player_registry or self._campaign_player_registry(campaign.id)
        by_actor_id = registry.get("by_actor_id", registry.get("by_user_id", {}))
        if isinstance(by_actor_id, dict):
            with self._session_factory() as session:
                from text_game_engine.persistence.sqlalchemy.models import Player
                for aid in by_actor_id:
                    row = session.query(Player).filter_by(
                        campaign_id=str(campaign.id),
                        actor_id=str(aid),
                    ).first()
                    if row is None:
                        continue
                    loc = str(self.get_player_state(row).get("location") or "").strip()
                    if loc and self._state_container_matches_location(container_text, loc):
                        candidates.add(loc)
        char_map = characters if isinstance(characters, dict) else self.get_campaign_characters(campaign)
        if isinstance(char_map, dict):
            for payload in char_map.values():
                if isinstance(payload, dict):
                    loc = str(payload.get("location") or "").strip()
                    if loc and self._state_container_matches_location(container_text, loc):
                        candidates.add(loc)
        if isinstance(campaign_state, dict):
            for payload in campaign_state.values():
                if isinstance(payload, dict):
                    loc = str(payload.get("location") or "").strip()
                    if loc and self._state_container_matches_location(container_text, loc):
                        candidates.add(loc)
        if len(candidates) == 1:
            return next(iter(candidates))
        fallback = str(container_text).replace("_", "-").strip("-")
        return fallback or None

    def _character_update_hits_player(
        self,
        campaign_id: str,
        slug: str,
        payload: object,
    ) -> dict[str, object] | None:
        registry = self._campaign_player_registry(campaign_id)
        by_slug = registry.get("by_slug", {})
        normalized = self._player_slug_key(slug)
        match = by_slug.get(normalized)
        if match is not None:
            return match
        if isinstance(payload, dict):
            name = str(payload.get("name") or "").strip()
            if name:
                name_slug = self._player_slug_key(name)
                match = by_slug.get(name_slug)
                if match is not None:
                    return match
        return None

    def _resolve_sms_recipient(
        self,
        campaign_id: str,
        recipient_text: str,
    ) -> dict[str, object] | None:
        text = " ".join(str(recipient_text or "").strip().lower().split())
        if not text:
            return None
        registry = self._campaign_player_registry(campaign_id)
        by_slug = registry.get("by_slug", {})
        by_actor_id = registry.get("by_actor_id", registry.get("by_user_id", {}))
        slug_key = self._player_slug_key(text)
        if slug_key and slug_key in by_slug:
            return by_slug[slug_key]
        for aid, entry in by_actor_id.items():
            name = str(entry.get("name") or "").strip()
            if name and self._normalize_match_text(name) == self._normalize_match_text(text):
                return entry
        characters = self.get_campaign_characters(None)  # caller should supply campaign
        if isinstance(characters, dict):
            for slug, payload in characters.items():
                if not isinstance(payload, dict):
                    continue
                name = str(payload.get("name") or "").strip()
                if name and self._normalize_match_text(name) == self._normalize_match_text(text):
                    return {"kind": "npc", "slug": slug, "name": name}
        return None

    # ------------------------------------------------------------------
    # Game Time System
    # ------------------------------------------------------------------

    @staticmethod
    def _game_period_from_hour(hour: int) -> str:
        if 5 <= hour <= 11:
            return "morning"
        if 12 <= hour <= 16:
            return "afternoon"
        if 17 <= hour <= 20:
            return "evening"
        return "night"

    def _game_time_to_total_minutes(self, game_time: Dict[str, int]) -> int:
        day = self._coerce_non_negative_int(game_time.get("day", 1), default=1) or 1
        hour = min(23, max(0, self._coerce_non_negative_int(game_time.get("hour", 0), default=0)))
        minute = min(59, max(0, self._coerce_non_negative_int(game_time.get("minute", 0), default=0)))
        return ((max(1, day) - 1) * 24 * 60) + (hour * 60) + minute

    def _game_time_from_total_minutes(
        self,
        total_minutes: int,
        *,
        start_day_of_week: str = "monday",
    ) -> Dict[str, object]:
        total = max(0, int(total_minutes))
        day = (total // (24 * 60)) + 1
        within = total % (24 * 60)
        hour = within // 60
        minute = within % 60
        period = self._game_period_from_hour(hour)
        day_of_week = self._weekday_for_day(
            day=day,
            start_day_of_week=start_day_of_week,
        )
        return {
            "day": day,
            "hour": hour,
            "minute": minute,
            "day_of_week": day_of_week,
            "period": period,
            "date_label": f"{day_of_week.title()}, Day {day}, {period.title()}",
        }

    def _infer_start_day_of_week_from_game_time(
        self,
        game_time: object,
    ) -> str | None:
        if not isinstance(game_time, dict):
            return None
        weekday = self._normalize_weekday_name(game_time.get("day_of_week"))
        day = self._coerce_non_negative_int(game_time.get("day", 1), default=1) or 1
        if weekday not in self.WEEKDAY_NAMES:
            return None
        try:
            weekday_index = self.WEEKDAY_NAMES.index(weekday)
        except ValueError:
            return None
        start_index = (weekday_index - (day - 1)) % len(self.WEEKDAY_NAMES)
        return self.WEEKDAY_NAMES[start_index]

    def _format_game_time_label(
        self,
        game_time: Dict[str, int],
        *,
        campaign_state: Optional[Dict[str, object]] = None,
    ) -> str:
        snapshot = (
            game_time
            if isinstance(game_time, dict)
            else self._extract_game_time_snapshot({"game_time": game_time})
        )
        start_day_of_week = None
        if isinstance(campaign_state, dict):
            start_day_of_week = self._campaign_start_day_of_week(campaign_state)
        else:
            start_day_of_week = self._infer_start_day_of_week_from_game_time(snapshot)
        if not start_day_of_week:
            existing_label = str(snapshot.get("date_label") or "").strip() if isinstance(snapshot, dict) else ""
            if existing_label:
                return existing_label
            start_day_of_week = "monday"
        canonical = self._game_time_from_total_minutes(
            self._game_time_to_total_minutes(snapshot),
            start_day_of_week=start_day_of_week,
        )
        return str(
            canonical.get("date_label")
            or (
                f"{str(canonical.get('day_of_week') or 'monday').title()}, "
                f"Day {canonical.get('day', 1)}, "
                f"{str(canonical.get('period') or 'time').title()}"
            )
        ).strip()

    def _speed_multiplier_from_state(self, campaign_state: Dict[str, object]) -> float:
        raw = 1.0
        if isinstance(campaign_state, dict):
            raw = campaign_state.get("speed_multiplier", 1.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 1.0
        if value <= 0:
            return 1.0
        return max(0.1, min(10.0, value))

    @classmethod
    def _time_model_from_state(cls, campaign_state: Dict[str, object] | None) -> str:
        raw = str((campaign_state or {}).get("time_model") or "").strip().lower()
        if raw == cls.TIME_MODEL_INDIVIDUAL_CLOCKS:
            return cls.TIME_MODEL_INDIVIDUAL_CLOCKS
        return cls.TIME_MODEL_SHARED_CLOCK

    @classmethod
    def _calendar_policy_from_state(cls, campaign_state: Dict[str, object] | None) -> str:
        raw = str((campaign_state or {}).get("calendar_policy") or "").strip().lower()
        if raw == cls.CALENDAR_POLICY_LOOSE:
            return cls.CALENDAR_POLICY_LOOSE
        return cls.CALENDAR_POLICY_CONSEQUENTIAL

    @classmethod
    def _calendar_event_targets_player(
        cls,
        event: object,
        *,
        actor_id: object = None,
        player_state: dict[str, object] | None = None,
    ) -> bool:
        target_tokens = cls._calendar_target_tokens_from_event(event)
        if not target_tokens:
            return False
        aliases: set[str] = set()

        def _add(raw: object) -> None:
            text = str(raw or "").strip()
            if not text:
                return
            aliases.add(" ".join(text.lower().split())[:160])
            slug_key = cls._player_slug_key(text)
            if slug_key:
                aliases.add(slug_key[:160])

        _add(actor_id)
        _add(cls._player_visibility_slug(actor_id))
        if isinstance(actor_id, str) and actor_id.strip():
            _add(f"<@{actor_id.strip()}>")
        if isinstance(player_state, dict):
            _add(player_state.get("character_name"))
        for token in target_tokens:
            normalized = " ".join(str(token or "").strip().lower().split())[:160]
            slug_key = cls._player_slug_key(token)
            if normalized in aliases or (slug_key and slug_key in aliases):
                return True
        return False

    def _current_game_time_for_prompt(
        self,
        campaign_state: dict[str, object],
        player_state: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object], str, str]:
        global_game_time = self._game_time_from_total_minutes(
            self._game_time_to_total_minutes(campaign_state.get("game_time") or {}),
            start_day_of_week=self._campaign_start_day_of_week(campaign_state),
        )
        time_model = self._time_model_from_state(campaign_state)
        calendar_policy = self._calendar_policy_from_state(campaign_state)
        if time_model == self.TIME_MODEL_INDIVIDUAL_CLOCKS:
            raw_player_time = player_state.get("game_time") if isinstance(player_state, dict) else None
            if isinstance(raw_player_time, dict) and raw_player_time:
                player_game_time = self._game_time_from_total_minutes(
                    self._game_time_to_total_minutes(raw_player_time),
                    start_day_of_week=self._campaign_start_day_of_week(campaign_state),
                )
                return player_game_time, global_game_time, time_model, calendar_policy
        return global_game_time, global_game_time, time_model, calendar_policy

    def _effective_min_turn_advance_minutes(self, speed_multiplier: float) -> int:
        try:
            speed = float(speed_multiplier)
        except (TypeError, ValueError):
            speed = 1.0
        return max(1, int(round(self.MIN_TURN_ADVANCE_MINUTES * speed)))

    def _effective_standard_turn_advance_minutes(self, speed_multiplier: float) -> int:
        try:
            speed = float(speed_multiplier)
        except (TypeError, ValueError):
            speed = 1.0
        scaled_default = int(round(self.DEFAULT_TURN_ADVANCE_MINUTES * speed))
        return max(self._effective_min_turn_advance_minutes(speed), scaled_default)

    def _turn_time_beat_guidance(self, min_turn_minutes: int) -> str:
        try:
            minutes = max(0, int(min_turn_minutes))
        except (TypeError, ValueError):
            minutes = self.MIN_TURN_ADVANCE_MINUTES
        for bucket in self.TURN_TIME_BEAT_GUIDANCE_BUCKETS:
            if minutes < bucket.min_minutes:
                continue
            if bucket.max_minutes is None or minutes <= bucket.max_minutes:
                return bucket.rule_text
        return self.TURN_TIME_BEAT_GUIDANCE_BUCKETS[-1].rule_text

    def _estimate_turn_time_advance_minutes(
        self, action_text: str, narration_text: str
    ) -> int:
        action_l = str(action_text or "").lower()
        combined = f"{action_l}\n{str(narration_text or '').lower()}"
        if any(token in combined for token in ("time skip", "timeskip", "time-skip")):
            return 60
        if any(
            token in combined
            for token in (
                "sleep",
                "rest",
                "nap",
                "wait",
                "travel",
                "drive",
                "ride",
                "fly",
                "train",
                "journey",
            )
        ):
            return 30
        if any(
            token in combined
            for token in ("fight", "combat", "attack", "shoot", "chase", "run")
        ):
            return 8
        if any(
            token in action_l
            for token in ("look", "examine", "inspect", "ask", "say", "talk")
        ):
            return 3
        return self.DEFAULT_TURN_ADVANCE_MINUTES

    def _extract_time_skip_request(
        self,
        action_text: object,
    ) -> Optional[Dict[str, object]]:
        text = " ".join(str(action_text or "").strip().split())
        if not text:
            return None
        match = re.match(
            r"^(?:time[\s-]*skip|timeskip)\b(?:\s+(.*))?$",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        desc = str(match.group(1) or "").strip()
        minutes = 60
        total = 0
        found = False
        for raw_value, unit in re.findall(
            r"(\d+(?:\.\d+)?)\s*(d(?:ays?)?|h(?:ours?|rs?)?|m(?:in(?:ute)?s?)?)\b",
            desc,
            flags=re.IGNORECASE,
        ):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            unit_l = unit.lower()
            if unit_l.startswith("d"):
                total += int(round(value * 24 * 60))
            elif unit_l.startswith("h"):
                total += int(round(value * 60))
            else:
                total += int(round(value))
            found = True
        if found and total > 0:
            minutes = total
        minutes = max(1, min(7 * 24 * 60, int(minutes)))
        return {
            "minutes": minutes,
            "description": desc,
        }

    def _is_ooc_action_text(self, action_text: object) -> bool:
        return bool(re.match(r"\s*\[OOC\b", str(action_text or ""), re.IGNORECASE))

    @staticmethod
    def _ooc_help_requested(action_text: object) -> bool:
        text = " ".join(str(action_text or "").strip().lower().split())
        if not text:
            return False
        markers = (
            "hint",
            "stuck",
            "confused",
            "unclear",
            "lost",
            "what do i do",
            "what should i do",
            "where do i go",
            "railroad",
            "railroading",
            "forced",
            "pushy",
            "guide me",
            "nudge",
            "help me",
        )
        return any(marker in text for marker in markers)

    def _ooc_prompt_extra_lines(
        self,
        campaign: Campaign,
        campaign_state: Dict[str, object],
        player: Player,
        player_state: Dict[str, object],
        action_text: str,
    ) -> list[str]:
        if not self._is_ooc_action_text(action_text):
            return []

        lines = [
            "OOC_DIRECTIVE: PLAYER_ACTION is out-of-character direct player-to-GM communication.",
            "You MUST address the OOC substance directly in this response instead of continuing the same pressure unchanged.",
            "Treat OOC hint/stuck/railroading feedback as authoritative signal about player experience.",
            "Give concrete, actionable guidance or scene adjustment grounded in visible facts. Do not hand-wave, moralize, or repeat the blocked premise.",
            "Default to little or no in-world advancement on this turn unless the player explicitly requested both OOC clarification and an in-world action.",
        ]
        if not self._ooc_help_requested(action_text):
            return lines

        viewer_slug = self._player_slug_key(player_state.get("character_name"))
        viewer_location_key = self._room_key_from_player_state(player_state).lower()
        visible_hints = self._plot_hints_for_viewer(
            campaign_state,
            viewer_actor_id=str(player.actor_id or ""),
            viewer_slug=viewer_slug,
            viewer_location_key=viewer_location_key,
            limit=5,
        )
        visible_threads = []
        for row in self._plot_threads_from_state(campaign_state).values():
            if not isinstance(row, dict):
                continue
            if str(row.get("status") or "") != "active":
                continue
            if not self._plot_thread_visible_to_viewer(
                row,
                viewer_actor_id=str(player.actor_id or ""),
                viewer_slug=viewer_slug,
                viewer_location_key=viewer_location_key,
            ):
                continue
            visible_threads.append(row)
        if visible_hints:
            lines.append(
                f"OOC_VISIBLE_HINTS: {self._dump_json(visible_hints[:5])}"
            )
        elif visible_threads:
            fallback_threads = []
            for row in visible_threads[:5]:
                fallback_threads.append(
                    {
                        "thread": str(row.get('thread') or '').strip(),
                        "setup": " ".join(str(row.get("setup") or "").split())[:220],
                        "hint": " ".join(str(row.get("hint") or "").split())[:220],
                    }
                )
            lines.append(
                f"OOC_ACTIVE_VISIBLE_THREADS: {self._dump_json(fallback_threads)}"
            )
        lines.append(
            "If the player asked for help, respond with specific next-step options or an explicit clue, not generic encouragement."
        )
        return lines

    # -- Passive NPC reuse / SMS follow-up nudges --
    # Probability constants for per-turn nudge rolls.
    NPC_NUDGE_CHANCE = 0.30       # 30% chance to nudge NPC reuse
    SMS_REPLY_NUDGE_CHANCE = 0.40  # 40% chance to nudge SMS reply when unanswered threads exist

    def _passive_npc_sms_nudge_lines(
        self,
        campaign: Campaign,
        campaign_state: Dict[str, object],
        player: Player,
        player_state: Dict[str, object],
    ) -> list[str]:
        """Generate prompt nudge lines encouraging passive NPC activity.

        Called on every turn. Uses probability rolls to decide whether to
        inject a nudge for:
        - Bringing an off-scene NPC back into the current scene
        - Having an NPC reply to or initiate an SMS conversation
        """
        lines: list[str] = []

        # --- Passive SMS reply nudge ---
        if random.random() < self.SMS_REPLY_NUDGE_CHANCE:
            actor_id = str(player.actor_id or "")
            unread_summary = self._sms_unread_summary_for_player(
                campaign_state,
                actor_id=actor_id,
                player_state=player_state,
            )
            unread_threads = self._coerce_non_negative_int(
                unread_summary.get("threads", 0), default=0
            )
            # Also check for threads where the last message was FROM the player (unanswered by NPC)
            sms_threads = self._sms_threads_from_state(campaign_state)
            player_aliases = self._sms_player_aliases(actor_id=actor_id, player_state=player_state)
            contact_roster = self._sms_contact_roster(campaign)
            unanswered_by_npc: list[str] = []
            unanswered_seen: set[str] = set()
            for thread_key, thread_data in sms_threads.items():
                if not isinstance(thread_data, dict):
                    continue
                messages = thread_data.get("messages")
                if not isinstance(messages, list) or not messages:
                    continue
                last_msg = messages[-1] if messages else None
                if not isinstance(last_msg, dict):
                    continue
                from_norm = self._sms_normalize_thread_key(last_msg.get("from"))
                if from_norm and from_norm in player_aliases:
                    resolved_contact = self._sms_resolved_contact(
                        str(thread_key),
                        thread_data,
                        viewer_actor_id=actor_id,
                        player_state=player_state,
                        contact_roster=contact_roster,
                        visible_messages=messages,
                    )
                    canonical_thread = self._sms_normalize_thread_key(
                        resolved_contact.get("thread") or thread_key
                    )
                    if canonical_thread and canonical_thread in unanswered_seen:
                        continue
                    if canonical_thread:
                        unanswered_seen.add(canonical_thread)
                    label = str(
                        resolved_contact.get("label")
                        or thread_data.get("label")
                        or thread_key
                    ).strip()[:40]
                    if label:
                        unanswered_by_npc.append(label)
            if unanswered_by_npc:
                picks = unanswered_by_npc[:2]
                names_str = " and ".join(picks)
                lines.append(
                    f"SMS_REPLY_NUDGE: The player has unanswered outgoing SMS to {names_str}. "
                    f"Consider having the NPC reply via sms_write during this turn — "
                    f"people generally respond to texts, and silence without narrative reason feels like a dropped thread. "
                    f"If the NPC would plausibly delay, narrate a brief reason (busy, thinking, ignoring)."
                )
            elif unread_threads > 0:
                labels = unread_summary.get("labels", [])
                if labels:
                    names_str = ", ".join(str(l) for l in labels[:2])
                    lines.append(
                        f"SMS_ACTIVITY_NUDGE: There are unread SMS threads from {names_str}. "
                        f"If narratively appropriate, the character's phone could buzz, "
                        f"or an NPC could mention they tried to reach the player."
                    )

        return lines

    def _increment_auto_fix_counter(
        self,
        campaign_state: Dict[str, object],
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
        counters = campaign_state.get(self.AUTO_FIX_COUNTERS_KEY)
        if not isinstance(counters, dict):
            counters = {}
            campaign_state[self.AUTO_FIX_COUNTERS_KEY] = counters
        current = self._coerce_non_negative_int(counters.get(safe_key, 0), default=0)
        counters[safe_key] = current + safe_amount

    def _ensure_game_time_progress(
        self,
        campaign_state: Dict[str, object],
        pre_turn_game_time: Dict[str, int],
        *,
        action_text: str,
        narration_text: str,
    ) -> Dict[str, object]:
        if not isinstance(campaign_state, dict):
            return campaign_state
        pre_snapshot = (
            pre_turn_game_time
            if isinstance(pre_turn_game_time, dict)
            else self._extract_game_time_snapshot(campaign_state)
        )
        cur_snapshot = self._extract_game_time_snapshot(campaign_state)
        pre_total = self._game_time_to_total_minutes(pre_snapshot)
        cur_total = self._game_time_to_total_minutes(cur_snapshot)
        start_day_of_week = self._campaign_start_day_of_week(campaign_state)
        time_skip_request = self._extract_time_skip_request(action_text)

        # Keep derived fields canonical when model already advanced time.
        if cur_total > pre_total:
            if time_skip_request is None and not self._is_ooc_action_text(action_text):
                speed_multiplier = self._speed_multiplier_from_state(campaign_state)
                min_step = self._effective_min_turn_advance_minutes(speed_multiplier)
                cur_total = max(cur_total, pre_total + min_step)
            campaign_state["game_time"] = self._game_time_from_total_minutes(
                cur_total,
                start_day_of_week=start_day_of_week,
            )
            return campaign_state

        # Meta/OOC turns do not auto-advance in-game time.
        if self._is_ooc_action_text(action_text):
            campaign_state["game_time"] = self._game_time_from_total_minutes(
                cur_total,
                start_day_of_week=start_day_of_week,
            )
            return campaign_state

        if time_skip_request is not None:
            base_minutes = self._coerce_non_negative_int(
                time_skip_request.get("minutes", 60), default=60
            )
        else:
            base_minutes = self._estimate_turn_time_advance_minutes(
                action_text, narration_text
            )
        speed_multiplier = self._speed_multiplier_from_state(campaign_state)
        min_step = self._effective_min_turn_advance_minutes(speed_multiplier)
        scaled_minutes = int(round(base_minutes * speed_multiplier))
        delta_minutes = max(min_step, scaled_minutes)
        if time_skip_request is None:
            delta_minutes = min(self.MAX_TURN_ADVANCE_MINUTES, delta_minutes)
        else:
            delta_minutes = min(7 * 24 * 60, delta_minutes)

        # If model froze or regressed time, force monotonic advance from pre-turn time.
        new_total = max(pre_total, cur_total) + delta_minutes
        campaign_state["game_time"] = self._game_time_from_total_minutes(
            new_total,
            start_day_of_week=start_day_of_week,
        )
        self._increment_auto_fix_counter(
            campaign_state,
            "game_time_auto_advance",
        )
        self._zork_log(
            "GAME TIME AUTO-ADVANCE",
            (
                f"pre=Day {pre_snapshot.get('day')} {int(pre_snapshot.get('hour', 0)):02d}:"
                f"{int(pre_snapshot.get('minute', 0)):02d} "
                f"post_model=Day {cur_snapshot.get('day')} {int(cur_snapshot.get('hour', 0)):02d}:"
                f"{int(cur_snapshot.get('minute', 0)):02d} "
                f"delta_min={delta_minutes} speed={speed_multiplier}"
                + (
                    f" explicit_time_skip={time_skip_request.get('minutes')}"
                    if time_skip_request is not None
                    else ""
                )
            ),
        )
        return campaign_state

    def _record_turn_game_time(
        self,
        campaign_state: Dict[str, object],
        turn_id: Optional[int],
        game_time: Optional[Dict[str, int]],
    ) -> None:
        if not isinstance(campaign_state, dict) or turn_id is None:
            return
        if not isinstance(game_time, dict):
            return
        turn_key = str(int(turn_id))
        index = campaign_state.get(self.TURN_TIME_INDEX_KEY)
        if not isinstance(index, dict):
            index = {}
            campaign_state[self.TURN_TIME_INDEX_KEY] = index
        index[turn_key] = {
            "day": self._coerce_non_negative_int(game_time.get("day", 1), default=1) or 1,
            "hour": min(
                23, max(0, self._coerce_non_negative_int(game_time.get("hour", 0), default=0))
            ),
            "minute": min(
                59, max(0, self._coerce_non_negative_int(game_time.get("minute", 0), default=0))
            ),
        }
        if len(index) > self.MAX_TURN_TIME_ENTRIES:
            keyed = []
            for key in index.keys():
                try:
                    keyed.append((int(key), key))
                except (TypeError, ValueError):
                    continue
            keyed.sort()
            to_drop = len(index) - self.MAX_TURN_TIME_ENTRIES
            for _, key in keyed[:to_drop]:
                index.pop(key, None)

    def _action_requests_clock_time(self, action_text: str) -> bool:
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

    def _narration_has_explicit_clock_time(self, narration_text: str) -> bool:
        text = str(narration_text or "")
        if not text:
            return False
        return bool(re.search(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", text))

    def _player_known_game_time(
        self,
        player: Optional[Player],
        *,
        fallback_time: Optional[Dict[str, int]] = None,
    ) -> Dict[str, int]:
        fallback_snapshot = self._extract_game_time_snapshot(
            {"game_time": fallback_time or {}}
        )
        if player is None:
            return fallback_snapshot
        player_state = self.get_player_state(player)
        known_time = player_state.get("game_time")
        if not isinstance(known_time, dict) or not known_time:
            return fallback_snapshot
        return self._extract_game_time_snapshot({"game_time": known_time})

    def _set_player_known_game_time(
        self,
        player: Player,
        game_time: Dict[str, int],
        *,
        campaign_state: Optional[Dict[str, object]] = None,
        start_day_of_week: str | None = None,
    ) -> None:
        player_state = self.get_player_state(player)
        resolved_start_day = start_day_of_week
        if not resolved_start_day and isinstance(campaign_state, dict):
            resolved_start_day = self._campaign_start_day_of_week(campaign_state)
        if not resolved_start_day:
            resolved_start_day = self._infer_start_day_of_week_from_game_time(game_time)
        if not resolved_start_day:
            resolved_start_day = "monday"
        player_state["game_time"] = self._game_time_from_total_minutes(
            self._game_time_to_total_minutes(game_time),
            start_day_of_week=resolved_start_day,
        )
        with self._session_factory() as session:
            row = session.get(Player, player.id)
            if row is not None:
                row.state_json = self._dump_json(player_state)
                row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.commit()
                player.state_json = row.state_json

    def _sync_game_time_to_player_state(
        self,
        state_update: Dict[str, object],
        player_state_update: Dict[str, object],
    ) -> Dict[str, object]:
        """Copy game_time from state_update into player_state_update.

        The model places game_time in state_update (campaign-wide), but the
        player card is built from player_state which is only populated via
        player_state_update.  Without this bridge the player card's game_time
        goes stale.
        """
        if not isinstance(state_update, dict) or not isinstance(player_state_update, dict):
            return player_state_update
        gt = state_update.get("game_time")
        if isinstance(gt, dict) and gt:
            player_state_update.setdefault("game_time", gt)
        return player_state_update

    # ── Location Sync ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_location_text(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    def _location_state_key(self, value: object) -> str:
        text = self._normalize_location_text(value).lower()
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:100]

    def _active_location_modifications_for_prompt(
        self,
        campaign_state: Dict[str, object],
        player_state: Dict[str, object],
    ) -> Dict[str, object]:
        if not isinstance(campaign_state, dict) or not isinstance(player_state, dict):
            return {}
        raw_locations = campaign_state.get("locations")
        if not isinstance(raw_locations, dict):
            return {}
        candidate_keys: List[str] = []
        for raw in (
            player_state.get("location"),
            player_state.get("room_title"),
            player_state.get("room_summary"),
        ):
            key = self._location_state_key(raw)
            if key and key not in candidate_keys:
                candidate_keys.append(key)
        if not candidate_keys:
            return {}
        for key in candidate_keys:
            row = raw_locations.get(key)
            if not isinstance(row, dict):
                continue
            mods = row.get("modifications")
            if isinstance(mods, list):
                clean_mods = []
                for item in mods[:24]:
                    item_text = " ".join(str(item or "").strip().split())[:180]
                    if item_text:
                        clean_mods.append(item_text)
                if clean_mods:
                    return {
                        "location_key": key,
                        "modifications": clean_mods,
                    }
            elif isinstance(mods, dict) and mods:
                return {
                    "location_key": key,
                    "modifications": mods,
                }
        return {}

    def _resolve_player_location_for_state_sync(
        self, player_state: Dict[str, object]
    ) -> str:
        if not isinstance(player_state, dict):
            return ""
        for key in ("location", "room_title", "room_summary"):
            text = self._normalize_location_text(player_state.get(key))
            if text:
                return text[:160]
        return ""

    def _player_entity_aliases_for_state_sync(
        self,
        player_state: Dict[str, object],
    ) -> List[str]:
        aliases: List[str] = []
        seen: set[str] = set()

        def _add(raw: object) -> None:
            slug = self._player_slug_key(raw)
            if not slug or slug in seen:
                return
            seen.add(slug)
            aliases.append(slug)

        name = str(player_state.get("character_name") or "").strip()
        if not name:
            return aliases
        _add(name)
        for sep in (",", "(", " - "):
            if sep in name:
                _add(name.split(sep, 1)[0].strip())
        words = [part for part in re.split(r"\s+", name) if part]
        if words:
            _add(words[0])
        return aliases

    def _sync_player_states_from_campaign_entities(
        self,
        campaign: Campaign,
        campaign_state: Dict[str, object],
        *,
        skip_actor_id: Optional[str] = None,
    ) -> int:
        if not isinstance(campaign_state, dict):
            return 0
        synced = 0
        relevant_keys = (
            "location",
            "room_title",
            "room_summary",
            "room_description",
            "exits",
            "current_status",
        )
        with self._session_factory() as session:
            players = session.query(Player).filter(Player.campaign_id == campaign.id).all()
            for target in players:
                actor_id = getattr(target, "actor_id", None)
                if skip_actor_id is not None and actor_id == skip_actor_id:
                    continue
                target_state = self.get_player_state(target)
                entity_state = None
                for alias in self._player_entity_aliases_for_state_sync(target_state):
                    candidate = campaign_state.get(alias)
                    if isinstance(candidate, dict) and any(
                        key in candidate for key in relevant_keys
                    ):
                        entity_state = candidate
                        break
                if not isinstance(entity_state, dict):
                    continue

                changed = False
                old_location = str(target_state.get("location") or "").strip()
                new_location = str(entity_state.get("location") or "").strip()
                if new_location and new_location != old_location:
                    target_state["location"] = new_location
                    changed = True
                    if "room_title" in entity_state:
                        target_state["room_title"] = entity_state.get("room_title")
                    else:
                        target_state.pop("room_title", None)
                    if "room_summary" in entity_state:
                        target_state["room_summary"] = entity_state.get("room_summary")
                    else:
                        target_state.pop("room_summary", None)
                    if "room_description" in entity_state:
                        target_state["room_description"] = entity_state.get("room_description")
                    else:
                        target_state.pop("room_description", None)
                    if "exits" in entity_state:
                        target_state["exits"] = entity_state.get("exits")
                    else:
                        target_state.pop("exits", None)

                for key in ("current_status",):
                    if key in entity_state and entity_state.get(key) != target_state.get(key):
                        target_state[key] = entity_state.get(key)
                        changed = True

                if changed:
                    target.state_json = self._dump_json(target_state)
                    synced += 1
            session.commit()
        return synced

    def _normalize_co_located_player_slugs(
        self,
        raw_value: object,
        *,
        actor_slug: str = "",
    ) -> List[str]:
        if not isinstance(raw_value, list):
            return []
        actor_slug_key = self._player_slug_key(actor_slug)
        out: List[str] = []
        seen: set[str] = set()
        for item in raw_value:
            slug = self._player_slug_key(item)
            if not slug or slug == actor_slug_key or slug in seen:
                continue
            seen.add(slug)
            out.append(slug)
        return out

    def _sync_marked_co_located_players(
        self,
        campaign_id: str,
        source_actor_id: str,
        source_state: Dict[str, object],
        player_slugs: List[str],
    ) -> int:
        if not isinstance(source_state, dict) or not isinstance(player_slugs, list):
            return 0
        slug_set = {
            self._player_slug_key(item)
            for item in player_slugs
            if self._player_slug_key(item)
        }
        if not slug_set:
            return 0
        has_room_context = any(
            source_state.get(key)
            for key in ("room_id", "location", "room_title", "room_summary", "room_description")
        )
        if not has_room_context:
            return 0
        changed = 0
        with self._session_factory() as session:
            targets = (
                session.query(Player).filter(
                    Player.campaign_id == campaign_id,
                    Player.actor_id != source_actor_id,
                )
                .all()
            )
            for target in targets:
                target_state = self.get_player_state(target)
                target_slug = self._player_slug_key(target_state.get("character_name"))
                fallback_slug = self._player_slug_key(f"player-{getattr(target, 'actor_id', '')}")
                if target_slug not in slug_set and fallback_slug not in slug_set:
                    continue
                before = dict(target_state)
                for key in self.ROOM_STATE_KEYS:
                    src_val = source_state.get(key)
                    if src_val is None:
                        target_state.pop(key, None)
                    else:
                        target_state[key] = src_val
                if target_state == before:
                    continue
                target.state_json = self._dump_json(target_state)
                target.last_active_at = datetime.now(timezone.utc).replace(tzinfo=None)
                changed += 1
            session.commit()
        return changed

    def _auto_sync_companion_locations(
        self,
        campaign_state: Dict[str, object],
        *,
        player_state: Dict[str, object],
        narration_text: str,
    ) -> int:
        if not isinstance(campaign_state, dict):
            return 0
        player_location = self._resolve_player_location_for_state_sync(player_state)
        if not player_location:
            return 0
        changed = 0
        for raw_key, raw_value in campaign_state.items():
            key = str(raw_key or "")
            if key in self.MODEL_STATE_EXCLUDE_KEYS:
                continue
            if not isinstance(raw_value, dict):
                continue
            if "location" not in raw_value:
                continue
            current_location = self._normalize_location_text(raw_value.get("location"))
            if not current_location or current_location == player_location:
                continue
            follows_flag = any(
                bool(raw_value.get(flag))
                for flag in (
                    "follows_player",
                    "following_player",
                    "with_player",
                    "companion",
                    "pet",
                    "at_heels",
                )
            )
            if not follows_flag:
                names = self._entity_name_candidates_for_sync(key, raw_value)
                if not self._narration_implies_entity_with_player(
                    narration_text, names
                ):
                    continue
            raw_value["location"] = player_location
            changed += 1
        return changed

    def _auto_sync_character_locations(
        self,
        campaign: Campaign,
        *,
        player_state: Dict[str, object],
        narration_text: str,
    ) -> int:
        player_location = self._resolve_player_location_for_state_sync(player_state)
        if not player_location:
            return 0
        characters = self.get_campaign_characters(campaign)
        if not isinstance(characters, dict) or not characters:
            return 0
        changed = 0
        for slug, entry in characters.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("deceased_reason"):
                continue
            current_location = self._normalize_location_text(entry.get("location"))
            if not current_location or current_location == player_location:
                continue
            names = self._entity_name_candidates_for_sync(slug, entry)
            if not (
                self._narration_implies_entity_with_player(narration_text, names)
                or self._narration_mentions_entity_in_active_scene(
                    narration_text, names
                )
            ):
                continue
            entry["location"] = player_location
            changed += 1
        if changed:
            with self._session_factory() as session:
                row = session.get(Campaign, campaign.id)
                if row:
                    row.characters_json = self._dump_json(characters)
                    session.commit()
        return changed

    def _sync_npc_locations_from_state_to_roster(
        self,
        campaign: Campaign,
        state_update: Dict[str, object],
    ) -> int:
        """Propagate NPC location data from state_update overlay into characters_json.

        When the model puts NPC mutable fields (especially ``location`` and
        ``current_status``) inside ``state_update.<slug>`` instead of
        ``character_updates.<slug>``, the campaign_state overlay gets updated
        but the persistent roster (``characters_json``) stays stale.  This
        method detects those cases and patches the roster so both stores agree.
        """
        if not isinstance(state_update, dict):
            return 0
        characters = self.get_campaign_characters(campaign)
        if not isinstance(characters, dict) or not characters:
            return 0
        overlay_mutable = {"location", "current_status", "allegiance", "evolving_personality"}
        changed = 0
        for slug, overlay in state_update.items():
            if not isinstance(overlay, dict):
                continue
            if slug not in characters:
                continue
            entry = characters[slug]
            if not isinstance(entry, dict):
                continue
            for field in overlay_mutable:
                if field not in overlay:
                    continue
                new_val = overlay[field]
                if new_val is None:
                    continue
                old_val = entry.get(field)
                if old_val != new_val:
                    entry[field] = new_val
                    changed += 1
        if changed:
            with self._session_factory() as session:
                row = session.get(Campaign, campaign.id)
                if row:
                    row.characters_json = self._dump_json(characters)
                    session.commit()
            self._zork_log(
                "NPC ROSTER SYNC FROM STATE_UPDATE",
                f"Patched {changed} field(s) in characters_json from state_update overlay",
            )
        return changed

    def _sync_active_player_character_location(
        self,
        campaign: Campaign,
        *,
        player_state: Dict[str, object],
    ) -> int:
        player_location = self._resolve_player_location_for_state_sync(player_state)
        if not player_location:
            return 0
        character_name = self._normalize_location_text(
            player_state.get("character_name")
        ).lower()
        if not character_name:
            return 0
        characters = self.get_campaign_characters(campaign)
        if not isinstance(characters, dict) or not characters:
            return 0

        target_slug = self._resolve_existing_character_slug(characters, character_name)
        if target_slug is None:
            for slug, entry in characters.items():
                if not isinstance(entry, dict):
                    continue
                entry_name = self._normalize_location_text(entry.get("name")).lower()
                if entry_name and entry_name == character_name:
                    target_slug = slug
                    break
        if target_slug is None:
            return 0

        entry = characters.get(target_slug)
        if not isinstance(entry, dict):
            return 0
        current_location = self._normalize_location_text(entry.get("location"))
        if current_location == player_location:
            return 0
        entry["location"] = player_location
        with self._session_factory() as session:
            row = session.get(Campaign, campaign.id)
            if row:
                row.characters_json = self._dump_json(characters)
                session.commit()
        return 1

    # ── Turn Visibility / Scene Output ──────────────────────────────────

    def _is_emptyish_turn_payload(
        self,
        *,
        narration: str,
        state_update: Dict[str, object],
        player_state_update: Dict[str, object],
        summary_update: object,
        xp_awarded: object,
        scene_image_prompt: object,
        character_updates: Dict[str, object],
        location_updates: Dict[str, object] | None = None,
        calendar_update: object,
    ) -> bool:
        text = " ".join(str(narration or "").strip().lower().split())
        trivial_narration = text in {
            "",
            "the world shifts, but nothing clear emerges.",
            "a hollow silence answers. try again.",
            "a hollow silence answers.",
        }
        short_narration = len(text) < 24
        has_world = bool(state_update) or bool(character_updates) or bool(location_updates) or bool(calendar_update)
        has_player = bool(player_state_update)
        has_summary = bool(str(summary_update or "").strip())
        has_image = bool(str(scene_image_prompt or "").strip())
        try:
            has_xp = int(xp_awarded or 0) > 0
        except (TypeError, ValueError):
            has_xp = False
        has_signal = has_world or has_player or has_summary or has_image or has_xp
        if trivial_narration and not has_signal:
            return True
        if short_narration and not has_signal:
            return True
        return False

    def _default_turn_visibility_meta(
        self,
        campaign: Campaign,
        actor: Optional[Player],
        is_private_context: bool,
    ) -> Dict[str, object]:
        registry = self._campaign_player_registry(
            str(campaign.id), self._session_factory
        )
        actor_entry = (
            registry.get("by_actor_id", {}).get(actor.actor_id)
            if actor is not None
            else None
        )
        actor_slug = str((actor_entry or {}).get("slug") or "").strip()
        actor_actor_id = (actor_entry or {}).get("actor_id")
        actor_state = self.get_player_state(actor) if actor is not None else {}
        actor_location_key = self._room_key_from_player_state(actor_state)
        scope = (
            "local"
            if actor_location_key and actor_location_key.lower() != "unknown-room"
            else "public"
        )
        visible_player_slugs = [actor_slug] if actor_slug else []
        visible_actor_ids = [actor_actor_id] if actor_actor_id is not None else []
        if scope == "public":
            visible_player_slugs = []
            visible_actor_ids = []
        return {
            "scope": scope,
            "actor_player_slug": actor_slug or None,
            "actor_actor_id": actor_actor_id,
            "visible_player_slugs": visible_player_slugs,
            "visible_actor_ids": visible_actor_ids,
            "location_key": (
                actor_location_key
                if scope == "local"
                and actor_location_key
                and actor_location_key.lower() != "unknown-room"
                else None
            ),
            "context_key": None,
            "aware_npc_slugs": [],
            "source": (
                "local-default" if scope == "local" else "public-default"
            ),
        }

    def _normalize_turn_visibility(
        self,
        campaign: Campaign,
        actor: Optional[Player],
        raw_visibility: object,
        *,
        is_private_context: bool,
    ) -> Dict[str, object]:
        default_meta = self._default_turn_visibility_meta(
            campaign, actor, is_private_context
        )
        if not isinstance(raw_visibility, dict):
            return default_meta

        scope = str(raw_visibility.get("scope") or "").strip().lower()
        if scope not in {"public", "private", "limited", "local"}:
            scope = str(default_meta.get("scope") or "public")

        registry = self._campaign_player_registry(
            str(campaign.id), self._session_factory
        )
        by_slug = registry.get("by_slug", {})
        actor_slug = str(default_meta.get("actor_player_slug") or "").strip()
        visible_player_slugs: List[str] = []
        visible_actor_ids: List[str] = []

        raw_player_slugs = raw_visibility.get("player_slugs")
        if isinstance(raw_player_slugs, list):
            player_items = raw_player_slugs
        elif isinstance(raw_player_slugs, str):
            player_items = [raw_player_slugs]
        else:
            player_items = []

        seen_player_slugs: set[str] = set()
        for item in player_items:
            slug = self._player_slug_key(item)
            if not slug or slug in seen_player_slugs:
                continue
            resolved = by_slug.get(slug)
            if resolved is None:
                continue
            seen_player_slugs.add(slug)
            visible_player_slugs.append(slug)
            resolved_actor_id = resolved.get("actor_id")
            if isinstance(resolved_actor_id, str) and resolved_actor_id:
                visible_actor_ids.append(resolved_actor_id)

        if scope in {"private", "limited", "local"} and actor_slug:
            if actor_slug not in seen_player_slugs:
                visible_player_slugs.insert(0, actor_slug)
                seen_player_slugs.add(actor_slug)
            actor_actor_id = default_meta.get("actor_actor_id")
            if isinstance(actor_actor_id, str) and actor_actor_id and actor_actor_id not in visible_actor_ids:
                visible_actor_ids.insert(0, actor_actor_id)

        aware_npc_slugs: List[str] = []
        characters = self.get_campaign_characters(campaign)
        raw_npc_slugs = raw_visibility.get("npc_slugs")
        if isinstance(raw_npc_slugs, list):
            npc_items = raw_npc_slugs
        elif isinstance(raw_npc_slugs, str):
            npc_items = [raw_npc_slugs]
        else:
            npc_items = []
        seen_npc_slugs: set[str] = set()
        for item in npc_items:
            slug = str(item or "").strip()
            if not slug or slug in seen_npc_slugs:
                continue
            if isinstance(characters, dict) and self._resolve_existing_character_slug(
                characters, slug
            ):
                resolved_slug = self._resolve_existing_character_slug(characters, slug)
                if resolved_slug and resolved_slug not in seen_npc_slugs:
                    aware_npc_slugs.append(resolved_slug)
                    seen_npc_slugs.add(resolved_slug)

        reason = self._trim_text(str(raw_visibility.get("reason") or "").strip(), 240)
        location_key = str(default_meta.get("location_key") or "").strip()
        return {
            "scope": scope,
            "actor_player_slug": actor_slug or None,
            "actor_actor_id": default_meta.get("actor_actor_id"),
            "visible_player_slugs": visible_player_slugs,
            "visible_actor_ids": visible_actor_ids,
            "location_key": location_key or None,
            "context_key": str(raw_visibility.get("context_key") or "").strip() or None,
            "aware_npc_slugs": aware_npc_slugs,
            "reason": reason or None,
            "source": "model",
        }

    @staticmethod
    def _resolved_turn_visibility_keys(
        turn_visibility: Dict[str, object],
        *,
        scene_output_raw: object,
        player_state_update: object,
        fallback_location_key: str = "",
    ) -> Dict[str, Optional[str]]:
        location_key = ""
        context_key = ""
        if isinstance(scene_output_raw, dict):
            location_key = str(scene_output_raw.get("location_key") or "").strip()
            context_key = str(scene_output_raw.get("context_key") or "").strip()
        if not location_key and isinstance(player_state_update, dict):
            location_key = str(player_state_update.get("location") or "").strip()
        if not location_key:
            location_key = str(
                turn_visibility.get("location_key") or fallback_location_key or ""
            ).strip()
        if not context_key:
            context_key = str(turn_visibility.get("context_key") or "").strip()
        return {
            "location_key": location_key or None,
            "context_key": context_key or None,
        }

    def _turn_visible_to_npc(
        self,
        turn: Turn,
        npc_slug: str,
    ) -> bool:
        """Check whether *npc_slug* would plausibly know about *turn*.

        An NPC "knows" about a turn if:
        - The turn has public scope, OR
        - The NPC appeared as speaker/actor/listener/aware_npc in any beat
        """
        meta = self._safe_turn_meta(turn)
        visibility = meta.get("visibility")
        if isinstance(visibility, dict):
            scope = str(visibility.get("scope") or "").strip().lower()
            if scope in {"", "public"}:
                return True
        elif not visibility:
            # Legacy turns without visibility metadata -- treat as public
            return True

        slug_norm = self._player_slug_key(npc_slug)
        if not slug_norm:
            return False

        # Check scene_output beats for NPC presence
        scene_output = meta.get("scene_output")
        if isinstance(scene_output, dict):
            beats = scene_output.get("beats")
            if isinstance(beats, list):
                for beat in beats:
                    if not isinstance(beat, dict):
                        continue
                    if self._player_slug_key(beat.get("speaker")) == slug_norm:
                        return True
                    for field in ("actors", "listeners", "aware_npc_slugs"):
                        entries = beat.get(field)
                        if isinstance(entries, list):
                            for entry in entries:
                                if self._player_slug_key(entry) == slug_norm:
                                    return True

        # Check turn-level visibility participant lists
        if isinstance(visibility, dict):
            for field in ("visible_player_slugs",):
                raw = visibility.get(field)
                if isinstance(raw, list):
                    for entry in raw:
                        if self._player_slug_key(entry) == slug_norm:
                            return True

        return False

    def _turn_visible_to_all_scene_npcs(
        self,
        turn: Turn,
        npc_slugs: set[str],
    ) -> bool:
        """Return True only if ALL listed NPCs would know about this turn."""
        for npc_slug in npc_slugs:
            if not self._turn_visible_to_npc(turn, npc_slug):
                return False
        return True

    def _beat_visible_to_all_scene_npcs(
        self,
        beat: Dict[str, object],
        npc_slugs: set[str],
    ) -> bool:
        if not npc_slugs:
            return True
        beat_visibility = str(beat.get("visibility") or "").strip().lower()
        if beat_visibility in {"", "public"}:
            return True
        beat_npc_slugs = {
            self._player_slug_key(entry)
            for entry in (
                list(beat.get("actors") or [])
                + list(beat.get("listeners") or [])
                + list(beat.get("aware_npc_slugs") or [])
                + [beat.get("speaker")]
            )
            if self._player_slug_key(entry)
        }
        return all(
            self._player_slug_key(npc_slug) in beat_npc_slugs
            for npc_slug in npc_slugs
            if self._player_slug_key(npc_slug)
        )

    def _recent_turn_receiver_hints(
        self,
        campaign: Campaign,
        *,
        viewer_actor_id: str,
        party_snapshot: List[Dict[str, object]],
        player_state: Dict[str, object],
    ) -> Dict[str, List[str]]:
        player_slugs: List[str] = []
        seen_player_slugs: set[str] = set()
        for entry in party_snapshot:
            if not isinstance(entry, dict):
                continue
            raw_actor_id = entry.get("actor_id")
            actor_id = str(raw_actor_id or "").strip()
            if actor_id and actor_id == viewer_actor_id:
                continue
            slug = self._player_slug_key(
                entry.get("player_slug") or entry.get("name") or ""
            )
            if slug and slug not in seen_player_slugs:
                seen_player_slugs.add(slug)
                player_slugs.append(slug)
        npc_slugs = sorted(self._active_scene_npc_slugs(campaign, player_state))
        return {
            "player_slugs": player_slugs[:8],
            "npc_slugs": npc_slugs[:12],
        }

    def _turn_relevant_to_scene_receivers(
        self,
        turn: Turn,
        *,
        requested_player_slugs: set[str],
        requested_npc_slugs: set[str],
    ) -> bool:
        meta = self._safe_turn_meta(turn)
        visibility = meta.get("visibility")
        if not isinstance(visibility, dict):
            return False
        scope = str(visibility.get("scope") or "").strip().lower()
        if scope not in {"private", "limited"}:
            return False

        aware_npc_slugs = {
            str(item or "").strip()
            for item in list(visibility.get("aware_npc_slugs") or [])
            if str(item or "").strip()
        }

        visible_player_slugs = {
            self._player_slug_key(item)
            for item in list(visibility.get("visible_player_slugs") or [])
            if self._player_slug_key(item)
        }
        player_match = True
        npc_match = True
        if requested_player_slugs:
            player_match = bool(
                visible_player_slugs.intersection(requested_player_slugs)
            )
        if requested_npc_slugs:
            npc_match = bool(aware_npc_slugs.intersection(requested_npc_slugs))
        return player_match and npc_match

    def _turn_relevant_to_requested_npc_history(
        self,
        turn: Turn,
        *,
        requested_npc_slugs: set[str],
    ) -> bool:
        for raw_slug in requested_npc_slugs:
            npc_slug = self._player_slug_key(raw_slug)
            if npc_slug and self._turn_visible_to_npc(turn, npc_slug):
                return True
        return False

    def _turn_visible_in_recent_turns_context(
        self,
        turn: Turn,
        *,
        viewer_actor_id: str,
        viewer_slug: str,
        viewer_location_key: str,
        viewer_private_context_key: str,
    ) -> bool:
        meta = self._safe_turn_meta(turn)
        if bool(meta.get("suppress_context")):
            return False
        if str(getattr(turn, "actor_id", "") or "").strip() == str(viewer_actor_id or "").strip():
            return True
        visibility = meta.get("visibility")
        if isinstance(visibility, dict):
            scope = str(visibility.get("scope") or "").strip().lower()
            turn_context_key = str(
                visibility.get("context_key") or meta.get("context_key") or ""
            ).strip()
            turn_location_keys = {
                k for k in (
                    self._normalize_location_key(visibility.get("location_key")),
                    self._normalize_location_key(meta.get("location_key")),
                ) if k
            }
            viewer_location_key_norm = self._normalize_location_key(viewer_location_key)
            if scope in {"", "public"}:
                return True
            if scope == "local":
                return bool(
                    viewer_location_key_norm
                    and turn_location_keys
                    and viewer_location_key_norm in turn_location_keys
                ) or self._viewer_participated_in_turn_scene_output(
                    turn,
                    viewer_actor_id=viewer_actor_id,
                    viewer_slug=viewer_slug,
                )
            if scope in {"private", "limited"} and turn_context_key:
                return self._turn_visible_to_viewer(
                    turn,
                    viewer_actor_id,
                    viewer_slug,
                    viewer_location_key,
                )
            return False
        return self._turn_visible_to_viewer(
            turn,
            viewer_actor_id,
            viewer_slug,
            viewer_location_key,
        )

    def _recent_turns_text_for_viewer(
        self,
        campaign: Campaign,
        turns: List[Turn],
        *,
        viewer_actor_id: str,
        viewer_slug: str,
        viewer_location_key: str,
        viewer_private_context_key: str,
        requested_player_slugs: set[str],
        requested_npc_slugs: set[str],
        scene_npc_slugs: Optional[set[str]] = None,
        focus_on_requested_receivers: bool = False,
    ) -> str:
        recent_lines: List[str] = []
        _OOC_RE = re.compile(r"^\s*\[OOC\b", re.IGNORECASE)
        _ERROR_PHRASES = (
            "a hollow silence answers",
            "the world shifts, but nothing clear emerges",
        )
        viewer_location_key_norm = self._normalize_location_key(viewer_location_key)
        registry = self._campaign_player_registry(
            str(campaign.id), self._session_factory
        )
        player_names: Dict[str, str] = {}
        for raw_actor_id, info in registry.get("by_actor_id", {}).items():
            actor_id = str(raw_actor_id or "").strip()
            if not actor_id:
                continue
            name = str(info.get("name") or "").strip()
            if name:
                player_names[actor_id] = name

        for turn in turns:
            content = (turn.content or "").strip()
            if not content:
                continue
            meta = self._safe_turn_meta(turn)
            if focus_on_requested_receivers and (
                requested_player_slugs or requested_npc_slugs
            ):
                visible = False
            else:
                visible = self._turn_visible_in_recent_turns_context(
                    turn,
                    viewer_actor_id=viewer_actor_id,
                    viewer_slug=viewer_slug,
                    viewer_location_key=viewer_location_key,
                    viewer_private_context_key=viewer_private_context_key,
                )
            if (
                not visible
                and (requested_player_slugs or requested_npc_slugs)
                and not scene_npc_slugs
                and turn.actor_id == viewer_actor_id
                and self._turn_relevant_to_scene_receivers(
                    turn,
                    requested_player_slugs=requested_player_slugs,
                    requested_npc_slugs=requested_npc_slugs,
                )
            ):
                visible = True
            if (
                not visible
                and requested_npc_slugs
                and not scene_npc_slugs
                and self._turn_relevant_to_requested_npc_history(
                    turn,
                    requested_npc_slugs=requested_npc_slugs,
                )
            ):
                visible = True
            if not visible:
                continue
            # LCD filtering: skip turns that scene NPCs don't know about
            if scene_npc_slugs and not self._turn_visible_to_all_scene_npcs(
                turn, scene_npc_slugs
            ):
                continue

            scene_output_lines = self._scene_output_recent_lines(
                turn,
                self.get_campaign_state(campaign),
                meta.get("scene_output"),
                viewer_actor_id=viewer_actor_id,
                viewer_slug=viewer_slug,
                viewer_location_key=viewer_location_key,
                viewer_private_context_key=viewer_private_context_key,
                requested_npc_slugs=requested_npc_slugs,
                scene_npc_slugs=scene_npc_slugs,
            )
            if scene_output_lines and turn.kind == "narrator":
                recent_lines.extend(scene_output_lines)
                continue

            campaign_state = self.get_campaign_state(campaign)
            if turn.kind == "player":
                if _OOC_RE.match(content):
                    continue
                clipped = self._strip_inventory_mentions(content)
                if not clipped:
                    continue
                name = player_names.get(str(turn.actor_id or ""))
                recent_lines.extend(
                    self._recent_turn_fallback_lines(
                        turn,
                        campaign_state,
                        content_text=clipped,
                        player_name=name,
                    )
                )
            elif turn.kind == "narrator":
                if content.lower() in _ERROR_PHRASES:
                    continue
                clipped = self._strip_narration_footer(content)
                if not clipped:
                    continue
                recent_lines.extend(
                    self._recent_turn_fallback_lines(
                        turn,
                        campaign_state,
                        content_text=clipped,
                    )
                )
        return "\n".join(recent_lines) if recent_lines else "None"

    def _recent_turns_location_hint(
        self,
        turns: List[Turn],
        *,
        viewer_actor_id: str,
        viewer_slug: str,
        viewer_location_key: str,
        viewer_private_context_key: str,
    ) -> Dict[str, str]:
        current_location = str(viewer_location_key or "").strip().lower() or "unknown-room"
        current_location_norm = self._normalize_location_key(current_location)
        last_other_location = ""
        for turn in reversed(turns):
            if not self._turn_visible_to_viewer(
                turn,
                viewer_actor_id,
                viewer_slug,
                viewer_location_key,
            ):
                continue
            meta = self._safe_turn_meta(turn)
            visibility = meta.get("visibility")
            turn_location = ""
            if isinstance(visibility, dict):
                turn_location = str(visibility.get("location_key") or "").strip().lower()
            if not turn_location:
                turn_location = str(meta.get("location_key") or "").strip().lower()
            if not turn_location:
                continue
            if self._normalize_location_key(turn_location) != current_location_norm:
                last_other_location = turn_location
                break
        return {
            "current_location_key": current_location,
            "last_other_location_key": last_other_location or "none",
        }

    def _recent_turn_fallback_lines(
        self,
        turn: Turn,
        campaign_state: Dict[str, object],
        *,
        content_text: str,
        player_name: str = "",
    ) -> List[str]:
        text = str(content_text or "").strip()
        if not text:
            return []
        turn_number = int(getattr(turn, "id", 0) or 0)
        index = (
            campaign_state.get(self.TURN_TIME_INDEX_KEY)
            if isinstance(campaign_state, dict)
            else {}
        )
        if not isinstance(index, dict):
            index = {}
        entry = index.get(str(turn_number))
        meta = self._safe_turn_meta(turn)
        visibility = meta.get("visibility")
        scope = "public"
        visible_actor_ids: List[str] = []
        aware_npc_slugs: List[str] = []
        actor_slug = ""
        location_key = str(meta.get("location_key") or "").strip() or None
        context_key = str(meta.get("context_key") or "").strip() or None
        if isinstance(visibility, dict):
            scope = str(visibility.get("scope") or "").strip().lower() or "public"
            actor_slug = self._player_slug_key(
                visibility.get("actor_player_slug") or ""
            )
            location_key = (
                str(visibility.get("location_key") or location_key or "").strip()
                or None
            )
            context_key = (
                str(visibility.get("context_key") or context_key or "").strip()
                or None
            )
            for item in list(visibility.get("visible_actor_ids") or []):
                actor_id = str(item or "").strip()
                if actor_id and actor_id not in visible_actor_ids:
                    visible_actor_ids.append(actor_id)
            aware_npc_slugs = [
                str(item or "").strip()
                for item in list(visibility.get("aware_npc_slugs") or [])
                if str(item or "").strip()
            ]
        if not actor_slug and turn.kind == "player":
            actor_slug = self._player_slug_key(player_name) or f"player-{turn.actor_id}"
        time_source = entry if isinstance(entry, dict) else meta.get("game_time")
        beat_type = "player_action" if turn.kind == "player" else "narration"
        speaker = actor_slug or ("narrator" if turn.kind == "narrator" else "player")
        actors = [actor_slug] if actor_slug else []
        beat = {
            "kind": "beat",
            "turn_id": turn_number,
            "index": 0,
            "reasoning": (
                "Compatibility fallback from player turn text."
                if turn.kind == "player"
                else "Compatibility fallback from plain narration."
            ),
            "type": beat_type,
            "speaker": speaker,
            "actors": actors,
            "listeners": [],
            "visibility": scope,
            "visible_actor_ids": visible_actor_ids,
            "aware_npc_slugs": aware_npc_slugs,
            "location_key": location_key,
            "context_key": context_key,
            "text": text,
        }
        if isinstance(time_source, dict):
            beat["day"] = (
                self._coerce_non_negative_int(time_source.get("day", 1), default=1) or 1
            )
            beat["hour"] = min(
                23,
                max(0, self._coerce_non_negative_int(time_source.get("hour", 0), default=0)),
            )
            beat["minute"] = min(
                59,
                max(
                    0,
                    self._coerce_non_negative_int(time_source.get("minute", 0), default=0),
                ),
            )
        return [json.dumps(beat, ensure_ascii=False, separators=(",", ":"))]

    def _scene_output_text_from_raw(self, raw_scene_output: object) -> str:
        if not isinstance(raw_scene_output, dict):
            return ""
        raw_beats = raw_scene_output.get("beats")
        if not isinstance(raw_beats, list):
            return ""
        texts: List[str] = []
        for beat in raw_beats:
            if not isinstance(beat, dict):
                continue
            text = str(
                beat.get("text") or beat.get("summary") or ""
            ).strip()
            if not text:
                continue
            texts.append(text)
        if not texts:
            return ""
        return self._trim_text("\n\n".join(texts), self.MAX_NARRATION_CHARS)

    def _normalize_scene_output(
        self,
        campaign: Campaign,
        raw_scene_output: object,
        *,
        fallback_narration: str,
        turn_visibility: Dict[str, object],
        fallback_location_key: str,
        actor_actor_id: Optional[str],
        actor_player_slug: str,
    ) -> Optional[Dict[str, object]]:
        base_visibility = (
            str(turn_visibility.get("scope") or "").strip().lower() or "local"
        )
        base_visible_actor_ids: List[str] = []
        for item in list(turn_visibility.get("visible_actor_ids") or []):
            actor_id = str(item or "").strip()
            if actor_id:
                base_visible_actor_ids.append(actor_id)
        base_aware_npc_slugs = [
            str(item or "").strip()
            for item in list(turn_visibility.get("aware_npc_slugs") or [])
            if str(item or "").strip()
        ]
        scene_output = raw_scene_output if isinstance(raw_scene_output, dict) else {}
        raw_beats = scene_output.get("beats")
        if not isinstance(raw_beats, list):
            raw_beats = []

        characters = self.get_campaign_characters(campaign)
        beats: List[Dict[str, object]] = []
        for raw_beat in raw_beats[:24]:
            if not isinstance(raw_beat, dict):
                continue
            text = str(
                raw_beat.get("text") or raw_beat.get("summary") or ""
            ).strip()
            if not text:
                continue
            beat_visibility = (
                str(raw_beat.get("visibility") or "").strip().lower() or base_visibility
            )
            if beat_visibility not in {"public", "private", "limited", "local"}:
                beat_visibility = base_visibility
            reasoning = self._trim_text(
                str(raw_beat.get("reasoning") or "").strip(),
                180,
            )
            if not reasoning:
                if beat_visibility == "private":
                    reasoning = "Private beat for the acting character."
                elif beat_visibility == "limited":
                    reasoning = "Limited beat for explicit participants."
                elif beat_visibility == "local":
                    reasoning = "Visible to the current room."
                else:
                    reasoning = "Public beat visible to everyone."
            beat_type = str(raw_beat.get("type") or "narration").strip().lower() or "narration"
            speaker = str(raw_beat.get("speaker") or "narrator").strip() or "narrator"
            actors: List[str] = []
            raw_actors = raw_beat.get("actors")
            if isinstance(raw_actors, list):
                for item in raw_actors:
                    actor_text = str(item or "").strip()
                    if actor_text and actor_text not in actors:
                        actors.append(actor_text)
            elif isinstance(raw_actors, str):
                actor_text = str(raw_actors or "").strip()
                if actor_text:
                    actors.append(actor_text)
            if not actors and speaker and speaker != "narrator":
                actors.append(speaker)

            listeners: List[str] = []
            raw_listeners = raw_beat.get("listeners")
            if isinstance(raw_listeners, list):
                for item in raw_listeners:
                    listener_text = str(item or "").strip()
                    if listener_text and listener_text not in listeners:
                        listeners.append(listener_text)
            elif isinstance(raw_listeners, str):
                listener_text = str(raw_listeners or "").strip()
                if listener_text:
                    listeners.append(listener_text)

            visible_actor_ids: List[str] = []
            raw_aware_ids = raw_beat.get("visible_actor_ids")
            if not isinstance(raw_aware_ids, list):
                raw_aware_ids = raw_beat.get("aware_actor_ids")
            if isinstance(raw_aware_ids, list):
                for item in raw_aware_ids:
                    aware_id = str(item or "").strip()
                    if aware_id and aware_id not in visible_actor_ids:
                        visible_actor_ids.append(aware_id)
            if not visible_actor_ids and beat_visibility in {"private", "limited"}:
                for item in base_visible_actor_ids:
                    if item not in visible_actor_ids:
                        visible_actor_ids.append(item)
            if not visible_actor_ids and beat_visibility == "private" and actor_actor_id is not None:
                visible_actor_ids.append(str(actor_actor_id))

            aware_npc_slugs: List[str] = []
            raw_aware_npcs = raw_beat.get("aware_npc_slugs")
            if isinstance(raw_aware_npcs, list):
                for item in raw_aware_npcs:
                    candidate = str(item or "").strip()
                    if not candidate:
                        continue
                    resolved = (
                        self._resolve_existing_character_slug(characters, candidate)
                        if isinstance(characters, dict)
                        else None
                    )
                    final_slug = resolved or candidate
                    if final_slug not in aware_npc_slugs:
                        aware_npc_slugs.append(final_slug)
            if not aware_npc_slugs and beat_visibility in {"private", "limited"}:
                for slug in base_aware_npc_slugs:
                    if slug not in aware_npc_slugs:
                        aware_npc_slugs.append(slug)

            beat_location_key = (
                str(raw_beat.get("location_key") or "").strip()
                or str(scene_output.get("location_key") or "").strip()
                or str(turn_visibility.get("location_key") or "").strip()
                or str(fallback_location_key or "").strip()
            )
            aware_npc_slugs = self._filter_aware_npc_slugs_for_location_key(
                campaign,
                beat_location_key,
                aware_npc_slugs,
            )
            beat_context_key = (
                str(raw_beat.get("context_key") or "").strip()
                or str(scene_output.get("context_key") or "").strip()
                or str(turn_visibility.get("context_key") or "").strip()
            )
            beats.append(
                {
                    "reasoning": reasoning,
                    "type": beat_type,
                    "speaker": speaker,
                    "actors": actors,
                    "listeners": listeners,
                    "visibility": beat_visibility,
                    "visible_actor_ids": visible_actor_ids,
                    "aware_npc_slugs": aware_npc_slugs,
                    "text": self._trim_text(text, self.MAX_NARRATION_CHARS),
                    "location_key": beat_location_key or None,
                    "context_key": beat_context_key or None,
                }
            )

        if not beats:
            fallback_text = str(fallback_narration or "").strip()
            if not fallback_text:
                return None
            beats = [
                {
                    "reasoning": "Compatibility fallback from plain narration.",
                    "type": "narration",
                    "speaker": "narrator",
                    "actors": [],
                    "listeners": [],
                    "visibility": base_visibility,
                    "visible_actor_ids": base_visible_actor_ids
                    if base_visibility in {"private", "limited"}
                    else ([] if actor_actor_id is None or base_visibility != "private" else [str(actor_actor_id)]),
                    "aware_npc_slugs": base_aware_npc_slugs if base_visibility in {"private", "limited"} else [],
                    "text": self._trim_text(fallback_text, self.MAX_NARRATION_CHARS),
                    "location_key": str(turn_visibility.get("location_key") or fallback_location_key or "").strip() or None,
                    "context_key": str(turn_visibility.get("context_key") or "").strip() or None,
                }
            ]

        rendered_text = self._trim_text(
            "\n\n".join(str(beat.get("text") or "").strip() for beat in beats if str(beat.get("text") or "").strip()),
            self.MAX_NARRATION_CHARS,
        )

        normalized = {
            "location_key": (
                str(scene_output.get("location_key") or "").strip()
                or str(turn_visibility.get("location_key") or "").strip()
                or str(fallback_location_key or "").strip()
                or None
            ),
            "context_key": (
                str(scene_output.get("context_key") or "").strip()
                or str(turn_visibility.get("context_key") or "").strip()
                or None
            ),
            "actor_player_slug": actor_player_slug or None,
            "beats": beats,
            "rendered_text": rendered_text or None,
        }
        return normalized

    def _scene_output_rendered_text(
        self, scene_output: Optional[Dict[str, object]]
    ) -> str:
        if not isinstance(scene_output, dict):
            return ""
        beats = scene_output.get("beats")
        if not isinstance(beats, list):
            return ""
        texts = [
            str(beat.get("text") or "").strip()
            for beat in beats
            if isinstance(beat, dict) and str(beat.get("text") or "").strip()
        ]
        if not texts:
            return ""
        return self._trim_text("\n\n".join(texts), self.MAX_NARRATION_CHARS)

    def _scene_output_jsonl(
        self,
        *,
        turn_id: Optional[int],
        game_time: Dict[str, int],
        scene_output: Optional[Dict[str, object]],
        turn_visibility: Optional[Dict[str, object]],
    ) -> str:
        if not isinstance(scene_output, dict):
            return ""
        beats = scene_output.get("beats")
        if not isinstance(beats, list) or not beats:
            return ""
        lines: List[str] = []
        header = {
            "kind": "turn",
            "turn_id": turn_id,
            "day": self._coerce_non_negative_int(game_time.get("day", 1), default=1) if isinstance(game_time, dict) else 1,
            "hour": self._coerce_non_negative_int(game_time.get("hour", 0), default=0) if isinstance(game_time, dict) else 0,
            "minute": self._coerce_non_negative_int(game_time.get("minute", 0), default=0) if isinstance(game_time, dict) else 0,
            "location_key": scene_output.get("location_key"),
            "context_key": scene_output.get("context_key"),
            "visibility": str((turn_visibility or {}).get("scope") or "").strip().lower() or "local",
        }
        lines.append(json.dumps(header, ensure_ascii=False, separators=(",", ":")))
        for beat_index, beat in enumerate(beats):
            if not isinstance(beat, dict):
                continue
            line = {
                "kind": "beat",
                "turn_id": turn_id,
                "index": beat_index,
                "reasoning": str(beat.get("reasoning") or "").strip(),
                "type": str(beat.get("type") or "narration").strip(),
                "speaker": str(beat.get("speaker") or "narrator").strip(),
                "actors": list(beat.get("actors") or []),
                "listeners": list(beat.get("listeners") or []),
                "visibility": str(beat.get("visibility") or "local").strip(),
                "visible_actor_ids": list(beat.get("visible_actor_ids") or []),
                "aware_npc_slugs": list(beat.get("aware_npc_slugs") or []),
                "location_key": beat.get("location_key"),
                "context_key": beat.get("context_key"),
                "text": str(beat.get("text") or "").strip(),
            }
            lines.append(json.dumps(line, ensure_ascii=False, separators=(",", ":")))
        return "\n".join(lines)

    def _scene_output_recent_lines(
        self,
        turn: Turn,
        campaign_state: Dict[str, object],
        scene_output: object,
        *,
        viewer_actor_id: Optional[str] = None,
        viewer_slug: str = "",
        viewer_location_key: str = "",
        viewer_private_context_key: str = "",
        requested_npc_slugs: Optional[set[str]] = None,
        scene_npc_slugs: Optional[set[str]] = None,
    ) -> List[str]:
        if not isinstance(scene_output, dict):
            return []
        beats = scene_output.get("beats")
        if not isinstance(beats, list) or not beats:
            return []
        turn_number = int(getattr(turn, "id", 0) or 0)
        time_index = campaign_state.get(self.TURN_TIME_INDEX_KEY) if isinstance(campaign_state, dict) else {}
        if not isinstance(time_index, dict):
            time_index = {}
        entry = time_index.get(str(turn_number))
        meta = self._safe_turn_meta(turn)
        time_source = entry if isinstance(entry, dict) else meta.get("game_time")
        beat_lines: List[str] = []
        meta_location_key_norm = self._normalize_location_key(meta.get("location_key"))
        fallback_location_key = self._normalize_location_key(
            scene_output.get("location_key")
        )
        fallback_context_key = str(scene_output.get("context_key") or "").strip()
        viewer_slug_key = self._player_slug_key(viewer_slug)
        viewer_location_key_norm = self._normalize_location_key(viewer_location_key)
        viewer_private_context_key_norm = str(viewer_private_context_key or "").strip()
        is_actor_turn = str(getattr(turn, "actor_id", "") or "").strip() == str(viewer_actor_id or "").strip()
        for beat_index, beat in enumerate(beats):
            if not isinstance(beat, dict):
                continue
            beat_visibility = str(beat.get("visibility") or "local").strip().lower() or "local"
            beat_location_keys = {
                k for k in (
                    self._normalize_location_key(beat.get("location_key")),
                    fallback_location_key,
                    meta_location_key_norm,
                ) if k
            }
            beat_context_key = (
                str(beat.get("context_key") or "").strip()
                or fallback_context_key
            )
            beat_visible_actor_ids: List[str] = []
            for item in list(beat.get("visible_actor_ids") or []):
                aware_id = str(item or "").strip()
                if aware_id and aware_id not in beat_visible_actor_ids:
                    beat_visible_actor_ids.append(aware_id)
            beat_actor_listener_slugs = {
                self._player_slug_key(item)
                for item in list(beat.get("actors") or []) + list(beat.get("listeners") or [])
                if self._player_slug_key(item)
            }
            beat_visible = False
            if beat_visibility in {"", "public"}:
                beat_visible = True
            elif beat_visibility == "local":
                beat_visible = bool(
                    is_actor_turn
                    or (
                        viewer_actor_id is not None
                        and str(viewer_actor_id).strip() in beat_visible_actor_ids
                    )
                    or (viewer_slug_key and viewer_slug_key in beat_actor_listener_slugs)
                    or (
                        viewer_location_key_norm
                        and beat_location_keys
                        and viewer_location_key_norm in beat_location_keys
                    )
                )
            elif beat_visibility in {"private", "limited"}:
                beat_visible = (
                    (viewer_actor_id is not None and str(viewer_actor_id) in beat_visible_actor_ids)
                    or (viewer_slug_key and viewer_slug_key in beat_actor_listener_slugs)
                    or (
                        viewer_private_context_key_norm
                        and beat_context_key
                        and viewer_private_context_key_norm == beat_context_key
                    )
                )
            if not beat_visible and requested_npc_slugs and not scene_npc_slugs:
                beat_npc_slugs = {
                    self._player_slug_key(e)
                    for e in (
                        list(beat.get("actors") or [])
                        + list(beat.get("listeners") or [])
                        + list(beat.get("aware_npc_slugs") or [])
                        + [beat.get("speaker")]
                    )
                    if self._player_slug_key(e)
                }
                requested_npc_keys = {
                    self._player_slug_key(npc)
                    for npc in requested_npc_slugs
                    if self._player_slug_key(npc)
                }
                if beat_npc_slugs.intersection(requested_npc_keys):
                    beat_visible = True
            if not beat_visible:
                continue
            if scene_npc_slugs and not self._beat_visible_to_all_scene_npcs(
                beat,
                scene_npc_slugs,
            ):
                continue
            beat_row = {
                "kind": "beat",
                "turn_id": turn_number,
                "index": beat_index,
                "reasoning": str(beat.get("reasoning") or "").strip(),
                "type": str(beat.get("type") or "narration").strip(),
                "speaker": str(beat.get("speaker") or "narrator").strip(),
                "actors": list(beat.get("actors") or []),
                "listeners": list(beat.get("listeners") or []),
                "visibility": str(beat.get("visibility") or "local").strip(),
                "visible_actor_ids": beat_visible_actor_ids,
                "aware_npc_slugs": list(beat.get("aware_npc_slugs") or []),
                "location_key": beat.get("location_key"),
                "context_key": beat.get("context_key"),
                "text": str(beat.get("text") or "").strip(),
            }
            if isinstance(time_source, dict):
                beat_row["day"] = self._coerce_non_negative_int(time_source.get("day", 1), default=1) or 1
                beat_row["hour"] = min(23, max(0, self._coerce_non_negative_int(time_source.get("hour", 0), default=0)))
                beat_row["minute"] = min(59, max(0, self._coerce_non_negative_int(time_source.get("minute", 0), default=0)))
            beat_lines.append(
                json.dumps(
                    beat_row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        if not beat_lines:
            return []
        return beat_lines

    def _compute_recent_turns_metadata(
        self,
        recent_turns: List[Turn],
        *,
        viewer_actor_id: str,
        viewer_slug: str,
    ) -> Dict[str, object]:
        """Compute quantitative metadata from loaded recent_turns for dynamic RECENT_TURNS_NOTE."""
        turn_count = len(recent_turns)
        time_span_minutes = 0
        if turn_count >= 2:
            first_created = getattr(recent_turns[-1], "created_at", None)
            last_created = getattr(recent_turns[0], "created_at", None)
            if first_created and last_created:
                try:
                    delta = (last_created - first_created).total_seconds()
                    time_span_minutes = max(0, int(delta / 60))
                except Exception:
                    pass
        speaker_counts: Dict[str, int] = {}
        listener_counts: Dict[str, int] = {}
        private_turn_count = 0
        viewer_last_turn_ago = turn_count  # default: never acted
        viewer_slug_lower = (viewer_slug or "").strip().lower()
        for idx, turn in enumerate(recent_turns):
            meta = self._safe_turn_meta(turn)
            visibility = meta.get("visibility")
            if isinstance(visibility, dict):
                scope = str(visibility.get("scope") or "").strip().lower()
                if scope in {"private", "limited"}:
                    private_turn_count += 1
            # Check if viewer acted this turn
            actor_slug = ""
            if isinstance(visibility, dict):
                actor_slug = str(visibility.get("actor_player_slug") or "").strip().lower()
            if not actor_slug:
                actor_slug = self._player_slug_key(meta.get("actor_player_slug") or "")
            if viewer_slug_lower and actor_slug == viewer_slug_lower and idx < viewer_last_turn_ago:
                viewer_last_turn_ago = idx
            # Extract speakers/listeners from beats
            scene_output = meta.get("scene_output")
            if isinstance(scene_output, dict):
                beats = scene_output.get("beats")
                if isinstance(beats, list):
                    for beat in beats:
                        if not isinstance(beat, dict):
                            continue
                        spk = str(beat.get("speaker") or "").strip()
                        if spk and spk != "narrator":
                            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
                        for lis in (beat.get("listeners") or []):
                            lis_str = str(lis or "").strip()
                            if lis_str:
                                listener_counts[lis_str] = listener_counts.get(lis_str, 0) + 1
        active_speakers = [
            name for name, _ in sorted(speaker_counts.items(), key=lambda x: -x[1])
        ][:6]
        active_listeners = sorted(listener_counts.items(), key=lambda x: -x[1])[:6]
        return {
            "turn_count": turn_count,
            "time_span_minutes": time_span_minutes,
            "active_speakers": active_speakers,
            "active_listeners": active_listeners,
            "private_turn_count": private_turn_count,
            "viewer_last_turn_ago": viewer_last_turn_ago if viewer_last_turn_ago < turn_count else None,
        }

    def _turn_embedding_metadata(
        self,
        *,
        visibility: Optional[Dict[str, object]],
        actor_player_slug: object,
        location_key: object,
        session_id: object,
    ) -> Dict[str, object]:
        visibility = visibility if isinstance(visibility, dict) else {}
        return {
            "actor_player_slug": self._player_slug_key(actor_player_slug),
            "visibility_scope": str(visibility.get("scope") or "public").strip().lower(),
            "visible_player_slugs": list(visibility.get("visible_player_slugs") or []),
            "visible_actor_ids": list(visibility.get("visible_actor_ids") or []),
            "aware_npc_slugs": list(visibility.get("aware_npc_slugs") or []),
            "location_key": str(location_key or "").strip(),
            "session_id": session_id,
        }

    def _memory_tool_text_value(self, text: object, max_chars: int = 4000) -> str:
        value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(value) > max_chars:
            value = value[:max_chars].rsplit(" ", 1)[0].strip() + "..."
        return value

    def _memory_tool_jsonl(self, records: List[Dict[str, object]]) -> str:
        lines: List[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            lines.append(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
        return "\n".join(lines) if lines else "None"

    # ── State Update / Merge ──────────────────────────────────────────────

    def _merge_state_update_with_conflict_resolution(
        self,
        authoritative_state: Dict[str, object],
        delta_update: Dict[str, object],
    ) -> Dict[str, object]:
        """Deep-merge *delta_update* into *authoritative_state* with per-key
        conflict resolution suitable for concurrent Phase 3 commits.

        Rules:
        * ``game_time`` -- monotonic max (keeps the later time).
        * ``current_chapter`` / ``current_scene`` -- left for the normal
          advancement guards that run after this merge.
        * Nested dicts -- shallow merge (``{**current[key], **delta[key]}``).
        * Scalar keys -- last-writer-wins (delta overwrites).
        * ``None`` values -- delete the key (existing ``_apply_state_update``
          behaviour).
        """
        if not isinstance(delta_update, dict):
            return authoritative_state

        # Handle game_time monotonic-max separately.
        delta_time = delta_update.get("game_time")
        if isinstance(delta_time, dict) and delta_time:
            current_time = authoritative_state.get("game_time", {})
            if isinstance(current_time, dict):
                delta_minutes = self._game_time_to_total_minutes(delta_time)
                current_minutes = self._game_time_to_total_minutes(current_time)
                if delta_minutes > current_minutes:
                    authoritative_state["game_time"] = delta_time
                # else: keep current -- it's already further ahead.
            else:
                authoritative_state["game_time"] = delta_time
            delta_update = {k: v for k, v in delta_update.items() if k != "game_time"}

        # Merge remaining keys.
        pruned_keys: List[str] = []
        for key, value in delta_update.items():
            if value is None:
                authoritative_state.pop(key, None)
                pruned_keys.append(key)
            elif (
                isinstance(value, str)
                and value.strip().lower() in self._COMPLETED_VALUES
            ):
                authoritative_state.pop(key, None)
                pruned_keys.append(key)
            elif isinstance(value, dict):
                existing = authoritative_state.get(key)
                if isinstance(existing, dict):
                    # Shallow merge: delta keys overwrite, existing keys retained.
                    merged = {**existing, **value}
                    # Honour None deletes within nested dict.
                    merged = {k: v for k, v in merged.items() if v is not None}
                    authoritative_state[key] = merged
                else:
                    authoritative_state[key] = value
            else:
                authoritative_state[key] = value

        if pruned_keys:
            self._auto_resolve_stale_plot_threads(authoritative_state, pruned_keys)
        return authoritative_state

    def _merge_character_updates(
        self,
        existing_chars: Dict[str, object],
        delta_updates: Dict[str, object],
    ) -> Dict[str, object]:
        """Per-slug merge for character_updates with conflict resolution.

        * Same NPC updated by two turns: merge mutable fields (last-writer-wins
          per field).
        * Deletion from either turn: deletion wins.
        * New NPC from either turn: add.
        """
        if not isinstance(delta_updates, dict):
            return existing_chars
        for slug, delta_char in delta_updates.items():
            if delta_char is None:
                # Deletion wins.
                existing_chars.pop(slug, None)
                continue
            existing = existing_chars.get(slug)
            if existing is None or not isinstance(existing, dict):
                existing_chars[slug] = delta_char
            elif isinstance(delta_char, dict):
                # Per-field merge.
                for field_key, field_val in delta_char.items():
                    if field_val is None:
                        existing.pop(field_key, None)
                    else:
                        existing[field_key] = field_val
        return existing_chars

    def _ensure_minimum_state_update_contract(
        self,
        campaign_state: Dict[str, object],
        state_update: object,
    ) -> Dict[str, object]:
        out = dict(state_update) if isinstance(state_update, dict) else {}
        current_time = self._extract_game_time_snapshot(campaign_state)
        start_day_of_week = self._campaign_start_day_of_week(campaign_state)
        provided_time = out.get("game_time")
        if isinstance(provided_time, dict):
            out["game_time"] = self._game_time_from_total_minutes(
                self._game_time_to_total_minutes(provided_time),
                start_day_of_week=start_day_of_week,
            )
        else:
            out["game_time"] = self._game_time_from_total_minutes(
                self._game_time_to_total_minutes(current_time),
                start_day_of_week=start_day_of_week,
            )
        if bool(campaign_state.get("on_rails", False)):
            out["current_chapter"] = self._coerce_non_negative_int(
                out.get("current_chapter", campaign_state.get("current_chapter", 0)),
                default=self._coerce_non_negative_int(
                    campaign_state.get("current_chapter", 0), default=0
                ),
            )
            out["current_scene"] = self._coerce_non_negative_int(
                out.get("current_scene", campaign_state.get("current_scene", 0)),
                default=self._coerce_non_negative_int(
                    campaign_state.get("current_scene", 0), default=0
                ),
            )
            return out

        active_chapters = self._chapters_for_prompt(
            campaign_state,
            active_only=True,
            limit=1,
        )
        active_row = active_chapters[0] if active_chapters else {}
        default_chapter = self._chapter_slug_key(
            active_row.get("slug") or campaign_state.get("current_chapter")
        )
        default_scene = self._chapter_slug_key(
            active_row.get("current_scene") or campaign_state.get("current_scene")
        )

        chapter_slug = self._chapter_slug_key(out.get("current_chapter"))
        scene_slug = self._chapter_slug_key(out.get("current_scene"))

        # Guard: reject chapter/scene slugs that don't match the chapter plan.
        # This prevents the model from hallucinating old or nonexistent values.
        all_chapters = self._chapter_plan_from_state(campaign_state)
        if chapter_slug and all_chapters and chapter_slug not in all_chapters:
            self._zork_log(
                "CHAPTER REGRESSION BLOCKED",
                f"Model sent current_chapter={chapter_slug!r} which is not in "
                f"chapter plan {sorted(all_chapters.keys())!r}; falling back to {default_chapter!r}",
            )
            chapter_slug = ""
        resolved_chapter = chapter_slug or default_chapter or ""
        if scene_slug and resolved_chapter and resolved_chapter in all_chapters:
            valid_scenes = all_chapters[resolved_chapter].get("scenes") or []
            if valid_scenes and scene_slug not in valid_scenes:
                self._zork_log(
                    "SCENE REGRESSION BLOCKED",
                    f"Model sent current_scene={scene_slug!r} which is not in "
                    f"chapter {resolved_chapter!r} scenes {valid_scenes!r}; falling back to {default_scene!r}",
                )
                scene_slug = ""
        out["current_chapter"] = resolved_chapter
        out["current_scene"] = scene_slug or default_scene or ""
        return out

    def _guard_state_null_character_prunes(
        self,
        state_update: object,
        existing_chars: Dict[str, dict],
        *,
        resolution_context: str = "",
        campaign_state: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if not isinstance(state_update, dict):
            return {}
        if not isinstance(existing_chars, dict):
            return dict(state_update)
        candidate_deletes: Dict[str, object] = {}
        for raw_key, value in state_update.items():
            if value is not None:
                continue
            resolved = self._resolve_existing_character_slug(existing_chars, raw_key)
            if resolved:
                candidate_deletes[str(raw_key)] = None
        if not candidate_deletes:
            return dict(state_update)
        allowed_deletes = self._sanitize_character_removals(
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

    def _sanitize_character_removals(
        self,
        existing_chars: Dict[str, dict],
        updates: object,
        *,
        resolution_context: str = "",
        campaign_state: Optional[Dict[str, object]] = None,
        counter_key: str = "character_remove_blocked",
    ) -> Dict[str, object]:
        if not isinstance(updates, dict):
            return {}
        # Character deletion is now fully model-controlled (reasoning + structured updates).
        # Keep this hook for compatibility, but do not block removals.
        return dict(updates)

    def _normalize_story_progression(self, value: object) -> Optional[Dict[str, object]]:
        if not isinstance(value, dict):
            return None
        target = " ".join(str(value.get("target") or "").strip().lower().split())
        target = target.replace("_", "-")
        allowed_targets = {"hold", "next-scene", "next-chapter"}
        if target not in allowed_targets:
            target = "hold"
        advance_raw = value.get("advance")
        if isinstance(advance_raw, bool):
            advance = advance_raw
        else:
            advance_text = " ".join(str(advance_raw or "").strip().lower().split())
            advance = advance_text in {"1", "true", "yes", "y", "advance"}
        if target == "hold":
            advance = False
        reason = " ".join(str(value.get("reason") or "").strip().split())[:300]
        return {
            "advance": advance,
            "target": target,
            "reason": reason,
        }

    def _apply_story_progression_hint(
        self,
        campaign_state: Dict[str, object],
        story_progression: Optional[Dict[str, object]],
        state_update: Dict[str, object],
    ) -> bool:
        if not bool(campaign_state.get("on_rails", False)):
            return False
        if not isinstance(state_update, dict):
            return False
        if "current_chapter" in state_update or "current_scene" in state_update:
            return False
        if not isinstance(story_progression, dict):
            return False
        if not bool(story_progression.get("advance")):
            return False

        outline = campaign_state.get("story_outline")
        if not isinstance(outline, dict):
            return False
        chapters = outline.get("chapters")
        if not isinstance(chapters, list) or not chapters:
            return False

        old_ch = self._coerce_non_negative_int(
            campaign_state.get("current_chapter", 0), default=0
        )
        old_sc = self._coerce_non_negative_int(
            campaign_state.get("current_scene", 0), default=0
        )
        old_ch = min(old_ch, len(chapters) - 1)
        current_entry = chapters[old_ch] if 0 <= old_ch < len(chapters) else {}
        scenes = current_entry.get("scenes")
        if not isinstance(scenes, list):
            scenes = []

        target = str(story_progression.get("target") or "hold")
        new_ch = old_ch
        new_sc = old_sc
        if target == "next-chapter":
            if old_ch + 1 >= len(chapters):
                return False
            new_ch = old_ch + 1
            new_sc = 0
            if isinstance(current_entry, dict):
                current_entry["completed"] = True
        elif target == "next-scene":
            if scenes and old_sc + 1 < len(scenes):
                new_sc = old_sc + 1
            elif old_ch + 1 < len(chapters):
                new_ch = old_ch + 1
                new_sc = 0
                if isinstance(current_entry, dict):
                    current_entry["completed"] = True
            else:
                return False
        else:
            return False

        campaign_state["current_chapter"] = new_ch
        campaign_state["current_scene"] = new_sc
        return True

    # ── Prompt Construction ───────────────────────────────────────────────

    def _recompose_prompt_with_tail(
        self,
        prompt: str,
        turn_prompt_tail: str,
        *inserted_blocks: object,
    ) -> str:
        prompt_text = str(prompt or "")
        tail_text = str(turn_prompt_tail or "").strip()
        base = prompt_text
        if tail_text:
            with_newline = f"\n{tail_text}\n"
            without_newline = f"\n{tail_text}"
            if prompt_text.endswith(with_newline):
                base = prompt_text[: -len(with_newline)]
            elif prompt_text.endswith(without_newline):
                base = prompt_text[: -len(without_newline)]
            elif prompt_text.endswith(tail_text):
                base = prompt_text[: -len(tail_text)]
        tail_marker = "\nPLAYER_ACTION "
        marker_index = base.rfind(tail_marker)
        if marker_index == -1 and base.startswith("PLAYER_ACTION "):
            marker_index = 0
        if marker_index >= 0:
            base = base[:marker_index]
        parts = [base.rstrip("\n")]
        for block in inserted_blocks:
            text = str(block or "").strip()
            if text:
                parts.append(text)
        if tail_text:
            parts.append(tail_text)
        return "\n".join(part for part in parts if part)

    def _auto_advance_on_rails_story_context(
        self,
        campaign_state: Dict[str, object],
        *,
        action_text: str,
        narration: str,
        summary_update: object,
        state_update: Dict[str, object],
        player_state_update: Dict[str, object],
        character_updates: Dict[str, object],
        location_updates: Dict[str, object] | None = None,
        calendar_update: object,
    ) -> bool:
        if not bool(campaign_state.get("on_rails", False)):
            return False
        outline = campaign_state.get("story_outline")
        if not isinstance(outline, dict):
            return False
        chapters = outline.get("chapters")
        if not isinstance(chapters, list) or not chapters:
            return False
        if not isinstance(state_update, dict):
            return False
        if "current_chapter" in state_update or "current_scene" in state_update:
            return False

        action_clean = " ".join(str(action_text or "").strip().lower().split())
        if not action_clean:
            return False
        if action_clean in {
            "look",
            "l",
            "inventory",
            "inv",
            "i",
            "calendar",
            "cal",
            "events",
            "roster",
            "characters",
            "npcs",
        }:
            return False
        if action_clean.startswith("[ooc"):
            return False

        if self._is_emptyish_turn_payload(
            narration=narration,
            state_update=state_update,
            player_state_update=player_state_update if isinstance(player_state_update, dict) else {},
            summary_update=summary_update,
            xp_awarded=0,
            scene_image_prompt=None,
            character_updates=character_updates if isinstance(character_updates, dict) else {},
            location_updates=location_updates if isinstance(location_updates, dict) else {},
            calendar_update=calendar_update,
        ):
            return False

        old_ch = self._coerce_non_negative_int(
            campaign_state.get("current_chapter", 0), default=0
        )
        old_sc = self._coerce_non_negative_int(
            campaign_state.get("current_scene", 0), default=0
        )
        old_ch = min(old_ch, len(chapters) - 1)
        current_entry = chapters[old_ch] if 0 <= old_ch < len(chapters) else {}
        scenes = current_entry.get("scenes")
        if not isinstance(scenes, list):
            scenes = []

        looks_major = self._looks_like_major_narrative_beat(
            narration=narration,
            summary_update=summary_update,
            state_update=state_update,
            character_updates=character_updates if isinstance(character_updates, dict) else {},
            calendar_update=calendar_update,
        )
        has_player_motion = bool(
            isinstance(player_state_update, dict)
            and any(
                key in player_state_update
                for key in ("location", "room_title", "room_summary", "room_description")
            )
        )
        has_scene_signal = bool(str(summary_update or "").strip()) or has_player_motion or looks_major
        if not has_scene_signal:
            return False

        new_ch = old_ch
        new_sc = old_sc
        if scenes and old_sc + 1 < len(scenes):
            new_sc = old_sc + 1
        elif old_ch + 1 < len(chapters):
            new_ch = old_ch + 1
            new_sc = 0
            if isinstance(current_entry, dict):
                current_entry["completed"] = True
        else:
            return False

        campaign_state["current_chapter"] = new_ch
        campaign_state["current_scene"] = new_sc
        return True

    def _looks_like_major_narrative_beat(
        self,
        *,
        narration: str,
        summary_update: object,
        state_update: Dict[str, object],
        character_updates: Dict[str, object],
        calendar_update: object,
    ) -> bool:
        text = " ".join(
            (
                f"{str(narration or '')} "
                f"{str(summary_update or '')}"
            ).lower().split()
        )
        major_cues = (
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
            "truth",
            "identity",
            "confession",
            "explodes",
            "escape",
            "ambush",
        )
        if any(cue in text for cue in major_cues):
            return True
        if isinstance(character_updates, dict):
            for row in character_updates.values():
                if isinstance(row, dict) and str(row.get("deceased_reason") or "").strip():
                    return True
        if isinstance(calendar_update, dict) and calendar_update:
            if isinstance(calendar_update.get("add"), list) or isinstance(
                calendar_update.get("remove"), list
            ):
                return True
        return False

    def _source_lookup_requested_by_action(self, action_text: object) -> bool:
        if self._is_ooc_action_text(action_text):
            return False
        text = " ".join(str(action_text or "").strip().lower().split())
        if not text:
            return False
        intent_markers = (
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
        return any(marker in text for marker in intent_markers)

    # ── Memory / Source Material ──────────────────────────────────────────

    def _derive_auto_memory_queries(
        self,
        action_text: str,
        player_state: Dict[str, object],
        party_snapshot: List[Dict[str, object]],
        limit: int = 4,
    ) -> List[str]:
        out: List[str] = []
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

        _push(player_state.get("location"))
        _push(player_state.get("room_title"))
        player_name = " ".join(
            str(player_state.get("character_name") or "").strip().lower().split()
        )
        for row in party_snapshot[: self.MAX_PARTY_CONTEXT_PLAYERS]:
            if not isinstance(row, dict):
                continue
            name = " ".join(str(row.get("name") or "").strip().split())
            if not name:
                continue
            if name.lower() == player_name:
                continue
            _push(name)
            if len(out) >= limit:
                break
        _push(action_text)
        return out[: max(1, int(limit or 4))]

    def _should_force_auto_memory_search(self, action_text: str) -> bool:
        if self._is_ooc_action_text(action_text):
            return False
        text = " ".join(str(action_text or "").strip().lower().split())
        if not text or text.startswith("!"):
            return False
        if len(text) < 6:
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

    def _record_memory_search_usage_and_hints(
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
                key=lambda kv: (
                    self._coerce_non_negative_int(kv[1].get("count", 0), default=0),
                    kv[0],
                ),
                reverse=True,
            )
            usage = dict(ranked[: self.MEMORY_SEARCH_USAGE_MAX_TERMS])

        campaign_state[self.MEMORY_SEARCH_USAGE_KEY] = usage
        with self._session_factory() as session:
            db_row = session.get(Campaign, campaign.id)
            if db_row:
                db_row.state_json = self._dump_json(campaign_state)
                db_row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.commit()

        characters = self.get_campaign_characters(campaign)
        hints: List[Dict[str, object]] = []
        seen_keys: set[str] = set()
        for term_key in updated_keys:
            if term_key in seen_keys:
                continue
            seen_keys.add(term_key)
            hint_row = usage.get(term_key) or {}
            count = self._coerce_non_negative_int(hint_row.get("count", 0), default=0)
            if count < self.MEMORY_SEARCH_ROSTER_HINT_THRESHOLD:
                continue
            if not self._memory_search_term_looks_character_like(term_key):
                continue
            if isinstance(characters, dict) and self._resolve_existing_character_slug(characters, term_key):
                continue
            hints.append(
                {
                    "term": str(hint_row.get("label") or term_key),
                    "slug": term_key,
                    "count": count,
                }
            )
        return hints

    @staticmethod
    def _source_wildcard_matches(text: str, wildcard: str) -> bool:
        pattern = str(wildcard or "%").strip()
        if not pattern or pattern in {"*", "%", "%%"}:
            return True
        regex = re.escape(pattern.replace("*", "%")).replace("%", ".*")
        return bool(re.match(rf"(?is)^{regex}$", str(text or "").strip()))

    def _browse_builtin_source_keys(
        self,
        *,
        document_key: Optional[str] = None,
        wildcard: str = "%",
        limit: int = 255,
    ) -> List[str]:
        built_in_key = self.communication_rulebook_document_key()
        requested_key = (
            SourceMaterialMemory._normalize_source_document_key(str(document_key or ""))
            if document_key
            else ""
        )
        if requested_key and requested_key != built_in_key:
            return []
        pattern = str(wildcard or "%").strip()
        broad_browse = pattern in {"", "*", "%", "%%"}
        out: List[str] = []
        seen: set[str] = set()
        for line in self._communication_rulebook_lines():
            key_text = line.split(":", 1)[0].strip() if ":" in line else line
            target_text = key_text if broad_browse else line
            if not self._source_wildcard_matches(target_text, pattern):
                continue
            if broad_browse:
                entry = key_text if requested_key else f"{built_in_key}: {key_text}"
            else:
                entry = line
            normalized = " ".join(str(entry or "").lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(entry)
            if len(out) >= max(1, int(limit)):
                break
        return out

    def _search_builtin_source_material(
        self,
        query: str,
        *,
        document_key: Optional[str] = None,
        top_k: int = 5,
        before_lines: int = 0,
        after_lines: int = 0,
    ) -> List[Tuple[str, str, int, str, float]]:
        built_in_key = self.communication_rulebook_document_key()
        requested_key = (
            SourceMaterialMemory._normalize_source_document_key(str(document_key or ""))
            if document_key
            else ""
        )
        if requested_key and requested_key != built_in_key:
            return []
        query_text = " ".join(str(query or "").lower().split())
        if not query_text:
            return []
        query_terms = [term for term in re.split(r"[^a-z0-9]+", query_text) if term]
        lines = self._communication_rulebook_lines()
        scored: List[Tuple[int, float]] = []
        for idx, line in enumerate(lines, start=1):
            hay = line.lower()
            key_text = line.split(":", 1)[0].strip().lower() if ":" in line else ""
            score = 0.0
            if query_text in hay:
                score = 1.0
                if query_text in key_text:
                    score += 0.2
            elif query_terms:
                overlap = sum(1 for term in query_terms if term in hay)
                if overlap:
                    score = overlap / max(1, len(query_terms))
                    key_overlap = sum(1 for term in query_terms if term in key_text)
                    if key_overlap:
                        score += key_overlap / max(1, len(query_terms) * 2)
            if score > 0.0:
                scored.append((idx, score))
        scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        before_n = max(0, int(before_lines or 0))
        after_n = max(0, int(after_lines or 0))
        out: List[Tuple[str, str, int, str, float]] = []
        for center_idx, score in scored[: max(1, int(top_k))]:
            start_idx = max(1, center_idx - before_n)
            end_idx = min(len(lines), center_idx + after_n)
            window = [lines[i - 1] for i in range(start_idx, end_idx + 1)]
            out.append(
                (
                    built_in_key,
                    self.COMMUNICATION_RULEBOOK_DOCUMENT_LABEL,
                    center_idx,
                    "\n".join(window),
                    float(score),
                )
            )
        return out

    def _browse_source_keys(
        self,
        campaign_id: str,
        *,
        document_key: Optional[str] = None,
        wildcard: str = "%",
        limit: int = 255,
    ) -> List[str]:
        built_in = self._browse_builtin_source_keys(
            document_key=document_key,
            wildcard=wildcard,
            limit=limit,
        )
        requested_key = (
            SourceMaterialMemory._normalize_source_document_key(str(document_key or ""))
            if document_key
            else ""
        )
        if requested_key == self.communication_rulebook_document_key():
            return built_in[: max(1, int(limit))]
        rows = SourceMaterialMemory.browse_source_keys(
            campaign_id,
            document_key=document_key,
            wildcard=wildcard,
            limit=limit,
        )
        merged: List[str] = []
        seen: set[str] = set()
        for row in [*built_in, *rows]:
            normalized = " ".join(str(row or "").lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(str(row))
            if len(merged) >= max(1, int(limit)):
                break
        return merged

    def _search_source_material(
        self,
        query: str,
        campaign_id: str,
        *,
        document_key: Optional[str] = None,
        top_k: int = 5,
        before_lines: int = 0,
        after_lines: int = 0,
    ) -> List[Tuple[str, str, int, str, float]]:
        requested_key = (
            SourceMaterialMemory._normalize_source_document_key(str(document_key or ""))
            if document_key
            else ""
        )
        built_in_key = self.communication_rulebook_document_key()
        db_hits: List[Tuple[str, str, int, str, float]] = []
        if requested_key != built_in_key:
            db_hits = SourceMaterialMemory.search_source_material(
                query,
                campaign_id,
                document_key=document_key,
                top_k=top_k,
                before_lines=before_lines,
                after_lines=after_lines,
            )
        built_in_hits = self._search_builtin_source_material(
            query,
            document_key=document_key,
            top_k=top_k,
            before_lines=before_lines,
            after_lines=after_lines,
        )
        merged = [*built_in_hits, *db_hits]
        merged.sort(key=lambda row: float(row[4] or 0.0), reverse=True)
        seen: set[Tuple[str, int]] = set()
        out: List[Tuple[str, str, int, str, float]] = []
        for row in merged:
            row_key = (str(row[0] or "").strip(), int(row[2] or 0))
            if row_key in seen:
                continue
            seen.add(row_key)
            out.append(row)
            if len(out) >= max(1, int(top_k)):
                break
        return out

    def _list_source_material_documents(
        self,
        campaign_id: str,
        *,
        limit: int = 20,
    ) -> List[Dict[str, object]]:
        docs = SourceMaterialMemory.list_source_material_documents(
            campaign_id,
            limit=max(1, int(limit)),
        )
        built_in_doc = {
            "document_key": self.communication_rulebook_document_key(),
            "document_label": self.COMMUNICATION_RULEBOOK_DOCUMENT_LABEL,
            "chunk_count": len(self.DEFAULT_GM_COMMUNICATION_RULES),
            "sample_chunk": "\n".join(self._communication_rulebook_lines()[:6]),
        }
        return [built_in_doc, *docs][: max(1, int(limit))]

    # ── Misc Core ─────────────────────────────────────────────────────────

    async def _compress_autobiography_with_model(
        self,
        *,
        character_slug: str,
        character_name: str,
        current_autobiography: str,
        raw_entries: list[dict[str, object]],
    ) -> str:
        current_text = self._sanitize_autobiography_text(
            current_autobiography,
            max_chars=self.MAX_AUTOBIOGRAPHY_TEXT_CHARS,
        )
        entry_lines: list[str] = []
        for row in raw_entries[-16:]:
            if not isinstance(row, dict):
                continue
            a_text = self._sanitize_autobiography_text(
                row.get("a"),
                max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
            )
            b_text = self._sanitize_autobiography_text(
                row.get("b"),
                max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
            )
            c_text = self._sanitize_autobiography_text(
                row.get("c"),
                max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
            )
            legacy_text = self._sanitize_autobiography_text(
                row.get("text"),
                max_chars=self.MAX_AUTOBIOGRAPHY_ENTRY_CHARS,
            )
            if not (a_text or b_text or c_text or legacy_text):
                continue
            stamp = ""
            gt = row.get("game_time")
            if isinstance(gt, dict):
                day = self._coerce_non_negative_int(gt.get("day"), default=0)
                hour = min(23, max(0, self._coerce_non_negative_int(gt.get("hour"), default=0)))
                minute = min(59, max(0, self._coerce_non_negative_int(gt.get("minute"), default=0)))
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
            payload_text = json.dumps(raw_row, ensure_ascii=True)
            entry_lines.append(f"- {stamp}{payload_text}".strip())
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
        user_prompt = (
            f"CHARACTER: {character_name} ({character_slug})\n"
            f"CURRENT_AUTOBIOGRAPHY: {current_text or '(none)'}\n"
            "RAW_ENTRIES:\n"
            f"{chr(10).join(entry_lines) or '(none)'}\n"
            f"Write a compressed autobiography no longer than {self.MAX_AUTOBIOGRAPHY_TEXT_CHARS} characters."
        )
        if self._completion_port is None:
            return current_text or ""
        response = await self._completion_port.complete(
            system_prompt,
            user_prompt,
            temperature=0.3,
            max_tokens=700,
        )
        cleaned = self._clean_response(response or "")
        cleaned = re.sub(r"^```[\w-]*\s*", "", cleaned).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        return self._sanitize_autobiography_text(
            cleaned,
            max_chars=self.MAX_AUTOBIOGRAPHY_TEXT_CHARS,
        )

    def _fallback_narration_from_payload(self, payload: Dict[str, object]) -> str:
        if not isinstance(payload, dict):
            return ""
        player_state_update = payload.get("player_state_update")
        if isinstance(player_state_update, dict):
            room_summary = str(player_state_update.get("room_summary") or "").strip()
            if room_summary:
                return room_summary[:300]
            room_title = str(player_state_update.get("room_title") or "").strip()
            if room_title:
                return f"{room_title}."
        summary_update = str(payload.get("summary_update") or "").strip()
        if summary_update:
            return summary_update.splitlines()[0][:300]
        character_updates = payload.get("character_updates")
        if isinstance(character_updates, dict) and character_updates:
            return "Character roster updated."
        calendar_update = payload.get("calendar_update")
        if isinstance(calendar_update, dict) and calendar_update:
            return "Calendar updated."
        state_update = payload.get("state_update")
        if isinstance(state_update, dict) and state_update:
            return "Noted."
        if isinstance(player_state_update, dict) and player_state_update:
            return "Noted."
        return ""

    def _brief_event_summary(
        self,
        *,
        action_text: str,
        summary_update: object,
        narration_text: str,
    ) -> str:
        summary = " ".join(str(summary_update or "").strip().split())
        if summary:
            return self._trim_text(summary, 260)
        narration_lines = []
        for line in str(narration_text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("inventory:"):
                continue
            if stripped.startswith("\u23f0"):
                continue
            narration_lines.append(stripped)
            if len(narration_lines) >= 2:
                break
        if narration_lines:
            return self._trim_text(" ".join(narration_lines), 260)
        return self._trim_text(" ".join(str(action_text or "").strip().split()), 180)
