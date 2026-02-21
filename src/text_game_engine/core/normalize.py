from __future__ import annotations

import json
import re
from typing import Any

from .ports import ActorResolverPort
from .types import GiveItemInstruction


def normalize_campaign_name(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-zA-Z0-9 _-]", "", value)
    return (value.lower()[:64] or "main")


def parse_json_dict(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        data = json.loads(text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def dump_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def apply_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def normalize_give_item(
    raw: dict[str, Any] | None,
    actor_resolver: ActorResolverPort | None,
) -> tuple[GiveItemInstruction | None, str | None]:
    if not isinstance(raw, dict):
        return None, None
    item = str(raw.get("item") or "").strip()
    if not item:
        return None, "missing_item"

    to_actor_id = raw.get("to_actor_id")
    if to_actor_id is not None:
        to_actor_id = str(to_actor_id).strip() or None

    mention = raw.get("to_discord_mention")
    mention = str(mention).strip() if mention is not None else None

    if not to_actor_id and mention and actor_resolver is not None:
        to_actor_id = actor_resolver.resolve_discord_mention(mention)

    instruction = GiveItemInstruction(
        item=item,
        to_actor_id=to_actor_id,
        to_discord_mention=mention,
    )
    if instruction.to_actor_id:
        return instruction, None
    return instruction, "unresolved_target"
