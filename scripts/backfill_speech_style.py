#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re

from text_game_engine.persistence.sqlalchemy import (
    build_engine,
    build_session_factory,
    create_schema,
)
from text_game_engine.persistence.sqlalchemy.models import Campaign


DEFAULT_SPEECH_STYLE = (
    "Uses concise, concrete sentences. Avoids grand speeches and flowery language. "
    "Asks direct questions only when needed and responds with clear intent."
)


def infer_speech_style(entry: dict) -> str:
    personality = " ".join(str(entry.get("personality") or "").strip().lower().split())
    if "formal" in personality or "professional" in personality:
        return (
            "Speaks in measured, professional sentences. Chooses precise words and avoids slang. "
            "Keeps emotional language restrained unless directly provoked."
        )
    if "sarcastic" in personality or "cynical" in personality:
        return (
            "Uses short, sharp sentences with dry sarcasm. Prefers concrete complaints over abstract emotion. "
            "Avoids earnest sentiment unless trust is clearly established."
        )
    if "warm" in personality or "kind" in personality:
        return (
            "Uses calm, supportive phrasing with medium-length sentences. "
            "Names practical details before emotional interpretation. Avoids dramatic exaggeration."
        )
    return DEFAULT_SPEECH_STYLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing speech_style fields in campaign character rosters.")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("TGE_DATABASE_URL", "sqlite:///./text_game_engine.db"),
        help="SQLAlchemy database URL (default: env TGE_DATABASE_URL or sqlite:///./text_game_engine.db)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = build_engine(args.database_url)
    create_schema(engine)
    session_factory = build_session_factory(engine)

    campaigns_scanned = 0
    campaigns_changed = 0
    characters_changed = 0

    with session_factory() as session:
        campaigns = session.query(Campaign).order_by(Campaign.id.asc()).all()
        for campaign in campaigns:
            campaigns_scanned += 1
            try:
                characters = json.loads(campaign.characters_json or "{}")
            except Exception:
                continue
            if not isinstance(characters, dict):
                continue
            changed = False
            for _slug, entry in characters.items():
                if not isinstance(entry, dict):
                    continue
                existing = str(entry.get("speech_style") or "").strip()
                if existing:
                    continue
                style = infer_speech_style(entry)
                style = re.sub(r"\s+", " ", style).strip()
                if not style:
                    style = DEFAULT_SPEECH_STYLE
                entry["speech_style"] = style
                changed = True
                characters_changed += 1
            if changed:
                campaign.characters_json = json.dumps(characters, ensure_ascii=True)
                campaigns_changed += 1
        session.commit()

    print(
        "speech_style backfill complete: "
        f"campaigns_scanned={campaigns_scanned} "
        f"campaigns_changed={campaigns_changed} "
        f"characters_changed={characters_changed}"
    )


if __name__ == "__main__":
    main()
