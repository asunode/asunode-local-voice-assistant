#!/usr/bin/env python3

import unicodedata
from datetime import datetime

from local_datetime import (
    format_turkish_date,
    format_turkish_day,
    format_turkish_time,
    get_local_now,
)


LOCAL_COMMANDS = {
    "gunaydin": "Günaydın {user_name}. Hazırım.",
    "iyi aksamlar": "İyi akşamlar {user_name}. Buradayım.",
    "yi aksamlar": "İyi akşamlar {user_name}. Buradayım.",
}

TIME_COMMANDS = {
    "saat kac",
    "saatkac",
    "su an saat kac",
    "su an saatkac",
    "suan saat kac",
    "suan saatkac",
    "simdi saat kac",
    "simdi saatkac",
}

DATE_COMMANDS = {
    "bugunun tarihi nedir",
    "bugunun tarihi ne",
    "bugunun tarihi",
    "bugun tarih nedir",
    "bugun tarih ne",
    "bugun tarih",
}

DAY_COMMANDS = {
    "bugun gunlerden ne",
    "bugun gunlerden nedir",
    "bugun hangi gun",
    "bugun hangi gunudur",
}


def normalize_command(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFKD",
        text.casefold().replace("ı", "i"),
    )

    characters = []

    for character in normalized:
        category = unicodedata.category(character)

        if category.startswith("M"):
            continue

        if category.startswith("P"):
            characters.append(" ")
            continue

        characters.append(character)

    words = "".join(characters).split()

    if words and words[0] in {
        "kuleye",
        "kuleya",
        "kureye",
        "kureya",
    }:
        words = words[1:]
    elif words and words[0] in {"kule", "kure"}:
        words = words[1:]

        if words and words[0] in {"ye", "ya"}:
            words = words[1:]

    return " ".join(words)


def match_local_command(
    text: str,
    user_name: str,
    timezone_name: str,
    now: datetime | None = None,
) -> str | None:
    normalized = normalize_command(text)
    response = LOCAL_COMMANDS.get(normalized)

    if response is not None:
        return response.format(user_name=user_name)

    if normalized not in (
        TIME_COMMANDS
        | DATE_COMMANDS
        | DAY_COMMANDS
    ):
        return None

    current_time = now or get_local_now(timezone_name)

    if normalized in TIME_COMMANDS:
        return (
            f"Şu an saat {format_turkish_time(current_time)}, "
            f"{user_name}."
        )

    if normalized in DATE_COMMANDS:
        return (
            f"Bugün {format_turkish_date(current_time)}, "
            f"{user_name}."
        )

    return (
        f"Bugün {format_turkish_day(current_time)}, "
        f"{user_name}."
    )
