#!/usr/bin/env python3

import unicodedata


LOCAL_COMMANDS = {
    "gunaydin": "Günaydın Sinan. Hazırım.",
    "iyi aksamlar": "İyi akşamlar Sinan. Buradayım.",
    "yi aksamlar": "İyi akşamlar Sinan. Buradayım.",
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


def match_local_command(text: str) -> str | None:
    return LOCAL_COMMANDS.get(normalize_command(text))
