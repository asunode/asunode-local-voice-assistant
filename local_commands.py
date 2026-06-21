#!/usr/bin/env python3

import unicodedata


LOCAL_COMMANDS = {
    "gunaydin": "Günaydın {user_name}. Hazırım.",
    "iyi aksamlar": "İyi akşamlar {user_name}. Buradayım.",
    "yi aksamlar": "İyi akşamlar {user_name}. Buradayım.",
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
) -> str | None:
    response = LOCAL_COMMANDS.get(normalize_command(text))

    if response is None:
        return None

    return response.format(user_name=user_name)
