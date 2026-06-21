#!/usr/bin/env python3

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_TIMEZONE = "Europe/Istanbul"

TURKISH_MONTHS = (
    "Ocak",
    "Şubat",
    "Mart",
    "Nisan",
    "Mayıs",
    "Haziran",
    "Temmuz",
    "Ağustos",
    "Eylül",
    "Ekim",
    "Kasım",
    "Aralık",
)

TURKISH_DAYS = (
    "Pazartesi",
    "Salı",
    "Çarşamba",
    "Perşembe",
    "Cuma",
    "Cumartesi",
    "Pazar",
)


def get_local_now(
    timezone_name: str,
) -> datetime:
    try:
        timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        timezone = ZoneInfo(DEFAULT_TIMEZONE)

    return datetime.now(timezone)


def format_turkish_date(
    value: datetime,
) -> str:
    return (
        f"{value.day} "
        f"{TURKISH_MONTHS[value.month - 1]} "
        f"{value.year}"
    )


def format_turkish_day(
    value: datetime,
) -> str:
    return TURKISH_DAYS[value.weekday()]


def format_turkish_time(
    value: datetime,
) -> str:
    return value.strftime("%H:%M")


def format_turkish_datetime(
    value: datetime,
) -> str:
    return (
        f"{format_turkish_date(value)} "
        f"{format_turkish_day(value)}, "
        f"{format_turkish_time(value)}"
    )
