"""Shared, dependency-free normalization helpers used by identity resolution
and evidence retrieval alike (kept separate to avoid a circular import
between `identity.py` and `retrieval.py`).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

_PHONE_DIGITS_RE = re.compile(r"\D")


def parse_iso8601(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def normalize_email(value: str) -> str:
    return value.strip().lower()


def normalize_phone(value: str) -> str:
    return _PHONE_DIGITS_RE.sub("", value)
