"""Common naming utilities for BerryQL (DRY consolidation).

Provides consistent camelCase/PascalCase to snake_case conversion used by
FieldMapper, query analyzer, and resolved data helpers.
"""
from __future__ import annotations

import re

__all__ = ["camel_to_snake", "snake_to_camel"]

def camel_to_snake(name: str) -> str:
    """Convert camelCase or PascalCase identifier to snake_case.

    Idempotent for already snake_case input. Handles sequences of capitals.
    """
    if not isinstance(name, str) or not name:
        return name  # type: ignore
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def snake_to_camel(name: str, upper_first: bool = False) -> str:
    """Convert snake_case identifier to camelCase or PascalCase.

    upper_first=False returns lowerCamelCase (default), True returns UpperCamelCase.
    Idempotent for already camelCase strings without underscores.
    """
    if not isinstance(name, str) or not name:
        return name  # type: ignore
    if '_' not in name:
        if upper_first:
            return name[0].upper() + name[1:]
        return name
    parts = [p for p in name.split('_') if p]
    if not parts:
        return ''
    first = parts[0].lower() if not upper_first else parts[0].capitalize()
    rest = ''.join(p.capitalize() for p in parts[1:])
    return first + rest
