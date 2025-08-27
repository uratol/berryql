"""BerryQL public API and lightweight lazy exports.

This __init__ avoids importing heavy submodules at import time to prevent
circular imports when foundational layers (like database models) import
BerryQL utilities (e.g., enum helpers).

Exposes:
- get_active_schema, set_active_schema (without importing registry eagerly)
- Lazy attributes: BerrySchema, BerryType, BerryDomain, StrawberryConfig
- Lazy functions: field, relation, aggregate, count, custom, custom_object, domain, mutation
- enum_column (resolved lazily from .sql.enum_helpers)
"""
from __future__ import annotations

from typing import Any

_ACTIVE_SCHEMA: Any = None


def set_active_schema(schema: Any) -> None:
    global _ACTIVE_SCHEMA
    _ACTIVE_SCHEMA = schema


def get_active_schema() -> Any:
    if _ACTIVE_SCHEMA is None:
        raise RuntimeError("Active Berry schema not set. Ensure schema.py initialized it.")
    return _ACTIVE_SCHEMA


def __getattr__(name: str):  # PEP 562 lazy exports
    if name in {'BerrySchema', 'BerryType', 'BerryDomain', 'StrawberryConfig'}:
        from . import registry as _registry
        return getattr(_registry, name)
    if name in {'field', 'relation', 'aggregate', 'count', 'custom', 'custom_object', 'domain', 'mutation'}:
        from .core import fields as _fields
        return getattr(_fields, name)
    if name == 'enum_column':
        from .sql.enum_helpers import enum_column as _enum_column
        return _enum_column
    raise AttributeError(name)


__all__ = [
    'BerrySchema', 'BerryType', 'BerryDomain', 'StrawberryConfig',
    'field', 'relation', 'aggregate', 'count', 'custom', 'custom_object', 'domain', 'mutation',
    'enum_column',
    'get_active_schema', 'set_active_schema',
]
