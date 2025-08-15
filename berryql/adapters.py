"""Compatibility shim for adapters.

This module now re-exports from berryql.adapters package.
"""
from .adapters import (  # type: ignore[F401]
    BaseAdapter,
    SQLiteAdapter,
    PostgresAdapter,
    MSSQLAdapter,
    get_adapter,
)
