from __future__ import annotations

from .base import BaseAdapter
from .sqlite import SQLiteAdapter
from .postgres import PostgresAdapter
from .mssql import MSSQLAdapter


def get_adapter(dialect_name: str) -> BaseAdapter:
    dn = (dialect_name or '').lower()
    if dn.startswith('postgres'):
        return PostgresAdapter()
    if dn.startswith('mssql') or 'pyodbc' in dn:
        return MSSQLAdapter()
    return SQLiteAdapter()


__all__ = [
    'BaseAdapter',
    'SQLiteAdapter',
    'PostgresAdapter',
    'MSSQLAdapter',
    'get_adapter',
]
