from __future__ import annotations
from typing import Any
from sqlalchemy import func, text as _text

class BaseAdapter:
    name = 'base'
    def json_object(self, *args):
        raise NotImplementedError
    def json_array_agg(self, expr):
        raise NotImplementedError
    def json_array_coalesce(self, expr):
        raise NotImplementedError
    def supports_relation_pushdown(self) -> bool:
        return True

class SQLiteAdapter(BaseAdapter):
    name = 'sqlite'
    def json_object(self, *args):
        return func.json_object(*args)
    def json_array_agg(self, expr):
        return func.json_group_array(expr)
    def json_array_coalesce(self, expr):
        return func.coalesce(expr, '[]')

class PostgresAdapter(BaseAdapter):
    name = 'postgres'
    def json_object(self, *args):
        return func.json_build_object(*args)
    def json_array_agg(self, expr):
        return func.json_agg(expr)
    def json_array_coalesce(self, expr):
        return func.coalesce(expr, _text("'[]'::json"))

class MSSQLAdapter(BaseAdapter):
    name = 'mssql'
    def json_object(self, *args):
        # Basic single pair fallback; full JSON support TBD
        return func.concat('{', args[0], ':', args[1], '}') if len(args) >= 2 else func.concat('{','}')
    def json_array_agg(self, expr):
        return None  # signal unsupported
    def json_array_coalesce(self, expr):
        return None
    def supports_relation_pushdown(self) -> bool:
        return False

def get_adapter(dialect_name: str) -> BaseAdapter:
    dn = (dialect_name or '').lower()
    if dn.startswith('postgres'):
        return PostgresAdapter()
    if dn.startswith('mssql') or 'pyodbc' in dn:
        return MSSQLAdapter()
    return SQLiteAdapter()

__all__ = ['get_adapter','BaseAdapter','SQLiteAdapter','PostgresAdapter','MSSQLAdapter']
