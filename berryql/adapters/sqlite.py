from __future__ import annotations
from sqlalchemy import func, literal
from .base import BaseAdapter

class SQLiteAdapter(BaseAdapter):
    name = 'sqlite'
    def json_object(self, *args):
        return func.json_object(*args)
    def json_array_agg(self, expr):
        return func.json_group_array(expr)
    def json_array_coalesce(self, expr):
        # Use a bound literal to avoid deprecation warnings for implicit string coercion
        return func.coalesce(expr, literal('[]'))
