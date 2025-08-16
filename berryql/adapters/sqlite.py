from __future__ import annotations
from sqlalchemy import func, literal
from sqlalchemy.sql.sqltypes import LargeBinary as _LB
from .base import BaseAdapter

class SQLiteAdapter(BaseAdapter):
    name = 'sqlite'
    def json_object(self, *args):
        # SQLite JSON cannot hold BLOBs; convert binary-like values to hex text
        conv: list = []
        for i, a in enumerate(args or ()):  # args come as key, value, key, value...
            if i % 2 == 0:
                conv.append(a)
            else:
                v = a
                try:
                    t = getattr(v, 'type', None)
                    is_bin = isinstance(t, _LB)
                except Exception:
                    is_bin = False
                if is_bin:
                    v = func.lower(func.hex(v))
                conv.append(v)
        return func.json_object(*conv)
    def json_array_agg(self, expr):
        return func.json_group_array(expr)
    def json_array_coalesce(self, expr):
        # Use a bound literal to avoid deprecation warnings for implicit string coercion
        return func.coalesce(expr, literal('[]'))
