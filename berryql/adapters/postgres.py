from __future__ import annotations
from sqlalchemy import func, text as _text
from .base import BaseAdapter

class PostgresAdapter(BaseAdapter):
    name = 'postgres'
    def json_object(self, *args):
        return func.json_build_object(*args)
    def json_array_agg(self, expr):
        return func.json_agg(expr)
    def json_array_coalesce(self, expr):
        # Prefer a typed literal over raw text for JSON to avoid deprecation warnings
        try:
            from sqlalchemy.dialects.postgresql import JSON as _PG_JSON
            from sqlalchemy import literal
            return func.coalesce(expr, literal('[]', type_=_PG_JSON()))
        except Exception:
            # Fallback keeps previous behavior if dialect types aren't available at import time
            return func.coalesce(expr, _text("'[]'::json"))
