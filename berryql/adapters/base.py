from __future__ import annotations
from typing import Any
from sqlalchemy import func, literal

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

    # Table identifier helper; adapters can override for dialect-specific quoting/qualification
    def table_ident(self, model_cls) -> str:
        try:
            # Prefer SQLAlchemy Table if available
            tbl = getattr(model_cls, '__table__', None)
            if tbl is not None:
                schema = getattr(tbl, 'schema', None)
                name = getattr(tbl, 'name', None) or getattr(model_cls, '__tablename__', None)
                if schema:
                    return f"{schema}.{name}"
                return str(name)
        except Exception:
            pass
        # Fallback: use __tablename__
        try:
            return str(getattr(model_cls, '__tablename__'))
        except Exception:
            return ''
