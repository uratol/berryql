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
