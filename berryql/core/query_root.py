from __future__ import annotations

from typing import Any, Optional


class QueryRootAssembler:
    """Owns query-root assembly orchestration.

    Dynamic Strawberry details remain private to the registry implementation;
    callers enter through this service so schema registration and root execution
    have separate lifecycle boundaries.
    """

    def __init__(self, schema: Any):
        self.schema = schema

    def build(self, *, strawberry_config: Optional[Any] = None) -> Any:
        return self.schema._build_query_impl(strawberry_config=strawberry_config)


__all__ = ["QueryRootAssembler"]
