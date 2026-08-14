from __future__ import annotations

from typing import Any, Mapping


class RelationResolverFactory:
    """Factory boundary for generated relation resolvers and fallback execution."""

    def __init__(self, schema: Any):
        self.schema = schema

    def create(
        self,
        metadata: Mapping[str, Any],
        *,
        single: bool,
        field_name: str,
        parent_type: Any,
    ) -> Any:
        return self.schema._create_relation_resolver_impl_factory(dict(metadata), single, field_name, parent_type)


__all__ = ["RelationResolverFactory"]
