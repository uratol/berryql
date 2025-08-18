"""Next-gen declarative Berry layer (clean-slate).

Public API:
- BerrySchema, BerryType, BerryDomain, StrawberryConfig
- field, relation, aggregate, count, custom, custom_object, domain

Additionally exposes get_active_schema/set_active_schema so resolvers can access
the single schema instance configured in app.graphql.schema without importing it
directly (avoiding circular imports).
"""
from typing import Optional
from .registry import BerrySchema, BerryType, BerryDomain, StrawberryConfig
from .core.fields import field, relation, aggregate, count, custom, custom_object, domain

_ACTIVE_SCHEMA: Optional[BerrySchema] = None

def set_active_schema(schema: BerrySchema) -> None:
    global _ACTIVE_SCHEMA
    _ACTIVE_SCHEMA = schema

def get_active_schema() -> BerrySchema:
    if _ACTIVE_SCHEMA is None:
        raise RuntimeError("Active Berry schema not set. Ensure schema.py initialized it.")
    return _ACTIVE_SCHEMA

__all__ = [
    'BerrySchema', 'BerryType', 'BerryDomain', 'StrawberryConfig',
    'field', 'relation', 'aggregate', 'count', 'custom', 'custom_object', 'domain',
    'get_active_schema', 'set_active_schema'
]
