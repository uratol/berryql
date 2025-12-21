from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from sqlalchemy import func

# Global operator registry (extensible)
OPERATOR_REGISTRY: Dict[str, Callable[[Any, Any], Any]] = {
    'eq': lambda col, v: col == v,
    'ne': lambda col, v: col != v,
    'lt': lambda col, v: col < v,
    'lte': lambda col, v: col <= v,
    'gt': lambda col, v: col > v,
    'gte': lambda col, v: col >= v,
    'like': lambda col, v: col.like(v),
    'not_like': lambda col, v: ~col.like(v),
    'ilike': lambda col, v: getattr(col, 'ilike', lambda x: func.lower(col).like(func.lower(x)))(v),
    'not_ilike': lambda col, v: ~getattr(col, 'ilike', lambda x: func.lower(col).like(func.lower(x)))(v),
    'in': lambda col, v: col.in_(v if isinstance(v, (list, tuple, set)) else [v]),
    'not_in': lambda col, v: ~col.in_(v if isinstance(v, (list, tuple, set)) else [v]),
    'between': lambda col, v: col.between(v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else None,
    'not_between': lambda col, v: ~col.between(v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else None,
    'contains': lambda col, v: col.contains(v),
    'starts_with': lambda col, v: col.like(f"{v}%"),
    'ends_with': lambda col, v: col.like(f"%{v}"),
}

@dataclass
class FilterSpec:
    column: Optional[str] = None
    op: Optional[str] = None
    ops: Optional[List[str]] = None
    transform: Optional[Callable[[Any], Any]] = None
    alias: Optional[str] = None
    builder: Optional[Callable[..., Any]] = None
    required: bool = False
    description: Optional[str] = None
    # Explicit GraphQL argument type override (e.g., int, str, bool, datetime)
    # If provided, this takes precedence over inferring from `column`.
    arg_type: Optional[Any] = None

    def clone_with(self, **overrides: Any) -> "FilterSpec":
        data = {
            'column': self.column,
            'op': self.op,
            'ops': self.ops,
            'transform': self.transform,
            'alias': self.alias,
            'builder': self.builder,
            'required': self.required,
            'description': self.description,
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        return FilterSpec(**data)

def normalize_filter_spec(raw: Any) -> FilterSpec:
    if isinstance(raw, FilterSpec):
        return raw
    if callable(raw):
        return FilterSpec(builder=raw)
    if isinstance(raw, dict):
        return FilterSpec(
            column=raw.get('column'),
            op=raw.get('op'),
            ops=raw.get('ops'),
            transform=raw.get('transform'),
            alias=raw.get('alias'),
            builder=raw.get('builder'),
            required=raw.get('required', False),
            description=raw.get('description'),
            arg_type=raw.get('arg_type') or raw.get('type') or raw.get('returns')
        )
    raise TypeError(f"Unsupported filter spec form: {raw!r}")

def register_operator(name: str, fn: Callable[[Any, Any], Any]):  # pragma: no cover - simple
    OPERATOR_REGISTRY[name] = fn
