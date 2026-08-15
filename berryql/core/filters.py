from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from sqlalchemy import func
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import Select


def _is_sql_operand(value: Any) -> bool:
    """True when ``value`` is a SQLAlchemy selectable or column expression.

    Trusted scopes may embed SQL constructs as operator operands, e.g.
    ``{'business_id': {'in': select(...).scalar_subquery()}}``.  Such operands
    must reach ``column.in_()`` unwrapped and carry no list-length semantics.
    Values arriving from GraphQL callers are always plain JSON types and can
    never satisfy this check.
    """
    if isinstance(value, (list, tuple, set, dict, str, bytes)):
        return False
    if isinstance(value, (Select, ColumnElement)):
        return True
    # Duck-typing fallback for dialect-specific wrappers and similar
    # constructs that are not direct subclasses of the public types.
    return callable(getattr(value, "self_group", None)) and callable(getattr(value, "compile", None))


def _in_value(value: Any) -> Any:
    """Normalize an ``in``/``not_in`` operand for ``column.in_()``.

    - sequences pass through unchanged,
    - SQL selectables/expressions pass through unwrapped so SQLAlchemy renders
      ``col IN (SELECT ...)`` (wrapping a plain ``Select`` in a list raises
      ``ArgumentError: IN expression list, SELECT construct, or bound parameter
      object expected``),
    - scalars are wrapped into a single-element list (legacy behavior).
    """
    if isinstance(value, (list, tuple, set)) or _is_sql_operand(value):
        return value
    return [value]


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
    'in': lambda col, v: col.in_(_in_value(v)),
    'not_in': lambda col, v: ~col.in_(_in_value(v)),
    'between': lambda col, v: col.between(v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else None,
    'not_between': lambda col, v: ~col.between(v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else None,
    'contains': lambda col, v: col.contains(v),
    'starts_with': lambda col, v: col.like(f"{v}%"),
    'ends_with': lambda col, v: col.like(f"%{v}"),
}

@dataclass(frozen=True)
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
    # Canonical Berry field paths read by a transform/builder. ``fields`` is
    # retained as a friendly declaration alias; both are additive metadata.
    depends_on: Tuple[str, ...] = field(default_factory=tuple)
    fields: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'ops', list(self.ops) if self.ops is not None else None)
        object.__setattr__(self, 'depends_on', _dependency_tuple(self.depends_on))
        object.__setattr__(self, 'fields', _dependency_tuple(self.fields))

    @property
    def dependencies(self) -> Tuple[str, ...]:
        values = []
        for value in ((self.column,) if self.column else ()) + self.depends_on + self.fields:
            if value not in values:
                values.append(value)
        return tuple(values)

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
            'arg_type': self.arg_type,
            'depends_on': self.depends_on,
            'fields': self.fields,
        }
        # ``None`` is a meaningful override for op/ops during multi-op
        # expansion, so preserve every explicitly supplied key.
        data.update(overrides)
        return FilterSpec(**data)


def _dependency_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    raise TypeError("FilterSpec dependency metadata must be a string or iterable")

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
            arg_type=raw.get('arg_type') or raw.get('type') or raw.get('returns'),
            depends_on=raw.get('depends_on') or (),
            fields=raw.get('fields') or (),
        )
    raise TypeError(f"Unsupported filter spec form: {raw!r}")

def register_operator(name: str, fn: Callable[[Any, Any], Any]):  # pragma: no cover - simple
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Operator name must be a non-empty string")
    if not callable(fn):
        raise TypeError("Operator implementation must be callable")
    OPERATOR_REGISTRY[name] = fn


class OperatorRegistry:
    """Schema-local operator registry seeded from compatibility defaults."""

    def __init__(self, operators: Optional[Mapping[str, Callable[[Any, Any], Any]]] = None):
        self._operators = dict(OPERATOR_REGISTRY)
        for name, implementation in (operators or {}).items():
            self.register(name, implementation)

    def register(self, name: str, implementation: Callable[[Any, Any], Any]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Operator name must be a non-empty string")
        if not callable(implementation):
            raise TypeError("Operator implementation must be callable")
        self._operators[name] = implementation

    def get(self, name: str) -> Optional[Callable[[Any, Any], Any]]:
        return self._operators.get(name)

    def snapshot(self) -> Mapping[str, Callable[[Any, Any], Any]]:
        return dict(self._operators)


@dataclass(frozen=True)
class FilterLimits:
    """Optional caller-filter cost limits.

    ``None`` keeps the historical permissive behaviour for each dimension.
    Trusted scopes are validated for shape but are not subject to caller cost
    limits.
    """

    max_clauses: Optional[int] = None
    max_in_items: Optional[int] = None
    max_json_length: Optional[int] = None
    max_depth: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("max_clauses", "max_in_items", "max_json_length", "max_depth"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"FilterLimits.{name} must be a non-negative integer or None")
