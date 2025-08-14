from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

@dataclass
class FieldDef:
    name: str
    kind: str
    meta: Dict[str, Any]

class FieldDescriptor:
    def __init__(self, *, kind: str, **meta):
        self.kind = kind
        self.meta = dict(meta)
        self.name: str | None = None

    def __set_name__(self, owner, name):  # pragma: no cover - simple
        self.name = name

    def build(self, parent_name: str) -> FieldDef:
        return FieldDef(name=self.name or '', kind=self.kind, meta=self.meta)

def field(**meta) -> FieldDescriptor:
    return FieldDescriptor(kind='scalar', **meta)

def relation(target: Any = None, *, single: bool | None = None, **meta) -> FieldDescriptor:
    m = dict(meta)
    if target is not None:
        m['target'] = target.__name__ if hasattr(target, '__name__') and not isinstance(target, str) else target
    if single is not None:
        m['single'] = single
    return FieldDescriptor(kind='relation', **m)

def aggregate(source: str, **meta) -> FieldDescriptor:
    return FieldDescriptor(kind='aggregate', source=source, **meta)

def count(source: str) -> FieldDescriptor:
    return aggregate(source, op='count')

def custom(builder: Callable[..., Any], *, returns: Any | None = None) -> FieldDescriptor:
    return FieldDescriptor(kind='custom', builder=builder, returns=returns)

def custom_object(builder: Callable[..., Any], *, returns: Any) -> FieldDescriptor:
    return FieldDescriptor(kind='custom_object', builder=builder, returns=returns)
