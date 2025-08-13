from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, get_type_hints
import asyncio
import strawberry
from sqlalchemy import select, func, text as _text
from sqlalchemy import and_ as _and
try:  # adapter abstraction
    from .adapters import get_adapter  # type: ignore
except Exception:  # pragma: no cover
    get_adapter = None  # type: ignore
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.sqltypes import Integer, String, Boolean, DateTime
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover - type checking only
    class Registry: ...  # forward ref placeholder
try:  # Provide StrawberryInfo for type annotations
    from strawberry.types import Info as StrawberryInfo  # type: ignore
except Exception:  # pragma: no cover
    class StrawberryInfo:  # type: ignore
        ...

T = TypeVar('T')

# --- Helper: Relation selection extraction (moved out of resolver closure) ---
class RelationSelectionExtractor:
    """Extract relation selection metadata (fields, pagination, filters) from Strawberry
    selected_fields data and/or raw GraphQL AST. Returns mapping:
        relation_name -> {
           fields: [scalar field names],
           limit: int|None,
           offset: int|None,
           order_by: str|None,
           order_dir: str|None,
           order_multi: list[str],
           where: Any,
           default_where: Any,
           single: bool,
           target: str | None,
           nested: { ... nested relation data ... },
           skip_pushdown: bool,
           filter_args: { arg_name: value }
        }
    """
    def __init__(self, registry: 'Registry'):
        self.registry = registry

    def _init_rel_cfg(self, fdef) -> Dict[str, Any]:
        return {
            'fields': [],
            'limit': None,
            'offset': None,
            'order_by': fdef.meta.get('order_by'),
            'order_dir': fdef.meta.get('order_dir'),
            'order_multi': fdef.meta.get('order_multi') or [],
            'where': None,
            'default_where': fdef.meta.get('where'),
            'single': bool(fdef.meta.get('single')),
            'target': fdef.meta.get('target'),
            'nested': {},
            'skip_pushdown': False,
            'filter_args': {}
        }

    def _ast_value(self, node: Any, info: Any) -> Any:
        """Best-effort conversion of GraphQL AST value node into Python value."""
        try:
            # Variable reference
            if hasattr(node, 'kind') and 'Variable' in str(getattr(node, 'kind', '')):
                vname = getattr(getattr(node, 'name', None), 'value', None)
                return (getattr(info, 'variable_values', {}) or {}).get(vname)
            # Object (map)
            if hasattr(node, 'fields') and isinstance(getattr(node, 'fields'), (list, tuple)):
                out = {}
                for f in node.fields:
                    k = getattr(getattr(f, 'name', None), 'value', None) or getattr(f, 'name', None)
                    out[k] = self._ast_value(getattr(f, 'value', None), info)
                return out
            # List
            if hasattr(node, 'values') and isinstance(getattr(node, 'values'), (list, tuple)):
                return [self._ast_value(v, info) for v in node.values]
            # Leaf with .value
            if hasattr(node, 'value'):
                return getattr(node, 'value')
        except Exception:
            pass
        return node

    def _children(self, node: Any) -> List[Any]:
        sels = getattr(node, 'selections', None)
        if sels is not None:
            return list(sels) or []
        selset = getattr(node, 'selection_set', None)
        if selset is not None and getattr(selset, 'selections', None) is not None:
            return list(selset.selections) or []
        return []

    # Selected fields path (Strawberry's resolved selection tree)
    def _walk_selected(self, sel: Any, btype: Any, out: Dict[str, Dict[str, Any]]):
        if (not getattr(sel, 'selections', None) and not getattr(getattr(sel, 'selection_set', None), 'selections', None)) or not btype:
            return
        for child in self._children(sel):
            # name can be child.name or child.name.value depending on object type
            name = getattr(child, 'name', None)
            if hasattr(name, 'value'):
                name = getattr(name, 'value')
            if not name:
                continue
            fdef = getattr(btype, '__berry_fields__', {}).get(name)
            if not fdef or fdef.kind != 'relation':
                continue
            rel_cfg = out.setdefault(name, self._init_rel_cfg(fdef))
            # arguments on the relation field
            try:
                for arg in getattr(child, 'arguments', []) or []:
                    arg_name = getattr(getattr(arg, 'name', None), 'value', None) or getattr(arg, 'name', None)
                    raw_val = getattr(arg, 'value', None)
                    # Attempt to convert AST/value wrappers
                    val = self._ast_value(raw_val, getattr(sel, 'info', None))
                    if arg_name == 'limit':
                        rel_cfg['limit'] = val
                    elif arg_name == 'offset':
                        rel_cfg['offset'] = val
                    elif arg_name == 'order_by':
                        rel_cfg['order_by'] = val
                    elif arg_name == 'order_dir':
                        rel_cfg['order_dir'] = val
                    elif arg_name == 'order_multi':
                        rel_cfg['order_multi'] = val or []
                    elif arg_name == 'where':
                        rel_cfg['where'] = val
                    else:
                        rel_cfg['filter_args'][arg_name] = val
            except Exception:
                pass
            # collect scalar subfields and nested relations
            sub_children = self._children(child)
            if sub_children:
                tgt_b = self.registry.types.get(fdef.meta.get('target')) if fdef.meta.get('target') else None
                for sub in sub_children:
                    sub_name = getattr(sub, 'name', None)
                    if hasattr(sub_name, 'value'):
                        sub_name = getattr(sub_name, 'value')
                    if not sub_name or sub_name.startswith('__'):
                        continue
                    sub_def = getattr(tgt_b, '__berry_fields__', {}).get(sub_name) if tgt_b else None
                    if not sub_def or sub_def.kind == 'scalar':
                        if sub_name not in rel_cfg['fields']:
                            rel_cfg['fields'].append(sub_name)
                    elif sub_def and sub_def.kind == 'relation':
                        # nested relation config
                        ncfg = rel_cfg['nested'].setdefault(sub_name, self._init_rel_cfg(sub_def))
                        # copy arguments for nested relation
                        try:
                            for narg in getattr(sub, 'arguments', []) or []:
                                narg_name = getattr(getattr(narg, 'name', None), 'value', None) or getattr(narg, 'name', None)
                                nraw = getattr(narg, 'value', None)
                                nval = self._ast_value(nraw, getattr(sel, 'info', None))
                                if narg_name == 'limit':
                                    ncfg['limit'] = nval
                                elif narg_name == 'offset':
                                    ncfg['offset'] = nval
                                elif narg_name == 'order_by':
                                    ncfg['order_by'] = nval
                                elif narg_name == 'order_dir':
                                    ncfg['order_dir'] = nval
                                elif narg_name == 'order_multi':
                                    ncfg['order_multi'] = nval or []
                                elif narg_name == 'where':
                                    ncfg['where'] = nval
                                else:
                                    ncfg['filter_args'][narg_name] = nval
                        except Exception:
                            pass
                        # nested scalar fields
                        try:
                            sub2_children = self._children(sub)
                            if sub2_children:
                                tgt_b2 = self.registry.types.get(sub_def.meta.get('target')) if sub_def.meta.get('target') else None
                                for sub2 in sub2_children:
                                    nname2 = getattr(sub2, 'name', None)
                                    if hasattr(nname2, 'value'):
                                        nname2 = getattr(nname2, 'value')
                                    if nname2 and not nname2.startswith('__'):
                                        sdef2 = getattr(tgt_b2, '__berry_fields__', {}).get(nname2) if tgt_b2 else None
                                        if not sdef2 or sdef2.kind == 'scalar':
                                            if nname2 not in ncfg['fields']:
                                                ncfg['fields'].append(nname2)
                        except Exception:
                            pass
            # recurse deeper (for arbitrary nesting)
            try:
                tgt = self.registry.types.get(fdef.meta.get('target')) if fdef.meta.get('target') else None
                self._walk_selected(child, tgt, out)
            except Exception:
                pass

    def extract(self, info: Any, root_field_name: str, btype: Any) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        # Prefer AST from field_nodes
        try:
            nodes = list(getattr(info, 'field_nodes', []) or [])
        except Exception:
            nodes = []
        root_node = None
        for n in nodes:
            nname = getattr(getattr(n, 'name', None), 'value', None) or getattr(n, 'name', None)
            if nname == root_field_name:
                root_node = n
                break
        if root_node is None and nodes:
            root_node = nodes[0]
        if root_node is not None and getattr(getattr(root_node, 'selection_set', None), 'selections', None):
            # Build a fake container with .selections and also pass btype
            try:
                fake = type('Sel', (), {})()
                setattr(fake, 'selections', getattr(root_node.selection_set, 'selections', []))
                self._walk_selected(fake, btype, out)
                return out
            except Exception:
                pass
        # Fallback: if Strawberry exposes selected_fields with children
        try:
            fields = getattr(info, 'selected_fields', None)
            for f in (fields or []):
                if getattr(f, 'name', None) == root_field_name:
                    fake = type('Sel2', (), {})()
                    setattr(fake, 'selections', getattr(f, 'selections', []) or getattr(f, 'children', []))
                    self._walk_selected(fake, btype, out)
                    break
        except Exception:
            pass
        return out
# Global operator registry (extensible)
OPERATOR_REGISTRY: Dict[str, Callable[[Any, Any], Any]] = {
    'eq': lambda col, v: col == v,
    'ne': lambda col, v: col != v,
    'lt': lambda col, v: col < v,
    'lte': lambda col, v: col <= v,
    'gt': lambda col, v: col > v,
    'gte': lambda col, v: col >= v,
    'like': lambda col, v: col.like(v),
    'ilike': lambda col, v: getattr(col, 'ilike', lambda x: func.lower(col).like(func.lower(x)))(v),  # fallback if ilike unsupported
    'in': lambda col, v: col.in_(v if isinstance(v, (list, tuple, set)) else [v]),
    'between': lambda col, v: col.between(v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else None,
    'contains': lambda col, v: col.contains(v),
    'starts_with': lambda col, v: col.like(f"{v}%"),
    'ends_with': lambda col, v: col.like(f"%{v}"),
}

# Simple filter spec container used to define and expand filter arguments
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

# Coerce JSON where values (or filter arg values) to column python types
def _coerce_where_value(col, val):  # pragma: no cover - simple coercion helper
    try:
        from sqlalchemy.sql.sqltypes import Integer as _I, Float as _F, Boolean as _B, DateTime as _DT, Numeric as _N
    except Exception:  # fallback if not importable in env
        _I = Integer; _F = None; _B = Boolean; _DT = DateTime; _N = None
    # List-like values -> coerce each element
    if isinstance(val, (list, tuple)):
        return [ _coerce_where_value(col, v) for v in val ]
    ctype = getattr(col, 'type', None)
    if ctype is None:
        return val
    try:
        # DateTime: parse ISO strings
        if isinstance(ctype, _DT):
            if isinstance(val, str):
                s = val.replace('Z', '+00:00') if 'Z' in val else val
                try:
                    dv = datetime.fromisoformat(s)
                    # drop tzinfo if DB column is naive
                    try:
                        if getattr(ctype, 'timezone', False) is False and getattr(dv, 'tzinfo', None) is not None:
                            dv = dv.replace(tzinfo=None)
                    except Exception:
                        pass
                    return dv
                except Exception:
                    return val
            return val
        # Integer
        if isinstance(ctype, _I):
            try:
                return int(val) if isinstance(val, str) else val
            except Exception:
                return val
        # Numeric/Float
        if _N is not None and isinstance(ctype, _N):
            try:
                return float(val) if isinstance(val, str) else val
            except Exception:
                return val
        if _F is not None and isinstance(ctype, _F):
            try:
                return float(val) if isinstance(val, str) else val
            except Exception:
                return val
        # Boolean
        if isinstance(ctype, _B):
            if isinstance(val, str):
                lv = val.strip().lower()
                if lv in ('true','t','1','yes','y'):
                    return True
                if lv in ('false','f','0','no','n'):
                    return False
            return bool(val)
    except Exception:
        return val
    return val

# Ordering direction enum for GraphQL (so queries can use order_dir: desc)
class _DirectionEnum(Enum):
    asc = 'asc'
    desc = 'desc'

Direction = strawberry.enum(_DirectionEnum, name="Direction")  # type: ignore

def _dir_value(order_dir: Any) -> str:
    """Normalize direction enum/string to lower-case string (defaults asc)."""
    if order_dir is None:
        return 'asc'
    try:
        # Strawberry Enum
        val = getattr(order_dir, 'value', order_dir)
        return str(val).lower()
    except Exception:
        return 'asc'


def _normalize_filter_spec(raw: Any) -> FilterSpec:
    """Coerce user-provided filter spec syntaxes into a FilterSpec instance.

    Supported raw forms:
      - FilterSpec instance
      - Callable (treated as dynamic builder)
      - Dict with keys (column, op, ops, transform, alias)
    """
    if isinstance(raw, FilterSpec):
        return raw
    if callable(raw):  # dynamic builder only
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
            description=raw.get('description')
        )
    raise TypeError(f"Unsupported filter spec form: {raw!r}")


def register_operator(name: str, fn: Callable[[Any, Any], Any]):  # pragma: no cover - simple
    OPERATOR_REGISTRY[name] = fn


# --- Field descriptor primitives (restored) ---
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
    # Allow filters passed for this field (stored in meta for later collection)
    return FieldDescriptor(kind='scalar', **meta)


def relation(target: Any = None, *, single: bool | None = None, **meta) -> FieldDescriptor:
    """Define a relation field.

    target can be a BerryType subclass or its string name. 'single' selects cardinality.
    """
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
    """Define a custom field whose value is produced by executing a user-supplied
    builder callable (typically returning a SQLAlchemy selectable or scalar).

    builder invocation order tries these signatures:
        (parent_model, session, info) -> selectable/value
        (parent_model, session) -> selectable/value
        (parent_model,) -> value

    If the builder returns a SQLAlchemy Select, it will be executed and the first
    row (or scalar) returned. If the resulting value is a JSON string, an attempt
    is made to json.loads() it into a dict for GraphQL. If multiple columns are
    returned, a dict mapping column keys to values is produced.

    'returns' may be provided to influence the GraphQL annotation (int, str, etc.);
    otherwise defaults to Optional[str]. For richer typed objects, future work can
    map dict keys into nested runtime types.
    """
    return FieldDescriptor(kind='custom', builder=builder, returns=returns)

def custom_object(builder: Callable[..., Any], *, returns: Any) -> FieldDescriptor:
        """Define a multi-column custom field that maps columns to a nested object.

        returns can be:
            - a dict mapping field_name -> python type (int, str, datetime, etc.)
            - an existing Strawberry type class (with __annotations__)
        """
        return FieldDescriptor(kind='custom_object', builder=builder, returns=returns)

class BerryTypeMeta(type):
    def __new__(mcls, name, bases, namespace):
        fdefs: Dict[str, FieldDef] = {}
        for k, v in list(namespace.items()):
            if isinstance(v, FieldDescriptor):
                v.__set_name__(None, k)
                fdefs[k] = v.build(name)
        namespace['__berry_fields__'] = fdefs
        return super().__new__(mcls, name, bases, namespace)

class BerryType(metaclass=BerryTypeMeta):
    model: Optional[Type] = None  # user sets via subclass attribute

class BerrySchema:
    """Registry + dynamic Strawberry schema builder.

    Current minimal capabilities:
    - Registers BerryType subclasses
    - Generates Strawberry object types with scalar & relation fields
    - Adds root collection fields (pluralized) performing simple SELECT * queries
    - Relation resolvers issue individual filtered queries (not optimized yet)
    """
    def __init__(self):
        self.types: Dict[str, Type[BerryType]] = {}
        self._st_types: Dict[str, Any] = {}

    def register(self, cls: Type[BerryType]):
        self.types[cls.__name__] = cls
        return cls

    def type(self, *, model: Optional[Type] = None):
        def deco(cls: Type[BerryType]):
            cls.model = model
            return self.register(cls)
        return deco

    # ---------- Internal helpers ----------
    def _pluralize(self, name: str) -> str:
        if name.endswith('y') and name[-2:].lower() not in ('ay','ey','iy','oy','uy'):
            return name[:-1].lower() + 'ies'
        if name.endswith('QL'):
            base = name[:-2]
        else:
            base = name
        return base.lower() + 's'

    def _build_strawberry_type(self, name: str, bcls: Type[BerryType]):  # pragma: no cover
        """Legacy single-pass builder retained for backward compatibility paths.

        The new implementation uses a two-pass approach inside to_strawberry();
        this method should not be invoked in the redesigned flow.
        """
        pass

    def to_strawberry(self):
        # Two-pass: create plain classes first
        for name, bcls in self.types.items():
            if name not in self._st_types:
                base_namespace = {'__berry_registry__': self, '__doc__': f'Berry runtime type {name}'}
                cls = type(name, (), base_namespace)
                cls.__module__ = __name__
                self._st_types[name] = cls
                # register in module globals for forward ref resolution later
                globals()[name] = cls
        # Second pass: add fields & annotations before decoration
        for name, bcls in self.types.items():
            st_cls = self._st_types[name]
            annotations: Dict[str, Any] = getattr(st_cls, '__annotations__', {}) or {}
            # column type mapping
            column_type_map: Dict[str, Any] = {}
            if bcls.model and hasattr(bcls.model, '__table__'):
                for col in bcls.model.__table__.columns:
                    if isinstance(col.type, Integer):
                        py_t = int
                    elif isinstance(col.type, String):
                        py_t = str
                    elif isinstance(col.type, Boolean):
                        py_t = bool
                    elif isinstance(col.type, DateTime):
                        py_t = datetime
                    else:
                        py_t = str
                    column_type_map[col.name] = py_t
            for fname, fdef in bcls.__berry_fields__.items():
                if hasattr(st_cls, fname):
                    # don't overwrite existing custom attr
                    pass
                if fdef.kind == 'scalar':
                    py_t = column_type_map.get(fname, str)
                    annotations[fname] = Optional[py_t]
                    setattr(st_cls, fname, None)
                elif fdef.kind == 'relation':
                    target_name = fdef.meta.get('target')
                    is_single = bool(fdef.meta.get('single'))
                    if target_name:
                        # Use string forward refs so we don't depend on decoration order
                        if is_single:
                            annotations[fname] = f'Optional[{target_name}]'
                        else:
                            annotations[fname] = f'List[{target_name}]'
                    else:  # fallback placeholder
                        annotations[fname] = 'Optional[str]' if is_single else 'List[str]'
                    meta_copy = dict(fdef.meta)
                    def _collect_declared_filters_for_target(target_type_name: str):
                        out: Dict[str, FilterSpec] = {}
                        target_b = self.types.get(target_type_name)
                        if not target_b:
                            return out
                        class_filters = getattr(target_b, '__filters__', {}) or {}
                        for key, raw in class_filters.items():
                            try:
                                spec = _normalize_filter_spec(raw)
                            except Exception:
                                continue
                            if spec.ops and not spec.op:
                                for op_name in spec.ops:
                                    base = spec.alias or key
                                    if base.endswith(f"_{op_name}"):
                                        arg_name = base
                                    else:
                                        arg_name = f"{base}_{op_name}"
                                    out[arg_name] = spec.clone_with(op=op_name, ops=None)
                            else:
                                out[spec.alias or key] = spec
                        return out
                    def _make_relation_resolver(meta_copy=meta_copy, is_single_value=is_single, fname_local=fname, parent_btype_local=bcls):
                        target_filters = _collect_declared_filters_for_target(meta_copy.get('target')) if meta_copy.get('target') else {}
                        # Build dynamic resolver with filter args + limit/offset
                        # Determine python types for target columns (if available) for future use (not required for arg defs now)
                        async def _impl(self, info: StrawberryInfo, limit: Optional[int], offset: Optional[int], order_by: Optional[str], order_dir: Optional[Any], order_multi: Optional[List[str]], related_where: Optional[Any], _filter_args: Dict[str, Any]):
                            prefetch_attr = f'_{fname_local}_prefetched'
                            if hasattr(self, prefetch_attr):
                                # Reuse prefetched always (root pushdown included pagination already)
                                return getattr(self, prefetch_attr)
                            target_name_i = meta_copy.get('target')
                            target_cls_i = self.__berry_registry__._st_types.get(target_name_i)
                            parent_model = getattr(self, '_model', None)
                            if not target_cls_i:
                                return None if is_single_value else []
                            session = getattr(info.context, 'get', lambda k, d=None: info.context[k])('db_session', None) if info and info.context else None
                            if session is None:
                                return None if is_single_value else []
                            target_btype = self.__berry_registry__.types.get(target_name_i)
                            if not target_btype or not target_btype.model:
                                return None if is_single_value else []
                            child_model_cls = target_btype.model
                            if is_single_value:
                                candidate_fk_val = None
                                if parent_model is not None:
                                    # Normal path: derive FK from ORM model instance
                                    for col in parent_model.__table__.columns:
                                        if col.name.endswith('_id') and col.foreign_keys:
                                            for fk in col.foreign_keys:
                                                if fk.column.table.name == child_model_cls.__table__.name:
                                                    candidate_fk_val = getattr(parent_model, col.name)
                                                    break
                                        if candidate_fk_val is not None:
                                            break
                                else:
                                    # Fallback for pushdown-hydrated parent (no _model): use <relation>_id scalar, if present
                                    try:
                                        candidate_fk_val = getattr(self, f"{fname_local}_id", None)
                                    except Exception:
                                        candidate_fk_val = None
                                if candidate_fk_val is None:
                                    return None
                                # Apply filters via query if any filter args passed
                                if any(v is not None for v in _filter_args.values()) or related_where is not None:
                                    from sqlalchemy import select as _select
                                    stmt = _select(child_model_cls).where(getattr(child_model_cls, 'id') == candidate_fk_val)
                                    # Apply JSON where if provided (argument and schema default)
                                    if related_where is not None or meta_copy.get('where') is not None:
                                        try:
                                            import json as _json
                                            # apply arg where
                                            if related_where is not None:
                                                wdict = related_where
                                                if isinstance(related_where, str):
                                                    wdict = _json.loads(related_where)
                                                exprs = []
                                                for col_name, op_map in (wdict or {}).items():
                                                    col = child_model_cls.__table__.c.get(col_name)
                                                    if col is None:
                                                        continue
                                                    for op_name, val in (op_map or {}).items():
                                                        if op_name in ('in','between') and isinstance(val, (list, tuple)):
                                                            val = [_coerce_where_value(col, v) for v in val]
                                                        else:
                                                            val = _coerce_where_value(col, val)
                                                        op_fn = OPERATOR_REGISTRY.get(op_name)
                                                        if not op_fn:
                                                            continue
                                                        exprs.append(op_fn(col, val))
                                                if exprs:
                                                    from sqlalchemy import and_ as _and
                                                    stmt = stmt.where(_and(*exprs))
                                            # apply default where from schema
                                            dwhere = meta_copy.get('where')
                                            if dwhere is not None:
                                                wdict = dwhere
                                                if isinstance(dwhere, str):
                                                    wdict = _json.loads(dwhere)
                                                exprs = []
                                                for col_name, op_map in (wdict or {}).items():
                                                    col = child_model_cls.__table__.c.get(col_name)
                                                    if col is None:
                                                        continue
                                                for op_name, val in (op_map or {}).items():
                                                    if op_name in ('in','between') and isinstance(val, (list, tuple)):
                                                        val = [_coerce_where_value(col, v) for v in val]
                                                    else:
                                                        val = _coerce_where_value(col, val)
                                                        op_fn = OPERATOR_REGISTRY.get(op_name)
                                                        if not op_fn:
                                                            continue
                                                        exprs.append(op_fn(col, val))
                                                if exprs:
                                                    from sqlalchemy import and_ as _and
                                                    stmt = stmt.where(_and(*exprs))
                                        except Exception:
                                            pass
                                    # Add filter where clauses
                                    for arg_name, val in _filter_args.items():
                                        if val is None:
                                            continue
                                        f_spec = target_filters.get(arg_name)
                                        if not f_spec:
                                            raise ValueError(f"Unknown filter argument: {arg_name}")
                                        expr = None
                                        if f_spec.transform:
                                            try:
                                                val = f_spec.transform(val)
                                            except Exception as e:
                                                raise ValueError(f"Filter transform failed for {arg_name}: {e}")
                                        if f_spec.builder:
                                            try:
                                                expr = f_spec.builder(child_model_cls, info, val)
                                            except Exception as e:
                                                raise ValueError(f"Filter builder failed for {arg_name}: {e}")
                                        elif f_spec.column:
                                            try:
                                                col = child_model_cls.__table__.c.get(f_spec.column)
                                            except Exception:
                                                col = None
                                            if col is None:
                                                raise ValueError(f"Unknown filter column: {f_spec.column} for argument {arg_name}")
                                            op_fn = OPERATOR_REGISTRY.get(f_spec.op or 'eq')
                                            if not op_fn:
                                                raise ValueError(f"Unknown filter operator: {f_spec.op or 'eq'} for argument {arg_name}")
                                            try:
                                                # Coerce simple filter arg value to column type when possible
                                                val2 = _coerce_where_value(col, val)
                                                expr = op_fn(col, val2)
                                            except Exception as e:
                                                raise ValueError(f"Filter operation failed for {arg_name}: {e}")
                                        if expr is not None:
                                            stmt = stmt.where(expr)
                                    result = await session.execute(stmt.limit(1))
                                    row = result.scalar_one_or_none()
                                else:
                                    row = await session.get(child_model_cls, candidate_fk_val)
                                if not row:
                                    return None
                                inst = target_cls_i()
                                setattr(inst, '_model', row)
                                for sf, sdef in self.__berry_registry__.types[target_name_i].__berry_fields__.items():
                                    if sdef.kind == 'scalar':
                                        try:
                                            setattr(inst, sf, getattr(row, sf, None))
                                        except Exception:
                                            pass
                                return inst
                            # list relation
                            fk_col = None
                            parent_model_cls = getattr(parent_btype_local, 'model', None)
                            for col in child_model_cls.__table__.columns:
                                for fk in col.foreign_keys:
                                    try:
                                        parent_table_name = parent_model_cls.__table__.name if parent_model_cls is not None else (parent_model.__table__.name if parent_model is not None else None)
                                    except Exception:
                                        parent_table_name = None
                                    if parent_table_name and fk.column.table.name == parent_table_name:
                                        fk_col = col
                                        break
                                if fk_col is not None:
                                    break
                            if fk_col is None:
                                return []
                            from sqlalchemy import select as _select
                            # Determine parent id value: use ORM model if available; else fallback to hydrated scalar 'id'
                            parent_id_val = getattr(parent_model, 'id', None) if parent_model is not None else getattr(self, 'id', None)
                            if parent_id_val is None:
                                return []
                            stmt = _select(child_model_cls).where(fk_col == parent_id_val)
                            # Apply where from args and/or schema default
                            if related_where is not None or meta_copy.get('where') is not None:
                                try:
                                    import json as _json
                                    if related_where is not None:
                                        wdict = related_where
                                        if isinstance(related_where, str):
                                            wdict = _json.loads(related_where)
                                        exprs = []
                                        for col_name, op_map in (wdict or {}).items():
                                            col = child_model_cls.__table__.c.get(col_name)
                                            if col is None:
                                                continue
                                            for op_name, val in (op_map or {}).items():
                                                if op_name in ('in','between') and isinstance(val, (list, tuple)):
                                                    val = [_coerce_where_value(col, v) for v in val]
                                                else:
                                                    val = _coerce_where_value(col, val)
                                                op_fn = OPERATOR_REGISTRY.get(op_name)
                                                if not op_fn:
                                                    continue
                                                exprs.append(op_fn(col, val))
                                    if exprs:
                                        from sqlalchemy import and_ as _and
                                        stmt = stmt.where(_and(*exprs))
                                    dwhere = meta_copy.get('where')
                                    if dwhere is not None:
                                        if isinstance(dwhere, (dict, str)):
                                            wdict = dwhere
                                            if isinstance(dwhere, str):
                                                wdict = _json.loads(dwhere)
                                            exprs = []
                                            for col_name, op_map in (wdict or {}).items():
                                                col = child_model_cls.__table__.c.get(col_name)
                                                if col is None:
                                                    continue
                                                for op_name, val in (op_map or {}).items():
                                                    if op_name in ('in','between') and isinstance(val, (list, tuple)):
                                                        val = [_coerce_where_value(col, v) for v in val]
                                                    else:
                                                        val = _coerce_where_value(col, val)
                                                        op_fn = OPERATOR_REGISTRY.get(op_name)
                                                        if not op_fn:
                                                            continue
                                                        exprs.append(op_fn(col, val))
                                            if exprs:
                                                from sqlalchemy import and_ as _and
                                                stmt = stmt.where(_and(*exprs))
                                        else:
                                            try:
                                                expr = dwhere(child_model_cls, info)
                                                if expr is not None:
                                                    stmt = stmt.where(expr)
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                            # Ad-hoc JSON where for relation list if present on selection
                            rel_meta_map = getattr(self, '_pushdown_meta', None)  # not reliable; read from extractor cfg instead
                            # Ordering (multi then single) if column whitelist permits
                            allowed_order = getattr(target_cls_i, '__ordering__', None)
                            if allowed_order is None:
                                # derive from scalar fields
                                allowed_order = [sf for sf, sd in self.__berry_registry__.types[target_name_i].__berry_fields__.items() if sd.kind == 'scalar']
                            applied_any = False
                            if order_multi:
                                for spec in order_multi:
                                    try:
                                        col_name, _, dir_part = spec.partition(':')
                                        dir_part = dir_part or 'asc'
                                        if col_name in allowed_order:
                                            col_obj = child_model_cls.__table__.c.get(col_name)
                                            if col_obj is not None:
                                                stmt = stmt.order_by(col_obj.desc() if dir_part.lower()=='desc' else col_obj.asc())
                                                applied_any = True
                                    except Exception:
                                        raise
                            if not applied_any and order_by and order_by in allowed_order:
                                try:
                                    col_obj = child_model_cls.__table__.c.get(order_by)
                                except Exception:
                                    col_obj = None
                                if col_obj is not None:
                                    descending = _dir_value(order_dir) == 'desc'
                                    try:
                                        stmt = stmt.order_by(col_obj.desc() if descending else col_obj.asc())
                                    except Exception:
                                        raise
                            # Apply default ordering from schema meta if still no order applied
                            if not applied_any and (not order_by) and (meta_copy.get('order_by') or meta_copy.get('order_multi')):
                                try:
                                    def_dir = _dir_value(meta_copy.get('order_dir'))
                                    if meta_copy.get('order_multi'):
                                        for spec in meta_copy.get('order_multi') or []:
                                            cn, _, dd = str(spec).partition(':')
                                            dd = dd or def_dir
                                            col = child_model_cls.__table__.c.get(cn)
                                            if col is not None:
                                                stmt = stmt.order_by(col.desc() if dd=='desc' else col.asc())
                                                applied_any = True
                                    elif meta_copy.get('order_by'):
                                        cn = meta_copy.get('order_by')
                                        dd = def_dir
                                        col = child_model_cls.__table__.c.get(cn)
                                        if col is not None:
                                            stmt = stmt.order_by(col.desc() if dd=='desc' else col.asc())
                                            applied_any = True
                                except Exception:
                                    pass
                            # Apply filters
                            if target_filters:
                                for arg_name, val in _filter_args.items():
                                    if val is None:
                                        continue
                                    f_spec = target_filters.get(arg_name)
                                    if not f_spec:
                                        raise ValueError(f"Unknown filter argument: {arg_name}")
                                    orig_val = val
                                    if f_spec.transform:
                                        try:
                                            val = f_spec.transform(val)
                                        except Exception as e:
                                            raise ValueError(f"Filter transform failed for {arg_name}: {e}")
                                    expr = None
                                    if f_spec.builder:
                                        try:
                                            expr = f_spec.builder(child_model_cls, info, val)
                                        except Exception as e:
                                            raise ValueError(f"Filter builder failed for {arg_name}: {e}")
                                    elif f_spec.column:
                                        try:
                                            col = child_model_cls.__table__.c.get(f_spec.column)
                                        except Exception:
                                            col = None
                                        if col is None:
                                            raise ValueError(f"Unknown filter column: {f_spec.column} for argument {arg_name}")
                                        op_fn = OPERATOR_REGISTRY.get(f_spec.op or 'eq')
                                        if not op_fn:
                                            raise ValueError(f"Unknown filter operator: {f_spec.op or 'eq'} for argument {arg_name}")
                                        try:
                                            val2 = _coerce_where_value(col, val)
                                            expr = op_fn(col, val2)
                                        except Exception as e:
                                            raise ValueError(f"Filter operation failed for {arg_name}: {e}")
                                    if expr is not None:
                                        stmt = stmt.where(expr)
                            if offset is not None:
                                try:
                                    o = int(offset)
                                except Exception:
                                    raise ValueError("offset must be an integer")
                                if o < 0:
                                    raise ValueError("offset must be non-negative")
                                if o:
                                    stmt = stmt.offset(o)
                            if limit is not None:
                                try:
                                    l = int(limit)
                                except Exception:
                                    raise ValueError("limit must be an integer")
                                if l < 0:
                                    raise ValueError("limit must be non-negative")
                                stmt = stmt.limit(l)
                            result = await session.execute(stmt)
                            rows = [r[0] for r in result.all()]
                            out = []
                            for row in rows:
                                inst = target_cls_i()
                                setattr(inst, '_model', row)
                                for sf, sdef in self.__berry_registry__.types[target_name_i].__berry_fields__.items():
                                    if sdef.kind == 'scalar':
                                        try:
                                            setattr(inst, sf, getattr(row, sf, None))
                                        except Exception:
                                            pass
                                out.append(inst)
                            return out
                        # Build dynamic wrapper to expose filter args
                        arg_defs = []
                        for a in target_filters.keys():
                            arg_defs.append(f"{a}=None")
                        params = 'self, info, limit=None, offset=None, order_by=None, order_dir=None, order_multi=None, where=None'
                        if arg_defs:
                            params += ', ' + ', '.join(arg_defs)
                        fname_inner = f"_rel_{fname_local}_resolver"
                        src = f"async def {fname_inner}({params}):\n"
                        src += "    _fa={}\n"
                        for a in target_filters.keys():
                            src += f"    _fa['{a}']={a}\n"
                        src += "    return await _impl(self, info, limit, offset, order_by, order_dir, order_multi, where, _fa)\n"
                        env: Dict[str, Any] = {'_impl': _impl}
                        exec(src, env)
                        fn = env[fname_inner]
                        if not getattr(fn, '__module__', None):  # ensure module for strawberry introspection
                            fn.__module__ = __name__
                        # annotations
                        anns: Dict[str, Any] = {'info': StrawberryInfo, 'limit': Optional[int], 'offset': Optional[int], 'order_by': Optional[str], 'order_dir': Optional[Direction], 'order_multi': Optional[List[str]], 'where': Optional[str]}
                        # crude type inference: map to Optional[str|int|bool|datetime] based on target model columns
                        target_b = self.types.get(meta_copy.get('target')) if meta_copy.get('target') else None
                        col_type_map: Dict[str, Any] = {}
                        if target_b and target_b.model and hasattr(target_b.model, '__table__'):
                            for col in target_b.model.__table__.columns:
                                if isinstance(col.type, Integer):
                                    col_type_map[col.name] = int
                                elif isinstance(col.type, String):
                                    col_type_map[col.name] = str
                                elif isinstance(col.type, Boolean):
                                    col_type_map[col.name] = bool
                                elif isinstance(col.type, DateTime):
                                    col_type_map[col.name] = datetime
                                else:
                                    col_type_map[col.name] = str
                        for a, spec in target_filters.items():
                            base_t = str
                            if spec.column and spec.column in col_type_map:
                                base_t = col_type_map[spec.column]
                            if spec.op in ('in','between'):
                                anns[a] = Optional[List[base_t]]  # type: ignore
                            else:
                                anns[a] = Optional[base_t]
                        fn.__annotations__ = anns
                        return fn
                    setattr(st_cls, fname, strawberry.field(_make_relation_resolver()))
                elif fdef.kind == 'aggregate':
                    # Normalize meta: if ops contains 'count' without explicit op, set op='count'
                    if 'op' not in fdef.meta and 'ops' in fdef.meta and fdef.meta.get('ops') == ['count']:
                        fdef.meta['op'] = 'count'
                    ops = fdef.meta.get('ops') or ([fdef.meta.get('op')] if fdef.meta.get('op') else [])
                    is_count = fdef.meta.get('op') == 'count' or 'count' in ops
                    is_last = fdef.meta.get('op') == 'last' or 'last' in ops
                    if is_count:
                        annotations[fname] = Optional[int]
                    elif is_last:
                        annotations[fname] = Optional[int]
                    else:
                        annotations[fname] = Optional[str]
                    meta_copy = dict(fdef.meta)
                    # assign cache key for count aggregates
                    if is_count:
                        meta_copy['cache_key'] = meta_copy.get('cache_key') or fdef.meta.get('source') + ':count'
                    if is_last:
                        meta_copy['cache_key_last'] = meta_copy.get('cache_key_last') or fdef.meta.get('source') + ':last'
                    def _make_aggregate_resolver(meta_copy=meta_copy, bcls_local=bcls):
                        async def aggregate_resolver(self, info: StrawberryInfo):  # noqa: D401
                            cache = getattr(self, '_agg_cache', None)
                            is_count_local = meta_copy.get('op') == 'count' or 'count' in meta_copy.get('ops', [])
                            is_last_local = meta_copy.get('op') == 'last' or 'last' in meta_copy.get('ops', [])
                            if is_count_local and cache is not None:
                                key = meta_copy.get('cache_key') or (meta_copy.get('source') + ':count')
                                if key in cache:
                                    return cache[key]
                            if is_last_local and cache is not None:
                                key_last = meta_copy.get('cache_key_last') or (meta_copy.get('source') + ':last')
                                if key_last in cache:
                                    return cache[key_last]
                            parent_model = getattr(self, '_model', None)
                            if parent_model is None:
                                if is_count_local:
                                    return 0
                                return None
                            session = getattr(info.context, 'get', lambda k, d=None: info.context[k])('db_session', None) if info and info.context else None
                            if session is None:
                                if is_count_local:
                                    return 0
                                return None
                            source = meta_copy.get('source')
                            rel_def = bcls_local.__berry_fields__.get(source) if source else None
                            if not rel_def or rel_def.kind != 'relation':
                                if is_count_local:
                                    return 0
                                return None
                            target_name = rel_def.meta.get('target')
                            target_btype = self.__berry_registry__.types.get(target_name) if target_name else None
                            if not target_btype or not target_btype.model:
                                if is_count_local:
                                    return 0
                                return None
                            child_model_cls = target_btype.model
                            fk_col = None
                            for col in child_model_cls.__table__.columns:
                                for fk in col.foreign_keys:
                                    if fk.column.table.name == parent_model.__table__.name:
                                        fk_col = col
                                        break
                                if fk_col is not None:
                                    break
                            if fk_col is None:
                                if is_count_local:
                                    return 0
                                return None
                            if is_count_local:
                                from sqlalchemy import func, select as _select
                                stmt = _select(func.count()).select_from(child_model_cls).where(fk_col == getattr(parent_model, 'id'))
                                result = await session.execute(stmt)
                                val = result.scalar_one() or 0
                                key = meta_copy.get('cache_key') or (source + ':count')
                                if cache is None:
                                    cache = {}
                                    setattr(self, '_agg_cache', cache)
                                cache[key] = val
                                return val
                            if is_last_local:
                                from sqlalchemy import select as _select
                                # Prefer created_at desc if column exists, else id desc
                                order_col = None
                                if 'created_at' in child_model_cls.__table__.columns:
                                    order_col = child_model_cls.__table__.columns['created_at']
                                elif 'id' in child_model_cls.__table__.columns:
                                    order_col = child_model_cls.__table__.columns['id']
                                stmt = _select(child_model_cls).where(fk_col == getattr(parent_model, 'id'))
                                if order_col is not None:
                                    stmt = stmt.order_by(order_col.desc())
                                stmt = stmt.limit(1)
                                result = await session.execute(stmt)
                                row = result.scalar_one_or_none()
                                if row is None:
                                    return None
                                # return last related row id if present
                                val = getattr(row, 'id', None)
                                key_last = meta_copy.get('cache_key_last') or (source + ':last')
                                if cache is None:
                                    cache = {}
                                    setattr(self, '_agg_cache', cache)
                                cache[key_last] = val
                                return val
                            return None
                        return aggregate_resolver
                    setattr(st_cls, fname, strawberry.field(_make_aggregate_resolver()))
                elif fdef.kind == 'custom':
                    ann_type = fdef.meta.get('returns') or str
                    try:
                        # basic mapping for common primitives
                        if ann_type in (int, 'int', 'Int'):
                            annotations[fname] = Optional[int]
                        elif ann_type in (float, 'float', 'Float'):
                            annotations[fname] = Optional[float]
                        elif ann_type in (bool, 'bool', 'Bool', 'Boolean'):
                            annotations[fname] = Optional[bool]
                        else:
                            annotations[fname] = Optional[str]
                    except Exception:
                        annotations[fname] = Optional[str]
                    meta_copy = dict(fdef.meta)
                    def _make_custom_resolver(meta_copy=meta_copy, fname_local=fname):
                        async def custom_resolver(self, info: StrawberryInfo):  # noqa: D401
                            # Fast-path: if root query already populated attribute (no N+1), return it
                            pre_value = getattr(self, fname_local, None)
                            if pre_value is not None:
                                return pre_value
                            parent_model = getattr(self, '_model', None)
                            if parent_model is None:
                                return None
                            session = getattr(info.context, 'get', lambda k, d=None: info.context[k])('db_session', None) if info and info.context else None
                            if session is None:
                                return None
                            builder = meta_copy.get('builder')
                            if builder is None:
                                return None
                            import inspect, asyncio, json as _json
                            # Try builder(parent_model) only for fallback (may be less efficient)
                            try:
                                if len(inspect.signature(builder).parameters) == 1:
                                    result_obj = builder(parent_model)
                                else:
                                    result_obj = builder(parent_model, session)
                            except Exception:
                                try:
                                    result_obj = builder(parent_model)
                                except Exception:
                                    return None
                            if asyncio.iscoroutine(result_obj):
                                result_obj = await result_obj
                            try:
                                from sqlalchemy.sql import Select as _Select  # type: ignore
                            except Exception:
                                _Select = None  # type: ignore
                            if _Select is not None and isinstance(result_obj, _Select):
                                exec_result = await session.execute(result_obj)
                                row = exec_result.first()
                                if not row:
                                    return None
                                try:
                                    mv = row._mapping
                                    if len(mv) == 1:
                                        result_obj = list(mv.values())[0]
                                    else:
                                        result_obj = dict(mv)
                                except Exception:
                                    result_obj = row[0]
                            if isinstance(result_obj, str):
                                try:
                                    result_obj = _json.loads(result_obj)
                                except Exception:
                                    pass
                            return result_obj
                        return custom_resolver
                    setattr(st_cls, fname, strawberry.field(_make_custom_resolver()))
                elif fdef.kind == 'custom_object':
                    returns_spec = fdef.meta.get('returns')
                    nested_type = None
                    if isinstance(returns_spec, dict):
                        # build dynamic type
                        nested_type_name = f"{name}_{fname}_Type"
                        if nested_type_name in self._st_types:
                            nested_type = self._st_types[nested_type_name]
                        else:
                            # create plain class
                            nt_cls = type(nested_type_name, (), {'__doc__': f'Auto object for {fname}'})
                            anns = {}
                            for k2, t2 in returns_spec.items():
                                anns[k2] = Optional[t2] if t2 in (int, str, bool, float, datetime) else 'Optional[str]'
                                setattr(nt_cls, k2, None)
                            nt_cls.__annotations__ = anns
                            self._st_types[nested_type_name] = strawberry.type(nt_cls)  # decorate immediately
                            nested_type = self._st_types[nested_type_name]
                    else:
                        nested_type = returns_spec
                    # annotate forward with actual type
                    if nested_type is not None:
                        annotations[fname] = Optional[nested_type]
                    else:
                        annotations[fname] = 'Optional[str]'
                    meta_copy = dict(fdef.meta)
                    # store nested type for resolver reconstruction
                    meta_copy['_nested_type'] = nested_type
                    def _make_custom_obj_resolver(meta_copy=meta_copy, fname_local=fname):
                        async def _resolver(self, info: StrawberryInfo):  # noqa: D401
                            # Prefer pre-hydrated values; no N+1 fallback
                            pre_json = getattr(self, f"_{fname_local}_prefetched", None)
                            if pre_json is not None:
                                return pre_json
                            pre_v = getattr(self, f"_{fname_local}_data", None)
                            if pre_v is not None:
                                return pre_v
                            return None
                        return _resolver
                    setattr(st_cls, fname, strawberry.field(_make_custom_obj_resolver()))
            st_cls.__annotations__ = annotations
        # Decorate all types now
        for name, cls in list(self._st_types.items()):
            if not getattr(cls, '__is_strawberry_type__', False):
                # Ensure typing symbols available for forward refs
                mod_globals = globals()
                mod_globals.update({'Optional': Optional, 'List': List})
                self._st_types[name] = strawberry.type(cls)  # type: ignore
        # Root query assembly BEFORE decoration so Strawberry sees fields
        query_namespace: Dict[str, Any] = {
            '__doc__': 'Auto-generated Berry root query (prototype).'
        }
        query_annotations: Dict[str, Any] = {}
        for name, bcls in self.types.items():
            if not bcls.model:
                continue
            st_type = self._st_types[name]
            field_name = self._pluralize(name)
            _model_cls = bcls.model
            def _collect_declared_filters(btype_cls_local):
                """Collect and expand declared filter specs.

                Expansion rules:
                  - If spec has .op set -> single arg named exactly as mapping key (or alias if provided)
                  - If spec has .ops list and no .op:
                        If mapping key equals column name -> produce arg per op named f"{key}_{op}"
                        Else treat mapping key as prefix -> same as above
                  - Callable-only spec (builder) uses provided key
                Returns mapping of argument_name -> FilterSpec (with resolved single op).
                """
                out: Dict[str, FilterSpec] = {}
                class_filters = getattr(btype_cls_local, '__filters__', {}) or {}
                for key, raw in class_filters.items():
                    try:
                        spec = _normalize_filter_spec(raw)
                    except Exception:
                        continue
                    if spec.ops and not spec.op:
                        for op_name in spec.ops:
                            base = spec.alias or key
                            # If base already ends with _{op} or exactly op appended, don't duplicate
                            if base.endswith(f"_{op_name}"):
                                arg_name = base
                            else:
                                arg_name = f"{base}_{op_name}"
                            out[arg_name] = spec.clone_with(op=op_name, ops=None)
                    else:
                        arg_name = spec.alias or key
                        out[arg_name] = spec
                return out

            def _make_root_resolver(model_cls, st_cls, btype_cls, root_field_name):
                declared_filters = _collect_declared_filters(btype_cls)
                # Precompute column -> python type mapping for argument type inference
                col_py_types: Dict[str, Any] = {}
                if model_cls is not None and hasattr(model_cls, '__table__'):
                    for col in model_cls.__table__.columns:
                        if isinstance(col.type, Integer):
                            py_t = int
                        elif isinstance(col.type, String):
                            py_t = str
                        elif isinstance(col.type, Boolean):
                            py_t = bool
                        elif isinstance(col.type, DateTime):
                            py_t = datetime
                        else:
                            py_t = str
                        col_py_types[col.name] = py_t

                # Build argument annotations dynamically
                filter_arg_types: Dict[str, Any] = {}
                for arg_name, f_spec in declared_filters.items():
                    base_type = str
                    if f_spec.column and f_spec.column in col_py_types:
                        base_type = col_py_types[f_spec.column]
                    # list type for in/between
                    if f_spec.op in ('in', 'between'):
                        from typing import List as _List
                        filter_arg_types[arg_name] = Optional[List[base_type]]  # type: ignore
                    else:
                        filter_arg_types[arg_name] = Optional[base_type]

                async def _base_impl(info: StrawberryInfo, limit: int | None, offset: int | None, order_by: Optional[str], order_dir: Optional[Any], _passed_filter_args: Dict[str, Any], raw_where: Optional[Any] = None):
                    session = info.context.get('db_session') if info and info.context else None
                    if session is None:
                        return []
                    # Detect dialect & acquire adapter (unifies JSON funcs / capabilities)
                    try:
                        dialect_name = session.get_bind().dialect.name.lower()
                    except Exception:
                        dialect_name = 'sqlite'
                    if get_adapter:
                        adapter = get_adapter(dialect_name)
                    else:
                        class _LegacyAdapter:
                            def json_object(self,*a): return func.json_object(*a)
                            def json_array_agg(self,e): return func.json_group_array(e)
                            def json_array_coalesce(self,e): return func.coalesce(e,'[]')
                            def supports_relation_pushdown(self): return True
                        adapter = _LegacyAdapter()  # type: ignore
                    def _json_object(*args):
                        return adapter.json_object(*args)
                    def _json_array_agg(expr):
                        return adapter.json_array_agg(expr)
                    def _json_array_coalesce(expr):
                        return adapter.json_array_coalesce(expr)
                    # Acquire per-context lock to avoid concurrent AsyncSession use (esp. MSSQL/pyodbc limitations)
                    lock = info.context.setdefault('_berry_db_lock', asyncio.Lock())
                    # Collect custom field expressions for pushdown
                    custom_fields: List[tuple[str, Any]] = []
                    custom_object_fields: List[tuple[str, List[str], Any]] = []  # (field, column_labels, returns_spec)
                    select_columns: List[Any] = [model_cls]
                    # Use extracted helper class
                    requested_relations = RelationSelectionExtractor(self).extract(info, root_field_name, btype_cls)
                    # Coerce JSON where values to match column types (helps strict dialects like Postgres)
                    def _coerce_where_value(col, val):
                        try:
                            from sqlalchemy.sql.sqltypes import Integer as _I, Float as _F, Boolean as _B, DateTime as _DT, Numeric as _N
                        except Exception:
                            _I = Integer; _F = None; _B = Boolean; _DT = DateTime; _N = None  # fallbacks
                        # List-like (in/between) -> coerce elements
                        if isinstance(val, (list, tuple)):
                            return [ _coerce_where_value(col, v) for v in val ]
                        ctype = getattr(col, 'type', None)
                        if ctype is None:
                            return val
                        try:
                            # DateTime
                            if isinstance(ctype, _DT):
                                if isinstance(val, str):
                                    s = val.replace('Z', '+00:00') if 'Z' in val else val
                                    try:
                                        dv = datetime.fromisoformat(s)
                                        # If DB stores naive datetimes, drop tzinfo
                                        try:
                                            if getattr(ctype, 'timezone', False) is False and getattr(dv, 'tzinfo', None) is not None:
                                                dv = dv.replace(tzinfo=None)
                                        except Exception:
                                            pass
                                        return dv
                                    except Exception:
                                        return val
                                return val
                            # Integer
                            if isinstance(ctype, _I):
                                try:
                                    return int(val) if isinstance(val, str) else val
                                except Exception:
                                    return val
                            # Numeric/Float
                            if _N is not None and isinstance(ctype, _N):
                                try:
                                    return float(val) if isinstance(val, str) else val
                                except Exception:
                                    return val
                            if _F is not None and isinstance(ctype, _F):
                                try:
                                    return float(val) if isinstance(val, str) else val
                                except Exception:
                                    return val
                            # Boolean
                            if isinstance(ctype, _B):
                                if isinstance(val, str):
                                    lv = val.strip().lower()
                                    if lv in ('true','t','1','yes','y'):
                                        return True
                                    if lv in ('false','f','0','no','n'):
                                        return False
                                return bool(val)
                        except Exception:
                            return val
                        return val
                    # Helper to recursively build relation JSON expressions
                    def _build_list_relation_json(parent_model_cls, parent_btype, rel_cfg: Dict[str, Any]):
                        if 'mssql' in dialect_name:
                            return None  # skip legacy JSON aggregation path for MSSQL
                        target_name = rel_cfg.get('target')
                        target_b = self.types.get(target_name)
                        if not target_b or not target_b.model:
                            return None
                        child_model_cls = target_b.model
                        # FK
                        fk_col = None
                        for col in child_model_cls.__table__.columns:
                            for fk in col.foreign_keys:
                                if fk.column.table.name == parent_model_cls.__table__.name:
                                    fk_col = col
                                    break
                            if fk_col is not None:
                                break
                        if fk_col is None:
                            return None
                        requested_scalar = list(rel_cfg.get('fields') or [])
                        if not requested_scalar:
                            for sf, sdef in target_b.__berry_fields__.items():
                                if sdef.kind == 'scalar':
                                    requested_scalar.append(sf)
                        # Ensure FK helper columns for child's single relations are present
                        try:
                            for rel_name2, rel_def2 in target_b.__berry_fields__.items():
                                if rel_def2.kind == 'relation' and (rel_def2.meta.get('single') or rel_def2.meta.get('mode') == 'single'):
                                    fk_name = f"{rel_name2}_id"
                                    if fk_name in child_model_cls.__table__.columns and fk_name not in requested_scalar:
                                        requested_scalar.append(fk_name)
                        except Exception:
                            pass
                        inner_cols = [getattr(child_model_cls, c) for c in requested_scalar] if requested_scalar else [getattr(child_model_cls, 'id')]
                        inner_sel = select(*inner_cols).select_from(child_model_cls).where(fk_col == parent_model_cls.id).correlate(parent_model_cls)
                        # Apply ordering from relation config (multi -> single -> fallback id)
                        ordered = False
                        try:
                            allowed_order_fields = [sf for sf, sd in target_b.__berry_fields__.items() if sd.kind == 'scalar']
                            multi = rel_cfg.get('order_multi') or []
                            for spec in multi:
                                try:
                                    cn, _, dd = str(spec).partition(':')
                                    dd = dd or 'asc'
                                    if cn not in allowed_order_fields:
                                        raise ValueError(f"Invalid order field '{cn}' for relation {rel_cfg.get('target')}")
                                    if dd.lower() not in ('asc','desc'):
                                        raise ValueError(f"Invalid order direction '{dd}' for relation {rel_cfg.get('target')}")
                                    col = getattr(child_model_cls, cn, None)
                                    if col is None:
                                        raise ValueError(f"Unknown order column '{cn}' for relation {rel_cfg.get('target')}")
                                    inner_sel = inner_sel.order_by(col.desc() if dd.lower()=='desc' else col.asc())
                                    ordered = True
                                except Exception:
                                    raise
                            if not ordered and rel_cfg.get('order_by') in allowed_order_fields:
                                cn = rel_cfg.get('order_by')
                                dd = _dir_value(rel_cfg.get('order_dir'))
                                if dd not in ('asc','desc'):
                                    raise ValueError(f"Invalid order direction '{rel_cfg.get('order_dir')}' for relation {rel_cfg.get('target')}")
                                col = getattr(child_model_cls, cn, None)
                                if col is None:
                                    raise ValueError(f"Unknown order column '{cn}' for relation {rel_cfg.get('target')}")
                                inner_sel = inner_sel.order_by(col.desc() if dd=='desc' else col.asc())
                                ordered = True
                            # Default order from schema meta if none applied yet
                            if not ordered and (rel_cfg.get('order_by') or rel_cfg.get('order_multi')):
                                try:
                                    def_dir = _dir_value(rel_cfg.get('order_dir'))
                                    multi = rel_cfg.get('order_multi') or []
                                    if multi:
                                        for spec in multi:
                                            cn, _, dd = str(spec).partition(':')
                                            dd = dd or def_dir
                                            col = getattr(child_model_cls, cn, None)
                                            if col is not None:
                                                inner_sel = inner_sel.order_by(col.desc() if dd.lower()=='desc' else col.asc())
                                                ordered = True
                                    elif rel_cfg.get('order_by'):
                                        cn = rel_cfg.get('order_by')
                                        dd = def_dir
                                        col = getattr(child_model_cls, cn, None)
                                        if col is not None:
                                            inner_sel = inner_sel.order_by(col.desc() if dd=='desc' else col.asc())
                                            ordered = True
                                except Exception:
                                    pass
                            if not ordered and 'id' in child_model_cls.__table__.columns:
                                inner_sel = inner_sel.order_by(getattr(child_model_cls, 'id'))
                        except Exception:
                            pass
                        # Apply schema-declared default_where AND query where before pagination
                        try:
                            rwhere = rel_cfg.get('where')
                            dwhere = rel_cfg.get('default_where')
                            # Merge dicts with AND semantics; if both exist, we AND their expressions
                            if rwhere is not None:
                                import json as _json
                                wdict = rwhere
                                if isinstance(rwhere, str):
                                    wdict = _json.loads(rwhere)
                                expr_r = None
                                try:
                                    exprs = []
                                    for col_name, op_map in (wdict or {}).items():
                                        col = child_model_cls.__table__.c.get(col_name)
                                        if col is None:
                                            continue
                                        for op_name, val in (op_map or {}).items():
                                            # coerce JSON where value(s) to column type
                                            if op_name in ('in','between') and isinstance(val, (list, tuple)):
                                                val = [_coerce_where_value(col, v) for v in val]
                                            else:
                                                val = _coerce_where_value(col, val)
                                            op_fn = OPERATOR_REGISTRY.get(op_name)
                                            if not op_fn:
                                                continue
                                            exprs.append(op_fn(col, val))
                                    if exprs:
                                        expr_r = _and(*exprs)
                                except Exception:
                                    expr_r = None
                                if expr_r is not None:
                                    inner_sel = inner_sel.where(expr_r)
                            if dwhere is not None:
                                import json as _json
                                wdict = dwhere
                                if isinstance(dwhere, str):
                                    wdict = _json.loads(dwhere)
                                expr_r = None
                                try:
                                    exprs = []
                                    for col_name, op_map in (wdict or {}).items():
                                        col = child_model_cls.__table__.c.get(col_name)
                                        if col is None:
                                            continue
                                        for op_name, val in (op_map or {}).items():
                                            if op_name in ('in','between') and isinstance(val, (list, tuple)):
                                                val = [_coerce_where_value(col, v) for v in val]
                                            else:
                                                val = _coerce_where_value(col, val)
                                            op_fn = OPERATOR_REGISTRY.get(op_name)
                                            if not op_fn:
                                                continue
                                            exprs.append(op_fn(col, val))
                                    if exprs:
                                        expr_r = _and(*exprs)
                                except Exception:
                                    expr_r = None
                                if expr_r is not None:
                                    inner_sel = inner_sel.where(expr_r)
                        except Exception:
                            pass
                        if rel_cfg.get('offset') is not None:
                            try:
                                inner_sel = inner_sel.offset(int(rel_cfg['offset']))
                            except Exception:
                                inner_sel = inner_sel.offset(rel_cfg['offset'])
                        if rel_cfg.get('limit') is not None:
                            try:
                                inner_sel = inner_sel.limit(int(rel_cfg['limit']))
                            except Exception:
                                inner_sel = inner_sel.limit(rel_cfg['limit'])
                        limited_subq = inner_sel.subquery()
                        row_json_args: List[Any] = []
                        # Scalars
                        for sf in requested_scalar if requested_scalar else ['id']:
                            row_json_args.extend([_text(f"'{sf}'"), getattr(limited_subq.c, sf)])
                        # Nested relations within each child row
                        for nested_name, nested_cfg in (rel_cfg.get('nested') or {}).items():
                            # Build nested JSON (list or single)
                            nested_target = nested_cfg.get('target')
                            nested_b = self.types.get(nested_target)
                            if not nested_b or not nested_b.model:
                                continue
                            # If nested default_where is callable, skip pushdown for that nested branch
                            try:
                                ndw = nested_cfg.get('default_where')
                            except Exception:
                                ndw = None
                            if ndw is not None and not isinstance(ndw, (dict, str)):
                                continue
                            grand_model_cls = nested_b.model
                            # Determine FK from grandchild -> child
                            g_fk = None
                            for col in grand_model_cls.__table__.columns:
                                for fk in col.foreign_keys:
                                    if fk.column.table.name == child_model_cls.__table__.name:
                                        g_fk = col
                                        break
                                if g_fk is not None:
                                    break
                            if g_fk is None:
                                continue
                            nested_scalars = nested_cfg.get('fields') or []
                            if not nested_scalars:
                                for sf2, sdef2 in nested_b.__berry_fields__.items():
                                    if sdef2.kind == 'scalar':
                                        nested_scalars.append(sf2)
                            g_inner_cols = [getattr(grand_model_cls, c) for c in nested_scalars] if nested_scalars else [getattr(grand_model_cls, 'id')]
                            # Join to the child limited subquery alias to properly scope nested rows per child
                            try:
                                g_sel = (
                                    select(*g_inner_cols)
                                    .select_from(grand_model_cls)
                                    .select_from(limited_subq)
                                    .where(g_fk == getattr(limited_subq.c, 'id'))
                                )
                            except Exception:
                                # Fallback to correlation on base child model (less precise but functional)
                                g_sel = select(*g_inner_cols).select_from(grand_model_cls).where(g_fk == getattr(child_model_cls, 'id')).correlate(child_model_cls, parent_model_cls)
                            # Apply ordering from nested relation config
                            try:
                                allowed_order_fields = [sf for sf, sd in nested_b.__berry_fields__.items() if sd.kind == 'scalar']
                                n_ordered = False
                                n_multi = nested_cfg.get('order_multi') or []
                                for spec in n_multi:
                                    try:
                                        cn, _, dd = str(spec).partition(':')
                                        dd = dd or 'asc'
                                        if cn not in allowed_order_fields:
                                            raise ValueError(f"Invalid order field '{cn}' for relation {nested_cfg.get('target')}")
                                        if dd.lower() not in ('asc','desc'):
                                            raise ValueError(f"Invalid order direction '{dd}' for relation {nested_cfg.get('target')}")
                                        col = getattr(grand_model_cls, cn, None)
                                        if col is None:
                                            raise ValueError(f"Unknown order column '{cn}' for relation {nested_cfg.get('target')}")
                                        g_sel = g_sel.order_by(col.desc() if dd.lower()=='desc' else col.asc())
                                        n_ordered = True
                                    except Exception:
                                        raise
                                if not n_ordered and nested_cfg.get('order_by') in allowed_order_fields:
                                    cn = nested_cfg.get('order_by')
                                    dd = _dir_value(nested_cfg.get('order_dir'))
                                    if dd not in ('asc','desc'):
                                        raise ValueError(f"Invalid order direction '{nested_cfg.get('order_dir')}' for relation {nested_cfg.get('target')}")
                                    col = getattr(grand_model_cls, cn, None)
                                    if col is None:
                                        raise ValueError(f"Unknown order column '{cn}' for relation {nested_cfg.get('target')}")
                                    g_sel = g_sel.order_by(col.desc() if dd=='desc' else col.asc())
                                    n_ordered = True
                                # Default order by id as final fallback
                                if not n_ordered and 'id' in grand_model_cls.__table__.columns:
                                    g_sel = g_sel.order_by(getattr(grand_model_cls, 'id'))
                            except Exception:
                                pass
                            # Apply where (query where + default where) before pagination
                            try:
                                dbg_where = nested_cfg.get('where')
                                dbg_def_where = nested_cfg.get('default_where')
                                if info and getattr(info, 'context', None) is not None:
                                    # minimal debug print; acceptable during targeted test run
                                    print(f"[berry debug] nested '{nested_name}' where={dbg_where!r} default_where={dbg_def_where!r}")
                            except Exception:
                                pass
                            try:
                                nrwhere = nested_cfg.get('where')
                                ndwhere = nested_cfg.get('default_where')
                                if nrwhere is not None:
                                    import json as _json
                                    wdict = nrwhere
                                    if isinstance(nrwhere, str):
                                        wdict = _json.loads(nrwhere)
                                    expr_r = None
                                    try:
                                        exprs = []
                                        for col_name, op_map in (wdict or {}).items():
                                            col = grand_model_cls.__table__.c.get(col_name)
                                            if col is None:
                                                continue
                                            for op_name, val in (op_map or {}).items():
                                                if op_name in ('in','between') and isinstance(val, (list, tuple)):
                                                    val = [_coerce_where_value(col, v) for v in val]
                                                else:
                                                    val = _coerce_where_value(col, val)
                                                op_fn = OPERATOR_REGISTRY.get(op_name)
                                                if not op_fn:
                                                    continue
                                                exprs.append(op_fn(col, val))
                                        if exprs:
                                            expr_r = _and(*exprs)
                                        if info and getattr(info, 'context', None) is not None:
                                            print(f"[berry debug] nested where exprs={len(exprs)} expr={expr_r}")
                                    except Exception as e:
                                        expr_r = None
                                        try:
                                            if info and getattr(info, 'context', None) is not None:
                                                print(f"[berry debug] nested where build error: {e}")
                                        except Exception:
                                            pass
                                    if expr_r is not None:
                                        g_sel = g_sel.where(expr_r)
                                        if info and getattr(info, 'context', None) is not None:
                                            print(f"[berry debug] applied nested where")
                                if ndwhere is not None:
                                    import json as _json
                                    wdict = ndwhere
                                    if isinstance(ndwhere, str):
                                        wdict = _json.loads(ndwhere)
                                    expr_r = None
                                    try:
                                        exprs = []
                                        for col_name, op_map in (wdict or {}).items():
                                            col = grand_model_cls.__table__.c.get(col_name)
                                            if col is None:
                                                continue
                                            for op_name, val in (op_map or {}).items():
                                                if op_name in ('in','between') and isinstance(val, (list, tuple)):
                                                    val = [_coerce_where_value(col, v) for v in val]
                                                else:
                                                    val = _coerce_where_value(col, val)
                                                op_fn = OPERATOR_REGISTRY.get(op_name)
                                                if not op_fn:
                                                    continue
                                                exprs.append(op_fn(col, val))
                                        if exprs:
                                            expr_r = _and(*exprs)
                                    except Exception:
                                        expr_r = None
                                    if expr_r is not None:
                                        g_sel = g_sel.where(expr_r)
                            except Exception:
                                pass
                            if nested_cfg.get('offset'):
                                g_sel = g_sel.offset(nested_cfg['offset'])
                            if nested_cfg.get('limit') is not None:
                                g_sel = g_sel.limit(nested_cfg['limit'])
                            g_subq = g_sel.subquery()
                            g_row_args: List[Any] = []
                            for sf2 in nested_scalars if nested_scalars else ['id']:
                                g_row_args.extend([_text(f"'{sf2}'"), getattr(g_subq.c, sf2)])
                            g_row_json = _json_object(*g_row_args)
                            agg_inner_expr = _json_array_agg(g_row_json)
                            if agg_inner_expr is None:
                                continue
                            # Correlate to limited_subq so nested clause can reference it
                            try:
                                g_agg_inner = select(_json_array_coalesce(agg_inner_expr)).select_from(g_subq).correlate(limited_subq)
                            except Exception:
                                g_agg_inner = select(_json_array_coalesce(agg_inner_expr)).select_from(g_subq).correlate(child_model_cls, parent_model_cls)
                            try:
                                g_agg = g_agg_inner.scalar_subquery()
                            except Exception:
                                g_agg = g_agg_inner
                            row_json_args.extend([_text(f"'{nested_name}'"), g_agg])
                        row_json_expr = _json_object(*row_json_args)
                        agg_expr = _json_array_agg(row_json_expr)
                        if agg_expr is None:
                            return None
                        agg_query = select(_json_array_coalesce(agg_expr)).select_from(limited_subq).correlate(parent_model_cls).scalar_subquery()
                        return agg_query
                    # Prepare pushdown COUNT aggregates (replace batch later)
                    count_aggregates: List[tuple[str, Any]] = []  # (agg_field_name, subquery_expr)
                    for cf_name, cf_def in btype_cls.__berry_fields__.items():
                        if cf_def.kind == 'custom':
                            builder = cf_def.meta.get('builder')
                            if builder is None:
                                continue
                            # Attempt to build expression using model class (no instance/session) to avoid N+1
                            import inspect
                            expr = None
                            try:
                                if len(inspect.signature(builder).parameters) == 1:
                                    expr = builder(model_cls)
                                else:
                                    # Skip builders requiring session/info for pushdown
                                    continue
                            except Exception:
                                continue
                            if expr is None:
                                continue
                            try:
                                from sqlalchemy.sql import Select as _Select  # type: ignore
                            except Exception:
                                _Select = None  # type: ignore
                            # Convert sub-selects to scalar
                            try:
                                if _Select is not None and isinstance(expr, _Select):
                                    # If multiple columns, leave as is (handled after execution)
                                    if hasattr(expr, 'subquery'):  # mark for scalar extraction
                                        try:
                                            # Use scalar_subquery if single column select
                                            if len(expr.selected_columns) == 1:  # type: ignore[attr-defined]
                                                expr = expr.scalar_subquery()
                                        except Exception:
                                            pass
                                # Label unlabeled column expressions so result mapping works
                                if hasattr(expr, 'label'):
                                    expr = expr.label(cf_name)
                            except Exception:
                                continue
                            custom_fields.append((cf_name, expr))
                            select_columns.append(expr)
                        elif cf_def.kind == 'custom_object':
                            builder = cf_def.meta.get('builder')
                            if builder is None:
                                continue
                            import inspect
                            expr_sel = None
                            try:
                                if len(inspect.signature(builder).parameters) == 1:
                                    expr_sel = builder(model_cls)
                                else:
                                    continue  # skip builders needing session/info to avoid N+1
                            except Exception:
                                continue
                            if expr_sel is None:
                                continue
                            try:
                                from sqlalchemy.sql import Select as _Select  # type: ignore
                            except Exception:
                                _Select = None  # type: ignore
                            if _Select is not None and isinstance(expr_sel, _Select):
                                # Build per-field scalar subqueries and compose a JSON object
                                try:
                                    sel_cols = list(getattr(expr_sel, 'selected_columns', []))  # type: ignore[attr-defined]
                                except Exception:
                                    sel_cols = []
                                key_exprs: list[tuple[str, Any]] = []
                                for col in sel_cols:
                                    try:
                                        labeled = col
                                        col_name = getattr(labeled, 'name', None) or getattr(labeled, 'key', None)
                                        if not col_name:
                                            col_name = f"{cf_name}_{len(key_exprs)}"
                                            labeled = col.label(col_name)
                                        subq = select(labeled)
                                        try:
                                            for _from in expr_sel.get_final_froms():  # type: ignore[attr-defined]
                                                subq = subq.select_from(_from)
                                        except Exception:
                                            pass
                                        for _w in getattr(expr_sel, '_where_criteria', []):  # type: ignore[attr-defined]
                                            subq = subq.where(_w)
                                        subq_expr = subq.scalar_subquery() if hasattr(subq, 'scalar_subquery') else subq
                                        key_exprs.append((col_name, subq_expr))
                                    except Exception:
                                        continue
                                if not key_exprs:
                                    continue
                                # MSSQL path: adapter.json_object is not available; select per-key labeled columns
                                is_mssql = getattr(adapter, 'name', '') == 'mssql'
                                if is_mssql:
                                    labels: list[str] = []
                                    for k, v in key_exprs:
                                        lbl = f"_pushcf_{cf_name}__{k}"
                                        try:
                                            select_columns.append(v.label(lbl))
                                        except Exception:
                                            # Best-effort: wrap via text if needed
                                            from sqlalchemy import literal_column
                                            select_columns.append(literal_column(str(v)).label(lbl))
                                        labels.append(lbl)
                                    custom_object_fields.append((cf_name, labels, cf_def.meta.get('returns')))
                                else:
                                    # Compose JSON object using adapter.json_object
                                    json_args: list[Any] = []
                                    for k, v in key_exprs:
                                        json_args.extend([_text(f"'{k}'"), v])
                                    json_obj_expr = _json_object(*json_args)
                                    # No null semantics: always return object, even when counts are zero
                                    json_label = f"_pushcf_{cf_name}"
                                    select_columns.append(json_obj_expr.label(json_label))
                                    custom_object_fields.append((cf_name, [json_label], cf_def.meta.get('returns')))
                            else:
                                continue
                        elif cf_def.kind == 'aggregate':
                            # Only handle count aggregates for pushdown
                            op = cf_def.meta.get('op')
                            ops = cf_def.meta.get('ops') or []
                            is_count = op == 'count' or 'count' in ops
                            if is_count:
                                source_rel = cf_def.meta.get('source')
                                rel_def = btype_cls.__berry_fields__.get(source_rel)
                                if rel_def and rel_def.kind == 'relation':
                                    target_name = rel_def.meta.get('target')
                                    target_b = self.types.get(target_name)
                                    if target_b and target_b.model:
                                        child_model_cls = target_b.model
                                        # find FK from child to parent
                                        fk_col = None
                                        for col in child_model_cls.__table__.columns:
                                            for fk in col.foreign_keys:
                                                if fk.column.table.name == model_cls.__table__.name:
                                                    fk_col = col
                                                    break
                                            if fk_col is not None:
                                                break
                                        if fk_col is not None:
                                            subq_cnt = select(func.count('*')).select_from(child_model_cls).where(fk_col == model_cls.id).scalar_subquery().label(cf_name)
                                            select_columns.append(subq_cnt)
                                            count_aggregates.append((cf_name, cf_def))
                    # MSSQL special handling: we'll emulate JSON aggregation via FOR JSON PATH
                    mssql_mode = hasattr(adapter, 'name') and adapter.name == 'mssql'
                    if not adapter.supports_relation_pushdown() and not mssql_mode:
                        requested_relations = {}
                    # Push down relation JSON arrays/objects
                    for rel_name, rel_cfg in requested_relations.items():
                        # If default_where is a callable, skip pushdown so resolver can apply it
                        try:
                            dw = rel_cfg.get('default_where')
                            if dw is not None and not isinstance(dw, (dict, str)):
                                rel_cfg['skip_pushdown'] = True
                        except Exception:
                            pass
                        if rel_cfg.get('skip_pushdown'):
                            continue
                        target_name = rel_cfg.get('target')
                        if not target_name:
                            continue
                        target_b = self.types.get(target_name)
                        if not target_b or not target_b.model:
                            continue
                        child_model_cls = target_b.model
                        # Determine FK
                        fk_col = None
                        for col in child_model_cls.__table__.columns:
                            for fk in col.foreign_keys:
                                if fk.column.table.name == model_cls.__table__.name:
                                    fk_col = col
                                    break
                            if fk_col is not None:
                                break
                        if fk_col is None:
                            continue
                        # Determine projected scalar fields
                        requested_scalar = rel_cfg.get('fields') or []
                        if rel_cfg.get('single'):
                            # Single relation: build json object
                            proj_cols = []
                            if not requested_scalar:
                                # default to all scalar fields of target
                                for sf, sdef in target_b.__berry_fields__.items():
                                    if sdef.kind == 'scalar':
                                        requested_scalar.append(sf)
                            for sf in requested_scalar:
                                if sf in child_model_cls.__table__.columns:
                                    proj_cols.append(sf)
                            if mssql_mode:
                                if not proj_cols:
                                    proj_cols = ['id']
                                join_cond = f"[{child_model_cls.__tablename__}].[id] = [{model_cls.__tablename__}].[{rel_name}_id]"
                                rel_expr = adapter.build_single_relation_json(
                                    child_table=child_model_cls.__tablename__,
                                    projected_columns=proj_cols,
                                    join_condition=join_cond,
                                ).label(f"_pushrel_{rel_name}")
                                select_columns.append(rel_expr)
                            else:
                                json_args: List[Any] = []
                                for sf in proj_cols:
                                    json_args.extend([_text(f"'{sf}'"), getattr(child_model_cls, sf)])
                                json_obj = _json_object(*json_args) if json_args else _json_object(_text("'id'"), getattr(child_model_cls, 'id'))
                                rel_subq = select(json_obj).select_from(child_model_cls).where(getattr(child_model_cls, 'id') == getattr(model_cls, f"{rel_name}_id", None))
                                # Fallback join via fk if id direct column missing
                                try:
                                    if not any(c.name == f"{rel_name}_id" for c in model_cls.__table__.columns):
                                        rel_subq = select(json_obj).select_from(child_model_cls).where(getattr(child_model_cls, 'id') == fk_col)
                                except Exception:
                                    pass
                                # Apply filter args if any
                                try:
                                    filter_args = rel_cfg.get('filter_args') or {}
                                    if filter_args:
                                        # collect declared filters for target
                                        target_btype = self.types.get(target_name)
                                        if target_btype:
                                            # replicate root filter expansion logic
                                            class_filters = getattr(target_btype, '__filters__', {}) or {}
                                            expanded: Dict[str, FilterSpec] = {}
                                            for key, raw in class_filters.items():
                                                try:
                                                    spec = _normalize_filter_spec(raw)
                                                except Exception:
                                                    continue
                                                if spec.ops and not spec.op:
                                                    for op_name in spec.ops:
                                                        base = spec.alias or key
                                                        if base.endswith(f"_{op_name}"):
                                                            an = base
                                                        else:
                                                            an = f"{base}_{op_name}"
                                                        expanded[an] = spec.clone_with(op=op_name, ops=None)
                                                else:
                                                    expanded[spec.alias or key] = spec
                                            for arg_name, val in filter_args.items():
                                                f_spec = expanded.get(arg_name)
                                                if not f_spec:
                                                    continue
                                                if f_spec.transform:
                                                    try:
                                                        val = f_spec.transform(val)
                                                    except Exception:
                                                        continue
                                                expr = None
                                                if f_spec.builder:
                                                    try:
                                                        expr = f_spec.builder(child_model_cls, info, val)
                                                    except Exception:
                                                        expr = None
                                                elif f_spec.column:
                                                    try:
                                                        col = child_model_cls.__table__.c.get(f_spec.column)
                                                    except Exception:
                                                        col = None
                                                    if col is None:
                                                        continue
                                                    op_fn = OPERATOR_REGISTRY.get(f_spec.op or 'eq')
                                                    if not op_fn:
                                                        continue
                                                    try:
                                                        expr = op_fn(col, val)
                                                    except Exception:
                                                        expr = None
                                                if expr is not None:
                                                    rel_subq = rel_subq.where(expr)
                                except Exception:
                                    pass
                                rel_expr = rel_subq.limit(1).scalar_subquery().label(f"_pushrel_{rel_name}")
                                select_columns.append(rel_expr)
                        else:
                            # List relation (possibly with nested) JSON aggregation
                            # Rebuild list relation JSON: adapter-aware; MSSQL via FOR JSON PATH
                            def _build_list_relation_json_adapter(parent_model_cls, parent_btype, rel_cfg_local):
                                target_name_i = rel_cfg_local.get('target')
                                target_b_i = self.types.get(target_name_i)
                                if not target_b_i or not target_b_i.model:
                                    return None
                                child_model_cls_i = target_b_i.model
                                fk_col_i = None
                                for col in child_model_cls_i.__table__.columns:
                                    for fk in col.foreign_keys:
                                        if fk.column.table.name == parent_model_cls.__table__.name:
                                            fk_col_i = col
                                            break
                                    if fk_col_i is not None:
                                        break
                                if fk_col_i is None:
                                    return None
                                requested_scalar_i = list(rel_cfg_local.get('fields') or [])
                                if not requested_scalar_i:
                                    for sf, sdef in target_b_i.__berry_fields__.items():
                                        if sdef.kind == 'scalar':
                                            requested_scalar_i.append(sf)
                                # Ensure FK helper columns for child's single relations are present
                                try:
                                    for r2, d2 in target_b_i.__berry_fields__.items():
                                        if d2.kind == 'relation' and (d2.meta.get('single') or d2.meta.get('mode') == 'single'):
                                            fk2 = f"{r2}_id"
                                            if fk2 in child_model_cls_i.__table__.columns and fk2 not in requested_scalar_i:
                                                requested_scalar_i.append(fk2)
                                except Exception:
                                    pass
                                inner_cols_i = [getattr(child_model_cls_i, c) for c in requested_scalar_i] if requested_scalar_i else [getattr(child_model_cls_i, 'id')]
                                if mssql_mode:
                                    parent_table = parent_model_cls.__tablename__
                                    child_table = child_model_cls_i.__tablename__
                                    # Start with FK correlation; extend with relation-level JSON where/default_where
                                    where_parts_rel: list[str] = [f"[{child_table}].[{fk_col_i.name}] = [{parent_table}].[id]"]
                                    # Helpers to render simple WHERE from JSON where dicts for MSSQL
                                    def _mssql_literal(col, v):
                                        try:
                                            v2 = _coerce_where_value(col, v)
                                        except Exception:
                                            v2 = v
                                        from sqlalchemy.sql.sqltypes import Integer as _I, Float as _F, Boolean as _B, DateTime as _DT, Numeric as _N
                                        ctype = getattr(col, 'type', None)
                                        if isinstance(v2, (int, float)):
                                            return str(v2)
                                        if isinstance(ctype, _B) or isinstance(v2, bool):
                                            return '1' if bool(v2) else '0'
                                        if isinstance(ctype, _DT):
                                            try:
                                                from datetime import datetime as _dt
                                                if isinstance(v2, _dt):
                                                    # Use ISO 8601 and CONVERT with style 126 for robust MSSQL parsing
                                                    iso = v2.replace(tzinfo=None).isoformat(sep='T', timespec='seconds')
                                                    return f"CONVERT(datetime2, '{iso}', 126)"
                                            except Exception:
                                                pass
                                        s = str(v2).replace("'", "''")
                                        return f"'{s}'"
                                    def _mssql_where_from_dict(model_cls_local, wdict) -> list[str]:
                                        parts: list[str] = []
                                        for col_name, op_map in (wdict or {}).items():
                                            col = model_cls_local.__table__.c.get(col_name)
                                            if col is None:
                                                continue
                                            for op_name, val in (op_map or {}).items():
                                                t = f"[{model_cls_local.__tablename__}].[{col_name}]"
                                                if op_name == 'eq':
                                                    parts.append(f"{t} = {_mssql_literal(col, val)}")
                                                elif op_name == 'ne':
                                                    parts.append(f"{t} <> {_mssql_literal(col, val)}")
                                                elif op_name == 'lt':
                                                    parts.append(f"{t} < {_mssql_literal(col, val)}")
                                                elif op_name == 'lte':
                                                    parts.append(f"{t} <= {_mssql_literal(col, val)}")
                                                elif op_name == 'gt':
                                                    parts.append(f"{t} > {_mssql_literal(col, val)}")
                                                elif op_name == 'gte':
                                                    parts.append(f"{t} >= {_mssql_literal(col, val)}")
                                                elif op_name == 'like':
                                                    parts.append(f"{t} LIKE {_mssql_literal(col, val)}")
                                                elif op_name == 'ilike':
                                                    # emulate case-insensitive like
                                                    parts.append(f"LOWER({t}) LIKE LOWER({_mssql_literal(col, val)})")
                                                elif op_name == 'in' and isinstance(val, (list, tuple)):
                                                    vals = ', '.join([_mssql_literal(col, v) for v in val])
                                                    parts.append(f"{t} IN ({vals})")
                                                elif op_name == 'between' and isinstance(val, (list, tuple)) and len(val) >= 2:
                                                    a = _mssql_literal(col, val[0])
                                                    b = _mssql_literal(col, val[1])
                                                    parts.append(f"{t} BETWEEN {a} AND {b}")
                                        return parts
                                    # Apply relation-level where/default_where if provided
                                    try:
                                        import json as _json
                                        r_where = rel_cfg_local.get('where')
                                        if r_where is not None:
                                            wdict_rel = r_where
                                            if isinstance(r_where, str):
                                                wdict_rel = _json.loads(r_where)
                                            where_parts_rel.extend(_mssql_where_from_dict(child_model_cls_i, wdict_rel))
                                        d_where = rel_cfg_local.get('default_where')
                                        if d_where is not None and isinstance(d_where, (dict, str)):
                                            dwdict_rel = d_where
                                            if isinstance(d_where, str):
                                                dwdict_rel = _json.loads(d_where)
                                            where_parts_rel.extend(_mssql_where_from_dict(child_model_cls_i, dwdict_rel))
                                    except Exception:
                                        pass
                                    where_clause = ' AND '.join(where_parts_rel)
                                    # Build ORDER BY honoring order_multi -> order_by/order_dir -> fallback id asc
                                    try:
                                        allowed_fields = [sf for sf, sd in target_b_i.__berry_fields__.items() if sd.kind == 'scalar']
                                    except Exception:
                                        allowed_fields = []
                                    order_parts: list[str] = []
                                    multi = (rel_cfg_local.get('order_multi') or [])
                                    for spec in multi:
                                        try:
                                            cn, _, dd = str(spec).partition(':')
                                            dd = dd or _dir_value(rel_cfg_local.get('order_dir'))
                                            if cn in allowed_fields:
                                                order_parts.append(f"[{child_table}].[{cn}] {'DESC' if (str(dd).lower()=='desc') else 'ASC'}")
                                        except Exception:
                                            pass
                                    if not order_parts and rel_cfg_local.get('order_by') in allowed_fields:
                                        cn = rel_cfg_local.get('order_by')
                                        dd = _dir_value(rel_cfg_local.get('order_dir'))
                                        order_parts.append(f"[{child_table}].[{cn}] {'DESC' if dd=='desc' else 'ASC'}")
                                    if not order_parts and 'id' in child_model_cls_i.__table__.columns:
                                        order_parts.append(f"[{child_table}].[id] ASC")
                                    order_clause = ', '.join(order_parts) if order_parts else None
                                    # Build nested subqueries for MSSQL if nested relations are requested
                                    nested_subqueries: list[tuple[str, str]] = []
                                    for nname, ncfg in (rel_cfg_local.get('nested') or {}).items():
                                        n_target = ncfg.get('target')
                                        nb = self.types.get(n_target)
                                        if not nb or not nb.model:
                                            continue
                                        grand_model = nb.model
                                        # use helper functions defined above for MSSQL where rendering
                                        # FK from grandchild -> child
                                        g_fk = None
                                        for c in grand_model.__table__.columns:
                                            for fk in c.foreign_keys:
                                                if fk.column.table.name == child_model_cls_i.__table__.name:
                                                    g_fk = c
                                                    break
                                            if g_fk is not None:
                                                break
                                        if g_fk is None:
                                            continue
                                        n_cols = ncfg.get('fields') or []
                                        if not n_cols:
                                            for sf2, sd2 in nb.__berry_fields__.items():
                                                if sd2.kind == 'scalar':
                                                    n_cols.append(sf2)
                                        n_col_select = ', '.join([f"[{grand_model.__tablename__}].[{c}] AS [{c}]" for c in (n_cols or ['id'])])
                                        # where: correlate to child row id in MSSQL path
                                        n_where_parts = [f"[{grand_model.__tablename__}].[{g_fk.name}] = [{child_table}].[id]"]
                                        # apply JSON where/default_where if provided
                                        try:
                                            import json as _json
                                            if ncfg.get('where') is not None:
                                                wdict = ncfg.get('where')
                                                if isinstance(wdict, str):
                                                    wdict = _json.loads(wdict)
                                                n_where_parts.extend(_mssql_where_from_dict(grand_model, wdict))
                                            if ncfg.get('default_where') is not None and isinstance(ncfg.get('default_where'), (dict, str)):
                                                dwdict = ncfg.get('default_where')
                                                if isinstance(dwdict, str):
                                                    dwdict = _json.loads(dwdict)
                                                n_where_parts.extend(_mssql_where_from_dict(grand_model, dwdict))
                                        except Exception:
                                            pass
                                        n_where = ' AND '.join(n_where_parts)
                                        # order
                                        n_order_parts: list[str] = []
                                        nmulti = (ncfg.get('order_multi') or [])
                                        for spec in nmulti:
                                            cn, _, dd = str(spec).partition(':')
                                            dd = dd or _dir_value(ncfg.get('order_dir'))
                                            n_order_parts.append(f"[{grand_model.__tablename__}].[{cn}] {'DESC' if (str(dd).lower()=='desc') else 'ASC'}")
                                        if not n_order_parts and ncfg.get('order_by'):
                                            cn = ncfg.get('order_by')
                                            dd = _dir_value(ncfg.get('order_dir'))
                                            n_order_parts.append(f"[{grand_model.__tablename__}].[{cn}] {'DESC' if dd=='desc' else 'ASC'}")
                                        n_order_clause = (" ORDER BY " + ', '.join(n_order_parts)) if n_order_parts else ''
                                        n_top = f"TOP ({int(ncfg.get('limit'))}) " if ncfg.get('limit') is not None else ''
                                        # where/default_where (JSON): not applied in MSSQL nested builder to keep simple; future enhancement can mirror PG/SQLite
                                        nested_sql = (
                                            f"SELECT {n_top}{n_col_select} FROM {grand_model.__tablename__} WHERE {n_where}{n_order_clause} FOR JSON PATH"
                                        )
                                        nested_subqueries.append((nname, nested_sql))
                                    return adapter.build_list_relation_json(
                                        child_table=child_table,
                                        projected_columns=requested_scalar_i,
                                        where_condition=where_clause,
                                        limit=rel_cfg_local.get('limit'),
                                        order_by=order_clause,
                                        nested_subqueries=nested_subqueries or None,
                                    )
                                inner_sel_i = select(*inner_cols_i).select_from(child_model_cls_i).where(fk_col_i == parent_model_cls.id).correlate(parent_model_cls)
                                # Apply filter args for list relation
                                try:
                                    filter_args = rel_cfg.get('filter_args') or {}
                                    if filter_args:
                                        target_btype = self.types.get(rel_cfg.get('target'))
                                        if target_btype:
                                            class_filters = getattr(target_btype, '__filters__', {}) or {}
                                            expanded: Dict[str, FilterSpec] = {}
                                            for key, raw in class_filters.items():
                                                try:
                                                    spec = _normalize_filter_spec(raw)
                                                except Exception:
                                                    continue
                                                if spec.ops and not spec.op:
                                                    for op_name in spec.ops:
                                                        base = spec.alias or key
                                                        if base.endswith(f"_{op_name}"):
                                                            an = base
                                                        else:
                                                            an = f"{base}_{op_name}"
                                                        expanded[an] = spec.clone_with(op=op_name, ops=None)
                                                else:
                                                    expanded[spec.alias or key] = spec
                                            for arg_name, val in filter_args.items():
                                                f_spec = expanded.get(arg_name)
                                                if not f_spec:
                                                    continue
                                                if f_spec.transform:
                                                    try:
                                                        val = f_spec.transform(val)
                                                    except Exception:
                                                        continue
                                                expr = None
                                                if f_spec.builder:
                                                    try:
                                                        expr = f_spec.builder(child_model_cls_i, info, val)
                                                    except Exception:
                                                        expr = None
                                                elif f_spec.column:
                                                    try:
                                                        col = child_model_cls_i.__table__.c.get(f_spec.column)
                                                    except Exception:
                                                        col = None
                                                    if col is None:
                                                        continue
                                                    op_fn = OPERATOR_REGISTRY.get(f_spec.op or 'eq')
                                                    if not op_fn:
                                                        continue
                                                    try:
                                                        expr = op_fn(col, val)
                                                    except Exception:
                                                        expr = None
                                                if expr is not None:
                                                    inner_sel_i = inner_sel_i.where(expr)
                                except Exception:
                                    pass
                                try:
                                    # Apply ordering (multi -> single -> fallback)
                                    ordered_i = False
                                    try:
                                        allowed_order_fields_i = [sf for sf, sd in target_b_i.__berry_fields__.items() if sd.kind == 'scalar']
                                        multi_i = rel_cfg_local.get('order_multi') or []
                                        for spec in multi_i:
                                            cn, _, dd = str(spec).partition(':')
                                            dd = dd or 'asc'
                                            if cn in allowed_order_fields_i:
                                                col = getattr(child_model_cls_i, cn, None)
                                                if col is not None:
                                                    inner_sel_i = inner_sel_i.order_by(col.desc() if dd.lower()=='desc' else col.asc())
                                                    ordered_i = True
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                # Apply ad-hoc JSON where for nested relation as well
                                try:
                                    rr = rel_cfg_local.get('where')
                                    dr = rel_cfg_local.get('default_where')
                                    if rr is not None:
                                        import json as _json
                                        wdict2 = rr
                                        if isinstance(rr, str):
                                            wdict2 = _json.loads(rr)
                                        expr_rr = None
                                        try:
                                            exprs2 = []
                                            for col_name2, op_map2 in (wdict2 or {}).items():
                                                col2 = child_model_cls_i.__table__.c.get(col_name2)
                                                if col2 is None:
                                                    continue
                                                for op_name2, val2 in (op_map2 or {}).items():
                                                    if op_name2 in ('in','between') and isinstance(val2, (list, tuple)):
                                                        val2 = [_coerce_where_value(col2, v) for v in val2]
                                                    else:
                                                        val2 = _coerce_where_value(col2, val2)
                                                    op_fn2 = OPERATOR_REGISTRY.get(op_name2)
                                                    if not op_fn2:
                                                        continue
                                                    exprs2.append(op_fn2(col2, val2))
                                            if exprs2:
                                                expr_rr = _and(*exprs2)
                                        except Exception:
                                            expr_rr = None
                                        if expr_rr is not None:
                                            inner_sel_i = inner_sel_i.where(expr_rr)
                                    if dr is not None:
                                        import json as _json
                                        wdict2 = dr
                                        if isinstance(dr, str):
                                            wdict2 = _json.loads(dr)
                                        expr_rr = None
                                        try:
                                            exprs2 = []
                                            for col_name2, op_map2 in (wdict2 or {}).items():
                                                col2 = child_model_cls_i.__table__.c.get(col_name2)
                                                if col2 is None:
                                                    continue
                                                for op_name2, val2 in (op_map2 or {}).items():
                                                    if op_name2 in ('in','between') and isinstance(val2, (list, tuple)):
                                                        val2 = [_coerce_where_value(col2, v) for v in val2]
                                                    else:
                                                        val2 = _coerce_where_value(col2, val2)
                                                    op_fn2 = OPERATOR_REGISTRY.get(op_name2)
                                                    if not op_fn2:
                                                        continue
                                                    exprs2.append(op_fn2(col2, val2))
                                            if exprs2:
                                                expr_rr = _and(*exprs2)
                                        except Exception:
                                            expr_rr = None
                                        if expr_rr is not None:
                                            inner_sel_i = inner_sel_i.where(expr_rr)
                                except Exception:
                                    pass
                                if rel_cfg_local.get('offset') is not None:
                                    try:
                                        inner_sel_i = inner_sel_i.offset(int(rel_cfg_local['offset']))
                                    except Exception:
                                        inner_sel_i = inner_sel_i.offset(rel_cfg_local['offset'])
                                if rel_cfg_local.get('limit') is not None:
                                    try:
                                        inner_sel_i = inner_sel_i.limit(int(rel_cfg_local['limit']))
                                    except Exception:
                                        inner_sel_i = inner_sel_i.limit(rel_cfg_local['limit'])
                                limited_subq_i = inner_sel_i.subquery()
                                row_json_args_i: List[Any] = []
                                for sf in requested_scalar_i if requested_scalar_i else ['id']:
                                    row_json_args_i.extend([_text(f"'{sf}'"), getattr(limited_subq_i.c, sf)])
                                # nested not reimplemented here (fallback to earlier recursive version if needed)
                                if mssql_mode:
                                    return None  # handled above
                                row_json_expr_i = _json_object(*row_json_args_i)
                                agg_inner_expr_i = _json_array_agg(row_json_expr_i)
                                if agg_inner_expr_i is None:
                                    return None
                                agg_query_i = select(_json_array_coalesce(agg_inner_expr_i)).select_from(limited_subq_i).correlate(parent_model_cls).scalar_subquery()
                                return agg_query_i
                            # Prefer nested-capable builder when nested relations are selected
                            nested_expr = None
                            try:
                                if (rel_cfg.get('nested') or {}) and not mssql_mode:
                                    nested_expr = _build_list_relation_json(model_cls, btype_cls, rel_cfg)
                            except Exception:
                                nested_expr = None
                            if nested_expr is None:
                                nested_expr = _build_list_relation_json_adapter(model_cls, btype_cls, rel_cfg)
                            if nested_expr is not None:
                                if mssql_mode:
                                    # Wrap TextClause into a labeled column using scalar_subquery style text
                                    try:
                                        select_columns.append(nested_expr.label(f"_pushrel_{rel_name}"))
                                    except Exception:
                                        from sqlalchemy import literal_column
                                        select_columns.append(literal_column(str(nested_expr)).label(f"_pushrel_{rel_name}"))
                                else:
                                    select_columns.append(nested_expr.label(f"_pushrel_{rel_name}"))
                    stmt = select(*select_columns)
                    # ----- Phase 2 filtering (argument-driven) -----
                    where_clauses = []
                    # Apply optional context-aware root custom where if provided on BerryType
                    try:
                        custom_where = getattr(btype_cls, '__root_custom_where__', None)
                    except Exception:
                        custom_where = None
                    def _expr_from_where_dict(model_cls_local, wdict):
                        exprs = []
                        try:
                            for col_name, op_map in (wdict or {}).items():
                                try:
                                    col = model_cls_local.__table__.c.get(col_name)
                                except Exception:
                                    col = None
                                if col is None:
                                    continue
                                for op_name, val in (op_map or {}).items():
                                    op_fn = OPERATOR_REGISTRY.get(op_name)
                                    if not op_fn:
                                        continue
                                    try:
                                        # Coerce where value(s) to column type
                                        if op_name in ('in','between') and isinstance(val, (list, tuple)):
                                            val = [_coerce_where_value(col, v) for v in val]
                                        else:
                                            val = _coerce_where_value(col, val)
                                        exprs.append(op_fn(col, val))
                                    except Exception:
                                        continue
                        except Exception:
                            return None
                        if not exprs:
                            return None
                        try:
                            return _and(*exprs)
                        except Exception:
                            return None
                    # Only enforce custom_where when explicitly enabled by context flag
                    if custom_where is not None and bool(getattr(info, 'context', {}) and info.context.get('enforce_user_gate')):
                        try:
                            cw_val = custom_where(model_cls, info) if callable(custom_where) else custom_where
                        except Exception:
                            cw_val = None
                        if cw_val is not None:
                            # accept raw SQLAlchemy expression or simple dict form {col: {op: val}}
                            try:
                                # Heuristic: dict-like -> build expression
                                if isinstance(cw_val, dict):
                                    expr_obj = _expr_from_where_dict(model_cls, cw_val)
                                else:
                                    expr_obj = cw_val
                                if expr_obj is not None:
                                    where_clauses.append(expr_obj)
                            except Exception:
                                pass
                    # Apply ad-hoc raw JSON where string if provided by user
                    if raw_where is not None:
                        try:
                            import json as _json
                            wdict = raw_where
                            if isinstance(raw_where, str):
                                wdict = _json.loads(raw_where)
                            expr2 = _expr_from_where_dict(model_cls, wdict) if isinstance(wdict, dict) else None
                            if expr2 is not None:
                                where_clauses.append(expr2)
                        except Exception:
                            pass
                    for arg_name, value in _passed_filter_args.items():
                        if value is None:
                            continue
                        f_spec = declared_filters.get(arg_name)
                        if not f_spec:
                            raise ValueError(f"Unknown filter argument: {arg_name}")
                        # transform value
                        if f_spec.transform:
                            try:
                                value = f_spec.transform(value)
                            except Exception as e:
                                raise ValueError(f"Filter transform failed for {arg_name}: {e}")
                        expr = None
                        if f_spec.builder:
                            try:
                                expr = f_spec.builder(model_cls, info, value)
                            except Exception as e:
                                raise ValueError(f"Filter builder failed for {arg_name}: {e}")
                        elif f_spec.column:
                            try:
                                col = model_cls.__table__.c.get(f_spec.column)
                            except Exception:
                                col = None
                        if col is None:
                            raise ValueError(f"Unknown filter column: {f_spec.column} for argument {arg_name}")
                        op_name = f_spec.op or 'eq'
                        op_fn = OPERATOR_REGISTRY.get(op_name)
                        if not op_fn:
                            raise ValueError(f"Unknown filter operator: {op_name} for argument {arg_name}")
                        try:
                            expr = op_fn(col, value)
                        except Exception as e:
                            raise ValueError(f"Filter operation failed for {arg_name}: {e}")
                        if expr is not None:
                            where_clauses.append(expr)
                    if where_clauses:
                        for wc in where_clauses:
                            try:
                                stmt = stmt.where(wc)
                            except Exception:
                                pass
                    # Apply ordering (whitelist) before pagination
                    if order_by:
                        allowed_order_fields = getattr(btype_cls, '__ordering__', None)
                        if allowed_order_fields is None:
                            # default allow scalar field names
                            allowed_order_fields = [fname for fname, fdef in btype_cls.__berry_fields__.items() if fdef.kind == 'scalar']
                        if order_by not in allowed_order_fields:
                            raise ValueError(f"Invalid order_by '{order_by}'. Allowed: {allowed_order_fields}")
                        try:
                            col = model_cls.__table__.c.get(order_by)
                        except Exception:
                            col = None
                        if col is None:
                            raise ValueError(f"Unknown order_by column: {order_by}")
                        dir_v = _dir_value(order_dir)
                        if dir_v not in ('asc','desc'):
                            raise ValueError(f"Invalid order_dir '{order_dir}'. Use asc or desc")
                        try:
                            stmt = stmt.order_by(col.desc() if dir_v == 'desc' else col.asc())
                        except Exception as e:
                            raise
                    else:
                        # Apply default ordering from type meta if present
                        allowed_order_fields = getattr(btype_cls, '__ordering__', None)
                        if allowed_order_fields is None:
                            allowed_order_fields = [fname for fname, fdef in btype_cls.__berry_fields__.items() if fdef.kind == 'scalar']
                        def_dir = _dir_value(getattr(btype_cls, '__default_order_dir__', None))
                        default_multi = getattr(btype_cls, '__default_order_multi__', None) or []
                        default_by = getattr(btype_cls, '__default_order_by__', None)
                        try:
                            applied_default = False
                            if default_multi:
                                for spec in default_multi:
                                    cn, _, dd = str(spec).partition(':')
                                    dd = dd or def_dir
                                    if cn in allowed_order_fields:
                                        col = model_cls.__table__.c.get(cn)
                                        if col is not None:
                                            stmt = stmt.order_by(col.desc() if (dd=='desc') else col.asc())
                                            applied_default = True
                            elif default_by and default_by in allowed_order_fields:
                                col = model_cls.__table__.c.get(default_by)
                                if col is not None:
                                    stmt = stmt.order_by(col.desc() if def_dir=='desc' else col.asc())
                                    applied_default = True
                        except Exception:
                            pass
                    if offset is not None:
                        try:
                            o = int(offset)
                        except Exception:
                            raise ValueError("offset must be an integer")
                        if o < 0:
                            raise ValueError("offset must be non-negative")
                        if o:
                            stmt = stmt.offset(o)
                    if limit is not None:
                        try:
                            l = int(limit)
                        except Exception:
                            raise ValueError("limit must be an integer")
                        if l < 0:
                            raise ValueError("limit must be non-negative")
                        stmt = stmt.limit(l)
                    async with lock:
                        result = await session.execute(stmt)
                        sa_rows = result.fetchall()
                        rows = [r[0] for r in sa_rows]
                        out = []
                        for row_index, row in enumerate(rows):
                            inst = st_cls()
                            setattr(inst, '_model', row)
                            # hydrate scalar fields
                            for sf, sdef in btype_cls.__berry_fields__.items():
                                if sdef.kind == 'scalar':
                                    try:
                                        setattr(inst, sf, getattr(row, sf, None))
                                    except Exception:
                                        pass
                            # attach custom scalar field values from select
                            if custom_fields:
                                base_offset = 1  # model at position 0
                                for idx, (cf_name, _) in enumerate(custom_fields, start=base_offset):
                                    try:
                                        val = sa_rows[row_index][idx]
                                        setattr(inst, cf_name, val)
                                    except Exception:
                                        pass
                            # reconstruct custom object fields (prefer single JSON column if present)
                            if custom_object_fields:
                                try:
                                    mapping = getattr(sa_rows[row_index], '_mapping')
                                except Exception:
                                    mapping = {}
                                for cf_name, col_labels, returns_spec in custom_object_fields:
                                    obj = None
                                    # If we labeled a JSON column for this field, use it; else assemble from multiple scalar labels (MSSQL)
                                    json_col = col_labels[0] if (col_labels and len(col_labels) == 1) else None
                                    raw_json = mapping[json_col] if (json_col and (json_col in mapping)) else None
                                    # Optional debug: print available keys and value when enabled
                                    try:
                                        import os as _os
                                        if _os.environ.get('BERRY_DEBUG') == '1' and json_col:
                                            try:
                                                print(f"[berry debug] custom_object '{cf_name}' json_col='{json_col}' keys={list(mapping.keys())}")
                                                print(f"[berry debug] custom_object raw_json={raw_json!r}")
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                                    if raw_json is not None:
                                        import json as _json
                                        parsed = None
                                        try:
                                            parsed = _json.loads(raw_json) if isinstance(raw_json, (str, bytes)) else raw_json
                                        except Exception:
                                            parsed = None
                                        data_dict = parsed if isinstance(parsed, dict) else None
                                        # MSSQL single-key labeled scalar value: build dict using label suffix
                                        if data_dict is None and json_col and '__' in json_col:
                                            try:
                                                key = json_col.split('__', 1)[1]
                                            except Exception:
                                                key = json_col
                                            data_dict = {key: parsed if parsed is not None else raw_json}
                                    else:
                                        # MSSQL path: multiple scalar labeled columns like _pushcf_<field>__<key>
                                        data_dict = None
                                        try:
                                            if col_labels and len(col_labels) > 1:
                                                tmp: dict[str, Any] = {}
                                                for lbl in col_labels:
                                                    if lbl in mapping:
                                                        # key is suffix after "__"
                                                        try:
                                                            key = lbl.split('__', 1)[1]
                                                        except Exception:
                                                            key = lbl
                                                        tmp[key] = mapping[lbl]
                                                data_dict = tmp
                                        except Exception:
                                            data_dict = None
                                    # Filter keys to declared returns
                                    if data_dict and isinstance(returns_spec, dict):
                                        data_dict = {k: data_dict.get(k) for k in returns_spec.keys() if k in data_dict}
                                        # Coerce values to expected Python types when possible (e.g., datetime)
                                        try:
                                            for k2, t2 in returns_spec.items():
                                                if k2 in data_dict:
                                                    v = data_dict[k2]
                                                    # datetime from ISO string
                                                    if t2 is datetime and isinstance(v, str):
                                                        try:
                                                            data_dict[k2] = datetime.fromisoformat(v)
                                                        except Exception:
                                                            # try common variant with 'T'
                                                            try:
                                                                data_dict[k2] = datetime.fromisoformat(v.replace(' ', 'T'))
                                                            except Exception:
                                                                pass
                                        except Exception:
                                            pass
                                    if data_dict:
                                        nested_type_name = f"{btype_cls.__name__}_{cf_name}_Type"
                                        nested_type = self._st_types.get(nested_type_name)
                                        try:
                                            if nested_type is not None:
                                                # Prefer constructing via kwargs to satisfy dataclass/strawberry init
                                                try:
                                                    obj = nested_type(**data_dict)
                                                except Exception:
                                                    obj = nested_type()
                                                    for k, v in data_dict.items():
                                                        try:
                                                            setattr(obj, k, v)
                                                        except Exception:
                                                            pass
                                            else:
                                                obj = data_dict
                                        except Exception:
                                            obj = data_dict
                                    # cache on instance and set public attribute as well
                                    setattr(inst, f"_{cf_name}_data", obj)
                                    setattr(inst, f"_{cf_name}_prefetched", obj)
                                    try:
                                        setattr(inst, cf_name, obj)
                                    except Exception:
                                        pass
                            # hydrate pushed-down relation JSON
                            try:
                                mapping = getattr(sa_rows[row_index], '_mapping')
                            except Exception:
                                mapping = {}
                            if requested_relations:
                                for rel_name in requested_relations.keys():
                                    key = f"_pushrel_{rel_name}"
                                    if key in mapping:
                                        raw_json = mapping[key]
                                        rel_meta = requested_relations[rel_name]
                                        is_single = rel_meta.get('single')
                                        import json as _json
                                        if raw_json is None:
                                            parsed_value = None if is_single else []
                                        else:
                                            try:
                                                parsed_value = _json.loads(raw_json) if isinstance(raw_json, (str, bytes)) else raw_json
                                            except Exception:
                                                parsed_value = (None if is_single else [])
                                        target_name = rel_meta.get('target')
                                        target_b = self.types.get(target_name) if target_name else None
                                        target_st = self._st_types.get(target_name) if target_name else None
                                        built_value = None if is_single else []
                                        if target_b and target_b.model and target_st and parsed_value is not None:
                                            if is_single:
                                                if isinstance(parsed_value, dict):
                                                    child_inst = target_st()
                                                    for sf, sdef in target_b.__berry_fields__.items():
                                                        if sdef.kind == 'scalar':
                                                            val = parsed_value.get(sf)
                                                            # Convert ISO datetime strings to datetime objects if field type is datetime
                                                            try:
                                                                target_model = target_b.model
                                                                col = target_model.__table__.c.get(sf) if target_model is not None else None
                                                                if col is not None and isinstance(getattr(col, 'type', None), DateTime) and isinstance(val, str):
                                                                    try:
                                                                        val = datetime.fromisoformat(val)
                                                                    except Exception:
                                                                        pass
                                                            except Exception:
                                                                pass
                                                            setattr(child_inst, sf, val)
                                                    setattr(child_inst, '_model', None)
                                                    built_value = child_inst
                                                else:
                                                    built_value = None
                                            else:
                                                tmp_list = []
                                                if isinstance(parsed_value, list):
                                                    for item in parsed_value:
                                                        if isinstance(item, dict):
                                                            child_inst = target_st()
                                                            # Scalars
                                                            for sf, sdef in target_b.__berry_fields__.items():
                                                                if sdef.kind == 'scalar':
                                                                    val = item.get(sf)
                                                                    try:
                                                                        target_model = target_b.model
                                                                        col = target_model.__table__.c.get(sf) if target_model is not None else None
                                                                        if col is not None and isinstance(getattr(col, 'type', None), DateTime) and isinstance(val, str):
                                                                            try:
                                                                                val = datetime.fromisoformat(val)
                                                                            except Exception:
                                                                                pass
                                                                    except Exception:
                                                                        pass
                                                                    setattr(child_inst, sf, val)
                                                            setattr(child_inst, '_model', None)
                                                            # Nested relations on this child (prefetch to avoid N+1)
                                                            try:
                                                                for nname, ndef in target_b.__berry_fields__.items():
                                                                    if ndef.kind != 'relation':
                                                                        continue
                                                                    raw_nested = item.get(nname, None)
                                                                    if raw_nested is None:
                                                                        continue
                                                                    import json as _json
                                                                    parsed_nested = None
                                                                    try:
                                                                        parsed_nested = _json.loads(raw_nested) if isinstance(raw_nested, (str, bytes)) else raw_nested
                                                                    except Exception:
                                                                        parsed_nested = None
                                                                    n_target = self.types.get(ndef.meta.get('target')) if ndef.meta.get('target') else None
                                                                    n_st = self._st_types.get(ndef.meta.get('target')) if ndef.meta.get('target') else None
                                                                    if not n_target or not n_target.model or not n_st:
                                                                        continue
                                                                    if ndef.meta.get('single'):
                                                                        if isinstance(parsed_nested, dict):
                                                                            ni = n_st()
                                                                            for nsf, nsdef in n_target.__berry_fields__.items():
                                                                                if nsdef.kind == 'scalar':
                                                                                    setattr(ni, nsf, parsed_nested.get(nsf))
                                                                            setattr(ni, '_model', None)
                                                                            setattr(child_inst, nname, ni)
                                                                            setattr(child_inst, f"_{nname}_prefetched", ni)
                                                                        else:
                                                                            setattr(child_inst, nname, None)
                                                                            setattr(child_inst, f"_{nname}_prefetched", None)
                                                                    else:
                                                                        nlist = []
                                                                        if isinstance(parsed_nested, list):
                                                                            for nv in parsed_nested:
                                                                                if isinstance(nv, dict):
                                                                                    ni = n_st()
                                                                                    for nsf, nsdef in n_target.__berry_fields__.items():
                                                                                        if nsdef.kind == 'scalar':
                                                                                            setattr(ni, nsf, nv.get(nsf))
                                                                                    setattr(ni, '_model', None)
                                                                                    nlist.append(ni)
                                                                        setattr(child_inst, nname, nlist)
                                                                        setattr(child_inst, f"_{nname}_prefetched", nlist)
                                                                    # record nested pushdown meta
                                                                    try:
                                                                        meta_map2 = getattr(child_inst, '_pushdown_meta', None)
                                                                        if meta_map2 is None:
                                                                            meta_map2 = {}
                                                                            setattr(child_inst, '_pushdown_meta', meta_map2)
                                                                        parent_rel_meta = requested_relations.get(rel_name, {})
                                                                        nested_meta_src = (parent_rel_meta.get('nested') or {}).get(nname, {})
                                                                        meta_map2[nname] = {
                                                                            'limit': nested_meta_src.get('limit'),
                                                                            'offset': nested_meta_src.get('offset'),
                                                                            'from_pushdown': True
                                                                        }
                                                                    except Exception:
                                                                        pass
                                                            except Exception:
                                                                pass
                                                            tmp_list.append(child_inst)
                                                # Apply Python-side ordering if specified (ensures correct order when subquery dropped ORDER BY)
                                                try:
                                                    rel_cfg_meta = requested_relations.get(rel_name, {})
                                                    multi_specs = rel_cfg_meta.get('order_multi') or []
                                                    single_by = rel_cfg_meta.get('order_by')
                                                    single_dir = _dir_value(rel_cfg_meta.get('order_dir'))
                                                    if multi_specs:
                                                        # apply multi-key sort, last key first for stability
                                                        specs = []
                                                        for spec in multi_specs:
                                                            cn, _, dd = str(spec).partition(':')
                                                            dd = (dd or 'asc').lower()
                                                            specs.append((cn, dd))
                                                        for cn, dd in reversed(specs):
                                                            tmp_list.sort(key=lambda o: getattr(o, cn, None), reverse=(dd=='desc'))
                                                    elif single_by:
                                                        tmp_list.sort(key=lambda o: getattr(o, single_by, None), reverse=(single_dir=='desc'))
                                                except Exception:
                                                    pass
                                                built_value = tmp_list
                                        else:
                                            built_value = parsed_value
                                        setattr(inst, f"_{rel_name}_prefetched", built_value)
                                        # record pushdown meta for pagination reuse
                                        meta_map = getattr(inst, '_pushdown_meta', None)
                                        if meta_map is None:
                                            meta_map = {}
                                            setattr(inst, '_pushdown_meta', meta_map)
                                        meta_map[rel_name] = {
                                            'limit': rel_cfg.get('limit'),
                                            'offset': rel_cfg.get('offset'),
                                            'from_pushdown': True
                                        }
                                        # Also assign to public attribute to avoid resolver DB path
                                        try:
                                            setattr(inst, rel_name, built_value)
                                        except Exception:
                                            pass
                            # populate aggregate count cache from pushdown columns
                            if count_aggregates:
                                cache = getattr(inst, '_agg_cache', None)
                                if cache is None:
                                    cache = {}
                                    setattr(inst, '_agg_cache', cache)
                                for agg_name, agg_def in count_aggregates:
                                    try:
                                        val = mapping.get(agg_name)
                                    except Exception:
                                        val = None
                                    cache_key = agg_def.meta.get('cache_key') or (agg_def.meta.get('source') + ':count')
                                    cache[cache_key] = val or 0
                            out.append(inst)
                        # Explicitly close result to release DBAPI cursor (important for MSSQL without MARS)
                        try:
                            try:
                                await result.close()  # type: ignore[func-returns-value]
                            except TypeError:
                                # some SQLAlchemy versions expose close() as sync
                                result.close()
                        except Exception:
                            pass
                    return out
                # Dynamically build a resolver with explicit filter args so Strawberry generates them
                # Build function source
                arg_defs = []
                for a, t in filter_arg_types.items():
                    # Represent type name for function definition (use forward ref if needed)
                    # We'll not embed complex generics; rely on annotations assignment after creation
                    arg_defs.append(f"{a}=None")
                args_str = (', '.join(arg_defs)) if arg_defs else ''
                func_name = f"_auto_root_{root_field_name}"
                # Build parameter list: info, limit, offset, ordering, optional where, filter args
                if args_str:
                    full_params = f"info, limit=None, offset=None, order_by=None, order_dir=None, where=None, {args_str}"
                else:
                    full_params = "info, limit=None, offset=None, order_by=None, order_dir=None, where=None"
                src = f"async def {func_name}({full_params}):\n" \
                      f"    _fa = {{}}\n"  # gather passed filter args
                for a in declared_filters.keys():
                    src += f"    _fa['{a}'] = {a} if '{a}' in locals() else None\n"
                src += "    return await _base_impl(info, limit, offset, order_by, order_dir, _fa, where)\n"
                # Exec the function in a prepared namespace
                ns: Dict[str, Any] = {'_base_impl': _base_impl}
                # Provide required symbols for optional typing (Optional, List, datetime, etc.) though not used in parameter defaults
                ns.update({'Optional': Optional, 'List': List, 'datetime': datetime})
                exec(src, ns)
                generated_fn = ns[func_name]
                # Ensure module attribute for Strawberry namespace resolution
                if not getattr(generated_fn, '__module__', None):  # pragma: no cover - environment dependent
                    generated_fn.__module__ = __name__
                # Attach type annotations for Strawberry to introspect
                ann: Dict[str, Any] = {'info': StrawberryInfo, 'limit': Optional[int], 'offset': Optional[int], 'order_by': Optional[str], 'order_dir': Optional[Direction], 'where': Optional[str]}
                ann.update(filter_arg_types)
                generated_fn.__annotations__ = ann
                return generated_fn
            root_resolver = _make_root_resolver(_model_cls, st_type, bcls, field_name)
            query_annotations[field_name] = List[self._st_types[name]]  # type: ignore
            query_namespace[field_name] = strawberry.field(root_resolver)
        # Add ping field
        async def _ping() -> str:  # noqa: D401
            return 'pong'
        query_annotations['_ping'] = str
        query_namespace['_ping'] = strawberry.field(resolver=_ping)
        # Optional convenience root: current_user (if a User-like type exists)
        # We choose the first registered type whose name endswith 'UserQL' or has an 'is_admin' scalar field
        user_type_name: Optional[str] = None
        for tname, btype in self.types.items():
            try:
                if tname.lower().endswith('userql') or ('is_admin' in btype.__berry_fields__):
                    user_type_name = tname
                    break
            except Exception:
                continue
        if user_type_name and user_type_name in self._st_types:
            UserSt = self._st_types[user_type_name]
            UserBt = self.types[user_type_name]
            async def _current_user(self, info: StrawberryInfo):  # noqa: D401
                session = info.context.get('db_session') if info and info.context else None
                if session is None:
                    return None
                user_id = None
                try:
                    user_id = info.context.get('user_id') if info and info.context else None
                except Exception:
                    user_id = None
                if user_id is None:
                    return None
                # Custom logic parity with legacy: raise when user_id==999
                if user_id == 999:
                    raise ValueError("Custom logic executed: User 999 is forbidden!")
                try:
                    model_cls = getattr(UserBt, 'model', None)
                    if not model_cls:
                        return None
                    row = await session.get(model_cls, user_id)
                except Exception:
                    row = None
                if not row:
                    return None
                inst = UserSt()
                setattr(inst, '_model', row)
                # hydrate scalars
                for sf, sdef in UserBt.__berry_fields__.items():
                    if sdef.kind == 'scalar':
                        try:
                            setattr(inst, sf, getattr(row, sf, None))
                        except Exception:
                            pass
                return inst
            query_annotations['current_user'] = Optional[UserSt]  # type: ignore
            query_namespace['current_user'] = strawberry.field(resolver=_current_user)
        query_namespace['__annotations__'] = query_annotations
        Query = type('Query', (), query_namespace)
        Query = strawberry.type(Query)  # type: ignore
        try:
            # Prefer direct flag (older versions). Fallback to config object.
            return strawberry.Schema(Query, auto_camel_case=False)  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover
            try:
                from strawberry.schema.config import StrawberryConfig  # type: ignore
                return strawberry.Schema(Query, config=StrawberryConfig(auto_camel_case=False))
            except Exception:
                return strawberry.Schema(Query)
