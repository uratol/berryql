from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, get_type_hints
import asyncio
import strawberry
from sqlalchemy import select, func, text as _text
try:  # adapter abstraction
    from .adapters import get_adapter  # type: ignore
except Exception:  # pragma: no cover
    get_adapter = None  # type: ignore
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.sqltypes import Integer, String, Boolean, DateTime
from datetime import datetime
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
           single: bool,
           target: str | None,
           nested: { ... future nested relation data ... },
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
            'single': fdef.meta.get('single') or fdef.meta.get('mode') == 'single',
            'target': fdef.meta.get('target'),
            'nested': {},
            'skip_pushdown': False,
            'filter_args': {}
        }

    # Selected fields path (Strawberry's resolved selection tree)
    def _walk_selected(self, sel, btype, out: Dict[str, Dict[str, Any]]):
        if not getattr(sel, 'selections', None) or not btype:
            return
        for child in sel.selections:
            name = getattr(child, 'name', None)
            if not name:
                continue
            fdef = getattr(btype, '__berry_fields__', {}).get(name)
            if not fdef or fdef.kind != 'relation':
                continue
            rel_cfg = out.setdefault(name, self._init_rel_cfg(fdef))
            # arguments
            try:
                for arg in getattr(child, 'arguments', []) or []:
                    raw_val = getattr(arg, 'value', None)
                    try:
                        if hasattr(raw_val, 'value') and raw_val is not None and raw_val.__class__.__name__ != 'datetime':
                            inner_v = getattr(raw_val, 'value')
                            if isinstance(inner_v, (int, str, float, bool)):
                                raw_val = inner_v
                    except Exception:
                        pass
                    if arg.name == 'limit':
                        rel_cfg['limit'] = raw_val
                    elif arg.name == 'offset':
                        rel_cfg['offset'] = raw_val
                    else:
                        rel_cfg['filter_args'][arg.name] = raw_val
            except Exception:
                pass
            # scalar subfields
            if getattr(child, 'selections', None):
                tgt_b = self.registry.types.get(fdef.meta.get('target')) if fdef.meta.get('target') else None
                for sub in child.selections:
                    sub_name = getattr(sub, 'name', None)
                    if sub_name and not sub_name.startswith('__'):
                        sub_def = getattr(tgt_b, '__berry_fields__', {}).get(sub_name) if tgt_b else None
                        if not sub_def or sub_def.kind == 'scalar':
                            if sub_name not in rel_cfg['fields']:
                                rel_cfg['fields'].append(sub_name)
            # recurse nested relations
            try:
                self._walk_selected(child, self.registry.types.get(fdef.meta.get('target')), out)
            except Exception:
                pass

    # Raw GraphQL AST path
    def _value_from_ast(self, node):
        try:
            from graphql.language import ast as _gast
            if isinstance(node, _gast.IntValueNode):
                return int(node.value)
            if isinstance(node, _gast.FloatValueNode):
                return float(node.value)
            if isinstance(node, _gast.StringValueNode):
                return node.value
            if isinstance(node, _gast.BooleanValueNode):
                return bool(node.value)
            if isinstance(node, _gast.NullValueNode):
                return None
            if isinstance(node, _gast.ListValueNode):
                return [self._value_from_ast(v) for v in node.values]
            if hasattr(node, 'value'):
                return getattr(node, 'value')
        except Exception:
            return None
        return None

    def _walk_ast(self, selection_set, btype, out: Dict[str, Dict[str, Any]]):
        if not selection_set or not btype:
            return
        for child in getattr(selection_set, 'selections', []) or []:
            name_node = getattr(child, 'name', None)
            name = getattr(name_node, 'value', None) if name_node and not isinstance(name_node, str) else name_node
            if not name:
                continue
            fdef = getattr(btype, '__berry_fields__', {}).get(name)
            if not fdef or fdef.kind != 'relation':
                continue
            rel_cfg = out.setdefault(name, self._init_rel_cfg(fdef))
            for arg in getattr(child, 'arguments', []) or []:
                arg_name_node = getattr(arg, 'name', None)
                arg_name = getattr(arg_name_node, 'value', None) if arg_name_node and not isinstance(arg_name_node, str) else arg_name_node
                if not arg_name:
                    continue
                if arg_name == 'limit':
                    rel_cfg['limit'] = self._value_from_ast(arg.value)
                elif arg_name == 'offset':
                    rel_cfg['offset'] = self._value_from_ast(arg.value)
                else:
                    rel_cfg['filter_args'][arg_name] = self._value_from_ast(arg.value)
            # scalar subfields
            if getattr(child, 'selection_set', None):
                tgt_b = self.registry.types.get(fdef.meta.get('target')) if fdef.meta.get('target') else None
                sub_scalars = []
                for sub in getattr(child.selection_set, 'selections', []) or []:
                    sub_name_node = getattr(sub, 'name', None)
                    sub_name = getattr(sub_name_node, 'value', None) if sub_name_node and not isinstance(sub_name_node, str) else sub_name_node
                    if not sub_name:
                        continue
                    sub_def = getattr(tgt_b, '__berry_fields__', {}).get(sub_name) if tgt_b else None
                    if not sub_def or sub_def.kind == 'scalar':
                        sub_scalars.append(sub_name)
                if sub_scalars:
                    for s in sub_scalars:
                        if s not in rel_cfg['fields']:
                            rel_cfg['fields'].append(s)
                # recurse deeper
                try:
                    self._walk_ast(getattr(child, 'selection_set', None), self.registry.types.get(fdef.meta.get('target')), out)
                except Exception:
                    pass

    def extract(self, info: 'StrawberryInfo', root_field_name: str, btype_cls) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        # strawberry selected_fields
        try:
            if info and getattr(info, 'selected_fields', None):
                for root_sel in [sf for sf in info.selected_fields if getattr(sf, 'name', None) == root_field_name]:
                    self._walk_selected(root_sel, btype_cls, out)
        except Exception:
            pass
        # AST fallback
        try:
            raw_info = getattr(info, '_raw_info', None)
            if raw_info and hasattr(raw_info, 'field_nodes'):
                need_ast = (not out) or any((cfg.get('limit') is None and cfg.get('offset') is None and not cfg.get('filter_args')) for cfg in out.values())
                if need_ast:
                    for fn in raw_info.field_nodes:
                        name_node = getattr(fn, 'name', None)
                        root_name = getattr(name_node, 'value', None) if name_node and not isinstance(name_node, str) else name_node
                        if root_name == root_field_name:
                            self._walk_ast(getattr(fn, 'selection_set', None), btype_cls, out)
        except Exception:
            pass
        return out

# --- Filtering DSL (Phase 1: foundational pieces) ---

@dataclass
class FilterSpec:
    """Represents a declared filter argument.

    Forms supported (early phase):
      - Column + single op: FilterSpec(column='name', op='ilike')
      - Column + multiple ops (suffix form): FilterSpec(column='created_at', ops=['gt','lt'])
      - Callable builder (dynamic): FilterSpec(builder=lambda model_cls, info, value: ...)

    In later phases this will also drive GraphQL argument autogeneration.
    For Phase 1, values are supplied via context (see root resolver notes).
    """
    column: Optional[str] = None
    op: Optional[str] = None
    ops: Optional[List[str]] = None
    transform: Optional[Callable[[Any], Any]] = None
    builder: Optional[Callable[..., Any]] = None  # signature: (model_cls, info, value) -> SQLA expression
    alias: Optional[str] = None  # future: override argument name
    required: bool = False
    description: Optional[str] = None

    def clone_with(self, **updates) -> 'FilterSpec':  # pragma: no cover - trivial helper
        d = self.__dict__.copy()
        d.update(updates)
        return FilterSpec(**d)


def ColumnFilter(column: str, *, op: Optional[str] = None, ops: Optional[List[str]] = None, transform: Optional[Callable[[Any], Any]] = None, **extras) -> FilterSpec:
    return FilterSpec(column=column, op=op, ops=ops, transform=transform, **extras)


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


def relation(target: Any = None, *, single: bool | None = None, mode: str | None = None, **meta) -> FieldDescriptor:
    """Define a relation field.

    target can be a BerryType subclass or its string name. single/mode selects cardinality.
    """
    m = dict(meta)
    if target is not None:
        m['target'] = target.__name__ if hasattr(target, '__name__') and not isinstance(target, str) else target
    if single is not None:
        m['single'] = single
    if mode is not None:
        m['mode'] = mode
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
                    is_single = fdef.meta.get('single') or fdef.meta.get('mode') == 'single'
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
                    def _make_relation_resolver(meta_copy=meta_copy, is_single_value=is_single, fname_local=fname):
                        target_filters = _collect_declared_filters_for_target(meta_copy.get('target')) if meta_copy.get('target') else {}
                        # Build dynamic resolver with filter args + limit/offset
                        # Determine python types for target columns (if available) for future use (not required for arg defs now)
                        async def _impl(self, info: StrawberryInfo, limit: Optional[int], offset: Optional[int], _filter_args: Dict[str, Any]):
                            prefetch_attr = f'_{fname_local}_prefetched'
                            if hasattr(self, prefetch_attr):
                                # Reuse prefetched always (root pushdown included pagination already)
                                return getattr(self, prefetch_attr)
                            target_name_i = meta_copy.get('target')
                            target_cls_i = self.__berry_registry__._st_types.get(target_name_i)
                            parent_model = getattr(self, '_model', None)
                            if not target_cls_i or parent_model is None:
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
                                for col in parent_model.__table__.columns:
                                    if col.name.endswith('_id') and col.foreign_keys:
                                        for fk in col.foreign_keys:
                                            if fk.column.table.name == child_model_cls.__table__.name:
                                                candidate_fk_val = getattr(parent_model, col.name)
                                                break
                                    if candidate_fk_val is not None:
                                        break
                                if candidate_fk_val is None:
                                    return None
                                # Apply filters via query if any filter args passed
                                if any(v is not None for v in _filter_args.values()):
                                    from sqlalchemy import select as _select
                                    stmt = _select(child_model_cls).where(getattr(child_model_cls, 'id') == candidate_fk_val)
                                    # Add filter where clauses
                                    for arg_name, val in _filter_args.items():
                                        if val is None:
                                            continue
                                        f_spec = target_filters.get(arg_name)
                                        if not f_spec:
                                            continue
                                        expr = None
                                        if f_spec.transform:
                                            try:
                                                val = f_spec.transform(val)
                                            except Exception:
                                                continue
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
                            for col in child_model_cls.__table__.columns:
                                for fk in col.foreign_keys:
                                    if fk.column.table.name == parent_model.__table__.name:
                                        fk_col = col
                                        break
                                if fk_col is not None:
                                    break
                            if fk_col is None:
                                return []
                            from sqlalchemy import select as _select
                            stmt = _select(child_model_cls).where(fk_col == getattr(parent_model, 'id'))
                            # Apply filters
                            if target_filters:
                                for arg_name, val in _filter_args.items():
                                    if val is None:
                                        continue
                                    f_spec = target_filters.get(arg_name)
                                    if not f_spec:
                                        continue
                                    orig_val = val
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
                                        stmt = stmt.where(expr)
                            if offset:
                                stmt = stmt.offset(offset)
                            if limit is not None:
                                stmt = stmt.limit(limit)
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
                        params = 'self, info, limit=None, offset=None'
                        if arg_defs:
                            params += ', ' + ', '.join(arg_defs)
                        fname_inner = f"_rel_{fname_local}_resolver"
                        src = f"async def {fname_inner}({params}):\n" \
                              f"    _fa={{}}\n"
                        for a in target_filters.keys():
                            src += f"    _fa['{a}']={a}\n"
                        src += "    return await _impl(self, info, limit, offset, _fa)\n"
                        env: Dict[str, Any] = {'_impl': _impl}
                        exec(src, env)
                        fn = env[fname_inner]
                        if not getattr(fn, '__module__', None):  # ensure module for strawberry introspection
                            fn.__module__ = __name__
                        # annotations
                        anns: Dict[str, Any] = {'info': StrawberryInfo, 'limit': Optional[int], 'offset': Optional[int]}
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
                    def _make_custom_resolver(meta_copy=meta_copy):
                        async def custom_resolver(self, info: StrawberryInfo):  # noqa: D401
                            # Fast-path: if root query already populated attribute (no N+1), return it
                            pre_value = getattr(self, fname, None)
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
                    def _make_custom_obj_resolver(meta_copy=meta_copy):
                        async def _resolver(self, info: StrawberryInfo):  # noqa: D401
                            pre_v = getattr(self, f"_{fname}_data", None)
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

                async def _base_impl(info: StrawberryInfo, limit: int | None, offset: int | None, _passed_filter_args: Dict[str, Any]):
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
                        inner_cols = [getattr(child_model_cls, c) for c in requested_scalar] if requested_scalar else [getattr(child_model_cls, 'id')]
                        inner_sel = select(*inner_cols).select_from(child_model_cls).where(fk_col == parent_model_cls.id).correlate(parent_model_cls)
                        try:
                            if 'id' in child_model_cls.__table__.columns:
                                inner_sel = inner_sel.order_by(getattr(child_model_cls, 'id'))
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
                            g_sel = select(*g_inner_cols).select_from(grand_model_cls).where(g_fk == getattr(child_model_cls, 'id')).correlate(child_model_cls, parent_model_cls)
                            try:
                                if 'id' in grand_model_cls.__table__.columns:
                                    g_sel = g_sel.order_by(getattr(grand_model_cls, 'id'))
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
                                    continue  # skip complex builders for pushdown
                            except Exception:
                                continue
                            if expr_sel is None:
                                continue
                            # Expect a Select with multiple labeled columns
                            try:
                                from sqlalchemy.sql import Select as _Select  # type: ignore
                            except Exception:
                                _Select = None  # type: ignore
                            if _Select is not None and isinstance(expr_sel, _Select):
                                try:
                                    sel_cols = list(getattr(expr_sel, 'selected_columns', []))  # type: ignore[attr-defined]
                                except Exception:
                                    sel_cols = []
                                col_labels: List[str] = []
                                for col in sel_cols:
                                    try:
                                        labeled = col
                                        col_name = getattr(labeled, 'name', None) or getattr(labeled, 'key', None)
                                        if not col_name:
                                            col_name = f"{cf_name}_{len(col_labels)}"
                                            labeled = col.label(col_name)
                                        # Build correlated scalar subquery
                                        subq = select(labeled)
                                        # add FROM / WHERE of original
                                        try:
                                            for _from in expr_sel.get_final_froms():  # type: ignore[attr-defined]
                                                subq = subq.select_from(_from)
                                        except Exception:
                                            pass
                                        for _w in getattr(expr_sel, '_where_criteria', []):  # type: ignore[attr-defined]
                                            subq = subq.where(_w)
                                        if hasattr(subq, 'scalar_subquery'):
                                            subq_expr = subq.scalar_subquery().label(col_name)
                                        else:
                                            subq_expr = subq.label(col_name)
                                        select_columns.append(subq_expr)
                                        col_labels.append(col_name)
                                    except Exception:
                                        continue
                                if col_labels:
                                    custom_object_fields.append((cf_name, col_labels, cf_def.meta.get('returns')))
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
                                inner_cols_i = [getattr(child_model_cls_i, c) for c in requested_scalar_i] if requested_scalar_i else [getattr(child_model_cls_i, 'id')]
                                if mssql_mode:
                                    parent_table = parent_model_cls.__tablename__
                                    child_table = child_model_cls_i.__tablename__
                                    where_clause = f"[{child_table}].[{fk_col_i.name}] = [{parent_table}].[id]"
                                    order_clause = f"[{child_table}].[id]" if 'id' in child_model_cls_i.__table__.columns else None
                                    return adapter.build_list_relation_json(
                                        child_table=child_table,
                                        projected_columns=requested_scalar_i,
                                        where_condition=where_clause,
                                        limit=rel_cfg_local.get('limit'),
                                        order_by=order_clause,
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
                                    if 'id' in child_model_cls_i.__table__.columns:
                                        inner_sel_i = inner_sel_i.order_by(getattr(child_model_cls_i, 'id'))
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
                    for arg_name, value in _passed_filter_args.items():
                        if value is None:
                            continue
                        f_spec = declared_filters.get(arg_name)
                        if not f_spec:
                            continue
                        # transform value
                        if f_spec.transform:
                            try:
                                value = f_spec.transform(value)
                            except Exception:
                                continue
                        expr = None
                        if f_spec.builder:
                            try:
                                expr = f_spec.builder(model_cls, info, value)
                            except Exception:
                                expr = None
                        elif f_spec.column:
                            try:
                                col = model_cls.__table__.c.get(f_spec.column)
                            except Exception:
                                col = None
                            if col is None:
                                continue
                            op_name = f_spec.op or 'eq'
                            op_fn = OPERATOR_REGISTRY.get(op_name)
                            if not op_fn:
                                continue
                            try:
                                expr = op_fn(col, value)
                            except Exception:
                                expr = None
                        if expr is not None:
                            where_clauses.append(expr)
                    if where_clauses:
                        for wc in where_clauses:
                            try:
                                stmt = stmt.where(wc)
                            except Exception:
                                pass
                    if offset:
                        stmt = stmt.offset(offset)
                    if limit is not None:
                        stmt = stmt.limit(limit)
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
                            # reconstruct custom object fields (all scalar subqueries executed in SELECT)
                            if custom_object_fields:
                                row_mapping = {}
                                full_row = sa_rows[row_index]
                                try:
                                    for k in getattr(full_row, '_mapping').keys():
                                        row_mapping[k] = full_row._mapping[k]
                                except Exception:
                                    for i, v in enumerate(full_row):
                                        row_mapping[f'_col_{i}'] = v
                                for cf_name, col_labels, returns_spec in custom_object_fields:
                                    data_dict = {}
                                    if isinstance(returns_spec, dict):
                                        for k2 in returns_spec.keys():
                                            if k2 in row_mapping:
                                                data_dict[k2] = row_mapping[k2]
                                    nested_type_name = f"{btype_cls.__name__}_{cf_name}_Type"
                                    nested_type = self._st_types.get(nested_type_name)
                                    if nested_type and data_dict:
                                        try:
                                            obj = nested_type(**data_dict)
                                        except Exception:
                                            obj = None
                                    else:
                                        obj = data_dict or None
                                    setattr(inst, f"_{cf_name}_data", obj)
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
                                                parsed_value = None if is_single else []
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
                                                            setattr(child_inst, sf, parsed_value.get(sf))
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
                                                            for sf, sdef in target_b.__berry_fields__.items():
                                                                if sdef.kind == 'scalar':
                                                                    setattr(child_inst, sf, item.get(sf))
                                                            setattr(child_inst, '_model', None)
                                                            tmp_list.append(child_inst)
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
                # Build parameter list: info, limit, offset, filter args
                if args_str:
                    full_params = f"info, limit=None, offset=None, {args_str}"
                else:
                    full_params = "info, limit=None, offset=None"
                src = f"async def {func_name}({full_params}):\n" \
                      f"    _fa = {{}}\n"  # gather passed filter args
                for a in declared_filters.keys():
                    src += f"    _fa['{a}'] = {a} if '{a}' in locals() else None\n"
                src += "    return await _base_impl(info, limit, offset, _fa)\n"
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
                ann: Dict[str, Any] = {'info': StrawberryInfo, 'limit': Optional[int], 'offset': Optional[int]}
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
