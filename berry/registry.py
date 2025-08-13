from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, get_type_hints
import asyncio
import strawberry
from sqlalchemy import select
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.sqltypes import Integer, String, Boolean, DateTime
from datetime import datetime
try:  # Provide StrawberryInfo for type annotations
    from strawberry.types import Info as StrawberryInfo  # type: ignore
except Exception:  # pragma: no cover
    class StrawberryInfo:  # type: ignore
        ...

T = TypeVar('T')

@dataclass
class FieldDef:
    name: str
    resolver: Optional[Callable] = None
    kind: str = 'scalar'  # scalar | relation | aggregate | custom | custom_object
    target: Optional[str] = None  # relation target type name
    meta: Dict[str, Any] = dc_field(default_factory=dict)

class FieldDescriptor:
    def __init__(self, *, kind: str = 'scalar', **meta):
        self.kind = kind
        self.meta = meta
        self.name: Optional[str] = None

    def __set_name__(self, owner, name):  # noqa: D401
        self.name = name

    def build(self, owner_name: str) -> FieldDef:
        return FieldDef(name=self.name, kind=self.kind, meta=self.meta)

def field(**meta) -> FieldDescriptor:
    return FieldDescriptor(kind='scalar', **meta)

def relation(target: str | Type | None = None, **meta) -> FieldDescriptor:
    m = dict(meta)
    if target is not None:
        if isinstance(target, str):
            m['target'] = target
        else:
            m['target'] = target.__name__
    # allow caller to specify single=True for belongs-to
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
                    def _make_relation_resolver(meta_copy=meta_copy, is_single_value=is_single):
                        async def relation_resolver(self, info: StrawberryInfo, limit: Optional[int] = None, offset: Optional[int] = None):  # noqa: D401
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
                        return relation_resolver
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
            def _make_root_resolver(model_cls, st_cls, btype_cls):
                async def _resolver(info: StrawberryInfo, limit: int | None = None, offset: int | None = None):  # noqa: D401
                    session = info.context.get('db_session') if info and info.context else None
                    if session is None:
                        return []
                    # Acquire per-context lock to avoid concurrent AsyncSession use (esp. MSSQL/pyodbc limitations)
                    lock = info.context.setdefault('_berry_db_lock', asyncio.Lock())
                    # Collect custom field expressions for pushdown
                    custom_fields: List[tuple[str, Any]] = []
                    custom_object_fields: List[tuple[str, List[str], Any]] = []  # (field, column_labels, returns_spec)
                    select_columns: List[Any] = [model_cls]
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
                    stmt = select(*select_columns)
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
                            out.append(inst)
                        # Batch aggregate counts if any count aggregate fields defined
                        agg_fields = [ (fname, fdef) for fname, fdef in btype_cls.__berry_fields__.items() if fdef.kind == 'aggregate' and (fdef.meta.get('op') == 'count' or 'count' in fdef.meta.get('ops', [])) ]
                        if agg_fields and rows:
                            parent_ids = [getattr(r, 'id') for r in rows]
                            from sqlalchemy import select as _select, func as _func
                            for agg_name, agg_def in agg_fields:
                                source_rel = agg_def.meta.get('source')
                                rel_def = btype_cls.__berry_fields__.get(source_rel)
                                if not rel_def or rel_def.kind != 'relation':
                                    continue
                                target_name = rel_def.meta.get('target')
                                target_btype = self.types.get(target_name)
                                if not target_btype or not target_btype.model:
                                    continue
                                child_model_cls = target_btype.model
                                # find fk col referencing parent
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
                                stmt_cnt = _select(fk_col, _func.count().label('c')).select_from(child_model_cls).where(fk_col.in_(parent_ids)).group_by(fk_col)
                                res_cnt = await session.execute(stmt_cnt)
                                map_counts = {pid: cnt for pid, cnt in res_cnt.all()}
                                cache_key = agg_def.meta.get('cache_key') or source_rel + ':count'
                                for inst in out:
                                    pid = getattr(inst._model, 'id')
                                    cache = getattr(inst, '_agg_cache', None)
                                    if cache is None:
                                        cache = {}
                                        setattr(inst, '_agg_cache', cache)
                                    cache[cache_key] = map_counts.get(pid, 0)
                    return out
                return _resolver
            root_resolver = _make_root_resolver(_model_cls, st_type, bcls)
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
