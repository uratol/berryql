from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, get_type_hints
import strawberry
from sqlalchemy import select
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.sqltypes import Integer, String, Boolean, DateTime
from datetime import datetime
try:
    from strawberry.types import Info as StrawberryInfo  # type: ignore
except Exception:  # pragma: no cover
    StrawberryInfo = Any  # fallback

T = TypeVar('T')

@dataclass
class FieldDef:
    name: str
    resolver: Optional[Callable] = None
    kind: str = 'scalar'  # scalar | relation | aggregate
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
                        async def relation_resolver(self, info, limit: Optional[int] = None, offset: Optional[int] = None):  # noqa: D401
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
                    if is_count:
                        annotations[fname] = Optional[int]
                    else:
                        annotations[fname] = Optional[str]
                    meta_copy = dict(fdef.meta)
                    # assign cache key for count aggregates
                    if is_count:
                        meta_copy['cache_key'] = meta_copy.get('cache_key') or fdef.meta.get('source') + ':count'
                    def _make_aggregate_resolver(meta_copy=meta_copy, bcls_local=bcls):
                        async def aggregate_resolver(self, info):  # noqa: D401
                            cache = getattr(self, '_agg_cache', None)
                            is_count_local = meta_copy.get('op') == 'count' or 'count' in meta_copy.get('ops', [])
                            if is_count_local and cache is not None:
                                key = meta_copy.get('cache_key') or (meta_copy.get('source') + ':count')
                                if key in cache:
                                    print(f"DEBUG aggregate_resolver cache hit for {key}: {cache[key]}")
                                    return cache[key]
                            parent_model = getattr(self, '_model', None)
                            if parent_model is None:
                                print("DEBUG aggregate_resolver: no parent_model")
                                return 0 if is_count_local else None
                            session = getattr(info.context, 'get', lambda k, d=None: info.context[k])('db_session', None) if info and info.context else None
                            if session is None:
                                print("DEBUG aggregate_resolver: no session")
                                return 0 if is_count_local else None
                            source = meta_copy.get('source')
                            rel_def = bcls_local.__berry_fields__.get(source) if source else None
                            if not rel_def or rel_def.kind != 'relation':
                                print(f"DEBUG aggregate_resolver: invalid rel_def for source {source}")
                                return 0 if is_count_local else None
                            target_name = rel_def.meta.get('target')
                            target_btype = self.__berry_registry__.types.get(target_name) if target_name else None
                            if not target_btype or not target_btype.model:
                                print(f"DEBUG aggregate_resolver: target_btype/model missing for {target_name}")
                                return 0 if is_count_local else None
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
                                print("DEBUG aggregate_resolver: fk_col not found")
                                return 0 if is_count_local else None
                            if is_count_local:
                                from sqlalchemy import func, select as _select
                                stmt = _select(func.count()).select_from(child_model_cls).where(fk_col == getattr(parent_model, 'id'))
                                result = await session.execute(stmt)
                                val = result.scalar_one() or 0
                                print(f"DEBUG aggregate_resolver direct count for parent {getattr(parent_model,'id',None)} = {val}")
                                key = meta_copy.get('cache_key') or (source + ':count')
                                if cache is None:
                                    cache = {}
                                    setattr(self, '_agg_cache', cache)
                                cache[key] = val
                                return val
                            return None
                        return aggregate_resolver
                    setattr(st_cls, fname, strawberry.field(_make_aggregate_resolver()))
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
                async def _resolver(info, limit: int | None = None, offset: int | None = None):  # noqa: D401
                    session = info.context.get('db_session') if info and info.context else None
                    if session is None:
                        return []
                    stmt = select(model_cls)
                    if offset:
                        stmt = stmt.offset(offset)
                    if limit is not None:
                        stmt = stmt.limit(limit)
                    result = await session.execute(stmt)
                    rows = [r[0] for r in result.all()]
                    out = []
                    for row in rows:
                        inst = st_cls()
                        setattr(inst, '_model', row)
                        # hydrate scalar fields
                        for sf, sdef in btype_cls.__berry_fields__.items():
                            if sdef.kind == 'scalar':
                                try:
                                    setattr(inst, sf, getattr(row, sf, None))
                                except Exception:
                                    pass
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
                            print(f"DEBUG batch aggregate for field {agg_name} counts map: {map_counts}")
                            cache_key = agg_def.meta.get('cache_key') or source_rel + ':count'
                            for inst in out:
                                pid = getattr(inst._model, 'id')
                                cache = getattr(inst, '_agg_cache', None)
                                if cache is None:
                                    cache = {}
                                    setattr(inst, '_agg_cache', cache)
                                cache[cache_key] = map_counts.get(pid, 0)
                                print(f"DEBUG batch cache set {cache_key} for parent {pid} = {cache[cache_key]}")
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
