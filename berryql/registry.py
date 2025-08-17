from __future__ import annotations
from dataclasses import dataclass, field as dc_field
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, get_type_hints
import asyncio
import strawberry
from sqlalchemy import select, func, text as _text
from sqlalchemy import and_ as _and
from .adapters import get_adapter  # adapter abstraction
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import load_only
from sqlalchemy.sql.sqltypes import Integer, String, Boolean, DateTime
from sqlalchemy.types import TypeDecorator as _SATypeDecorator
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY as PG_ARRAY, JSONB as PG_JSONB
import uuid as _py_uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from .core.utils import get_db_session as _get_db
if TYPE_CHECKING:  # pragma: no cover - type checking only
    class Registry: ...  # forward ref placeholder

# Silence Strawberry's LazyType deprecation warnings to keep test output clean.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"LazyType is deprecated.*")

# Project logger
_logger = logging.getLogger("berryql")
try:  # Provide StrawberryInfo for type annotations
    from strawberry.types import Info as StrawberryInfo  # type: ignore
except Exception:  # pragma: no cover
    class StrawberryInfo:  # type: ignore
        ...
try:  # Provide StrawberryConfig for type annotations (optional)
    from strawberry.schema.config import StrawberryConfig  # type: ignore
except Exception:  # pragma: no cover
    class StrawberryConfig:  # type: ignore
        def __init__(self, auto_camel_case: bool | None = None):
            self.auto_camel_case = auto_camel_case

T = TypeVar('T')

# DRY split: import core building blocks
from .core.fields import FieldDef, FieldDescriptor, field, relation, aggregate, count, custom, custom_object, DomainDescriptor
from .core.filters import FilterSpec, OPERATOR_REGISTRY, register_operator, normalize_filter_spec as _normalize_filter_spec
from .core.selection import RelationSelectionExtractor, RootSelectionExtractor
from .core.analyzer import QueryAnalyzer
from .core.utils import (
    Direction,
    dir_value as _dir_value,
    coerce_where_value as _coerce_where_value,
    coerce_literal as _coerce_literal,
    normalize_relation_cfg as _normalize_rel_cfg,
    expr_from_where_dict as _expr_from_where_dict,
    to_where_dict as _to_where_dict,
    normalize_order_multi_values as _norm_order_multi,
)
from .sql.builders import RelationSQLBuilders, RootSQLBuilders
from .core.hydration import Hydrator

__all__ = ['BerrySchema', 'BerryType', 'BerryDomain']

# --- Helper: Relation selection extraction (moved out of resolver closure) ---
## RelationSelectionExtractor imported from core.selection

## RootSelectionExtractor imported from core.selection
# Global operator registry (extensible) imported from core.filters

# Simple filter spec container used to define and expand filter arguments imported from core.filters

# Coerce JSON where values (or filter arg values) helper imported from core.utils as _coerce_where_value

# Ordering direction enum for GraphQL imported as Direction; dir normalization via _dir_value


# Filter spec normalizer imported from core.filters as _normalize_filter_spec



# Field primitives imported from core.fields

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

class BerryDomain:
        """Marker base for a domain group. Holds relation FieldDescriptors and optional strawberry fields.

        Attributes (set by schema.domain decorator meta):
            __domain_name__: the GraphQL field name in Query (string)
            __domain_guard__: optional callable (model_cls, info) -> dict|SA expr applied to all relations as default where
        """
        pass

class BerrySchema:
    """Registry + dynamic Strawberry schema builder.
    """
    def __init__(self):
        self.types: Dict[str, Type[BerryType]] = {}
        self._st_types: Dict[str, Any] = {}
        # Optional: user-defined root Query fields (via @berry_schema.query)
        self._root_query_fields = None
        # Keep reference to user-declared Query class to copy @strawberry.field methods
        self._user_query_cls = None
        # Keep references to user-declared Mutation and Subscription classes
        self._user_mutation_cls = None
        self._user_subscription_cls = None
        # Domain registry: name -> (domain class, options)
        self._domains = {}

    def register(self, cls: Type[BerryType]):
        self.types[cls.__name__] = cls
        return cls

    def type(self, *, model: Optional[Type] = None):
        def deco(cls: Type[BerryType]):
            cls.model = model
            return self.register(cls)
        return deco

    def mutation(self):
        """Decorator to declare the root Mutation using Strawberry fields.

        Example:
            @berry_schema.mutation()
            class Mutation:
                @strawberry.mutation
                async def do_something(...): ...
        """
        def deco(cls: Type[Any]):
            # Simply store reference; fields are copied during schema build
            self._user_mutation_cls = cls
            return cls
        return deco

    def subscription(self):
        """Decorator to declare the root Subscription using Strawberry subscription fields.

        Example:
            @berry_schema.subscription()
            class Subscription:
                @strawberry.subscription
                async def tick(...): ...
        """
        def deco(cls: Type[Any]):
            # Simply store reference; fields are copied during schema build
            self._user_subscription_cls = cls
            return cls
        return deco

    def query(self):
        """Decorator to declare the root Query using FieldDescriptors.

        Example:
            @berry_schema.query
            class Query:
                users = relation('UserQL', where=..., order_by='id')
                userById = relation('UserQL', single=True, where=...)
        """
        def deco(cls: Type[Any]):
            qfields: Dict[str, FieldDef] = {}
            for k, v in list(vars(cls).items()):
                if isinstance(v, FieldDescriptor):
                    v.__set_name__(None, k)
                    qfields[k] = v.build('Query')
                # Capture DomainDescriptor markers for exposure later
                if isinstance(v, DomainDescriptor):
                    v.__set_name__(None, k)
                    dom_cls = v.domain_cls
                    dom_name = v.name or v.attr_name or getattr(dom_cls, '__domain_name__', None) or k
                    # Persist the chosen field name and class for build time
                    self._domains.setdefault(dom_name, {'class': dom_cls, 'expose': True, 'options': dict(v.meta)})
            self._root_query_fields = qfields
            # Save user-declared class for copying strawberry fields later
            self._user_query_cls = cls
            return cls
        return deco

    def domain(self, *, name: Optional[str] = None, guard: Optional[Callable[..., Any]] = None, description: Optional[str] = None):
        """Decorator to register a domain class.

        Example:
            @berry_schema.domain(name='userDomain', guard=...)
            class UserDomain(BerryDomain):
                users = relation('UserQL')
        """
        def deco(cls: Type[BerryDomain]):
            dom_name = name or getattr(cls, '__name__', 'Domain')
            setattr(cls, '__domain_name__', dom_name)
            if guard is not None:
                setattr(cls, '__domain_guard__', guard)
            if description is not None:
                setattr(cls, '__doc__', description)
            # Register placeholder; exposure onto Query happens when Query decorator processes DomainDescriptors
            self._domains.setdefault(dom_name, {'class': cls, 'expose': False, 'options': {}})
            return cls
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

    # ---------- DRY helpers ----------
    def _sa_python_type(self, sqlatype: Any) -> Any:
        """Map a SQLAlchemy column type to a Python (annotation) type.

        Handles common primitives and PostgreSQL-specific wrappers (UUID, ARRAY, JSONB).
        Defaults to str for unknown types (safe GraphQL scalar mapping).
        """
        try:
            # Unwrap TypeDecorator when possible and special-case BinaryBlob/BinaryArray
            if isinstance(sqlatype, _SATypeDecorator):
                # Special-case our BinaryArray -> List[str]
                try:
                    if type(sqlatype).__name__.lower() == 'binaryarray':
                        return List[str]  # type: ignore[index]
                except Exception:
                    pass
                # Special-case our BinaryBlob -> str
                try:
                    if type(sqlatype).__name__.lower() == 'binaryblob':
                        return str
                except Exception:
                    pass
                # Try to recurse into underlying impl
                try:
                    inner_impl = getattr(sqlatype, 'impl', None)
                    if inner_impl is not None and inner_impl is not sqlatype:
                        t = self._sa_python_type(inner_impl)
                        if t is not None:
                            return t
                except Exception:
                    pass
            if isinstance(sqlatype, Integer):
                return int
            if isinstance(sqlatype, String):
                return str
            if isinstance(sqlatype, Boolean):
                return bool
            if isinstance(sqlatype, DateTime):
                return datetime
            if isinstance(sqlatype, PG_UUID):
                return _py_uuid.UUID
            if isinstance(sqlatype, PG_ARRAY):
                inner = getattr(sqlatype, 'item_type', None)
                inner_t = self._sa_python_type(inner) if inner is not None else str
                # guard List typing for non-subscriptable case
                try:
                    return List[inner_t]  # type: ignore[index]
                except Exception:
                    return list
            if isinstance(sqlatype, PG_JSONB):
                return str
        except Exception:
            pass
        return str

    def _get_pk_name(self, model_cls: Any) -> str:
        """Return the primary key column name for a SQLAlchemy ORM model.

        Raises ValueError if no primary key column can be determined.
        """
        try:
            if model_cls is not None and hasattr(model_cls, "__table__"):
                try:
                    pk_cols = list(getattr(model_cls.__table__, "primary_key").columns)
                except Exception:
                    pk_cols = [c for c in model_cls.__table__.columns if getattr(c, "primary_key", False)]
                if pk_cols:
                    # Prefer the first PK column (single-column PK expected)
                    return pk_cols[0].name
        except Exception:
            pass
        raise ValueError(f"Primary key column not found for model: {getattr(model_cls, '__name__', model_cls)}")

    def _get_pk_column(self, model_cls: Any):
        """Return the SQLAlchemy Column object for the model's primary key.

        Raises ValueError if not found.
        """
        pk_name = self._get_pk_name(model_cls)
        try:
            # Prefer Column from model.__table__.c to keep compatibility with label()/order_by()
            col = getattr(model_cls.__table__, 'c', None)
            if col is not None:
                cobj = col.get(pk_name)
                if cobj is not None:
                    return cobj
        except Exception:
            pass
        try:
            cobj = getattr(model_cls, pk_name, None)
            if cobj is not None:
                return cobj
        except Exception:
            pass
        # If the name resolved but attribute is missing, still raise to fail fast
        raise ValueError(f"Primary key column object not accessible for model: {getattr(model_cls, '__name__', model_cls)}")

    def _build_column_type_map(self, model_cls: Any) -> Dict[str, Any]:
        """Return a mapping of column name -> Python type for the given ORM model."""
        out: Dict[str, Any] = {}
        try:
            if model_cls is not None and hasattr(model_cls, '__table__'):
                for col in model_cls.__table__.columns:
                    out[col.name] = self._sa_python_type(getattr(col, 'type', None))
        except Exception:
            pass
        return out

    def _expand_filter_args(self, arg_spec: Optional[Dict[str, Any]]) -> Dict[str, 'FilterSpec']:
        """Normalize a relation arguments specification to concrete FilterSpecs keyed by arg name.

        - Supports single op via .op and multi-op via .ops with suffixes.
        - Accepts raw values compatible with normalize_filter_spec, or callables as builder-only specs.
        """
        if not isinstance(arg_spec, dict):
            return {}
        expanded: Dict[str, FilterSpec] = {}
        for key, raw in arg_spec.items():
            try:
                if callable(raw):
                    spec = FilterSpec(builder=raw)
                else:
                    spec = _normalize_filter_spec(raw)
            except Exception:
                continue
            if spec.ops and not spec.op:
                for op_name in spec.ops:
                    base = spec.alias or key
                    an = base if base.endswith(f"_{op_name}") else f"{base}_{op_name}"
                    expanded[an] = spec.clone_with(op=op_name, ops=None)
            else:
                expanded[spec.alias or key] = spec
        return expanded

    def _normalize_order_multi_values(self, multi: Any) -> List[str]:
        """Normalize a potentially heterogeneous order_multi list to a list[str] of 'col:dir'."""
        return _norm_order_multi(multi)

    def _find_parent_fk_column_name(self, parent_model_cls: Any, child_model_cls: Any, rel_name: str) -> Optional[str]:
        """Return the name of the FK column on parent that references child, or a conventional '<rel>_id' fallback.

        Tries model metadata first (parent __table__ foreign_keys targeting child's table).
        Falls back to '<rel>_id' when present. Returns None when not found.
        """
        # Prefer model metadata foreign key discovery
        try:
            if parent_model_cls is not None and hasattr(parent_model_cls, '__table__') and \
               child_model_cls is not None and hasattr(child_model_cls, '__table__'):
                for pc in parent_model_cls.__table__.columns:
                    for fk in pc.foreign_keys:
                        try:
                            if fk.column.table.name == child_model_cls.__table__.name:
                                return pc.name
                        except Exception:
                            continue
        except Exception:
            pass
        # Conventional fallback: '<relation>_id'
        fallback = f"{rel_name}_id"
        try:
            if parent_model_cls is not None:
                # check via table columns when possible
                if hasattr(parent_model_cls, '__table__'):
                    try:
                        if any(c.name == fallback for c in parent_model_cls.__table__.columns):
                            return fallback
                    except Exception:
                        pass
                # lenient: hasattr on ORM class attribute
                if hasattr(parent_model_cls, fallback):
                    return fallback
        except Exception:
            pass
        return None

    def _get_context_lock(self, info: 'StrawberryInfo') -> 'asyncio.Lock':
        """Return a per-request asyncio.Lock stored on info.context to serialize DB access when needed."""
        try:
            ctx = getattr(info, 'context', None)
        except Exception:
            ctx = None
        lock = None
        if isinstance(ctx, dict):
            lock = ctx.get('_berry_db_lock')
            if lock is None:
                lock = asyncio.Lock()
                try:
                    ctx['_berry_db_lock'] = lock
                except Exception:
                    pass
        elif ctx is not None:
            try:
                lock = getattr(ctx, '_berry_db_lock')
            except Exception:
                lock = None
            if lock is None:
                lock = asyncio.Lock()
                try:
                    setattr(ctx, '_berry_db_lock', lock)
                except Exception:
                    pass
        if lock is None:
            lock = asyncio.Lock()
        return lock

    def to_strawberry(self, *, strawberry_config: Optional[StrawberryConfig] = None):
        # Persist naming behavior for extractor logic
        try:
            # Prefer explicit name_converter when available
            if strawberry_config is not None and hasattr(strawberry_config, 'name_converter'):
                self._name_converter = getattr(strawberry_config, 'name_converter')
                # Derive auto_camel_case hint from converter when present
                self._auto_camel_case = bool(getattr(self._name_converter, 'auto_camel_case', False))
            else:
                # Fallback to auto_camel_case on config if exposed (older Strawberry)
                if strawberry_config is not None and hasattr(strawberry_config, 'auto_camel_case'):
                    self._auto_camel_case = bool(getattr(strawberry_config, 'auto_camel_case'))
                else:
                    self._auto_camel_case = False
                self._name_converter = None
        except Exception:
            self._auto_camel_case = False
            self._name_converter = None
        # Always rebuild Strawberry runtime classes fresh per call to honor config changes
        # and avoid stale definitions across multiple to_strawberry invocations.
        self._st_types = {}
        # Two-pass: create plain classes first
        for name, bcls in self.types.items():
            base_namespace = {'__berry_registry__': self, '__doc__': f'Berry runtime type {name}'}
            cls = type(name, (), base_namespace)
            cls.__module__ = __name__
            self._st_types[name] = cls
    # Second pass: add fields & annotations before decoration
        for name, bcls in self.types.items():
            st_cls = self._st_types[name]
            annotations: Dict[str, Any] = getattr(st_cls, '__annotations__', {}) or {}
            # Also capture user-declared annotations on the BerryType subclass itself so we can
            # expose regular strawberry fields (with their own resolvers) alongside Berry fields.
            try:
                user_annotations: Dict[str, Any] = getattr(bcls, '__annotations__', {}) or {}
            except Exception:
                user_annotations = {}
            # column type mapping
            column_type_map: Dict[str, Any] = self._build_column_type_map(getattr(bcls, 'model', None))
            for fname, fdef in bcls.__berry_fields__.items():
                is_private = isinstance(fname, str) and fname.startswith('_')
                if hasattr(st_cls, fname):
                    # don't overwrite existing custom attr
                    pass
                if fdef.kind == 'scalar':
                    if is_private:
                        # Skip exposing private scalars
                        continue
                    # Prefer mapped source column for type inference when provided
                    try:
                        src_col = (fdef.meta or {}).get('column')
                    except Exception:
                        src_col = None
                    py_t = column_type_map.get(src_col or fname, str)
                    annotations[fname] = Optional[py_t]
                    setattr(st_cls, fname, None)
                elif fdef.kind == 'relation':
                    target_name = fdef.meta.get('target')
                    is_single = bool(fdef.meta.get('single'))
                    post_process = fdef.meta.get('post_process')
                    if target_name:
                        # Use actual class objects from registry to avoid global forward-ref collisions
                        target_cls_ref = self._st_types.get(target_name)
                        if target_cls_ref is not None:
                            if not is_private:
                                if is_single:
                                    annotations[fname] = Optional[target_cls_ref]  # type: ignore[index]
                                else:
                                    annotations[fname] = List[target_cls_ref]  # type: ignore[index]
                        else:
                            if not is_private:
                                # Use real typing objects instead of string annotations to avoid LazyType warnings
                                if is_single:
                                    annotations[fname] = Optional[str]
                                else:
                                    annotations[fname] = List[str]  # type: ignore[index]
                    else:  # fallback placeholder
                        if not is_private:
                            # Use real typing objects instead of string annotations to avoid LazyType warnings
                            if is_single:
                                annotations[fname] = Optional[str]
                            else:
                                annotations[fname] = List[str]  # type: ignore[index]
                    meta_copy = dict(fdef.meta)
                    def _collect_declared_filters_for_target(target_type_name: str):
                        # Deprecated path: type-level __filters__ removed; keep stub for parity
                        return {}
                    def _make_relation_resolver(meta_copy=meta_copy, is_single_value=is_single, fname_local=fname, parent_btype_local=bcls):
                        # Build filter specs exclusively from relation-specific arguments
                        target_filters: Dict[str, FilterSpec] = {}
                        # overlay relation-specific arguments
                        rel_args_spec = meta_copy.get('arguments')
                        if isinstance(rel_args_spec, dict):
                            target_filters.update(self._expand_filter_args(rel_args_spec))
                        # Build dynamic resolver with filter args + limit/offset
                        # Determine python types for target columns (if available) for future use (not required for arg defs now)
                        async def _impl(self, info: StrawberryInfo, limit: Optional[int], offset: Optional[int], order_by: Optional[str], order_dir: Optional[Any], order_multi: Optional[List[str]], related_where: Optional[Any], _filter_args: Dict[str, Any]):
                            prefetch_attr = f'_{fname_local}_prefetched'
                            # Simplified: no special handling for 'source' wrapper
                            if hasattr(self, prefetch_attr):
                                # If relation was prefetched via SQL pushdown, return it directly.
                                prefetched = getattr(self, prefetch_attr)
                                val = prefetched if is_single_value else list(prefetched or [])
                                # Apply python-side post-process if provided
                                try:
                                    pp = meta_copy.get('post_process')
                                except Exception:
                                    pp = None
                                if pp is not None:
                                    try:
                                        import inspect, asyncio
                                        res = pp(val, info)
                                        if inspect.isawaitable(res):
                                            res = await res
                                        return res
                                    except Exception:
                                        return val
                                return val
                            target_name_i = meta_copy.get('target')
                            target_cls_i = self.__berry_registry__._st_types.get(target_name_i)
                            parent_model = getattr(self, '_model', None)
                            if not target_cls_i:
                                return None if is_single_value else []
                            session = _get_db(info)
                            if session is None:
                                return None if is_single_value else []
                            target_btype = self.__berry_registry__.types.get(target_name_i)
                            if not target_btype or not target_btype.model:
                                return None if is_single_value else []
                            child_model_cls = target_btype.model
                            if is_single_value:
                                candidate_fk_val = None
                                fallback_parent_id = None
                                # Try helper '<relation>_id' on parent instance
                                try:
                                    candidate_fk_val = getattr(self, f"{fname_local}_id", None)
                                except Exception:
                                    candidate_fk_val = None
                                # Record parent id for potential child->parent fallback
                                try:
                                    if parent_model is not None:
                                        try:
                                            pk_name_parent = self.__berry_registry__._get_pk_name(parent_model.__class__)
                                            fallback_parent_id = getattr(parent_model, pk_name_parent, None)
                                        except Exception:
                                            fallback_parent_id = None
                                    else:
                                        fallback_parent_id = None
                                except Exception:
                                    fallback_parent_id = None
                                if fallback_parent_id is None:
                                    try:
                                        # Fallback to hydrated attribute 'id' when present on the Strawberry instance
                                        fallback_parent_id = getattr(self, 'id', None)
                                    except Exception:
                                        fallback_parent_id = None
                                if candidate_fk_val is None and parent_model is not None:
                                    # Derive FK from ORM model instance when available (parent has FK to child)
                                    for col in parent_model.__table__.columns:
                                        if col.name.endswith('_id') and col.foreign_keys:
                                            for fk in col.foreign_keys:
                                                if fk.column.table.name == child_model_cls.__table__.name:
                                                    candidate_fk_val = getattr(parent_model, col.name)
                                                    break
                                        if candidate_fk_val is not None:
                                            break
                                # If we didn't find a direct FK to child, try fetching the first child where child.parent_id == parent.id
                                if candidate_fk_val is None and fallback_parent_id is not None:
                                    from sqlalchemy import select as _select
                                    # Attempt to find an FK from child -> parent (common case)
                                    child_fk_to_parent = None
                                    for col in child_model_cls.__table__.columns:
                                        for fk in col.foreign_keys:
                                            try:
                                                if fk.column.table.name == (parent_model.__table__.name if parent_model is not None else None):
                                                    child_fk_to_parent = col
                                                    break
                                            except Exception:
                                                continue
                                        if child_fk_to_parent is not None:
                                            break
                                    if child_fk_to_parent is not None:
                                        # Build a select for the first child row for this parent
                                        stmt = _select(child_model_cls).where(child_fk_to_parent == fallback_parent_id)
                                        # Apply where/default_where and filter args similarly to list path
                                        if related_where is not None or meta_copy.get('where') is not None:
                                            if related_where is not None:
                                                wdict = _to_where_dict(related_where, strict=True)
                                                if wdict:
                                                    expr = _expr_from_where_dict(child_model_cls, wdict, strict=True)
                                                    if expr is not None:
                                                        stmt = stmt.where(expr)
                                            dwhere = meta_copy.get('where')
                                            if dwhere is not None:
                                                if isinstance(dwhere, (dict, str)):
                                                    wdict = _to_where_dict(dwhere, strict=False)
                                                    if wdict:
                                                        expr = _expr_from_where_dict(child_model_cls, wdict, strict=False)
                                                        if expr is not None:
                                                            stmt = stmt.where(expr)
                                                elif callable(dwhere):
                                                    expr = dwhere(child_model_cls, info)
                                                    if expr is not None:
                                                        stmt = stmt.where(expr)
                                        # Ordering: honor order_by/order_multi if provided; default by id
                                        try:
                                            allowed_order = [sf for sf, sd in self.__berry_registry__.types[target_name_i].__berry_fields__.items() if sd.kind == 'scalar']
                                        except Exception:
                                            allowed_order = []
                                        ordered_any = False
                                        if order_multi:
                                            for spec in order_multi:
                                                try:
                                                    cn, _, dd = str(spec).partition(':')
                                                    dd = dd or 'asc'
                                                    if not allowed_order or cn in allowed_order:
                                                        col = child_model_cls.__table__.c.get(cn)
                                                        if col is not None:
                                                            stmt = stmt.order_by(col.desc() if dd.lower()=='desc' else col.asc())
                                                            ordered_any = True
                                                except Exception:
                                                    pass
                                        if not ordered_any and order_by and (not allowed_order or order_by in allowed_order):
                                            col = child_model_cls.__table__.c.get(order_by)
                                            if col is not None:
                                                desc = _dir_value(order_dir) == 'desc' if order_dir is not None else False
                                                stmt = stmt.order_by(col.desc() if desc else col.asc())
                                                ordered_any = True
                                        if not ordered_any:
                                            try:
                                                pk_col_child = self.__berry_registry__._get_pk_column(child_model_cls)
                                                stmt = stmt.order_by(pk_col_child.asc())
                                            except Exception:
                                                pass
                                        result = await session.execute(stmt.limit(1))
                                        row = result.scalar_one_or_none()
                                        if row is None:
                                            return None
                                        inst = target_cls_i()
                                        setattr(inst, '_model', row)
                                        for sf, sdef in self.__berry_registry__.types[target_name_i].__berry_fields__.items():
                                            if sdef.kind == 'scalar':
                                                try:
                                                    setattr(inst, sf, getattr(row, sf, None))
                                                except Exception:
                                                    pass
                                        # Apply post-process for single value if provided
                                        try:
                                            pp = meta_copy.get('post_process')
                                        except Exception:
                                            pp = None
                                        if pp is not None:
                                            try:
                                                import inspect
                                                res = pp(inst, info)
                                                if inspect.isawaitable(res):
                                                    res = await res
                                                return res
                                            except Exception:
                                                return inst
                                        return inst
                                if candidate_fk_val is None:
                                    return None
                                # Apply filters via query if any filter args passed
                                if any(v is not None for v in _filter_args.values()) or related_where is not None:
                                    from sqlalchemy import select as _select
                                    # Log potential N+1 when no prefetched value and we need a targeted select
                                    try:
                                        reason = None
                                        try:
                                            meta_map = getattr(self, '_pushdown_meta', None)
                                            if isinstance(meta_map, dict):
                                                reason = (meta_map.get(fname_local) or {}).get('skip_reason')
                                        except Exception:
                                            reason = None
                                        if reason:
                                            _logger.warning(
                                                "berryql: falling back to per-parent select for single relation '%s' (reason=%s)",
                                                fname_local,
                                                reason,
                                            )
                                        else:
                                            _logger.warning(
                                                "berryql: falling back to per-parent select for single relation '%s' (no pushdown)",
                                                fname_local,
                                            )
                                    except Exception:
                                        pass
                                    try:
                                        pk_col_child = self.__berry_registry__._get_pk_column(child_model_cls)
                                        stmt = _select(child_model_cls).where(pk_col_child == candidate_fk_val)
                                    except Exception:
                                        # If PK not resolvable, fail fast to signal schema/model issue
                                        raise
                                    # Apply JSON where if provided (argument and schema default)
                                    if related_where is not None or meta_copy.get('where') is not None:
                                        # Strict for user-provided; permissive for schema default
                                        wdict_arg = _to_where_dict(related_where, strict=True) if related_where is not None else None
                                        if wdict_arg:
                                            expr = _expr_from_where_dict(child_model_cls, wdict_arg, strict=True)
                                            if expr is not None:
                                                stmt = stmt.where(expr)
                                        dwhere = meta_copy.get('where')
                                        if dwhere is not None:
                                            wdict_def = _to_where_dict(dwhere, strict=False)
                                            if wdict_def:
                                                expr = _expr_from_where_dict(child_model_cls, wdict_def, strict=False)
                                                if expr is not None:
                                                    stmt = stmt.where(expr)
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
                                    # Log potential N+1 when doing a direct get per parent
                                    try:
                                        reason = None
                                        try:
                                            meta_map = getattr(self, '_pushdown_meta', None)
                                            if isinstance(meta_map, dict):
                                                reason = (meta_map.get(fname_local) or {}).get('skip_reason')
                                        except Exception:
                                            reason = None
                                        if reason:
                                            _logger.warning(
                                                "berryql: falling back to per-parent get() for single relation '%s' (reason=%s)",
                                                fname_local,
                                                reason,
                                            )
                                        else:
                                            _logger.warning(
                                                "berryql: falling back to per-parent get() for single relation '%s' (no pushdown)",
                                                fname_local,
                                            )
                                    except Exception:
                                        pass
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
                                # Apply post-process for single value if provided
                                try:
                                    pp = meta_copy.get('post_process')
                                except Exception:
                                    pp = None
                                if pp is not None:
                                    try:
                                        import inspect
                                        res = pp(inst, info)
                                        if inspect.isawaitable(res):
                                            res = await res
                                        return res
                                    except Exception:
                                        return inst
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
                            if parent_model is not None:
                                try:
                                    pk_name_parent = self.__berry_registry__._get_pk_name(parent_model.__class__)
                                    parent_id_val = getattr(parent_model, pk_name_parent, None)
                                except Exception:
                                    parent_id_val = None
                            else:
                                parent_id_val = getattr(self, 'id', None)
                            if parent_id_val is None:
                                return []
                            # Determine requested scalar fields for this relation, prefer extractor meta from root if present
                            requested_fields: list[str] = []
                            try:
                                meta_map0 = getattr(self, '_pushdown_meta', None)
                                if isinstance(meta_map0, dict):
                                    cfg = meta_map0.get(fname_local) or {}
                                    req = cfg.get('fields') or []
                                    if isinstance(req, (list, tuple)):
                                        requested_fields = [str(x) for x in req]
                            except Exception:
                                requested_fields = []
                            if not requested_fields:
                                # Fallback to local AST for this resolver
                                try:
                                    raw_info = getattr(info, '_raw_info', None) or info
                                    fnodes = list(getattr(raw_info, 'field_nodes', []) or [])
                                    if fnodes:
                                        node = fnodes[0]
                                        selset = getattr(node, 'selection_set', None)
                                        sels = list(getattr(selset, 'selections', []) or []) if selset is not None else []
                                        for s in sels:
                                            nm = None
                                            try:
                                                nobj = getattr(s, 'name', None)
                                                nm = getattr(nobj, 'value', None) if nobj is not None else None
                                                if nm is None:
                                                    nm = getattr(s, 'name', None)
                                            except Exception:
                                                nm = None
                                            if nm and not str(nm).startswith('__'):
                                                requested_fields.append(str(nm))
                                except Exception:
                                    requested_fields = []
                            # Build minimal column list: always include id for downstream nested resolvers; plus requested scalars if any
                            cols = []
                            pk_expr = None
                            try:
                                pk_expr = self.__berry_registry__._get_pk_column(child_model_cls)
                            except Exception:
                                pk_expr = None
                            if pk_expr is not None:
                                # Keep label as 'id' for downstream hydration compatibility
                                cols.append(pk_expr.label('id'))
                            # Filter requested_fields to scalars known on target type
                            try:
                                scalars_on_target = {sf for sf, sd in self.__berry_registry__.types[target_name_i].__berry_fields__.items() if sd.kind == 'scalar'}
                            except Exception:
                                scalars_on_target = set()
                            for fn in requested_fields:
                                if fn == 'id':
                                    continue
                                if scalars_on_target and fn not in scalars_on_target:
                                    continue
                                try:
                                    col_obj = child_model_cls.__table__.c.get(fn)
                                except Exception:
                                    col_obj = None
                                if col_obj is not None:
                                    cols.append(col_obj.label(fn))
                            if cols:
                                stmt = _select(*cols).select_from(child_model_cls).where(fk_col == parent_id_val)
                            else:
                                # Fallback to selecting id if nothing resolvable
                                try:
                                    pk_expr_fallback = self.__berry_registry__._get_pk_column(child_model_cls)
                                    stmt = _select(pk_expr_fallback).select_from(child_model_cls).where(fk_col == parent_id_val)
                                except Exception:
                                    return []
                            # Log potential N+1 for list relation when resolver runs a per-parent query
                            try:
                                reason = None
                                try:
                                    meta_map = getattr(self, '_pushdown_meta', None)
                                    if isinstance(meta_map, dict):
                                        reason = (meta_map.get(fname_local) or {}).get('skip_reason')
                                except Exception:
                                    reason = None
                                if reason:
                                    _logger.warning(
                                        "berryql: falling back to per-parent query for relation '%s' (reason=%s)",
                                        fname_local,
                                        reason,
                                    )
                                else:
                                    _logger.warning(
                                        "berryql: falling back to per-parent query for relation '%s' (no lateral pushdown)",
                                        fname_local,
                                    )
                            except Exception:
                                pass
                            # Apply where from args and/or schema default
                            if related_where is not None or meta_copy.get('where') is not None:
                                # Strictly validate and apply argument-provided where
                                if related_where is not None:
                                    wdict = _to_where_dict(related_where, strict=True)
                                    if wdict:
                                        expr = _expr_from_where_dict(child_model_cls, wdict, strict=True)
                                        if expr is not None:
                                            stmt = stmt.where(expr)
                                # Default where from schema meta: keep permissive
                                dwhere = meta_copy.get('where')
                                if dwhere is not None:
                                    if isinstance(dwhere, (dict, str)):
                                        wdict = _to_where_dict(dwhere, strict=False)
                                        if wdict:
                                            expr = _expr_from_where_dict(child_model_cls, wdict, strict=False)
                                            if expr is not None:
                                                stmt = stmt.where(expr)
                                    elif callable(dwhere):
                                        expr = dwhere(child_model_cls, info)
                                        if expr is not None:
                                            stmt = stmt.where(expr)
                            # Ad-hoc JSON where for relation list if present on selection
                            rel_meta_map = getattr(self, '_pushdown_meta', None)  # not reliable; read from extractor cfg instead
                            # Ordering (multi then single) if column whitelist permits
                            allowed_order = getattr(target_cls_i, '__ordering__', None)
                            if allowed_order is None:
                                # derive from scalar fields
                                allowed_order = [sf for sf, sd in self.__berry_registry__.types[target_name_i].__berry_fields__.items() if sd.kind == 'scalar']
                            applied_any = False
                            # Validate invalid order_by up front
                            if order_by and order_by not in allowed_order:
                                raise ValueError(f"Invalid order_by '{order_by}'. Allowed: {allowed_order}")
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
                                    # If no order_dir provided, default to ASC when explicit order_by is present
                                    descending = _dir_value(order_dir) == 'desc' if order_dir is not None else False
                                    try:
                                        stmt = stmt.order_by(col_obj.desc() if descending else col_obj.asc())
                                        applied_any = True
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
                            rows = []
                            try:
                                rows = result.mappings().all()  # type: ignore[attr-defined]
                            except Exception:
                                rows = [getattr(r, '_mapping', None) or r for r in result.fetchall()]
                            results_list = []
                            for m in rows:
                                # Build kwargs for dataclass init from mapping keys
                                init_kwargs: Dict[str, Any] = {}
                                try:
                                    mapping_obj = getattr(m, '_mapping', None) or m
                                except Exception:
                                    mapping_obj = m
                                try:
                                    from collections.abc import Mapping as _Mapping  # type: ignore
                                except Exception:
                                    _Mapping = None  # type: ignore
                                if (_Mapping is not None and isinstance(mapping_obj, _Mapping)) or isinstance(mapping_obj, dict):
                                    try:
                                        items_iter = mapping_obj.items()
                                    except Exception:
                                        items_iter = []  # type: ignore
                                    for k, v in items_iter:
                                        try:
                                            init_kwargs[str(k)] = v
                                        except Exception:
                                            pass
                                else:
                                    # Fallback sequence handling
                                    try:
                                        init_kwargs['id'] = mapping_obj[0]
                                    except Exception:
                                        pass
                                try:
                                    inst = target_cls_i(**init_kwargs)
                                except Exception:
                                    # Fallback to empty init if something went wrong
                                    inst = target_cls_i()
                                    # Best-effort set attributes
                                    for k, v in init_kwargs.items():
                                        try:
                                            setattr(inst, k, v)
                                        except Exception:
                                            pass
                                # Helper attribute for downstream resolvers (best-effort)
                                try:
                                    setattr(inst, '_model', None)
                                except Exception:
                                    pass
                                results_list.append(inst)
                            # Apply post-process for list if provided
                            try:
                                pp = meta_copy.get('post_process')
                            except Exception:
                                pp = None
                            if pp is not None:
                                try:
                                    import inspect
                                    res = pp(results_list, info)
                                    if inspect.isawaitable(res):
                                        res = await res
                                    return res
                                except Exception:
                                    return results_list
                            return results_list
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
                            col_type_map = self._build_column_type_map(target_b.model)
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
                    # Attach resolver: if private, expose as plain method only; else as GraphQL field
                    if is_private:
                        setattr(st_cls, fname, _make_relation_resolver())
                    else:
                        # Attach resolver with explicit argument annotations to make relation args visible in schema
                        setattr(st_cls, fname, strawberry.field(resolver=_make_relation_resolver()))
                elif fdef.kind == 'aggregate':
                    if is_private:
                        # Skip exposing private aggregates
                        continue
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
                        async def aggregate_resolver(self, info: StrawberryInfo):  # noqa: D401
                            cache = getattr(self, '_agg_cache', None)
                            is_count_local = meta_copy.get('op') == 'count' or 'count' in meta_copy.get('ops', [])
                            if is_count_local and cache is not None:
                                key = meta_copy.get('cache_key') or (meta_copy.get('source') + ':count')
                                if key in cache:
                                    return cache[key]
                            parent_model = getattr(self, '_model', None)
                            if parent_model is None:
                                if is_count_local:
                                    return 0
                                return None
                            session = _get_db(info)
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
                                try:
                                    pk_name_parent = self.__berry_registry__._get_pk_name(parent_model.__class__)
                                    parent_pk_value = getattr(parent_model, pk_name_parent)
                                except Exception:
                                    # Fail fast: no PK value on parent
                                    return 0
                                stmt = _select(func.count()).select_from(child_model_cls).where(fk_col == parent_pk_value)
                                result = await session.execute(stmt)
                                val = result.scalar_one() or 0
                                key = meta_copy.get('cache_key') or (source + ':count')
                                if cache is None:
                                    cache = {}
                                    setattr(self, '_agg_cache', cache)
                                cache[key] = val
                                return val
                            return None
                        return aggregate_resolver
                    setattr(st_cls, fname, strawberry.field(_make_aggregate_resolver()))
                elif fdef.kind == 'custom':
                    if is_private:
                        # Skip exposing private custom fields
                        continue
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
                            session = _get_db(info)
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
                    if is_private:
                        # Skip exposing private custom object fields
                        continue
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
                                anns[k2] = Optional[t2] if t2 in (int, str, bool, float, datetime) else Optional[str]
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
                        annotations[fname] = Optional[str]
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
            # Merge in any user-declared strawberry fields that are not part of Berry field defs.
            # Strategy:
            # 1) Copy attributes explicitly annotated on the BerryType subclass (classic dataclass-like fields).
            # 2) Additionally scan class attributes for objects produced by @strawberry.field (method resolvers),
            #    and copy them verbatim so strawberry.type can pick them up on the generated class.
            try:
                # 1) Copy annotated attributes
                for uf, utype in (user_annotations or {}).items():
                    if uf in bcls.__berry_fields__:
                        continue  # skip berry-defined fields
                    if uf in annotations:
                        continue  # already defined
                    if uf.startswith('__'):
                        continue
                    annotations[uf] = utype
                    try:
                        setattr(st_cls, uf, getattr(bcls, uf))
                    except Exception:
                        pass
                # 2) Copy @strawberry.field method-based resolvers
                for uf, val in vars(bcls).items():
                    if uf in bcls.__berry_fields__:
                        continue
                    if uf in annotations or hasattr(st_cls, uf):
                        continue
                    if uf.startswith('__') or uf.startswith('_'):
                        continue
                    try:
                        mod = getattr(getattr(val, "__class__", object), "__module__", "") or ""
                        looks_strawberry = (
                            mod.startswith("strawberry")
                            or hasattr(val, "resolver")
                            or hasattr(val, "base_resolver")
                        )
                        if looks_strawberry:
                            # Rebuild a new strawberry.field bound to the generated class using the original resolver function
                            fn = getattr(val, 'resolver', None)
                            if fn is None:
                                br = getattr(val, 'base_resolver', None)
                                fn = getattr(br, 'func', None)
                            if fn is None:
                                fn = getattr(val, 'func', None)
                            if callable(fn):
                                # Let Strawberry infer the return type from the resolver; avoid injecting
                                # potentially version-specific StrawberryAnnotation wrappers into __annotations__.
                                setattr(st_cls, uf, strawberry.field(resolver=fn))
                            else:
                                # Fallback to copying as-is
                                setattr(st_cls, uf, val)
                    except Exception:
                        # best-effort
                        pass
            except Exception:
                pass
            st_cls.__annotations__ = annotations
        # Decorate all types now
        for name, cls in list(self._st_types.items()):
            if not getattr(cls, '__is_strawberry_type__', False):
                # Ensure typing symbols available for forward refs
                mod_globals = globals()
                mod_globals.update({'Optional': Optional, 'List': List})
                self._st_types[name] = strawberry.type(cls)  # type: ignore
        # Expose generated Strawberry types in this module's globals to satisfy LazyType lookups
        try:
            for _tname, _tcls in self._st_types.items():
                globals()[_tname] = _tcls
        except Exception:
            pass
        # Hand off to builder that assembles the Query type and Schema
        return self._build_query(strawberry_config=strawberry_config)

    def _build_query(self, *, strawberry_config: Optional[StrawberryConfig] = None):
        # Root query assembly: create class, then attach fields before decoration
        query_annotations: Dict[str, Any] = {}
        QueryPlain = type('Query', (), {'__doc__': 'Auto-generated Berry root query (prototype).'})
        setattr(QueryPlain, '__module__', __name__)
        # Helper: collect declared filters from a BerryType (auto roots only)
        def _collect_declared_filters(btype_cls_local, rel_args_spec: Optional[dict] = None):
            """Collect and expand declared filter specs.

            Expansion rules:
              - If spec has .op set -> single arg named exactly as mapping key (or alias if provided)
              - If spec has .ops list and no .op: expand to multiple args with suffixes
              - Callable-only spec (builder) uses provided key
            Returns: mapping of argument_name -> FilterSpec (with resolved single op).
            """
            out: Dict[str, FilterSpec] = {}
            # Only explicit arguments are used; type-level __filters__ is no longer supported
            class_filters = rel_args_spec if isinstance(rel_args_spec, dict) else {}
            for key, raw in class_filters.items():
                try:
                    if callable(raw):
                        spec = FilterSpec(builder=raw)
                    else:
                        spec = _normalize_filter_spec(raw)
                except Exception:
                    continue
                if spec.ops and not spec.op:
                    for op_name in spec.ops:
                        base = spec.alias or key
                        arg_name = base if base.endswith(f"_{op_name}") else f"{base}_{op_name}"
                        out[arg_name] = spec.clone_with(op=op_name, ops=None)
                else:
                    out[spec.alias or key] = spec
            return out

        # Core root resolver factory (used by auto roots and user-declared Query fields)
        def _make_root_resolver(model_cls, st_cls, btype_cls, root_field_name, relation_defaults: Optional[Dict[str, Any]] = None, is_single: bool = False):
            rel_args_spec = (relation_defaults or {}).get('arguments') if relation_defaults else None
            declared_filters = _collect_declared_filters(btype_cls, rel_args_spec)
            # Precompute column -> python type mapping for argument type inference
            col_py_types: Dict[str, Any] = self._build_column_type_map(model_cls)

            # Build argument annotations dynamically
            filter_arg_types: Dict[str, Any] = {}
            for arg_name, f_spec in declared_filters.items():
                base_type = str
                if f_spec.column and f_spec.column in col_py_types:
                    base_type = col_py_types[f_spec.column]
                if f_spec.op in ('in', 'between'):
                    filter_arg_types[arg_name] = Optional[List[base_type]]  # type: ignore
                else:
                    filter_arg_types[arg_name] = Optional[base_type]

            async def _base_impl(info: StrawberryInfo, limit: int | None, offset: int | None, order_by: Optional[str], order_dir: Optional[Any], order_multi: Optional[List[str]], _passed_filter_args: Dict[str, Any], raw_where: Optional[Any] = None):
                out = []
                session = _get_db(info)
                if session is None:
                    return out
                # Detect dialect & acquire adapter (unifies JSON funcs / capabilities)
                try:
                    dialect_name = session.get_bind().dialect.name.lower()
                except Exception:
                    dialect_name = 'sqlite'
                adapter = get_adapter(dialect_name)
                def _json_object(*args):
                    return adapter.json_object(*args)
                def _json_array_agg(expr):
                    return adapter.json_array_agg(expr)
                def _json_array_coalesce(expr):
                    return adapter.json_array_coalesce(expr)
                # Acquire per-context lock to avoid concurrent AsyncSession use (esp. MSSQL/pyodbc limitations)
                # info.context may be a dict or an object; handle both safely.
                lock = self._get_context_lock(info)
                # Centralized SQL builders: instantiate on demand to avoid scope issues
                # Collect custom field expressions for pushdown
                custom_fields: List[tuple[str, Any]] = []
                custom_object_fields: List[tuple[str, List[str], Any]] = []  # (field, column_labels, returns_spec)
                # Start with an empty selection; we'll add only the needed root columns
                # (e.g., id and any explicitly requested scalars) plus pushdown expressions.
                # Avoid selecting the full entity to keep SQL projections minimal.
                select_columns: List[Any] = []
                # Centralized query analysis: requested fields/relations and helper FKs
                _plan = QueryAnalyzer(self).analyze(info, root_field_name, btype_cls)
                requested_relations = _plan.requested_relations
                requested_scalar_root = _plan.requested_scalar_root
                requested_custom_root = _plan.requested_custom_root
                requested_custom_obj_root = _plan.requested_custom_obj_root
                requested_aggregates_root = _plan.requested_aggregates_root
                # Coercion of JSON where values moved to core.utils.coerce_where_value
                # Helper calls to RelationSQLBuilders are instantiated inline where needed
                # Prepare pushdown COUNT aggregates (centralized)
                count_aggregates: List[tuple[str, Any]] = []  # (agg_field_name, def)
                # Also build select_columns for count aggregates
                for cf_name, cf_def in btype_cls.__berry_fields__.items():
                    if cf_def.kind == 'custom':
                        # Include only if explicitly requested
                        if cf_name not in requested_custom_root:
                            continue
                        # Use centralized builder
                        expr = RelationSQLBuilders(self).build_custom_scalar_pushdown(
                            model_cls=model_cls,
                            field_name=cf_name,
                            builder=cf_def.meta.get('builder')
                        )
                        if expr is None:
                            continue
                        custom_fields.append((cf_name, expr))
                        select_columns.append(expr)
                    elif cf_def.kind == 'custom_object':
                        # Include only if explicitly requested
                        if cf_name not in requested_custom_obj_root:
                            continue
                        # Use centralized builder
                        built = RelationSQLBuilders(self).build_custom_object_pushdown(
                            model_cls=model_cls,
                            field_name=cf_name,
                            builder=cf_def.meta.get('builder'),
                            adapter=adapter,
                            json_object_fn=_json_object,
                            info=info,
                        )
                        if built is None:
                            continue
                        cols_to_add, labels = built
                        for c in cols_to_add:
                            select_columns.append(c)
                        custom_object_fields.append((cf_name, labels, cf_def.meta.get('returns')))
                    elif cf_def.kind == 'aggregate' and cf_name in requested_aggregates_root:
                        # centralized below; delay collecting to single pass
                        pass
                # centralized aggregate construction
                cols_aggs, meta_aggs = RootSQLBuilders(self).build_count_aggregates(
                    model_cls=model_cls,
                    btype_cls=btype_cls,
                    requested_aggregates=requested_aggregates_root,
                )
                for c in cols_aggs:
                    select_columns.append(c)
                count_aggregates.extend(meta_aggs)
                # MSSQL special handling: we'll emulate JSON aggregation via FOR JSON PATH
                mssql_mode = hasattr(adapter, 'name') and adapter.name == 'mssql'
                # Determine helper FK columns on parent for single relations. Include them proactively
                # whenever a single relation is requested so resolvers can operate even if pushdown
                # isn't used (or additional filtering is applied at resolve time).
                required_fk_parent_cols: set[str] = set()
                required_fk_parent_cols: set[str] = set(_plan.required_fk_parent_cols or set())
                # Track per-relation pushdown status and skip reasons
                rel_push_status: Dict[str, Dict[str, Any]] = {}
                # Push down relation JSON arrays/objects
                for rel_name, rel_cfg in requested_relations.items():
                    # initialize status entry
                    if rel_name not in rel_push_status:
                        rel_push_status[rel_name] = {'pushed': False, 'reason': None}
                    # Allow pushdown even when relation 'where' or filter args are provided.
                    # Both JSON where and declared filter args are handled in the pushdown builders below.
                    target_name = rel_cfg.get('target')
                    if not target_name:
                        try:
                            rel_push_status[rel_name].update({'pushed': False, 'reason': 'no target type'})
                        except Exception:
                            pass
                        continue
                    target_b = self.types.get(target_name)
                    if not target_b or not target_b.model:
                        try:
                            rel_push_status[rel_name].update({'pushed': False, 'reason': 'target model missing'})
                        except Exception:
                            pass
                        continue
                    # Early validation of relation order_by for pushdown path
                    try:
                        allowed_fields_rel = [sf for sf, sd in target_b.__berry_fields__.items() if sd.kind == 'scalar']
                    except Exception:
                        allowed_fields_rel = []
                    ob_rel = rel_cfg.get('order_by')
                    if ob_rel and ob_rel not in allowed_fields_rel:
                        raise ValueError(f"Invalid order_by '{ob_rel}'. Allowed: {allowed_fields_rel}")
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
                        try:
                            rel_push_status[rel_name].update({'pushed': False, 'reason': 'no FK child->parent'})
                        except Exception:
                            pass
                        continue
                    # Determine projected scalar fields
                    requested_scalar = rel_cfg.get('fields') or []
                    if rel_cfg.get('single'):
                        # If this single relation is a computed wrapper over another relation (source),
                        # skip pushdown here and let the resolver reuse the source's prefetched value.
                        try:
                            if btype_cls.__berry_fields__.get(rel_name) and btype_cls.__berry_fields__[rel_name].meta.get('source'):
                                rel_push_status[rel_name].update({'pushed': False, 'reason': 'computed from source'})
                                continue
                        except Exception:
                            pass
                        # Single relation: build json object
                        # Projected columns
                        proj_cols: list[str] = []
                        if not requested_scalar:
                            for sf, sdef in target_b.__berry_fields__.items():
                                if sdef.kind == 'scalar':
                                    requested_scalar.append(sf)
                        for sf in requested_scalar:
                            if sf in child_model_cls.__table__.columns:
                                proj_cols.append(sf)
                        # Determine FK helper name for correlation
                        parent_fk_col_name = self._find_parent_fk_column_name(model_cls, child_model_cls, rel_name)
                        # Delegate to builders
                        rel_expr_core = RelationSQLBuilders(self).build_single_relation_object(
                            adapter=adapter,
                            parent_model_cls=model_cls,
                            child_model_cls=child_model_cls,
                            rel_name=rel_name,
                            projected_columns=proj_cols,
                            parent_fk_col_name=parent_fk_col_name,
                            json_object_fn=_json_object,
                            json_array_coalesce_fn=_json_array_coalesce,
                            to_where_dict=_to_where_dict,
                            expr_from_where_dict=_expr_from_where_dict,
                            info=info,
                            rel_where=rel_cfg.get('where'),
                            rel_default_where=rel_cfg.get('default_where'),
                            filter_args=rel_cfg.get('filter_args') or {},
                            arg_specs=rel_cfg.get('arg_specs') or {},
                        )
                        if rel_expr_core is None:
                            try:
                                rel_push_status[rel_name].update({'pushed': False, 'reason': 'builder returned None'})
                            except Exception:
                                pass
                            continue
                        try:
                            select_columns.append(rel_expr_core.label(f"_pushrel_{rel_name}"))
                        except Exception:
                            select_columns.append(rel_expr_core)
                        try:
                            rel_push_status[rel_name].update({'pushed': True, 'reason': None})
                        except Exception:
                            pass
                    else:
                        # List relation (possibly with nested) JSON aggregation
                        # Prefer nested-capable builder when nested relations are selected (non-MSSQL path)
                        # Prefer nested-capable builder when nested relations are selected
                        nested_expr = None
                        try:
                            if (rel_cfg.get('nested') or {}) and not mssql_mode:
                                nested_expr = RelationSQLBuilders(self).build_list_relation_json_recursive(
                                    parent_model_cls=model_cls,
                                    parent_btype=btype_cls,
                                    rel_cfg=rel_cfg,
                                    json_object_fn=_json_object,
                                    json_array_agg_fn=_json_array_agg,
                                    json_array_coalesce_fn=_json_array_coalesce,
                                    to_where_dict=_to_where_dict,
                                    expr_from_where_dict=_expr_from_where_dict,
                                    dir_value_fn=_dir_value,
                                    info=info,
                                )
                        except Exception:
                            nested_expr = None
                        if nested_expr is None:
                            # Prepare list of scalar fields
                            requested_scalar_i = list(rel_cfg.get('fields') or [])
                            try:
                                if requested_scalar_i:
                                    tmp: list[str] = []
                                    for sf in requested_scalar_i:
                                        sdef = target_b.__berry_fields__.get(sf)
                                        if sdef and sdef.kind == 'scalar':
                                            tmp.append(sf)
                                    requested_scalar_i = tmp
                            except Exception:
                                pass
                            if not requested_scalar_i:
                                for sf, sdef in target_b.__berry_fields__.items():
                                    if sdef.kind == 'scalar':
                                        requested_scalar_i.append(sf)
                            # FK from child to parent
                            fk_col_i = None
                            for col in child_model_cls.__table__.columns:
                                for fk in col.foreign_keys:
                                    if fk.column.table.name == model_cls.__table__.name:
                                        fk_col_i = col
                                        break
                                if fk_col_i is not None:
                                    break
                            if fk_col_i is not None:
                                nested_expr = RelationSQLBuilders(self).build_list_relation_json_adapter(
                                    adapter=adapter,
                                    parent_model_cls=model_cls,
                                    child_model_cls=child_model_cls,
                                    requested_scalar=requested_scalar_i,
                                    fk_child_to_parent_col=fk_col_i,
                                    rel_cfg=rel_cfg,
                                    json_object_fn=_json_object,
                                    json_array_agg_fn=_json_array_agg,
                                    json_array_coalesce_fn=_json_array_coalesce,
                                    to_where_dict=_to_where_dict,
                                    expr_from_where_dict=_expr_from_where_dict,
                                    info=info,
                                )
                        if nested_expr is not None:
                            if mssql_mode:
                                # Wrap TextClause into a labeled column using scalar_subquery style text
                                try:
                                    select_columns.append(nested_expr.label(f"_pushrel_{rel_name}"))
                                except Exception:
                                    from sqlalchemy import literal_column
                                    select_columns.append(literal_column(str(nested_expr)).label(f"_pushrel_{rel_name}"))
                                try:
                                    rel_push_status[rel_name].update({'pushed': True, 'reason': None})
                                except Exception:
                                    pass
                                try:
                                    rr_entry = requested_relations.get(rel_name, {})
                                    rr_entry['from_pushdown'] = True
                                    rr_entry['skip_reason'] = None
                                except Exception:
                                    pass
                            else:
                                select_columns.append(nested_expr.label(f"_pushrel_{rel_name}"))
                                try:
                                    rel_push_status[rel_name].update({'pushed': True, 'reason': None})
                                except Exception:
                                    pass
                                try:
                                    rr_entry = requested_relations.get(rel_name, {})
                                    rr_entry['from_pushdown'] = True
                                    rr_entry['skip_reason'] = None
                                except Exception:
                                    pass
                        else:
                            # builder couldn't produce a pushdown expression
                            try:
                                if not rel_push_status.get(rel_name, {}).get('reason'):
                                    rel_push_status[rel_name].update({'pushed': False, 'reason': 'builder returned None'})
                            except Exception:
                                pass
                stmt = select(*select_columns)
                # Add minimal root columns directly into the SELECT to avoid pulling all entity columns
                # Prefer selecting only requested scalars and id when relations are requested.
                try:
                    base_root_cols = RootSQLBuilders(self).build_base_root_columns(
                        model_cls=model_cls,
                        requested_scalar_root=requested_scalar_root,
                        requested_relations=requested_relations,
                        required_fk_parent_cols=required_fk_parent_cols,
                    )
                    if base_root_cols or select_columns:
                        # When projecting labeled columns only, we must add an explicit FROM
                        # so ORDER BY and correlated subselects can reference the parent table.
                        stmt = select(*base_root_cols, *select_columns).select_from(model_cls)
                    else:
                        stmt = select(model_cls)
                except Exception:
                    stmt = select(model_cls)
                else:
                # ----- Phase 2 filtering (argument-driven) -----
                    stmt = RootSQLBuilders(self).apply_root_filters(
                        stmt,
                        model_cls=model_cls,
                        btype_cls=btype_cls,
                        info=info,
                        raw_where=raw_where,
                        declared_filters=declared_filters,
                        passed_filter_args=_passed_filter_args,
                    )
                    stmt = RootSQLBuilders(self).apply_ordering(
                        stmt,
                        model_cls=model_cls,
                        btype_cls=btype_cls,
                        order_by=order_by,
                        order_dir=order_dir,
                        order_multi=order_multi,
                    )
                    stmt = RootSQLBuilders(self).apply_pagination(stmt, limit=limit, offset=offset)
                    async with lock:
                        result = await session.execute(stmt)
                        sa_rows = result.fetchall()
                        out = []
                        for row_index, sa_row in enumerate(sa_rows):
                            inst = st_cls()
                            try:
                                mapping = getattr(sa_row, '_mapping')
                            except Exception:
                                mapping = {}
                            _hydr = Hydrator(self)
                            # Attach model and copy base mapping fields (for direct labeled columns)
                            _hydr.attach_model(inst, sa_row)
                            _hydr.copy_mapping_fields(inst, mapping)
                            # Base scalars and helper FKs
                            _hydr.hydrate_base_scalars(
                                inst,
                                mapping,
                                requested_scalar_root=requested_scalar_root,
                                requested_relations=requested_relations,
                                required_fk_parent_cols=required_fk_parent_cols,
                            )
                            # Custom fields
                            _hydr.hydrate_custom_scalars(inst, mapping, custom_fields=custom_fields)
                            _hydr.hydrate_custom_objects(inst, mapping, custom_object_fields=custom_object_fields, btype_cls=btype_cls)
                            # Relations (from pushdown JSON)
                            if requested_relations:
                                _hydr.hydrate_relations(inst, mapping, requested_relations=requested_relations, rel_push_status=rel_push_status)
                            # Aggregates cache
                            _hydr.populate_aggregate_cache(inst, mapping, count_aggregates=count_aggregates)
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
            # Build resolver function programmatically (outside _base_impl)
            # Always include declared filter args for both declared and auto roots
            arg_defs = []
            for a, t in filter_arg_types.items():
                arg_defs.append(f"{a}=None")
            args_str = (', '.join(arg_defs)) if arg_defs else ''
            func_name = f"_root_{root_field_name}"
            # Build parameter list: info, limit, offset, ordering, optional where, order_multi, filter args
            if args_str:
                full_params = f"self, info, limit=None, offset=None, order_by=None, order_dir=None, order_multi=None, where=None, {args_str}"
            else:
                full_params = "self, info, limit=None, offset=None, order_by=None, order_dir=None, order_multi=None, where=None"
            src = f"async def {func_name}({full_params}):\n" \
                  f"    _fa = {{}}\n"
            for a in declared_filters.keys():
                src += f"    _fa['{a}'] = {a} if '{a}' in locals() else None\n"
            # Apply defaults for declared roots if provided
            src += "    _ob = order_by if order_by is not None else (_defaults.get('order_by') if _defaults is not None else None)\n"
            src += "    _od = order_dir if order_dir is not None else (_defaults.get('order_dir') if _defaults is not None else None)\n"
            src += "    _om = order_multi if order_multi is not None else (_defaults.get('order_multi') if _defaults is not None else None)\n"
            src += "    _rw = where if where is not None else (_defaults.get('where') if _defaults is not None else None)\n"
            src += "    _lim = limit\n"
            src += "    if _is_single and _lim is None: _lim = 1\n"
            src += "    _rows = await _base_impl(info, _lim, offset, _ob, _od, _om, _fa, _rw)\n"
            src += "    return (_rows[0] if _rows else None) if _is_single else _rows\n"
            ns: Dict[str, Any] = {'_base_impl': _base_impl, '_defaults': relation_defaults, '_is_single': bool(is_single)}
            ns.update({'Optional': Optional, 'List': List, 'datetime': datetime})
            exec(src, ns)
            generated_fn = ns[func_name]
            if not getattr(generated_fn, '__module__', None):
                generated_fn.__module__ = __name__
            ann: Dict[str, Any] = {'info': StrawberryInfo, 'limit': Optional[int], 'offset': Optional[int], 'order_by': Optional[str], 'order_dir': Optional[Direction], 'order_multi': Optional[List[str]], 'where': Optional[str]}
            ann.update(filter_arg_types)
            generated_fn.__annotations__ = ann
            return generated_fn

        # Helper to add fields from user-declared Query
        def _add_declared_root_fields_to_class(query_cls):
            assert self._root_query_fields is not None
            for fname, fdef in self._root_query_fields.items():
                if fdef.kind != 'relation':
                    continue
                target_name = fdef.meta.get('target')
                if not target_name:
                    continue
                target_b = self.types.get(target_name)
                target_st = self._st_types.get(target_name)
                if not target_b or not target_b.model or target_st is None:
                    continue
                # Defaults sourced from relation meta on Query
                rel_defaults = {
                    'order_by': fdef.meta.get('order_by'),
                    'order_dir': fdef.meta.get('order_dir'),
                    'order_multi': fdef.meta.get('order_multi') or [],
                    'where': fdef.meta.get('where'),
                    'arguments': fdef.meta.get('arguments')
                }
                is_single_root = bool(fdef.meta.get('single'))
                root_resolver = _make_root_resolver(target_b.model, target_st, target_b, fname, relation_defaults=rel_defaults, is_single=is_single_root)
                # Annotations
                if is_single_root:
                    query_annotations[fname] = Optional[self._st_types[target_name]]  # type: ignore
                else:
                    query_annotations[fname] = List[self._st_types[target_name]]  # type: ignore
                # Attach field on class with explicit resolver
                setattr(query_cls, fname, strawberry.field(resolver=root_resolver))

        # Only add user-declared root fields
        if self._root_query_fields:
            _add_declared_root_fields_to_class(QueryPlain)
        # Build and attach any declared domain fields (supports nested domains)
        if self._domains:
            # Helper to (recursively) build a Strawberry type for a domain class and cache it
            _domain_type_cache: Dict[Type[Any], Any] = {}

            def _ensure_domain_type(dom_cls: Type[Any]):
                # Return cached if already built
                try:
                    if dom_cls in _domain_type_cache:
                        return _domain_type_cache[dom_cls]
                except Exception:
                    pass
                type_name = f"{getattr(dom_cls, '__name__', 'Domain')}Type"
                if type_name in self._st_types:
                    DomSt_local = self._st_types[type_name]
                    _domain_type_cache[dom_cls] = DomSt_local
                    return DomSt_local
                # Create plain runtime class
                DomSt_local = type(type_name, (), {'__doc__': f'Domain container for {getattr(dom_cls, "__name__", type_name)}'})
                # Ensure the class appears to live in this module to prevent LazyType path lookups
                DomSt_local.__module__ = __name__
                _domain_type_cache[dom_cls] = DomSt_local  # pre-cache to break cycles
                ann_local: Dict[str, Any] = {}
                # Attach relation fields from domain class
                for fname, fval in list(vars(dom_cls).items()):
                    if isinstance(fval, FieldDescriptor):
                        fval.__set_name__(None, fname)
                        fdef = fval.build(type_name)
                        if fdef.kind != 'relation':
                            continue
                        target_name = fdef.meta.get('target')
                        target_b = self.types.get(target_name) if target_name else None
                        target_st = self._st_types.get(target_name) if target_name else None
                        if not target_b or not target_b.model or not target_st:
                            continue
                        # Merge domain-level guard as default where if provided
                        defaults = {
                            'order_by': fdef.meta.get('order_by'),
                            'order_dir': fdef.meta.get('order_dir'),
                            'order_multi': fdef.meta.get('order_multi') or [],
                            'where': fdef.meta.get('where'),
                            'arguments': fdef.meta.get('arguments')
                        }
                        try:
                            guard = getattr(dom_cls, '__domain_guard__', None)
                        except Exception:
                            guard = None
                        if guard is not None and defaults.get('where') is None:
                            defaults['where'] = guard
                        is_single = bool(fdef.meta.get('single'))
                        resolver = _make_root_resolver(target_b.model, target_st, target_b, fname, relation_defaults=defaults, is_single=is_single)
                        if is_single:
                            ann_local[fname] = Optional[target_st]  # type: ignore
                        else:
                            ann_local[fname] = List[target_st]  # type: ignore
                        setattr(DomSt_local, fname, strawberry.field(resolver=resolver))
                # Attach nested domain fields declared via DomainDescriptor
                try:
                    from .core.fields import DomainDescriptor as _DomDesc
                except Exception:
                    _DomDesc = None  # type: ignore
                for fname, fval in list(vars(dom_cls).items()):
                    if _DomDesc is not None and isinstance(fval, _DomDesc):
                        # Determine field name and nested class
                        child_cls = fval.domain_cls
                        child_dom_st = _ensure_domain_type(child_cls)
                        # Use concrete class reference in annotations to avoid LazyType deprecation warnings
                        ann_local[fname] = child_dom_st  # type: ignore
                        def _make_nested_resolver(ChildSt):
                            async def _resolver(self, info: StrawberryInfo):  # noqa: D401
                                inst = ChildSt()
                                setattr(inst, '__berry_registry__', getattr(self, '__berry_registry__', self))
                                return inst
                            return _resolver
                        setattr(DomSt_local, fname, strawberry.field(resolver=_make_nested_resolver(child_dom_st)))
                # Copy any strawberry fields defined on the domain class (e.g., @strawberry.field)
                for uf, val in list(vars(dom_cls).items()):
                    if (_DomDesc is not None and isinstance(val, _DomDesc)) or isinstance(val, FieldDescriptor):
                        continue
                    if uf.startswith('__') or uf.startswith('_'):
                        continue
                    if hasattr(DomSt_local, uf):
                        continue
                    try:
                        mod = getattr(getattr(val, "__class__", object), "__module__", "") or ""
                        looks_st = (
                            mod.startswith("strawberry")
                            or hasattr(val, "resolver")
                            or hasattr(val, "base_resolver")
                        )
                        if looks_st:
                            fn = getattr(val, 'resolver', None)
                            if fn is None:
                                br = getattr(val, 'base_resolver', None)
                                fn = getattr(br, 'func', None)
                            if fn is None:
                                fn = getattr(val, 'func', None)
                            if callable(fn):
                                setattr(DomSt_local, uf, strawberry.field(resolver=fn))
                            else:
                                setattr(DomSt_local, uf, val)
                    except Exception:
                        pass
                DomSt_local.__annotations__ = ann_local
                # Decorate and cache
                self._st_types[type_name] = strawberry.type(DomSt_local)  # type: ignore
                # Also export into module globals to satisfy any LazyType lookups by name
                try:
                    globals()[type_name] = self._st_types[type_name]
                except Exception:
                    pass
                return self._st_types[type_name]

            for dom_name, cfg in list(self._domains.items()):
                dom_cls = cfg.get('class')
                if not dom_cls:
                    continue
                DomSt = _ensure_domain_type(dom_cls)
                # Expose on Query as a field that returns the domain container instance
                def _make_domain_resolver(DomSt_local):
                    async def _resolver(self, info: StrawberryInfo):  # noqa: D401
                        inst = DomSt_local()
                        setattr(inst, '__berry_registry__', self)
                        return inst
                    return _resolver
                query_annotations[dom_name] = DomSt  # type: ignore
                setattr(QueryPlain, dom_name, strawberry.field(resolver=_make_domain_resolver(DomSt)))
        # Merge any regular @strawberry.field resolvers defined on the user Query class
        try:
            if getattr(self, '_user_query_cls', None) is not None:
                for uf, val in vars(self._user_query_cls).items():
                    # Skip Berry FieldDescriptors (already handled), dunders/private, and existing names
                    if isinstance(val, FieldDescriptor):
                        continue
                    if uf.startswith('__') or uf.startswith('_'):
                        continue
                    if hasattr(QueryPlain, uf):
                        continue
                    try:
                        mod = getattr(getattr(val, "__class__", object), "__module__", "") or ""
                        looks_strawberry = (
                            mod.startswith("strawberry")
                            or hasattr(val, "resolver")
                            or hasattr(val, "base_resolver")
                        )
                        if looks_strawberry:
                            fn = getattr(val, 'resolver', None)
                            if fn is None:
                                br = getattr(val, 'base_resolver', None)
                                fn = getattr(br, 'func', None)
                            if fn is None:
                                fn = getattr(val, 'func', None)
                            if callable(fn):
                                setattr(QueryPlain, uf, strawberry.field(resolver=fn))
                            else:
                                setattr(QueryPlain, uf, val)
                    except Exception:
                        # best-effort
                        pass
        except Exception:
            pass
        # Add ping field
        async def _ping() -> str:  # noqa: D401
            return 'pong'
        query_annotations['_ping'] = str
        setattr(QueryPlain, '_ping', strawberry.field(resolver=_ping))
        # Set annotations last
        setattr(QueryPlain, '__annotations__', query_annotations)
        Query = QueryPlain
        Query = strawberry.type(Query)  # type: ignore
        # Optionally build Mutation and Subscription roots if provided by user
        Mutation = None
        Subscription = None
        try:
            if self._user_mutation_cls is not None:
                MPlain = type('Mutation', (), {'__doc__': 'Auto-generated Berry root mutation.'})
                setattr(MPlain, '__module__', __name__)
                # Copy annotated attributes and strawberry-decorated fields as-is
                try:
                    anns_m = dict(getattr(self._user_mutation_cls, '__annotations__', {}) or {})
                except Exception:
                    anns_m = {}
                for uf, utype in (anns_m or {}).items():
                    try:
                        setattr(MPlain, uf, getattr(self._user_mutation_cls, uf))
                    except Exception:
                        pass
                # Copy methods/attributes decorated by strawberry (mutation/field)
                for uf, val in vars(self._user_mutation_cls).items():
                    if uf.startswith('__'):
                        continue
                    if hasattr(MPlain, uf):
                        continue
                    try:
                        looks_strawberry = (
                            str(getattr(getattr(val, "__class__", object), "module", "") or "").startswith("strawberry")
                            or hasattr(val, "resolver")
                            or hasattr(val, "base_resolver")
                        )
                        if looks_strawberry:
                            # Prefer copying the field object as-is to preserve mutation semantics
                            setattr(MPlain, uf, val)
                    except Exception:
                        pass
                setattr(MPlain, '__annotations__', anns_m)
                Mutation = strawberry.type(MPlain)  # type: ignore
        except Exception:
            Mutation = None
        try:
            if self._user_subscription_cls is not None:
                SPlain = type('Subscription', (), {'__doc__': 'Auto-generated Berry root subscription.'})
                setattr(SPlain, '__module__', __name__)
                # Copy annotated attributes and strawberry subscription fields
                try:
                    anns_s = dict(getattr(self._user_subscription_cls, '__annotations__', {}) or {})
                except Exception:
                    anns_s = {}
                for uf, utype in (anns_s or {}).items():
                    try:
                        setattr(SPlain, uf, getattr(self._user_subscription_cls, uf))
                    except Exception:
                        pass
                for uf, val in vars(self._user_subscription_cls).items():
                    if uf.startswith('__'):
                        continue
                    if hasattr(SPlain, uf):
                        continue
                    try:
                        looks_strawberry = (
                            str(getattr(getattr(val, "__class__", object), "module", "") or "").startswith("strawberry")
                            or hasattr(val, "resolver")
                            or hasattr(val, "base_resolver")
                        )
                        if looks_strawberry:
                            # Preserve subscription objects as-is
                            setattr(SPlain, uf, val)
                    except Exception:
                        pass
                setattr(SPlain, '__annotations__', anns_s)
                Subscription = strawberry.type(SPlain)  # type: ignore
        except Exception:
            Subscription = None
        # Build the Schema honoring provided strawberry_config; keep backward compatibility.
        # Precedence: explicit strawberry_config -> default False (legacy behavior)
        if strawberry_config is not None:
            # First, try older Strawberry signature using auto_camel_case kwarg
            try:
                ac = getattr(strawberry_config, 'auto_camel_case', None)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"LazyType is deprecated.*")
                    if Mutation is not None and Subscription is not None:
                        return strawberry.Schema(Query, mutation=Mutation, subscription=Subscription, auto_camel_case=ac)  # type: ignore[arg-type]
                    if Mutation is not None:
                        return strawberry.Schema(Query, mutation=Mutation, auto_camel_case=ac)  # type: ignore[arg-type]
                    return strawberry.Schema(Query, auto_camel_case=ac)  # type: ignore[arg-type]
            except TypeError:
                # If unsupported, try newer signature with config kwarg
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"LazyType is deprecated.*")
                        if Mutation is not None and Subscription is not None:
                            return strawberry.Schema(Query, mutation=Mutation, subscription=Subscription, config=strawberry_config)  # type: ignore[arg-type]
                        if Mutation is not None:
                            return strawberry.Schema(Query, mutation=Mutation, config=strawberry_config)  # type: ignore[arg-type]
                        return strawberry.Schema(Query, config=strawberry_config)  # type: ignore[arg-type]
                except TypeError:
                    # Last resort: build without config
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"LazyType is deprecated.*")
                        if Mutation is not None and Subscription is not None:
                            return strawberry.Schema(Query, mutation=Mutation, subscription=Subscription)
                        if Mutation is not None:
                            return strawberry.Schema(Query, mutation=Mutation)
                        return strawberry.Schema(Query)
        # Default: preserve previous library behavior (auto_camel_case=False)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"LazyType is deprecated.*")
                if Mutation is not None and Subscription is not None:
                    return strawberry.Schema(Query, mutation=Mutation, subscription=Subscription, auto_camel_case=False)  # type: ignore[arg-type]
                if Mutation is not None:
                    return strawberry.Schema(Query, mutation=Mutation, auto_camel_case=False)  # type: ignore[arg-type]
                return strawberry.Schema(Query, auto_camel_case=False)  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"LazyType is deprecated.*")
                    if Mutation is not None and Subscription is not None:
                        return strawberry.Schema(Query, mutation=Mutation, subscription=Subscription, config=StrawberryConfig(auto_camel_case=False))  # type: ignore[arg-type]
                    if Mutation is not None:
                        return strawberry.Schema(Query, mutation=Mutation, config=StrawberryConfig(auto_camel_case=False))  # type: ignore[arg-type]
                    return strawberry.Schema(Query, config=StrawberryConfig(auto_camel_case=False))  # type: ignore[arg-type]
            except Exception:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"LazyType is deprecated.*")
                    if Mutation is not None and Subscription is not None:
                        return strawberry.Schema(Query, mutation=Mutation, subscription=Subscription)
                    if Mutation is not None:
                        return strawberry.Schema(Query, mutation=Mutation)
                    return strawberry.Schema(Query)
