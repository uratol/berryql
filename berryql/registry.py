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
from .sql.builders import RelationSQLBuilders

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
            # Unwrap TypeDecorator when possible and special-case BinaryArray
            if isinstance(sqlatype, _SATypeDecorator):
                # Special-case our BinaryArray -> List[str]
                try:
                    if type(sqlatype).__name__.lower() == 'binaryarray':
                        return List[str]  # type: ignore[index]
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
                    py_t = column_type_map.get(fname, str)
                    annotations[fname] = Optional[py_t]
                    setattr(st_cls, fname, None)
                elif fdef.kind == 'relation':
                    target_name = fdef.meta.get('target')
                    is_single = bool(fdef.meta.get('single'))
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
                            if hasattr(self, prefetch_attr):
                                # If relation was prefetched via SQL pushdown, return it directly.
                                prefetched = getattr(self, prefetch_attr)
                                if is_single_value:
                                    return prefetched
                                return list(prefetched or [])
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
                                # First, if the instance has a helper '<relation>_id' attribute, use it
                                try:
                                    candidate_fk_val = getattr(self, f"{fname_local}_id", None)
                                except Exception:
                                    candidate_fk_val = None
                                if candidate_fk_val is None and parent_model is not None:
                                    # Derive FK from ORM model instance when available
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
                                    stmt = _select(child_model_cls).where(getattr(child_model_cls, 'id') == candidate_fk_val)
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
                            id_expr = None
                            try:
                                id_expr = getattr(child_model_cls, '__table__', None)
                                id_expr = id_expr.c.get('id') if id_expr is not None else None
                                if id_expr is None:
                                    id_expr = getattr(child_model_cls, 'id', None)
                            except Exception:
                                id_expr = getattr(child_model_cls, 'id', None)
                            if id_expr is not None:
                                cols.append(id_expr.label('id'))
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
                                stmt = _select(id_expr).select_from(child_model_cls).where(fk_col == parent_id_val)
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
                                stmt = _select(func.count()).select_from(child_model_cls).where(fk_col == getattr(parent_model, 'id'))
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
                # Collect custom field expressions for pushdown
                custom_fields: List[tuple[str, Any]] = []
                custom_object_fields: List[tuple[str, List[str], Any]] = []  # (field, column_labels, returns_spec)
                # Start with an empty selection; we'll add only the needed root columns
                # (e.g., id and any explicitly requested scalars) plus pushdown expressions.
                # Avoid selecting the full entity to keep SQL projections minimal.
                select_columns: List[Any] = []
                # Use extracted helper classes: relations and root field kinds
                # Instantiate relation extractor with registry wired
                _rel_extractor = RelationSelectionExtractor(self)  # type: ignore
                requested_relations = _rel_extractor.extract(info, root_field_name, btype_cls)
                # Normalize any AST-node leftovers in relation configs to plain Python types
                # Using shared helpers from core.utils
                for rel_cfg in list(requested_relations.values()):
                    _normalize_rel_cfg(rel_cfg)
                # Use RootSelectionExtractor with registry for naming conversion (camelCase support)
                root_selected = RootSelectionExtractor(self).extract(info, root_field_name, btype_cls)
                requested_scalar_root: set[str] = set(root_selected.get('scalars', set()))
                requested_custom_root: set[str] = set(root_selected.get('custom', set()))
                requested_custom_obj_root: set[str] = set(root_selected.get('custom_object', set()))
                requested_aggregates_root: set[str] = set(root_selected.get('aggregate', set()))
                # Determine if we successfully extracted any selection at all. If yes, include
                # only explicitly requested custom/aggregate fields; if not, keep legacy behavior.
                selection_extracted = bool(
                    requested_scalar_root or requested_custom_root or requested_custom_obj_root or requested_relations
                )
                # Coercion of JSON where values moved to core.utils.coerce_where_value
                # Use builders to recursively build relation JSON for non-MSSQL
                def _build_list_relation_json(parent_model_cls, parent_btype, rel_cfg: Dict[str, Any]):
                    return _sql_builders.build_list_relation_json_recursive(
                        parent_model_cls=parent_model_cls,
                        parent_btype=parent_btype,
                        rel_cfg=rel_cfg,
                        json_object_fn=_json_object,
                        json_array_agg_fn=_json_array_agg,
                        json_array_coalesce_fn=_json_array_coalesce,
                        to_where_dict=_to_where_dict,
                        expr_from_where_dict=_expr_from_where_dict,
                        dir_value_fn=_dir_value,
                        info=info,
                    )
                # Prepare pushdown COUNT aggregates (replace batch later)
                count_aggregates: List[tuple[str, Any]] = []  # (agg_field_name, subquery_def)
                for cf_name, cf_def in btype_cls.__berry_fields__.items():
                    if cf_def.kind == 'custom':
                        # Include only if explicitly requested
                        if cf_name not in requested_custom_root:
                            continue
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
                                if hasattr(expr, 'subquery'):
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
                        # Include only if explicitly requested
                        if cf_name not in requested_custom_obj_root:
                            continue
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
                            agg_pairs: list[tuple[str, Any]] = []  # (key, aggregate/labeled expr)
                            for col in sel_cols:
                                try:
                                    labeled = col
                                    col_name = getattr(labeled, 'name', None) or getattr(labeled, 'key', None)
                                    if not col_name:
                                        col_name = f"{cf_name}_{len(key_exprs)}"
                                        labeled = col.label(col_name)
                                    # Keep the aggregate/labeled expression for single-select JSON
                                    agg_pairs.append((col_name, labeled))
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
                                        from sqlalchemy import literal_column
                                        select_columns.append(literal_column(str(v)).label(lbl))
                                    labels.append(lbl)
                                custom_object_fields.append((cf_name, labels, cf_def.meta.get('returns')))
                            else:
                                # Compose JSON object via a single SELECT using aggregated expressions
                                json_args: list[Any] = []
                                for k, agg_expr in agg_pairs:
                                    json_args.extend([_text(f"'{k}'"), agg_expr])
                                inner = select(_json_object(*json_args))
                                try:
                                    for _from in expr_sel.get_final_froms():  # type: ignore[attr-defined]
                                        inner = inner.select_from(_from)
                                except Exception:
                                    pass
                                for _w in getattr(expr_sel, '_where_criteria', []):  # type: ignore[attr-defined]
                                    inner = inner.where(_w)
                                try:
                                    json_obj_expr = inner.scalar_subquery()
                                except Exception:
                                    json_obj_expr = inner
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
                            # Include only if explicitly requested
                            if cf_name not in requested_aggregates_root:
                                continue
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
                # Centralized SQL builders
                _sql_builders = RelationSQLBuilders(self)
                # Determine helper FK columns on parent for single relations. Include them proactively
                # whenever a single relation is requested so resolvers can operate even if pushdown
                # isn't used (or additional filtering is applied at resolve time).
                required_fk_parent_cols: set[str] = set()
                try:
                    for rel_name, rel_cfg in list(requested_relations.items()):
                        # Collect helper FK columns for single relations; don't preemptively skip pushdown for callables
                        if rel_cfg.get('single'):
                            try:
                                target_name = rel_cfg.get('target')
                                child_model_cls = (self.types.get(target_name).model if target_name and self.types.get(target_name) else None)
                            except Exception:
                                child_model_cls = None
                            parent_fk_col_name = self._find_parent_fk_column_name(model_cls, child_model_cls, rel_name)
                            if parent_fk_col_name is not None:
                                required_fk_parent_cols.add(parent_fk_col_name)
                except Exception:
                    required_fk_parent_cols = set()
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
                        rel_expr_core = _sql_builders.build_single_relation_object(
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
                                nested_expr = _build_list_relation_json(model_cls, btype_cls, rel_cfg)
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
                                nested_expr = _sql_builders.build_list_relation_json_adapter(
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
                    base_root_cols: list[Any] = []
                    effective_root_cols: set[str] = set(requested_scalar_root or set())
                    # Always include primary id when available to ensure hydration works in nested/domain contexts
                    try:
                        if hasattr(model_cls, 'id') and 'id' not in effective_root_cols:
                            effective_root_cols.add('id')
                    except Exception:
                        pass
                    # Include id when any relation is requested (needed for correlation)
                    if requested_relations:
                        effective_root_cols.add('id')
                    # Include any required FK helper columns for resolvers
                    for fk in required_fk_parent_cols:
                        effective_root_cols.add(fk)
                    # Label columns explicitly so RowMapping uses plain keys (e.g., 'id', 'author_id')
                    for sf in effective_root_cols:
                        if hasattr(model_cls, sf):
                            try:
                                col_obj = getattr(model_cls, sf)
                                base_root_cols.append(col_obj.label(sf))
                            except Exception:
                                base_root_cols.append(getattr(model_cls, sf))
                    if base_root_cols or select_columns:
                        stmt = select(*base_root_cols, *select_columns)
                    else:
                        # Nothing explicit to project; fall back to entity
                        stmt = select(model_cls)
                except Exception:
                    # Fallback if anything above fails
                    stmt = select(model_cls)
                else:
                # ----- Phase 2 filtering (argument-driven) -----
                    where_clauses = []
                    # Apply optional context-aware root custom where if provided on BerryType
                    try:
                        custom_where = getattr(btype_cls, '__root_custom_where__', None)
                    except Exception:
                        custom_where = None
                    # Use shared where-dict -> SQLA expression builder
                    # Only enforce custom_where when explicitly enabled by context flag
                    try:
                        _ctx = getattr(info, 'context', None)
                    except Exception:
                        _ctx = None
                    enforce_gate = False
                    if isinstance(_ctx, dict):
                        enforce_gate = bool(_ctx.get('enforce_user_gate'))
                    elif _ctx is not None:
                        try:
                            enforce_gate = bool(getattr(_ctx, 'enforce_user_gate', False))
                        except Exception:
                            enforce_gate = False
                    if custom_where is not None and enforce_gate:
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
                    # Apply ad-hoc raw where if provided by user/defaults
                    if raw_where is not None:
                        wdict = raw_where
                        # Allow callable: (model_cls, info) -> dict/JSON or SA expr
                        if callable(raw_where):
                            wdict = raw_where(model_cls, info)
                        expr2 = None
                        # Try dict/JSON first
                        try:
                            wdict_parsed = _to_where_dict(wdict, strict=True) if not isinstance(wdict, dict) else wdict
                        except Exception as e:
                            # If not parseable and not a dict, treat as potential SA expr; if it's a string, raise
                            if isinstance(wdict, str):
                                raise ValueError(f"Invalid where JSON: {e}")
                            wdict_parsed = None
                        if isinstance(wdict_parsed, dict):
                            expr2 = _expr_from_where_dict(model_cls, wdict_parsed)
                        else:
                            # assume SQLAlchemy expression or None
                            expr2 = wdict if not isinstance(wdict, str) else None
                        if expr2 is None and isinstance(wdict, str):
                            raise ValueError("where must be a JSON object")
                        if expr2 is not None:
                            where_clauses.append(expr2)
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
                    if order_multi:
                        allowed_order_fields = getattr(btype_cls, '__ordering__', None)
                        if allowed_order_fields is None:
                            allowed_order_fields = [fname for fname, fdef in btype_cls.__berry_fields__.items() if fdef.kind == 'scalar']
                        for spec in (order_multi or []):
                            try:
                                cn, _, dd = str(spec).partition(':')
                                dd = (dd or 'asc').lower()
                                if cn not in allowed_order_fields:
                                    continue
                                col = model_cls.__table__.c.get(cn)
                                if col is None:
                                    continue
                                stmt = stmt.order_by(col.desc() if dd=='desc' else col.asc())
                            except Exception:
                                pass
                    elif order_by:
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
                        out = []
                        for row_index, sa_row in enumerate(sa_rows):
                            inst = st_cls()
                            # Prefer mapping access for both minimal and legacy entity selects
                            try:
                                mapping = getattr(sa_row, '_mapping')
                            except Exception:
                                mapping = {}
                            # Pre-hydrate any mapped root scalars onto the instance (id, helper FKs, requested scalars)
                            try:
                                # Mapping supports .items(); if not, try casting to dict
                                it = None
                                try:
                                    it = mapping.items()
                                except Exception:
                                    try:
                                        it = dict(mapping).items()
                                    except Exception:
                                        it = []
                                for key, val in list(it):
                                    try:
                                        setattr(inst, key, val)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            # Attach _model only when an ORM entity is present in row 0
                            try:
                                row0 = sa_row[0]
                                # Heuristic: ORM entity usually has __table__ attr on its class
                                if hasattr(getattr(row0, '__class__', object), '__table__'):
                                    setattr(inst, '_model', row0)
                                else:
                                    setattr(inst, '_model', None)
                            except Exception:
                                setattr(inst, '_model', None)
                            # hydrate only requested scalar fields from mapping
                            try:
                                if requested_scalar_root:
                                    for sf in requested_scalar_root:
                                        try:
                                            setattr(inst, sf, mapping.get(sf, None))
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            # ensure helper root columns (id and FK helpers) are present on instance
                            try:
                                needed_cols: set[str] = set()
                                # id often required for relation resolvers
                                if requested_relations:
                                    needed_cols.add('id')
                                for fk in required_fk_parent_cols:
                                    needed_cols.add(fk)
                                for col_name in needed_cols:
                                    if getattr(inst, col_name, None) is None and (col_name in mapping):
                                        try:
                                            setattr(inst, col_name, mapping.get(col_name))
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            # attach custom scalar field values from labeled select columns
                            if custom_fields:
                                for (cf_name, _) in custom_fields:
                                    try:
                                        if cf_name in mapping:
                                            setattr(inst, cf_name, mapping[cf_name])
                                    except Exception:
                                        pass
                            # reconstruct custom object fields (prefer single JSON column if present)
                            if custom_object_fields:
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
                                    # mapping already available above
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
                                                            # Nested relations on this child (prefetch to avoid N+1) with recursion
                                                            try:
                                                                def _hydrate_nested_list(parent_inst, parent_b, item_dict, nested_meta_src_map):
                                                                    for nname_i, ndef_i in parent_b.__berry_fields__.items():
                                                                        if ndef_i.kind != 'relation':
                                                                            continue
                                                                        raw_nested_i = item_dict.get(nname_i, None)
                                                                        if raw_nested_i is None:
                                                                            continue
                                                                        import json as _json
                                                                        try:
                                                                            parsed_nested_i = _json.loads(raw_nested_i) if isinstance(raw_nested_i, (str, bytes)) else raw_nested_i
                                                                        except Exception:
                                                                            parsed_nested_i = None
                                                                        n_target_i = self.types.get(ndef_i.meta.get('target')) if ndef_i.meta.get('target') else None
                                                                        n_st_i = self._st_types.get(ndef_i.meta.get('target')) if ndef_i.meta.get('target') else None
                                                                        if not n_target_i or not n_target_i.model or not n_st_i:
                                                                            continue
                                                                        if ndef_i.meta.get('single'):
                                                                            if isinstance(parsed_nested_i, dict):
                                                                                ni = n_st_i()
                                                                                for nsf, nsdef in n_target_i.__berry_fields__.items():
                                                                                    if nsdef.kind == 'scalar':
                                                                                        setattr(ni, nsf, parsed_nested_i.get(nsf))
                                                                                setattr(ni, '_model', None)
                                                                                setattr(parent_inst, nname_i, ni)
                                                                                setattr(parent_inst, f"_{nname_i}_prefetched", ni)
                                                                            else:
                                                                                setattr(parent_inst, nname_i, None)
                                                                                setattr(parent_inst, f"_{nname_i}_prefetched", None)
                                                                        else:
                                                                            nlist_i = []
                                                                            if isinstance(parsed_nested_i, list):
                                                                                for nv_i in parsed_nested_i:
                                                                                    if isinstance(nv_i, dict):
                                                                                        ni = n_st_i()
                                                                                        for nsf, nsdef in n_target_i.__berry_fields__.items():
                                                                                            if nsdef.kind == 'scalar':
                                                                                                setattr(ni, nsf, nv_i.get(nsf))
                                                                                        setattr(ni, '_model', None)
                                                                                        # recursively hydrate deeper nested under ni if any
                                                                                        try:
                                                                                            deeper_meta = (nested_meta_src_map.get(nname_i) or {}).get('nested') if isinstance(nested_meta_src_map, dict) else None
                                                                                            _hydrate_nested_list(ni, n_target_i, nv_i, deeper_meta or {})
                                                                                        except Exception:
                                                                                            pass
                                                                                        nlist_i.append(ni)
                                                                            setattr(parent_inst, nname_i, nlist_i)
                                                                            setattr(parent_inst, f"_{nname_i}_prefetched", nlist_i)
                                                                        # record meta for this nested level
                                                                        try:
                                                                            meta_map_nested = getattr(parent_inst, '_pushdown_meta', None)
                                                                            if meta_map_nested is None:
                                                                                meta_map_nested = {}
                                                                                setattr(parent_inst, '_pushdown_meta', meta_map_nested)
                                                                            src_meta = (nested_meta_src_map.get(nname_i) or {}) if isinstance(nested_meta_src_map, dict) else {}
                                                                            meta_map_nested[nname_i] = {
                                                                                'limit': src_meta.get('limit'),
                                                                                'offset': src_meta.get('offset'),
                                                                                'from_pushdown': bool(src_meta.get('from_pushdown', True)),
                                                                                'skip_reason': src_meta.get('skip_reason')
                                                                            }
                                                                        except Exception:
                                                                            pass

                                                                # First level nested under this child
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
                                                                                    # recursively hydrate deeper nested for each next level item
                                                                                    try:
                                                                                        parent_meta = requested_relations.get(rel_name, {})
                                                                                        nested_meta_map = (parent_meta.get('nested') or {})
                                                                                        deeper_meta = (nested_meta_map.get(nname) or {}).get('nested') or {}
                                                                                        _hydrate_nested_list(ni, n_target, nv, deeper_meta)
                                                                                    except Exception:
                                                                                        pass
                                                                                    nlist.append(ni)
                                                                    # Pagination handled in SQL; no Python-side slicing
                                                                    setattr(child_inst, nname, nlist)
                                                                    setattr(child_inst, f"_{nname}_prefetched", nlist)
                                                                    # record nested pushdown meta (deep-aware)
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
                                                                            'from_pushdown': bool(nested_meta_src.get('from_pushdown', True)),
                                                                            'skip_reason': nested_meta_src.get('skip_reason')
                                                                        }
                                                                    except Exception:
                                                                        pass
                                                            except Exception:
                                                                pass
                                                            tmp_list.append(child_inst)
                                                # Remove Python-side ordering for top-level relation; rely on DB ordering
                                                try:
                                                    pass
                                                except Exception:
                                                    pass
                                                # Pagination handled in SQL; no Python-side slicing
                                                built_value = tmp_list
                                        else:
                                            built_value = parsed_value
                                        # Only cache prefetched value for single relation if object is present;
                                        # when None, allow the resolver to perform a targeted DB fetch.
                                        if is_single and built_value is None:
                                            pass
                                        else:
                                            setattr(inst, f"_{rel_name}_prefetched", built_value)
                                        # record pushdown meta for pagination reuse
                                        meta_map = getattr(inst, '_pushdown_meta', None)
                                        if meta_map is None:
                                            meta_map = {}
                                            setattr(inst, '_pushdown_meta', meta_map)
                                        meta_map[rel_name] = {
                                            'limit': rel_cfg.get('limit'),
                                            'offset': rel_cfg.get('offset'),
                                            'from_pushdown': True,
                                            'skip_reason': None
                                        }
                                        # Don't assign to public attribute; resolver will read _prefetched and
                                        # still apply per-call filters/order/pagination deterministically.
                                    else:
                                        # Relation was requested but not pushed down: record skip reason
                                        try:
                                            meta_map = getattr(inst, '_pushdown_meta', None)
                                            if meta_map is None:
                                                meta_map = {}
                                                setattr(inst, '_pushdown_meta', meta_map)
                                            rel_meta_src = requested_relations.get(rel_name, {})
                                            reason_txt = None
                                            try:
                                                reason_txt = (rel_push_status.get(rel_name) or {}).get('reason')
                                            except Exception:
                                                reason_txt = None
                                            meta_map[rel_name] = {
                                                'limit': rel_meta_src.get('limit'),
                                                'offset': rel_meta_src.get('offset'),
                                                'from_pushdown': False,
                                                'skip_reason': reason_txt,
                                            }
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
