from __future__ import annotations
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, get_origin, get_args, Annotated
from enum import Enum
from datetime import datetime
import asyncio
import strawberry
from sqlalchemy import select
from .adapters import get_adapter  # adapter abstraction
from typing import TYPE_CHECKING
from .core.utils import get_db_session as _get_db
from .core.utils import _py_uuid
from strawberry.scalars import JSON as ST_JSON
from sqlalchemy.sql.sqltypes import Integer, String, Boolean, DateTime, JSON as SA_JSON, Numeric as SANumeric
try:
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY as PG_ARRAY, JSONB as PG_JSONB
except Exception:  # pragma: no cover
    PG_UUID = PG_ARRAY = PG_JSONB = object  # type: ignore
try:
    from sqlalchemy.types import TypeDecorator as _SATypeDecorator
except Exception:  # pragma: no cover
    class _SATypeDecorator:  # type: ignore
        pass
try:
    from sqlalchemy import Enum as SAEnumType
except Exception:  # pragma: no cover
    SAEnumType = object  # type: ignore
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
from .core.fields import FieldDef, FieldDescriptor, DomainDescriptor
from .core.filters import FilterSpec, OPERATOR_REGISTRY, normalize_filter_spec as _normalize_filter_spec
from .core.analyzer import QueryAnalyzer
from .core.utils import (
    Direction,
    dir_value as _dir_value,
    coerce_where_value as _coerce_where_value,
    expr_from_where_dict as _expr_from_where_dict,
    to_where_dict as _to_where_dict,
    normalize_order_multi_values as _norm_order_multi,
)
from .sql.builders import RelationSQLBuilders, RootSQLBuilders
from .core.hydration import Hydrator

# --- Standard argument descriptions for default query parameters ---
# These descriptions are attached to GraphQL arguments so that tools and UIs
# (e.g., GraphiQL) show helpful guidance and examples for users.
_ARG_DESC_ORDER_BY = (
    "Name of a scalar field to order by. Use together with order_dir."
    "Example: order_by: \"created_at\""
)
_ARG_DESC_ORDER_DIR = (
    "Sort direction for order_by. Accepted values: asc or desc. If not provided, defaults to asc."
    "Example: order_dir: desc"
)

_ARG_DESC_ORDER_MULTI = (
    "List of ordering specs in 'column:direction' format. Multiple entries define tie-breakers. "
    "Examples: ['id:asc'], ['created_at:desc', 'id:asc']"
)

_ARG_DESC_WHERE = (
    "JSON filter expressed as a JSON object (string). Supports simple operators like "
    "eq, ne, gt, gte, lt, lte, like, ilike, in, between, and/or. "
    "Examples: {\"id\": {\"eq\": 1}}, {\"created_at\": {\"between\": [\"2020-01-01T00:00:00\", \"2020-12-31T23:59:59\"]}}"
)

# --- Public hook descriptor to attach pre/post callbacks declaratively ----
class HooksDescriptor:
    """Descriptor that registers pre/post merge hooks on a BerryType.

    Usage inside a @berry_schema.type class body:

        from app.graphql.hooks.invitation import invitation_pre, invitation_post

        class InvitationQL(BerryType):
            hooks = berry_schema.hooks(pre=invitation_pre, post=invitation_post)

    This avoids direct use of private dunder attributes and keeps hook wiring near the type.
    """
    def __init__(self, pre=None, post=None):
        self._pre = pre
        self._post = post

    def _iter_funcs(self, val):
        try:
            if val is None:
                return []
            if isinstance(val, (list, tuple)):
                return [f for f in val if callable(f)]
            return [val] if callable(val) else []
        except Exception:
            return []

    def __set_name__(self, owner, name):
        # Append to existing pre hooks
        try:
            existing_pre = list(getattr(owner, '__merge_pre_cbs__', ()) or ())
        except Exception:
            existing_pre = []
        pre_funcs = self._iter_funcs(self._pre)
        if pre_funcs:
            for f in pre_funcs:
                if f not in existing_pre:
                    existing_pre.append(f)
            try:
                setattr(owner, '__merge_pre_cbs__', tuple(existing_pre))
            except Exception:
                pass

        # Append to existing post hooks
        try:
            existing_post = list(getattr(owner, '__merge_post_cbs__', ()) or ())
        except Exception:
            existing_post = []
        post_funcs = self._iter_funcs(self._post)
        if post_funcs:
            for f in post_funcs:
                if f not in existing_post:
                    existing_post.append(f)
            try:
                setattr(owner, '__merge_post_cbs__', tuple(existing_post))
            except Exception:
                pass

        # Optionally hide the attribute from instances by replacing with a sentinel
        try:
            setattr(owner, name, None)
        except Exception:
            pass

__all__ = ['BerrySchema', 'BerryType', 'BerryDomain']

UNSET = getattr(strawberry, 'UNSET')

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
        # Pick methods tagged via decorators
        try:
            pre_list = []
            post_list = []
            for k, v in list(namespace.items()):
                try:
                    if getattr(v, '__berry_merge_pre__', False):
                        pre_list.append(v)
                    if getattr(v, '__berry_merge_post__', False):
                        post_list.append(v)
                except Exception:
                    continue
            if pre_list:
                namespace['__merge_pre_cbs__'] = tuple(pre_list)
            if post_list:
                namespace['__merge_post_cbs__'] = tuple(post_list)
        except Exception:
            pass
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
        # Names of domains explicitly attached to Query via DomainDescriptor
        # This ensures we expose only intended domains on Query (and not mutation-only domains)
        self._domains_exposed_on_query = set()

    # No external registration of callbacks; only type decorators are supported.

    def register(self, cls: Type[BerryType]):
        self.types[cls.__name__] = cls
        return cls

    # No external registration method; use @berry_schema.pre/@berry_schema.post on the type.

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
                users = relation('UserQL', scope=..., order_by='id')
                userById = relation('UserQL', single=True, scope=...)
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
                    # Always overwrite to mark this domain exposed on Query
                    self._domains[dom_name] = {'class': dom_cls, 'expose': True, 'options': dict(v.meta)}
                    # Track explicit exposure on Query
                    try:
                        self._domains_exposed_on_query.add(dom_name)
                    except Exception:
                        pass
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

    # ---- Decorators for type-level mutation callbacks (single names) ---
    def pre(self, fn: Optional[Callable[..., Any]] = None):
        """Decorator to mark a method as the pre-merge callback."""
        def _deco(f):
            try:
                setattr(f, '__berry_merge_pre__', True)
            except Exception:
                pass
            return f
        return _deco(fn) if callable(fn) else _deco

    def post(self, fn: Optional[Callable[..., Any]] = None):
        """Decorator to mark a method as the post-merge callback."""
        def _deco(f):
            try:
                setattr(f, '__berry_merge_post__', True)
            except Exception:
                pass
            return f
        return _deco(fn) if callable(fn) else _deco

    # Public factory for hook registration inside type classes
    def hooks(self, *, pre: Any | None = None, post: Any | None = None):
        """Return a descriptor that registers pre/post hooks when set on a type class.

        Example:
            class MyType(BerryType):
                hooks = berry_schema.hooks(pre=pre_fn, post=post_fn)
        """
        return HooksDescriptor(pre=pre, post=post)


    

    # ---------- Input/Mutation helpers ----------
    def _ensure_input_type(self, btype_cls: Type[BerryType], _stack: Optional[List[str]] = None):
        """Build (or return cached) Strawberry input type mirroring a BerryType's writable shape.

        Includes scalar fields and relation fields (recursively using corresponding Input types).
        Aggregates and custom fields are excluded. All fields are optional to support partial updates.
        """
        # Resolve target type name
        try:
            tname = getattr(btype_cls, '__name__', None)
        except Exception:
            tname = None
        if not tname:
            raise ValueError("_ensure_input_type requires a BerryType class")
        # Initialize cycle-detection stack for this build path
        if _stack is None:
            _stack = []
        input_name = f"{tname}Input"
        # Return cached if present
        if input_name in self._st_types:
            return self._st_types[input_name]
        # Create plain class first and cache placeholder to break recursion cycles
        InPlain = type(input_name, (), {'__doc__': f'Input for upsert of {tname}'})
        InPlain.__module__ = __name__
        # Pre-cache placeholder to avoid infinite recursion on self-referential types
        self._st_types[input_name] = InPlain
        anns: Dict[str, Any] = {}

        # Column type map for scalar field type inference
        try:
            model_cls = getattr(btype_cls, 'model', None)
        except Exception:
            model_cls = None
        col_type_map: Dict[str, Any] = self._build_column_type_map(model_cls) if model_cls is not None else {}
        # Set of physical table columns for read-only/computed detection
        try:
            table_cols = set(c.name for c in getattr(getattr(model_cls, '__table__', None), 'columns', []) or []) if model_cls is not None else set()
        except Exception:
            table_cols = set()
        # Map of column name -> SQLAlchemy Column to fetch comments/descriptions
        try:
            col_obj_map: Dict[str, Any] = {c.name: c for c in getattr(getattr(model_cls, '__table__', None), 'columns', []) or []} if model_cls is not None else {}
        except Exception:
            col_obj_map = {}

        # Scalars and relations
        for fname, fdef in (getattr(btype_cls, '__berry_fields__', {}) or {}).items():
            if fdef.kind == 'scalar':
                # Skip private scalars
                if isinstance(fname, str) and fname.startswith('_'):
                    continue
                # Skip computed/read-only scalars from mutation input types
                try:
                    _meta = (getattr(fdef, 'meta', {}) or {})
                    if _meta.get('read_only') or _meta.get('computed'):
                        continue
                except Exception:
                    pass
                src_col = (fdef.meta or {}).get('column') if isinstance(fdef.meta, dict) else None
                # Include write_only fields even if not a real table column
                is_write_only = False
                try:
                    is_write_only = bool((fdef.meta or {}).get('write_only'))
                except Exception:
                    is_write_only = False
                if not is_write_only:
                    # If the source is not a real table column, treat as read-only (computed) and skip
                    try:
                        source_name = src_col or fname
                        if source_name not in table_cols:
                            continue
                    except Exception:
                        pass
                # Allow explicit override of Python type via meta.returns
                try:
                    if is_write_only:
                        py_t = (fdef.meta or {}).get('returns') or str
                    else:
                        py_t = (fdef.meta or {}).get('returns') or col_type_map.get(src_col or fname, str)
                except Exception:
                    py_t = (str if is_write_only else col_type_map.get(src_col or fname, str))
                try:
                    if isinstance(py_t, type) and issubclass(py_t, Enum):
                        # Mirror output: wrap Python Enum into a Strawberry enum for input as well
                        st_enum_name = getattr(py_t, '__name__', 'Enum')
                        st_enum = self._st_types.get(st_enum_name)
                        if st_enum is None:
                            st_enum = strawberry.enum(py_t, name=st_enum_name)  # type: ignore
                            self._st_types[st_enum_name] = st_enum
                            try:
                                globals()[st_enum_name] = st_enum
                            except Exception:
                                pass
                        anns[fname] = Optional[st_enum]  # type: ignore[index]
                    else:
                        anns[fname] = Optional[py_t]
                except Exception:
                    anns[fname] = Optional[str]
                # Compute description from field meta comment or SQLAlchemy Column.comment
                desc: str | None = None
                try:
                    desc = (getattr(fdef, 'meta', {}) or {}).get('comment')
                except Exception:
                    desc = None
                if not desc:
                    try:
                        source_name = src_col or fname
                        col_obj = col_obj_map.get(source_name)
                        if col_obj is not None:
                            desc = getattr(col_obj, 'comment', None)
                    except Exception:
                        pass
                # Use strawberry.field to attach description while keeping UNSET semantics
                try:
                    if desc:
                        setattr(InPlain, fname, strawberry.field(default=UNSET, description=desc))  # type: ignore[arg-type]
                    else:
                        setattr(InPlain, fname, UNSET)
                except Exception:
                    try:
                        if desc:
                            setattr(InPlain, fname, strawberry.field(default=None, description=desc))  # type: ignore[arg-type]
                        else:
                            setattr(InPlain, fname, None)
                    except Exception:
                        setattr(InPlain, fname, None)
            elif fdef.kind == 'relation':
                # Build nested inputs recursively
                # Respect read_only on relations to avoid cycles in input types
                try:
                    if (getattr(fdef, 'meta', {}) or {}).get('read_only'):
                        continue
                except Exception:
                    pass
                target_name = fdef.meta.get('target') if isinstance(fdef.meta, dict) else None
                is_single = bool(fdef.meta.get('single')) if isinstance(fdef.meta, dict) else False
                if not target_name:
                    continue
                target_b = self.types.get(target_name)
                if not target_b:
                    continue
                # If this relation points to an ancestor in the current build stack, include a slim input
                # that only allows identifying fields (pk and simple direct FKs) instead of full recursion.
                # IMPORTANT: Apply slimming only for single (to-one) relations to break cycles, but allow
                # full inputs for list (to-many) relations so nested payloads can include writable FK fields.
                use_slim = False
                try:
                    if (target_name in _stack) and is_single:
                        use_slim = True
                except Exception:
                    use_slim = False
                # Use a slim "RefInput" when the target type is an ancestor in the build stack
                # to prevent infinite recursion. The RefInput includes the target's primary key
                # and any write-only scalar helper fields (e.g., "characterName").
                if use_slim:
                    # Build a minimal ad-hoc input type for the target consisting of pk only.
                    # Name it deterministically for caching: e.g., PlotlineQLRefInput
                    ref_input_name = f"{target_name}RefInput"
                    if ref_input_name in self._st_types:
                        ref_input = self._st_types[ref_input_name]
                    else:
                        RefPlain = type(ref_input_name, (), {'__doc__': f'Reference input for {target_name} (pk only)'})
                        RefPlain.__module__ = __name__
                        # Determine pk field name and type
                        try:
                            t_model = getattr(target_b, 'model', None)
                            pk_name = self._get_pk_name(t_model) if t_model is not None else 'id'
                        except Exception:
                            pk_name = 'id'
                        # pk type: default to str for UUIDs; use column type map when available
                        try:
                            col_map = self._build_column_type_map(t_model) if t_model is not None else {}
                            pk_py_t = col_map.get(pk_name, str)
                        except Exception:
                            pk_py_t = str
                        # Start annotations with pk
                        ref_anns: Dict[str, Any] = {pk_name: Optional[pk_py_t]}
                        # Also include write-only scalar helper fields from the target type
                        try:
                            for tfname, tfdef in (getattr(target_b, '__berry_fields__', {}) or {}).items():
                                if tfdef.kind != 'scalar':
                                    continue
                                try:
                                    if not (tfdef.meta or {}).get('write_only'):
                                        continue
                                except Exception:
                                    continue
                                # Prefer explicit returns for helper types; default to str
                                try:
                                    t_py = (tfdef.meta or {}).get('returns') or col_map.get((tfdef.meta or {}).get('column') or tfname, str)
                                except Exception:
                                    t_py = str
                                ref_anns[tfname] = Optional[t_py]
                        except Exception:
                            pass
                        # Include control flag for delete semantics
                        try:
                            ref_anns['_Delete'] = Optional[bool]
                        except Exception:
                            pass
                        setattr(RefPlain, '__annotations__', ref_anns)
                        # Set defaults to UNSET
                        for an_name in list(ref_anns.keys()):
                            try:
                                setattr(RefPlain, an_name, UNSET)
                            except Exception:
                                setattr(RefPlain, an_name, None)
                        try:
                            ref_input = strawberry.input(RefPlain)  # type: ignore
                        except Exception:
                            ref_input = RefPlain
                        self._st_types[ref_input_name] = ref_input
                        try:
                            globals()[ref_input_name] = ref_input
                        except Exception:
                            pass
                    # Assign the slim ref input for this relation
                    try:
                        if is_single:
                            anns[fname] = Optional[ref_input]  # type: ignore[index]
                        else:
                            anns[fname] = Optional[List[ref_input]]  # type: ignore[index]
                        setattr(InPlain, fname, UNSET)
                    except Exception:
                        anns[fname] = Optional[str]
                        try:
                            setattr(InPlain, fname, None)
                        except Exception:
                            pass
                else:
                    # Recurse while tracking the current path
                    child_input = self._ensure_input_type(target_b, _stack=(_stack + [tname]))
                    try:
                        if is_single:
                            anns[fname] = Optional[child_input]  # type: ignore[index]
                            # Attach relation comment if provided
                            _rel_desc = None
                            try:
                                _rel_desc = (getattr(fdef, 'meta', {}) or {}).get('comment')
                            except Exception:
                                _rel_desc = None
                            if _rel_desc:
                                setattr(InPlain, fname, strawberry.field(default=UNSET, description=_rel_desc))  # type: ignore[arg-type]
                            else:
                                setattr(InPlain, fname, UNSET)
                        else:
                            anns[fname] = Optional[List[child_input]]  # type: ignore[index]
                            _rel_desc = None
                            try:
                                _rel_desc = (getattr(fdef, 'meta', {}) or {}).get('comment')
                            except Exception:
                                _rel_desc = None
                            if _rel_desc:
                                setattr(InPlain, fname, strawberry.field(default=UNSET, description=_rel_desc))  # type: ignore[arg-type]
                            else:
                                setattr(InPlain, fname, UNSET)
                    except Exception:
                        anns[fname] = Optional[str]
                        try:
                            setattr(InPlain, fname, None)
                        except Exception:
                            pass

        # Special control flag: allow delete semantics in mutation payloads
        try:
            anns['_Delete'] = Optional[bool]
            setattr(InPlain, '_Delete', UNSET)
        except Exception:
            try:
                setattr(InPlain, '_Delete', None)
            except Exception:
                pass

        # Now attach annotations and decorate as strawberry input
        setattr(InPlain, '__annotations__', anns)
        try:
            InType = strawberry.input(InPlain)  # type: ignore
        except Exception:
            InType = InPlain
        self._st_types[input_name] = InType
        try:
            globals()[input_name] = InType
        except Exception:
            pass
        return InType

    def _input_to_dict(self, obj: Any) -> Any:
        """Convert a Strawberry input instance (or nested list/dict) to plain Python dicts/lists.

        Best-effort: handles dataclasses and simple objects with __dict__ or attribute access.
        """
        from dataclasses import is_dataclass, fields as _dc_fields
        from enum import Enum as _PyEnum
        # None
        if obj is None:
            return None
        # Primitive
        if isinstance(obj, (str, int, float, bool)):
            return obj
        # Enum -> coerce to its value for DB storage
        try:
            if isinstance(obj, _PyEnum):
                return getattr(obj, 'value', obj.name)
        except Exception:
            pass
        # List/tuple
        if isinstance(obj, (list, tuple)):
            return [self._input_to_dict(x) for x in obj]
        # Dict-like
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                out[k] = self._input_to_dict(v)
            return out
        # Dataclass from strawberry.input
        try:
            if is_dataclass(obj):
                out: Dict[str, Any] = {}
                for f in _dc_fields(obj):
                    try:
                        v = getattr(obj, f.name)
                    except Exception:
                        continue
                    # Skip UNSET (omitted) fields; keep None as explicit null
                    try:
                        if v is UNSET:
                            continue
                    except Exception:
                        pass
                    out[f.name] = self._input_to_dict(v)
                return out
        except Exception:
            pass
        # Generic object: try __dict__
        try:
            d = getattr(obj, '__dict__', None)
            if isinstance(d, dict):
                out = {}
                for k, v in d.items():
                    if k.startswith('_'):
                        continue
                    out[k] = self._input_to_dict(v)
                return out
        except Exception:
            pass
        return obj

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
            # Allow passing a SQLAlchemy Column; if so, unwrap to its .type
            try:
                # A Column-like will have both 'type' and 'info' attributes
                if hasattr(sqlatype, 'type') and hasattr(sqlatype, 'info'):
                    sqlatype = getattr(sqlatype, 'type', sqlatype)
            except Exception:
                pass
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
            # Important: check SQLAlchemy Enum BEFORE String, since Enum is a subclass of String
            # SQLAlchemy Enum -> underlying Python Enum class when available
            try:
                if isinstance(sqlatype, SAEnumType):
                    enum_cls = getattr(sqlatype, 'enum_class', None)
                    if enum_cls is not None:
                        return enum_cls
            except Exception:
                pass
            if isinstance(sqlatype, Integer):
                return int
            if isinstance(sqlatype, Boolean):
                return bool
            if isinstance(sqlatype, DateTime):
                return datetime
            if isinstance(sqlatype, String):
                return str
            # Treat DECIMAL/NUMERIC as float for GraphQL scalar purposes
            try:
                if isinstance(sqlatype, SANumeric):
                    return float
            except Exception:
                pass
            # UUID types (PostgreSQL dialect and SQLAlchemy generic)
            if isinstance(sqlatype, PG_UUID):
                return _py_uuid.UUID
            try:
                from sqlalchemy import Uuid as _SA_UUID  # SQLAlchemy 2.0 generic UUID
            except Exception:
                _SA_UUID = None  # type: ignore[assignment]
            try:
                if _SA_UUID is not None and isinstance(sqlatype, _SA_UUID):
                    return _py_uuid.UUID
            except Exception:
                pass
            if isinstance(sqlatype, PG_ARRAY):
                inner = getattr(sqlatype, 'item_type', None)
                inner_t = self._sa_python_type(inner) if inner is not None else str
                # guard List typing for non-subscriptable case
                try:
                    return List[inner_t]  # type: ignore[index]
                except Exception:
                    return list
            # JSON (cross-dialect) -> Strawberry JSON scalar
            try:
                if isinstance(sqlatype, SA_JSON):
                    return ST_JSON  # type: ignore[return-value]
            except Exception:
                pass
            if isinstance(sqlatype, PG_JSONB):
                return ST_JSON  # type: ignore[return-value]
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
                    # Prefer the first PK column (single-column PK expected)
                    if pk_cols:
                        return pk_cols[0].name
                except Exception:
                    pass
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
                    # Pass the Column itself so _sa_python_type can inspect Column.info for enum hints
                    out[col.name] = self._sa_python_type(col)
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

    def _find_child_fk_column(self, parent_model_cls: Any, child_model_cls: Any, explicit_child_fk_name: Optional[str] = None):
        """Return SQLAlchemy Column on child that references parent via FK.

        Honors explicit_child_fk_name when provided; otherwise scans child's foreign keys
        to locate a column whose FK target table matches parent's table.
        Returns None if not found or models lack __table__ metadata.
        """
        try:
            if child_model_cls is None or not hasattr(child_model_cls, '__table__'):
                return None
            parent_table_name = None
            try:
                parent_table_name = parent_model_cls.__table__.name if hasattr(parent_model_cls, '__table__') else None
            except Exception:
                parent_table_name = None
            for col in child_model_cls.__table__.columns:
                try:
                    if explicit_child_fk_name and col.name == explicit_child_fk_name:
                        return col
                except Exception:
                    pass
                for fk in getattr(col, 'foreign_keys', []) or []:
                    try:
                        if parent_table_name and fk.column.table.name == parent_table_name:
                            return col
                    except Exception:
                        continue
        except Exception:
            return None
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

    # --- Public helpers ---------------------------------------------------
    def from_model(self, type_ref: Any, model_obj: Any) -> Any:
        """Instantiate a Strawberry runtime object for the given Berry type and attach a model.

        Usage:
            instance = berry_schema.from_model('PostQL', post_model)
            # or
            instance = berry_schema.from_model(PostQL, post_model)

        The returned instance has:
          - _model set to the provided SQLAlchemy model instance
          - scalar fields copied from the model (id, title, ...), when present
        """
        try:
            type_name = type_ref if isinstance(type_ref, str) else getattr(type_ref, '__name__', None)
        except Exception:
            type_name = None
        if not type_name:
            raise ValueError("from_model requires a BerryType class or its name")
        st_cls = self._st_types.get(type_name)
        b_cls = self.types.get(type_name)
        if st_cls is None or b_cls is None:
            raise ValueError(f"Unknown Berry type '{type_name}'. Ensure to_strawberry() was called.")
        inst = st_cls()
        try:
            setattr(inst, '_model', model_obj)
        except Exception:
            pass
        # Pre-populate scalar fields from model for GraphQL response objects.
        # Only copy values for actual SQLAlchemy table columns to avoid triggering async/computed props.
        try:
            _cols = list(getattr(getattr(getattr(b_cls, 'model', None), '__table__', None), 'columns', []) or [])
            _col_names = {c.name for c in _cols}
        except Exception:
            _col_names = set()
        for fname, fdef in (getattr(b_cls, '__berry_fields__', {}) or {}).items():
            if getattr(fdef, 'kind', None) == 'scalar':
                if _col_names and fname not in _col_names:
                    continue
                try:
                    _meta = getattr(fdef, 'meta', {}) or {}
                    if _meta.get('write_only'):
                        continue
                except Exception:
                    pass
                # Skip async property getters, if any
                try:
                    import inspect as _ins
                    cls_attr = getattr(type(model_obj), fname, None)
                    if isinstance(cls_attr, property) and _ins.iscoroutinefunction(getattr(cls_attr, 'fget', None)):
                        continue
                except Exception:
                    pass
                # Prefer instance dict to avoid descriptor side effects
                try:
                    if isinstance(getattr(model_obj, '__dict__', None), dict) and fname in model_obj.__dict__:
                        val = model_obj.__dict__.get(fname, None)
                    else:
                        val = getattr(model_obj, fname, None)
                except Exception:
                    val = None
                # Map back DB enum strings to Python Enum if needed
                try:
                    enum_cls = None
                    try:
                        col = getattr(getattr(b_cls, 'model', None).__table__.c, fname)
                        sa_type = getattr(col, 'type', None)
                        if isinstance(sa_type, SAEnumType):
                            enum_cls = getattr(sa_type, 'enum_class', None)
                    except Exception:
                        enum_cls = None
                    if enum_cls is not None and isinstance(val, str):
                        try:
                            val = enum_cls(val)
                        except Exception:
                            try:
                                val = enum_cls[val]
                            except Exception:
                                pass
                    setattr(inst, fname, val)
                except Exception:
                    pass
        return inst

    def to_strawberry(self, *, strawberry_config: Optional[StrawberryConfig] = None):
        # Persist naming behavior for extractor logic
        try:
            # Prefer explicit name_converter when available
            if strawberry_config is not None and hasattr(strawberry_config, 'name_converter'):
                self._name_converter = getattr(strawberry_config, 'name_converter')
                # Derive auto_camel_case hint from converter when present
                self._auto_camel_case = bool(getattr(self._name_converter, 'auto_camel_case', False))
                # Ensure names starting with '_' are preserved (e.g., special control fields like _Delete)
                try:
                    from strawberry.schema.name_converter import NameConverter as _NC  # type: ignore
                    base_nc = self._name_converter
                    class _PreserveUnderscoreNC(_NC):  # type: ignore[misc]
                        def __init__(self, base=None, auto_camel_case: bool = False):
                            super().__init__(auto_camel_case=auto_camel_case)
                            self.auto_camel_case = auto_camel_case
                            self._base = base
                        def apply_naming_config(self, name: str) -> str:  # type: ignore[override]
                            try:
                                if isinstance(name, str) and name.startswith('_'):
                                    return name
                            except Exception:
                                pass
                            if self._base is not None:
                                try:
                                    return self._base.apply_naming_config(name)  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                            return super().apply_naming_config(name)
                    # Replace the provided converter with a wrapper that preserves leading underscores
                    try:
                        ac = bool(getattr(base_nc, 'auto_camel_case', getattr(strawberry_config, 'auto_camel_case', False)))
                    except Exception:
                        ac = bool(getattr(strawberry_config, 'auto_camel_case', False))
                    wrapped = _PreserveUnderscoreNC(base_nc, auto_camel_case=ac)
                    try:
                        # Assign back both to config and to our cache
                        setattr(strawberry_config, 'name_converter', wrapped)
                    except Exception:
                        pass
                    self._name_converter = wrapped
                except Exception:
                    # Best-effort; continue with the provided converter
                    pass
            else:
                # Fallback to auto_camel_case on config if exposed (older Strawberry)
                if strawberry_config is not None and hasattr(strawberry_config, 'auto_camel_case'):
                    self._auto_camel_case = bool(getattr(strawberry_config, 'auto_camel_case'))
                else:
                    self._auto_camel_case = False
                self._name_converter = None
                # When no explicit converter is provided but auto_camel_case could be True,
                # we keep our own reference as None and handle defaults later.
        except Exception:
            self._auto_camel_case = False
            self._name_converter = None
        # Always rebuild Strawberry runtime classes fresh per call to honor config changes
        # and avoid stale definitions across multiple to_strawberry invocations.
        self._st_types = {}
        # Keep a mapping of type name -> description to enforce on the final Strawberry definitions
        type_descriptions: Dict[str, str] = {}
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
            # Prefer model docstring, then SQLAlchemy table comment for type description
            try:
                model_cls_doc = None
                model_cls = getattr(bcls, 'model', None)
                # 1) Class docstring
                model_cls_doc = getattr(model_cls, '__doc__', None)
                # 2) Fallback to table comment
                if not model_cls_doc and hasattr(getattr(model_cls, '__table__', None), 'comment'):
                    model_cls_doc = getattr(model_cls.__table__, 'comment')  # type: ignore[attr-defined]
                if model_cls_doc:
                    # Set both __doc__ (best-effort) and a stable attribute consumed at decoration time
                    setattr(st_cls, '__doc__', model_cls_doc)
                    try:
                        setattr(st_cls, '__berry_description__', str(model_cls_doc))
                        # Also cache on the local map to enforce after decoration (works across Strawberry versions)
                        type_descriptions[name] = str(model_cls_doc)
                    except Exception:
                        pass
            except Exception:
                pass
            # Also capture user-declared annotations on the BerryType subclass itself so we can
            # expose regular strawberry fields (with their own resolvers) alongside Berry fields.
            try:
                user_annotations: Dict[str, Any] = getattr(bcls, '__annotations__', {}) or {}
            except Exception:
                user_annotations = {}
            # column type mapping
            column_type_map: Dict[str, Any] = self._build_column_type_map(getattr(bcls, 'model', None))
            # Helper: build a safe scalar resolver that reads from the attached model
            def _make_scalar_resolver(fname_local: str, src_col_name: Optional[str] = None):
                def _resolver(self, info: StrawberryInfo):  # noqa: D401
                    # Prefer an explicitly set value on the Strawberry instance first.
                    # This allows from_model() or custom resolvers to override values when needed.
                    try:
                        dself = object.__getattribute__(self, '__dict__')
                        if isinstance(dself, dict) and fname_local in dself:
                            return dself.get(fname_local, None)
                    except Exception:
                        pass
                    # Otherwise, read from the attached SQLAlchemy model when available.
                    try:
                        m = getattr(self, '_model', None)
                    except Exception:
                        m = None
                    col_name = src_col_name or fname_local
                    if m is not None:
                        try:
                            dct = getattr(m, '__dict__', None)
                            if isinstance(dct, dict) and col_name in dct:
                                return dct.get(col_name, None)
                        except Exception:
                            pass
                        try:
                            return getattr(m, col_name, None)
                        except Exception:
                            return None
                    # Lastly, return whatever is set on the instance (if any)
                    try:
                        dself = object.__getattribute__(self, '__dict__')
                        return dself.get(fname_local, None) if isinstance(dself, dict) else None
                    except Exception:
                        return None
                return _resolver

            for fname, fdef in bcls.__berry_fields__.items():
                is_private = isinstance(fname, str) and fname.startswith('_')
                if hasattr(st_cls, fname):
                    # don't overwrite existing custom attr
                    pass
                if fdef.kind == 'scalar':
                    if is_private:
                        # Skip exposing private scalars
                        continue
                    # Hide write_only fields from output types entirely
                    try:
                        if (fdef.meta or {}).get('write_only'):
                            continue
                    except Exception:
                        pass
                    # Optional explicit field comment override
                    try:
                        explicit_field_comment = (fdef.meta or {}).get('comment')
                    except Exception:
                        explicit_field_comment = None
                    # Prefer mapped source column for type inference when provided
                    try:
                        src_col = (fdef.meta or {}).get('column')
                    except Exception:
                        src_col = None
                    py_t = column_type_map.get(src_col or fname)
                    if py_t is None:
                        # Try to infer from SQLAlchemy mapped attr (column_property / hybrid)
                        try:
                            model_attr = getattr(getattr(bcls, 'model', None), src_col or fname, None)
                            # Column-like has .type; column_property.expression has type too
                            t = getattr(getattr(model_attr, 'property', None), 'columns', None)
                            if t:
                                col0 = t[0]
                                py_t = self._sa_python_type(getattr(col0, 'type', None))
                            else:
                                sa_type = getattr(model_attr, 'type', None) or getattr(getattr(model_attr, 'expression', None), 'type', None)
                                if sa_type is not None:
                                    py_t = self._sa_python_type(sa_type)
                        except Exception:
                            py_t = None
                    if py_t is None:
                        # Allow explicit override via meta.returns or default to str
                        try:
                            py_t = (fdef.meta or {}).get('returns') or str
                        except Exception:
                            py_t = str
                    # Optional: discover field description from SQLAlchemy mapping metadata
                    field_description = None
                    try:
                        if explicit_field_comment:
                            field_description = explicit_field_comment
                        else:
                            model_cls_local = getattr(bcls, 'model', None)
                            col_name = src_col or fname
                            col_obj = None
                            if hasattr(getattr(model_cls_local, '__table__', None), 'c'):
                                try:
                                    col_obj = getattr(model_cls_local.__table__.c, col_name)
                                except Exception:
                                    try:
                                        col_obj = model_cls_local.__table__.c.get(col_name)
                                    except Exception:
                                        col_obj = None
                            if col_obj is not None:
                                # Column.comment is preferred; fallback to Column.info["description"|"doc"]
                                field_description = getattr(col_obj, 'comment', None)
                                if not field_description:
                                    info_dict = getattr(col_obj, 'info', {}) or {}
                                    field_description = info_dict.get('description') or info_dict.get('doc')
                            # If not a column, attempt relationship / attribute docs
                            if not field_description:
                                try:
                                    from sqlalchemy.inspection import inspect as sa_inspect  # local import to avoid hard dep at module import
                                    mapper = sa_inspect(model_cls_local) if model_cls_local is not None else None
                                except Exception:
                                    mapper = None
                                # RelationshipProperty: prefer .doc then .info['description']
                                try:
                                    if mapper is not None and hasattr(mapper, 'relationships') and (col_name in mapper.relationships):
                                        rel = mapper.relationships[col_name]
                                        field_description = getattr(rel, 'doc', None)
                                        if not field_description:
                                            rel_info = getattr(rel, 'info', {}) or {}
                                            field_description = rel_info.get('description') or rel_info.get('doc')
                                except Exception:
                                    pass
                                # Python attribute docstring (hybrid_property, @property, descriptor)
                                if not field_description and model_cls_local is not None:
                                    try:
                                        model_attr2 = getattr(model_cls_local, col_name, None)
                                        field_description = getattr(model_attr2, '__doc__', None)
                                    except Exception:
                                        pass
                    except Exception:
                        field_description = None
                    # Convert Python Enum classes to Strawberry enums on the fly
                    try:
                        is_enum_type = isinstance(py_t, type) and issubclass(py_t, Enum)
                        if is_enum_type:
                            # Build enum values list for description augmentation
                            try:
                                members = list(py_t)  # type: ignore[arg-type]
                                vals = []
                                for m in members:
                                    try:
                                        v = getattr(m, 'value', None)
                                        vals.append(str(v if v is not None else getattr(m, 'name', str(m))))
                                    except Exception:
                                        vals.append(str(getattr(m, 'name', str(m))))
                                values_str = ", ".join(vals)
                                enum_values_desc = f"Values: {values_str}" if values_str else None
                            except Exception:
                                enum_values_desc = None
                        else:
                            enum_values_desc = None
                        if is_enum_type:
                            # Cache by name to reuse across types
                            st_enum_name = getattr(py_t, '__name__', 'Enum')
                            st_enum = self._st_types.get(st_enum_name)
                            if st_enum is None:
                                st_enum = strawberry.enum(py_t, name=st_enum_name)  # type: ignore
                                self._st_types[st_enum_name] = st_enum
                                try:
                                    globals()[st_enum_name] = st_enum
                                except Exception:
                                    pass
                            annotations[fname] = Optional[st_enum]  # type: ignore[index]
                        else:
                            annotations[fname] = Optional[py_t]
                    except Exception:
                        annotations[fname] = Optional[py_t]
                    # Attach Strawberry field with resolver so values come from the attached model
                    try:
                        # If enum, append values list to description or set it when absent
                        try:
                            if enum_values_desc:
                                if field_description:
                                    field_description = f"{field_description} | {enum_values_desc}"
                                else:
                                    field_description = enum_values_desc
                        except Exception:
                            pass
                        # Prefer explicit mapped source column name when provided
                        try:
                            src_col_name = (fdef.meta or {}).get('column')
                        except Exception:
                            src_col_name = None
                        if field_description:
                            setattr(st_cls, fname, strawberry.field(resolver=_make_scalar_resolver(fname, src_col_name), description=str(field_description)))
                        else:
                            setattr(st_cls, fname, strawberry.field(resolver=_make_scalar_resolver(fname, src_col_name)))
                    except Exception:
                        # As a fallback, expose a plain attribute with None default
                        setattr(st_cls, fname, None)
                elif fdef.kind == 'relation':
                    target_name = fdef.meta.get('target')
                    is_single = bool(fdef.meta.get('single'))
                    # Relation description: prefer explicit meta.comment; else SQLAlchemy relationship doc/info; else target model's docstring or table comment
                    relation_description = None
                    try:
                        relation_description = (fdef.meta or {}).get('comment')
                    except Exception:
                        relation_description = None
                    # If no explicit comment, inspect the ORM relationship for doc/info
                    if not relation_description:
                        try:
                            model_cls_local = getattr(bcls, 'model', None)
                            try:
                                rel_attr_name = (fdef.meta or {}).get('source') or fname
                            except Exception:
                                rel_attr_name = fname
                            if model_cls_local is not None:
                                try:
                                    from sqlalchemy.inspection import inspect as sa_inspect
                                    mapper = sa_inspect(model_cls_local)
                                except Exception:
                                    mapper = None
                                if mapper is not None and hasattr(mapper, 'relationships') and (rel_attr_name in mapper.relationships):
                                    rel_prop = mapper.relationships[rel_attr_name]
                                    # Prefer relationship .doc; fallback to .info['description'|'doc']
                                    relation_description = getattr(rel_prop, 'doc', None)
                                    if not relation_description:
                                        try:
                                            rinfo = getattr(rel_prop, 'info', {}) or {}
                                            relation_description = rinfo.get('description') or rinfo.get('doc')
                                        except Exception:
                                            relation_description = None
                        except Exception:
                            relation_description = None
                    if not relation_description and target_name:
                        try:
                            tb = self.types.get(target_name)
                            if tb and getattr(tb, 'model', None) is not None:
                                # Prefer target model docstring first
                                relation_description = getattr(tb.model, '__doc__', None)
                                if not relation_description:
                                    relation_description = getattr(getattr(tb.model, '__table__', None), 'comment', None)
                        except Exception:
                            relation_description = None
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
                                        import inspect
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
                            # Acquire per-request lock for DB I/O in relation resolvers as well
                            lock = self.__berry_registry__._get_context_lock(info)
                            target_btype = self.__berry_registry__.types.get(target_name_i)
                            if not target_btype or not target_btype.model:
                                return None if is_single_value else []
                            child_model_cls = target_btype.model
                            if is_single_value:
                                candidate_fk_val = None
                                fallback_parent_id = None
                                # Allow explicit FK override on parent -> child
                                try:
                                    explicit_fk_name = meta_copy.get('fk_column_name') or (getattr(self, '_pushdown_meta', {}) or {}).get(fname_local, {}).get('fk_column_name')
                                except Exception:
                                    explicit_fk_name = None
                                # Try helper '<relation>_id' on parent instance
                                try:
                                    if explicit_fk_name:
                                        candidate_fk_val = getattr(self, explicit_fk_name, None)
                                    else:
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
                                        # If explicit FK override provided, use it directly
                                        if explicit_fk_name and col.name == explicit_fk_name:
                                            try:
                                                candidate_fk_val = getattr(parent_model, col.name)
                                            except Exception:
                                                candidate_fk_val = None
                                            if candidate_fk_val is not None:
                                                break
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
                                    # Attempt to find an FK from child -> parent (common case), honoring explicit override
                                    try:
                                        explicit_child_fk_name = meta_copy.get('fk_column_name') or (getattr(self, '_pushdown_meta', {}) or {}).get(fname_local, {}).get('fk_column_name')
                                    except Exception:
                                        explicit_child_fk_name = None
                                    child_fk_to_parent = self.__berry_registry__._find_child_fk_column(
                                        parent_model.__class__ if parent_model is not None else None,
                                        child_model_cls,
                                        explicit_child_fk_name,
                                    )
                                    if child_fk_to_parent is not None:
                                        # Build a select for the first child row for this parent
                                        stmt = _select(child_model_cls).where(child_fk_to_parent == fallback_parent_id)
                                        # Apply where/default_where and filter args similarly to list path
                                        # Apply relation-level scope (default filter) and argument-provided where
                                        eff_scope = meta_copy.get('scope')
                                        if related_where is not None or eff_scope is not None:
                                            if related_where is not None:
                                                wdict = _to_where_dict(related_where, strict=True)
                                                if wdict:
                                                    expr = _expr_from_where_dict(child_model_cls, wdict, strict=True)
                                                    if expr is not None:
                                                        stmt = stmt.where(expr)
                                            # 'scope' replaces legacy 'where' for schema default filters
                                            dwhere = eff_scope
                                            if dwhere is not None:
                                                if isinstance(dwhere, (dict, str)):
                                                    wdict = _to_where_dict(dwhere, strict=True)
                                                    if wdict:
                                                        expr = _expr_from_where_dict(child_model_cls, wdict, strict=True)
                                                        if expr is not None:
                                                            stmt = stmt.where(expr)
                                                elif callable(dwhere):
                                                    expr = dwhere(child_model_cls, info)
                                                    if expr is not None:
                                                        stmt = stmt.where(expr)
                                        # Also apply type-level scope when present
                                        try:
                                            tgt_b_for_type = self.__berry_registry__.types.get(target_name_i)
                                            t_scope = getattr(tgt_b_for_type, '__type_scope__', None) if tgt_b_for_type is not None else None
                                            if t_scope is None and tgt_b_for_type is not None:
                                                t_scope = getattr(tgt_b_for_type, 'scope', None)
                                        except Exception:
                                            t_scope = None
                                        if t_scope is not None:
                                            fragments = t_scope if isinstance(t_scope, list) else [t_scope]
                                            for frag in fragments:
                                                if isinstance(frag, (dict, str)):
                                                    wdict_t = _to_where_dict(frag, strict=True)
                                                    if wdict_t:
                                                        expr_t = _expr_from_where_dict(child_model_cls, wdict_t, strict=True)
                                                        if expr_t is not None:
                                                            stmt = stmt.where(expr_t)
                                                elif callable(frag):
                                                    expr_t = frag(child_model_cls, info)
                                                    if expr_t is not None:
                                                        stmt = stmt.where(expr_t)
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
                                        async with lock:
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
                                    eff_scope = meta_copy.get('scope')
                                    if related_where is not None or eff_scope is not None:
                                        # Strict for user-provided; permissive for schema default
                                        wdict_arg = _to_where_dict(related_where, strict=True) if related_where is not None else None
                                        if wdict_arg:
                                            expr = _expr_from_where_dict(child_model_cls, wdict_arg, strict=True)
                                            if expr is not None:
                                                stmt = stmt.where(expr)
                                        dwhere = eff_scope
                                        if dwhere is not None:
                                            if isinstance(dwhere, (dict, str)):
                                                wdict_def = _to_where_dict(dwhere, strict=True)
                                                if wdict_def:
                                                    expr = _expr_from_where_dict(child_model_cls, wdict_def, strict=True)
                                                    if expr is not None:
                                                        stmt = stmt.where(expr)
                                            elif callable(dwhere):
                                                expr = dwhere(child_model_cls, info)
                                                if expr is not None:
                                                    stmt = stmt.where(expr)
                                    # Also apply type-level scope when present
                                    try:
                                        tgt_b_for_type2 = self.__berry_registry__.types.get(target_name_i)
                                        t_scope2 = getattr(tgt_b_for_type2, '__type_scope__', None) if tgt_b_for_type2 is not None else None
                                        if t_scope2 is None and tgt_b_for_type2 is not None:
                                            t_scope2 = getattr(tgt_b_for_type2, 'scope', None)
                                    except Exception:
                                        t_scope2 = None
                                    if t_scope2 is not None:
                                        fragments2 = t_scope2 if isinstance(t_scope2, list) else [t_scope2]
                                        for frag2 in fragments2:
                                            if isinstance(frag2, (dict, str)):
                                                wdict_t2 = _to_where_dict(frag2, strict=True)
                                                if wdict_t2:
                                                    expr_t2 = _expr_from_where_dict(child_model_cls, wdict_t2, strict=True)
                                                    if expr_t2 is not None:
                                                        stmt = stmt.where(expr_t2)
                                            elif callable(frag2):
                                                expr_t2 = frag2(child_model_cls, info)
                                                if expr_t2 is not None:
                                                    stmt = stmt.where(expr_t2)
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
                                    async with lock:
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
                                    async with lock:
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
                            # Allow explicit override for child->parent FK
                            try:
                                explicit_child_fk = meta_copy.get('fk_column_name') or (getattr(self, '_pushdown_meta', {}) or {}).get(fname_local, {}).get('fk_column_name')
                            except Exception:
                                explicit_child_fk = None
                            fk_col = self.__berry_registry__._find_child_fk_column(
                                parent_model_cls if parent_model_cls is not None else (parent_model.__class__ if parent_model is not None else None),
                                child_model_cls,
                                explicit_child_fk,
                            )
                            if fk_col is None:
                                return []
                            from sqlalchemy import select as _select
                            # Determine parent id value robustly:
                            # Prefer ORM model's PK when available; otherwise attempt to read a non-callable
                            # attribute on the Strawberry instance (guarding against resolver functions).
                            if parent_model is not None:
                                try:
                                    pk_name_parent = self.__berry_registry__._get_pk_name(parent_model.__class__)
                                    parent_id_val = getattr(parent_model, pk_name_parent, None)
                                except Exception:
                                    parent_id_val = None
                            else:
                                # Try by declared PK name on the Berry type, fallback to 'id'
                                pk_attr_name = None
                                try:
                                    parent_model_cls2 = getattr(parent_btype_local, 'model', None)
                                    if parent_model_cls2 is not None:
                                        pk_attr_name = self.__berry_registry__._get_pk_name(parent_model_cls2)
                                except Exception:
                                    pk_attr_name = None
                                if not pk_attr_name:
                                    pk_attr_name = 'id'
                                parent_id_val = getattr(self, pk_attr_name, None)
                                # If the attribute is a resolver/callable, ignore and fallback to instance dict
                                try:
                                    if callable(parent_id_val):
                                        parent_id_val = getattr(self, '__dict__', {}).get(pk_attr_name, None)
                                except Exception:
                                    pass
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
                            # Normalize GraphQL names to Python field names when auto-camel-case is enabled or a name converter is used,
                            # then filter to scalars known on target type
                            try:
                                scalars_on_target = set()
                                for sf, sd in self.__berry_registry__.types[target_name_i].__berry_fields__.items():
                                    if sd.kind == 'scalar':
                                        # Hide write_only from SQL projection set
                                        try:
                                            if (getattr(sd, 'meta', {}) or {}).get('write_only'):
                                                continue
                                        except Exception:
                                            pass
                                        scalars_on_target.add(sf)
                                # Prepare reverse mapping helper: GraphQL name -> python field name
                                def _to_python_name(gql_name: str) -> str:
                                    # direct match first
                                    if gql_name in scalars_on_target:
                                        return gql_name
                                    # auto camelCase support
                                    try:
                                        if getattr(self.__berry_registry__, '_auto_camel_case', False):
                                            # decamelize
                                            n = str(gql_name)
                                            out = []
                                            for i, ch in enumerate(n):
                                                if ch.isupper() and i > 0 and n[i-1] != '_':
                                                    out.append('_')
                                                out.append(ch.lower())
                                            snake = ''.join(out)
                                            if snake in scalars_on_target:
                                                return snake
                                    except Exception:
                                        pass
                                    # try configured name converter
                                    try:
                                        name_conv = getattr(getattr(self.__berry_registry__, '_name_converter', None), 'apply_naming_config', None)
                                        if callable(name_conv):
                                            for py in scalars_on_target:
                                                try:
                                                    if name_conv(py) == gql_name:
                                                        return py
                                                except Exception:
                                                    continue
                                    except Exception:
                                        pass
                                    return str(gql_name)
                            except Exception:
                                scalars_on_target = set()
                            # Map requested field names to python names when possible
                            mapped_requested = []
                            try:
                                for _fn in requested_fields:
                                    mapped = _to_python_name(_fn)
                                    mapped_requested.append(mapped)
                            except Exception:
                                mapped_requested = list(requested_fields)
                            for fn in mapped_requested:
                                if fn == 'id':
                                    continue
                                if scalars_on_target and fn not in scalars_on_target:
                                    continue
                                # Honor Berry field alias mapping (meta.column) for scalars
                                source_name = None
                                try:
                                    fdef_req = self.__berry_registry__.types[target_name_i].__berry_fields__.get(fn)
                                    if fdef_req and getattr(fdef_req, 'kind', None) == 'scalar':
                                        source_name = (getattr(fdef_req, 'meta', {}) or {}).get('column')
                                except Exception:
                                    source_name = None
                                if source_name:
                                    try:
                                        src_col_obj = child_model_cls.__table__.c.get(source_name)
                                    except Exception:
                                        src_col_obj = None
                                    if src_col_obj is not None:
                                        try:
                                            cols.append(src_col_obj.label(fn))
                                        except Exception:
                                            cols.append(src_col_obj)
                                        continue
                                # Fallback to selecting column by the same name
                                try:
                                    col_obj = child_model_cls.__table__.c.get(fn)
                                except Exception:
                                    col_obj = None
                                if col_obj is not None:
                                    cols.append(col_obj.label(fn))
                            # Also include FK helper columns for any requested nested single relations,
                            # so their resolvers can work even when the FK wasn't explicitly selected
                            try:
                                rel_fields_on_target = {
                                    rf: rd for rf, rd in self.__berry_registry__.types[target_name_i].__berry_fields__.items()
                                    if rd.kind == 'relation'
                                }
                            except Exception:
                                rel_fields_on_target = {}
                            for rf_name, rf_def in rel_fields_on_target.items():
                                # Only if relation is requested directly under this selection
                                if rf_name not in requested_fields:
                                    continue
                                # Only for single relations (need <rel>_id)
                                try:
                                    is_single_rel = bool(rf_def.meta.get('single'))
                                except Exception:
                                    is_single_rel = False
                                if not is_single_rel:
                                    continue
                                # Compute target model to find FK direction
                                try:
                                    target_rel_name = rf_def.meta.get('target')
                                    target_rel_btype = self.__berry_registry__.types.get(target_rel_name) if target_rel_name else None
                                    target_rel_model = target_rel_btype.model if target_rel_btype and target_rel_btype.model else None
                                except Exception:
                                    target_rel_model = None
                                # Prefer explicit '<rel>_id' column when present; otherwise discover via helper
                                fk_col_obj = None
                                try:
                                    fk_col_obj = child_model_cls.__table__.c.get(f"{rf_name}_id")
                                except Exception:
                                    fk_col_obj = None
                                if fk_col_obj is None and target_rel_model is not None:
                                    try:
                                        # honor any explicit fk configured on the relation definition
                                        explicit_child_fk_name = None
                                        try:
                                            explicit_child_fk_name = (rf_def.meta or {}).get('fk_column_name')
                                        except Exception:
                                            explicit_child_fk_name = None
                                        fk_col_obj = self.__berry_registry__._find_child_fk_column(target_rel_model, child_model_cls, explicit_child_fk_name)
                                    except Exception:
                                        fk_col_obj = None
                                if fk_col_obj is not None:
                                    # Label as '<rel>_id' so nested single resolvers can pick it up
                                    try:
                                        label_name = f"{rf_name}_id"
                                        # Avoid duplicate additions
                                        if all(getattr(cl, 'name', None) != label_name for cl in cols):
                                            cols.append(fk_col_obj.label(label_name))
                                    except Exception:
                                        # Best-effort; ignore if labeling fails
                                        pass
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
                            eff_scope = meta_copy.get('scope')
                            if related_where is not None or eff_scope is not None:
                                # Strictly validate and apply argument-provided where
                                if related_where is not None:
                                    wdict = _to_where_dict(related_where, strict=True)
                                    if wdict:
                                        expr = _expr_from_where_dict(child_model_cls, wdict, strict=True)
                                        if expr is not None:
                                            stmt = stmt.where(expr)
                                # Default where from schema meta: enforce strict to surface issues
                                dwhere = eff_scope
                                if dwhere is not None:
                                    if isinstance(dwhere, (dict, str)):
                                        wdict = _to_where_dict(dwhere, strict=True)
                                        if wdict:
                                            expr = _expr_from_where_dict(child_model_cls, wdict, strict=True)
                                            if expr is not None:
                                                stmt = stmt.where(expr)
                                    elif callable(dwhere):
                                        expr = dwhere(child_model_cls, info)
                                        if expr is not None:
                                            stmt = stmt.where(expr)
                            # Combine with type-level default scope for list fallback
                            try:
                                tgt_b_for_type3 = self.__berry_registry__.types.get(target_name_i)
                                t_scope3 = getattr(tgt_b_for_type3, '__type_scope__', None) if tgt_b_for_type3 is not None else None
                                if t_scope3 is None and tgt_b_for_type3 is not None:
                                    t_scope3 = getattr(tgt_b_for_type3, 'scope', None)
                            except Exception:
                                t_scope3 = None
                            if t_scope3 is not None:
                                fragments3 = t_scope3 if isinstance(t_scope3, list) else [t_scope3]
                                for frag3 in fragments3:
                                    if isinstance(frag3, (dict, str)):
                                        wdict_t3 = _to_where_dict(frag3, strict=True)
                                        if wdict_t3:
                                            expr_t3 = _expr_from_where_dict(child_model_cls, wdict_t3, strict=True)
                                            if expr_t3 is not None:
                                                stmt = stmt.where(expr_t3)
                                    elif callable(frag3):
                                        expr_t3 = frag3(child_model_cls, info)
                                        if expr_t3 is not None:
                                            stmt = stmt.where(expr_t3)
                            # Ad-hoc JSON where for relation list if present on selection (not used; keep future hook comment)
                            # Ordering (multi then single) if column whitelist permits
                            allowed_order = getattr(target_cls_i, '__ordering__', None)
                            if allowed_order is None:
                                # derive from scalar fields
                                allowed_order = [sf for sf, sd in self.__berry_registry__.types[target_name_i].__berry_fields__.items() if sd.kind == 'scalar']
                            applied_any = False
                            # Validate invalid order_by up front
                            if order_by and isinstance(order_by, str) and order_by not in allowed_order:
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
                            if not applied_any and order_by:
                                # Accept callable (lambda M, info -> expr) or direct SA expression
                                expr = None
                                if callable(order_by):
                                    try:
                                        expr = order_by(child_model_cls, info)
                                    except Exception:
                                        expr = None
                                elif hasattr(order_by, 'desc') or hasattr(order_by, 'asc'):
                                    expr = order_by
                                elif isinstance(order_by, str) and order_by in allowed_order:
                                    try:
                                        expr = child_model_cls.__table__.c.get(order_by)
                                    except Exception:
                                        expr = None
                                if expr is not None:
                                    descending = _dir_value(order_dir) == 'desc' if order_dir is not None else False
                                    try:
                                        # Prefer calling .desc/.asc when available
                                        if hasattr(expr, 'desc') and hasattr(expr, 'asc'):
                                            stmt = stmt.order_by(expr.desc() if descending else expr.asc())
                                        else:
                                            stmt = stmt.order_by(expr)
                                        applied_any = True
                                    except Exception:
                                        pass
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
                                        expr2 = None
                                        # Support callables and SA expressions in schema default too
                                        if callable(cn):
                                            try:
                                                expr2 = cn(child_model_cls, info)
                                            except Exception:
                                                expr2 = None
                                        elif hasattr(cn, 'desc') or hasattr(cn, 'asc'):
                                            expr2 = cn
                                        else:
                                            col = child_model_cls.__table__.c.get(cn)
                                            expr2 = col
                                        if expr2 is not None:
                                            try:
                                                if hasattr(expr2, 'desc') and hasattr(expr2, 'asc'):
                                                    stmt = stmt.order_by(expr2.desc() if dd=='desc' else expr2.asc())
                                                else:
                                                    stmt = stmt.order_by(expr2)
                                                applied_any = True
                                            except Exception:
                                                pass
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
                            async with lock:
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
                        # Build dynamic wrapper; for single relations do not expose limit/offset/order params
                        arg_defs = []
                        for a in target_filters.keys():
                            arg_defs.append(f"{a}=None")
                        if is_single_value:
                            params = 'self, info, where=None'
                            if arg_defs:
                                params += ', ' + ', '.join(arg_defs)
                        else:
                            params = 'self, info, limit=None, offset=None, order_by=None, order_dir=None, order_multi=None, where=None'
                            if arg_defs:
                                params += ', ' + ', '.join(arg_defs)
                        fname_inner = f"_rel_{fname_local}_resolver"
                        src = f"async def {fname_inner}({params}):\n"
                        src += "    _fa={}\n"
                        for a in target_filters.keys():
                            src += f"    _fa['{a}']={a}\n"
                        if is_single_value:
                            # pass None for non-exposed params
                            src += "    return await _impl(self, info, None, None, None, None, None, where, _fa)\n"
                        else:
                            src += "    return await _impl(self, info, limit, offset, order_by, order_dir, order_multi, where, _fa)\n"
                        env: Dict[str, Any] = {'_impl': _impl}
                        exec(src, env)
                        fn = env[fname_inner]
                        if not getattr(fn, '__module__', None):  # ensure module for strawberry introspection
                            fn.__module__ = __name__
                        # annotations
                        if is_single_value:
                            anns: Dict[str, Any] = {
                                'info': StrawberryInfo,
                                'where': Annotated[Optional[str], strawberry.argument(description=_ARG_DESC_WHERE)],
                            }
                        else:
                            anns: Dict[str, Any] = {
                                'info': StrawberryInfo,
                                'limit': Optional[int],
                                'offset': Optional[int],
                                'order_by': Annotated[Optional[str], strawberry.argument(description=_ARG_DESC_ORDER_BY)],
                                'order_dir': Annotated[Optional[Direction], strawberry.argument(description=_ARG_DESC_ORDER_DIR)],
                                'order_multi': Annotated[Optional[List[str]], strawberry.argument(description=_ARG_DESC_ORDER_MULTI)],
                                'where': Annotated[Optional[str], strawberry.argument(description=_ARG_DESC_WHERE)],
                            }
                        # crude type inference: map to Optional[str|int|bool|datetime] based on target model columns
                        target_b = self.types.get(meta_copy.get('target')) if meta_copy.get('target') else None
                        col_type_map: Dict[str, Any] = {}
                        if target_b and target_b.model and hasattr(target_b.model, '__table__'):
                            col_type_map = self._build_column_type_map(target_b.model)
                        for a, spec in target_filters.items():
                            base_t = getattr(spec, 'arg_type', None) or str
                            if base_t is str and spec.column and spec.column in col_type_map:
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
                        if relation_description:
                            setattr(st_cls, fname, strawberry.field(resolver=_make_relation_resolver(), description=str(relation_description)))
                        else:
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
                            """Resolve aggregate values.

                            Improvement: when a nested object was hydrated purely from pushdown JSON
                            (so self._model is absent), count aggregates previously returned 0.
                            We now attempt a synthetic count using the instance's primary key attribute
                            if available (e.g., "id") so that count(...) works consistently for nested
                            pushdown results.
                            """
                            cache = getattr(self, '_agg_cache', None)
                            is_count_local = meta_copy.get('op') == 'count' or 'count' in meta_copy.get('ops', [])
                            if is_count_local and cache is not None:
                                key = meta_copy.get('cache_key') or (meta_copy.get('source') + ':count')
                                if key in cache:
                                    return cache[key]

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

                            # Determine parent model instance (if available) OR fallback to id attribute
                            parent_model = getattr(self, '_model', None)
                            parent_model_cls = parent_model.__class__ if parent_model is not None else getattr(bcls_local, 'model', None)

                            # Honor explicit fk if present on the relation meta
                            try:
                                explicit_child_fk_name = (rel_def.meta or {}).get('fk_column_name')
                            except Exception:
                                explicit_child_fk_name = None
                            fk_col = None
                            if parent_model_cls is not None:
                                fk_col = self.__berry_registry__._find_child_fk_column(parent_model_cls, child_model_cls, explicit_child_fk_name)
                            if fk_col is None:
                                if is_count_local:
                                    return 0
                                return None

                            if not is_count_local:
                                # Only count aggregates currently supported here; others default to None
                                return None

                            session = _get_db(info)
                            if session is None:
                                return 0

                            from sqlalchemy import func, select as _select
                            parent_pk_value = None
                            try:
                                # Prefer real model instance
                                if parent_model is not None:
                                    pk_name_parent = self.__berry_registry__._get_pk_name(parent_model.__class__)
                                    parent_pk_value = getattr(parent_model, pk_name_parent)
                                else:
                                    # Fallback: derive pk from the object attribute (e.g., id) when model missing
                                    if parent_model_cls is not None:
                                        pk_name_parent = self.__berry_registry__._get_pk_name(parent_model_cls)
                                        parent_pk_value = getattr(self, pk_name_parent, None)
                            except Exception:
                                parent_pk_value = None

                            if parent_pk_value is None:
                                return 0

                            try:
                                stmt = _select(func.count()).select_from(child_model_cls).where(fk_col == parent_pk_value)
                                result = await session.execute(stmt)
                                val = result.scalar_one() or 0
                            except Exception:
                                val = 0

                            key = meta_copy.get('cache_key') or (source + ':count')
                            if cache is None:
                                cache = {}
                                setattr(self, '_agg_cache', cache)
                            cache[key] = val
                            return val
                        return aggregate_resolver
                    agg_comment = None
                    try:
                        agg_comment = (fdef.meta or {}).get('comment')
                    except Exception:
                        agg_comment = None
                    if agg_comment:
                        setattr(st_cls, fname, strawberry.field(_make_aggregate_resolver(), description=str(agg_comment)))
                    else:
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
                            # Fast-path: if root query already populated attribute (no N+1), return it.
                            # Important: avoid accidentally returning the bound resolver method itself
                            # when the attribute isn't set on the instance (__dict__).
                            pre_value = None
                            try:
                                # Prefer instance dict value if present (set by hydration)
                                if hasattr(self, '__dict__') and fname_local in getattr(self, '__dict__', {}):
                                    pre_value = self.__dict__[fname_local]
                                else:
                                    # Fallback to explicit prefetched marker used by hydrators
                                    pre_value = getattr(self, f"_{fname_local}_prefetched", None)
                            except Exception:
                                pre_value = None
                            # Only short-circuit when we have a real value (and not a callable)
                            if pre_value is not None and not callable(pre_value):
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
                            # Try builder(model_cls) for robust evaluation; scope by PK when possible
                            try:
                                # Determine model class for the parent
                                model_cls_for_builder = type(parent_model)
                                if len(inspect.signature(builder).parameters) == 1:
                                    result_obj = builder(model_cls_for_builder)
                                else:
                                    result_obj = builder(model_cls_for_builder, session)
                            except Exception:
                                try:
                                    result_obj = builder(model_cls_for_builder)
                                except Exception:
                                    return None
                            if asyncio.iscoroutine(result_obj):
                                result_obj = await result_obj
                            try:
                                from sqlalchemy.sql import Select as _Select  # type: ignore
                            except Exception:
                                _Select = None  # type: ignore
                            if _Select is not None and isinstance(result_obj, _Select):
                                # If possible, constrain the SELECT to this instance by primary key
                                try:
                                    from sqlalchemy.inspection import inspect as _sa_inspect  # type: ignore
                                    pk_cols = list(getattr(_sa_inspect(model_cls_for_builder), 'primary_key', []) or [])
                                    if pk_cols:
                                        pk_col = pk_cols[0]
                                        pk_val = getattr(parent_model, getattr(pk_col, 'key', 'id'), None)
                                        if pk_val is not None:
                                            try:
                                                result_obj = result_obj.where(pk_col == pk_val)
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
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
                    c_comment = None
                    try:
                        c_comment = (fdef.meta or {}).get('comment')
                    except Exception:
                        c_comment = None
                    if c_comment:
                        setattr(st_cls, fname, strawberry.field(_make_custom_resolver(), description=str(c_comment)))
                    else:
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
                    co_comment = None
                    try:
                        co_comment = (fdef.meta or {}).get('comment')
                    except Exception:
                        co_comment = None
                    if co_comment:
                        setattr(st_cls, fname, strawberry.field(_make_custom_obj_resolver(), description=str(co_comment)))
                    else:
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
                # Prefer explicit description when available to ensure introspection shows it
                desc = None
                try:
                    desc = getattr(cls, '__berry_description__', None)
                except Exception:
                    desc = None
                if desc:
                    self._st_types[name] = strawberry.type(cls, description=desc)  # type: ignore
                else:
                    self._st_types[name] = strawberry.type(cls)  # type: ignore
        # Post-decoration safety net: directly set the description on the Strawberry definitions
        # using the cached map (more reliable than reading attributes from the decorated class).
        try:
            for _tname, _desc in list(type_descriptions.items()):
                try:
                    _tcls = self._st_types.get(_tname)
                except Exception:
                    _tcls = None
                if _tcls is None:
                    continue
                try:
                    defn = getattr(_tcls, '__strawberry_definition__', None)
                    if defn is not None and getattr(defn, 'description', None) in (None, ''):
                        setattr(defn, 'description', str(_desc))
                except Exception:
                    pass
        except Exception:
            pass
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
                base_type = getattr(f_spec, 'arg_type', None) or str
                if base_type is str and f_spec.column and f_spec.column in col_py_types:
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
                requested_other_root = getattr(_plan, 'requested_other_root', set()) or set()
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
                    # Only validate against allowed field names when order_by is a string.
                    # Callables and SQLAlchemy expressions are allowed and handled later.
                    if isinstance(ob_rel, str) and ob_rel and ob_rel not in allowed_fields_rel:
                        raise ValueError(f"Invalid order_by '{ob_rel}'. Allowed: {allowed_fields_rel}")
                    child_model_cls = target_b.model
                    # Determine FK from child->parent, allow explicit override via relation meta
                    try:
                        explicit_child_fk = rel_cfg.get('fk_column_name')
                    except Exception:
                        explicit_child_fk = None
                    fk_col = self._find_child_fk_column(model_cls, child_model_cls, explicit_child_fk)
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
                        # For single relation, allow explicit parent FK override
                        parent_fk_col_name = rel_cfg.get('fk_column_name') or self._find_parent_fk_column_name(model_cls, child_model_cls, rel_name)
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
                            type_default_where=rel_cfg.get('type_default_where'),
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
                            try:
                                explicit_child_fk = rel_cfg.get('fk_column_name')
                            except Exception:
                                explicit_child_fk = None
                            fk_col_i = self._find_child_fk_column(model_cls, child_model_cls, explicit_child_fk)
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
                    # If unknown Strawberry fields are requested at root, select the full entity
                    # so their resolvers can access underlying model data.
                    if requested_other_root:
                        stmt = select(model_cls, *base_root_cols, *select_columns)
                    elif base_root_cols or select_columns:
                        # When projecting labeled columns only, we must add an explicit FROM
                        # so ORDER BY and correlated subselects can reference the parent table.
                        stmt = select(*base_root_cols, *select_columns).select_from(model_cls)
                    else:
                        stmt = select(model_cls)
                except Exception:
                    stmt = select(model_cls)
                else:
                # ----- Phase 2 filtering (argument-driven) -----
                    stmt = await RootSQLBuilders(self).apply_root_filters(
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
            # Build parameter list: for single roots only expose where + filter args
            if is_single:
                if args_str:
                    full_params = f"self, info, where=None, {args_str}"
                else:
                    full_params = "self, info, where=None"
            else:
                if args_str:
                    full_params = f"self, info, limit=None, offset=None, order_by=None, order_dir=None, order_multi=None, where=None, {args_str}"
                else:
                    full_params = "self, info, limit=None, offset=None, order_by=None, order_dir=None, order_multi=None, where=None"
            src = f"async def {func_name}({full_params}):\n" \
                  f"    _fa = {{}}\n"
            for a in declared_filters.keys():
                src += f"    _fa['{a}'] = {a} if '{a}' in locals() else None\n"
            # Apply defaults for declared roots if provided
            src += "    _rw = where if where is not None else (_defaults.get('where') if _defaults is not None else None)\n"
            if is_single:
                # No order/offset/limit args for single roots; enforce limit 1 internally
                src += "    _rows = await _base_impl(info, 1, None, None, None, None, _fa, _rw)\n"
            else:
                src += "    _ob = order_by if order_by is not None else (_defaults.get('order_by') if _defaults is not None else None)\n"
                src += "    _od = order_dir if order_dir is not None else (_defaults.get('order_dir') if _defaults is not None else None)\n"
                src += "    _om = order_multi if order_multi is not None else (_defaults.get('order_multi') if _defaults is not None else None)\n"
                src += "    _lim = limit\n"
                src += "    _rows = await _base_impl(info, _lim, offset, _ob, _od, _om, _fa, _rw)\n"
            src += "    return (_rows[0] if _rows else None) if _is_single else _rows\n"
            ns: Dict[str, Any] = {'_base_impl': _base_impl, '_defaults': relation_defaults, '_is_single': bool(is_single)}
            ns.update({'Optional': Optional, 'List': List, 'datetime': datetime})
            exec(src, ns)
            generated_fn = ns[func_name]
            if not getattr(generated_fn, '__module__', None):
                generated_fn.__module__ = __name__
            if is_single:
                ann: Dict[str, Any] = {
                    'info': StrawberryInfo,
                    'where': Annotated[Optional[str], strawberry.argument(description=_ARG_DESC_WHERE)],
                }
            else:
                ann: Dict[str, Any] = {
                    'info': StrawberryInfo,
                    'limit': Optional[int],
                    'offset': Optional[int],
                    'order_by': Annotated[Optional[str], strawberry.argument(description=_ARG_DESC_ORDER_BY)],
                    'order_dir': Annotated[Optional[Direction], strawberry.argument(description=_ARG_DESC_ORDER_DIR)],
                    'order_multi': Annotated[Optional[List[str]], strawberry.argument(description=_ARG_DESC_ORDER_MULTI)],
                    'where': Annotated[Optional[str], strawberry.argument(description=_ARG_DESC_WHERE)],
                }
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
                    'where': fdef.meta.get('scope'),
                    'arguments': fdef.meta.get('arguments')
                }
                is_single_root = bool(fdef.meta.get('single'))
                root_resolver = _make_root_resolver(target_b.model, target_st, target_b, fname, relation_defaults=rel_defaults, is_single=is_single_root)
                # Annotations
                if is_single_root:
                    query_annotations[fname] = Optional[self._st_types[target_name]]  # type: ignore
                else:
                    query_annotations[fname] = List[self._st_types[target_name]]  # type: ignore
                # Determine description: explicit relation meta comment on Query field or target model docstring/table comment
                root_desc = None
                try:
                    root_desc = (fdef.meta or {}).get('comment')
                except Exception:
                    root_desc = None
                if not root_desc:
                    try:
                        # Prefer target model docstring first
                        root_desc = getattr(target_b.model, '__doc__', None)
                        if not root_desc:
                            root_desc = getattr(getattr(target_b.model, '__table__', None), 'comment', None)
                    except Exception:
                        root_desc = None
                # Attach field on class with explicit resolver and description when available
                if root_desc:
                    setattr(query_cls, fname, strawberry.field(resolver=root_resolver, description=str(root_desc)))
                else:
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
                # Create plain runtime class; prefer domain class docstring when available
                _dom_doc = getattr(dom_cls, '__doc__', None)
                DomSt_local = type(
                    type_name,
                    (),
                    {'__doc__': (_dom_doc if _dom_doc else f'Domain container for {getattr(dom_cls, "__name__", type_name)}')}
                )
                # Keep the container in this registry module to minimize circular resolution risk
                DomSt_local.__module__ = __name__
                _domain_type_cache[dom_cls] = DomSt_local  # pre-cache to break cycles
                ann_local: Dict[str, Any] = {}
                # Precompute names of mutation fields using explicit Strawberry flags only.
                # Avoid signature/argument heuristics so the distinction is obvious and stable.
                excluded_mutation_field_names: set[str] = set()
                for _fname, _fval in list(vars(dom_cls).items()):
                    try:
                        if _fname.startswith('__') or isinstance(_fval, FieldDescriptor):
                            continue
                        # Only consider Strawberry-decorated fields
                        _looks_st = (
                            str(getattr(getattr(_fval, '__class__', object), '__module__', '') or '').startswith('strawberry')
                            or hasattr(_fval, 'base_resolver')
                        )
                        if not _looks_st:
                            continue
                        # Explicit mutation flags from Strawberry
                        _br = getattr(_fval, 'base_resolver', None)
                        _is_mut = False
                        try:
                            if bool(getattr(_fval, 'is_mutation', False)):
                                _is_mut = True
                        except Exception:
                            pass
                        if not _is_mut:
                            try:
                                if _br is not None and bool(getattr(_br, 'is_mutation', False)):
                                    _is_mut = True
                            except Exception:
                                pass
                        if not _is_mut:
                            try:
                                _fd = getattr(_fval, 'field_definition', None)
                                if _fd is not None and bool(getattr(_fd, 'is_mutation', False)):
                                    _is_mut = True
                            except Exception:
                                pass
                        # Stable custom flag we set via our patched strawberry.mutation
                        if not _is_mut:
                            try:
                                if bool(getattr(_fval, '__berry_is_mutation__', False)):
                                    _is_mut = True
                                elif _br is not None and bool(getattr(_br, '__berry_is_mutation__', False)):
                                    _is_mut = True
                                else:
                                    _fd = getattr(_fval, 'field_definition', None)
                                    if _fd is not None and bool(getattr(_fd, '__berry_is_mutation__', False)):
                                        _is_mut = True
                            except Exception:
                                pass
                        if _is_mut:
                            excluded_mutation_field_names.add(_fname)
                    except Exception:
                        continue
                # Helper: map a return annotation to runtime Strawberry types (unwrap Annotated, preserve Optional/List)
                def _map_ret_annotation(t: Any) -> Any:
                    try:
                        if t is None:
                            return None
                        from typing import Annotated as _Annotated  # type: ignore
                        origin = get_origin(t)
                        args = list(get_args(t) or [])
                        # Unwrap Annotated[T, ...]
                        if origin is _Annotated:
                            return _map_ret_annotation(args[0]) if args else None
                        # Optional[T] is Union[T, NoneType]
                        if origin is getattr(__import__('typing'), 'Union', None):
                            # Filter out NoneType
                            non_none = [a for a in args if a is not type(None)]  # noqa: E721
                            if len(non_none) == 1 and len(args) == 2:
                                inner = _map_ret_annotation(non_none[0])
                                return Optional[inner]  # type: ignore
                            # Generic Union: map each
                            mapped = tuple(_map_ret_annotation(a) for a in args)
                            try:
                                from typing import Union as _Union  # type: ignore
                                return _Union[mapped]  # type: ignore
                            except Exception:
                                return t
                        # List[T]
                        if origin in (list, List):
                            inner = _map_ret_annotation(args[0]) if args else Any
                            return List[inner]  # type: ignore
                        # Direct string name of a Berry type
                        if isinstance(t, str):
                            # Map to runtime Strawberry type when available
                            return self._st_types.get(t, t)
                        # If it's a Berry type class, map by its __name__
                        try:
                            if t in self.types.values():
                                nm = getattr(t, '__name__', None)
                                if nm and nm in self._st_types:
                                    return self._st_types[nm]
                        except Exception:
                            pass
                        return t
                    except Exception:
                        return t
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
                            'where': fdef.meta.get('scope'),
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
                        # Description for domain relations
                        d_desc = None
                        try:
                            d_desc = (fdef.meta or {}).get('comment')
                        except Exception:
                            d_desc = None
                        if not d_desc:
                            try:
                                # Prefer target model docstring first
                                d_desc = getattr(target_b.model, '__doc__', None)
                                if not d_desc:
                                    d_desc = getattr(getattr(target_b.model, '__table__', None), 'comment', None)
                            except Exception:
                                d_desc = None
                        if d_desc:
                            setattr(DomSt_local, fname, strawberry.field(resolver=resolver, description=str(d_desc)))
                        else:
                            setattr(DomSt_local, fname, strawberry.field(resolver=resolver))
                # Copy regular @strawberry.field resolvers from the domain class verbatim so Strawberry
                # registers them with their existing base_resolver. Avoid rebuilding wrappers here to
                # keep behavior identical to the user-declared methods (including Annotated/lazy types).
                _domain_static_field_resolvers: Dict[str, Any] = {}
                # Helper: detect strawberry subscription fields to avoid exposing them on Query containers
                def _looks_subscription(obj) -> bool:
                    try:
                        if bool(getattr(obj, 'is_subscription', False)):
                            return True
                    except Exception:
                        pass
                    try:
                        br = getattr(obj, 'base_resolver', None)
                        if br is not None and bool(getattr(br, 'is_subscription', False)):
                            return True
                    except Exception:
                        pass
                    try:
                        fd = getattr(obj, 'field_definition', None)
                        if fd is not None and bool(getattr(fd, 'is_subscription', False)):
                            return True
                    except Exception:
                        pass
                    try:
                        if callable(getattr(obj, 'subscribe', None)):
                            return True
                    except Exception:
                        pass
                    return False
                for uf, val in list(vars(dom_cls).items()):
                    # Skip dunders, Berry FieldDescriptors and nested domains (handled elsewhere)
                    if uf.startswith('__') or isinstance(val, FieldDescriptor):
                        continue
                    # Never expose fields that appear on the mutation domain container
                    if uf in excluded_mutation_field_names:
                        continue
                    # Strong guard: skip anything that carries our mutation marker directly or on its resolver
                    try:
                        if bool(getattr(val, '__berry_is_mutation__', False)):
                            continue
                        _rv = getattr(val, 'resolver', None)
                        if _rv is not None and bool(getattr(_rv, '__berry_is_mutation__', False)):
                            continue
                    except Exception:
                        pass
                    # Strong guard: if the mutation domain container exposes this name,
                    # do not copy it onto the Query domain type.
                    try:
                        # Only consider Strawberry-decorated fields (objects carrying a base_resolver)
                        looks_strawberry = (
                            str(getattr(getattr(val, '__class__', object), '__module__', '') or '').startswith('strawberry')
                            or hasattr(val, 'base_resolver')
                        )
                    except Exception:
                        looks_strawberry = False
                    if not looks_strawberry:
                        continue
                    # Do not expose subscription fields on Query domain container
                    try:
                        if _looks_subscription(val):
                            continue
                    except Exception:
                        pass
                    # Skip actual mutations using only explicit Strawberry flags.
                    def _looks_mutation(obj) -> bool:
                        try:
                            if bool(getattr(obj, 'is_mutation', False)):
                                return True
                        except Exception:
                            pass
                        try:
                            br = getattr(obj, 'base_resolver', None)
                            if br is not None and bool(getattr(br, 'is_mutation', False)):
                                return True
                        except Exception:
                            pass
                        try:
                            fd = getattr(obj, 'field_definition', None)
                            if fd is not None and bool(getattr(fd, 'is_mutation', False)):
                                return True
                        except Exception:
                            pass
                        try:
                            if bool(getattr(obj, '__berry_is_mutation__', False)):
                                return True
                        except Exception:
                            pass
                        return False
                    _is_mut = _looks_mutation(val)
                    if _is_mut:
                        # Do not expose mutation fields on Query domain container
                        continue
                    # Attach the original Strawberry field object; Strawberry will use its resolver.
                    try:
                        setattr(DomSt_local, uf, val)
                        # Track the underlying implementation for optional pre-population in the domain resolver
                        try:
                            br = getattr(val, 'base_resolver', None)
                            if br is not None:
                                _fn_for_cache = getattr(br, 'wrapped_func', None) or getattr(br, 'func', None)
                            else:
                                _fn_for_cache = getattr(val, 'func', None)
                            if callable(_fn_for_cache):
                                _domain_static_field_resolvers[uf] = _fn_for_cache
                        except Exception:
                            pass
                        # Resolve return annotation to a concrete runtime Strawberry type and set field.type_annotation
                        try:
                            ret_ann = None
                            try:
                                _fn_src = getattr(br, 'wrapped_func', None) if br is not None else None
                            except Exception:
                                _fn_src = None
                            if _fn_src is None:
                                _fn_src = getattr(val, 'func', None)
                            if callable(_fn_src):
                                ret_ann = getattr(_fn_src, '__annotations__', {}).get('return')
                            ann_eff = ret_ann
                            if ann_eff is not None:
                                from typing import get_origin as _go, get_args as _ga
                                # Unwrap Annotated[T, ...] one level and eval any forward refs in dom module
                                try:
                                    from typing import Annotated as _Ann, ForwardRef as _FRef  # type: ignore
                                except Exception:
                                    _Ann = None  # type: ignore
                                    _FRef = None  # type: ignore
                                try:
                                    if _Ann is not None and _go(ann_eff) is _Ann:
                                        _args = list(_ga(ann_eff) or [])
                                        if _args:
                                            ann_eff = _args[0]
                                except Exception:
                                    pass
                                try:
                                    import sys as _sys
                                    _mod = _sys.modules.get(getattr(dom_cls, '__module__', __name__))
                                    if isinstance(ann_eff, str):
                                        try:
                                            ann_eff = eval(ann_eff, vars(_mod) if _mod else {})
                                        except Exception:
                                            pass
                                    elif _FRef is not None and isinstance(ann_eff, _FRef):
                                        try:
                                            _name = getattr(ann_eff, '__forward_arg__', '')
                                            if _name:
                                                ann_eff = eval(_name, vars(_mod) if _mod else {})
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                # Map BerryType classes and collections to the generated Strawberry runtime classes
                                try:
                                    mapped_ann = _map_ret_annotation(ann_eff)
                                except Exception:
                                    mapped_ann = ann_eff
                                # If effective type is a Strawberry type class, pre-register by name
                                try:
                                    _cand = mapped_ann
                                    _orig = _go(_cand)
                                    if _orig is None and isinstance(_cand, type):
                                        if getattr(_cand, '__strawberry_definition__', None) or getattr(_cand, '__is_strawberry_type__', False):
                                            _nm = getattr(_cand, '__name__', None)
                                            if _nm and not self._st_types.get(_nm):
                                                self._st_types[_nm] = _cand  # type: ignore[assignment]
                                except Exception:
                                    pass
                                # Finally, set field.type_annotation to the mapped annotation
                                try:
                                    from strawberry.annotation import StrawberryAnnotation as _SA  # type: ignore
                                    _fld_obj = getattr(DomSt_local, uf, None)
                                    if _fld_obj is not None:
                                        _fld_obj.type_annotation = _SA.from_annotation(mapped_ann)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Do not pre-populate values here; let Strawberry call the resolver.
                    except Exception:
                        # best-effort: skip if we can't attach the field cleanly
                        continue
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
                                # Bypass dataclass __init__ to avoid required kw-only args on resolver-backed fields
                                try:
                                    inst = object.__new__(ChildSt)
                                except Exception:
                                    try:
                                        import inspect as _inspect
                                        sig = _inspect.signature(ChildSt)
                                        kwargs = {p.name: None for p in sig.parameters.values() if p.kind is p.KEYWORD_ONLY}
                                        inst = ChildSt(**kwargs)
                                    except Exception:
                                        inst = ChildSt()
                                setattr(inst, '__berry_registry__', getattr(self, '__berry_registry__', self))
                                return inst
                            return _resolver
                        setattr(DomSt_local, fname, strawberry.field(resolver=_make_nested_resolver(child_dom_st)))
                # Note: We now intentionally copy regular @strawberry.field attributes from domain classes as read-only
                # custom fields, in addition to Berry relations and nested domains.
                DomSt_local.__annotations__ = ann_local
                # Attach cached resolvers map for use by domain root resolver
                try:
                    setattr(DomSt_local, '__berry_domain_static_fields__', _domain_static_field_resolvers)
                except Exception:
                    pass
                # Decorate and cache
                self._st_types[type_name] = strawberry.type(DomSt_local)  # type: ignore
                # Ensure the GraphQL type description reflects the domain docstring when available
                try:
                    _dom_desc2 = getattr(dom_cls, '__doc__', None)
                except Exception:
                    _dom_desc2 = None
                if _dom_desc2:
                    try:
                        _sd = getattr(self._st_types[type_name], '__strawberry_definition__', None)
                        if _sd is not None:
                            setattr(_sd, 'description', str(_dom_desc2))
                    except Exception:
                        pass
                # Mirror the resolver cache onto the decorated class (some Strawberry versions replace the class)
                try:
                    setattr(self._st_types[type_name], '__berry_domain_static_fields__', getattr(DomSt_local, '__berry_domain_static_fields__', {}) or _domain_static_field_resolvers)
                except Exception:
                    pass
                # Also export into module globals to satisfy any LazyType lookups by name
                try:
                    globals()[type_name] = self._st_types[type_name]
                except Exception:
                    pass
                return self._st_types[type_name]

            # Only expose domains explicitly declared on Query
            for dom_name in list(self._domains_exposed_on_query):
                cfg = self._domains.get(dom_name) or {}
                dom_cls = cfg.get('class')
                if not dom_cls:
                    continue
                DomSt = _ensure_domain_type(dom_cls)
                # Expose on Query as a field that returns the domain container instance
                def _make_domain_resolver(DomSt_local):
                    async def _resolver(self, info: StrawberryInfo):  # noqa: D401
                        # Prefer bypassing dataclass __init__ to avoid required kw-only args on resolver-backed fields
                        try:
                            inst = object.__new__(DomSt_local)
                        except Exception:
                            try:
                                import inspect as _inspect
                                sig = _inspect.signature(DomSt_local)
                                kwargs = {p.name: None for p in sig.parameters.values() if p.kind is p.KEYWORD_ONLY}
                                inst = DomSt_local(**kwargs)
                            except Exception:
                                inst = DomSt_local()
                        setattr(inst, '__berry_registry__', self)
                        # Pre-populate regular strawberry.field values to avoid missing attributes when Strawberry
                        # falls back to default_resolver on dataclass fields. This keeps behavior deterministic even
                        # when the field is defined without args and returns a computed constant.
                        try:
                            import inspect as _inspect
                            sfmap = getattr(DomSt_local, '__berry_domain_static_fields__', {}) or {}
                            for _fname, _res in list(sfmap.items()):
                                try:
                                    if _inspect.iscoroutinefunction(_res):
                                        val = await _res(inst)
                                    else:
                                        val = _res(inst)
                                    setattr(inst, _fname, val)
                                except Exception:
                                    # Best-effort: leave unset if computation fails
                                    pass
                        except Exception:
                            pass
                        return inst
                    return _resolver
                query_annotations[dom_name] = DomSt  # type: ignore
                # Expose domain docstring as GraphQL field description when present
                _dom_desc = None
                try:
                    _dom_desc = getattr(dom_cls, '__doc__', None)
                except Exception:
                    _dom_desc = None
                if _dom_desc:
                    setattr(QueryPlain, dom_name, strawberry.field(resolver=_make_domain_resolver(DomSt), description=str(_dom_desc)))
                else:
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
                # Build fields from user-declared resolvers and map return annotations to runtime types
                anns_m: Dict[str, Any] = {}
                # Delegate domain-scoped mutation containers and their auto upserts to mutations module
                try:
                    from . import mutations as _mut
                    _mut.add_mutation_domains(self, MPlain, anns_m)
                except Exception:
                    pass
                # Copy user-defined Strawberry-decorated attributes (@strawberry.mutation, etc.). Ignore plain callables.
                for uf, val in vars(self._user_mutation_cls).items():
                    if uf.startswith('__'):
                        continue
                    looks_strawberry = False
                    try:
                        mod = getattr(getattr(val, "__class__", object), "__module__", "") or ""
                        looks_strawberry = (
                            mod.startswith("strawberry") or hasattr(val, "resolver") or hasattr(val, "base_resolver")
                        )
                    except Exception:
                        looks_strawberry = False
                    if not looks_strawberry:
                        continue
                    try:
                        # Extract underlying resolver function from the Strawberry field
                        fn = getattr(val, 'resolver', None)
                        if fn is None:
                            br = getattr(val, 'base_resolver', None)
                            fn = getattr(br, 'wrapped_func', None) or getattr(br, 'func', None)
                        if fn is None:
                            fn = getattr(val, 'func', None)
                        # Map return annotation from Berry types to runtime Strawberry types
                        ret_ann = None
                        try:
                            ret_ann = getattr(fn, '__annotations__', {}).get('return') if callable(fn) else None
                        except Exception:
                            ret_ann = None
                        mapped_type = None
                        try:
                            if get_origin(ret_ann) is getattr(__import__('typing'), 'Annotated', None):
                                args = list(get_args(ret_ann) or [])
                                if args:
                                    ret_ann = args[0]
                            if isinstance(ret_ann, str) and ret_ann in self.types:
                                mapped_type = self._st_types.get(ret_ann)
                            elif ret_ann in self.types.values():
                                nm = getattr(ret_ann, '__name__', None)
                                if nm:
                                    mapped_type = self._st_types.get(nm)
                        except Exception:
                            mapped_type = None
                        if callable(fn) and mapped_type is not None:
                            try:
                                anns = getattr(fn, '__annotations__', None)
                                if isinstance(anns, dict):
                                    anns['return'] = mapped_type
                            except Exception:
                                pass
                        # Attach as a regular strawberry.field using the resolver
                        setattr(MPlain, uf, strawberry.field(resolver=fn))
                    except Exception:
                        pass
                # Attach explicitly declared top-level merge mutations via mutations module
                try:
                    if '_mut' not in locals():
                        from . import mutations as _mut  # type: ignore
                    _mut.add_top_level_merges(self, MPlain, anns_m)
                except Exception:
                    pass
                setattr(MPlain, '__annotations__', anns_m)
                Mutation = strawberry.type(MPlain)  # type: ignore
        except Exception:
            Mutation = None
        # Build Subscription root from user-declared class and domain subscription fields
        try:
            # Start with a plain container regardless of user class; we'll decide later if it's empty
            SPlain = None
            anns_s: Dict[str, Any] = {}
            if self._user_subscription_cls is not None:
                SPlain = type('Subscription', (), {'__doc__': 'Auto-generated Berry root subscription.'})
                setattr(SPlain, '__module__', __name__)
                # Copy annotated attributes and strawberry subscription fields from user class
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
                            setattr(SPlain, uf, val)
                    except Exception:
                        pass
            # Also lift @strawberry.subscription fields declared on domain classes onto root
            # Detect if any domain provides subscription fields
            def _is_subscription_field(obj: Any) -> bool:
                try:
                    if getattr(obj, 'is_subscription', False):
                        return True
                except Exception:
                    pass
                try:
                    if callable(getattr(obj, 'subscribe', None)):
                        return True
                except Exception:
                    pass
                # Fallback: inspect return annotation of underlying resolver for AsyncGenerator
                try:
                    fn = getattr(obj, 'resolver', None) or getattr(getattr(obj, 'base_resolver', None), 'func', None) or getattr(obj, 'func', None)
                    from typing import get_origin as _go
                    origin = _go(getattr(fn, '__annotations__', {}).get('return')) if callable(fn) else None
                    if origin is not None and 'AsyncGenerator' in str(origin):
                        return True
                except Exception:
                    pass
                return False
            # Ensure we have a container if we find any domain subscriptions
            dom_subs_found = False
            for _dom_name, _cfg in list(self._domains.items()):
                dom_cls = _cfg.get('class')
                if not dom_cls:
                    continue
                for uf, val in vars(dom_cls).items():
                    if uf.startswith('__'):
                        continue
                    try:
                        looks_strawberry = (
                            str(getattr(getattr(val, "__class__", object), "module", "") or "").startswith("strawberry")
                            or hasattr(val, "resolver")
                            or hasattr(val, "base_resolver")
                        )
                        if not looks_strawberry:
                            continue
                        if not _is_subscription_field(val):
                            continue
                        # We have a subscription field on a domain; create Subscription container if missing
                        if SPlain is None:
                            SPlain = type('Subscription', (), {'__doc__': 'Auto-generated Berry root subscription.'})
                            setattr(SPlain, '__module__', __name__)
                        dom_subs_found = True
                        # Extract underlying resolver function
                        fn = getattr(val, 'resolver', None)
                        if fn is None:
                            br = getattr(val, 'base_resolver', None)
                            fn = getattr(br, 'wrapped_func', None) or getattr(br, 'func', None)
                        if fn is None:
                            fn = getattr(val, 'func', None)
                        # Map return annotation to generated Strawberry runtime types
                        try:
                            ret_ann = getattr(fn, '__annotations__', {}).get('return') if callable(fn) else None
                        except Exception:
                            ret_ann = None
                        mapped_type = None
                        try:
                            from typing import get_origin as _go, get_args as _ga, Annotated as _Ann  # type: ignore
                        except Exception:
                            _go = None  # type: ignore
                            _ga = None  # type: ignore
                            _Ann = None  # type: ignore
                        try:
                            if _Ann is not None and _go and _go(ret_ann) is _Ann:
                                args = list(_ga(ret_ann) or [])
                                if args:
                                    ret_ann = args[0]
                            # Map BerryType names/classes to runtime types
                            if isinstance(ret_ann, str) and ret_ann in self.types:
                                mapped_type = self._st_types.get(ret_ann)
                            elif ret_ann in self.types.values():
                                nm = getattr(ret_ann, '__name__', None)
                                if nm:
                                    mapped_type = self._st_types.get(nm)
                        except Exception:
                            mapped_type = None
                        if callable(fn) and mapped_type is not None:
                            try:
                                anns = getattr(fn, '__annotations__', None)
                                if isinstance(anns, dict):
                                    anns['return'] = mapped_type
                            except Exception:
                                pass
                        # Attach as a strawberry.subscription using the resolver directly
                        # Prefix with domain name to avoid global name collisions and emulate namespacing.
                        # Naming:
                        # - If auto_camel_case is enabled (via config.name_converter or flag), use
                        #   {domain}{SubMethodName} where SubMethodName is CamelCased from the original method name.
                        # - Otherwise, use snake case joined with underscore: {domain}_{sub_method_name}
                        try:
                            _method_name = uf
                            _dom_prefix = _dom_name
                            use_camel = False
                            try:
                                if getattr(self, '_name_converter', None) is not None:
                                    use_camel = bool(getattr(self._name_converter, 'auto_camel_case', False))
                                elif hasattr(self, '_auto_camel_case'):
                                    use_camel = bool(getattr(self, '_auto_camel_case'))
                            except Exception:
                                use_camel = False
                            if use_camel:
                                # Prefer converter if available to match Strawberry behavior
                                try:
                                    if getattr(self, '_name_converter', None) is not None and hasattr(self._name_converter, 'apply_naming_config'):
                                        _camel = self._name_converter.apply_naming_config(_method_name)  # type: ignore[attr-defined]
                                    else:
                                        # Fallback: local snake_to_camel (lowerCamel)
                                        parts = [p for p in str(_method_name).split('_') if p]
                                        _camel = parts[0].lower() + ''.join(p.capitalize() for p in parts[1:]) if parts else str(_method_name)
                                except Exception:
                                    _camel = str(_method_name)
                                # Convert to PascalCase for the suffix when concatenated to domain
                                _suffix = (_camel[:1].upper() + _camel[1:]) if _camel else str(_method_name)
                                new_field_name = f"{_dom_prefix}{_suffix}"
                            else:
                                new_field_name = f"{_dom_prefix}_{_method_name}"
                        except Exception:
                            new_field_name = uf
                        try:
                            setattr(SPlain, new_field_name, strawberry.subscription(fn))  # type: ignore[arg-type]
                        except Exception:
                            # Fallback: expose as a regular field if subscription wrapper fails
                            setattr(SPlain, new_field_name, strawberry.field(resolver=fn))
                    except Exception:
                        continue
            # Finalize Subscription only if we actually have a class with any fields
            if SPlain is not None and (dom_subs_found or self._user_subscription_cls is not None):
                try:
                    setattr(SPlain, '__annotations__', getattr(SPlain, '__annotations__', {}) or anns_s)
                except Exception:
                    pass
                Subscription = strawberry.type(SPlain)  # type: ignore
            else:
                Subscription = None
        except Exception:
            Subscription = None
        # Safety: purge any empty/stale mutation domain types without fields before schema build
        try:
            for _tn, _tc in list(getattr(self, '_st_types', {}).items()):
                try:
                    if not isinstance(_tn, str) or not _tn.endswith('MutType'):
                        continue
                    has_fields = False
                    for _an in dir(_tc):
                        try:
                            _v = getattr(_tc, _an)
                        except Exception:
                            continue
                        _mod = getattr(getattr(_v, '__class__', object), '__module__', '') or ''
                        if _mod.startswith('strawberry') or hasattr(_v, 'resolver') or hasattr(_v, 'base_resolver'):
                            has_fields = True
                            break
                    if not has_fields:
                        try:
                            del self._st_types[_tn]
                        except Exception:
                            pass
                except Exception:
                    continue
        except Exception:
            pass
        # Build the Schema using StrawberryConfig exclusively (modern API).
        # If caller provided a config, honor it; otherwise default to snake_case globally.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"LazyType is deprecated.*")
            if strawberry_config is None:
                try:
                    from strawberry.schema.name_converter import NameConverter as _NC  # type: ignore
                    class _IdentityNameConverter(_NC):  # type: ignore[misc]
                        def __init__(self):
                            super().__init__(auto_camel_case=False)
                            self.auto_camel_case = False
                        def apply_naming_config(self, name: str) -> str:  # type: ignore[override]
                            try:
                                if isinstance(name, str) and name.startswith('_'):
                                    return name
                            except Exception:
                                pass
                            return name
                    _nc = _IdentityNameConverter()
                    strawberry_config = StrawberryConfig(name_converter=_nc, auto_camel_case=False)  # type: ignore[arg-type]
                except Exception:
                    strawberry_config = StrawberryConfig(auto_camel_case=False)  # type: ignore[arg-type]
            # Allow rebuilding schemas with the same type names without tripping duplicate validation.
            # This is safe here because we fully control type identity per to_strawberry call.
            try:
                setattr(strawberry_config, "_unsafe_disable_same_type_validation", True)  # type: ignore[attr-defined]
            except Exception:
                pass
            # Finally, construct schema with config
            # Build base schema
            if Mutation is not None and Subscription is not None:
                _inner_schema = strawberry.Schema(Query, mutation=Mutation, subscription=Subscription, config=strawberry_config)  # type: ignore[arg-type]
            elif Mutation is not None:
                _inner_schema = strawberry.Schema(Query, mutation=Mutation, config=strawberry_config)  # type: ignore[arg-type]
            else:
                _inner_schema = strawberry.Schema(Query, config=strawberry_config)  # type: ignore[arg-type]
            return _inner_schema

# --- Module-level helpers (public) ---------------------------------------------
def hooks(*, pre: Any | None = None, post: Any | None = None):
    """Create a hook descriptor for in-class registration.

    Usage:
        from app.graphql.berryql import hooks
        class MyType(BerryType):
            hooks = hooks(pre=pre_fn, post=post_fn)
    """
    return HooksDescriptor(pre=pre, post=post)
