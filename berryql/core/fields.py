from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

@dataclass
class FieldDef:
    """Internal, normalized field description collected by the registry.

    Attributes:
        name: The attribute name on the declaring type (e.g. "posts").
        kind: One of "scalar", "relation", "aggregate", "custom", "custom_object".
        meta: Arbitrary metadata captured from the descriptor factory. Keys vary by
            kind and are interpreted by the registry/adapters layer (e.g. target,
            single, order_by, arguments, op, returns, etc.).
    """

    name: str
    kind: str
    meta: Dict[str, Any]

class FieldDescriptor:
    """Descriptor placed on Berry types to declare fields.

    Users normally use helper factories like :func:`field`, :func:`relation`,
    :func:`aggregate`, :func:`custom`, or :func:`custom_object` which return a
    ``FieldDescriptor`` instance. The registry inspects the descriptor and
    converts it to a :class:`FieldDef` with normalized metadata.
    """

    def __init__(self, *, kind: str, **meta):
        self.kind = kind
        self.meta = dict(meta)
        self.name: str | None = None

    def __set_name__(self, owner, name):  # pragma: no cover - simple
        self.name = name

    def build(self, parent_name: str) -> FieldDef:
        """Build a :class:`FieldDef` consumed by the registry.

        Args:
            parent_name: Name of the parent type (unused here; provided for
                symmetry with other builders).

        Returns:
            FieldDef: Normalized representation for this field.
        """
        return FieldDef(name=self.name or '', kind=self.kind, meta=self.meta)

def field(column: Optional[str] = None, /, **meta) -> FieldDescriptor:
    """Declare a scalar field on a Berry type.

    Place this as a class attribute inside a ``@berry_schema.type`` class to
    expose a simple column/attribute as a GraphQL field or to reserve a slot for
    a computed scalar resolved by the adapters layer.

        Common metadata keys (interpreted by adapters/registry):
        - column: Override source column/attribute name on the model. You can pass
            this as a positional argument too, e.g. ``id = field('user_id')``.
    - description: Optional GraphQL field description.
    - nullable: Hint about nullability for schema generation.

    Examples:
        class PostQL(BerryType):
            id = field()
            title = field(description="Post title")
            # Map GraphQL 'id' to DB column 'user_id'
            id_alias = field('user_id', description="Public id")
            # Or expose DB column with same name directly
            user_id = field()

    Returns:
        FieldDescriptor: A descriptor captured by the registry.
    """
    if column is not None:
        meta = dict(meta)
        meta['column'] = column
    return FieldDescriptor(kind='scalar', **meta)

def relation(target: Any = None, *, single: bool | None = None, mutation: bool = False, **meta) -> FieldDescriptor:
    """Declare a relation to another Berry type.

    Use on both root Query/Domain classes and nested Berry types to model
    one-to-many or one-to-one relations.

        Args:
        target: Target Berry type, either as the class itself or its name
            (string). When omitted, the registry may infer it from conventions.
        single: When True, the relation returns at most one object (one-to-one).
            When False/None, the relation returns a list.
                **meta: Extra options interpreted by adapters/registry. Common keys:
            - order_by: Column name to order by (e.g. "id").
            - order_dir: "asc" or "desc".
                        - scope: Callable(model_cls, info) -> SQLAlchemy expression or dict
                            to pre-filter results (previously named "where").
            - arguments: Mapping of GraphQL argument names to either callables
              ``lambda M, info, v: <SA expression>`` or dict specifications
              {"column": <str>, "op": <str>, "transform": <callable>} to
              drive filtering.
                        - post_process: Python callable applied to the hydrated relation
                            value(s) right before returning from the resolver. It is never
                            translated to SQL. Signature suggestions:
                                single=True:  fn(child_instance, info) -> Any
                                single=False: fn(list_of_child_instances, info) -> Any
                            The callable may be sync or async.
                        - returns: Optional Python type for the GraphQL field when using
                            post_process to return a non-target type (e.g., str). For list
                            relations, provide the element type (e.g., str); the schema will
                            expose List[returns].

    Examples:
        class PostQL(BerryType):
            author = relation('UserQL', single=True)
            comments = relation('PostCommentQL', order_by='id')

        @berry_schema.query()
        class Query:
            users = relation('UserQL', order_by='id', order_dir='asc', arguments={
                'name_ilike': lambda M, info, v: M.name.ilike(f"%{v}%"),
            })

    Returns:
        FieldDescriptor: A descriptor captured by the registry.
    """
    m = dict(meta)
    if target is not None:
        m['target'] = target.__name__ if hasattr(target, '__name__') and not isinstance(target, str) else target
    if single is not None:
        m['single'] = single
    # Flag to enable generating an upsert mutation for this relation under Mutation domains
    m['mutation'] = bool(mutation)
    # Backward-incompatible rename: parameter 'where' became 'scope'.
    # If users pass scope=..., keep it under 'scope' key; no implicit aliasing from 'where'.
    return FieldDescriptor(kind='relation', **m)

def aggregate(source: str, **meta) -> FieldDescriptor:
    """Declare an aggregate derived from a relation.

    Aggregates compute a scalar over a related collection (e.g. count, min,
    max). They are derived from a relation field present on the same type.

    Args:
        source: Name of the relation attribute to aggregate (e.g. "posts").
        **meta: Extra options such as ``op`` (e.g. "count", "min", "max",
            adapter-specific parameters) and ``description``.

    Example:
        class UserQL(BerryType):
            posts = relation('PostQL')
            post_agg = aggregate('posts', op='count')

    Returns:
        FieldDescriptor: A descriptor captured by the registry.
    """
    return FieldDescriptor(kind='aggregate', source=source, **meta)

def count(source: str) -> FieldDescriptor:
    """Convenience wrapper for ``aggregate(..., op='count')``.

    Example:
        class UserQL(BerryType):
            posts = relation('PostQL')
            post_agg = count('posts')
    """
    return aggregate(source, op='count')

def custom(builder: Callable[..., Any], *, returns: Any | None = None) -> FieldDescriptor:
    """Declare a custom computed scalar backed by a builder function.

    The builder should return a SQLAlchemy selectable/expression that yields a
    single scalar value per parent row. This enables pushdown execution (no
    N+1) when supported by adapters.

    Args:
        builder: Callable invoked as ``builder(model_cls)`` (and may receive
            additional adapter-provided context). Must return a selectable or
            expression producing the scalar.
        returns: Optional Python type for GraphQL schema generation (e.g. ``int``).
            When omitted, the adapter may try to infer it.

    Example:
        class PostQL(BerryType):
            def _comment_text_len_builder(model_cls):
                from sqlalchemy import select, func
                from tests.models import PostComment
                return (
                    select(func.coalesce(func.sum(func.length(PostComment.content)), 0))
                    .where(PostComment.post_id == model_cls.id)
                )
            comment_text_len = custom(_comment_text_len_builder, returns=int)

    Returns:
        FieldDescriptor: A descriptor captured by the registry.
    """
    return FieldDescriptor(kind='custom', builder=builder, returns=returns)

def custom_object(builder: Callable[..., Any], *, returns: Any) -> FieldDescriptor:
    """Declare a custom computed object with a fixed shape.

    Similar to :func:`custom` but returns multiple named columns bundled into a
    nested GraphQL object. The ``returns`` mapping defines the object's shape
    and field types.

    Args:
        builder: Callable invoked as ``builder(model_cls)`` returning a
            SQLAlchemy selectable with labeled columns matching the keys of
            ``returns``.
        returns: A mapping of field name to Python type, e.g.
            ``{'min_created_at': datetime, 'comments_count': int}``.

    Example:
        class PostQL(BerryType):
            post_comments_agg_obj = custom_object(
                lambda model_cls: (
                    select(
                        func.min(PostComment.created_at).label('min_created_at'),
                        func.count(PostComment.id).label('comments_count')
                    ).where(PostComment.post_id == model_cls.id)
                ),
                returns={'min_created_at': datetime, 'comments_count': int}
            )

    Returns:
        FieldDescriptor: A descriptor captured by the registry.
    """
    return FieldDescriptor(kind='custom_object', builder=builder, returns=returns)

# --- Domains (namespacing) ---

@dataclass
class DomainDef:
    name: str
    meta: Dict[str, Any]

class DomainDescriptor:
    """Descriptor placed on Query classes to expose a registered domain.

    Using :func:`domain`, you can mount a :class:`BerryDomain` subclass under a
    namespaced field on Query or on another Domain to create nested groups.

    Example:
        @berry_schema.domain(name='userDomain')
        class UserDomain(BerryDomain):
            users = relation('UserQL')

        @berry_schema.query()
        class Query:
            userDomain = domain(UserDomain)

    The registry collects these and generates nested Strawberry types and
    resolvers accordingly.
    """
    def __init__(self, domain_cls: Type[Any], *, name: Optional[str] = None, **meta: Any):
        self.domain_cls = domain_cls
        self.name = name
        self.meta = dict(meta)
        self.attr_name: str | None = None

    def __set_name__(self, owner, name):  # pragma: no cover
        self.attr_name = name

def domain(domain_cls: Type[Any], *, name: Optional[str] = None, **meta: Any) -> DomainDescriptor:
    """Declare a domain field on Query or another Domain.

    Args:
        domain_cls: A ``BerryDomain`` subclass previously registered with the
            schema via ``@berry_schema.domain``.
        name: Optional explicit GraphQL field name; defaults to the attribute
            name where it is assigned.
        **meta: Reserved for future/extensible options.

    Returns:
        DomainDescriptor: A descriptor captured by the registry.

    Example:
        @berry_schema.domain(name='blogDomain')
        class BlogDomain(BerryDomain):
            posts = relation('PostQL', order_by='id')

        @berry_schema.query()
        class Query:
            blogDomain = domain(BlogDomain)
    """
    return DomainDescriptor(domain_cls, name=name, **meta)
