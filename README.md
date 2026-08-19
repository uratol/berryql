BerryQL
========

A tiny, declarative GraphQL mapper for Strawberry + SQLAlchemy that optimizes queries automatically.

BerryQL lets you define GraphQL types on top of SQLAlchemy models with a minimal DSL. At runtime it:

- Projects only the columns you ask for (column-level projection pushdown)
- Pushes down relations into a single SQL per root field when possible
- Supports relation filters, ordering, and pagination without N+1
- Adds simple aggregates (e.g., count) and custom SQL-backed fields/objects

It’s designed for async SQLAlchemy 2.x and Strawberry GraphQL.


Hello world example
-------------------

Here is a minimal end‑to‑end sketch using BerryQL with three types, relations, a query, and a merge mutation:

```python
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncSession
import strawberry
from strawberry.types import Info
from berryql import BerrySchema, BerryType, field, relation, mutation

berry_schema = BerrySchema()

# SQLAlchemy models (simplified)
class User:
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    posts: Mapped[list["Post"]] = relationship(back_populates="author")
    # extra relation for comments authored by the user (optional)
    comments: Mapped[list["PostComment"]] = relationship(back_populates="author")

class Post:
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    author_id: Mapped[int]
    author: Mapped[User] = relationship(back_populates="posts")
    comments: Mapped[list["PostComment"]] = relationship(back_populates="post")


class PostComment:
    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str]
    post_id: Mapped[int]
    author_id: Mapped[int]
    post: Mapped[Post] = relationship(back_populates="comments")
    author: Mapped[User] = relationship(back_populates="comments")


@berry_schema.type(model=Post)
class PostQL(BerryType):
    id = field()
    title = field()
    author_id = field()  # autoCamelCase → authorId
    # many‑to‑one relation: post → author
    author = relation("UserQL", single=True)
    # one‑to‑many relation: post → comments
    comments = relation("PostCommentQL")


@berry_schema.type(model=PostComment)
class PostCommentQL(BerryType):
    id = field()
    content = field()
    post_id = field(name="postId")
    author_id = field(name="authorId")
    post = relation("PostQL", single=True)
    author = relation("UserQL", single=True)


@berry_schema.type(model=User)
class UserQL(BerryType):
    id = field()
    name = field()
    # one‑to‑many relation: user → posts
    posts = relation("PostQL")


@berry_schema.query()
class Query:
    # root collection
    users = relation("UserQL")


@berry_schema.mutation()
class Mutation:
    # generated merge mutation: upsert Post rows from payload
    merge_posts = mutation("PostQL")


schema = berry_schema.to_strawberry()
```

GraphQL usage examples:

```graphql
{
    users {
        id
        name
        posts {
            id
            title
            comments {
                id
                content
            }
        }
    }
}
```

```graphql
mutation {
    mergePosts(
        payload: [{
            title: "Hello",
            authorId: 1,
            comments: [{ content: "Nice post" }]
        }]
    ) {
        id
        title
        comments { id content }
    }
}
```


5‑minute try-out
-----------------

If you just want to see it working quickly, you don’t need to design a schema from scratch – this repo already contains a full demo schema, models, and a FastAPI app.

**1. Create a virtualenv and install deps (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. Run tests (uses in‑memory SQLite)**

```powershell
pytest -q
```

This spins up the demo models (`tests/models.py`), Berry schema (`tests/schema.py`), and exercises queries, relations, domains, mutations, and subscriptions.

**3. Run the demo GraphQL API (FastAPI + Strawberry)**

```powershell
python -m uvicorn examples.main:app --reload --host 127.0.0.1 --port 8000
```

Then open GraphiQL at: http://127.0.0.1:8000/graphql

Try a simple query:

```graphql
{
    users {
        id
        name
        postAggObj { count }
    }
}
```

Or a mutation using Berry’s merge API via a domain:

```graphql
mutation {
    blogDomain {
        merge_posts(payload: [{ title: "Hello", content: "Body", authorId: 1 }]) {
            id
            title
            authorId
        }
    }
}
```

Environment variables (optional):

- `BERRYQL_TEST_DATABASE_URL`: async SQLAlchemy URL (e.g., `postgresql+asyncpg://…` or `mssql+aioodbc:///?odbc_connect=…`)
- `SQL_ECHO`: set `1` to log SQL (default `1`)

See `README_RUN_FASTAPI.md` for more.

**4. Minimal “how would I use this in my app?” sketch**

At a high level you will:

1. Define SQLAlchemy models (or reuse existing ones).
2. Map them to Berry types with `@berry_schema.type` and `field()/relation()`.
3. Define a `@berry_schema.query()` class for roots and optionally `@berry_schema.mutation()` / `@berry_schema.domain()` / `@berry_schema.subscription()` classes for mutations, domains, and subscriptions.
4. Call `berry_schema.to_strawberry()` and plug the resulting schema into Strawberry/FastAPI.

The rest of this README goes into the details of fields, relations, filters, JSON where, custom scalars/objects, domains, merge mutations, and subscriptions.


Dynamic field permissions
-------------------------

BerryQL can keep one stable GraphQL schema while resolving field and operation
capabilities from an external authorization system for each execution. Providers
may be synchronous or asynchronous and receive ``(berry_type, info)``:

```python
from berryql import (
    BerrySchema,
    FieldPermissions,
    FieldSet,
    OperationPermissions,
)


async def resolve_fields(berry_type, info):
    grants = await info.context["acl"].fields_for(
        actor=info.context["current_user"],
        resource=berry_type.model.__tablename__,
    )
    return FieldPermissions(
        select=FieldSet.only(grants.select_fields),
        filter=FieldSet.only(grants.filter_fields),
        order=FieldSet.only(grants.order_fields),
        create=FieldSet.only(grants.create_fields),
        update=FieldSet.only(grants.update_fields),
        operations=OperationPermissions(
            create=grants.can_create,
            update=grants.can_update,
            delete=grants.can_delete,
            replace=grants.can_replace,
        ),
    )


berry_schema = BerrySchema(field_permissions=resolve_fields)
```

``FieldSet.only(...)`` is an allow-list. ``FieldSet.all_except(...)`` is a
deny-list; ``FieldSet.all()`` and ``FieldSet.none()`` are also available. A
type-specific provider can be added with
``@berry_schema.type(model=Post, field_permissions=resolve_post_fields)``.
Schema-level and type-level policies are intersected independently for every
capability and operation. Results are resolved with concurrent single-flight and
cached once per Berry type in the current GraphQL execution. Reusing a context
for another execution does not reuse the previous policy.

``select`` controls returned values. Denied scalar, single relation, custom,
aggregate, and ordinary nullable Strawberry fields resolve to ``null``; denied
list relations and ordinary list fields resolve to ``[]``. Denied fields are
pruned from SQL projections and relation/custom/aggregate pushdown.
``filter`` controls caller-provided ``where`` and declared filter arguments;
``order`` controls ``order_by`` and ``order_multi``. Both are always intersected
with ``select`` so a hidden field cannot become a side channel.

``create`` and ``update`` sanitize payload fields recursively before merge
pre-hooks run. Primary keys remain usable as selectors without becoming writable
values on existing rows. ``operations`` separately controls create, update,
delete, and replace: ``_Insert: true`` requires create, ``_Insert: false``
requires update, ``_Delete`` requires delete, and ``_Replace`` requires replace
plus update access to its parent relation. Replace deletion also requires delete
access on the child type. On updates, BerryQL logs a warning only when a denied
scalar differs from an already-loaded old value; it never issues a query or lazy
load solely to produce that warning.

``FieldPermissions`` is capability-based from its first public release: field
access is declared independently through ``select``, ``filter``, ``order``,
``create``, and ``update``, while mutation-wide grants are declared through
``OperationPermissions``.

Field policy and row scope solve different problems. `FieldPermissions` is a
request/type policy and is cached once per Berry type for one GraphQL execution;
it must not inspect a particular result row. A type-level `scope(...)` is a row
policy and is ANDed with relation/domain scopes for root and nested reads and for
create, update, delete, replace, and re-parent checks. Applications that need
per-object decisions should express them as row scopes (or add an explicit
object-policy layer), not as a field-permission provider.

Policy, scope, predicate, ordering, and adapter failures are fail-closed. BerryQL
never converts one of these failures into an unfiltered query or broader mutation.
Authorization audit records contain only type, capability, operation, field, and
reason metadata; payload values and secrets are not recorded.


What queries look like (and what SQL runs)

- Only selected columns are fetched for each table.
- When selecting users with posts, BerryQL will execute one SQL for users and one for posts (root fields), aggregating nested rows without joining unrelated tables.
- For simple selections like `users { id }`, the SQL only selects the id column.

See tests for concrete assertions:

- `tests/test_sql_projection.py` ensures only requested columns are present and unrelated tables aren’t touched.
- `tests/test_relations_pagination_aggregate.py` ensures “one SQL per root field” when pushdown is supported.


Core concepts
-------------

- BerrySchema: registry for types and root query.
- BerryType: base for GraphQL types. Use Berry’s field descriptors on subclasses.
- field(): scalar column mapping.
- relation(target, single=False, …): relation to another Berry type. Supports:
    - arguments: map GraphQL args to SQL filters (column+op or builder callable)
    - scope: trusted default JSON-style row scope for the relation (dict or JSON string) or callable(model_cls, info)
    - order_by/order_dir/order_multi, limit/offset
    - single=True for to-one
- count(source): count aggregate of a relation.
- aggregate(source, ops=[…]): additional prebuilt aggregates (tests use ‘last’ to get last related id).
- custom(builder, returns=…): computed scalar; builder returns an SQLAlchemy Select or expression (preferred), or a value.
- custom_object(builder, returns={…}): computed object; returns-spec defines fields and their types.


Filtering arguments (relation.arguments)
---------------------------------------

Attach GraphQL args to a relation and map them to SQL with a simple spec:

- Column-based spec:
    - { 'column': 'created_at', 'op': 'between' }
- Expand to multiple ops automatically:
    - { 'column': 'created_at', 'ops': ['gt', 'lt'] }
- Builder (full control):
    - lambda Model, info, value: Model.name.ilike(f"%{value}%")
- Optional transform to coerce/parse the input.

Builder filters can declare every field they read with `depends_on` (or the
equivalent `fields` alias). BerryQL authorizes those dependencies before invoking
the transform or builder. Legacy builders without dependency metadata remain
valid and are treated as declaring no implicit field access; new builders should
always declare dependencies.

```python
arguments={
    "owned_by": FilterSpec(
        builder=build_owner_filter,
        arg_type=int,
        depends_on=("owner_id", "tenant_id"),
    )
}
```

Caller query cost can be bounded without changing the historical permissive
defaults:

```python
from berryql import BerrySchema, FilterLimits

berry_schema = BerrySchema(
    filter_limits=FilterLimits(
        max_clauses=20,
        max_in_items=100,
        max_json_length=8_192,
        max_depth=6,
    )
)
```

`BerrySchema(operators={...})` and `berry_schema.register_operator(...)` create
schema-local operators. The module-level `register_operator` remains a
compatibility default for schemas created afterwards.

At runtime BerryQL validates columns, operators, and types and applies them in SQL. When relation pushdown is skipped (e.g., because of a ‘where’ argument), filters are still applied safely in resolvers.

Supported operators include: eq, ne, lt, lte, gt, gte, like, ilike, in, between, contains, starts_with, ends_with. You can register more.


Ordering and pagination
-----------------------

- order_by: a single column
- order_dir: asc|desc
- order_multi: ["created_at:desc", "id:asc"]
- limit/offset: integers
- after: an opaque keyset cursor created with `encode_cursor(...)`

`order_multi` also accepts an optional null placement suffix, for example
`"created_at:desc:nulls_last"`. Paginated ordering always receives a primary-key
tie-breaker. Dotted single-relation fallback ordering batches each relation hop,
so its SQL query count is bounded by ordering depth rather than result-row count.

Invalid order_by values raise a GraphQL error with the allowed fields.

Applications can opt into server-side list bounds when creating the schema:

```python
from berryql import BerrySchema, PaginationConfig

berry_schema = BerrySchema(
    pagination=PaginationConfig(default_limit=100, max_limit=200),
)
```

When configured, generated list resolvers apply `default_limit` if the caller omits `limit`, cap oversized explicit limits to `max_limit`, and continue to honor `offset` for next-page reads. Leaving pagination unconfigured preserves BerryQL's historical unbounded behavior.

Keyset pagination is additive; existing `limit`/`offset` calls are unchanged.
The cursor values must match the effective ordering (including the automatic PK
tie-breaker when the requested order does not already include it):

```python
from berryql import encode_cursor

variables = {"after": encode_cursor(last_created_at, last_id)}
```


JSON where
----------

- Relation resolvers accept a where argument that’s either a dict or a JSON string with operators, for example:
    - { "created_at": { "between": ["2000-01-01T00:00:00", "2100-01-01T00:00:00"] } }
- Type-coercion is handled using the target column’s type.


Scalar aggregates
----------

- Count is pushed down as a correlated subquery and cached per parent row.


Custom fields and objects/aggregation
-------------------------

- Prefer builders that accept the model class and return a Select/aggregates expression; these can be pushed into the root SQL.
- For custom_object, specify returns as a dict, e.g., { 'min_created_at': datetime, 'comments_count': int }.
- On Postgres/SQLite, JSON composition uses native json functions; on MSSQL it uses FOR JSON PATH.

Subscriptions
-------------

BerryQL can also participate in Strawberry subscriptions via `@berry_schema.subscription()` classes:

- Define a subscription container with `@berry_schema.subscription()`.
- Inside, declare `@strawberry.subscription` methods that yield values (e.g. integers or BerryQL objects) using async generators.
- The test schema includes a simple `tick` subscription and a `new_post_event` subscription under a domain to exercise this path.

Root query
----------

Define explicit roots with @berry_schema.query(). Each root field is a relation() to a Berry type. The resulting Strawberry schema exposes these roots.

Example patterns used in tests:

- Root collections: users, posts
- Single by ID: userById(single=True)
- Root-level arguments for filtering/ordering/pagination
- Context-aware gating with where=lambda model_cls, info: … (see `tests/schema.py`)

Execution and context
---------------------

Execute queries with the Strawberry schema built from Berry:

- schema = berry_schema.to_strawberry()
- await schema.execute(query, context_value={ 'db_session': async_session, … })

Context keys recognized by the test schema:

- db_session (required): AsyncSession used for all SQL
- enforce_user_gate / user_id / current_user: example gating knobs in tests


Dialect support and adapters
----------------------------

BerryQL detects the SQLAlchemy dialect from the provided session and adapts JSON handling:

- SQLite: json_object/json_group_array
- Postgres: json_build_object/json_agg
- MSSQL: FOR JSON PATH (single and list relations, nested arrays)

Relation pushdown works on all three. When it’s not safe to push down (e.g., custom where/filters that require resolver logic), BerryQL falls back to per-relation queries and still avoids N+1 where practical.


Type naming and camelCase
-------------------------

BerryQL respects Strawberry name conversion. If you use auto_camel_case/name_converter in Strawberry config, selection extraction recognizes camelCase field names and maps them to your Python field names.


Testing and development
-----------------------

- Run tests with the bundled suite:
    - pytest -q
- Provide BERRYQL_TEST_DATABASE_URL to run against Postgres/MSSQL; else tests use in-memory SQLite (async) and echo SQL.


Mutations
---------

BerryQL supports two styles of mutations, both inside a class registered with `@berry_schema.mutation()`:

1) **BerryQL merge mutations** (generated resolvers)

- Use the `mutation("TypeName", ...)` helper on a domain or the root mutation class to create upsert-style mutations backed by the ORM model of that Berry type.
- Variants in the test schema include:
    - `merge_posts`, `merge_users`: bulk upserts from a `payload` list.
    - `merge_post`: single-payload variant (one object instead of a list).
    - Scoped mutations: pass `scope` (JSON or callable) to enforce filters server-side, e.g. `scope='{"author_id": {"eq": 1}}'`.
- Merge callbacks can be attached on the Berry type:
    - `@berry_schema.pre` / `@berry_schema.post` methods on the BerryType class.
    - `hooks = hooks(pre=..., post=...)` descriptor combining sync/async callbacks.
- Callbacks can modify input data, enforce invariants, and even mutate the ORM instance before it’s returned.

2) **Plain Python / Strawberry mutations**

- Plain async methods on the mutation class with return annotations pointing to Berry types; BerryQL resolves them to the generated Strawberry types.
- Classic Strawberry mutations decorated with `@strawberry.mutation`, ideal for returning primitives or simple payloads.

Example (simplified from `tests/schema.py`):

```python
@berry_schema.mutation()
class Mutation:
        # Generated merge mutations
        merge_posts = mutation('PostQL', comment="Create or update posts")
        merge_users = mutation('UserQL', comment="Create or update users")
        merge_post = mutation('PostQL', single=True, comment="Create or update a single post")

        # Full object mutation implemented manually
        @strawberry.mutation
        async def create_post(self, info: Info, title: str, content: str, author_id: int) -> PostQL:
                session: AsyncSession = info.context["db_session"]
                p = Post(title=title, content=content, author_id=author_id)
                session.add(p)
                await session.flush(); await session.commit()
                return berry_schema.from_model('PostQL', p)

        # ID-only mutation
        @strawberry.mutation
        async def create_post_id(self, info: Info, title: str, content: str, author_id: int) -> int:
                session: AsyncSession = info.context["db_session"]
                p = Post(title=title, content=content, author_id=author_id)
                session.add(p)
                await session.flush(); await session.commit()
                return int(p.id)
```

Merge mutations accept `payload` arguments inferred from the Berry type’s fields (including write-only helpers such as `author_email`) and return BerryQL objects that include read-only fields.

### Merge order, operation context, and lifecycle

Sibling relations are merged in their owning `BerryType` declaration order.
GraphQL input objects are semantically unordered, so client textual field
order, variable-dictionary insertion order, and Strawberry input/dataclass
order do not control database execution.

```python
class SceneQL(BerryType):
    shots = relation("ShotQL", order_by="sequence_number")
    beats = relation("BeatQL", order_by="sequence_number")
```

Here `shots` always merges before `beats`. This rule also applies to
single-relation precreation, `_Replace`, normal list/single merging, and
cascade traversal. Inherited fields keep their base-class relative order,
bases compose left-to-right, subclass fields append in declaration order, and
an overridden field moves to its declaration position in the subclass.

This makes explicit-primary-key cross-sibling references deterministic:

```graphql
scenes: [{
  id: $sceneId
  shots: [{
    id: $clientGeneratedShotId
    _Insert: true
    title: "Shot 1"
  }]
  beats: [{
    id: $beatId
    lines: [{
      id: $lineId
      shotId: $clientGeneratedShotId
    }]
  }]
}]
```

`_Insert` retains the supplied primary key, so the later relation can refer to
it after the earlier relation is flushed. BerryQL does not infer arbitrary FK
dependencies. A database-generated ID unknown to the client cannot be
referenced by a sibling in the same payload.

Every type-level pre/post hook receives the existing dictionary `ctx`. It now
also contains:

- `ctx["operation"]`: the shared `MergeOperationContext` for all descendants
  and every item in a list mutation.
- `ctx["node"]`: the current `MergeNodeContext`.
- The existing `parent`, `relation`, and `delete` keys.

The operation exposes shared `state`, touched primary keys grouped by model,
the request `session`, and deduplicated deferred validation:

```python
def line_post(model_cls, info, line, created, ctx):
    ctx.operation.defer_validation(
        key=("shot-line-continuity", line.scene_id),
        callback=validate_scene_shot_line_continuity,
        scene_id=line.scene_id,
    )
```

Re-registering a key is a no-op. Sync and async validators run in first
registration order after the complete graph/list and a final flush, while all
changes are still uncommitted. Validators may query that transaction but must
not call `commit()` or `rollback()`.

Schema-level hooks can be registered incrementally:

```python
berry_schema.merge_hooks(
    before_merge=[audit_payload],
    before_commit=[validate_operation],
    after_commit=[publish_notifications],
    on_error=[record_failure],
)
```

Each global hook receives `(info, operation, relevant_value)`, where the last
value is the converted payload, merge result, or original exception for that
phase. The lifecycle is:

```text
before_merge
  -> recursive type pre/post hooks
  -> final flush
  -> deferred validators
  -> before_commit
  -> commit
  -> after_commit
```

Any exception before commit explicitly rolls back the full mutation, then runs
`on_error` after rollback and re-raises the original exception. An `on_error`
failure is logged without masking that original error. `after_commit` runs
only after a successful commit and therefore must not be used for validation
that depends on rollback.

### Exception interception (translating errors for clients)

Runtime exceptions raised inside resolvers (queries, mutations, subscriptions)
can be intercepted and translated into clean, client-facing GraphQL errors —
for example, turning a raw SQLAlchemy `IntegrityError` into a friendly message
instead of leaking driver details.

Register handlers on the schema:

```python
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from berryql import UserFacingError

@berry_schema.error_handler(IntegrityError)
def translate_integrity(exc, context):
    return UserFacingError(
        "This email address is already registered",
        code="EMAIL_TAKEN",
    )

@berry_schema.error_handler  # catch-all fallback (no exception classes)
def translate_unexpected(exc, context):
    if isinstance(exc, SQLAlchemyError):
        return UserFacingError("Database error", code="DB_ERROR")
    return None  # keep the original error
```

Equivalently, `berry_schema.register_error_handler(handler, ExcType, ...)`
registers programmatically, before or after `to_strawberry()` is called.

- Handlers may be sync or async and accept `(exc)`, `(exc, context)`, or
  `(exc, context, gql_error)` where `context` is the operation's
  `context_value` and `gql_error` is the final `graphql.GraphQLError`.
- Return values control the translation: `None` keeps the original error, a
  `str` replaces the error message, and an exception instance (typically
  `UserFacingError`) replaces the message and attaches its `extensions`
  (exposed as `extensions` on the GraphQL error, e.g. `{"code": "EMAIL_TAKEN"}`).
- Type-specific handlers always take precedence over catch-all fallbacks,
  regardless of registration order; within each tier the earliest
  registration wins.
- Exceptions raised by a broken handler are logged and never mask the original
  error; validation errors (which have no original exception) are never
  translated.

The mechanism is built into the schema returned by `to_strawberry()` and works
with any Strawberry-compatible server (FastAPI `GraphQLRouter`, ASGI views,
direct `schema.execute(...)` calls, and the synchronous `execute_sync` path).

### Mutation payload control flags

Every generated input type carries optional control flags (underscore-prefixed) to tweak merge behavior:

- `_Delete: Boolean` — when true on a payload item, deletes that row (by its provided PK) instead of upserting.
- `_Insert: Boolean` — when true, forces insertion of a new row even if a PK is provided. The update-lookup is skipped and the row is inserted **with the provided PK** (explicit-PK insert); a duplicate PK errors at the DB level.
- `_Replace: [String]` — lists relation field names on the current item to operate in **replace mode**: upsert the listed children, then delete every other child of that parent not present in the payload (by PK). The delete runs before the upsert to avoid unique-constraint collisions; unknown relation names and single relations are rejected. Relation names are matched in **both** snake_case and camelCase, so this works whether or not Strawberry `autoCamelCase` is enabled (e.g. `_Replace: ["postComments"]` and `_Replace: ["post_comments"]` are equivalent).

Example — replace a post's comments (keep/update the listed ones, delete the rest):

```graphql
mutation {
  merge_posts(payload: [{
    id: 10,
    post_comments: [{ id: 123, content: "bar" }, { content: "new" }],
    _Replace: ["post_comments"]
  }]) { id post_comments { id content } }
}
```

Example — force a fresh insert despite passing an existing PK:

```graphql
mutation {
  merge_posts(payload: [{ id: 10, title: "Copy", _Insert: true }]) { id }
}
```


Domains
-------

Domains let you group related operations (queries and mutations) under a nested namespace while still benefiting from BerryQL’s relation/merge machinery.

- Define a domain by subclassing `BerryDomain` and decorating it with `@berry_schema.domain(name="userDomain")`, `@berry_schema.domain(name="blogDomain")`, etc.
- Inside a domain you can declare:
    - Relations to Berry types (e.g. `users`, `posts`, `postsAsyncFilter`) exactly like on the root query.
    - Domain-scoped merge mutations via `merge_posts = mutation('PostQL', ...)` and similar.
    - Regular Strawberry fields (e.g. `helloDomain`) and subscriptions.
- Nest domains using `domain(OtherDomain)` to build grouped hierarchies (see `groupDomain` in `tests/schema.py`).
- Domains can be exposed on both the root `Query` and root `Mutation` classes:
    - On `Query` they appear as read-only containers (no mutations exposed there).
    - On `Mutation` they expose only their mutation fields (e.g. `blogDomain { merge_posts ... }`, `asyncDomain { merge_posts ... }`).
- Domain-level filters and scopes work the same way as on roots: you can attach `scope` (JSON or callable/async callable) to relations and mutations to enforce contextual rules (user gating, author_id constraints, etc.).


FAQ
---

- Do I need to write resolvers? No for basic scalars/relations/aggregates; BerryQL generates resolvers. You can still add regular @strawberry.field resolvers to your BerryType classes alongside Berry fields.
- How does N+1 get avoided? By pushing down relation arrays/objects into one SQL per root field where possible; otherwise resolvers batch and apply filters with minimal columns.
- Is Sync SQLAlchemy supported? BerryQL targets async SQLAlchemy 2.x APIs; the demo and tests use AsyncSession.


License
-------

MIT (see LICENSE).

Support This Project
-------

If you find this library useful, please consider supporting its development:

👉 [![Donate](https://img.shields.io/badge/PayPal-Donate-blue)](https://www.paypal.com/donate/?business=SFJ6EM3NBP3PL&no_recurring=0&currency_code=USD)
