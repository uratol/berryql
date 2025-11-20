# BerryQL
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

Here is a minimal end‑to‑end sketch using BerryQL with two types, relations, a query, and a merge mutation:

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

class Post:
        id: Mapped[int] = mapped_column(primary_key=True)
        title: Mapped[str]
        author_id: Mapped[int]
        author: Mapped[User] = relationship(back_populates="posts")


@berry_schema.type(model=Post)
class PostQL(BerryType):
        id = field()
        title = field()
        author_id = field()  # autoCamelCase → authorId
    # many‑to‑one relation: post → author
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
        posts { id title }
    }
}
```

```graphql
mutation {
    mergePosts(payload: [{ title: "Hello", authorId: 1 }]) {
        id
        title
    }
}
```


5‑minute try-out
-----------------

If you just want to see it working quickly, you don’t need to design a schema from scratch – this repo already contains a full demo schema and models.

**1. Install and run tests (uses in‑memory SQLite)**

```bash
pip install -r requirements.txt
pytest -q
```

This spins up the demo models (`tests/models.py`), Berry schema (`tests/schema.py`), and exercises queries, relations, domains, and mutations.

**2. Run the demo GraphQL API (FastAPI + Strawberry)**

```bash
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

Behind the scenes this uses the Berry schema from `tests/schema.py` and an async SQLAlchemy session from the FastAPI app.

**3. Minimal “how would I use this in my app?” sketch**

At a high level you will:

1. Define SQLAlchemy models (or reuse existing ones).
2. Map them to Berry types with `@berry_schema.type` and `field()/relation()`.
3. Define a `@berry_schema.query()` class for roots and optionally `@berry_schema.mutation()` / `@berry_schema.domain()` classes for mutations and domains.
4. Call `berry_schema.to_strawberry()` and plug the resulting schema into Strawberry/FastAPI.

The rest of this README goes into the details of fields, relations, filters, JSON where, custom scalars/objects, domains, and merge mutations.


What queries look like (and what SQL runs)

- Only selected columns are fetched for each table.
- When selecting users with posts, BerryQL will execute one SQL for users and one for posts (root fields), aggregating nested rows without joining unrelated tables.
- For simple selections like `users { id }`, the SQL only selects the id column.

See tests for concrete assertions:

- `tests/test_sql_projection.py` ensures only requested columns are present and unrelated tables aren’t touched.
- `tests/test_relations_pagination_aggregate.py` ensures “one SQL per root field” when pushdown is supported.


Run the example API (GraphiQL)
------------------------------

There’s a minimal FastAPI app that mounts the Berry-generated Strawberry schema.

- File: `examples/main.py`
- Endpoint: http://127.0.0.1:8000/graphql (GraphiQL enabled)

Quick steps (PowerShell):

1) Create venv and install deps
     - python -m venv .venv
     - .venv\Scripts\Activate.ps1
     - pip install -r requirements.txt
2) Run the app
     - python -m uvicorn examples.main:app --reload --host 127.0.0.1 --port 8000

Environment variables (optional):

- BERRYQL_TEST_DATABASE_URL: async SQLAlchemy URL (e.g., postgresql+asyncpg://… or mssql+aioodbc:///?odbc_connect=…)
- SQL_ECHO: set 1 to log SQL (default 1)

See `README_RUN_FASTAPI.md` for more.


Core concepts
-------------

- BerrySchema: registry for types and root query.
- BerryType: base for GraphQL types. Use Berry’s field descriptors on subclasses.
- field(): scalar column mapping.
- relation(target, single=False, …): relation to another Berry type. Supports:
    - arguments: map GraphQL args to SQL filters (column+op or builder callable)
    - where: default JSON-style where for the relation (dict or JSON string) or callable(model_cls, info)
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

At runtime BerryQL validates columns, operators, and types and applies them in SQL. When relation pushdown is skipped (e.g., because of a ‘where’ argument), filters are still applied safely in resolvers.

Supported operators include: eq, ne, lt, lte, gt, gte, like, ilike, in, between, contains, starts_with, ends_with. You can register more.


Ordering and pagination
-----------------------

- order_by: a single column
- order_dir: asc|desc
- order_multi: ["created_at:desc", "id:asc"]
- limit/offset: integers

Invalid order_by values raise a GraphQL error with the allowed fields.


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
