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


Quick start
-----------

Install

- Core:
    - pip install berryql
- Optional DB drivers and helpers (choose what you need):
    - pip install "berryql[adapters]"  # asyncpg, pyodbc, python-dotenv

Requirements: Python 3.8+ (tested up to 3.13), Strawberry GraphQL, SQLAlchemy 2.x


Define your models (SQLAlchemy)

This repo ships demo models in `tests/models.py` (User, Post, PostComment). Any SQLAlchemy ORM models work.


Define Berry types

Map models to GraphQL using Berry’s DSL: scalars, relations, aggregates, and custom fields.

Example (excerpt adapted from tests):

- File: `tests/schema.py` builds the runtime Strawberry schema from Berry’s registry.
- You only need to provide an async SQLAlchemy session via GraphQL context as `db_session` or `db`.

Highlights in the example below:

- field(): map model columns
- relation(): to-one or to-many; supports arguments, ordering, pagination
- count(): quick aggregate of a relation (e.g., posts count)
- custom()/custom_object(): inject SQL selects for computed values (pushed down when possible)
- @berry_schema.query(): define root collections (users, posts, etc.)


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

BerryQL supports two mutation styles inside the class registered with `@berry_schema.mutation()`:

1) Plain Python method (BerryQL maps the return annotation to the runtime Strawberry type):

        - Annotate with the Berry type name (e.g., `-> "PostQL"`) or the Berry class; BerryQL resolves it to the generated Strawberry type.
        - Return an instance using `berry_schema.from_model('PostQL', orm_instance)` to attach the SQLAlchemy model and seed scalar fields.

2) Classic Strawberry mutation with `@strawberry.mutation`:

        - Best for returning primitives (e.g., integers) or simple payloads.

Example:

```python
@berry_schema.mutation()
class Mutation:
        # Full object mutation
        async def create_post(self, info: Info, title: str, content: str, author_id: int) -> "PostQL":
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

Querying the object mutation:

```graphql
mutation {
    create_post(title: "Hello", content: "Body", author_id: 1) {
        id
        title
        author_id
    }
}
```


FAQ
---

- Do I need to write resolvers? No for basic scalars/relations/aggregates; BerryQL generates resolvers. You can still add regular @strawberry.field resolvers to your BerryType classes alongside Berry fields.
- How does N+1 get avoided? By pushing down relation arrays/objects into one SQL per root field where possible; otherwise resolvers batch and apply filters with minimal columns.
- Is Sync SQLAlchemy supported? BerryQL targets async SQLAlchemy 2.x APIs; the demo and tests use AsyncSession.


License
-------

MIT (see LICENSE).
