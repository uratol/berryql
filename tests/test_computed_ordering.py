from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import case, event, func, select
from sqlalchemy.dialects import mssql
from sqlalchemy.ext.asyncio import AsyncSession

from berryql import BerrySchema, BerryType, FieldPermissions, FieldSet, StrawberryConfig, custom, field, relation
from berryql.core.ordering import OrderingError
from tests.models import Post, User


def _score(model, info):
    return select(
        case((model.author_id == info.context["preferred_author"], 0), else_=100) + func.length(model.title)
    ).correlate(model)


def _schema(builder=_score):
    def permissions(type_, info):
        return (
            FieldPermissions(order=FieldSet.none()) if info.context.get("deny_order") else FieldPermissions.allow_all()
        )

    registry = BerrySchema(field_permissions=permissions)

    @registry.type(model=Post)
    class ComputedPostQL(BerryType):
        id = field()
        title = field()
        weighted_score = custom(builder, returns=int)
        author = relation("ComputedUserQL", single=True)

    @registry.type(model=User)
    class ComputedUserQL(BerryType):
        id = field()
        name_length = custom(lambda model: select(func.length(model.name)), returns=int)
        posts = relation("ComputedPostQL")

    @registry.query()
    class Query:
        posts = relation("ComputedPostQL")
        ranked_posts = relation("ComputedPostQL", order_by="weighted_score")
        users = relation("ComputedUserQL")

    return registry, registry.to_strawberry(strawberry_config=StrawberryConfig(auto_camel_case=True))


def _expected(posts, author, descending=False):
    return sorted(
        posts, key=lambda post: len(post.title) + (0 if post.author_id == author else 100), reverse=descending
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("field_name", ["weightedScore", "weighted_score"])
@pytest.mark.parametrize("direction", ["asc", "desc"])
async def test_computed_ordering_without_selection_and_with_pagination(
    db_session, populated_db, engine, field_name, direction
):
    _, schema = _schema()
    author = populated_db["users"][0].id
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await schema.execute(
            "query($field: String!, $direction: Direction) { posts(orderBy: $field, orderDir: $direction, offset: 1, limit: 2) { id } }",
            variable_values={"field": field_name, "direction": direction},
            context_value={"db_session": db_session, "preferred_author": author},
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert not result.errors
    assert [row["id"] for row in result.data["posts"]] == [
        post.id for post in _expected(populated_db["posts"], author, direction == "desc")[1:3]
    ]
    assert len(statements) == 1


@pytest.mark.asyncio
async def test_computed_default_order_and_projection_are_request_specific(db_session, populated_db):
    _, schema = _schema()
    for user in populated_db["users"][:2]:
        result = await schema.execute(
            "{ rankedPosts { id weightedScore } }",
            context_value={"db_session": db_session, "preferred_author": user.id},
        )
        assert not result.errors
        expected = _expected(populated_db["posts"], user.id)
        assert result.data["rankedPosts"] == [
            {"id": post.id, "weightedScore": len(post.title) + (0 if post.author_id == user.id else 100)}
            for post in expected
        ]


NESTED_ORDERING = [
    'orderBy: "weightedScore", orderDir: desc',
    'orderBy: "weighted_score", orderDir: desc',
    'orderMulti: ["weightedScore:desc", "id:asc"]',
    'orderMulti: ["weighted_score:desc", "id:asc"]',
]


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering", NESTED_ORDERING)
async def test_mssql_nested_computed_order_compiles_without_selection(ordering):
    # Keep the MSSQL compilation path covered even in the default SQLite run.
    _, schema = _schema()
    db = AsyncMock(spec=AsyncSession)
    db.info = {}
    db.get_bind.return_value = SimpleNamespace(dialect=mssql.dialect())
    rows = Mock()
    rows.fetchall.return_value = []
    db.execute.return_value = rows
    result = await schema.execute(
        "{ users { id posts(" + ordering + ") { id } } }",
        context_value={"db_session": db, "preferred_author": 7},
    )
    assert not result.errors
    db.execute.assert_awaited_once()
    statement = db.execute.call_args.args[0]
    sql = str(statement.compile(dialect=mssql.dialect(), compile_kwargs={"literal_binds": True}))
    assert "CASE WHEN (posts.author_id = 7)" in sql
    assert "LEN(posts.title)" in sql
    assert "ORDER BY weighted" not in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering", NESTED_ORDERING)
async def test_nested_computed_order_receives_request_info(db_session, populated_db, ordering):
    _, schema = _schema()
    result = await schema.execute(
        "{ users { id posts(" + ordering + ") { id } } }",
        context_value={"db_session": db_session, "preferred_author": populated_db["users"][0].id},
    )
    assert not result.errors
    for user in result.data["users"]:
        expected = sorted(
            [post for post in populated_db["posts"] if post.author_id == user["id"]],
            key=lambda post: len(post.title),
            reverse=True,
        )
        assert [post["id"] for post in user["posts"]] == [post.id for post in expected]


@pytest.mark.asyncio
async def test_dotted_computed_order_without_projection(db_session, populated_db):
    _, schema = _schema()
    result = await schema.execute(
        '{ posts(orderBy: "author.nameLength", orderDir: desc, limit: 10) { id } }',
        context_value={"db_session": db_session},
    )
    assert not result.errors
    names = {user.id: user.name for user in populated_db["users"]}
    expected = sorted(populated_db["posts"], key=lambda post: (-len(names[post.author_id]), post.id))
    assert [row["id"] for row in result.data["posts"]] == [post.id for post in expected]


@pytest.mark.asyncio
async def test_computed_order_permission_is_checked_without_field_selection(db_session, populated_db, engine):
    _, schema = _schema()
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await schema.execute(
            '{ posts(orderBy: "weightedScore") { id } }',
            context_value={"db_session": db_session, "preferred_author": 1, "deny_order": True},
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert result.errors
    assert statements == []


async def _async_builder(model):
    return model.id


def _awaitable_builder(model):
    return _async_builder(model)


@pytest.mark.parametrize("builder", [_async_builder, _awaitable_builder, lambda model: {"invalid": 1}])
def test_computed_order_rejects_non_sql_or_async_builders(builder):
    registry, _ = _schema(builder)
    with pytest.raises(OrderingError, match="synchronous SQL builder|scalar SQL expression"):
        registry._resolve_order_expression(
            Post, registry.types["ComputedPostQL"], "weighted_score", info=SimpleNamespace(context={})
        )
