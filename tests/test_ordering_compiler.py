from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import event, select

from berryql import FieldSet
from berryql.adapters.mssql import MSSQLAdapter
from berryql.core.ordering import OrderTerm, OrderingCompiler, OrderingError
from tests.models import Post
from tests.test_field_permissions import (
    PermissionPostQL,
    _context,
    _permissions,
    permission_graphql_schema,
    permission_schema,
)


@pytest.mark.parametrize(
    "terms",
    [
        ["broken", "title:asc", "id:desc"],
        ["title:asc", "broken", "id:desc"],
        ["title:asc", "id:desc", "broken"],
        ["title:sideways"],
        ["title:asc:extra"],
        [":asc"],
    ],
)
def test_every_order_multi_term_is_strictly_validated(terms):
    compiler = OrderingCompiler(permission_schema)
    with pytest.raises(OrderingError):
        compiler.parse(order_multi=terms)


def test_order_by_and_direction_are_strictly_validated():
    compiler = OrderingCompiler(permission_schema)
    with pytest.raises(OrderingError, match="non-empty"):
        compiler.parse(order_by="")
    with pytest.raises(OrderingError, match="direction"):
        compiler.parse(order_by="id", order_dir="sideways")


@pytest.mark.asyncio
async def test_invalid_middle_term_does_not_execute_partial_sql(db_session, populated_db, engine):
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await permission_graphql_schema.execute(
            'query { posts(order_multi: ["title:asc", "broken", "id:desc"]) { id } }',
            context_value=_context(db_session),
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)

    assert result.errors
    assert "order_multi" in str(result.errors[0])
    assert statements == []


@pytest.mark.asyncio
async def test_denied_dotted_order_path_fails_before_sql(db_session, populated_db, engine):
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    context = _context(
        db_session,
        field_permissions={"PermissionUserQL": _permissions(order=FieldSet.all_except("name"))},
    )
    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await permission_graphql_schema.execute(
            'query { posts(order_by: "author.name") { id } }',
            context_value=context,
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)

    assert result.errors
    assert "not allowed for order" in str(result.errors[0])
    assert statements == []


def test_paginated_sql_adds_pk_tiebreaker_without_duplicate():
    compiler = OrderingCompiler(permission_schema)

    def resolve_expression(statement, path, join_cache):
        return statement, permission_schema._resolve_order_expression(Post, PermissionPostQL, path)

    terms = compiler.parse(order_multi=["title:desc"])
    statement = compiler.apply_sqlalchemy(
        select(Post.id),
        model_cls=Post,
        berry_type=PermissionPostQL,
        terms=terms,
        resolve_expression=resolve_expression,
        add_pk_tiebreaker=True,
    )
    order_clause = str(statement).split("ORDER BY", 1)[1]
    assert "posts.title DESC" in order_clause
    assert "posts.id ASC" in order_clause

    explicit_pk = compiler.parse(order_multi=["title:desc", "id:desc"])
    explicit_statement = compiler.apply_sqlalchemy(
        select(Post.id),
        model_cls=Post,
        berry_type=PermissionPostQL,
        terms=explicit_pk,
        resolve_expression=resolve_expression,
        add_pk_tiebreaker=True,
    )
    explicit_order = str(explicit_statement).split("ORDER BY", 1)[1]
    assert explicit_order.count("posts.id") == 1
    assert "posts.id DESC" in explicit_order

    mssql_order = MSSQLAdapter().build_order_clause(
        Post,
        "posts",
        "title",
        "desc",
        None,
        add_pk_tiebreaker=True,
    )
    assert mssql_order == "[posts].[title] DESC, [posts].[id] ASC"


@pytest.mark.asyncio
async def test_python_fallback_null_order_matches_dialect_rules():
    compiler = OrderingCompiler(permission_schema)
    items = [
        SimpleNamespace(id=1, value=None),
        SimpleNamespace(id=2, value=2),
        SimpleNamespace(id=3, value=1),
    ]

    async def ordered(direction, nulls_first):
        return await compiler.apply_python(
            items,
            [OrderTerm("value", direction)],
            lambda item, path: getattr(item, path),
            nulls_first=nulls_first,
        )

    sqlite_asc = await ordered("asc", lambda direction: direction == "asc")
    postgres_asc = await ordered("asc", lambda direction: direction == "desc")
    sqlite_desc = await ordered("desc", lambda direction: direction == "asc")
    postgres_desc = await ordered("desc", lambda direction: direction == "desc")

    assert [item.id for item in sqlite_asc] == [1, 3, 2]
    assert [item.id for item in postgres_asc] == [3, 2, 1]
    assert [item.id for item in sqlite_desc] == [2, 3, 1]
    assert [item.id for item in postgres_desc] == [1, 2, 3]
