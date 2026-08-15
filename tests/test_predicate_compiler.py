from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from berryql import FilterLimits
from berryql.adapters.mssql import MSSQLAdapter
from berryql.core.predicates import (
    Conjunction,
    PredicateCompiler,
    PredicateError,
    TrustedExpression,
    UnsupportedPredicateError,
)
from tests.models import Post, User


class _Info:
    def __init__(self):
        self.context = {}
        self.operation = object()
        self.variable_values = {}


def _compiler():
    return PredicateCompiler(SimpleNamespace(_auto_camel_case=False))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        {"id": {"eq": 1}},
        '{"id": {"eq": 1}}',
        [{"id": {"gte": 1}}, '{"id": {"lte": 2}}'],
    ],
)
async def test_predicate_matrix_compiles_dict_string_and_fragments(value):
    compiler = _compiler()
    predicate = await compiler.resolve(value, Post, _Info(), trusted=True)
    expression = compiler.compile_sqlalchemy(predicate, Post)
    sql = str(select(Post.id).where(expression))

    assert isinstance(predicate, Conjunction)
    assert "posts.id" in sql


@pytest.mark.asyncio
async def test_predicate_matrix_resolves_sync_async_and_expression_providers():
    compiler = _compiler()
    info = _Info()

    def sync_provider(model_cls, provided_info):
        assert model_cls is Post
        assert provided_info is info
        return {"id": {"eq": 1}}

    async def async_provider(model_cls, provided_info):
        assert model_cls is Post
        assert provided_info is info
        return model_cls.id == 1

    sync_predicate = await compiler.resolve(sync_provider, Post, info)
    async_predicate = await compiler.resolve(async_provider, Post, info)
    direct_predicate = await compiler.resolve(Post.id == 1, Post, info)

    assert "posts.id" in str(compiler.compile_sqlalchemy(sync_predicate, Post))
    assert isinstance(async_predicate, TrustedExpression)
    assert isinstance(direct_predicate, TrustedExpression)


@pytest.mark.asyncio
async def test_predicate_provider_is_called_once_across_pushdown_and_async_fallback():
    compiler = _compiler()
    info = _Info()
    calls = 0

    def awaitable_provider(model_cls, provided_info):
        nonlocal calls
        calls += 1

        async def result():
            return {"id": {"eq": 1}}

        return result()

    with pytest.raises(UnsupportedPredicateError):
        compiler.resolve_sync(awaitable_provider, Post, info)

    predicate = await compiler.resolve(awaitable_provider, Post, info)
    assert compiler.compile_sqlalchemy(predicate, Post) is not None
    assert calls == 1


@pytest.mark.asyncio
async def test_type_error_inside_provider_is_original_and_not_retried():
    compiler = _compiler()
    calls = 0

    def broken_provider(model_cls, info):
        nonlocal calls
        calls += 1
        raise TypeError("provider body failed")

    with pytest.raises(TypeError, match="provider body failed"):
        await compiler.resolve(broken_provider, Post, _Info())
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("{broken", "Invalid where JSON"),
        ({"missing": {"eq": 1}}, "Unknown where column"),
        ({"id": {"unknown": 1}}, "Unknown where operator"),
    ],
)
async def test_invalid_predicates_fail_closed(value, message):
    compiler = _compiler()

    with pytest.raises((PredicateError, ValueError), match=message):
        predicate = await compiler.resolve(value, Post, _Info())
        compiler.compile_sqlalchemy(predicate, Post)


def test_mssql_compiles_supported_forms_and_rejects_unsupported_scope():
    compiler = _compiler()
    adapter = MSSQLAdapter()
    info = _Info()

    dict_parts = compiler.compile_mssql_sync({"id": {"eq": 1}}, Post, adapter, info)
    expression_parts = compiler.compile_mssql_sync(Post.id == 1, Post, adapter, info)
    callable_dict_parts = compiler.compile_mssql_sync(
        lambda model_cls, provided_info: {"id": {"eq": 1}},
        Post,
        adapter,
        info,
    )
    callable_expression_parts = compiler.compile_mssql_sync(
        lambda model_cls, provided_info: model_cls.id == 1,
        Post,
        adapter,
        info,
    )
    assert len(dict_parts) == 1 and "id" in dict_parts[0].lower()
    assert len(expression_parts) == 1 and "id" in expression_parts[0].lower()
    assert callable_dict_parts == dict_parts
    assert callable_expression_parts == expression_parts

    async def async_scope(model_cls, provided_info):
        return {"id": {"eq": 1}}

    with pytest.raises(UnsupportedPredicateError, match="async"):
        compiler.compile_mssql_sync(async_scope, Post, adapter, info)

    with pytest.raises(UnsupportedPredicateError, match="cannot safely compile"):
        compiler.compile_mssql(TrustedExpression(object()), Post, adapter, strict=True)


def test_mssql_adapter_applies_compiled_expression_instead_of_dropping_it():
    compiler = _compiler()
    adapter = MSSQLAdapter()
    expression_parts = compiler.compile_mssql_sync(Post.id == 1, Post, adapter, _Info())

    sql = str(
        adapter.build_relation_list_json_full(
            parent_table=User,
            parent_pk_name="id",
            child_model=Post,
            fk_col_name="author_id",
            projected_columns=["id"],
            rel_where=None,
            rel_default_where=None,
            extra_where_sql=expression_parts,
        )
    )

    assert "posts.id = 1" in sql


# ---------------------------------------------------------------------------
# Regression: SQL operands in where-dicts (trusted scopes)
#
# 0.5.0's operand-shape validation rejected
# {'author_id': {'in': select(...).scalar_subquery()}} — the documented
# multi-tenant scope form (see mutations._enforce_scope docstring) — with
# "Where operator 'in' for '<column>' requires a non-empty list".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trusted_scope_dict_with_scalar_subquery_operand_compiles():
    compiler = _compiler()
    subquery = select(User.id).where(User.name == "alice").scalar_subquery()
    scope = {"author_id": {"in": subquery}}

    predicate = await compiler.resolve(scope, Post, _Info(), trusted=True)
    expression = compiler.compile_sqlalchemy(predicate, Post)
    sql = str(select(Post.id).where(expression))

    assert "posts.author_id IN" in sql
    assert "(SELECT users.id" in sql


@pytest.mark.asyncio
async def test_in_and_not_in_accept_plain_select_operands_unwrapped():
    compiler = _compiler()
    plain = select(User.id).where(User.name == "alice")

    for op, expected in (("in", "IN"), ("not_in", "NOT IN")):
        predicate = await compiler.resolve(
            {"author_id": {op: plain}}, Post, _Info(), trusted=True
        )
        expression = compiler.compile_sqlalchemy(predicate, Post)
        sql = str(select(Post.id).where(expression))
        # The Select must reach col.in_() unwrapped; wrapping it in a list
        # raises "IN expression list, SELECT construct, or bound parameter
        # object expected".
        assert f"{expected} (SELECT users.id" in sql


@pytest.mark.asyncio
async def test_between_accepts_sql_expression_items():
    compiler = _compiler()
    upper = select(User.id).where(User.id == 1).scalar_subquery()

    predicate = await compiler.resolve(
        {"id": {"between": [1, upper]}}, Post, _Info(), trusted=True
    )
    expression = compiler.compile_sqlalchemy(predicate, Post)
    sql = str(select(Post.id).where(expression))

    assert "BETWEEN" in sql
    assert "SELECT" in sql


def test_caller_empty_in_list_still_rejected():
    compiler = _compiler()

    with pytest.raises(PredicateError, match="non-empty"):
        compiler.parse({"id": {"in": []}}, Post, trusted=False)
    with pytest.raises(PredicateError, match="non-empty"):
        compiler.parse({"id": {"not_in": []}}, Post, trusted=False)


def test_caller_in_cost_limits_enforced_but_trusted_scopes_exempt():
    schema = SimpleNamespace(
        filter_limits=FilterLimits(max_in_items=2),
        _operators=None,
        _auto_camel_case=False,
    )
    compiler = PredicateCompiler(schema)

    with pytest.raises(PredicateError, match="max_in_items"):
        compiler.parse({"id": {"in": [1, 2, 3]}}, Post, trusted=False)

    # Trusted scopes are validated for shape but not subject to caller cost
    # limits (FilterLimits contract).
    compiler.parse({"id": {"in": [1, 2, 3]}}, Post, trusted=True)


def test_mssql_compiles_sql_operand_predicates_instead_of_dropping_them():
    compiler = _compiler()
    adapter = MSSQLAdapter()
    subquery = select(User.id).where(User.id == 1).scalar_subquery()

    parts = compiler.compile_mssql_sync(
        {"author_id": {"in": subquery}}, Post, adapter, _Info()
    )

    assert len(parts) == 1
    assert "IN" in parts[0].upper()
    assert "SELECT" in parts[0].upper()
