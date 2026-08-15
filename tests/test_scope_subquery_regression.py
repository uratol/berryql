"""End-to-end regression: scope where-dicts with SQL subquery operands.

0.5.0's operand-shape validation rejected
``{"author_id": {"in": select(...).scalar_subquery()}}`` — the documented
multi-tenant scope form (see the ``mutations._enforce_scope`` docstring) —
with ``Where operator 'in' for 'author_id' requires a non-empty list``,
breaking every scoped query and mutation in consumer apps.

These tests exercise the full query and mutation paths via a type-level
scope returning a subquery-based where-dict.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from tests.fixtures import *  # noqa: F401,F403
from tests.models import User
from tests.schema import PostQL, schema as berry_schema


@pytest.fixture
def author_one_type_scope():
    """Posts authored by user 1 — the canonical tenant-membership scope shape."""
    orig = getattr(PostQL, "__type_scope__", None)
    PostQL.__type_scope__ = {
        "author_id": {"in": select(User.id).where(User.id == 1).scalar_subquery()}
    }
    try:
        yield
    finally:
        if orig is None:
            try:
                delattr(PostQL, "__type_scope__")
            except Exception:
                pass
        else:
            PostQL.__type_scope__ = orig


@pytest.mark.asyncio
async def test_query_type_scope_with_subquery_dict_filters_rows(
    db_session, populated_db, author_one_type_scope
):
    q = """
    {
      posts { id author_id }
    }
    """
    res = await berry_schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    assert posts, "expected in-scope rows to be visible"
    assert {int(p["author_id"]) for p in posts} == {1}


@pytest.mark.asyncio
async def test_mutation_type_scope_with_subquery_dict_enforced(
    db_session, populated_db, author_one_type_scope
):
    m = """
    mutation($p: [PostQLInput!]!) {
      merge_posts(payload: $p) { id title author_id }
    }
    """

    # Out-of-scope: author_id = 2 is not in the membership subquery.
    res_other = await berry_schema.execute(
        m,
        variable_values={"p": [{"title": "A", "content": "B", "author_id": 2}]},
        context_value={"db_session": db_session},
    )
    assert res_other.errors is not None, "expected scope violation error"
    assert "out of scope" in str(res_other.errors[0]).lower()

    # In-scope: author_id = 1.
    res_self = await berry_schema.execute(
        m,
        variable_values={"p": [{"title": "C", "content": "D", "author_id": 1}]},
        context_value={"db_session": db_session},
    )
    assert res_self.errors is None, res_self.errors
    data = res_self.data["merge_posts"][0]
    assert int(data["author_id"]) == 1
