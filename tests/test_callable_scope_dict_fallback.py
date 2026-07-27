"""Regression test: a callable relation ``scope`` that returns a *dict* must be
honored on the per-parent fallback path.

Background
----------
``relation(..., scope=...)`` documents that a scope may be a callable returning
either a SQLAlchemy expression OR a dict/JSON-string. The JSON/LATERAL pushdown
path already converted dict returns, but the **per-parent fallback** path (used
when no pushdown is possible, e.g. polymorphic relations via
``fk_column_name='entity_id'``) used to pass the callable's return value
straight to ``Select.where()``.

That raised::

    ArgumentError: SQL expression for WHERE/HAVING role expected,
    got {'entity_type': {'eq': 'shot'}}.

These tests pin both layers:
  * the shared ``scope_to_sql_expr`` helper (dispatch unit tests), and
  * an end-to-end polymorphic relation whose scope is a callable returning a
    dict, queried nested so it falls back to per-parent loading.
"""
from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import DeclarativeBase

from berryql import BerrySchema, BerryType, field, relation
from berryql.core.utils import scope_to_sql_expr


# ---------------------------------------------------------------------------
# Unit tests for the shared helper (dispatch behavior)
#
# We do NOT exercise the dict->SQL conversion here (that is covered by the
# to_where_dict / expr_from_where_dict tests and the integration test below).
# These tests pin the *dispatch*: None, passthrough of a non-dict value, and
# that a callable is invoked and its result recursively normalized.
# ---------------------------------------------------------------------------


def test_scope_to_sql_expr_none_returns_none():
    assert scope_to_sql_expr(None, None) is None  # type: ignore[arg-type]


def test_scope_to_sql_expr_passthrough_non_dict():
    sentinel = object()
    # A non-dict/str/callable value is returned verbatim (assumed SQL expression).
    assert scope_to_sql_expr(None, sentinel) is sentinel  # type: ignore[arg-type]


def test_scope_to_sql_expr_callable_passthrough_non_dict():
    sentinel = object()

    def scope(M, _info):
        return sentinel

    assert scope_to_sql_expr(None, scope, info=None) is sentinel  # type: ignore[arg-type]


def test_scope_to_sql_expr_callable_returning_none():
    def scope(M, _info):
        return None

    assert scope_to_sql_expr(None, scope, info=None) is None  # type: ignore[arg-type]


def test_scope_to_sql_expr_callable_callable_returning_dict_is_normalized():
    """A callable returning a dict must NOT be returned as a raw dict.

    We assert only that the result is not the dict itself (i.e. it was
    normalized), using a model whose column matches the dict so conversion
    succeeds rather than raising.
    """
    from tests.models import Post  # reuse an existing model with an 'author_id' column

    def scope(M, _info):
        return {"author_id": {"eq": 1}}

    result = scope_to_sql_expr(Post, scope, info=None)
    assert result is not None
    assert not isinstance(result, dict)


# ---------------------------------------------------------------------------
# End-to-end: callable-dict scope on a polymorphic relation (per-parent fallback)
# ---------------------------------------------------------------------------

class _RegBase(DeclarativeBase):
    pass


class _Container(_RegBase):
    __tablename__ = "reg_container"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)


class _Ref(_RegBase):
    """Polymorphic ref attached to a container via entity_id (like Shot.refs)."""
    __tablename__ = "reg_ref"
    id = Column(Integer, primary_key=True)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(Integer, nullable=False)
    title = Column(String(50), nullable=False)


_reg_schema = BerrySchema()


@_reg_schema.type(model=_Container)
class ContainerQL(BerryType):
    id = field()
    name = field()
    # Polymorphic relation scoped by a CALLABLE that returns a DICT.
    # fk_column_name='entity_id' defeats LATERAL pushdown -> per-parent fallback.
    refs = relation(
        "RefQL",
        fk_column_name="entity_id",
        order_by="id",
        scope=lambda M, _info: {"entity_type": {"eq": "container"}},
    )


@_reg_schema.type(model=_Ref)
class RefQL(BerryType):
    id = field()
    entity_type = field()
    entity_id = field()
    title = field()


@_reg_schema.query()
class Query:
    containers = relation("ContainerQL", order_by="id")


_reg_strawberry = _reg_schema.to_strawberry()


@pytest.mark.asyncio
async def test_callable_dict_scope_polymorphic_relation_fallback(engine):
    """End-to-end: a polymorphic relation whose scope is a callable returning a
    dict must resolve without error and apply the filter correctly.

    Depending on the backend, BerryQL may serve this via JSON/LATERAL pushdown or
    via the per-parent fallback; both paths must honor a callable-dict scope. The
    unit tests above pin the helper used by the fallback path directly.
    """
    async with engine.begin() as conn:
        await conn.run_sync(_RegBase.metadata.create_all)

    from sqlalchemy.ext.asyncio import async_sessionmaker

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        session.add_all([
            _Container(id=1, name="c1"),
            _Container(id=2, name="c2"),
            # Refs for container 1 (matching scope) + a decoy of another entity_type
            _Ref(id=10, entity_type="container", entity_id=1, title="r1"),
            _Ref(id=11, entity_type="container", entity_id=1, title="r2"),
            _Ref(id=12, entity_type="container", entity_id=2, title="r3"),
            _Ref(id=13, entity_type="other", entity_id=1, title="decoy"),
        ])
        await session.commit()

    async with Session() as session:
        q = """
        query {
          containers(order_by: "id") {
            id
            name
            refs(order_by: "id") { id title entity_type }
          }
        }
        """
        res = await _reg_strawberry.execute(
            q, context_value={"db_session": session}
        )
        assert res.errors is None, f"Unexpected errors: {res.errors}"
        data = res.data["containers"]

        c1 = next(c for c in data if c["id"] == 1)
        c2 = next(c for c in data if c["id"] == 2)
        # Container 1 sees only its 'container' refs (decoy entity_type filtered out)
        assert [r["id"] for r in c1["refs"]] == [10, 11]
        assert all(r["entity_type"] == "container" for r in c1["refs"])
        # Container 2 sees its single ref
        assert [r["id"] for r in c2["refs"]] == [12]
