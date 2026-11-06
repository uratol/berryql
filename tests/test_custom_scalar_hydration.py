"""Prefetched SQL NULL is a loaded custom scalar, not a fallback request."""

from types import SimpleNamespace
from typing import Annotated
from unittest.mock import AsyncMock, Mock

import pytest
import strawberry
from sqlalchemy import Column, Integer, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

from berryql import BerrySchema, BerryType, FieldPermissions, FieldSet, custom, field


Base = declarative_base()
pytestmark = pytest.mark.asyncio


class Entry(Base):
    __tablename__ = "custom_scalar_hydration_entries"
    id = Column(Integer, primary_key=True)
    amount = Column(Integer, nullable=True)


@pytest.fixture
def sample():
    builder_calls = []

    def score_builder(model):
        builder_calls.append(model)
        return select(model.amount).correlate(model)

    def permissions(type_, info):
        if type_.__name__ == "HydratedEntryQL" and info.context.get("deny_score"):
            return FieldPermissions(select=FieldSet.only("id"))
        return FieldPermissions.allow_all()

    berry = BerrySchema(field_permissions=permissions)

    @berry.type(model=Entry)
    class HydratedEntryQL(BerryType):
        id = field()
        score = custom(score_builder, returns=int)

    @berry.query()
    class Query:
        @strawberry.field
        def source(self, info: strawberry.Info) -> Annotated[
            "HydratedEntryQL", strawberry.lazy("berryql.registry"),
        ]:
            return info.context["source"]

    schema = berry.to_strawberry()
    source = berry.from_model("HydratedEntryQL", Entry(id=7, amount=42))
    assert "score" not in source.__dict__
    assert not hasattr(source, "_score_prefetched")
    db = AsyncMock(spec=AsyncSession)
    db.info = {}
    rows = Mock()
    rows.first.return_value = SimpleNamespace(_mapping={"score": 42})
    db.execute.return_value = rows
    builder_calls.clear()
    return SimpleNamespace(
        schema=schema, source=source, db=db, rows=rows, builder_calls=builder_calls,
        context={"db_session": db, "source": source},
    )


@pytest.mark.parametrize("attribute", ["score", "_score_prefetched"])
@pytest.mark.parametrize("value", [None, 0, 17])
async def test_loaded_custom_scalar_does_not_run_builder_or_query(sample, attribute, value):
    setattr(sample.source, attribute, value)
    result = await sample.schema.execute("{ source { score } }", context_value=sample.context)

    assert result.errors is None, result.errors
    assert result.data == {"source": {"score": value}}
    assert sample.builder_calls == []
    sample.db.execute.assert_not_awaited()


@pytest.mark.parametrize("value", [None, 42])
@pytest.mark.parametrize("callable_attribute", [False, True])
async def test_unloaded_custom_scalar_runs_builder_and_query(sample, value, callable_attribute):
    if callable_attribute:
        sample.source.score = lambda: None
    sample.rows.first.return_value = SimpleNamespace(_mapping={"score": value})
    result = await sample.schema.execute("{ source { score } }", context_value=sample.context)

    assert result.errors is None, result.errors
    assert result.data == {"source": {"score": value}}
    assert len(sample.builder_calls) == 1
    sample.db.execute.assert_awaited_once()
    statement = sample.db.execute.await_args.args[0]
    assert statement.compile().params == {"id_1": 7}


async def test_prefetched_custom_scalar_still_checks_read_permissions(sample):
    sample.source.score = 17
    sample.context["deny_score"] = True
    result = await sample.schema.execute("{ source { score } }", context_value=sample.context)

    assert result.errors is None, result.errors
    assert result.data == {"source": {"score": None}}
    assert sample.builder_calls == []
    sample.db.execute.assert_not_awaited()
