from __future__ import annotations

import pytest
from sqlalchemy import JSON as SQLJSON, Boolean, Column, Integer, case, event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from strawberry.scalars import JSON

from berryql import BerrySchema, BerryType, custom, field, relation

Base = declarative_base()


class Entry(Base):
    __tablename__ = "custom_json_entries"
    id = Column(Integer, primary_key=True)
    visible = Column(Boolean, nullable=False)
    payload = Column(SQLJSON)


berry = BerrySchema()


@berry.type(model=Entry)
class EntryQL(BerryType):
    id = field()
    inline_payload = custom(
        lambda M: case((M.visible.is_(True), M.payload), else_=None),
        returns=JSON,
    )


@berry.query()
class Query:
    entries = relation("EntryQL", order_by="id")


schema = berry.to_strawberry()


def test_custom_json_declares_nullable_json_scalar():
    assert "inline_payload: JSON" in schema.as_str()
    assert "inline_payload: JSON!" not in schema.as_str()


@pytest.mark.asyncio
async def test_custom_json_projects_structured_values_and_null_in_one_query():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            values = [{"type": "form", "data": {"refs": ["one", "two"]}}, [1, True, None], "text", 42, False, None]
            db.add_all([Entry(id=i + 1, visible=True, payload=value) for i, value in enumerate(values)])
            db.add(Entry(id=100, visible=False, payload={"large": "x" * 10000}))
            await db.commit()
            queries = []

            def record_query(_connection, _cursor, statement, _parameters, _context, _executemany):
                queries.append(statement)

            event.listen(engine.sync_engine, "before_cursor_execute", record_query)
            result = await schema.execute(
                "{ entries { id payload: inline_payload } }", context_value={"db_session": db},
            )
            assert not result.errors
            assert [row["payload"] for row in result.data["entries"]] == [*values, None]
            assert len(queries) == 1
    finally:
        await engine.dispose()

