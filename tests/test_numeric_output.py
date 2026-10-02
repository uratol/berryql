from decimal import Decimal
from typing import Annotated, List, Optional

import pytest
import strawberry
from sqlalchemy import Column, ForeignKey, Integer, Numeric, event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, relationship

from berryql import BerrySchema, BerryType, FieldPermissions, FieldSet, custom, field, mutation, relation


Base = declarative_base()


class NumericParent(Base):
    __tablename__ = "numeric_output_parents"
    id = Column(Integer, primary_key=True)
    amount = Column(Numeric(12, 3))
    children = relationship("NumericChild", back_populates="parent")


class NumericChild(Base):
    __tablename__ = "numeric_output_children"
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey(NumericParent.id))
    amount = Column(Numeric(12, 3))
    parent = relationship(NumericParent, back_populates="children")


def _permissions(type_, info):
    return info.context.get("permissions", FieldPermissions.allow_all())


berry = BerrySchema(field_permissions=_permissions)


@berry.type(model=NumericChild)
class NumericChildQL(BerryType):
    id = field()
    parent_id = field()
    amount = field()
    parent = relation("NumericParentQL", single=True)


@berry.type(model=NumericParent)
class NumericParentQL(BerryType):
    id = field()
    amount = field()
    amount_alias = field("amount", read_only=True)
    projected_amount = custom(lambda M: M.amount, returns=float)
    float_list = field(returns=List[Optional[float]], read_only=True)
    children = relation("NumericChildQL", order_by="id")

    @strawberry.field
    def native_amount(self) -> Optional[float]:
        model = getattr(self, "_model", self)
        return model.__dict__.get("amount")


@berry.query()
class Query:
    parents = relation("NumericParentQL", order_by="id")
    children = relation("NumericChildQL", order_by="id")

    @strawberry.field
    def raw_parent(self, info: strawberry.Info) -> Annotated["NumericParentQL", strawberry.lazy("berryql.registry")]:
        return info.context["source"]


@berry.mutation()
class Mutation:
    merge_parent = mutation("NumericParentQL", single=True)


schema = berry.to_strawberry()


@pytest.mark.parametrize("source_kind", ["raw", "from_model", "attached", "hydrated"])
@pytest.mark.parametrize("value", [None, Decimal("0.000"), Decimal("12.345")])
async def test_decimal_output_from_all_resolver_sources(source_kind, value):
    model = NumericParent(id=1, amount=value)
    if source_kind == "raw":
        source = model
    elif source_kind == "from_model":
        source = berry.from_model("NumericParentQL", model)
    else:
        source = berry._st_types["NumericParentQL"]()
        if source_kind == "attached":
            source._model = model
        else:
            source.id = 1
            source.amount = value
            source.amount_alias = value

    selection = "id amount native_amount"
    if source_kind != "raw":
        selection += " amount_alias"
    result = await schema.execute(
        "{ raw_parent { " + selection + " } }",
        context_value={"source": source},
    )
    assert result.errors is None, result.errors
    number = None if value is None else float(value)
    expected = {"id": 1, "amount": number, "native_amount": number}
    if source_kind != "raw":
        expected["amount_alias"] = number
    assert result.data["raw_parent"] == expected
    assert model.amount == value
    if value is not None:
        assert isinstance(model.amount, Decimal)


async def test_float_lists_and_permissions():
    source = NumericParent(id=1, amount=Decimal("12.345"))
    source.float_list = [Decimal("0"), None, Decimal("12.345")]
    result = await schema.execute(
        "{ raw_parent { float_list } }", context_value={"source": source},
    )
    assert result.errors is None, result.errors
    assert result.data["raw_parent"]["float_list"] == [0.0, None, 12.345]
    assert isinstance(source.float_list[0], Decimal)

    result = await schema.execute(
        "{ raw_parent { id amount native_amount } }",
        context_value={"source": source, "permissions": FieldPermissions(select=FieldSet.only("id"))},
    )
    assert result.errors is None, result.errors
    assert result.data["raw_parent"] == {"id": 1, "amount": None, "native_amount": None}


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
async def test_non_finite_decimal_still_fails_graphql_serialization(value):
    result = await schema.execute(
        "{ raw_parent { amount } }",
        context_value={"source": NumericParent(id=1, amount=value)},
    )
    assert len(result.errors or []) == 1
    assert result.errors[0].path == ["raw_parent", "amount"]
    assert "Float cannot represent non numeric value" in result.errors[0].message


async def test_sql_projections_nested_relations_and_mutation_outputs():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            db.add_all([
                NumericParent(id=1, amount=Decimal("12.345")),
                NumericChild(id=1, parent_id=1, amount=Decimal("1.250")),
            ])
            await db.commit()
            db.expunge_all()
            statements = []

            def record_query(_connection, _cursor, statement, _parameters, _context, _executemany):
                statements.append(statement)

            event.listen(engine.sync_engine, "before_cursor_execute", record_query)
            result = await schema.execute(
                "{ parents { id amount amount_alias projected_amount children { id amount } } }",
                context_value={"db_session": db},
            )
            assert result.errors is None, result.errors
            assert result.data["parents"] == [{
                "id": 1, "amount": 12.345, "amount_alias": 12.345, "projected_amount": 12.345,
                "children": [{"id": 1, "amount": 1.25}],
            }]
            assert len(statements) == 1

            result = await schema.execute(
                "{ children { amount parent { amount } } }", context_value={"db_session": db},
            )
            assert result.errors is None, result.errors
            assert result.data["children"] == [{"amount": 1.25, "parent": {"amount": 12.345}}]

            result = await schema.execute(
                """mutation Update($payload: NumericParentQLInput!) {
                  merge_parent(payload: $payload) { id amount children { amount } }
                }""",
                variable_values={"payload": {"id": 1, "amount": 60.125}},
                context_value={"db_session": db},
            )
            assert result.errors is None, result.errors
            assert result.data["merge_parent"] == {
                "id": 1, "amount": 60.125, "children": [{"amount": 1.25}],
            }
            persisted = await db.get(NumericParent, 1)
            await db.refresh(persisted)
            assert persisted.amount == Decimal("60.125")
            assert isinstance(persisted.amount, Decimal)
    finally:
        await engine.dispose()
