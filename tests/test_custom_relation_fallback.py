"""Computed scalars must survive list loading below a native resolver."""
from typing import Annotated

import pytest
import strawberry
from sqlalchemy import Column, ForeignKey, Integer, case, event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from berryql import (
    BerrySchema, BerryType, FieldPermissions, FieldSet, StrawberryConfig,
    custom, field, relation,
)


Base = declarative_base()


class Parent(Base):
    __tablename__ = "custom_fallback_parents"
    id = Column(Integer, primary_key=True)


class Child(Base):
    __tablename__ = "custom_fallback_children"
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey(Parent.id), nullable=False)
    amount = Column(Integer, nullable=True)


def permissions(type_, info):
    if type_.__name__ == "CustomFallbackChildQL":
        return info.context.get("permissions", FieldPermissions.allow_all())
    return FieldPermissions.allow_all()


def unused_builder(model):
    raise AssertionError("An unselected or denied builder must not run")


berry = BerrySchema(field_permissions=permissions)


@berry.type(model=Parent)
class CustomFallbackParentQL(BerryType):
    id = field()
    children = relation("CustomFallbackChildQL", order_by="id")

    @strawberry.field
    def model_id(self) -> int:
        return self._model.id


@berry.type(model=Child)
class CustomFallbackChildQL(BerryType):
    id = field()
    amount = field()
    parent = relation("CustomFallbackParentQL", single=True, fk_column_name="parent_id")

    @strawberry.field
    async def native_amount(self) -> int | None:
        return self._model.amount
    direct_value = custom(lambda M: M.amount, returns=int)
    selected_value = custom(lambda M: select(M.amount), returns=int)
    scalar_value = custom(
        lambda M: select(M.amount).correlate(M).scalar_subquery(), returns=int,
    )
    case_value = custom(lambda M: case((M.amount > 0, M.amount), else_=0), returns=int)
    context_value = custom(lambda M, info: M.amount + info.context["add"], returns=int)
    unused_value = custom(unused_builder, returns=int)
    dependent_value = custom(unused_builder, returns=int, depends_on=["amount"])


@berry.query()
class Query:
    parents = relation("CustomFallbackParentQL", order_by="id")

    @strawberry.field
    def source(self, info: strawberry.Info) -> Annotated[
        "CustomFallbackParentQL", strawberry.lazy("berryql.registry"),
    ]:
        return info.context["source"]


schema = berry.to_strawberry(strawberry_config=StrawberryConfig(auto_camel_case=True))


@pytest.fixture
async def sample():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            parent = Parent(id=1)
            db.add_all([
                parent, Parent(id=2),
                Child(id=1, parent_id=1, amount=2),
                Child(id=2, parent_id=1, amount=0),
                Child(id=3, parent_id=1, amount=None),
                Child(id=4, parent_id=2, amount=999),
            ])
            await db.flush()
            queries = []

            def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
                if statement.lstrip().upper().startswith("SELECT"):
                    queries.append(statement)

            event.listen(engine.sync_engine, "before_cursor_execute", capture)
            yield {"db_session": db, "source": berry.from_model("CustomFallbackParentQL", parent),
                   "add": 3}, queries
    finally:
        await engine.dispose()


async def test_fallback_projects_sql_custom_scalars_without_extra_queries(sample):
    context, queries = sample
    result = await schema.execute("""
        { source { children {
          id directValue selectedValue scalarValue caseValue contextValue
        } } }
    """, context_value=context)
    assert result.errors is None, result.errors
    assert result.data["source"]["children"] == [
        {"id": 1, "directValue": 2, "selectedValue": 2, "scalarValue": 2,
         "caseValue": 2, "contextValue": 5},
        {"id": 2, "directValue": 0, "selectedValue": 0, "scalarValue": 0,
         "caseValue": 0, "contextValue": 3},
        {"id": 3, "directValue": None, "selectedValue": None, "scalarValue": None,
         "caseValue": 0, "contextValue": None},
    ]
    assert len(queries) == 1


async def test_fallback_custom_names_aliases_fragments_and_directives(sample):
    context, queries = sample
    result = await schema.execute("""
      query($skip: Boolean = true) {
        source { children {
          ...Values
          first: directValue second: directValue
          unusedValue @skip(if: $skip)
        } }
      }
      fragment Values on CustomFallbackChildQL { selectedValue }
    """, context_value=context)
    assert result.errors is None, result.errors
    assert result.data["source"]["children"] == [
        {"selectedValue": value, "first": value, "second": value}
        for value in [2, 0, None]
    ]
    assert len(queries) == 1


async def test_fallback_does_not_project_denied_custom_fields_or_dependencies(sample):
    context, queries = sample
    context["permissions"] = FieldPermissions(
        select=FieldSet.only("id", "direct_value", "dependent_value"),
    )
    result = await schema.execute("""
      { source { children { id directValue unusedValue dependentValue } } }
    """, context_value=context)
    assert result.errors is None, result.errors
    assert result.data["source"]["children"] == [
        {"id": i, "directValue": value, "unusedValue": None, "dependentValue": None}
        for i, value in [(1, 2), (2, 0), (3, None)]
    ]
    assert len(queries) == 1
    assert "dependent_value" not in queries[0]
    assert "unused_value" not in queries[0]


@pytest.mark.parametrize("root", ["source", "parents(limit: 1)"])
async def test_nested_native_field_receives_model_without_selected_columns(sample, root):
    context, queries = sample
    result = await schema.execute(
        "{ " + root + " { children { nativeAmount } } }", context_value=context,
    )
    assert result.errors is None, result.errors
    parent = result.data["source"] if root == "source" else result.data["parents"][0]
    assert parent["children"] == [{"nativeAmount": value} for value in [2, 0, None]]
    assert len(queries) == (1 if root == "source" else 2)


async def test_native_field_below_single_and_list_relations_receives_model(sample):
    context, _ = sample
    result = await schema.execute("""
        { parents(limit: 1) { children { parent { modelId } } } }
    """, context_value=context)
    assert result.errors is None, result.errors
    assert result.data["parents"] == [{"children": [{"parent": {"modelId": 1}}] * 3}]


async def test_nested_native_field_alias_fragment_and_pagination(sample):
    context, _ = sample
    result = await schema.execute("""
        { parents(limit: 1) { children(limit: 1, offset: 1) { ...Native } } }
        fragment Native on CustomFallbackChildQL { value: nativeAmount directValue }
    """, context_value=context)
    assert result.errors is None, result.errors
    assert result.data["parents"] == [{"children": [{"value": 0, "directValue": 0}]}]


async def test_skipped_native_field_keeps_relation_pushdown(sample):
    context, queries = sample
    result = await schema.execute("""
        { parents(limit: 1) { children { id nativeAmount @skip(if: true) } } }
    """, context_value=context)
    assert result.errors is None, result.errors
    assert result.data["parents"] == [{"children": [{"id": i} for i in [1, 2, 3]]}]
    assert len(queries) == 1


async def test_denied_native_field_is_not_called(sample):
    context, _ = sample
    context["permissions"] = FieldPermissions(select=FieldSet.only("id", "direct_value"))
    result = await schema.execute("""
        { parents(limit: 1) { children { id nativeAmount } } }
    """, context_value=context)
    assert result.errors is None, result.errors
    assert result.data["parents"] == [{"children": [
        {"nativeAmount": None, "id": value} for value in [1, 2, 3]
    ]}]
