"""Conditional fields must be omitted from SQL, not just GraphQL output."""

from types import SimpleNamespace
from typing import Annotated

import pytest
import strawberry
from graphql import parse
from sqlalchemy import event
from strawberry.types.nodes import convert_selections

from berryql import BerrySchema, BerryType, field, relation
from berryql.core.selection import RelationSelectionExtractor, RootSelectionExtractor
from tests.models import Post, User
from tests.schema import schema

fallback_berry = BerrySchema()


@fallback_berry.type(model=Post)
class ConditionalFallbackPostQL(BerryType):
    id = field()
    title = field()
    content = field()


@fallback_berry.type(model=User)
class ConditionalFallbackUserQL(BerryType):
    id = field()
    posts = relation("ConditionalFallbackPostQL", order_by="id")


@fallback_berry.query()
class ConditionalFallbackQuery:
    @strawberry.field
    def raw_user(self, info: strawberry.Info) -> Annotated[
        "ConditionalFallbackUserQL",
        strawberry.lazy("berryql.registry"),
    ]:
        return info.context["source"]


fallback_schema = fallback_berry.to_strawberry()


async def test_native_resolver_relation_fallback_honors_directives(
    db_session,
    populated_db,
    select_statements,
):
    source = fallback_berry.from_model("ConditionalFallbackUserQL", populated_db["users"][0])
    result = await fallback_schema.execute(
        """
      query($details: Boolean = false) {
        raw_user {
          posts {
            id
            content @include(if: $details)
            ...Title @skip(if: false)
          }
        }
      }
      fragment Title on ConditionalFallbackPostQL { title }
    """,
        context_value={"db_session": db_session, "source": source},
    )
    assert result.errors is None, result.errors
    posts = result.data["raw_user"]["posts"]
    assert posts and set(posts[0]) == {"id", "title"}
    sql = " ".join(select_statements)
    assert "posts.title" in sql
    assert "posts.content" not in sql


@pytest.fixture
def select_statements(db_session):
    statements = []
    engine = db_session.get_bind()

    def capture(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(" ".join(statement.lower().split()))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", capture)


@pytest.mark.parametrize("directive", ["@include(if: false)", "@skip(if: true)"])
async def test_excluded_root_fields_do_not_enter_sql(db_session, populated_db, select_statements, directive):
    result = await schema.execute(
        "{ posts(limit: 1) { id content "
        + directive
        + " comment_text_len "
        + directive
        + " post_comments_agg "
        + directive
        + " post_comments_agg_obj "
        + directive
        + " { comments_count } } }",
        context_value={"db_session": db_session},
    )
    assert result.errors is None, result.errors
    assert result.data["posts"] and set(result.data["posts"][0]) == {"id"}
    assert select_statements
    sql = " ".join(select_statements)
    assert "posts.content" not in sql
    assert "post_comments" not in sql


@pytest.mark.parametrize("selection", ["email @include(if: false)", "__typename"])
async def test_empty_root_projection_only_selects_identity(db_session, sample_users, select_statements, selection):
    result = await schema.execute(
        "{ users(limit: 1) { " + selection + " } }",
        context_value={"db_session": db_session},
    )
    assert result.errors is None, result.errors
    assert result.data["users"]
    sql = " ".join(select_statements)
    assert "users.id" in sql
    assert "users.email" not in sql
    assert "users.name" not in sql


async def test_empty_nested_projection_only_selects_identity(db_session, populated_db, select_statements):
    result = await schema.execute(
        "{ users(limit: 1) { id posts { content @skip(if: true) } } }",
        context_value={"db_session": db_session},
    )
    assert result.errors is None, result.errors
    assert result.data["users"][0]["posts"]
    assert result.data["users"][0]["posts"][0] == {}
    assert "posts.content" not in " ".join(select_statements)


@pytest.mark.parametrize(
    "selection",
    [
        "posts @include(if: false) { id }",
        "posts @skip(if: true) { id }",
        "...Hidden @skip(if: true)",
        "... on UserQL @include(if: false) { posts { id } }",
        "... @include(if: false) { posts { id } }",
    ],
)
async def test_excluded_relations_do_not_enter_sql(db_session, populated_db, select_statements, selection):
    query = "{ users(limit: 1) { id " + selection + " } }"
    if "Hidden" in selection:
        query += " fragment Hidden on UserQL { posts { id } }"
    result = await schema.execute(query, context_value={"db_session": db_session})
    assert result.errors is None, result.errors
    assert result.data["users"] and set(result.data["users"][0]) == {"id"}
    assert select_statements
    assert "posts" not in " ".join(select_statements)


async def test_merged_root_fields_and_aliases_keep_included_selections(db_session, sample_users, select_statements):
    result = await schema.execute(
        """
      {
        people: users(limit: 1) { id }
        people: users(limit: 1) { email @include(if: true) @skip(if: false) }
        people: users(limit: 1) { name @include(if: true) @skip(if: true) }
      }
    """,
        context_value={"db_session": db_session},
    )
    assert result.errors is None, result.errors
    assert result.data["people"] and set(result.data["people"][0]) == {"id", "email"}
    sql = " ".join(select_statements)
    assert "users.email" in sql
    assert "users.name" not in sql


async def test_directive_defaults_control_nested_fragments_and_relations(db_session, populated_db, select_statements):
    query = """
      query($details: Boolean = false, $skipComments: Boolean = true) {
        users(limit: 1) {
          id
          ...Private @include(if: $details)
          posts(limit: 1) {
            id
            ... on PostQL @include(if: $details) { content }
            post_comments @skip(if: $skipComments) { id content }
          }
        }
      }
      fragment Private on UserQL { email }
    """
    result = await schema.execute(query, context_value={"db_session": db_session})
    assert result.errors is None, result.errors
    users = result.data["users"]
    assert users and set(users[0]) == {"id", "posts"}
    assert users[0]["posts"] and set(users[0]["posts"][0]) == {"id"}
    sql = " ".join(select_statements)
    assert "users.email" not in sql
    assert "posts.content" not in sql
    assert "post_comments" not in sql

    select_statements.clear()
    result = await schema.execute(
        query,
        variable_values={"details": True, "skipComments": False},
        context_value={"db_session": db_session},
    )
    assert result.errors is None, result.errors
    assert "email" in result.data["users"][0]
    assert "content" in result.data["users"][0]["posts"][0]
    assert "post_comments" in result.data["users"][0]["posts"][0]
    sql = " ".join(select_statements)
    assert "users.email" in sql
    assert "post_comments" in sql


@pytest.mark.parametrize("selected_fields", [False, True])
def test_extractors_respect_conditional_fragments_and_merge_field_nodes(selected_fields):
    def field(kind="scalar", **meta):
        return SimpleNamespace(kind=kind, meta=meta)

    child = SimpleNamespace(__berry_fields__={"id": field(), "content": field()})
    parent = SimpleNamespace(
        __berry_fields__={
            "id": field(),
            "payload": field(),
            "snapshot": field("custom_object"),
            "children": field("relation", target="Child"),
        }
    )
    registry = SimpleNamespace(types={"Child": child}, _auto_camel_case=False)
    document = parse(
        """
      query($details: Boolean!, $skip: Boolean!) {
        items { id ...Details @include(if: $details) children { id } }
        items { children { content @skip(if: $skip) } }
      }
      fragment Details on Item {
        payload
            ... on Item @skip(if: false) { snapshot { title } }
        children { content }
      }
    """
    )
    raw = SimpleNamespace(
        field_nodes=document.definitions[0].selection_set.selections,
        fragments={"Details": document.definitions[1]},
        variable_values={"details": False, "skip": True},
    )
    info = SimpleNamespace(_raw_info=raw)
    if selected_fields:
        info.selected_fields = convert_selections(raw, raw.field_nodes)
    root = RootSelectionExtractor(registry).extract(info, "items", parent)
    relations = RelationSelectionExtractor(registry).extract(info, "items", parent)
    assert root["scalars"] == {"id"}
    assert not root["custom_object"]
    assert relations["children"]["fields"] == ["id"]

    raw.variable_values.update(details=True, skip=False)
    if selected_fields:
        info.selected_fields = convert_selections(raw, raw.field_nodes)
    root = RootSelectionExtractor(registry).extract(info, "items", parent)
    relations = RelationSelectionExtractor(registry).extract(info, "items", parent)
    assert root["scalars"] == {"id", "payload"}
    assert root["custom_object"] == {"snapshot"}
    assert set(relations["children"]["fields"]) == {"id", "content"}
