from datetime import datetime, timezone

import pytest

from berryql import BerrySchema, BerryType, PaginationConfig, field, relation
from tests.models import Post, User


bounded_berry_schema = BerrySchema(pagination=PaginationConfig(default_limit=2, max_limit=3))


@bounded_berry_schema.type(model=User)
class UserQL(BerryType):
    id = field()
    name = field()
    posts = relation("PostQL", order_by="id")


@bounded_berry_schema.type(model=Post)
class PostQL(BerryType):
    id = field()
    title = field()
    author_id = field()


@bounded_berry_schema.query()
class Query:
    users = relation("UserQL", order_by="id")
    user_by_id = relation(
        "UserQL",
        single=True,
        arguments={"id": {"column": "id", "op": "eq"}},
    )


bounded_schema = bounded_berry_schema.to_strawberry()


@pytest.mark.asyncio
async def test_pagination_config_applies_default_max_and_offset_to_roots(db_session, sample_users):
    context = {"db_session": db_session}

    default_res = await bounded_schema.execute("query { users { id } }", context_value=context)
    assert default_res.errors is None, default_res.errors
    assert len(default_res.data["users"]) == 2

    max_res = await bounded_schema.execute("query { users(limit: 99) { id } }", context_value=context)
    assert max_res.errors is None, max_res.errors
    assert len(max_res.data["users"]) == 3

    next_res = await bounded_schema.execute("query { users(limit: 2, offset: 2) { id } }", context_value=context)
    assert next_res.errors is None, next_res.errors
    assert [row["id"] for row in next_res.data["users"]] == [user.id for user in sample_users[2:4]]


@pytest.mark.asyncio
async def test_pagination_config_applies_to_nested_relations(db_session, sample_users):
    author = sample_users[0]
    posts = [
        Post(
            title=f"Pagination Config Post {index}",
            content="body",
            author_id=author.id,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        for index in range(5)
    ]
    db_session.add_all(posts)
    await db_session.flush()
    await db_session.commit()

    context = {"db_session": db_session}
    default_res = await bounded_schema.execute(
        "query($id: Int!) { user_by_id(id: $id) { id posts { id } } }",
        variable_values={"id": author.id},
        context_value=context,
    )
    assert default_res.errors is None, default_res.errors
    assert len(default_res.data["user_by_id"]["posts"]) == 2

    max_res = await bounded_schema.execute(
        "query($id: Int!) { user_by_id(id: $id) { id posts(limit: 99) { id } } }",
        variable_values={"id": author.id},
        context_value=context,
    )
    assert max_res.errors is None, max_res.errors
    assert len(max_res.data["user_by_id"]["posts"]) == 3

    next_res = await bounded_schema.execute(
        "query($id: Int!) { user_by_id(id: $id) { id posts(limit: 2, offset: 2) { id } } }",
        variable_values={"id": author.id},
        context_value=context,
    )
    assert next_res.errors is None, next_res.errors
    first_page_ids = [row["id"] for row in default_res.data["user_by_id"]["posts"]]
    next_page_ids = [row["id"] for row in next_res.data["user_by_id"]["posts"]]
    assert len(next_page_ids) == 2
    assert not set(first_page_ids).intersection(next_page_ids)
