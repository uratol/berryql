import pytest
from tests.fixtures import *  # noqa: F401,F403

# Build a tiny schema that uses aggregate(..., op='count') directly and
# compare it with the count(...) convenience wrapper to ensure parity.
from berryql import BerrySchema, BerryType, field, relation, aggregate, count
from tests.models import Post, PostComment  # type: ignore
from sqlalchemy import select, func


berry_schema = BerrySchema()


@berry_schema.type(model=PostComment)
class PostCommentQL(BerryType):
    id = field()
    content = field()
    post_id = field()


@berry_schema.type(model=Post)
class PostQL(BerryType):
    id = field()
    title = field()
    post_comments = relation('PostCommentQL', order_by='id')
    # Two equivalent aggregates using different helpers
    post_comments_count_via_count = count('post_comments')
    post_comments_count_via_aggregate = aggregate('post_comments', op='count')


@berry_schema.query()
class Query:
    posts = relation('PostQL', order_by='id', order_dir='asc')


schema = berry_schema.to_strawberry()


@pytest.mark.asyncio
async def test_aggregate_function_count_equivalence(db_session, populated_db):
    # Query both aggregate fields and ensure they are equal for every row
    query = """
    query {
      posts(limit: 5) { id post_comments_count_via_count post_comments_count_via_aggregate }
    }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    assert len(posts) > 0
    for p in posts:
        a = p["post_comments_count_via_count"]
        b = p["post_comments_count_via_aggregate"]
        assert isinstance(a, int) and isinstance(b, int)
        assert a == b
