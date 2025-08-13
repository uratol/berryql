import pytest
from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema

@pytest.mark.asyncio
async def test_post_comments_agg_obj_custom_object(db_session, populated_db):
    query = """
    query {
      posts(limit: 2) { id post_comments_agg_obj { min_created_at comments_count } }
    }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    assert len(posts) == 2
    agg = posts[0]["post_comments_agg_obj"]
    assert agg is None or set(agg.keys()) == {"min_created_at", "comments_count"}
    if agg:
        assert isinstance(agg["comments_count"], int)
