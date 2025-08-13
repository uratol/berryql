import pytest
from tests.common.fixtures import *  # noqa: F401,F403
from tests.new.schema import schema


@pytest.mark.asyncio
async def test_single_object_relations_and_count(db_session, populated_db):
    query = """
    query {
      posts(limit: 2) {
        id
        author { id name }
        post_comments(limit: 2) { id }
        post_comments_agg
      }
      users(limit: 1) { id name posts(limit: 2) { id title } post_agg post_agg_obj { count } }
    }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data
    assert "posts" in data
    assert len(data["posts"]) == 2
    first_post = data["posts"][0]
    assert "author" in first_post and isinstance(first_post["author"], dict)
    assert "post_comments_agg" in first_post
    assert "users" in data
    assert len(data["users"]) == 1
    assert "post_agg" in data["users"][0]
    assert "post_agg_obj" in data["users"][0]
    # post_agg_obj should be either None or have count key
    agg_obj = data["users"][0]["post_agg_obj"]
    if agg_obj is not None:
        assert set(agg_obj.keys()) == {"count"}
