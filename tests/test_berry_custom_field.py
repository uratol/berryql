import pytest
from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema

@pytest.mark.asyncio
async def test_post_comment_text_len_custom_field(db_session, populated_db):
    # Query first 2 posts including custom aggregated length field
    query = """
    query {
      posts(limit: 2) { id title comment_text_len }
    }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    assert len(posts) == 2
    # Values should be integers (0 when no comments)
    lengths = [p["comment_text_len"] for p in posts]
    assert all(isinstance(v, int) for v in lengths)
    # At least one of the first two seeded posts has comments so sum > 0
    assert any(v > 0 for v in lengths)
