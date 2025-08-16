import os
import pytest

from tests.schema import schema

@pytest.mark.asyncio
@pytest.mark.skipif(False, reason="converted to single binary blob; keep test active across DBs")
async def test_post_binary_array_roundtrip_postgres(db_session, sample_posts):
    q = """
    query {
      posts(order_by: "id") { id binary_blob }
    }
    """
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data
    assert data and 'posts' in data
    posts = data['posts']
    assert posts[0]['binary_blob'] == 'YQ=='
    assert posts[1]['binary_blob'] == 'eA=='
    assert posts[2]['binary_blob'] is None
    assert posts[3]['binary_blob'] == 'AAEC'
    assert posts[4]['binary_blob'] is None
