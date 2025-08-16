import os
import pytest

from tests.schema import schema

@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv('BERRYQL_TEST_DATABASE_URL', '').startswith('postgresql') is False, reason="requires Postgres URL to test BYTEA[]")
async def test_post_binary_array_roundtrip_postgres(db_session, sample_posts):
    # Query posts with binary_blobs field; expect base64 strings for non-null values
    q = """
    query {
      posts(order_by: "id") { id binary_blobs }
    }
    """
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data
    assert data and 'posts' in data
    posts = data['posts']
    # We inserted 5 posts; check first few expectations
    # Post 1: ["YQ==", "Yg==", "Yw=="] for a, b, c
    assert posts[0]['binary_blobs'] == ["YQ==", "Yg==", "Yw=="]
    # Post 2: ["eA==", "eQ=="] for x, y
    assert posts[1]['binary_blobs'] == ["eA==", "eQ=="]
    # Post 3: None
    assert posts[2]['binary_blobs'] is None
    # Post 4: ["AAEC"] for \x00\x01\x02
    assert posts[3]['binary_blobs'] == ["AAEC"]
    # Post 5: empty list
    assert posts[4]['binary_blobs'] == []
