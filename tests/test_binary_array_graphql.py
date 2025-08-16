import json
import os
import pytest

from tests.schema import schema


def _normalize_blobs(v):
    # Accept list[str] or JSON string representing a list[str]; allow None or []
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    # Fallback: wrap unknown scalars
    return [str(v)]


@pytest.mark.asyncio
async def test_posts_binary_blob_graphql(db_session, sample_posts):
    q = """
    query { posts(order_by: "id") { id binary_blob } }
    """
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    # Base64 values seeded in fixtures
    assert posts[0]["binary_blob"] == "YQ=="  # b"a"
    assert posts[1]["binary_blob"] == "eA=="  # b"x"
    assert posts[2]["binary_blob"] is None
    assert posts[3]["binary_blob"] == "AAEC"  # b"\x00\x01\x02"
    assert posts[4]["binary_blob"] is None
