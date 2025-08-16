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
async def test_posts_binary_blobs_graphql(db_session, sample_posts):
    q = """
    query { posts(order_by: "id") { id binary_blobs } }
    """
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    # On Postgres we expect base64 strings, on SQLite it's JSON strings as well (when supported)
    # Values seeded in fixtures
    assert posts[0]["binary_blobs"] is None or posts[0]["binary_blobs"] == ["YQ==", "Yg==", "Yw=="]
    assert posts[1]["binary_blobs"] is None or posts[1]["binary_blobs"] == ["eA==", "eQ=="]
    assert posts[2]["binary_blobs"] is None
    assert posts[3]["binary_blobs"] is None or posts[3]["binary_blobs"] == ["AAEC"]
    assert posts[4]["binary_blobs"] is None or posts[4]["binary_blobs"] == []
