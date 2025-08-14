import pytest
from tests.fixtures import populated_db  # noqa: F401
from tests.schema import schema

@pytest.mark.asyncio
async def test_user_regular_strawberry_field(db_session, populated_db):
    query = """
    query { users { id name name_upper } }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    users = res.data["users"]
    assert len(users) == 4
    # name_upper should be uppercase of name
    for u in users:
        if u["name"] is not None:
            assert u["name_upper"] == u["name"].upper()

@pytest.mark.asyncio
async def test_post_comment_regular_strawberry_field(db_session, populated_db):
    # Fetch users with posts and their comments to ensure content_preview resolves
    query = """
    query {
        posts { id title post_comments { id content content_preview } }
    }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    assert isinstance(posts, list)
    # For each comment, content_preview should be a shortened string (or equal when short)
    for p in posts:
        for c in p.get("post_comments", []) or []:
            cont = c.get("content")
            prev = c.get("content_preview")
            if cont is None:
                assert prev is None
            else:
                s = cont if len(cont) <= 10 else cont[:10] + "..."
                assert prev == s
