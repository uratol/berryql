import pytest

from tests.schema import schema


@pytest.mark.asyncio
async def test_nested_domain_queries(db_session, populated_db):
    q = """
    query {
      groupDomain {
        userDomain { users(limit: 1) { id name } }
        blogDomain { posts(limit: 1) { id title } }
      }
    }
    """
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data
    assert "groupDomain" in data
    gd = data["groupDomain"]
    assert "userDomain" in gd and "blogDomain" in gd
    users = gd["userDomain"]["users"]
    posts = gd["blogDomain"]["posts"]
    assert isinstance(users, list) and isinstance(posts, list)
    if users:
        assert set(["id", "name"]).issubset(users[0].keys())
    if posts:
        assert set(["id", "title"]).issubset(posts[0].keys())
