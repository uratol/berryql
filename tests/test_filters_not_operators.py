import pytest
from tests.schema import schema as berry_strawberry_schema

@pytest.mark.asyncio
async def test_user_filters_not_like(db_session, populated_db):
    # Filter users by name_not_like
    # Assuming populated_db has users like "Alice", "Bob", "Charlie", "Dave"
    q = """
    query { users(name_not_like: "A%") { id name } }
    """
    res = await berry_strawberry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data is not None
    names = [u['name'] for u in res.data['users']]
    assert "Alice Johnson" not in names
    assert "Bob Smith" in names

@pytest.mark.asyncio
async def test_user_filters_not_ilike(db_session, populated_db):
    # Filter users by name_not_ilike
    q = """
    query { users(name_not_ilike: "a%") { id name } }
    """
    res = await berry_strawberry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data is not None
    names = [u['name'] for u in res.data['users']]
    # "Alice" starts with "A", ilike "a%" matches "Alice"
    assert "Alice Johnson" not in names
    assert "Bob Smith" in names

@pytest.mark.asyncio
async def test_user_filters_not_in(db_session, populated_db):
    # Filter users by name_not_in
    q = """
    query { users(name_not_in: ["Alice Johnson", "Bob Smith"]) { id name } }
    """
    res = await berry_strawberry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data is not None
    names = [u['name'] for u in res.data['users']]
    assert "Alice Johnson" not in names
    assert "Bob Smith" not in names
    assert "Charlie Brown" in names

@pytest.mark.asyncio
async def test_user_filters_not_between(db_session, populated_db):
    # Filter users by created_at_not_between
    # We need to pick a range that includes some users but not others, or excludes all if we pick a wide range
    # Let's try to exclude everything by picking a very wide range
    q = """
    query { users(created_at_not_between: ["1900-01-01T00:00:00", "2200-01-01T00:00:00"]) { id name } }
    """
    res = await berry_strawberry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data is not None
    assert len(res.data['users']) == 0

    # Now try a range that shouldn't match anything (future), so all users should be returned
    q2 = """
    query { users(created_at_not_between: ["2200-01-01T00:00:00", "2300-01-01T00:00:00"]) { id name } }
    """
    res2 = await berry_strawberry_schema.execute(q2, context_value={'db_session': db_session})
    assert res2.errors is None, res2.errors
    assert len(res2.data['users']) > 0
