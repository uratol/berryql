import pytest
from tests.schema import schema as berry_strawberry_schema

@pytest.mark.asyncio
async def test_where_not_in(db_session, populated_db):
    # Filter users by where: {name: {not_in: [...]}}
    q = """
    query($w: String) {
      users(where: $w) { id name }
    }
    """
    w = '{"name": {"not_in": ["Alice Johnson", "Bob Smith"]}}'
    res = await berry_strawberry_schema.execute(q, variable_values={"w": w}, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data is not None
    names = [u['name'] for u in res.data['users']]
    assert "Alice Johnson" not in names
    assert "Bob Smith" not in names
    assert "Charlie Brown" in names

@pytest.mark.asyncio
async def test_where_not_between(db_session, populated_db):
    # Filter users by where: {created_at: {not_between: [...]}}
    # We need to pick a range that excludes everything
    q = """
    query($w: String) {
      users(where: $w) { id name }
    }
    """
    w = '{"created_at": {"not_between": ["1900-01-01T00:00:00", "2200-01-01T00:00:00"]}}'
    res = await berry_strawberry_schema.execute(q, variable_values={"w": w}, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data is not None
    assert len(res.data['users']) == 0
