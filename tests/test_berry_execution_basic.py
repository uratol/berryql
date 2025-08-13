import pytest
from tests.schema import schema as berry_schema

@pytest.mark.asyncio
async def test_berry_root_users(db_session, populated_db):
    query = """
    query { users { id name } }
    """
    result = await berry_schema.execute(query, context_value={'db_session': db_session})
    assert result.errors is None, result.errors
    assert 'users' in result.data
    assert len(result.data['users']) == 4
