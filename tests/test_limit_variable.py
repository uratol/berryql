import pytest
from tests.schema import schema

@pytest.mark.asyncio
async def test_limit_variable(db_session, populated_db):
    query = """
    query MyQuery($limit: Int) {
      userById(id: 1) {
        posts(limit: $limit){id}
      }
    }
    """
    variable_values = {"limit": 1}
    result = await schema.execute(query, variable_values=variable_values, context_value={'db_session': db_session})
    assert result.errors is None, result.errors
    posts = result.data['userById']['posts']
    # User 1 has 2 posts in populated_db. limit 1 should return 1.
    # If variable fails, it falls back to no limit (default), so 2.
    assert len(posts) == 1

@pytest.mark.asyncio
async def test_orderby_variable(db_session, populated_db):
    query = """
    query MyQuery($orderBy: String) {
      userById(id: 1) {
        posts(order_by: $orderBy){id}
      }
    }
    """
    variable_values = {"orderBy": "id"}
    result = await schema.execute(query, variable_values=variable_values, context_value={'db_session': db_session})
    assert result.errors is None, result.errors

@pytest.mark.asyncio
async def test_orderdir_variable(db_session, populated_db):
    query = """
    query MyQuery($orderDir: Direction) {
      userById(id: 1) {
        posts(order_by: "id", order_dir: $orderDir){id}
      }
    }
    """
    variable_values = {"orderDir": "desc"}
    result = await schema.execute(query, variable_values=variable_values, context_value={'db_session': db_session})
    assert result.errors is None, result.errors
