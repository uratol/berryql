import pytest
from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_root_where_json(db_session, populated_db):
    # Filter users by JSON where: id gt 1 and name ilike 'a'
    q = '''
    query {
      users(where: "{\\"id\\": {\\"gt\\": 1}, \\\"name\\\": {\\"ilike\\\": \\\"%a%\\\"}}") { id name }
    }
    '''
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    # Should at least exclude id=1
    ids = [u['id'] for u in res.data['users']]
    assert all(i > 1 for i in ids)


@pytest.mark.asyncio
async def test_relation_where_json(db_session, populated_db):
    # On User -> posts relation, filter by created_at gt very old date to include all; then lt now to still include
    q = '''
    query {
      users(name_ilike: "Alice") {
        id
        posts(where: "{\\"created_at\\": {\\"gt\\": \\\"2000-01-01T00:00:00\\\", \\\"lt\\\": \\\"2100-01-01T00:00:00\\\"}}") {
          id
          title
        }
      }
    }
    '''
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    users = res.data['users']
    assert len(users) >= 1
    posts = users[0]['posts']
    assert len(posts) >= 1


@pytest.mark.asyncio
async def test_relation_where_json_excludes(db_session, populated_db):
    # Exclude Alice's posts by a title ilike that doesn't match
    q = '''
    query {
      users(name_ilike: "Alice") {
        id
        posts(where: "{\\"title\\": {\\"ilike\\": \\\"%NoMatch%\\\"}}") { id }
      }
    }
    '''
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data['users'][0]['posts'] == []
