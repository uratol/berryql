import pytest
from tests.new.schema import schema as berry_schema

@pytest.mark.asyncio
async def test_relation_filters_list(db_session, populated_db):
    # Filter posts relation on User by title_ilike and limit
    q = """
    query { users(name_ilike: \"Alice\") { id name posts(title_ilike: \"GraphQL\", limit: 5) { id title } } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    u = res.data['users'][0]
    titles = [p['title'] for p in u['posts']]
    assert any('GraphQL' in t for t in titles)

@pytest.mark.asyncio
async def test_relation_filters_single(db_session, populated_db):
    # Fetch a post then filter its single author relation (redundant but exercise path)
    q = """
    query { posts(title_ilike: \"First\") { id title author(name_ilike: \"Alice\") { id name } } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data['posts'][0]['author']['name'].startswith('Alice')

@pytest.mark.asyncio
async def test_relation_pagination(db_session, populated_db):
    q = """
    query { users(name_ilike: \"Alice\") { id posts(limit:1, offset:0) { id } } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert len(res.data['users'][0]['posts']) == 1
