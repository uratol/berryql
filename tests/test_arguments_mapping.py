import pytest
from tests.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_userById_argument(db_session, populated_db):
  q = """
  query { userById(id: 1) { id name } }
  """
  res = await berry_schema.execute(q, context_value={'db_session': db_session})
  assert res.errors is None, res.errors
  assert res.data['userById'] is not None
  assert res.data['userById']['id'] == 1


@pytest.mark.asyncio
async def test_relation_arguments_pure_lambda(db_session, populated_db):
    # Verify pure lambda args: created_at_gt/lt and title_ilike are applied
    q = """
    query {
      users(name_ilike: "Alice") {
        id
        posts(title_ilike: "First", created_at_gt: "2000-01-01T00:00:00", created_at_lt: "2100-01-01T00:00:00") { id title }
      }
    }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert len(res.data['users']) >= 1
    titles = [p['title'] for p in res.data['users'][0]['posts']]
    assert any('First' in t for t in titles)


@pytest.mark.asyncio
async def test_root_posts_title_ilike_argument_lambda(db_session, populated_db):
    q = """
    query { posts(title_ilike: "GraphQL") { id title } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    titles = [p['title'] for p in res.data['posts']]
    assert any('GraphQL' in t for t in titles)

@pytest.mark.asyncio
async def test_single_relation_lambda_argument(db_session, populated_db):
  # Ensure arguments appear on single relations and are applied (PostQL.author)
  q = """
  query { posts(title_ilike: "First") { id title author(name_ilike: "Alice") { id name } } }
  """
  res = await berry_schema.execute(q, context_value={'db_session': db_session})
  assert res.errors is None, res.errors
  assert res.data['posts'][0]['author']['name'].startswith('Alice')
