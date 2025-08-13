import pytest
from tests.common.fixtures import *  # noqa: F401,F403
from tests.new.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_deep_nesting_with_filters_and_ordering(db_session, populated_db):
    # users -> posts(order by id desc, limit 2) -> post_comments(where rate gt 1, order asc)
    q = '''
    query {
      users(limit: 2) {
        id
        posts(limit: 2, order_by: "id", order_dir: desc) {
          id
          post_comments(where: """{"rate": {"gt": 1}}""", order_by: "id", order_dir: asc) {
            id
            rate
          }
        }
      }
    }
    '''
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    data = res.data
    assert 'users' in data
    for u in data['users']:
        for p in u.get('posts', []):
            rates = [c['rate'] for c in p.get('post_comments', [])]
            assert all(r is None or r > 1 for r in rates)
            ids = [c['id'] for c in p.get('post_comments', []) if c.get('id') is not None]
            assert ids == sorted(ids)


@pytest.mark.asyncio
async def test_mixed_pushdown_and_callable_default_where(db_session, populated_db):
    # posts_have_comments uses callable where => skip pushdown at that branch
    q = '''
    query {
      users(limit: 1) {
        id
        posts_have_comments(limit: 3) {
          id
          post_comments(limit: 1, order_by: "id") { id }
        }
      }
    }
    '''
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    users = res.data['users']
    assert len(users) >= 1
    for p in users[0].get('posts_have_comments', []):
        assert 'id' in p
        pcs = p.get('post_comments', [])
        assert isinstance(pcs, list)
