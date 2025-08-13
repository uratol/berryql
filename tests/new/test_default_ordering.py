import pytest
from tests.new.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_default_ordering_users(db_session, populated_db):
    # No explicit order args; should use default ordering defined on type
    q = """
    query { users { id created_at } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    ids = [u['id'] for u in res.data['users']]
    # Expect ascending by id if we set default that way
    assert ids == sorted(ids)


@pytest.mark.asyncio
async def test_default_ordering_comments_relation(db_session, populated_db):
    # Fetch a post and check its post_comments list uses default ordering
    q = """
    query {
      posts(limit: 1) {
        id
        post_comments { id created_at }
      }
    }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    comments = res.data['posts'][0]['post_comments']
    # Expect ascending by id if we set default that way
    ids = [c['id'] for c in comments]
    assert ids == sorted(ids)

@pytest.mark.asyncio
async def test_query_overrides_default_order_users(db_session, populated_db):
    # Default is id asc; query should override to desc
    q = """
    query { users(order_by: "id", order_dir: desc) { id } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    ids = [u['id'] for u in res.data['users']]
    assert ids == sorted(ids, reverse=True)

@pytest.mark.asyncio
async def test_query_overrides_default_order_comments_relation(db_session, populated_db):
    # Default is id asc on relation; query should override to desc
    q = """
    query {
      posts(limit: 1) {
        id
        post_comments(order_by: "id", order_dir: desc) { id }
      }
    }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    ids = [c['id'] for c in res.data['posts'][0]['post_comments']]
    assert ids == sorted(ids, reverse=True)
