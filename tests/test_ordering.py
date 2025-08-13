import pytest
from tests.schema import schema as berry_schema

@pytest.mark.asyncio
async def test_root_ordering_single(db_session, populated_db):
    q = """
    query { posts(order_by: "created_at", order_dir: desc, limit:3) { id created_at } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    data = res.data['posts']
    # created_at desc => first timestamp >= next
    created = [p['created_at'] for p in data]
    assert created == sorted(created, reverse=True)

@pytest.mark.asyncio
async def test_root_ordering_default_dir(db_session, populated_db):
    q = """
    query { posts(order_by: "id", limit:5) { id } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None
    ids = [p['id'] for p in res.data['posts']]
    assert ids == sorted(ids)

@pytest.mark.asyncio
async def test_relation_ordering_single(db_session, populated_db):
    q = """
    query { users(name_ilike: "Alice") { id posts(order_by: "created_at", order_dir: desc) { id created_at } } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    posts = res.data['users'][0]['posts']
    created = [p['created_at'] for p in posts]
    assert created == sorted(created, reverse=True)

@pytest.mark.asyncio
async def test_relation_ordering_multi(db_session, populated_db):
    q = """
    query {
      users(name_ilike: "Alice") {
        id
        posts(order_multi: ["created_at:desc", "id:asc"]) { id created_at }
      }
    }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    posts = res.data['users'][0]['posts']
    # Validate primary sort (created_at desc)
    created = [p['created_at'] for p in posts]
    assert created == sorted(created, reverse=True)
    # Where created_at ties (synthetic possibility), ids should ascend; create mapping to group
    from collections import defaultdict
    groups = defaultdict(list)
    for p in posts:
        groups[p['created_at']].append(p['id'])
    for ids in groups.values():
        assert ids == sorted(ids)

@pytest.mark.asyncio
async def test_relation_ordering_with_pagination(db_session, populated_db):
    q = """
    query {
      users(name_ilike: "Alice") { id posts(order_by: "created_at", order_dir: desc, limit:1) { id created_at } }
    }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    posts = res.data['users'][0]['posts']
    assert len(posts) == 1

@pytest.mark.asyncio
async def test_invalid_order_field_ignored(db_session, populated_db):
    q = """
    query { posts(order_by: "__nope__", limit:3) { id title } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is not None
    assert any('Invalid order_by' in str(e) for e in res.errors)
