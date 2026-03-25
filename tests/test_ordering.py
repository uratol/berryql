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
    query { users(name_ilike: "Alice") { id posts { id created_at } } }
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
async def test_relation_ordering_by_single_relation_model_field(db_session, populated_db):
        alice = populated_db['users'][0]
        bob = populated_db['users'][1]
        charlie = populated_db['users'][2]
        first_post = populated_db['posts'][0]
        second_post = populated_db['posts'][1]

        first_post.reviewer_id = bob.id
        second_post.reviewer_id = charlie.id
        await db_session.commit()

        q = """
        query {
            users(name_ilike: "Alice") {
                id
                posts(order_by: "reviewer.id", order_dir: desc) {
                    id
                    reviewer { id }
                }
            }
        }
        """
        res = await berry_schema.execute(q, context_value={'db_session': db_session})
        assert res.errors is None, res.errors
        posts = res.data['users'][0]['posts']
        reviewer_ids = [post['reviewer']['id'] for post in posts if post['reviewer'] is not None]
        assert reviewer_ids == [charlie.id, bob.id]


@pytest.mark.asyncio
async def test_relation_ordering_by_single_relation_scalar(db_session, populated_db):
        q = """
        query {
            posts(limit: 1, order_by: "id") {
                id
                post_comments(order_by: "author.id", order_dir: desc) {
                    id
                    author { id }
                }
            }
        }
        """
        res = await berry_schema.execute(q, context_value={'db_session': db_session})
        assert res.errors is None, res.errors
        comments = res.data['posts'][0]['post_comments']
        author_ids = [comment['author']['id'] for comment in comments]
        assert author_ids == sorted(author_ids, reverse=True)


@pytest.mark.asyncio
async def test_relation_ordering_multi_by_hierarchical_single_relation_path(db_session, populated_db):
        q = """
        query {
            users(name_ilike: "Charlie") {
                id
                post_comments(order_multi: ["post.author.id:desc", "id:asc"]) {
                    id
                    post {
                        id
                        author { id }
                    }
                }
            }
        }
        """
        res = await berry_schema.execute(q, context_value={'db_session': db_session})
        assert res.errors is None, res.errors
        comments = res.data['users'][0]['post_comments']
        sort_keys = [(comment['post']['author']['id'], comment['id']) for comment in comments]
        assert sort_keys == sorted(sort_keys, key=lambda item: (-item[0], item[1]))


@pytest.mark.asyncio
async def test_relation_ordering_multi_by_hierarchical_single_relation_model_field(db_session, populated_db):
        charlie = populated_db['users'][2]
        third_post = populated_db['posts'][2]
        fourth_post = populated_db['posts'][3]

        third_post.reviewer_id = charlie.id
        fourth_post.reviewer_id = charlie.id
        await db_session.commit()

        q = """
        query {
            users(name_ilike: "Alice") {
                id
                post_comments(order_multi: ["post.reviewer.id:desc", "id:asc"]) {
                    id
                    post {
                        id
                        reviewer { id }
                    }
                }
            }
        }
        """
        res = await berry_schema.execute(q, context_value={'db_session': db_session})
        assert res.errors is None, res.errors
        comments = res.data['users'][0]['post_comments']
        sort_keys = [(comment['post']['reviewer']['id'], comment['id']) for comment in comments]
        assert sort_keys == [(charlie.id, 4), (charlie.id, 6)]


@pytest.mark.asyncio
async def test_relation_ordering_multi_prefetched(db_session, populated_db):
        # Force a user with multiple posts; order by two columns and confirm stable ordering in-memory as well
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
        created = [p['created_at'] for p in posts]
        assert created == sorted(created, reverse=True)
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


@pytest.mark.asyncio
async def test_nested_relation_prefetched_respects_default_and_explicit_order(db_session, populated_db):
        # Fetch posts with nested post_comments; resolver may prefetch comments, and ordering must still apply.
        # 1) Default meta ordering on relation (created_at desc)
        q1 = """
        query {
            posts(limit: 1) {
                id
                post_comments { id created_at }
            }
        }
        """
        res1 = await berry_schema.execute(q1, context_value={'db_session': db_session})
        assert res1.errors is None, res1.errors
        created_default = [c['created_at'] for c in res1.data['posts'][0]['post_comments']]
        assert created_default == sorted(created_default, reverse=True)
        # 2) Explicit override should take precedence over default meta
        q2 = """
        query {
            posts(limit: 1) {
                id
                post_comments(order_by: "id", order_dir: desc) { id }
            }
        }
        """
        res2 = await berry_schema.execute(q2, context_value={'db_session': db_session})
        assert res2.errors is None, res2.errors
        ids_desc = [c['id'] for c in res2.data['posts'][0]['post_comments']]
        assert ids_desc == sorted(ids_desc, reverse=True)


@pytest.mark.asyncio
async def test_invalid_order_field_on_relation_raises(db_session, populated_db):
        # Ensure invalid order_by on a nested relation produces an error (matches DB path validation)
        q = """
        query {
            posts(limit: 1) {
                id
                post_comments(order_by: "__nope__") { id }
            }
        }
        """
        res = await berry_schema.execute(q, context_value={'db_session': db_session})
        assert res.errors is not None
        assert any('Invalid order_by' in str(e) for e in res.errors)
