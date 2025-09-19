import pytest

@pytest.mark.asyncio
async def test_relation_order_by_callable_author_created_at(db_session, populated_db):
    # Use the schema with PostQL.post_comments_ordered_asc ordering by User.created_at via callable
    from tests.schema import schema

    q = '''
    query {
      posts {
        id
        post_comments_ordered_asc { id author { id created_at } }
      }
    }
    '''
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data
    assert data and 'posts' in data
    # Find a post that has at least 2 comments and verify ordering
    found_checked = False
    for p in data['posts']:
        comments = p.get('post_comments_ordered_asc') or []
        if len(comments) >= 2:
            times = [c['author']['created_at'] for c in comments if c.get('author')]
            assert times == sorted(times), f"Expected ascending by author.createdAt, got {times}"
            found_checked = True
            break
    if not found_checked:
        pytest.skip("No post had at least 2 comments to verify ordering")
