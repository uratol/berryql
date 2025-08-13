import pytest
from tests.new.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_comment_post_backref(db_session, populated_db):
    # Fetch comments with their parent post; ensure values match and no errors
    q = """
    query {
      posts(limit: 2) {
        id
        post_comments {
          id
          post { id }
        }
      }
    }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    posts = res.data['posts']
    # Simple sanity: each comment.post.id equals the outer post id
    for p in posts:
        pid = p['id']
        for c in p['post_comments']:
            assert c['post']['id'] == pid


@pytest.mark.asyncio
async def test_negative_limit_raises(db_session, populated_db):
    q = """
    query { posts(limit: -1) { id } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is not None
    assert any('non-negative' in str(e) for e in res.errors)


@pytest.mark.asyncio
async def test_negative_offset_raises(db_session, populated_db):
    q = """
    query { users(offset: -5) { id } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is not None
    assert any('non-negative' in str(e) for e in res.errors)
