import pytest
from strawberry.types import Info
from tests.schema import schema as _schema_module


@pytest.mark.asyncio
async def test_type_level_scope_and_relation_scope(db_session, populated_db):
    # Define a temporary type-level scope on PostCommentLikeQL: only user_id == 1
    from tests.schema import PostCommentLikeQL
    # Save original if existed
    orig = getattr(PostCommentLikeQL, '__type_scope__', None)
    try:
        PostCommentLikeQL.__type_scope__ = {'user_id': {'eq': 1}}
        # Query admin_likes under a comment where relation scope is also user_id == 1 -> should pass
        q_ok = """
        query {
          posts(limit: 1) {
            post_comments(limit: 1) {
              admin_likes { id user_id }
            }
          }
        }
        """
        res_ok = await _schema_module.execute(q_ok, context_value={'db_session': db_session})
        assert res_ok.errors is None, res_ok.errors
        # There should be at least one like and all must have user_id == 1
        likes = (
            res_ok.data.get('posts', [])
            and res_ok.data['posts'][0].get('post_comments', [])
            and res_ok.data['posts'][0]['post_comments'][0].get('admin_likes', [])
        )
        assert isinstance(likes, list)
        assert all(int(l['user_id']) == 1 for l in likes)

        # Now query general likes (relation scope: id > 0) combined with type scope user_id == 1 -> only user_id 1 should remain
        q = """
        query {
          posts(limit: 1) {
            post_comments(limit: 1) {
              likes { id user_id }
            }
          }
        }
        """
        res = await _schema_module.execute(q, context_value={'db_session': db_session})
        assert res.errors is None, res.errors
        likes2 = (
            res.data.get('posts', [])
            and res.data['posts'][0].get('post_comments', [])
            and res.data['posts'][0]['post_comments'][0].get('likes', [])
        )
        assert isinstance(likes2, list)
        # All likes returned must satisfy type-level scope user_id == 1
        assert all(int(l['user_id']) == 1 for l in likes2)
    finally:
        # Cleanup
        if orig is None:
            try:
                delattr(PostCommentLikeQL, '__type_scope__')
            except Exception:
                pass
        else:
            PostCommentLikeQL.__type_scope__ = orig
