import pytest
from tests.schema import schema as _schema_module


@pytest.mark.asyncio
async def test_view_type_scope_on_posts(db_session, populated_db):
    # With only_view_user_id in context, ViewQL.type_scope should filter views to that user
    q = """
    query {
      posts {
        id
        views { id user_id entity_type }
      }
    }
    """
    res = await _schema_module.execute(
        q,
        context_value={"db_session": db_session, "only_view_user_id": 3},
    )
    assert res.errors is None, res.errors
    posts = res.data.get("posts", [])
    # Flatten views from the first two posts
    views = [v for p in (posts or []) for v in (p.get("views") or [])]
    assert views, "expected at least one view after filtering"
    # All must match the user_id from the context and have entity_type enforced by relation scope
    assert all(int(v["user_id"]) == 3 for v in views)
    assert all(v.get("entity_type") == "post" for v in views)


@pytest.mark.asyncio
async def test_view_type_scope_on_post_comments(db_session, populated_db):
    # Type-level scope should also apply under the comment->views relation
    q = """
    query {
      posts(limit: 3) {
        post_comments(limit: 3) {
          id
          views { id user_id entity_type }
        }
      }
    }
    """
    res = await _schema_module.execute(
        q,
        context_value={"db_session": db_session, "only_view_user_id": 1},
    )
    assert res.errors is None, res.errors
    posts = res.data.get("posts", [])
    # Flatten all comment views
    comments = [c for p in (posts or []) for c in (p.get("post_comments") or [])]
    views = [v for c in comments for v in (c.get("views") or [])]
    assert views, "expected at least one comment view after filtering"
    assert all(int(v["user_id"]) == 1 for v in views)
    assert all(v.get("entity_type") == "post_comment" for v in views)
