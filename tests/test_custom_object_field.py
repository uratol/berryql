import pytest
from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema

@pytest.mark.asyncio
async def test_post_comments_agg_obj_custom_object(db_session, populated_db):
  # Track SQL statements to ensure no N+1 occurs for custom_object
  from sqlalchemy import event
  engine = db_session.get_bind()
  query_counter = {"count": 0}
  def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: D401
    query_counter["count"] += 1
  event.listen(engine, "before_cursor_execute", before_cursor_execute)

  try:
    query = """
    query {
      posts(limit: 2) { id post_comments_agg_obj { min_created_at comments_count } }
    }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    assert len(posts) == 2
    agg0 = posts[0]["post_comments_agg_obj"]
    agg1 = posts[1]["post_comments_agg_obj"]
    # Shape is fixed; datetime should be parsed (or null when no comments)
    assert set(agg0.keys()) == {"min_created_at", "comments_count"}
    assert isinstance(agg0["comments_count"], int)
    # min_created_at is either null or ISO string emitted by GraphQL datetime scalar; just ensure it’s present
    # Since tests use Python datetime, Strawberry serializes to ISO; verify string-like
    if agg0["min_created_at"] is not None:
      assert isinstance(agg0["min_created_at"], str) and len(agg0["min_created_at"]) >= 10
    # Second post also has proper typing
    assert set(agg1.keys()) == {"min_created_at", "comments_count"}
    assert isinstance(agg1["comments_count"], int)
    if agg1["min_created_at"] is not None:
      assert isinstance(agg1["min_created_at"], str) and len(agg1["min_created_at"]) >= 10
    # Expect one SQL query for posts root field; no N+1 for custom_object
    assert query_counter['count'] == 1, f"Expected 1 SQL query for posts root field, got {query_counter['count']}"
  finally:
    event.remove(engine, "before_cursor_execute", before_cursor_execute)
