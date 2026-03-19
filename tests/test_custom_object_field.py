import os

import pytest
from sqlalchemy import event
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


@pytest.mark.asyncio
async def test_custom_and_custom_object_receive_context(db_session, populated_db):
  query = """
  query {
    posts(limit: 1) {
      title
      title_len_custom_context
      post_comments_ctx_obj { flag comments_count }
    }
  }
  """
  ctx = {
    "db_session": db_session,
    "custom_add": 5,
    "custom_obj_flag": 7,
  }
  res = await schema.execute(query, context_value=ctx)
  assert res.errors is None, res.errors
  posts = res.data["posts"]
  assert len(posts) == 1
  post = posts[0]
  assert post["title_len_custom_context"] == len(post["title"]) + 5
  obj = post["post_comments_ctx_obj"]
  assert obj["flag"] == 7
  assert isinstance(obj["comments_count"], int)


@pytest.mark.asyncio
async def test_context_aware_custom_scalar_mssql_root_sql_has_no_inner_from_posts(db_session, populated_db):
  engine = db_session.get_bind()
  statements = []

  def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: D401, ANN001
    try:
      if str(statement).lstrip().upper().startswith("SELECT"):
        statements.append(" ".join(str(statement).split()))
    except Exception:
      pass

  event.listen(engine, "before_cursor_execute", before_cursor_execute)
  try:
    query = """
    query {
      posts(limit: 1) {
        title
        title_len_custom_context
      }
    }
    """
    ctx = {
      "db_session": db_session,
      "custom_add": 5,
    }
    res = await schema.execute(query, context_value=ctx)
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    assert len(posts) == 1
    assert posts[0]["title_len_custom_context"] == len(posts[0]["title"]) + 5

    root_selects = [s.lower() for s in statements if " title_len_custom_context" in s.lower()]
    assert root_selects, f"No root SELECT with title_len_custom_context captured. Statements: {statements}"

    if 'mssql' in (os.getenv('BERRYQL_TEST_DATABASE_URL') or '').lower():
      sql = root_selects[0]
      marker = " as title_len_custom_context"
      assert marker in sql, sql
      expr_prefix = sql.split(marker, 1)[0]
      expr_segment = expr_prefix.rsplit(',', 1)[-1].strip()
      assert "from posts" not in expr_segment, expr_segment
      assert expr_segment.startswith("len(posts.title)") or expr_segment.startswith("len([posts].[title])"), expr_segment
  finally:
    event.remove(engine, "before_cursor_execute", before_cursor_execute)
