import pytest
from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema


import pytest
from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema


@pytest.mark.asyncio
async def test_single_object_relations_and_count(db_session, populated_db):
  from sqlalchemy import event
  engine = db_session.get_bind()
  query_counter = {"count": 0}

  def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: D401
    query_counter["count"] += 1

  event.listen(engine, "before_cursor_execute", before_cursor_execute)
  query = """
  query {
    posts(limit: 2) {
    id
    author { id name }
    post_comments(limit: 2) { id }
    post_comments_agg
    }
    users(limit: 1) { id name posts(limit: 2) { id title } post_agg post_agg_obj { count } }
  }
  """
  try:
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data
    assert "posts" in data
    assert len(data["posts"]) == 2
    first_post = data["posts"][0]
    assert "author" in first_post and isinstance(first_post["author"], dict)
    assert "post_comments_agg" in first_post
    assert "users" in data
    assert len(data["users"]) == 1
    assert "post_agg" in data["users"][0]
    assert "post_agg_obj" in data["users"][0]
    agg_obj = data["users"][0]["post_agg_obj"]
    if agg_obj is not None:
      assert set(agg_obj.keys()) == {"count"}
    # Expect one SQL query per root field (posts + users) on adapters with pushdown
    try:
      dialect_name = db_session.get_bind().dialect.name.lower()
    except Exception:
      dialect_name = 'sqlite'
  # MSSQL now supports relation pushdown; enforce same query count
    assert query_counter['count'] == 2, f"Expected 2 SQL queries (one per root field), got {query_counter['count']}"
  finally:
    event.remove(engine, "before_cursor_execute", before_cursor_execute)


@pytest.mark.asyncio
async def test_users_post_agg_obj_nullable(db_session, populated_db):
  """post_agg_obj should be null for users with no posts.

  Query mirrors the reported case:
  {
    users(limit: 5, order_by: "id", order_dir: asc) {
      id
      post_agg_obj { count }
    }
  }
  """
  query = """
  query {
    users(limit: 5, order_by: "id", order_dir: asc) {
      id
      post_agg_obj { count }
    }
  }
  """
  res = await schema.execute(query, context_value={"db_session": db_session})
  assert res.errors is None, res.errors
  users = res.data["users"]
  # With our fixtures there are 4 users (IDs start at 1), last user has no posts
  assert len(users) == 4
  # Expected counts per user in fixtures: [2, 2, 1, 0]
  expected_counts = [2, 2, 1, 0]
  # For all users, post_agg_obj should be non-null and expose only { count } with correct values
  for idx, u in enumerate(users):
    agg = u.get("post_agg_obj")
    assert agg is not None
    assert set(agg.keys()) == {"count"}
    assert isinstance(agg["count"], int)
    assert agg["count"] == expected_counts[idx]
