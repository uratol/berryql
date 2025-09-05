import pytest

from tests.schema import schema, berry_schema

@pytest.mark.asyncio
async def test_polymorphic_views_on_posts(populated_db, db_session):
  # Count SQL queries to ensure pushdown (no per-parent N+1)
  from sqlalchemy import event
  engine = db_session.get_bind()
  query_counter = {"count": 0}
  def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: D401
    if not statement.lstrip().upper().startswith("PRAGMA") and not statement.lstrip().upper().startswith("CREATE TABLE") and "INSERT INTO" not in statement:
      query_counter["count"] += 1
  event.listen(engine, "before_cursor_execute", before_cursor_execute)
  # Query posts with their polymorphic views
  q = '''
    query {
      posts(order_by: "id") {
        id
        title
        views { id user_id entity_type entity_id }
      }
    }
    '''
  try:
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, f"GraphQL errors: {res.errors}"
    data = res.data["posts"]
    # Post 1 has 2 views in fixtures, Post 2 has 1
    p1 = next(p for p in data if int(p["id"]) == int(populated_db['posts'][0].id))
    p2 = next(p for p in data if int(p["id"]) == int(populated_db['posts'][1].id))
    assert len(p1["views"]) == 2
    assert len(p2["views"]) == 1
    for v in p1["views"]:
      assert v["entity_type"] == "post"
      assert int(v["entity_id"]) == int(populated_db['posts'][0].id)
    # Expect a single SQL query for the posts root (views pushed down)
    assert query_counter['count'] == 1, f"Expected 1 SQL query for posts with views, got {query_counter['count']}"
  finally:
    event.remove(engine, "before_cursor_execute", before_cursor_execute)

@pytest.mark.asyncio
async def test_polymorphic_views_on_comments(populated_db, db_session):
  from sqlalchemy import event
  engine = db_session.get_bind()
  query_counter = {"count": 0}
  def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: D401
    if not statement.lstrip().upper().startswith("PRAGMA") and not statement.lstrip().upper().startswith("CREATE TABLE") and "INSERT INTO" not in statement:
      query_counter["count"] += 1
  event.listen(engine, "before_cursor_execute", before_cursor_execute)
  q = '''
    query {
      posts(order_by: "id") {
        id
        post_comments(order_by: "id") { id views { id user_id entity_type entity_id } }
      }
    }
    '''
  try:
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, f"GraphQL errors: {res.errors}"
    data = res.data["posts"]
    # First post has two comments; the first two comments have views seeded
    first_post = next(p for p in data if int(p["id"]) == int(populated_db['posts'][0].id))
    comments = first_post["post_comments"]
    assert len(comments) >= 2
    assert len(comments[0]["views"]) >= 1
    for v in comments[0]["views"]:
      assert v["entity_type"] == "post_comment"
      assert int(v["entity_id"]) == int(populated_db['post_comments'][0].id)
    # Expect a single SQL query for posts root with nested comments->views pushdown
    assert query_counter['count'] == 1, f"Expected 1 SQL query for posts with comments.views, got {query_counter['count']}"
  finally:
    event.remove(engine, "before_cursor_execute", before_cursor_execute)

@pytest.mark.asyncio
async def test_type_scope_filters_views_by_context_user(populated_db, db_session):
  # Limit views to a specific user via context-aware type-level scope on ViewQL
  user = populated_db['users'][2]  # user3 in fixtures
  q = '''
    query {
      posts(order_by: "id") {
        id
        views { id user_id entity_type entity_id }
      }
    }
  '''
  res = await schema.execute(q, context_value={"db_session": db_session, "only_view_user_id": int(user.id)})
  assert res.errors is None, f"GraphQL errors: {res.errors}"
  data = res.data["posts"]
  # Post 1 has two views overall (user2, user3) but filtered by user3 we expect one
  p1 = next(p for p in data if int(p["id"]) == int(populated_db['posts'][0].id))
  assert len(p1["views"]) == 1
  v = p1["views"][0]
  assert int(v["user_id"]) == int(user.id)
  assert v["entity_type"] == "post"
  assert int(v["entity_id"]) == int(populated_db['posts'][0].id)

  # Also verify comment views respect the same scope
  q2 = '''
    query {
      posts(order_by: "id") {
        id
        post_comments(order_by: "id") { id views { id user_id entity_type entity_id } }
      }
    }
  '''
  res2 = await schema.execute(q2, context_value={"db_session": db_session, "only_view_user_id": int(user.id)})
  assert res2.errors is None, f"GraphQL errors: {res2.errors}"
  data2 = res2.data["posts"]
  first_post = next(p for p in data2 if int(p["id"]) == int(populated_db['posts'][0].id))
  comments = first_post["post_comments"]
  # Fixture comment views are by user1, so under user3 filter expect zero
  for c in comments:
    assert len(c["views"]) == 0
