import pytest
from sqlalchemy import event

from tests.schema import schema as berry_schema


def _mentions_table(sql: str, table: str) -> bool:
  s = sql.lower()
  return (
    f" from {table}" in s
    or f' from "{table}"' in s
    or f" from [{table}]" in s
  )


def _mentions_column(sql: str, table: str, column: str) -> bool:
    s = sql.lower()
    return (
        f" {table}.{column}" in s
        or f' "{table}".{column}' in s
        or f" [{table}].[{column}]" in s
        or f"select {column}" in s
    )


@pytest.mark.asyncio
async def test_aliases_on_scalars_and_relations(db_session, populated_db):
    q = """
    query {
      users(limit: 1) {
        uid: id
        uname: name
        postsList: posts(limit: 1) { pid: id }
      }
    }
    """
    res = await berry_schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    users = res.data["users"]
    assert isinstance(users, list) and len(users) == 1
    u = users[0]
    # Aliased scalar fields should exist
    assert set(["uid", "uname"]).issubset(set(u.keys()))
    # Aliased relation field should exist and contain aliased child field
    assert "postsList" in u and isinstance(u["postsList"], list)
    if u["postsList"]:
        assert "pid" in u["postsList"][0]


@pytest.mark.asyncio
async def test_alias_projection_only_id(db_session, sample_users):
    # Capture executed SELECT statements to validate projection with alias
    engine = db_session.get_bind()
    statements: list[str] = []

    def _capture(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        try:
            if str(statement).lstrip().upper().startswith("SELECT"):
                statements.append(str(statement))
        except Exception:
            pass

    event.listen(engine, "before_cursor_execute", _capture)
    try:
        q = """
        query { users { userId: id } }
        """
        res = await berry_schema.execute(q, context_value={"db_session": db_session})
        assert res.errors is None, res.errors
        assert res.data is not None and "users" in res.data

        # Validate only id is projected (qualified or not), and no joins
        lowered = [s.lower() for s in statements]
        users_selects = [s for s in lowered if " from users" in s or ' from "users"' in s or " from [users]" in s or s.strip().startswith("select ")]
        assert users_selects, f"No users SELECT captured. Statements: {statements}"
        sql = users_selects[0]
        forbidden_cols = [" name", " email", " is_admin", " created_at", " users.name", " users.email", " users.is_admin", " users.created_at", ' "users".name', ' "users".email', ' "users".is_admin', ' "users".created_at', " [users].name", " [users].email", " [users].is_admin", " [users].created_at"]
        for tok in forbidden_cols:
            assert tok not in sql, f"Unexpected column in SQL: {tok} -> {sql}"
        assert " join " not in sql, sql
    finally:
        try:
            event.remove(engine, "before_cursor_execute", _capture)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_fragment_spreads_on_user_and_post(db_session, populated_db):
    q = """
    query {
      users(limit: 1) {
        ...UserBase
        posts(limit: 1) { ...PostBase }
      }
    }
    fragment UserBase on UserQL { id name }
    fragment PostBase on PostQL { id title }
    """
    res = await berry_schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    users = res.data["users"]
    assert users and set(["id", "name"]).issubset(set(users[0].keys()))
    posts = users[0].get("posts") or []
    if posts:
        assert set(["id", "title"]).issubset(set(posts[0].keys()))


@pytest.mark.asyncio
async def test_inline_fragments(db_session, populated_db):
    # Inline fragments should be honored by the selection extractor
    q = """
    query {
      users(limit: 1) {
        ... on UserQL { id is_admin }
        posts(limit: 1) {
          ... on PostQL { id title }
        }
      }
    }
    """
    res = await berry_schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    users = res.data["users"]
    assert users and set(["id", "is_admin"]).issubset(set(users[0].keys()))
    posts = users[0].get("posts") or []
    if posts:
        assert set(["id", "title"]).issubset(set(posts[0].keys()))


@pytest.mark.asyncio
async def test_named_fragment_projection_on_nested_relation(db_session, populated_db):
    engine = db_session.get_bind()
    statements: list[str] = []

    def _capture(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        try:
            if str(statement).lstrip().upper().startswith("SELECT"):
                statements.append(str(statement))
        except Exception:
            pass

    event.listen(engine, "before_cursor_execute", _capture)
    try:
        q = """
        query {
          users(limit: 1) {
            id
            posts(limit: 1) {
              ...PostFields
            }
          }
        }

        fragment PostFields on PostQL {
          id
          title
        }
        """
        res = await berry_schema.execute(q, context_value={"db_session": db_session})
        assert res.errors is None, res.errors
        users = res.data["users"]
        assert users and "posts" in users[0]
        posts = users[0]["posts"] or []
        if posts:
            assert set(["id", "title"]).issubset(set(posts[0].keys()))

        lowered = [" ".join(s.lower().split()) for s in statements]
        posts_selects = [s for s in lowered if _mentions_table(s, "posts")]
        assert posts_selects, f"No posts SELECT captured. Statements: {statements}"

        assert any(_mentions_column(sql, "posts", "title") for sql in posts_selects), posts_selects

        forbidden_columns = [
          "content",
          "metadata_json",
          "binary_blob",
        ]
        for sql in posts_selects:
          for column in forbidden_columns:
            assert not _mentions_column(sql, "posts", column), f"Unexpected column in SQL: {column} -> {sql}"
    finally:
        try:
            event.remove(engine, "before_cursor_execute", _capture)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_inline_fragment_projection_on_nested_single_relation(db_session, populated_db):
    engine = db_session.get_bind()
    statements: list[str] = []

    def _capture(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        try:
            if str(statement).lstrip().upper().startswith("SELECT"):
                statements.append(str(statement))
        except Exception:
            pass

    event.listen(engine, "before_cursor_execute", _capture)
    try:
        q = """
        query {
          posts(limit: 1) {
            id
            author {
              ... on UserQL {
                id
                name
              }
            }
          }
        }
        """
        res = await berry_schema.execute(q, context_value={"db_session": db_session})
        assert res.errors is None, res.errors
        posts = res.data["posts"]
        assert posts and "author" in posts[0]
        author = posts[0]["author"]
        assert author is not None
        assert set(["id", "name"]).issubset(set(author.keys()))

        lowered = [" ".join(s.lower().split()) for s in statements]
        user_selects = [s for s in lowered if _mentions_table(s, "users")]
        assert user_selects, f"No users SELECT captured. Statements: {statements}"

        assert any(_mentions_column(sql, "users", "name") for sql in user_selects), user_selects

        forbidden_columns = [
          "email",
          "is_admin",
          "created_at",
        ]
        for sql in user_selects:
          for column in forbidden_columns:
            assert not _mentions_column(sql, "users", column), f"Unexpected column in SQL: {column} -> {sql}"
    finally:
        try:
            event.remove(engine, "before_cursor_execute", _capture)
        except Exception:
            pass
