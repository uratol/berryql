import pytest
from sqlalchemy import event

from tests.schema import schema


@pytest.mark.asyncio
async def test_users_id_only_projection(db_session, sample_users):
    # Capture executed SELECT statements
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
        query = """
        query { users { id } }
        """
        res = await schema.execute(query, context_value={"db_session": db_session})
        assert res.errors is None, f"GraphQL errors: {res.errors}"
        assert res.data is not None

        # Find the users SELECT statement
        def _is_users_select(sql: str) -> bool:
            s = sql.lower()
            return " from users" in s or " from \"users\"" in s or " from [users]" in s or s.strip().startswith("select ")

        users_selects = [s for s in statements if _is_users_select(s)]
        assert users_selects, f"No users SELECT captured. Statements: {statements}"
        sql = users_selects[0].lower()

        # Should not include unrequested scalar columns (qualified or not)
        forbidden_cols = ["name", "email", "is_admin", "created_at"]
        for col in forbidden_cols:
            assert f" {col}" not in sql, f"Unexpected column in SQL: {col} -> {sql}"
            assert f" users.{col}" not in sql, f"Unexpected qualified column in SQL: users.{col} -> {sql}"
            assert f' "users".{col}' not in sql, f"Unexpected qualified column in SQL: 'users'.{col} -> {sql}"
            assert f" [users].{col}" not in sql, f"Unexpected qualified column in SQL: [users].{col} -> {sql}"

        # Should project only id from users table (allow qualifiers and ORDER BY)
        assert (
            "select id" in sql
            or "select users.id" in sql
            or 'select "users".id' in sql
            or "select [users].id" in sql
        ), sql
        # No joins to other tables
        assert " join " not in sql, sql

        # No relation subqueries (e.g., FROM posts/post_comments) should appear
        for t in ("posts", "post_comments"):
            assert f" from {t}" not in sql, sql
            assert f' from "{t}"' not in sql, sql
            assert f" from [{t}]" not in sql, sql

        # No custom field pushdown labels or JSON functions
        forbidden_tokens = [
            "_pushcf_",  # custom field/object labels
            "_pushrel_",  # relation pushdown labels
            "json_object(",
            "json_group_array",
            "for json path",
            "post_agg",
        ]
        for tok in forbidden_tokens:
            assert tok not in sql, f"Unexpected token in SQL: {tok} -> {sql}"
    finally:
        try:
            event.remove(engine, "before_cursor_execute", _capture)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_users_posts_id_only_projection(db_session, sample_users):
    # Capture executed SELECT statements
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
        query = """
        query { users { posts { id } } }
        """
        res = await schema.execute(query, context_value={"db_session": db_session})
        assert res.errors is None, f"GraphQL errors: {res.errors}"
        assert res.data is not None

        lowered = [s.lower() for s in statements]
        lowered_norm = [" ".join(s.split()) for s in lowered]

        # Ensure we selected from users and posts, but nothing else like post_comments
        def _mentions_table(sql: str, table: str) -> bool:
            return (
                f" from {table}" in sql
                or f' from "{table}"' in sql
                or f" from [{table}]" in sql
            )

        users_selects = [s for s in lowered_norm if _mentions_table(s, "users") or s.strip().startswith("select ")]
        posts_selects = [s for s in lowered_norm if _mentions_table(s, "posts")]

        assert users_selects, f"No users SELECT captured. Statements: {statements}"
        assert posts_selects, f"No posts-related SELECT captured. Statements: {statements}"

        # Users root SELECT should not include unrequested user scalar columns
        forbidden_user_cols = [" name", " email", " is_admin", " created_at", " users.name", " users.email", " users.is_admin", " users.created_at", ' "users".name', ' "users".email', ' "users".is_admin', ' "users".created_at', " [users].name", " [users].email", " [users].is_admin", " [users].created_at"]
        for sql in users_selects:
            for tok in forbidden_user_cols:
                assert tok not in sql, f"Unexpected user column in SQL: {tok} -> {sql}"
            # Prefer no joins at root; relation handled via subselect or separate query
            assert " join " not in sql, sql

        # Posts SELECTs must not project unrelated post columns; allow author_id in WHERE
        forbidden_post_cols = [" title", " content", " created_at", " posts.title", " posts.content", " posts.created_at", ' "posts".title', ' "posts".content', ' "posts".created_at', " [posts].title", " [posts].content", " [posts].created_at"]
        for sql in posts_selects:
            for tok in forbidden_post_cols:
                assert tok not in sql, f"Unexpected post column in SQL: {tok} -> {sql}"
            # Should not involve other tables like post_comments
            assert not _mentions_table(sql, "post_comments"), sql

        # Globally, ensure we never touch post_comments table in any captured SQL
        for sql in lowered_norm:
            assert not _mentions_table(sql, "post_comments"), f"Unexpected table post_comments in SQL: {sql}"
    finally:
        try:
            event.remove(engine, "before_cursor_execute", _capture)
        except Exception:
            pass
