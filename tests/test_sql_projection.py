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

        # Posts SELECTs must not project unrelated post columns; allow author_id in WHERE.
        # created_at may appear in ORDER BY due to default relation ordering.
        forbidden_post_cols = [
            " title",
            " content",
            " posts.title",
            " posts.content",
            ' "posts".title',
            ' "posts".content',
            " [posts].title",
            " [posts].content",
        ]
        for sql in posts_selects:
            for tok in forbidden_post_cols:
                assert tok not in sql, f"Unexpected post column in SQL: {tok} -> {sql}"
            # Should not involve other tables like post_comments
            assert not _mentions_table(sql, "post_comments"), sql

        # Ensure ordering by created_at desc is applied in the posts subquery (support MSSQL quoting)
        def _has_order_by_created_desc(s: str) -> bool:
            return (
                "order by posts.created_at desc" in s
                or "order by [posts].[created_at] desc" in s
                or 'order by "posts".created_at desc' in s
                or "order by [posts].[created_at] desc for json path" in s
            )
        assert any(_has_order_by_created_desc(sql) for sql in posts_selects), posts_selects

        # Globally, ensure we never touch post_comments table in any captured SQL
        for sql in lowered_norm:
            assert not _mentions_table(sql, "post_comments"), f"Unexpected table post_comments in SQL: {sql}"
    finally:
        try:
            event.remove(engine, "before_cursor_execute", _capture)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_sql_level_where_and_order_defaults_and_args(db_session, sample_users):
        # Capture executed SELECT statements
        engine = db_session.get_bind()
        statements: list[str] = []

        def _capture(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
                try:
                        if str(statement).lstrip().upper().startswith("SELECT"):
                                statements.append(str(statement))
                except Exception:
                        pass

        from tests.schema import schema as berry_schema

        event.listen(engine, "before_cursor_execute", _capture)
        try:
                # 1) Defaults: User.posts has default ORDER BY created_at desc
                q1 = """
                query { users { posts { id } } }
                """
                res1 = await berry_schema.execute(q1, context_value={"db_session": db_session})
                assert res1.errors is None, res1.errors
                lowered_norm = [" ".join(s.lower().split()) for s in statements]
                posts_selects = [s for s in lowered_norm if " from posts" in s or ' from "posts"' in s or ' from [posts]' in s]
                # Support adapter-specific quoting styles
                def _has_order_by_created_desc(s: str) -> bool:
                    return (
                        "order by posts.created_at desc" in s
                        or "order by [posts].[created_at] desc" in s
                        or 'order by "posts".created_at desc' in s
                    )
                assert any(_has_order_by_created_desc(s) for s in posts_selects), posts_selects

                # 2) Explicit args: override ORDER BY to id asc and add WHERE range
                statements.clear()
                q2 = """
                query {
                    users(name_ilike: "Alice") {
                        id
                        posts(order_by: "id", order_dir: asc, where: "{\\"created_at\\": {\\"gt\\": \\\"1900-01-01T00:00:00\\\", \\\"lt\\": \\\"2100-01-01T00:00:00\\\"}}") {
                            id
                        }
                    }
                }
                """
                res2 = await berry_schema.execute(q2, context_value={"db_session": db_session})
                assert res2.errors is None, res2.errors
                lowered_norm = [" ".join(s.lower().split()) for s in statements]
                posts_selects = [s for s in lowered_norm if " from posts" in s or ' from "posts"' in s or ' from [posts]' in s]
                # ORDER BY id asc present (support MSSQL quoting)
                def _has_order_by_id_asc(s: str) -> bool:
                    return (
                        "order by posts.id asc" in s
                        or "order by [posts].[id] asc" in s
                        or 'order by "posts".id asc' in s
                        or "order by [posts].[id] asc offset" in s
                    )
                assert any(_has_order_by_id_asc(s) for s in posts_selects), posts_selects
                # WHERE must include author_id correlation or parameterized predicate (support quoting)
                def _mentions_col(s: str, table: str, col: str) -> bool:
                    return (f"{table}.{col}" in s) or (f'"{table}".{col}' in s) or (f"[{table}].[{col}]" in s)
                assert any(" where " in s and _mentions_col(s, "posts", "author_id") for s in posts_selects), posts_selects
                # WHERE created_at gt and lt present (as two predicates)
                assert any(_mentions_col(s, "posts", "created_at") and " > " in s for s in posts_selects), posts_selects
                assert any(_mentions_col(s, "posts", "created_at") and " < " in s for s in posts_selects), posts_selects

                # 3) Default WHERE via relation posts_recent on User
                statements.clear()
                q3 = """
                query { users { posts_recent { id } } }
                """
                res3 = await berry_schema.execute(q3, context_value={"db_session": db_session})
                assert res3.errors is None, res3.errors
                lowered_norm = [" ".join(s.lower().split()) for s in statements]
                posts_selects = [s for s in lowered_norm if " from posts" in s or ' from "posts"' in s or ' from [posts]' in s]
                # Ensure default where is pushed into SQL (author_id predicate present)
                assert any(" where " in s and _mentions_col(s, "posts", "author_id") for s in posts_selects), posts_selects
                assert any(_mentions_col(s, "posts", "created_at") and " > " in s for s in posts_selects), posts_selects
        finally:
                try:
                        event.remove(engine, "before_cursor_execute", _capture)
                except Exception:
                        pass
