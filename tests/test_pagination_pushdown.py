import pytest
from sqlalchemy import event

from tests.schema import schema as berry_schema


def _normalize(sql: str) -> str:
    return " ".join(sql.lower().split())


def _is_mssql(sqls: list[str]) -> bool:
    # Detect MSSQL either by FOR JSON PATH (nested relations) or by bracketed identifiers/OFFSET FETCH syntax
    lowered = [s.lower() for s in sqls]
    if any("for json path" in s for s in lowered):
        return True
    # OFFSET n ROWS FETCH NEXT m ROWS ONLY is typical of MSSQL pagination
    if any(" offset " in s and (" fetch next " in s or " fetch first " in s) for s in lowered):
        return True
    # Bracketed identifiers like [users] also suggest MSSQL
    if any("[users]" in s or "[posts]" in s for s in lowered):
        return True
    return False


@pytest.mark.asyncio
async def test_relation_pagination_sql_pushdown(db_session, sample_users):
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
        query { users { id posts(limit: 2, offset: 1, order_by: "id") { id } } }
        """
        res = await berry_schema.execute(q, context_value={"db_session": db_session})
        assert res.errors is None, res.errors

        lowered = [_normalize(s) for s in statements]
        posts_selects = [s for s in lowered if " from posts" in s or ' from "posts"' in s or ' from [posts]' in s]
        assert posts_selects, f"No posts SELECT captured. Statements: {statements}"

        if _is_mssql(statements):
            # Expect ORDER BY id asc with OFFSET/FETCH
            assert any("order by [posts].[id] asc" in s or 'order by "posts".id asc' in s or 'order by posts.id asc' in s for s in posts_selects), posts_selects
            assert any((" offset 1 rows" in s) or (" offset ? rows" in s) for s in posts_selects), posts_selects
            assert any((" fetch next 2 rows only" in s) or (" fetch first 2 rows only" in s) or (" fetch next ? rows only" in s) or (" fetch first ? rows only" in s) for s in posts_selects), posts_selects
        else:
            # SQLite/Postgres style LIMIT/OFFSET (accept parameterized placeholders too: ?, $n::integer)
            assert any((" limit 2" in s) or (" limit ?" in s) or (" limit $" in s) for s in posts_selects), posts_selects
            assert any((" offset 1" in s) or (" offset ?" in s) or (" offset $" in s) for s in posts_selects), posts_selects
    finally:
        try:
            event.remove(engine, "before_cursor_execute", _capture)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_root_pagination_sql_pushdown(db_session, sample_users):
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
        query { users(limit: 3, offset: 1, order_by: "id") { id } }
        """
        res = await berry_schema.execute(q, context_value={"db_session": db_session})
        assert res.errors is None, res.errors

        lowered = [_normalize(s) for s in statements]
        users_selects = [s for s in lowered if s.strip().startswith("select ") and (" from users" in s or ' from "users"' in s or ' from [users]' in s)]
        assert users_selects, f"No users SELECT captured. Statements: {statements}"

        if _is_mssql(statements):
            assert any("order by users.id asc" in s or 'order by "users".id asc' in s or 'order by [users].[id] asc' in s for s in users_selects), users_selects
            assert any((" offset 1 rows" in s) or (" offset ? rows" in s) for s in users_selects), users_selects
            assert any((" fetch next 3 rows only" in s) or (" fetch first 3 rows only" in s) or (" fetch next ? rows only" in s) or (" fetch first ? rows only" in s) for s in users_selects), users_selects
        else:
            assert any((" limit 3" in s) or (" limit ?" in s) or (" limit $" in s) for s in users_selects), users_selects
            assert any((" offset 1" in s) or (" offset ?" in s) or (" offset $" in s) for s in users_selects), users_selects
    finally:
        try:
            event.remove(engine, "before_cursor_execute", _capture)
        except Exception:
            pass
