import pytest
import re
from sqlalchemy import event

from tests.schema import schema as berry_schema


def _normalize(sql: str) -> str:
    return " ".join(sql.lower().split())


def _mentions_table(sql: str, table: str) -> bool:
    s = sql.lower()
    return (f" from {table}" in s) or (f' from "{table}"' in s) or (f" from [{table}]" in s)


@pytest.mark.asyncio
async def test_relation_callable_where_pushdown_posts_have_comments(db_session, sample_users):
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
        query { users { posts_have_comments { id } } }
        """
        res = await berry_schema.execute(q, context_value={"db_session": db_session})
        assert res.errors is None, res.errors

        lowered = [_normalize(s) for s in statements]
        # Selects that include FROM posts (either as main FROM or in a subquery)
        posts_selects = [s for s in lowered if _mentions_table(s, "posts")]
        assert posts_selects, f"No posts-related SELECT captured. Statements: {statements}"

        # Ensure the callable default where hits post_comments within the same SQL (correlated subquery)
        assert any(_mentions_table(s, "post_comments") for s in posts_selects), posts_selects

        # Also ensure correlation between posts and users is present in WHERE.
        # Accept either explicit table correlation (users.id) or parameterized predicate.
        corr_tokens = [
                "posts.author_id = users.id",
                '"posts".author_id = "users".id',
                "[posts].[author_id] = [users].[id]",
                "posts.author_id = ?",
                '"posts".author_id = ?',
                "[posts].[author_id] = ?",
            ]
        # Accept also Postgres positional bind ($1, $2, possibly with ::type casts)
        pg_bind_pattern = re.compile(r"(?<![\w])posts\.author_id\s*=\s*\$\d+(::[a-z_]+)?")
        assert any((any(tok in s for tok in corr_tokens) or pg_bind_pattern.search(s)) for s in posts_selects), posts_selects
    finally:
        try:
            event.remove(engine, "before_cursor_execute", _capture)
        except Exception:
            pass
