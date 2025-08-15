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
async def test_relation_argument_title_ilike_is_pushed_down(db_session, sample_users):
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
        q = '{ users { id posts(limit: 5, title_ilike: "GraphQL") { id title } } }'
        res = await berry_schema.execute(q, context_value={"db_session": db_session})
        assert res.errors is None, res.errors

        lowered = [_normalize(s) for s in statements]
        # Find SELECTs that touch posts
        posts_selects = [s for s in lowered if _mentions_table(s, "posts")]
        assert posts_selects, f"No posts-related SELECT captured. Statements: {statements}"
        # De-dup by text and operate on the unique shapes
        distinct_posts = list(dict.fromkeys(posts_selects))

        # Ensure only one posts-related SELECT shape (no N+1)
        assert len(distinct_posts) == 1, distinct_posts

        s = distinct_posts[0]
        # Ensure LIKE/ILIKE predicate is present. Allow bind parameters instead of literal value.
        has_like = (" ilike " in s) or (" like " in s)
        has_bind = ("$1" in s or "$2" in s or "$3" in s or "?" in s or re.search(r"%\(.*?\)s", s) is not None)
        has_literal = "%graphql%" in s
        assert has_like and (has_bind or has_literal), s
        # Also ensure correlation between posts and users is present (join predicate or parameterized form)
        corr_tokens = [
            "posts.author_id = users.id",
            '"posts".author_id = "users".id',
            "[posts].[author_id] = [users].[id]",
            "posts.author_id = ?",
            '"posts".author_id = ?',
            "[posts].[author_id] = ?",
        ]
        pg_bind_pattern = re.compile(r"(?<![\w])posts\.author_id\s*=\s*\$\d+(::[a-z_]+)?")
        assert any((any(tok in sel for tok in corr_tokens) or pg_bind_pattern.search(sel)) for sel in distinct_posts), distinct_posts
    finally:
        try:
            event.remove(engine, "before_cursor_execute", _capture)
        except Exception:
            pass
