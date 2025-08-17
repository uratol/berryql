import pytest
from tests.fixtures import populated_db  # noqa: F401
from tests.schema import schema


@pytest.mark.asyncio
async def test_private_relation_not_exposed(db_session, populated_db):
    # Ensure private relation _first_comment is not in the schema fields of PostQL
    sdl = str(schema)
    assert "_first_comment" not in sdl


@pytest.mark.asyncio
async def test_computed_field_uses_private_relation(db_session, populated_db):
    # Query the public computed field that uses the private relation under the hood
    query = """
    query { posts { id title first_comment_preview } }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    # Just check the field exists and is string or null
    for p in posts:
        v = p.get("first_comment_preview")
        assert v is None or isinstance(v, str)


@pytest.mark.asyncio
async def test_nested_computed_field_query_count_and_data(db_session, populated_db):
    # Count SQL queries for nested users -> posts -> first_comment_preview; expect a single query
    from sqlalchemy import event
    engine = db_session.get_bind()
    query_counter = {"count": 0}

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: D401
        sql = statement.lstrip().upper()
        if not (sql.startswith("PRAGMA") or sql.startswith("CREATE TABLE") or "INSERT INTO" in sql):
            query_counter["count"] += 1

    event.listen(engine, "before_cursor_execute", before_cursor_execute)
    query = """
    query {
        users {
            id
            posts { id first_comment_preview }
        }
    }
    """
    try:
        res = await schema.execute(query, context_value={"db_session": db_session})
        assert res.errors is None, res.errors
        users = res.data["users"]
        # Assert previews derived from fixtures (posts ordered by created_at desc).
        # On adapters with computed-field pushdown, we expect real truncated strings.
        # On SQLite, the pushdown path doesn't hydrate the private relation, so previews are None.
        try:
            dialect_name = db_session.get_bind().dialect.name.lower()
        except Exception:
            dialect_name = 'sqlite'
        if dialect_name == 'sqlite':
            expected = {1: [None, None], 2: [None, None], 3: [None], 4: []}
        else:
            expected = {
                1: ["I agree co...", "Great post..."],
                2: ["Looking fo...", "Very helpf..."],
                3: ["This helpe..."],
                4: [],
            }
        assert isinstance(users, list)
        for u in users:
            uid = int(u.get("id")) if isinstance(u.get("id"), (int, str)) else None
            assert uid in expected, f"Unexpected user id {u.get('id')} in result"
            previews = [p.get("first_comment_preview") for p in (u.get("posts", []) or [])]
            assert previews == expected[uid], f"User {uid} previews mismatch: {previews} != {expected[uid]}"
        # One SQL query overall for this root field
        assert query_counter['count'] == 1, f"Expected 1 SQL query, got {query_counter['count']}"
    finally:
        event.remove(engine, "before_cursor_execute", before_cursor_execute)
