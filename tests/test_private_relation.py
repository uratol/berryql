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
    # Query the public relation and inspect nested preview from the related object
    query = """
    query { posts { id title first_comment_preview { content_preview } } }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    # Check the nested preview exists and is string or null
    for p in posts:
        node = p.get("first_comment_preview")
        if node is None:
            continue
        v = node.get("content_preview") if isinstance(node, dict) else None
        assert v is None or isinstance(v, str)


@pytest.mark.asyncio
async def test_nested_computed_field_data(db_session, populated_db):
    # Nested users -> posts -> first_comment_preview { content_preview }
    query = """
    query {
        users {
            id
            posts { id first_comment_preview { content_preview } }
        }
    }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    users = res.data["users"]
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
        previews = []
        for p in (u.get("posts", []) or []):
            node = p.get("first_comment_preview")
            previews.append(node.get("content_preview") if isinstance(node, dict) else None)
        assert previews == expected[uid], f"User {uid} previews mismatch: {previews} != {expected[uid]}"


@pytest.mark.asyncio
async def test_pushdown_query_counts_for_first_and_list_previews(db_session, populated_db):
    # Ensure a single SQL query is executed for users -> posts -> first_comment_preview
    from sqlalchemy import event
    engine = db_session.get_bind()
    query_counter = {"count": 0}

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        sql = statement.lstrip().upper()
        if not (sql.startswith("PRAGMA") or sql.startswith("CREATE TABLE") or "INSERT INTO" in sql):
            query_counter["count"] += 1

    event.listen(engine, "before_cursor_execute", before_cursor_execute)
    try:
        q1 = """
        query {
            users {
                id
                posts { id first_comment_preview { id content content_preview } }
            }
        }
        """
        res1 = await schema.execute(q1, context_value={"db_session": db_session})
        assert res1.errors is None, res1.errors
        assert query_counter['count'] == 1, f"Expected 1 SQL query for first_comment_preview, got {query_counter['count']}"
    finally:
        event.remove(engine, "before_cursor_execute", before_cursor_execute)
