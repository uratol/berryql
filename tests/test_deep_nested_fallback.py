import pytest
from sqlalchemy import event
from tests.schema import schema

# This test now asserts that a deeply nested, pushdown-friendly query executes in a single SQL SELECT
# on Postgres by leveraging JSON aggregation pushdown (users -> posts -> post_comments).


@pytest.mark.asyncio
async def test_deep_nested_pushdown_one_select(engine, db_session, populated_db):
    # Count SELECT statements executed by SQLAlchemy during the GraphQL query
    calls = {"selects": 0}

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        try:
            if str(statement).lstrip().upper().startswith("SELECT"):
                calls["selects"] += 1
        except Exception:
            pass

    event.listen(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
    try:
        query = {
            'query': (
                '{ users { posts { post_comments { likes { id user { id name } } } } } } '  # Deeply nested query
            )
        }
        res = await schema.execute(query['query'], context_value={'db_session': db_session})
        assert res.errors is None, f"GraphQL errors: {res.errors}"
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", before_cursor_execute)

    # Expect a single SELECT due to pushdown of nested relations
    assert calls["selects"] == 1, f"Expected 1 SELECT, got {calls['selects']}"
