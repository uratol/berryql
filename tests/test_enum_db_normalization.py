import asyncio
import re
import pytest
from sqlalchemy import event
from tests.schema import schema

pytestmark = pytest.mark.asyncio


async def test_insert_uses_lowercase_status_value(db_session, populated_db):
    captured_params = []

    # Capture parameters on the DBAPI cursor before execution
    @event.listens_for(db_session.sync_session.bind, "before_cursor_execute")  # type: ignore[attr-defined]
    def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # pragma: no cover - instrumentation
        try:
            if isinstance(statement, str) and re.search(r"INSERT\s+INTO\s+posts", statement, re.IGNORECASE):
                # parameters can be a list of tuples or a single tuple
                if isinstance(parameters, (list, tuple)) and parameters and isinstance(parameters[0], (list, tuple)):
                    captured_params.extend(parameters)
                else:
                    captured_params.append(parameters)
        except Exception:
            pass

    # Run mutation that sets status to PUBLISHED (should be stored as 'published')
    mutation = (
        "mutation Upsert($payload: PostQLInput!) {\n"
        "  merge_post(payload: $payload) { id title status author_id }\n"
        "}"
    )
    variables = {"payload": {"title": "Enum Captured", "content": "C2", "author_id": 1, "status": "PUBLISHED"}}
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors

    # Ensure we captured at least one INSERT into posts
    assert any(isinstance(p, (list, tuple)) for p in captured_params)

    # Find a parameters tuple matching posts insert shape and check last param (status)
    # With SQLite echo format we expect: (title, content, author_id, created_at, binary_blob, status)
    status_values = []
    for p in captured_params:
        try:
            if isinstance(p, (list, tuple)) and len(p) >= 6:
                status_values.append(p[-1])
        except Exception:
            continue
    assert status_values, f"No captured status values; captured={captured_params!r}"
    # All captured status parameters for posts should be lowercase values
    for sv in status_values:
        if isinstance(sv, str):
            assert sv.islower(), f"Expected lowercase storage value, got {sv!r}"
        # Allow Enum instance in fixture inserts (if any), but mutation path must be lowercase