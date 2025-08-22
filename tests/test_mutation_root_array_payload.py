import pytest

from tests.schema import schema


pytestmark = pytest.mark.asyncio


async def test_root_merge_posts_rejects_array_payload(db_session, populated_db):
    # Root-level merge_posts expects a single PostQLInput, not a list.
    mutation = (
        "mutation Upsert($payload: PostQLInput!) {\n"
    "  merge_posts(payload: $payload) { id title }\n"
        "}"
    )
    variables = {
        # Intentionally send an array to root-level payload
        "payload": [
            {"title": "A1", "content": "B1", "author_id": 1},
            {"title": "A2", "content": "B2", "author_id": 1},
        ]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    # Expect GraphQL validation/type error because payload type is PostQLInput, not a list
    assert res.errors is not None
    # A generic check that mentions the variable type mismatch
    msg = str(res.errors[0])
    assert "PostQLInput" in msg or "expected type" in msg.lower()
