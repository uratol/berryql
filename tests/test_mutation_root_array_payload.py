import pytest

from tests.schema import schema


pytestmark = pytest.mark.asyncio


async def test_root_merge_posts_accepts_array_payload(db_session, populated_db):
    # Root-level merge_posts should accept a list payload for non-single relations.
    mutation = (
        "mutation Upsert($payload: [PostQLInput!]!) {\n"
        "  merge_posts(payload: $payload) { id title }\n"
        "}"
    )
    variables = {
        "payload": [
            {"title": "A1", "content": "B1", "author_id": 1},
            {"title": "A2", "content": "B2", "author_id": 1},
        ]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    obj = res.data["merge_posts"]
    assert isinstance(obj, list)
    assert len(obj) == 2
    titles = {o["title"] for o in obj}
    assert titles == {"A1", "A2"}
