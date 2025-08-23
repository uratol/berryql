import pytest

from tests.schema import schema

pytestmark = pytest.mark.asyncio


async def test_merge_post_single_payload_with_nested_author(db_session, populated_db):
    # Single-payload merge supports nested single relation author
    mutation = (
        "mutation Upsert($payload: PostQLInput!) {\n"
        "  merge_post(payload: $payload) { id title author_id author { id name } }\n"
        "}"
    )
    variables = {
        "payload": {
            "title": "Single with nested",
            "content": "Body",
            "author": {"name": "Nested Single", "email": "ns@example.com", "is_admin": False},
        }
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    post = res.data["merge_post"]
    assert post["id"] is not None
    assert post["author_id"] is not None
    assert post["author"]["id"] == post["author_id"]
    assert post["author"]["name"] == "Nested Single"
