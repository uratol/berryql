import pytest

from tests.schema import schema

pytestmark = pytest.mark.asyncio


async def test_merge_post_accepts_single_payload(db_session, populated_db):
    # Single-payload merge should accept one PostQLInput object and return a Post
    mutation = (
        "mutation Upsert($payload: PostQLInput!) {\n"
        "  merge_post(payload: $payload) { id title author_id }\n"
        "}"
    )
    variables = {
        "payload": {"title": "Single", "content": "C1", "author_id": 1}
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    post = res.data["merge_post"]
    assert isinstance(post, dict)
    assert post["title"] == "Single"
    assert post["id"] is not None
