import pytest

from tests.schema import schema

pytestmark = pytest.mark.asyncio


async def test_upsert_post_creates_author_first(db_session, populated_db):
    # Create a post with nested author (single relation). No author_id is provided.
    mutation = (
        "mutation Upsert($payload: PostQLInput!) {"
        "  merge_post(payload: $payload) { id title author_id author { id name } }"
        "}"
    )
    variables = {
        "payload": {
            "title": "Post with new author",
            "content": "Body",
            # Nested single relation: author
            "author": {
                "name": "Nested Author",
                "email": "nested_author@example.com",
                "is_admin": False
            }
        }
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    if res.errors:
        # Surface full error for diagnostics
        raise AssertionError(f"GraphQL errors: {res.errors}")
    if res.data is None:
        raise AssertionError("No data returned from GraphQL execution.")
    post = res.data["merge_post"]
    assert post["id"] is not None
    assert post["author_id"] is not None
    assert post["author"] is not None
    assert post["author"]["id"] == post["author_id"]
    assert post["author"]["name"] == "Nested Author"
