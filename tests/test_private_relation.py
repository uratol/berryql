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
