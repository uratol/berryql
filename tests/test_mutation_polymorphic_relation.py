import pytest
from tests.schema import schema

@pytest.mark.asyncio
async def test_polymorphic_post_views_mutation_defaults_entity_type(db_session, populated_db):
    # Create a Post and a View via nested mutation under Post.views without specifying entity_type.
    # Scope on views relation enforces {"entity_type": {"eq": "post"}}, which should be defaulted in mutation.
    mutation = (
        "mutation Upsert($payload: [PostQLInput!]!) {\n"
            "  merge_posts(payload: $payload) {\n"
        "    id\n"
        "    title\n"
        "    views { id entity_type entity_id user_id }\n"
        "  }\n"
        "}"
    )
    variables = {
        "payload": [{
            "title": "With view",
            "content": "Body",
            "author_id": 1,
            # Create a view referencing this post; omit entity_type so scope should fill it
            "views": [
                {"user_id": 1}  # entity_id will be set by mutation via FK; entity_type from scope
            ],
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, f"Errors: {res.errors}"
    data = res.data["merge_posts"]
    assert data["id"] is not None
    assert data["views"] and len(data["views"]) == 1
    v = data["views"][0]
    assert v["entity_type"] == "post"
    # entity_id should match post id
    assert int(v["entity_id"]) == int(data["id"])  # ints as strings possible
