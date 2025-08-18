import asyncio
import pytest

from tests.schema import schema
from tests.fixtures import populated_db  # noqa: F401


@pytest.mark.asyncio
async def test_mutation_create_post_and_subscription(db_session, populated_db):
    # Start subscription first
    sub = await schema.subscribe("subscription { post_created { id title author_id } }")
    assert hasattr(sub, "__anext__") or hasattr(sub, "__aiter__"), "subscribe() should return async iterator"

    # Run mutation to create a post
    mutation = (
        "mutation { create_post(title: \"From test\", content: \"Body\", author_id: %d) { id title author_id } }"
        % populated_db["users"][0].id
    )
    mres = await schema.execute(mutation, context_value={"db_session": db_session})
    assert mres.errors is None, mres.errors
    created = mres.data["create_post"]
    assert isinstance(created, dict)
    new_id = int(created["id"]) if created and "id" in created else None
    assert isinstance(new_id, int)

    # Pull first subscription event
    it = sub if hasattr(sub, "__anext__") else sub.__aiter__()
    event = await it.__anext__()
    assert getattr(event, "errors", None) is None, getattr(event, "errors", None)
    payload = event.data["post_created"]
    assert int(payload["id"]) == int(new_id)
    assert payload["title"] == "From test"
