import pytest

from tests.schema import schema, CALLBACK_EVENTS
from tests.fixtures import populated_db  # noqa: F401


@pytest.mark.asyncio
async def test_merge_delete_invokes_callbacks(db_session, populated_db):
    # Ensure callbacks captured
    CALLBACK_EVENTS.clear()
    post = populated_db["posts"][0]
    pid = int(post.id)

    # Delete existing post via _Delete flag; callbacks set on root Query.posts
    mutation = (
    "mutation($p: [PostQLInput!]!) { merge_posts(payload: $p) { id title author_id } }"
    )
    variables = {"p": [{"id": pid, "_Delete": True}]}
    res = await schema.execute(
        mutation,
        variable_values=variables,
        context_value={"db_session": db_session, "test_callbacks": True},
    )
    assert res.errors is None, res.errors
    # We return the deleted instance snapshot; id should match
    out = res.data["merge_posts"]
    assert int(out["id"]) == pid

    # Pre and post should both have been logged with created=False
    kinds = [e.get("event") for e in CALLBACK_EVENTS]
    assert "pre" in kinds and "post" in kinds
    assert any(e.get("event") == "post" and e.get("created") is False for e in CALLBACK_EVENTS)

    # Verify it's gone
    q = f"query {{ posts(where: \"{{\\\"id\\\": {{\\\"eq\\\": {pid}}}}}\") {{ id }} }}"
    res2 = await schema.execute(q, context_value={"db_session": db_session})
    assert res2.errors is None, res2.errors
    assert res2.data["posts"] == []
