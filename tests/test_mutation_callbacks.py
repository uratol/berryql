import pytest

from tests.schema import schema, CALLBACK_EVENTS
from tests.fixtures import populated_db  # noqa: F401


@pytest.mark.asyncio
async def test_upsert_callbacks_create(db_session, populated_db):
    # Clear log
    CALLBACK_EVENTS.clear()
    uid = populated_db["users"][0].id
    # Use root-level upsert with callbacks
    mutation = (
        "mutation($payload: PostQLInput!) { merge_post(payload: $payload) { id title author_id } }"
    )
    variables = {
        "payload": {"title": "New", "content": "Body", "author_id": uid}
    }
    res = await schema.execute(
        mutation,
        variable_values=variables,
        context_value={"db_session": db_session, "test_callbacks": True},
    )
    assert res.errors is None, res.errors
    post = res.data["merge_post"]
    assert post["title"].startswith("[pre]New")
    assert post["title"].endswith("[post]")
    # Ensure both pre and post were logged
    kinds = [e["event"] for e in CALLBACK_EVENTS]
    assert "pre" in kinds and "post" in kinds


@pytest.mark.asyncio
async def test_upsert_callbacks_update(db_session, populated_db):
    CALLBACK_EVENTS.clear()
    uid = populated_db["users"][1].id
    # First create
    create = "mutation($p: PostQLInput!) { merge_post(payload: $p) { id title } }"
    payload = {"title": "Once", "content": "B", "author_id": uid}
    res1 = await schema.execute(
        create,
        variable_values={"p": payload},
        context_value={"db_session": db_session, "test_callbacks": True},
    )
    assert res1.errors is None, res1.errors
    pid = int(res1.data["merge_post"]["id"])

    # Update: title change only
    upd = "mutation($p: PostQLInput!) { merge_post(payload: $p) { id title } }"
    payload2 = {"id": pid, "title": "Twice"}
    res2 = await schema.execute(
        upd,
        variable_values={"p": payload2},
        context_value={"db_session": db_session, "test_callbacks": True},
    )
    assert res2.errors is None, res2.errors
    title = res2.data["merge_post"]["title"]
    # Should have pre and post markers around new title
    assert title.startswith("[pre]Twice") and title.endswith("[post]")
    # Confirm at least one post event with created=False
    assert any(e.get("event") == "post" and not e.get("created") for e in CALLBACK_EVENTS)
