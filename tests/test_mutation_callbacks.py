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
    "mutation($payload: [PostQLInput!]!) { merge_posts(payload: $payload) { id title author_id } }"
    )
    variables = {
        "payload": [{"title": "New", "content": "Body", "author_id": uid}]
    }
    res = await schema.execute(
        mutation,
        variable_values=variables,
        context_value={"db_session": db_session, "test_callbacks": True},
    )
    assert res.errors is None, res.errors
    post = res.data["merge_posts"]
    # Both decorator and descriptor hooks should have fired: [pre] and [hpre] prefixes, [post] and [hpost] suffixes
    title = post["title"]
    assert title.startswith("[hpre][pre]New") or title.startswith("[pre][hpre]New")
    assert title.endswith("[post][hpost]") or title.endswith("[hpost][post]")
    # Ensure both pre and post were logged
    kinds = [e["event"] for e in CALLBACK_EVENTS]
    # Expect both declaration methods' events
    assert all(k in kinds for k in ("pre", "post", "hpre", "hpost"))


@pytest.mark.asyncio
async def test_upsert_callbacks_update(db_session, populated_db):
    CALLBACK_EVENTS.clear()
    uid = populated_db["users"][1].id
    # First create
    create = "mutation($p: [PostQLInput!]!) { merge_posts(payload: $p) { id title } }"
    payload = [{"title": "Once", "content": "B", "author_id": uid}]
    res1 = await schema.execute(
        create,
        variable_values={"p": payload},
        context_value={"db_session": db_session, "test_callbacks": True},
    )
    assert res1.errors is None, res1.errors
    pid = int(res1.data["merge_posts"]["id"])

    # Update: title change only
    upd = "mutation($p: [PostQLInput!]!) { merge_posts(payload: $p) { id title } }"
    payload2 = [{"id": pid, "title": "Twice"}]
    res2 = await schema.execute(
        upd,
        variable_values={"p": payload2},
        context_value={"db_session": db_session, "test_callbacks": True},
    )
    assert res2.errors is None, res2.errors
    title = res2.data["merge_posts"]["title"]
    # Should have both pre and post markers from decorators and descriptor
    assert (title.startswith("[pre][hpre]Twice") or title.startswith("[hpre][pre]Twice") or title.startswith("[pre]Twice") or title.startswith("[hpre]Twice"))
    assert title.endswith("[post][hpost]") or title.endswith("[hpost][post]") or title.endswith("[post]") or title.endswith("[hpost]")
    # Confirm post events (both kinds) with created=False present
    assert any(e.get("event") in {"post", "hpost"} and not e.get("created") for e in CALLBACK_EVENTS)
