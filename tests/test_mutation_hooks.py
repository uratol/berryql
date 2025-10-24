import pytest

from tests.schema import schema, CALLBACK_EVENTS
from tests.fixtures import populated_db  # noqa: F401


@pytest.mark.asyncio
async def test_merge_hooks_create(db_session, populated_db):
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
    # Both decorator and descriptor hooks should have fired. Check presence and relative positions.
    title = post["title"]
    assert all(tag in title for tag in ("[pre]", "[hpre]", "[post]", "[hpost]", "[h2post]"))
    # Ensure base word appears after pre markers and before post markers
    base_ix = title.find("New")
    assert base_ix != -1
    assert max(title.find("[pre]"), title.find("[hpre]")) < base_ix
    assert base_ix < min(title.rfind("[post]"), title.rfind("[hpost]"), title.rfind("[h2post]"))
    # Ensure both pre and post were logged
    kinds = [e["event"] for e in CALLBACK_EVENTS]
    # Expect both declaration methods' events, including array-based hooks
    for k in ("pre", "post", "hpre", "hpost", "h2post"):
        assert k in kinds


@pytest.mark.asyncio
async def test_merge_hooks_update(db_session, populated_db):
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
    # Should have both pre and post markers from decorators and descriptors (two descriptor hooks)
    assert all(tag in title for tag in ("[pre]", "[hpre]", "[post]", "[hpost]", "[h2post]"))
    base_ix = title.find("Twice")
    assert base_ix != -1
    assert max(title.find("[pre]"), title.find("[hpre]")) < base_ix
    assert base_ix < min(title.rfind("[post]"), title.rfind("[hpost]"), title.rfind("[h2post]"))
    # Confirm post events (all kinds) with created=False present
    assert any(e.get("event") in {"post", "hpost", "h2post"} and not e.get("created") for e in CALLBACK_EVENTS)


@pytest.mark.asyncio
async def test_merge_hooks_nested_relation(db_session, populated_db):
    # Verify hooks fire when creating a nested Post under a User merge
    CALLBACK_EVENTS.clear()
    uid = populated_db["users"][0].id

    # Merge a user with a nested posts payload; expect Post hooks to run
    mutation = (
        "mutation($p: [UserQLInput!]!) { "
        "  merge_users(payload: $p) { id name posts { id title author_id } } "
        "}"
    )
    variables = {
        "p": [
            {
                "id": uid,
                "posts": [
                    {"title": "New", "content": "Body", "author_id": uid}
                ],
            }
        ]
    }

    res = await schema.execute(
        mutation,
        variable_values=variables,
        context_value={"db_session": db_session, "test_callbacks": True},
    )
    assert res.errors is None, res.errors

    user = res.data["merge_users"]
    assert user["id"] == str(uid) or user["id"] == uid
    posts = user.get("posts") or []
    assert isinstance(posts, list) and len(posts) >= 1
    title = posts[0]["title"]

    # Hooks from decorators and descriptor should have annotated the title
    assert all(tag in title for tag in ("[pre]", "[hpre]", "[post]", "[hpost]", "[h2post]"))
    base_ix = title.find("New")
    assert base_ix != -1
    assert max(title.find("[pre]"), title.find("[hpre]")) < base_ix
    assert base_ix < min(title.rfind("[post]"), title.rfind("[hpost]"), title.rfind("[h2post]"))

    # Ensure both pre and post events were logged for the nested Post
    kinds = [e.get("event") for e in CALLBACK_EVENTS]
    for k in ("pre", "post", "hpre", "hpost", "h2post"):
        assert k in kinds
