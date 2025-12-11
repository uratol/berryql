import pytest

from tests.schema import schema, CALLBACK_EVENTS
from tests.fixtures import populated_db  # noqa: F401


@pytest.mark.asyncio
async def test_merge_hooks_async(db_session, populated_db):
    CALLBACK_EVENTS.clear()
    uid = populated_db["users"][0].id
    # Invoke async callbacks via domain field under Mutation
    mutation = (
        "mutation($p: [PostQLInput!]!) { asyncDomain { merge_posts(payload: $p) { id title author_id } } }"
    )
    variables = {
        "p": [{"title": "Async", "content": "Body", "author_id": uid}]
    }
    res = await schema.execute(
        mutation,
        variable_values=variables,
        context_value={"db_session": db_session, "test_callbacks_async": True},
    )
    assert res.errors is None, res.errors
    edge = res.data.get("asyncDomain")
    assert isinstance(edge, dict)
    post_list = edge["merge_posts"]
    assert isinstance(post_list, list)
    assert len(post_list) == 1
    post = post_list[0]
    # Title should have async pre and post markers from decorator-declared async hooks
    assert post["title"].startswith("[apre]Async")
    assert post["title"].endswith("[apost]")
    kinds = [e["event"] for e in CALLBACK_EVENTS]
    assert "apre" in kinds and "apost" in kinds
