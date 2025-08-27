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


async def test_posts_query_returns_enum_names(db_session, populated_db):
    # Query posts and ensure status is a GraphQL enum (returned as its name string)
    query = (
        "query { posts { id status } }"
    )
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    rows = res.data["posts"]
    assert isinstance(rows, list)
    # Build id -> status mapping
    # Strawberry returns Python Enum instances in ExecutionResult.data; compare by .name
    got = {int(r["id"]): r["status"].name if hasattr(r["status"], "name") else r["status"] for r in rows}
    # Seeded fixtures produce deterministic statuses
    # 1: PUBLISHED, 2: PUBLISHED, 3: DRAFT, 4: PUBLISHED, 5: ARCHIVED
    assert got[1] == "PUBLISHED"
    assert got[2] == "PUBLISHED"
    assert got[3] == "DRAFT"
    assert got[4] == "PUBLISHED"
    assert got[5] == "ARCHIVED"


async def test_merge_post_enum_create_and_update(db_session, populated_db):
    # Create with explicit enum in payload
    mutation_create = (
        "mutation Upsert($payload: PostQLInput!) {\n"
        "  merge_post(payload: $payload) { id title status author_id }\n"
        "}"
    )
    variables = {
        "payload": {"title": "Enum Post", "content": "C2", "author_id": 1, "status": "PUBLISHED"}
    }
    res1 = await schema.execute(mutation_create, variable_values=variables, context_value={"db_session": db_session})
    assert res1.errors is None, res1.errors
    p = res1.data["merge_post"]
    pid = int(p["id"])
    assert (p["status"].name if hasattr(p["status"], "name") else p["status"]) == "PUBLISHED"

    # Update the same post to ARCHIVED via enum
    mutation_update = (
        "mutation Upsert($payload: PostQLInput!) {\n"
        "  merge_post(payload: $payload) { id status }\n"
        "}"
    )
    variables2 = {"payload": {"id": pid, "status": "ARCHIVED"}}
    res2 = await schema.execute(mutation_update, variable_values=variables2, context_value={"db_session": db_session})
    assert res2.errors is None, res2.errors
    p2 = res2.data["merge_post"]
    assert int(p2["id"]) == pid
    assert (p2["status"].name if hasattr(p2["status"], "name") else p2["status"]) == "ARCHIVED"

    # Verify via query
    query = "query { posts { id status } }"
    res3 = await schema.execute(query, context_value={"db_session": db_session})
    assert res3.errors is None, res3.errors
    found = next((r for r in res3.data["posts"] if int(r["id"]) == pid), None)
    assert found is not None
    assert (found["status"].name if hasattr(found["status"], "name") else found["status"]) == "ARCHIVED"
