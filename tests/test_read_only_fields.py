import pytest

from tests.schema import schema

pytestmark = pytest.mark.asyncio


async def test_read_only_fields_excluded_from_mutation_inputs():
    # Introspect PostQLInput and ensure read-only fields are not present
    sdl_query = """
    { __type(name: "PostQLInput") { name inputFields { name } } }
    """
    res = await schema.execute(sdl_query)
    assert res.errors is None, res.errors
    fields = {f["name"] for f in res.data["__type"]["inputFields"]}
    # created_at and content_length were marked read_only
    assert "created_at" not in fields
    assert "content_length" not in fields
    # writable fields still present
    assert "title" in fields and "content" in fields and "author_id" in fields


async def test_read_only_fields_still_queryable(db_session, populated_db):
    # Ensure read-only fields remain on query output types
    q = """
    query {
      posts { id created_at content_length }
    }
    """
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    rows = res.data["posts"]
    assert isinstance(rows, list)
    # All rows should have created_at, and content_length may be int (>=0)
    assert all("created_at" in r for r in rows)


# Domain mutation inputs are validated by GraphQL against PostQLInput shape; we already
# assert PostQLInput excludes read-only fields above, and below we validate runtime response.


async def test_domain_merge_posts_response_includes_read_only_fields(db_session, populated_db):
    # Create a post via domain mutation and ensure created_at is returned
    m = (
    "mutation Upsert($payload: [PostQLInput!]!) {\n"
    "  blogDomain {\n"
    "    merge_posts(payload: $payload) { id title created_at }\n"
    "  }\n"
    "}"
    )
    variables = {"payload": [{"title": "ROnly", "content": "Body", "author_id": 1}]}
    res = await schema.execute(m, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    obj = res.data["blogDomain"]["merge_posts"][0]
    assert obj["title"] == "ROnly"
    # created_at should be present (non-null string or datetime-like)
    assert obj["created_at"] is not None


async def test_domain_mutation_rejects_read_only_fields_in_payload():
    # Supplying read-only fields in the domain mutation payload should be rejected
    mutation = (
    "mutation($p: [PostQLInput!]!) {\n"
    "  blogDomain { merge_posts(payload: $p) { id } }\n"
    "}"
    )
    variables = {
    "p": [
        {
        "title": "X",
        "content": "Y",
        "author_id": 1,
        # read-only fields should not be accepted by schema
        "created_at": "2020-01-01T00:00:00",
        "content_length": 42,
        }
    ]
    }
    res = await schema.execute(mutation, variable_values=variables)
    # Expect validation error mentioning the unknown fields on PostQLInput
    assert res.errors, "Expected validation error for read-only fields in input"
    msg = "\n".join(str(e) for e in res.errors)
    assert "PostQLInput" in msg and "created_at" in msg


async def test_root_merge_posts_response_includes_read_only_fields(db_session, populated_db):
    # Ensure root merge returns read-only fields like created_at
    m = (
    "mutation($p: [PostQLInput!]!) {\n"
    "  merge_posts(payload: $p) { id title created_at }\n"
    "}"
    )
    variables = {"p": [{"title": "RRoot", "content": "Body", "author_id": 1}]}
    res = await schema.execute(m, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    obj = res.data["merge_posts"][0]
    assert obj["title"] == "RRoot"
    assert obj["created_at"] is not None
