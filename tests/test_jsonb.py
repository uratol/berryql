import os
import pytest
from tests.schema import schema

pytestmark = pytest.mark.asyncio

pg_only = pytest.mark.skipif(os.getenv('BERRYQL_TEST_DATABASE_URL','').startswith('postgresql') is False, reason='requires Postgres URL')

@pg_only
async def test_query_posts_metadata_json_postgres(db_session, sample_posts):
    q = """
    query { posts(order_by: "id") { id metadata_json } }
    """
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None
    posts = res.data["posts"]
    # Expect JSON objects/values as parsed JSON scalars
    assert posts[0]["metadata_json"] == {"tags": ["intro", "hello"], "views": 10}
    assert posts[1]["metadata_json"] == {"tags": ["graphql"], "rating": 4.5}
    assert posts[2]["metadata_json"] is None
    assert posts[3]["metadata_json"] == {"flags": {"featured": True}, "count": 3}
    assert posts[4]["metadata_json"] == {}

@pg_only
async def test_mutation_posts_metadata_json_write_and_read(db_session, sample_users):
    m = """
    mutation($p: [PostQLInput!]!) {
      merge_posts(payload: $p) { id metadata_json title author_id }
    }
    """
    vars = {
        "p": [
            {"title": "JSONB New", "content": "c", "author_id": sample_users[0].id, "metadata_json": {"a": 1, "b": [1,2,3]}},
        ]
    }
    r = await schema.execute(m, variable_values=vars, context_value={"db_session": db_session})
    assert r.errors is None
    ps = r.data["merge_posts"]
    # merge_posts returns the last merged object when payload is a list
    assert isinstance(ps, dict)
    got = ps
    assert got["metadata_json"] == {"a": 1, "b": [1,2,3]}

    # Query back
    q = """
    query { posts(order_by: "id", where: "{\\\"title\\\": {\\\"eq\\\": \\\"JSONB New\\\"}}") { id metadata_json } }
    """
    r2 = await schema.execute(q, context_value={"db_session": db_session})
    assert r2.errors is None
    rows = r2.data["posts"]
    assert len(rows) == 1
    assert rows[0]["metadata_json"] == {"a": 1, "b": [1,2,3]}
