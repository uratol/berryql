import pytest
from tests.schema import schema

@pytest.mark.asyncio
async def test_mutation_domain_create_post_and_query(db_session, populated_db):
    # Use mutation domains to create and then fetch via query
    m = """
    mutation {
      create_post(title: "FromDomain", content: "Body", author_id: 1) { id title author_id }
    }
    """
    res = await schema.execute(m, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    post = res.data["create_post"]
    assert post["title"] == "FromDomain"
    pid = int(post["id"])  # noqa: F841

    # Query via group domain to ensure domains and mutations co-exist
    q = """
    query {
      groupDomain { blogDomain { posts(limit: 1, order_by: "id") { id title } } }
    }
    """
    res2 = await schema.execute(q, context_value={"db_session": db_session})
    assert res2.errors is None, res2.errors
    gd = res2.data["groupDomain"]["blogDomain"]["posts"]
    assert isinstance(gd, list)


@pytest.mark.asyncio
async def test_domain_scoped_mutation(db_session, populated_db):
    m = """
    mutation {
      blogDomain { create_post_mut(title: "ByDomain", content: "Body", author_id: 1) { id title author_id } }
    }
    """
    res = await schema.execute(m, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data["blogDomain"]["create_post_mut"]
    assert data["title"] == "ByDomain"
