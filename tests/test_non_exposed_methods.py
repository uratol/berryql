import pytest


@pytest.mark.asyncio
async def test_regular_methods_not_exposed_on_query(db_session, populated_db):
    from tests.schema import berry_schema
    schema = berry_schema.to_strawberry()
    # Ensure hello exists (control)
    res = await schema.execute("query { hello }", context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    assert res.data == {"hello": "world"}
    # Call to a plain method must fail
    res2 = await schema.execute("query { should_not_be_exposed }", context_value={"db_session": db_session})
    assert res2.errors, "Expected error for non-exposed regular method on Query"


@pytest.mark.asyncio
async def test_regular_methods_not_exposed_on_mutation_root(db_session, populated_db):
    from tests.schema import berry_schema
    schema = berry_schema.to_strawberry()
    # Control: properly exposed mutation works
    m = """
    mutation { create_post_id(title: \"t\", content: \"c\", author_id: 1) }
    """
    ok = await schema.execute(m, context_value={"db_session": db_session})
    assert ok.errors is None, ok.errors
    # Plain method must not be available
    bad = await schema.execute("mutation { should_not_be_mutation }", context_value={"db_session": db_session})
    assert bad.errors, "Expected error for non-exposed regular method on Mutation"


@pytest.mark.asyncio
async def test_regular_methods_not_exposed_in_domain_containers(db_session, populated_db):
    from tests.schema import berry_schema
    schema = berry_schema.to_strawberry()
    # Domain resolver should work
    q_ok = """
    query { groupDomain { blogDomain { posts(limit: 1) { id } } } }
    """
    res = await schema.execute(q_ok, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    # But the plain method on the domain should not appear as a field
    q_bad = """
    query { groupDomain { blogDomain { should_not_be_on_domain } } }
    """
    res2 = await schema.execute(q_bad, context_value={"db_session": db_session})
    assert res2.errors, "Expected error for non-exposed regular method on domain container"
    # And also not in mutation domain container
    m_bad = """
    mutation { blogDomain { should_not_be_on_domain } }
    """
    res3 = await schema.execute(m_bad, context_value={"db_session": db_session})
    assert res3.errors, "Expected error for non-exposed regular method on mutation domain container"
