import pytest
from tests.old.schema import schema


@pytest.fixture
async def graphql_schema():
    return schema

@pytest.fixture
async def graphql_context(db_session, sample_users):
    current_user = sample_users[0] if sample_users else None
    return {'db_session': db_session, 'user_id': 1, 'current_user': current_user}

@pytest.mark.asyncio
async def test_invalid_where_json(graphql_schema, graphql_context):
    # where passed as malformed JSON via root users field (string)
    bad_query = """
    query { users(where: "{invalid") { id } }
    """
    result = await graphql_schema.execute(bad_query, context_value=graphql_context)
    # Expect an error mentioning Invalid where JSON
    assert result.errors, "Should raise error for invalid where JSON"
    assert any('Invalid where JSON' in str(e) for e in result.errors)

@pytest.mark.asyncio
async def test_invalid_order_by_json(graphql_schema, graphql_context):
    bad_query = """
    query { users(orderBy: "{invalid") { id } }
    """
    result = await graphql_schema.execute(bad_query, context_value=graphql_context)
    assert result.errors, "Should raise error for invalid order_by JSON"
    assert any('Invalid order_by JSON' in str(e) for e in result.errors)

@pytest.mark.asyncio
async def test_invalid_where_field(graphql_schema, graphql_context):
    # Use correct JSON but field not in model
    bad_query = """
    query { users(where: "{\\"nonexistentField\\": {\\"eq\\": 1}}") { id } }
    """
    result = await graphql_schema.execute(bad_query, context_value=graphql_context)
    assert result.errors, "Should raise error for invalid field in where"
    assert any('not found in model' in str(e) for e in result.errors)

@pytest.mark.asyncio
async def test_invalid_where_operator(graphql_schema, graphql_context):
    bad_query = """
    query { users(where: "{\\"id\\": {\\"badOp\\": 1}}") { id } }
    """
    result = await graphql_schema.execute(bad_query, context_value=graphql_context)
    assert result.errors, "Should raise error for unsupported operator"
    assert any('Unsupported operator' in str(e) for e in result.errors)
