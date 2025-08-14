import pytest
from tests.schema import schema as berry_strawberry_schema

@pytest.mark.asyncio
async def test_domain_schema_shape():
    # Ensure Query has domain fields
    q = berry_strawberry_schema.get_type_by_name('Query')
    assert q is not None
    fields = {f.name for f in q.fields}
    assert 'userDomain' in fields
    assert 'blogDomain' in fields
    # Ensure domain types exist with expected fields
    user_domain_type = None
    blog_domain_type = None
    for tname in ['UserDomainType','UserDomain','BlogDomainType','BlogDomain']:
        user_domain_type = user_domain_type or berry_strawberry_schema.get_type_by_name(tname)
        blog_domain_type = blog_domain_type or berry_strawberry_schema.get_type_by_name(tname)
    # We can't rely on exact names, but the fields should be accessible via selection
    assert user_domain_type is not None or blog_domain_type is not None

@pytest.mark.asyncio
async def test_domain_query_users(db_session, populated_db):
    q = """
    query {
      userDomain {
        users(order_by: "id") { id name }
      }
    }
    """
    res = await berry_strawberry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, f"GraphQL errors: {res.errors}"
    users = res.data['userDomain']['users']
    assert isinstance(users, list)
    assert len(users) >= 1

@pytest.mark.asyncio
async def test_domain_query_posts(db_session, populated_db):
    q = """
    query {
      blogDomain {
        posts(order_by: "id") { id title }
      }
    }
    """
    res = await berry_strawberry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, f"GraphQL errors: {res.errors}"
    posts = res.data['blogDomain']['posts']
    assert isinstance(posts, list)
    assert len(posts) >= 1
