import pytest
from tests.schema import schema

@pytest.mark.asyncio
async def test_users_strawberry_native_field(db_session, populated_db):
    query = """
    query {
        userDomain {
            users_strawberry {
                id
                name
                posts {
                    id
                    title
                }
            }
        }
    }
    """
    result = await schema.execute(query, context_value={'db_session': db_session})
    assert result.errors is None, result.errors
    data = result.data['userDomain']['users_strawberry']
    assert len(data) > 0
    
    # Verify posts are present (populated_db should have posts)
    # We need to check if at least one user has posts
    has_posts = False
    for user in data:
        if user['posts']:
            has_posts = True
            # Verify post fields
            assert 'id' in user['posts'][0]
            assert 'title' in user['posts'][0]
            break
    assert has_posts, "No posts found for any user"

@pytest.mark.asyncio
async def test_users_berryql_native_field(db_session, populated_db):
    query = """
    query {
        userDomain {
            users_berryql {
                id
                name
                posts {
                    id
                    title
                }
            }
        }
    }
    """
    result = await schema.execute(query, context_value={'db_session': db_session})
    assert result.errors is None, result.errors
    data = result.data['userDomain']['users_berryql']
    assert len(data) > 0
    
    # Verify posts are present (populated_db should have posts)
    # We need to check if at least one user has posts
    has_posts = False
    for user in data:
        if user['posts']:
            has_posts = True
            # Verify post fields
            assert 'id' in user['posts'][0]
            assert 'title' in user['posts'][0]
            break
    assert has_posts, "No posts found for any user"
