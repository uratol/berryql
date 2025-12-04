import pytest
from tests.schema import schema
from tests.models import User, Post

@pytest.mark.asyncio
async def test_users_strawberry_field(db_session):
    # Setup data
    u1 = User(name="User 1", email="u1@example.com")
    u2 = User(name="User 2", email="u2@example.com")
    db_session.add_all([u1, u2])
    await db_session.flush()
    
    p1 = Post(title="Post 1", content="Content 1", author_id=u1.id)
    p2 = Post(title="Post 2", content="Content 2", author_id=u1.id)
    p3 = Post(title="Post 3", content="Content 3", author_id=u2.id)
    db_session.add_all([p1, p2, p3])
    await db_session.commit()

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
    
    result = await schema.execute(query, context_value={"db_session": db_session})
    assert result.errors is None
    data = result.data['userDomain']['users_strawberry']
    
    assert len(data) == 2
    # Sort by id to ensure stable access or find by id
    data.sort(key=lambda x: int(x['id']))
    
    u1_data = next(u for u in data if int(u['id']) == u1.id)
    u2_data = next(u for u in data if int(u['id']) == u2.id)
    
    assert u1_data['name'] == "User 1"
    assert len(u1_data['posts']) == 2
    post_titles_1 = set(p['title'] for p in u1_data['posts'])
    assert post_titles_1 == {"Post 1", "Post 2"}
    
    assert u2_data['name'] == "User 2"
    assert len(u2_data['posts']) == 1
    assert u2_data['posts'][0]['title'] == "Post 3"

@pytest.mark.asyncio
async def test_users_berryql_field(db_session):
    # Setup data
    u1 = User(name="User 1", email="u1@example.com")
    u2 = User(name="User 2", email="u2@example.com")
    db_session.add_all([u1, u2])
    await db_session.flush()
    
    p1 = Post(title="Post 1", content="Content 1", author_id=u1.id)
    p2 = Post(title="Post 2", content="Content 2", author_id=u1.id)
    p3 = Post(title="Post 3", content="Content 3", author_id=u2.id)
    db_session.add_all([p1, p2, p3])
    await db_session.commit()

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
    
    result = await schema.execute(query, context_value={"db_session": db_session})
    assert result.errors is None
    data = result.data['userDomain']['users_berryql']
    
    assert len(data) == 2
    # Sort by id to ensure stable access or find by id
    data.sort(key=lambda x: int(x['id']))
    
    u1_data = next(u for u in data if int(u['id']) == u1.id)
    u2_data = next(u for u in data if int(u['id']) == u2.id)
    
    assert u1_data['name'] == "User 1"
    assert len(u1_data['posts']) == 2
    post_titles_1 = set(p['title'] for p in u1_data['posts'])
    assert post_titles_1 == {"Post 1", "Post 2"}
    
    assert u2_data['name'] == "User 2"
    assert len(u2_data['posts']) == 1
    assert u2_data['posts'][0]['title'] == "Post 3"
