
import pytest
from tests.schema import schema
from tests.models import Post, User

@pytest.mark.asyncio
async def test_cascade_set_null_persistent(db_session, populated_db):
    # Create a reviewer user
    reviewer = User(name="Reviewer", email="reviewer@example.com", is_admin=True)
    db_session.add(reviewer)
    await db_session.flush()
    
    # Create a post with a reviewer
    post = Post(title="Reviewed Post", content="Content", author_id=populated_db['users'][0].id, reviewer_id=reviewer.id)
    db_session.add(post)
    await db_session.commit()
    
    # Verify post has reviewer
    post_id = post.id
    reviewer_id = reviewer.id
    
    p = await db_session.get(Post, post_id)
    assert p.reviewer_id == reviewer_id
    
    # Delete the reviewer using mutation
    query = """
    mutation($id: Int!) {
        merge_users(payload: {id: $id, _Delete: true}) {
            id
        }
    }
    """
    
    res = await schema.execute(query, variable_values={"id": reviewer_id}, context_value={"db_session": db_session})
    assert res.errors is None
    
    # Verify reviewer is deleted
    db_session.expire_all()
    assert await db_session.get(User, reviewer_id) is None
    
    # Verify post still exists and reviewer_id is NULL
    p_ref = await db_session.get(Post, post_id)
    assert p_ref is not None
    assert p_ref.reviewer_id is None
