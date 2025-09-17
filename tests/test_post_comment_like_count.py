import pytest
from tests.schema import schema
from tests.fixtures import *  # noqa: F401,F403

# Expected likes distribution from fixtures:
# comments[0]: 2 likes (user1, user3)
# comments[1]: 1 like  (user1)
# comments[2]: 1 like  (user3)
# comments[3]: 0
# comments[4]: 0
# comments[5]: 0
# comments[6]: 0

@pytest.mark.asyncio
async def test_post_comment_like_count_field(db_session, populated_db):
    # Simpler query: rely on default ordering configured on relation; no extraneous variables
    query = """
    query {
      users(limit: 1) {  # fetch first user (admin) who authored first posts
        id
        posts(limit: 2) {
          id
          post_comments { id like_count likes { id } }
        }
      }
    }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data
    assert 'users' in data
    posts_data = data['users'][0]['posts'] if data['users'] else []
    # Flatten comments from returned posts
    all_comments = []
    for p in posts_data:
        pcs = p.get('post_comments') or []
        all_comments.extend(pcs)
        # Ensure each comment has integer like_count >= 0
        for c in pcs:
            assert isinstance(c['like_count'], int)
            assert c['like_count'] >= 0
    # We seeded 2 posts (limit 2) - first two posts have comment indices:
    # post1: comments[0], comments[1]
    # post2: comments[2]
    # So collected comments should be at least 3 (depending on ordering filters applied)
    assert len(all_comments) >= 3
    # Build mapping for diagnostics
    like_counts = {c['id']: c['like_count'] for c in all_comments}
    # Basic invariant: like_count integer >=0
    assert all(isinstance(v, int) and v >= 0 for v in like_counts.values())
    # At least one comment actually has likes relation length > 0
    has_likes = False
    for p in posts_data:
        for c in (p.get('post_comments') or []):
            likes_list = c.get('likes') or []
            if likes_list:
                has_likes = True
    assert has_likes, "Expected at least one comment with likes relation items"
    # Ensure like_count itself reflects at least one non-zero count
    assert any(c['like_count'] > 0 for c in all_comments), "Expected at least one comment with like_count > 0"
