import pytest
from tests.schema import schema as berry_strawberry_schema


@pytest.mark.asyncio
async def test_posts_commented_by_filters_posts_having_comment_by_user(db_session, populated_db):
    # In fixtures, comments authored by:
    # user2 on post1, post2, post5; user3 on post1, post3; user1 on post3, post4
    users = populated_db['users']
    user1, user2, user3, _ = users

    # Query posts commented by user2
    q = f"""
    query {{
      posts(commented_by: {int(user2.id)}) {{ id title }}
    }}
    """
    res = await berry_strawberry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data is not None
    posts = res.data['posts']
    # Expect post1, post2, post5 as per fixtures
    ids = {int(p['id']) for p in posts}
    expected = {int(populated_db['posts'][0].id), int(populated_db['posts'][1].id), int(populated_db['posts'][4].id)}
    assert expected.issubset(ids)

    # Query posts commented by user1
    q2 = f"""
    query {{
      posts(commented_by: {int(user1.id)}) {{ id title }}
    }}
    """
    res2 = await berry_strawberry_schema.execute(q2, context_value={'db_session': db_session})
    assert res2.errors is None, res2.errors
    assert res2.data is not None
    posts2 = res2.data['posts']
    ids2 = {int(p['id']) for p in posts2}
    expected2 = {int(populated_db['posts'][2].id), int(populated_db['posts'][3].id)}
    assert expected2.issubset(ids2)

    # No posts should be returned for an unknown user id
    q3 = """
    query {
      posts(commented_by: 9999) { id }
    }
    """
    res3 = await berry_strawberry_schema.execute(q3, context_value={'db_session': db_session})
    assert res3.errors is None, res3.errors
    assert res3.data is not None
    assert isinstance(res3.data['posts'], list)
    # Depending on DB, it could be empty; assert that none of the post ids match fixtures when filtering impossible
    assert len(res3.data['posts']) == 0
