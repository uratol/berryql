import pytest

from tests.schema import schema, berry_schema

@pytest.mark.asyncio
async def test_polymorphic_views_on_posts(populated_db, db_session):
    # Query posts with their polymorphic views
    q = '''
    query {
      posts(order_by: "id") {
        id
        title
        views { id user_id entity_type entity_id }
      }
    }
    '''
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, f"GraphQL errors: {res.errors}"
    data = res.data["posts"]
    # Post 1 has 2 views in fixtures, Post 2 has 1
    p1 = next(p for p in data if int(p["id"]) == int(populated_db['posts'][0].id))
    p2 = next(p for p in data if int(p["id"]) == int(populated_db['posts'][1].id))
    assert len(p1["views"]) == 2
    assert len(p2["views"]) == 1
    for v in p1["views"]:
        assert v["entity_type"] == "post"
        assert int(v["entity_id"]) == int(populated_db['posts'][0].id)

@pytest.mark.asyncio
async def test_polymorphic_views_on_comments(populated_db, db_session):
    q = '''
    query {
      posts(order_by: "id") {
        id
        post_comments(order_by: "id") { id views { id user_id entity_type entity_id } }
      }
    }
    '''
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, f"GraphQL errors: {res.errors}"
    data = res.data["posts"]
    # First post has two comments; the first two comments have views seeded
    first_post = next(p for p in data if int(p["id"]) == int(populated_db['posts'][0].id))
    comments = first_post["post_comments"]
    assert len(comments) >= 2
    assert len(comments[0]["views"]) >= 1
    for v in comments[0]["views"]:
        assert v["entity_type"] == "post_comment"
        assert int(v["entity_id"]) == int(populated_db['post_comments'][0].id)
