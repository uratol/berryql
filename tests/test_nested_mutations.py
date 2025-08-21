import pytest

from tests.schema import schema


@pytest.mark.asyncio
async def test_upsert_post_with_nested_comments_and_likes(db_session, populated_db):
    # Use top-level auto-generated mutation upsert_post with nested relations
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]
    u3 = populated_db['users'][2]

    mutation = (
        """
        mutation Upsert($payload: PostQLInput!) {
          upsert_post(payload: $payload) {
            id
            title
            author_id
            post_comments(order_by: \"id\", order_dir: desc) {
              id
              content
              author_id
              likes(order_by: \"id\") { id user_id }
            }
          }
        }
        """
    )
    variables = {
        "payload": {
            "title": "Nested Create",
            "content": "Body",
            "author_id": int(u1.id),
            "post_comments": [
                {"content": "c1", "author_id": int(u2.id)},
                {
                    "content": "c2",
                    "author_id": int(u3.id),
                    # nested likes for this comment
                    "likes": [
                        {"user_id": int(u1.id)},
                        {"user_id": int(u2.id)}
                    ]
                },
            ],
        }
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    post = res.data["upsert_post"]
    assert post["title"] == "Nested Create"
    assert int(post["author_id"]) == int(u1.id)
    pcs = post["post_comments"]
    assert isinstance(pcs, list) and len(pcs) == 2
    # Find the one with likes and validate
    pc_with_likes = next((x for x in pcs if x.get("content") == "c2"), None)
    assert pc_with_likes is not None
    likes = pc_with_likes["likes"]
    assert isinstance(likes, list) and len(likes) == 2

    # Query back to ensure persistence and FK wiring
    q = """
    query {
      posts(where: "{\\\"id\\\": {\\\"eq\\\": %d}}") {
        id
        post_comments { id content likes { id user_id } }
      }
    }
    """ % int(post["id"])
    res2 = await schema.execute(q, context_value={"db_session": db_session})
    assert res2.errors is None, res2.errors
    posts = res2.data["posts"]
    assert len(posts) == 1
    assert len(posts[0]["post_comments"]) == 2
    # the comment with likes should still have 2 likes
    pc2 = next((x for x in posts[0]["post_comments"] if x.get("content") == "c2"), None)
    assert pc2 is not None
    assert len(pc2["likes"]) == 2


@pytest.mark.asyncio
async def test_mutation_domain_upsert_posts(db_session, populated_db):
    # Upsert via domain-scoped mutation under blogDomain
    u1 = populated_db['users'][0]
    m = (
        """
        mutation($payload: PostQLInput!) {
          blogDomain { 
            upsert_posts(payload: $payload) {
              id
              title
            }
          }
        }
        """
    )
    variables = {
        "payload": {
            "title": "Domain Upsert",
            "content": "From domain",
            "author_id": int(u1.id),
        }
    }
    res = await schema.execute(m, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data["blogDomain"]["upsert_posts"]
    assert data["title"] == "Domain Upsert"
