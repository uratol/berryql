import pytest

from tests.schema import schema


@pytest.mark.asyncio
async def test_upsert_post_with_nested_comments_and_likes(db_session, populated_db):
  # Use top-level auto-generated mutation merge_posts with nested relations
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]
    u3 = populated_db['users'][2]

    mutation = (
        """
  mutation Upsert($payload: [PostQLInput!]!) {
          merge_posts(payload: $payload) {
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
    "payload": [{
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
    }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    post_list = res.data["merge_posts"]
    assert isinstance(post_list, list)
    assert len(post_list) == 1
    post = post_list[0]
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
async def test_nested_comment_merge_coerces_iso_datetime_strings(db_session, populated_db):
    u1 = populated_db["users"][0]
    u2 = populated_db["users"][1]
    created_iso = "2026-03-19T12:34:56"
    updated_iso = "2026-03-20T08:09:10"

    create_mutation = """
        mutation Upsert($payload: PostQLInput!) {
            merge_post(payload: $payload) {
                id
                post_comments(order_by: "id") {
                    id
                    content
                    created_at
                }
            }
        }
    """
    create_variables = {
        "payload": {
            "title": "Datetime Nested Create",
            "content": "Body",
            "author_id": int(u1.id),
            "post_comments": [
                {
                    "content": "dt comment",
                    "author_id": int(u2.id),
                    "created_at": created_iso,
                }
            ],
        }
    }
    res1 = await schema.execute(create_mutation, variable_values=create_variables, context_value={"db_session": db_session})
    assert res1.errors is None, res1.errors
    post = res1.data["merge_post"]
    comment = next((row for row in post["post_comments"] if row["content"] == "dt comment"), None)
    assert comment is not None
    assert str(comment["created_at"]).startswith(created_iso)

    post_id = int(post["id"])
    comment_id = int(comment["id"])

    update_mutation = """
        mutation Upsert($payload: PostQLInput!) {
            merge_post(payload: $payload) {
                id
                post_comments(order_by: "id") {
                    id
                    content
                    created_at
                }
            }
        }
    """
    update_variables = {
        "payload": {
            "id": post_id,
            "post_comments": [
                {
                    "id": comment_id,
                    "content": "dt comment updated",
                    "created_at": updated_iso,
                }
            ],
        }
    }
    res2 = await schema.execute(update_mutation, variable_values=update_variables, context_value={"db_session": db_session})
    assert res2.errors is None, res2.errors
    updated_comment = next((row for row in res2.data["merge_post"]["post_comments"] if int(row["id"]) == comment_id), None)
    assert updated_comment is not None
    assert updated_comment["content"] == "dt comment updated"
    assert str(updated_comment["created_at"]).startswith(updated_iso)

    query = """
    query {
      posts(where: "{\\\"id\\\": {\\\"eq\\\": %d}}") {
        id
        post_comments(order_by: \"id\") {
          id
          content
          created_at
        }
      }
    }
    """ % post_id
    res3 = await schema.execute(query, context_value={"db_session": db_session})
    assert res3.errors is None, res3.errors
    persisted_comment = next(
        (row for row in res3.data["posts"][0]["post_comments"] if int(row["id"]) == comment_id),
        None,
    )
    assert persisted_comment is not None
    assert persisted_comment["content"] == "dt comment updated"
    assert str(persisted_comment["created_at"]).startswith(updated_iso)


@pytest.mark.asyncio
async def test_mutation_domain_upsert_posts(db_session, populated_db):
    # Upsert via domain-scoped mutation under blogDomain
    u1 = populated_db['users'][0]
    m = (
        """
    mutation($payload: [PostQLInput!]!) {
          blogDomain { 
            merge_posts(payload: $payload) {
              id
              title
            }
          }
        }
        """
    )
    variables = {
    "payload": [{
            "title": "Domain Upsert",
            "content": "From domain",
            "author_id": int(u1.id),
    }]
    }
    res = await schema.execute(m, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data_list = res.data["blogDomain"]["merge_posts"]
    assert isinstance(data_list, list)
    assert len(data_list) == 1
    data = data_list[0]
    assert data["title"] == "Domain Upsert"
