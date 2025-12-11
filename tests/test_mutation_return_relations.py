import pytest
from tests.schema import schema

@pytest.mark.asyncio
async def test_mutation_single_payload_single_relation(db_session, populated_db):
    """Case 1: Single Mutation -> Single Relation"""
    u1 = populated_db['users'][0]
    
    # merge_post is defined with single=True
    mutation = """
        mutation Upsert($payload: PostQLInput!) {
            merge_post(payload: $payload) {
                id
                title
                author {
                    id
                    name
                }
            }
        }
    """
    
    variables = {
        "payload": {
            "title": "Single Post Single Rel",
            "content": "Content",
            "author_id": int(u1.id)
        }
    }
    
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    post = res.data["merge_post"]
    assert isinstance(post, dict)
    assert post["title"] == "Single Post Single Rel"
    assert post["author"]["name"] == u1.name

@pytest.mark.asyncio
async def test_mutation_single_payload_multiple_relation(db_session, populated_db):
    """Case 2: Single Mutation -> Multiple Relation"""
    u1 = populated_db['users'][0]
    
    # Create post with comments first
    mutation_create = """
        mutation Upsert($payload: PostQLInput!) {
            merge_post(payload: $payload) {
                id
            }
        }
    """
    variables_create = {
        "payload": {
            "title": "Single Post Multi Rel",
            "content": "Content",
            "author_id": int(u1.id),
            "post_comments": [
                {"content": "c1", "author_id": int(u1.id)},
                {"content": "c2", "author_id": int(u1.id)}
            ]
        }
    }
    res = await schema.execute(mutation_create, variable_values=variables_create, context_value={"db_session": db_session})
    assert res.errors is None
    post_id = res.data["merge_post"]["id"]

    # Update and fetch comments
    mutation = """
        mutation Upsert($payload: PostQLInput!) {
            merge_post(payload: $payload) {
                id
                post_comments {
                    id
                    content
                }
            }
        }
    """
    
    variables = {
        "payload": {
            "id": post_id,
            "title": "Single Post Multi Rel Updated"
        }
    }
    
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    post = res.data["merge_post"]
    assert isinstance(post, dict)
    assert len(post["post_comments"]) == 2
    assert post["post_comments"][0]["content"] in ["c1", "c2"]

@pytest.mark.asyncio
async def test_mutation_multiple_payload_single_relation(db_session, populated_db):
    """Case 3: Multiple Mutation -> Single Relation"""
    u1 = populated_db['users'][0]
    
    # merge_posts is defined with single=False (default)
    mutation = """
        mutation Upsert($payload: [PostQLInput!]!) {
            merge_posts(payload: $payload) {
                id
                title
                author {
                    id
                    name
                }
            }
        }
    """
    
    variables = {
        "payload": [
            {
                "title": "Multi Post 1",
                "content": "Content 1",
                "author_id": int(u1.id)
            },
            {
                "title": "Multi Post 2",
                "content": "Content 2",
                "author_id": int(u1.id)
            }
        ]
    }
    
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["merge_posts"]
    assert isinstance(posts, list)
    assert len(posts) == 2
    
    assert posts[0]["title"] == "Multi Post 1"
    assert posts[0]["author"]["name"] == u1.name
    
    assert posts[1]["title"] == "Multi Post 2"
    assert posts[1]["author"]["name"] == u1.name

@pytest.mark.asyncio
async def test_mutation_multiple_payload_multiple_relation(db_session, populated_db):
    """Case 4: Multiple Mutation -> Multiple Relation"""
    u1 = populated_db['users'][0]
    
    mutation = """
        mutation Upsert($payload: [PostQLInput!]!) {
            merge_posts(payload: $payload) {
                id
                title
                post_comments {
                    id
                    content
                }
            }
        }
    """
    
    variables = {
        "payload": [
            {
                "title": "Multi Post Multi Rel 1",
                "content": "Content 1",
                "author_id": int(u1.id),
                "post_comments": [{"content": "p1c1", "author_id": int(u1.id)}]
            },
            {
                "title": "Multi Post Multi Rel 2",
                "content": "Content 2",
                "author_id": int(u1.id),
                "post_comments": [{"content": "p2c1", "author_id": int(u1.id)}, {"content": "p2c2", "author_id": int(u1.id)}]
            }
        ]
    }
    
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["merge_posts"]
    assert isinstance(posts, list)
    assert len(posts) == 2
    
    assert posts[0]["title"] == "Multi Post Multi Rel 1"
    assert len(posts[0]["post_comments"]) == 1
    assert posts[0]["post_comments"][0]["content"] == "p1c1"
    
    assert posts[1]["title"] == "Multi Post Multi Rel 2"
    assert len(posts[1]["post_comments"]) == 2
