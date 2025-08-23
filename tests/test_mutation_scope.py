import pytest

from tests.schema import schema
import asyncio


@pytest.mark.asyncio
async def test_domain_mutation_scope_async_guard(db_session, populated_db):
    # Set an async domain guard on BlogDomain to allow only author_id == 1
    from tests.schema import BlogDomain, schema as berry_schema
    orig_guard = getattr(BlogDomain, "__domain_guard__", None)

    async def _guard(model_cls, info):
        await asyncio.sleep(0)
        return {"author_id": {"eq": 1}}

    try:
        setattr(BlogDomain, "__domain_guard__", _guard)
        m = (
            """
            mutation($p: [PostQLInput!]!) {
              blogDomain { merge_posts(payload: $p) { id title author_id } }
            }
            """
        )
        # Out-of-scope: author_id = 2
        res1 = await berry_schema.execute(
            m,
            variable_values={"p": [{"title": "A", "content": "B", "author_id": 2}]},
            context_value={"db_session": db_session},
        )
        assert res1.errors is not None, "expected scope violation error for async guard"
        assert "out of scope" in str(res1.errors[0]).lower()

        # In-scope: author_id = 1
        res2 = await berry_schema.execute(
            m,
            variable_values={"p": [{"title": "C", "content": "D", "author_id": 1}]},
            context_value={"db_session": db_session},
        )
        assert res2.errors is None, res2.errors
        data = res2.data["blogDomain"]["merge_posts"]
        assert int(data["author_id"]) == 1
    finally:
        # Restore guard
        if orig_guard is None:
            try:
                delattr(BlogDomain, "__domain_guard__")
            except Exception:
                pass
        else:
            setattr(BlogDomain, "__domain_guard__", orig_guard)


@pytest.mark.asyncio
async def test_nested_mutation_scope_rejects_out_of_scope_like(db_session, populated_db):
    # admin_likes on PostCommentQL has scope: user_id == 1; try to create with user_id != 1
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]

    m = (
        """
                mutation Upsert($payload: [PostQLInput!]!) {
                    merge_posts(payload: $payload) { id title }
        }
        """
    )
    variables = {
        "payload": [{
            "title": "Scoped Create",
            "content": "Body",
            "author_id": int(u1.id),
            "post_comments": [
                {
                    "content": "c1",
                    "author_id": int(u2.id),
                    # Attempt to create via admin_likes with a non-admin user id
                    "admin_likes": [
                        {"user_id": int(u2.id)}
                    ]
                }
            ],
        }]
    }
    res = await schema.execute(m, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is not None, "expected scope violation error"
    msg = str(res.errors[0]) if res.errors else ""
    assert "out of scope" in msg.lower()


@pytest.mark.asyncio
async def test_nested_mutation_scope_allows_in_scope_like(db_session, populated_db):
    # admin_likes allows user_id == 1
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]

    m = (
        """
                mutation Upsert($payload: [PostQLInput!]!) {
                    merge_posts(payload: $payload) {
            id
            post_comments { id content admin_likes { id user_id } }
          }
        }
        """
    )
    variables = {
        "payload": [{
            "title": "Scoped OK",
            "content": "Body",
            "author_id": int(u1.id),
            "post_comments": [
                {
                    "content": "c1",
                    "author_id": int(u2.id),
                    "admin_likes": [
                        {"user_id": int(u1.id)}
                    ]
                }
            ],
        }]
    }
    res = await schema.execute(m, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    assert res.data is not None, "No data returned from mutation"
    post = res.data["merge_posts"]
    pcs = post["post_comments"]
    assert isinstance(pcs, list) and len(pcs) == 1
    likes = pcs[0]["admin_likes"]
    assert isinstance(likes, list) and len(likes) == 1
    assert int(likes[0]["user_id"]) == int(u1.id)


@pytest.mark.asyncio
async def test_nested_mutation_scope_update_rejected(db_session, populated_db):
    # Create in-scope like, then try to update it out of scope (user_id -> 2)
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]

    # First create a post with one comment and one admin_like (user_id=1)
    create_m = (
        """
                mutation Upsert($payload: [PostQLInput!]!) {
                    merge_posts(payload: $payload) {
            id
            post_comments { id content admin_likes { id user_id } }
          }
        }
        """
    )
    create_vars = {
        "payload": [{
            "title": "Scoped Update Base",
            "content": "Body",
            "author_id": int(u1.id),
            "post_comments": [
                {
                    "content": "c1",
                    "author_id": int(u2.id),
                    "admin_likes": [
                        {"user_id": int(u1.id)}
                    ]
                }
            ],
        }]
    }
    res1 = await schema.execute(create_m, variable_values=create_vars, context_value={"db_session": db_session})
    assert res1.errors is None, res1.errors
    assert res1.data is not None, "No data returned from mutation"
    post = res1.data["merge_posts"]
    pc = post["post_comments"][0]
    like_id = int(pc["admin_likes"][0]["id"])

    # Now attempt to update that like out of scope
    upd_m = (
        """
                mutation Upsert($payload: [PostQLInput!]!) {
                    merge_posts(payload: $payload) { id }
        }
        """
    )
    upd_vars = {
        "payload": [{
            "id": int(post["id"]),
            "post_comments": [
                {
                    "id": int(pc["id"]),
                    "admin_likes": [
                        {"id": like_id, "user_id": int(u2.id)}
                    ]
                }
            ],
        }]
    }
    res2 = await schema.execute(upd_m, variable_values=upd_vars, context_value={"db_session": db_session})
    assert res2.errors is not None, "expected scope violation on update"
    assert "out of scope" in str(res2.errors[0]).lower()
