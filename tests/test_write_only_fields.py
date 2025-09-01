import pytest


@pytest.mark.asyncio
async def test_write_only_hidden_from_query(db_session, populated_db):
    from tests.schema import schema
    q = """
    query { posts { id title authorEmail } }
    """
    # authorEmail should not exist on query type; expect GraphQL error
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors, "Expected error for unknown field authorEmail on PostQL"

    q = """
    query { posts { id title author_email } }
    """
    # author_email should not exist on query type; expect GraphQL error
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors, "Expected error for unknown field author_email on PostQL"

@pytest.mark.asyncio
async def test_write_only_input_resolves_author_email(db_session, populated_db):
    from tests.schema import schema
    # pick an existing user email from fixtures
    u = populated_db['users'][0]
    email = str(getattr(u, 'email'))
    m = (
        """
        mutation($p: [PostQLInput!]!) {
          merge_posts(payload: $p) { id title author_id }
        }
        """
    )
    vars = {"p": [{"title": "ByEmail", "content": "X", "author_email": email}]}
    res = await schema.execute(m, variable_values=vars, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data["merge_posts"]
    assert data["title"] == "ByEmail"
    # author_id should reflect the resolved id
    assert str(data.get("author_id")) == str(getattr(u, 'id'))


@pytest.mark.asyncio
async def test_write_only_on_nested_comment(db_session, populated_db):
    from tests.schema import schema
    u = populated_db['users'][1]
    email = str(getattr(u, 'email'))
    m = (
        """
        mutation($p: [PostQLInput!]!) {
          merge_posts(payload: $p) {
            id
            post_comments(order_by: \"id\") { id content author_id }
          }
        }
        """
    )
    vars = {
        "p": [{
            "title": "WithCommentEmail",
            "content": "Body",
            "author_id": int(populated_db['users'][0].id),
            "post_comments": [
                {"content": "c1", "author_email": email}
            ]
        }]
    }
    res = await schema.execute(m, variable_values=vars, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    pcs = res.data["merge_posts"]["post_comments"]
    assert isinstance(pcs, list) and len(pcs) == 1
    assert str(pcs[0]["author_id"]) == str(getattr(u, 'id'))
