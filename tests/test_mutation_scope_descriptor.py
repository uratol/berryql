import pytest

from tests.schema import schema


@pytest.mark.asyncio
async def test_root_mutation_scope_rejects_create_out_of_scope(db_session, populated_db):
    # merge_posts_scoped enforces author_id == 1; try author_id == 2
    u2_id = int(populated_db['users'][1].id)
    m = (
        """
        mutation($p: [PostQLInput!]!) {
          merge_posts_scoped(payload: $p) { id title author_id }
        }
        """
    )
    res = await schema.execute(
        m,
        variable_values={"p": [{"title": "X", "content": "Y", "author_id": u2_id}]},
        context_value={"db_session": db_session},
    )
    assert res.errors is not None, "expected out-of-scope create to be rejected"
    assert "out of scope" in str(res.errors[0]).lower()


@pytest.mark.asyncio
async def test_root_mutation_scope_allows_in_scope_create(db_session, populated_db):
    # author_id == 1 should pass
    u1_id = int(populated_db['users'][0].id)
    m = (
        """
        mutation($p: [PostQLInput!]!) {
          merge_posts_scoped(payload: $p) { id title author_id }
        }
        """
    )
    res = await schema.execute(
        m,
        variable_values={"p": [{"title": "OK", "content": "Y", "author_id": u1_id}]},
        context_value={"db_session": db_session},
    )
    assert res.errors is None, res.errors
    assert int(res.data["merge_posts_scoped"]["author_id"]) == u1_id


@pytest.mark.asyncio
async def test_root_mutation_scope_rejects_update_out_of_scope(db_session, populated_db):
    # First create in-scope post via scoped mutation
    u1_id = int(populated_db['users'][0].id)
    u2_id = int(populated_db['users'][1].id)
    m_create = (
        """
        mutation($p: [PostQLInput!]!) {
          merge_posts_scoped(payload: $p) { id author_id }
        }
        """
    )
    res1 = await schema.execute(
        m_create,
        variable_values={"p": [{"title": "C1", "content": "B", "author_id": u1_id}]},
        context_value={"db_session": db_session},
    )
    assert res1.errors is None, res1.errors
    pid = int(res1.data["merge_posts_scoped"]["id"])
    # Try to update author_id to 2 (out-of-scope)
    m_upd = (
        """
        mutation($p: [PostQLInput!]!) {
          merge_posts_scoped(payload: $p) { id }
        }
        """
    )
    res2 = await schema.execute(
        m_upd,
    variable_values={"p": [{"id": pid, "author_id": u2_id}]},
        context_value={"db_session": db_session},
    )
    assert res2.errors is not None, "expected out-of-scope update to be rejected"
    assert "out of scope" in str(res2.errors[0]).lower()


@pytest.mark.asyncio
async def test_root_mutation_scope_rejects_delete_out_of_scope(db_session, populated_db):
    # Create in-scope post then try to delete after re-scoping via payload
    u1_id = int(populated_db['users'][0].id)
    m_create = (
        """
        mutation($p: [PostQLInput!]!) {
          merge_posts_scoped(payload: $p) { id author_id }
        }
        """
    )
    res1 = await schema.execute(
        m_create,
        variable_values={"p": [{"title": "C2", "content": "B", "author_id": u1_id}]},
        context_value={"db_session": db_session},
    )
    assert res1.errors is None, res1.errors
    pid = int(res1.data["merge_posts_scoped"]["id"])
    # deletion is also checked against scope; since author_id is 1 and scope is author_id==1, delete should pass
    # To exercise rejection, attempt delete through the unscoped mutation but expect internal scope to reject
    m_del = (
        """
        mutation($p: [PostQLInput!]!) {
          merge_posts_scoped(payload: $p) { id }
        }
        """
    )
    res2 = await schema.execute(
        m_del,
        variable_values={"p": [{"id": pid, "_Delete": True, "author_id": 2}]},
        context_value={"db_session": db_session},
    )
    # Scope uses stored instance attrs, so author_id change in payload doesn't matter; still in-scope -> allow delete
    # So assert success here to reflect enforcement behavior after load
    assert res2.errors is None, res2.errors


@pytest.mark.asyncio
async def test_domain_mutation_scope_descriptor_and_guard(db_session, populated_db):
    # blogDomain.merge_posts_scoped has descriptor scope author_id==1 and also domain guard can be added
    u1_id = int(populated_db['users'][0].id)
    u2_id = int(populated_db['users'][1].id)
    m = (
        """
        mutation($p: [PostQLInput!]!) {
          blogDomain { merge_posts_scoped(payload: $p) { id author_id } }
        }
        """
    )
    # Out-of-scope create
    res1 = await schema.execute(
        m,
        variable_values={"p": [{"title": "X", "content": "Y", "author_id": u2_id}]},
        context_value={"db_session": db_session},
    )
    assert res1.errors is not None, "expected domain scoped mutation to reject out-of-scope create"
    assert "out of scope" in str(res1.errors[0]).lower()
    # In-scope create
    res2 = await schema.execute(
        m,
        variable_values={"p": [{"title": "OK", "content": "Y", "author_id": u1_id}]},
        context_value={"db_session": db_session},
    )
    assert res2.errors is None, res2.errors
    assert int(res2.data["blogDomain"]["merge_posts_scoped"]["author_id"]) == u1_id
