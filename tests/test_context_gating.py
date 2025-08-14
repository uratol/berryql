import pytest
from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_admin_users_can_see_all_users(db_session, sample_users, populated_db):
    # Context uses Alice (admin) from common fixtures: id=1, is_admin=True
    admin_ctx = {
        'db_session': db_session,
        'user_id': sample_users[0].id,
        'current_user': sample_users[0],
        'enforce_user_gate': True,
    }
    query = """
    query { users { id name is_admin } }
    """
    res = await berry_schema.execute(query, context_value=admin_ctx)
    assert res.errors is None, res.errors
    assert len(res.data['users']) == 4
    assert any(u['is_admin'] for u in res.data['users'])


@pytest.mark.asyncio
async def test_non_admin_users_see_only_themselves(db_session, sample_users, populated_db):
    # Bob Smith is non-admin (index 1)
    bob = sample_users[1]
    ctx = {'db_session': db_session, 'user_id': bob.id, 'current_user': bob, 'enforce_user_gate': True}
    query = """
    query { users { id name is_admin } }
    """
    res = await berry_schema.execute(query, context_value=ctx)
    assert res.errors is None, res.errors
    assert len(res.data['users']) == 1
    assert res.data['users'][0]['id'] == bob.id


@pytest.mark.asyncio
async def test_no_user_context_returns_empty(db_session, populated_db):
    ctx = {'db_session': db_session, 'user_id': None, 'current_user': None, 'enforce_user_gate': True}
    q = """
    query { users { id } }
    """
    res = await berry_schema.execute(q, context_value=ctx)
    assert res.errors is None, res.errors
    assert res.data['users'] == []


