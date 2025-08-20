import pytest
from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_root_async_where_callable_filters_ids(db_session, populated_db):
    q = """
    query { usersAsyncGate { id } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    ids = [u['id'] for u in res.data['usersAsyncGate']]
    # async where enforces id > 1 by default
    assert ids and all(i > 1 for i in ids)


@pytest.mark.asyncio
async def test_root_async_where_none_no_filter(db_session, populated_db):
    q = """
    query { usersAsyncGate { id } }
    """
    # When async_gate_return_none flag is set, the async where returns None (no filter)
    res = await berry_schema.execute(q, context_value={'db_session': db_session, 'async_gate_return_none': True})
    assert res.errors is None, res.errors
    ids = [u['id'] for u in res.data['usersAsyncGate']]
    assert ids and min(ids) == 1
