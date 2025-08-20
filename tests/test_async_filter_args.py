import pytest
from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_root_filter_arg_async_builder(db_session, populated_db):
    # Use the blogDomain.postsAsyncFilter declared with an async builder
    q = """
    query { blogDomain { postsAsyncFilter(created_at_gt: "1900-01-01T00:00:00") { id } } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    # Should return some posts
    data = res.data['blogDomain']['postsAsyncFilter']
    assert isinstance(data, list)
    assert len(data) >= 1
