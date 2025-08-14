import pytest


@pytest.mark.asyncio
async def test_query_regular_strawberry_field(db_session, populated_db):
    from tests.schema import berry_schema
    schema = berry_schema.to_strawberry()
    res = await schema.execute("query { hello }", context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data == {'hello': 'world'}
