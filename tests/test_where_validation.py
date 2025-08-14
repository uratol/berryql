import pytest
from tests.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_root_where_malformed_json(db_session, populated_db):
    # where expects a JSON object string; malformed should raise error
    q = """
    query($w: String) { posts(where: $w) { id } }
    """
    res = await berry_schema.execute(q, variable_values={"w": "{id: 1"}, context_value={'db_session': db_session})
    assert res.errors is not None
    assert any('Invalid where JSON' in str(e) or 'where must be a JSON object' in str(e) for e in res.errors)


@pytest.mark.asyncio
async def test_root_where_unknown_field(db_session, populated_db):
    q = """
    query($w: String) { posts(where: $w) { id } }
    """
    res = await berry_schema.execute(q, variable_values={"w": "{\"__nope__\": {\"eq\": 1}}"}, context_value={'db_session': db_session})
    # At root we currently validate order_by only; unknown where column should bubble as unknown where column
    assert res.errors is not None
    assert any('Unknown where column' in str(e) for e in res.errors)


@pytest.mark.asyncio
async def test_root_where_unknown_operator(db_session, populated_db):
    q = """
    query($w: String) { posts(where: $w) { id } }
    """
    res = await berry_schema.execute(q, variable_values={"w": "{\"id\": {\"__bad__\": 1}}"}, context_value={'db_session': db_session})
    assert res.errors is not None
    assert any('Unknown where operator' in str(e) for e in res.errors)


@pytest.mark.asyncio
async def test_relation_where_malformed_json(db_session, populated_db):
    q = """
    query($w: String) { users { id posts(where: $w) { id } } }
    """
    res = await berry_schema.execute(q, variable_values={"w": "{id: 1"}, context_value={'db_session': db_session})
    assert res.errors is not None
    assert any('Invalid where JSON' in str(e) or 'where must be a JSON object' in str(e) for e in res.errors)


@pytest.mark.asyncio
async def test_relation_where_unknown_field(db_session, populated_db):
    q = """
    query($w: String) { users { id posts(where: $w) { id } } }
    """
    res = await berry_schema.execute(q, variable_values={"w": "{\"__nope__\": {\"eq\": 1}}"}, context_value={'db_session': db_session})
    assert res.errors is not None
    assert any('Unknown where column' in str(e) for e in res.errors)


@pytest.mark.asyncio
async def test_relation_where_unknown_operator(db_session, populated_db):
    q = """
    query($w: String) { users { id posts(where: $w) { id } } }
    """
    res = await berry_schema.execute(q, variable_values={"w": "{\"id\": {\"__bad__\": 1}}"}, context_value={'db_session': db_session})
    assert res.errors is not None
    assert any('Unknown where operator' in str(e) for e in res.errors)


@pytest.mark.asyncio
async def test_relation_invalid_order_by_raises(db_session, populated_db):
    q = """
    query { users { id posts(order_by: "__nope__") { id } } }
    """
    res = await berry_schema.execute(q, context_value={'db_session': db_session})
    assert res.errors is not None
    assert any('Invalid order_by' in str(e) for e in res.errors)
