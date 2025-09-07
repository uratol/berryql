import pytest
import uuid
from tests.schema import schema as berry_schema
from tests.fixtures import *  # noqa: F401,F403


@pytest.mark.asyncio
async def test_uuid_where_ne_string(db_session, populated_db):
    # Use where JSON string with UUID inequality to exclude one item
    items = populated_db.get('uuid_items')
    assert items and len(items) >= 2
    exclude_id = str(items[0].id)
    q = '''
    query($w: String) {
      uuidItems(where: $w) { id name }
    }
    '''
    w = '{"id": {"ne": "%s"}}' % exclude_id
    res = await berry_schema.execute(q, variable_values={"w": w}, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    ids = [row['id'] for row in res.data['uuidItems']]
    assert exclude_id not in ids
    # Sanity: remaining ids are valid UUID strings
    for s in ids:
        uuid.UUID(str(s))


@pytest.mark.asyncio
async def test_uuid_where_ne_variable_object_error(db_session, populated_db):
  # When passing an object value to where (not JSON string), error should mention Invalid where JSON or JSON object
    q = '''
  query($excludeId: UUID!) {
      uuidItems(where: { id: { ne: $excludeId } }) { id name }
    }
    '''
    # pick any existing id
    any_id = str(populated_db['uuid_items'][0].id)
    res = await berry_schema.execute(q, variable_values={"excludeId": any_id}, context_value={'db_session': db_session})
    # Depending on Strawberry parsing, this is likely a GraphQL validation error that String cannot represent non string value
    assert res.errors is not None
    msg = '\n'.join(str(e) for e in res.errors)
    assert ('Invalid where JSON' in msg) or ('where must be a JSON object' in msg) or ('String cannot represent a non string value' in msg)
