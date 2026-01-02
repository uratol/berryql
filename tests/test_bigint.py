import pytest
import uuid
import json
from tests.schema import schema
from tests.models import GenericItem

@pytest.mark.asyncio
async def test_bigint_exposure(db_session):
    # Create a GenericItem with a value larger than 32-bit integer
    # 2^31 - 1 = 2147483647
    # Let's use something clearly larger
    big_value = 3000000000 
    
    item = GenericItem(
        id=uuid.uuid4(),
        name="BigInt Item",
        code="BIG",
        count=big_value,
        active=True
    )
    db_session.add(item)
    await db_session.commit()
    
    query = """
    query($w: String) {
        genericItems(where: $w) {
            id
            name
            count
        }
    }
    """
    
    w = json.dumps({"code": {"eq": "BIG"}})
    result = await schema.execute(query, variable_values={'w': w}, context_value={'db_session': db_session})
    assert result.errors is None, result.errors
    assert 'genericItems' in result.data
    items = result.data['genericItems']
    assert len(items) == 1
    assert items[0]['count'] == big_value
