import pytest

from berryql import encode_cursor
from tests.schema import schema


@pytest.mark.asyncio
async def test_root_keyset_pagination_is_stable(db_session, sample_users):
    first = await schema.execute(
        'query { users(limit: 2, order_by: "id") { id } }',
        context_value={"db_session": db_session},
    )
    assert first.errors is None
    first_ids = [row["id"] for row in first.data["users"]]
    assert first_ids

    cursor = encode_cursor(first_ids[-1])
    second = await schema.execute(
        'query($after: String!) { users(limit: 2, order_by: "id", after: $after) { id } }',
        variable_values={"after": cursor},
        context_value={"db_session": db_session},
    )
    assert second.errors is None
    second_ids = [row["id"] for row in second.data["users"]]
    assert set(first_ids).isdisjoint(second_ids)
    assert second_ids == sorted(second_ids)
