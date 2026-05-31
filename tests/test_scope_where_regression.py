import json

import pytest

from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_root_scope_is_not_bypassed_by_where(db_session, sample_users, populated_db):
    bob = sample_users[1]
    alice = sample_users[0]
    ctx = {"db_session": db_session, "user_id": bob.id, "current_user": bob, "enforce_user_gate": True}

    q = """
    query($where: String) {
      users(where: $where, order_by: "id") { id name }
    }
    """

    res_other = await berry_schema.execute(
        q,
        variable_values={"where": json.dumps({"id": {"eq": alice.id}})},
        context_value=ctx,
    )
    assert res_other.errors is None, res_other.errors
    assert res_other.data["users"] == []

    res_self = await berry_schema.execute(
        q,
        variable_values={"where": json.dumps({"id": {"eq": bob.id}})},
        context_value=ctx,
    )
    assert res_self.errors is None, res_self.errors
    assert res_self.data["users"] == [{"id": bob.id, "name": bob.name}]


@pytest.mark.asyncio
async def test_domain_scope_is_not_bypassed_by_where(db_session, sample_users, populated_db):
    bob = sample_users[1]
    alice = sample_users[0]
    ctx = {"db_session": db_session, "user_id": bob.id, "current_user": bob, "enforce_user_gate": True}

    q = """
    query($where: String) {
      userDomain {
        users(where: $where, order_by: "id") { id name }
      }
    }
    """

    res_other = await berry_schema.execute(
        q,
        variable_values={"where": json.dumps({"id": {"eq": alice.id}})},
        context_value=ctx,
    )
    assert res_other.errors is None, res_other.errors
    assert res_other.data["userDomain"]["users"] == []

    res_self = await berry_schema.execute(
        q,
        variable_values={"where": json.dumps({"id": {"eq": bob.id}})},
        context_value=ctx,
    )
    assert res_self.errors is None, res_self.errors
    assert res_self.data["userDomain"]["users"] == [{"id": bob.id, "name": bob.name}]