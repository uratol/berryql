from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from berryql import FieldPermissions, FieldSet, OperationPermissions
from berryql.core.policy import PolicyConfigurationError, PolicyEngine


class _Schema:
    def __init__(self, provider=None):
        self._field_permissions_provider = provider
        self._auto_camel_case = False
        self.types = {}


class _Info:
    def __init__(self, context, operation=None, variable_values=None):
        self.context = context
        self.operation = operation if operation is not None else object()
        self.variable_values = variable_values if variable_values is not None else {}


def _type_with(provider=None):
    fields = {name: SimpleNamespace(kind="scalar", meta={}) for name in ("id", "title", "content")}
    return type(
        "PolicyUnitType",
        (),
        {
            "__berry_fields__": fields,
            "__field_permissions_provider__": provider,
        },
    )


@pytest.mark.asyncio
async def test_policy_engine_concurrent_resolution_is_single_flight():
    calls = 0

    async def provider(berry_type, info):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return FieldPermissions(select=FieldSet.only("id"))

    berry_type = _type_with()
    engine = PolicyEngine(_Schema(provider))
    info = _Info({})

    resolved = await asyncio.gather(*(engine.resolve(berry_type, info) for _ in range(8)))

    assert calls == 1
    assert all(item is resolved[0] for item in resolved)


@pytest.mark.asyncio
async def test_policy_engine_failed_task_is_retried_and_context_reuse_isolated():
    calls = 0

    async def provider(berry_type, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary ACL failure")
        return FieldPermissions(select=FieldSet.only("id"))

    berry_type = _type_with()
    engine = PolicyEngine(_Schema(provider))
    context = {}
    first_info = _Info(context)

    with pytest.raises(RuntimeError, match="temporary ACL failure"):
        await engine.resolve(berry_type, first_info)
    retry = await engine.resolve(berry_type, first_info)
    assert retry.select.allows("id")
    assert calls == 2

    second_info = _Info(context, operation=object(), variable_values={})
    await engine.resolve(berry_type, second_info)
    assert calls == 3
    state = context[PolicyEngine._CACHE_ATTR]
    assert len(state.values) == 1


@pytest.mark.asyncio
async def test_policy_engine_intersects_every_capability_and_operation():
    global_permissions = FieldPermissions(
        select=FieldSet.only("id", "title"),
        filter=FieldSet.only("id", "title"),
        order=FieldSet.only("title"),
        create=FieldSet.only("title", "content"),
        update=FieldSet.only("content"),
        operations=OperationPermissions(create=True, update=True, delete=False, replace=True),
    )
    local_permissions = FieldPermissions(
        select=FieldSet.all_except("title"),
        filter=FieldSet.all(),
        order=FieldSet.all(),
        create=FieldSet.only("title"),
        update=FieldSet.all_except("title"),
        operations=OperationPermissions(create=True, update=False, delete=True, replace=False),
    )
    berry_type = _type_with(local_permissions)
    engine = PolicyEngine(_Schema(global_permissions))

    resolved = await engine.resolve(berry_type, _Info({}))

    assert resolved.select.allows("id")
    assert not resolved.select.allows("title")
    assert resolved.filter.allows("id")
    assert not resolved.filter.allows("title")
    assert not resolved.order.allows("title")
    assert resolved.create.allows("title")
    assert not resolved.create.allows("content")
    assert resolved.update.allows("content")
    assert resolved.operations == OperationPermissions(create=True, update=False, delete=False, replace=False)


@pytest.mark.asyncio
async def test_policy_engine_rejects_unknown_canonical_field():
    berry_type = _type_with()
    engine = PolicyEngine(_Schema(FieldPermissions(select=FieldSet.only("missing_field"))))

    with pytest.raises(PolicyConfigurationError, match="missing_field"):
        await engine.resolve(berry_type, _Info({}))
