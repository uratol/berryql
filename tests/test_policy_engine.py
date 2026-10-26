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


def test_field_metadata_is_discovered_once_per_type(monkeypatch):
    berry_type = _type_with()
    berry_type.__annotations__ = {"created_at": str, "native_value": str}
    berry_type.__berry_fields__["title"].meta = {"alias": "displayTitle", "column": "stored_title"}
    engine = PolicyEngine(_Schema())
    original = engine._declared_field_names
    calls = []

    def discover(type_):
        calls.append(type_)
        return original(type_)

    monkeypatch.setattr(engine, "_declared_field_names", discover)
    for _ in range(100):
        for reference, expected in [
            ("title", "title"),
            ("displayTitle", "title"),
            ("stored_title", "title"),
            ("createdAt", "created_at"),
            ("native_value", "native_value"),
        ]:
            assert engine.canonical_field_name(berry_type, reference) == expected
    assert calls == [berry_type]


def test_field_metadata_and_name_converters_are_schema_local():
    berry_type = _type_with()
    berry_type.__berry_fields__["content"].meta = {"graphql_name": "body"}
    first_schema = _Schema()
    first_schema._name_converter = SimpleNamespace(apply_naming_config=lambda name: "first_" + name)
    second_schema = _Schema()
    second_schema._name_converter = SimpleNamespace(apply_naming_config=lambda name: "second_" + name)
    first, second = PolicyEngine(first_schema), PolicyEngine(second_schema)
    assert first.canonical_field_name(berry_type, "first_content") == "content"
    assert second.canonical_field_name(berry_type, "second_content") == "content"
    assert second._canonical_reference(berry_type, "first_content") is None
    assert first.canonical_field_name(berry_type, "body") == "content"
    other_type = _type_with()
    assert first._canonical_reference(other_type, "body") is None


@pytest.mark.asyncio
async def test_static_metadata_cache_does_not_share_execution_permissions():
    def provider(type_, info):
        return FieldPermissions(select=FieldSet.only(*info.context["allowed"]))

    berry_type = _type_with()
    engine = PolicyEngine(_Schema(provider))
    first = _Info({"allowed": ["title"]})
    second = _Info({"allowed": ["content"]})
    for _ in range(3):
        assert await engine.allows_field(berry_type, first, "select", "title")
        assert not await engine.allows_field(berry_type, second, "select", "title")
        assert await engine.allows_field(berry_type, second, "select", "content")


def test_field_metadata_preserves_alias_precedence():
    berry_type = _type_with()
    berry_type.__berry_fields__["title"].meta = {"alias": "display_title"}
    berry_type.__berry_fields__["content"].meta = {"alias": "displayTitle"}
    engine = PolicyEngine(_Schema())
    assert engine.canonical_field_name(berry_type, "displayTitle") == "title"


def test_registering_a_type_invalidates_static_metadata():
    from berryql import BerrySchema

    registry = BerrySchema()
    berry_type = _type_with()
    registry.register(berry_type)
    engine = registry._policy_engine
    assert engine._canonical_reference(berry_type, "headline") is None
    berry_type.__berry_fields__["title"].meta["alias"] = "headline"
    registry.register(berry_type)
    assert engine.canonical_field_name(berry_type, "headline") == "title"


def test_rebuilding_schema_invalidates_name_converter_metadata():
    from strawberry.schema.name_converter import NameConverter

    from berryql import BerrySchema, BerryType, StrawberryConfig, field
    from tests.models import User

    class PrefixConverter(NameConverter):
        def __init__(self, prefix):
            super().__init__(auto_camel_case=False)
            self.prefix = prefix

        def apply_naming_config(self, name):
            return self.prefix + name

    registry = BerrySchema()

    @registry.type(model=User)
    class RebuiltUserQL(BerryType):
        id = field()
        name = field()

    registry.to_strawberry(strawberry_config=StrawberryConfig(name_converter=PrefixConverter("first_")))
    assert registry._policy_engine.canonical_field_name(RebuiltUserQL, "first_name") == "name"
    registry.to_strawberry(strawberry_config=StrawberryConfig(name_converter=PrefixConverter("second_")))
    assert registry._policy_engine.canonical_field_name(RebuiltUserQL, "second_name") == "name"
    assert registry._policy_engine._canonical_reference(RebuiltUserQL, "first_name") is None
