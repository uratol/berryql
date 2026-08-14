from types import SimpleNamespace

import pytest
from sqlalchemy import column

from berryql import (
    BerrySchema,
    FieldPermissions,
    FieldSet,
    FilterLimits,
    FilterSpec,
    decode_cursor,
    encode_cursor,
)
from berryql.core.analyzer import (
    PushdownDecision,
    PushdownReason,
    RelationPlan,
    ValueSource,
)
from berryql.core.ordering import OrderingCompiler
from berryql.core.policy import PolicyConfigurationError, PolicyEngine
from berryql.core.predicates import PredicateCompiler, PredicateError


def test_filter_spec_clone_preserves_type_and_dependency_metadata():
    spec = FilterSpec(
        column="id",
        ops=["in", "not_in"],
        arg_type=int,
        depends_on=("owner.id",),
        fields=("tenant_id",),
    )

    clone = spec.clone_with(op="in", ops=None)

    assert clone.arg_type is int
    assert clone.ops is None
    assert clone.dependencies == ("id", "owner.id", "tenant_id")


def test_predicate_limits_and_value_shapes_are_strict():
    schema = SimpleNamespace(
        filter_limits=FilterLimits(max_clauses=1, max_in_items=2, max_json_length=50),
        _operators=None,
        _auto_camel_case=False,
    )
    compiler = PredicateCompiler(schema)
    model = SimpleNamespace(__table__=SimpleNamespace(c={"id": column("id")}))

    with pytest.raises(PredicateError, match="exactly two"):
        compiler.parse({"id": {"between": [1]}}, model, trusted=False)
    with pytest.raises(PredicateError, match="max_in_items"):
        compiler.parse({"id": {"in": [1, 2, 3]}}, model, trusted=False)
    with pytest.raises(PredicateError, match="max_clauses"):
        compiler.parse({"id": {"gt": 1, "lt": 4}}, model, trusted=False)


def test_operator_registry_is_schema_local():
    left = BerrySchema(operators={"is_even": lambda col, _value: col % 2 == 0})
    right = BerrySchema()

    assert left._predicate_compiler.operators.get("is_even") is not None
    assert right._predicate_compiler.operators.get("is_even") is None


def test_relation_plan_uses_typed_provenance_and_pushdown_reason():
    plan = RelationPlan(
        target="ItemQL",
        fields=("id",),
        order_by="id",
        order_by_source=ValueSource.CALLER,
    )
    decision = PushdownDecision.fallback(PushdownReason.ADAPTER_UNSUPPORTED, "no lateral")

    assert plan.get("order_by_source") is ValueSource.CALLER
    assert not any(key.startswith("_has_explicit") for key in plan)
    assert decision.reason is PushdownReason.ADAPTER_UNSUPPORTED


def test_cursor_round_trip_and_invalid_payload():
    cursor = encode_cursor(10, "stable")
    assert decode_cursor(cursor) == (10, "stable")
    with pytest.raises(ValueError, match="Invalid after cursor"):
        decode_cursor("not-a-cursor")


def test_policy_canonicalizes_camel_case_and_rejects_unknown_deny_names():
    class ItemQL:
        __annotations__ = {"created_at": str}
        __berry_fields__ = {}

    schema = SimpleNamespace(_field_permissions_provider=None, _auto_camel_case=True, types={})
    engine = PolicyEngine(schema)
    normalized = engine._normalize_permissions(ItemQL, FieldPermissions(select=FieldSet.only("createdAt")))
    assert normalized.select.fields == frozenset({"created_at"})

    with pytest.raises(PolicyConfigurationError, match="missingField"):
        engine._normalize_permissions(ItemQL, FieldPermissions(select=FieldSet.all_except("missingField")))


def test_null_ordering_is_additive():
    compiler = OrderingCompiler(SimpleNamespace())
    term = compiler.parse_multi(["name:desc:nulls_last"])[0]
    assert term.direction == "desc"
    assert term.nulls == "last"
