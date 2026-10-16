from __future__ import annotations

import enum
import json

import pytest
import strawberry
from sqlalchemy import Boolean, Column, Enum as SAEnum, ForeignKey, Integer, String, func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, deferred

from berryql import (
    BerrySchema,
    BerryType,
    ConfigurationError,
    StrawberryConfig,
    ValidationError,
    field,
    mutation,
    relation,
)


class Base(DeclarativeBase):
    pass


class WorkerKind(enum.Enum):
    EMPLOYEE = "EMPLOYEE"
    CONTRACTOR = "CONTRACTOR"


class Worker(Base):
    __tablename__ = "required_workers"
    id = Column(Integer, primary_key=True)
    legal_first_name = Column(String, nullable=False)
    worker_type = Column(SAEnum(WorkerKind), nullable=False)
    employment_category = Column(String)
    abn = deferred(Column(String))
    pay_basis = Column(String)
    hourly_rate = Column(Integer)
    tfn = Column(String)
    python_default = Column(Boolean, nullable=False, default=False)
    server_default = Column(String, nullable=False, server_default=text("'ready'"))
    callable_default = Column(String, nullable=False, default=lambda: "from_callable")
    optional_field = Column(String)


class Note(Base):
    __tablename__ = "required_notes"
    id = Column(Integer, primary_key=True)
    worker_id = Column(Integer, ForeignKey(Worker.id), nullable=False)
    note_text = Column(String, nullable=False)


CATEGORY_RULE = "Required for EMPLOYEE: FULL_TIME, PART_TIME or CASUAL."
ABN_RULE = "Required for CONTRACTOR: 11-digit ABN."
RATE_RULE = "Required when payBasis is HOURLY."


def build_schema(*, camel=True, pre=None, name_converter=None, optional_requirement=None, default_requirement=None):
    berry = BerrySchema()

    @berry.type(model=Note)
    class NoteQL(BerryType):
        id = field()
        worker_id = field()
        note_text = field()

    @berry.type(model=Worker)
    class WorkerQL(BerryType):
        id = field()
        first_name = field("legal_first_name", required="create", comment="Legal first name.")
        worker_type = field()
        employment_category = field(required_when=lambda w: w.worker_type == "EMPLOYEE", rule=CATEGORY_RULE)
        abn = field(required_when=lambda w: w.worker_type == "CONTRACTOR", rule=ABN_RULE)
        pay_basis = field()
        hourly_rate = field(required_when=lambda w: w.pay_basis == "HOURLY", rule=RATE_RULE)
        tfn = field(required=True, allow_unknown=True)
        python_default = field(**(default_requirement or {}))
        server_default = field()
        callable_default = field()
        optional_field = field(**{"required": False, **(optional_requirement or {})})
        notes = relation("NoteQL")

    if pre:
        WorkerQL.__merge_pre_cbs__ = (pre,)

    @berry.query()
    class Query:
        workers = relation("WorkerQL")

    @berry.mutation()
    class Mutation:
        merge_workers = mutation("WorkerQL")
        merge_worker = mutation("WorkerQL", single=True)

    config = StrawberryConfig(auto_camel_case=camel)
    if name_converter is not None:
        config.name_converter = name_converter
    return berry, berry.to_strawberry(strawberry_config=config)


@pytest.fixture
async def worker_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session
    await engine.dispose()


def employee(**changes):
    return dict(firstName="Ada", workerType="EMPLOYEE", employmentCategory="CASUAL", tfn=None, **changes)


async def merge(schema, session, payload):
    return await schema.execute(
        "mutation($payload: [WorkerQLInput!]!) { mergeWorkers(payload: $payload) { id firstName } }",
        variable_values={"payload": payload if isinstance(payload, list) else [payload]},
        context_value={"db_session": session},
    )


def validation(result, fields, rules=None):
    assert len(result.errors or []) == 1, result.errors
    error = result.errors[0]
    assert error.message == "Missing required fields"
    assert error.extensions["code"] == "VALIDATION"
    assert error.extensions["fields"] == fields
    if rules is not None:
        assert error.extensions["rules"] == rules
    return error


async def count_workers(session):
    return await session.scalar(select(func.count()).select_from(Worker))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"required": "update"},
        {"required": 1},
        {"required_when": lambda w: True},
        {"required_when": 1, "rule": "required"},
        {"rule": " "},
        {"allow_unknown": "yes"},
    ],
)
def test_invalid_declarations_fail_early(kwargs):
    with pytest.raises(ConfigurationError):
        field(**kwargs)


def test_async_predicate_is_rejected():
    async def predicate(record):
        return True

    with pytest.raises(ConfigurationError):
        field(required_when=predicate, rule="required")


async def test_input_metadata_is_discoverable_and_shared_inputs_remain_nullable():
    berry, schema = build_schema()
    result = await schema.execute(
        '{ __type(name: "WorkerQLInput") { inputFields { name description type { kind name } defaultValue } } }'
    )
    assert result.errors is None
    fields = {f["name"]: f for f in result.data["__type"]["inputFields"]}
    for name in ("firstName", "workerType", "tfn", "abn"):
        assert fields[name]["type"]["kind"] != "NON_NULL"
        assert fields[name]["defaultValue"] is None
    assert fields["firstName"]["description"].startswith("Legal first name.\n\n")
    metadata = json.loads(fields["abn"]["description"].split("BerryQL validation: ")[1])
    assert metadata == {"requiredOnCreate": False, "allowUnknown": False, "requiredWhen": ABN_RULE, "rule": ABN_RULE}
    gql_fields = schema._schema.get_type("WorkerQLInput").fields
    assert gql_fields["firstName"].extensions["requiredOnCreate"] is True
    assert gql_fields["workerType"].extensions["requiredOnCreate"] is True
    assert gql_fields["tfn"].extensions["allowUnknown"] is True
    for name in ("id", "pythonDefault", "serverDefault", "optionalField"):
        assert gql_fields[name].extensions["requiredOnCreate"] is False
    input_type = berry._st_types["WorkerQLInput"]
    assert input_type().first_name is strawberry.UNSET


@pytest.mark.parametrize("missing", ["firstName", "workerType", "tfn"])
async def test_create_requires_an_answer(worker_session, missing):
    _, schema = build_schema()
    payload = employee()
    del payload[missing]
    result = await merge(schema, worker_session, payload)
    validation(result, [missing])
    assert await count_workers(worker_session) == 0


@pytest.mark.parametrize("name", ["firstName", "workerType", "employmentCategory"])
async def test_explicit_null_is_rejected_without_allow_unknown(worker_session, name):
    _, schema = build_schema()
    payload = employee()
    payload[name] = None
    validation(await merge(schema, worker_session, payload), [name])
    assert await count_workers(worker_session) == 0


async def test_reports_all_missing_conditional_fields(worker_session):
    _, schema = build_schema()
    payload = employee(payBasis="HOURLY")
    del payload["employmentCategory"]
    validation(
        await merge(schema, worker_session, payload),
        ["employmentCategory", "hourlyRate"],
        {
            "employmentCategory": CATEGORY_RULE,
            "hourlyRate": RATE_RULE,
        },
    )
    assert await count_workers(worker_session) == 0


async def test_null_unknown_and_falsy_values_are_answers(worker_session):
    _, schema = build_schema()
    result = await merge(schema, worker_session, employee(payBasis="HOURLY", hourlyRate=0))
    assert result.errors is None, result.errors
    row = await worker_session.get(Worker, result.data["mergeWorkers"][0]["id"])
    assert row.legal_first_name == "Ada"
    assert row.tfn is None
    assert row.hourly_rate == 0
    assert row.python_default is False
    assert row.server_default == "ready"


async def test_partial_updates_keep_existing_values_and_unknown_null(worker_session):
    _, schema = build_schema()
    created = await merge(schema, worker_session, employee())
    worker_id = created.data["mergeWorkers"][0]["id"]
    updated = await merge(schema, worker_session, {"id": worker_id, "firstName": "Grace", "tfn": None})
    assert updated.errors is None, updated.errors
    assert updated.data["mergeWorkers"][0]["firstName"] == "Grace"
    validation(await merge(schema, worker_session, {"id": worker_id, "firstName": None}), ["firstName"])
    assert (await worker_session.get(Worker, worker_id)).legal_first_name == "Grace"


async def test_condition_uses_existing_row_plus_changes_and_loads_deferred_columns(worker_session):
    _, schema = build_schema()
    created = await merge(schema, worker_session, employee())
    worker_id = created.data["mergeWorkers"][0]["id"]
    worker_session.expunge_all()
    validation(
        await merge(schema, worker_session, {"id": worker_id, "workerType": "CONTRACTOR"}), ["abn"], {"abn": ABN_RULE}
    )
    row = await worker_session.get(Worker, worker_id)
    assert row.worker_type == WorkerKind.EMPLOYEE
    result = await merge(schema, worker_session, {"id": worker_id, "workerType": "CONTRACTOR", "abn": "12345678901"})
    assert result.errors is None, result.errors
    worker_session.expunge_all()
    result = await merge(schema, worker_session, {"id": worker_id, "firstName": "Grace"})
    assert result.errors is None, result.errors
    validation(await merge(schema, worker_session, {"id": worker_id, "abn": None}), ["abn"])


async def test_existing_condition_is_checked_when_dependency_is_omitted(worker_session):
    _, schema = build_schema()
    created = await merge(schema, worker_session, employee(payBasis="HOURLY", hourlyRate=10))
    worker_id = created.data["mergeWorkers"][0]["id"]
    validation(
        await merge(schema, worker_session, {"id": worker_id, "hourlyRate": None}),
        ["hourlyRate"],
        {"hourlyRate": RATE_RULE},
    )
    result = await merge(schema, worker_session, {"id": worker_id, "payBasis": "SALARY", "hourlyRate": None})
    assert result.errors is None, result.errors


@pytest.mark.parametrize("controls", [{"id": 99}, {"id": 99, "_Insert": True}])
async def test_create_detection_with_explicit_pk(worker_session, controls):
    _, schema = build_schema()
    validation(await merge(schema, worker_session, controls), ["firstName", "workerType", "tfn"])
    assert await count_workers(worker_session) == 0


async def test_force_insert_does_not_borrow_existing_required_values(worker_session):
    _, schema = build_schema()
    created = await merge(schema, worker_session, employee())
    worker_id = created.data["mergeWorkers"][0]["id"]
    validation(
        await merge(schema, worker_session, {"id": worker_id, "_Insert": True}), ["firstName", "workerType", "tfn"]
    )
    assert await count_workers(worker_session) == 1
    missing_update = await merge(schema, worker_session, {"id": 99, "_Insert": False})
    assert "requires an existing" in missing_update.errors[0].message


async def test_batch_failure_rolls_back_preceding_items(worker_session):
    _, schema = build_schema()
    validation(await merge(schema, worker_session, [employee(), {"tfn": None}]), ["firstName", "workerType"])
    assert await count_workers(worker_session) == 0


async def test_nested_requirements_see_derived_fk_and_rollback_parent(worker_session):
    _, schema = build_schema()
    validation(await merge(schema, worker_session, employee(notes=[{}])), ["noteText"])
    assert await count_workers(worker_session) == 0
    result = await merge(schema, worker_session, employee(notes=[{"noteText": "Welcome"}]))
    assert result.errors is None, result.errors
    assert await worker_session.scalar(select(func.count()).select_from(Note)) == 1


async def test_pre_hooks_supply_answers_and_can_raise_the_same_validation_error(worker_session):
    def pre(data):
        if data.get("first_name") == "blocked":
            raise ValidationError("ABN is already registered", fields=["abn"], rules={"abn": "ABN must be unique."})
        return {"first_name": "from hook", "worker_type": "CONTRACTOR", "abn": "12345678901", "tfn": None, **data}

    _, schema = build_schema(pre=pre)
    result = await merge(schema, worker_session, {})
    assert result.errors is None, result.errors
    assert result.data["mergeWorkers"][0]["firstName"] == "from hook"
    result = await merge(schema, worker_session, {"firstName": "blocked"})
    assert result.errors[0].message == "ABN is already registered"
    assert result.errors[0].extensions == {
        "code": "VALIDATION",
        "fields": ["abn"],
        "rules": {"abn": "ABN must be unique."},
    }
    assert await count_workers(worker_session) == 1


async def test_delete_skips_required_checks_for_legacy_invalid_rows(worker_session):
    row = Worker(legal_first_name="Legacy", worker_type=WorkerKind.CONTRACTOR)
    worker_session.add(row)
    await worker_session.commit()
    _, schema = build_schema()
    result = await merge(schema, worker_session, {"id": row.id, "_Delete": True})
    assert result.errors is None, result.errors
    assert await count_workers(worker_session) == 0


async def test_single_merge_and_snake_case_error_names(worker_session):
    _, schema = build_schema(camel=False)
    result = await schema.execute(
        "mutation { merge_worker(payload: {tfn: null, worker_type: CONTRACTOR}) { id } }",
        context_value={"db_session": worker_session},
    )
    validation(result, ["first_name", "abn"])


async def test_custom_name_converter_is_used_in_errors(worker_session):
    from strawberry.schema.name_converter import NameConverter

    class CustomNames(NameConverter):
        def apply_naming_config(self, name):
            return name if name.startswith("_") else "custom_" + name

    _, schema = build_schema(name_converter=CustomNames(auto_camel_case=False))
    result = await schema.execute(
        "mutation { custom_merge_worker(custom_payload: {custom_tfn: null}) { custom_id } }",
        context_value={"db_session": worker_session},
    )
    validation(result, ["custom_first_name", "custom_worker_type"])


@pytest.mark.parametrize(
    "predicate",
    [
        lambda w: w.server_default == "ready",
        lambda w: w.callable_default == "from_callable",
        lambda w: w.abn is None,
    ],
)
async def test_predicates_see_resolved_defaults_and_omitted_nullable_values(worker_session, predicate):
    _, schema = build_schema(optional_requirement={"required_when": predicate, "rule": "Required for this state."})
    validation(await merge(schema, worker_session, employee()), ["optionalField"])
    assert await count_workers(worker_session) == 0
    result = await merge(schema, worker_session, employee(optionalField="answer"))
    assert result.errors is None, result.errors


async def test_pending_default_does_not_cause_premature_rejection(worker_session):
    _, schema = build_schema(
        optional_requirement={
            "required_when": lambda w: w.server_default != "ready",
            "rule": "Required unless ready.",
        }
    )
    result = await merge(schema, worker_session, employee())
    assert result.errors is None, result.errors


async def test_conditional_unknown_requires_explicit_answer_on_create(worker_session):
    _, schema = build_schema(
        optional_requirement={
            "required_when": lambda w: w.worker_type == "EMPLOYEE",
            "rule": "Answer or explicitly mark unknown.",
            "allow_unknown": True,
        }
    )
    validation(await merge(schema, worker_session, employee()), ["optionalField"])
    result = await merge(schema, worker_session, employee(optionalField=None))
    assert result.errors is None, result.errors
    result = await merge(schema, worker_session, {"id": result.data["mergeWorkers"][0]["id"], "firstName": "Grace"})
    assert result.errors is None, result.errors


async def test_explicit_requirement_overrides_a_python_default(worker_session):
    _, schema = build_schema(default_requirement={"required": "create"})
    validation(await merge(schema, worker_session, employee()), ["pythonDefault"])
    result = await merge(schema, worker_session, employee(pythonDefault=False))
    assert result.errors is None, result.errors
    validation(
        await merge(schema, worker_session, {"id": result.data["mergeWorkers"][0]["id"], "pythonDefault": None}),
        ["pythonDefault"],
    )
