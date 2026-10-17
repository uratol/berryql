from types import MappingProxyType

import pytest
import strawberry
from graphql import get_named_type
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import DeclarativeBase
from strawberry.schema.schema_converter import GraphQLCoreConverter

from berryql import (
    BerryDomain,
    BerrySchema,
    BerryType,
    ConfigurationError,
    StrawberryConfig,
    domain,
    field,
    mutation,
    relation,
)
from berryql.core.fields import MutationDescriptor
from berryql.mutations import publish_mutation_metadata

BACKREF = GraphQLCoreConverter.DEFINITION_BACKREF
METADATA = {
    "requiredSkills": ["workforce.hire"],
    "application": {"enabled": True, "limit": 2, "optional": None},
}


class Base(DeclarativeBase):
    pass


class Worker(Base):
    __tablename__ = "metadata_workers"
    id = Column(Integer, primary_key=True)
    name = Column(String)


def build_schema(metadata=METADATA, *, camel=False):
    berry = BerrySchema()

    @berry.type(model=Worker)
    class WorkerQL(BerryType):
        id = field()
        name = field()

    @berry.domain()
    class WorkforceDomain(BerryDomain):
        merge_worker = mutation("WorkerQL", single=True, comment="Merge one worker.", metadata=metadata)
        merge_workers = mutation("WorkerQL", metadata=metadata)
        merge_unannotated = mutation("WorkerQL")

        @strawberry.mutation(name="hireWorker", description="Hire a worker.", metadata=metadata)
        def hire_worker(self) -> bool:
            return True

        @strawberry.mutation
        def unannotated(self) -> bool:
            return True

    @berry.domain()
    class ParentDomain(BerryDomain):
        workforce = domain(WorkforceDomain)

    @berry.query()
    class Query:
        workers = relation("WorkerQL")

    @berry.mutation()
    class Mutation:
        workforce = domain(WorkforceDomain)
        parent = domain(ParentDomain)
        merge_worker = mutation("WorkerQL", single=True, comment="Merge one worker.", metadata=metadata)
        merge_workers = mutation("WorkerQL", metadata=metadata)
        merge_unannotated = mutation("WorkerQL")

        @strawberry.mutation(metadata=metadata)
        def hire_worker(self) -> bool:
            return True

        @strawberry.mutation
        def unannotated(self) -> bool:
            return True

    return berry.to_strawberry(strawberry_config=StrawberryConfig(auto_camel_case=camel))


@pytest.mark.parametrize("camel", [False, True])
@pytest.mark.parametrize("in_domain", [False, True])
@pytest.mark.parametrize("single", [False, True])
def test_generated_merge_metadata(camel, in_domain, single):
    schema = build_schema(camel=camel)
    gql_type = schema._schema.get_type("WorkforceDomainMutType") if in_domain else schema._schema.mutation_type
    field_name = (
        ("mergeWorker" if single else "mergeWorkers") if camel else ("merge_worker" if single else "merge_workers")
    )
    gql_field = gql_type.fields[field_name]
    assert {key: value for key, value in gql_field.extensions.items() if key != BACKREF} == METADATA
    assert gql_field.extensions[BACKREF].metadata == METADATA
    assert gql_field.description == ("Merge one worker." if single else None)
    # A nested domain mounts the same metadata-bearing mutation container.
    nested_type = get_named_type(schema._schema.get_type("ParentDomainMutType").fields["workforce"].type)
    assert nested_type is get_named_type(schema._schema.mutation_type.fields["workforce"].type)


@pytest.mark.parametrize("in_domain", [False, True])
def test_native_strawberry_mutation_metadata(in_domain):
    schema = build_schema()
    gql_type = schema._schema.get_type("WorkforceDomainMutType") if in_domain else schema._schema.mutation_type
    gql_field = gql_type.fields["hireWorker" if in_domain else "hire_worker"]
    assert {key: value for key, value in gql_field.extensions.items() if key != BACKREF} == METADATA
    if in_domain:
        assert gql_field.description == "Hire a worker."


@pytest.mark.parametrize("metadata", [None, {}])
def test_mutations_without_metadata_have_only_strawberry_backref(metadata):
    schema = build_schema(metadata)
    for gql_type in (schema._schema.mutation_type, schema._schema.get_type("WorkforceDomainMutType")):
        for gql_field in gql_type.fields.values():
            assert set(gql_field.extensions) == {BACKREF}


def test_metadata_does_not_leak_to_other_mutations():
    schema = build_schema()
    for gql_type in (schema._schema.mutation_type, schema._schema.get_type("WorkforceDomainMutType")):
        for name in ("merge_unannotated", "unannotated"):
            assert set(gql_type.fields[name].extensions) == {BACKREF}


def test_publishing_preserves_existing_extension_keys():
    schema = build_schema({**METADATA, BACKREF: "application value"})
    gql_field = schema._schema.mutation_type.fields["merge_worker"]
    definition = gql_field.extensions[BACKREF]
    gql_field.extensions["requiredSkills"] = ["existing.skill"]
    gql_field.extensions["other"] = "keep"
    publish_mutation_metadata(schema, ())
    assert gql_field.extensions[BACKREF] is definition
    assert gql_field.extensions["requiredSkills"] == ["existing.skill"]
    assert gql_field.extensions["other"] == "keep"
    assert gql_field.extensions["application"] == METADATA["application"]


@pytest.mark.parametrize("factory", [mutation, MutationDescriptor])
@pytest.mark.parametrize(
    "metadata",
    [
        {"invalid": object()},
        {"invalid": {"set"}},
        {"nested": [b"bytes"]},
        {"invalid": float("nan")},
        {"invalid": float("inf")},
        ["not a mapping"],
        {1: "not a string key"},
    ],
)
def test_invalid_metadata_fails_during_declaration(factory, metadata):
    with pytest.raises(ConfigurationError, match="JSON-serializable"):
        factory(target="WorkerQL", metadata=metadata)


def test_circular_metadata_fails_during_declaration():
    metadata = {}
    metadata["circular"] = metadata
    with pytest.raises(ConfigurationError, match="JSON-serializable"):
        mutation("WorkerQL", metadata=metadata)


def test_accepts_mapping_metadata():
    descriptor = mutation("WorkerQL", metadata=MappingProxyType(METADATA))
    assert descriptor.meta["metadata"] == METADATA


def test_metadata_does_not_change_sdl():
    assert build_schema().as_str() == build_schema(None).as_str()
