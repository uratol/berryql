"""Shared merge requirements and discoverable input-field metadata."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import strawberry
from sqlalchemy import Column

from .errors import ConfigurationError, ValidationError


_PENDING_DEFAULT = object()


class _PendingDefault(Exception):
    """A predicate needs a default that SQLAlchemy has not resolved yet."""


class _ScalarState:
    """Read-only scalar attributes, without ORM descriptors or relationships."""

    __slots__ = ("_values",)

    def __init__(self, values):
        self._values = values

    def __getattr__(self, name):
        if name not in self._values:
            raise AttributeError(name)
        value = self._values[name]
        if value is _PENDING_DEFAULT:
            raise _PendingDefault(name)
        return None if value is strawberry.UNSET else value


@dataclass(frozen=True)
class FieldRequirement:
    name: str
    source: str
    required_on_create: bool
    predicate: Callable[[Any], bool] | None
    rule: str | None
    allow_unknown: bool

    @property
    def metadata(self) -> dict[str, Any]:
        result = {
            "requiredOnCreate": self.required_on_create,
            "allowUnknown": self.allow_unknown,
        }
        if self.rule:
            result["rule"] = self.rule
        if self.predicate is not None:
            result["requiredWhen"] = self.rule
        return result

    def describe(self, description: str | None) -> str | None:
        if not (self.required_on_create or self.predicate is not None or self.rule):
            return description
        guidance = "BerryQL validation: " + json.dumps(self.metadata, ensure_ascii=False)
        return f"{description}\n\n{guidance}" if description else guidance


def field_requirement(name, field_def, column=None) -> FieldRequirement:
    meta = field_def.meta or {}
    required = meta.get("required")
    if required is None:
        required = bool(
            column is not None
            and not column.nullable
            and column.default is None
            and column.server_default is None
            and column.identity is None
            and column.computed is None
            and column is not getattr(column.table, "_autoincrement_column", None)
        )
    return FieldRequirement(
        name=name,
        source=meta.get("column") or name,
        required_on_create=required is True or required == "create",
        predicate=meta.get("required_when"),
        rule=meta.get("rule") or ("Required on create; cannot be cleared on update." if required else None),
        allow_unknown=bool(meta.get("allow_unknown")),
    )


def _column(model, source):
    mapper = getattr(model, "__mapper__", None)
    prop = mapper.column_attrs.get(source) if mapper is not None else None
    if prop is not None:
        column = prop.columns[0]
        return column if isinstance(column, Column) else None
    table = getattr(model, "__table__", None)
    return table.c.get(source) if table is not None else None


def requirements_for(berry_type):
    model = getattr(berry_type, "model", None)
    for name, field_def in (getattr(berry_type, "__berry_fields__", {}) or {}).items():
        meta = field_def.meta or {}
        if field_def.kind != "scalar" or name.startswith("_") or meta.get("read_only") or meta.get("computed"):
            continue
        column = _column(model, meta.get("column") or name)
        if column is None and not meta.get("write_only"):
            continue
        requirement = field_requirement(name, field_def, column)
        if requirement.required_on_create or requirement.predicate is not None:
            yield requirement


def input_field_name(schema, berry_type, name):
    input_type = schema._ensure_input_type(berry_type)
    definition = input_type.__strawberry_definition__
    field = next(f for f in definition.fields if f.python_name == name)
    converter = getattr(schema, "_name_converter", None)
    if converter is not None:
        return converter.from_field(field)
    if field.graphql_name:
        return field.graphql_name
    from .naming import to_camel

    return to_camel(name) if getattr(schema, "_auto_camel_case", False) else name


async def validate_required_fields(schema, berry_type, instance, values, *, creating, session, final=False):
    """Validate without mutating the row or giving predicates ORM access.

    ``values`` is the sanitized payload after pre-hooks and derived foreign
    keys. Missing keys retain UNSET semantics; null is an explicit answer.
    """
    requirements = list(requirements_for(berry_type))
    has_predicates = any(r.predicate is not None for r in requirements)
    if not requirements or (final and not has_predicates):
        return
    model = berry_type.model
    mapper = getattr(model, "__mapper__", None)
    loaded = dict(vars(instance))
    if (not creating or final) and has_predicates and mapper is not None:
        # Deferred/expired local columns must be fetched asynchronously once;
        # predicates themselves only receive ordinary scalar attributes.
        missing = [p.key for p in mapper.column_attrs if isinstance(p.columns[0], Column) and p.key not in loaded]
        if missing:
            with session.no_autoflush:
                await session.refresh(instance, attribute_names=missing)
            loaded = dict(vars(instance))
    state = {}
    if mapper is not None:
        for prop in mapper.column_attrs:
            value = loaded.get(prop.key, strawberry.UNSET)
            column = prop.columns[0]
            if creating and not final and value is strawberry.UNSET and isinstance(column, Column):
                if column.default is not None and column.default.is_scalar:
                    value = column.default.arg
                elif column.default is not None or column.server_default is not None:
                    value = _PENDING_DEFAULT
            state[prop.key] = value
    if not final:
        state.update(values)
    for name, field_def in berry_type.__berry_fields__.items():
        if field_def.kind != "scalar":
            continue
        source = (field_def.meta or {}).get("column") or name
        if not final or (field_def.meta or {}).get("write_only"):
            state[name] = values.get(name, state.get(source, strawberry.UNSET))
        else:
            state[name] = state.get(source, strawberry.UNSET)
    state = {name: value.value if isinstance(value, Enum) else value for name, value in state.items()}
    record = _ScalarState(state)
    fields = []
    rules = {}
    for requirement in requirements:
        supplied = values.get(requirement.name, values.get(requirement.source, strawberry.UNSET))
        value = state.get(requirement.name, strawberry.UNSET)
        active = requirement.required_on_create and (creating or supplied is not strawberry.UNSET)
        if requirement.predicate is not None:
            try:
                conditional = requirement.predicate(record)
            except _PendingDefault:
                # The final pass after flush sees real callable/server defaults.
                conditional = False
            if inspect.isawaitable(conditional):
                if inspect.iscoroutine(conditional):
                    conditional.close()
                raise ConfigurationError("required_when must return a synchronous boolean")
            active = active or bool(conditional)
        missing = value is strawberry.UNSET or (value is None and not requirement.allow_unknown)
        if creating and requirement.required_on_create and supplied is strawberry.UNSET:
            missing = True
        if creating and supplied is strawberry.UNSET and value is None:
            # A database NULL is not evidence of an explicit unknown answer.
            missing = True
        if active and missing:
            public_name = input_field_name(schema, berry_type, requirement.name)
            fields.append(public_name)
            rules[public_name] = requirement.rule or "Required on create; cannot be cleared on update."
    if fields:
        raise ValidationError(fields=fields, rules=rules)


def publish_input_requirements(graphql_schema):
    """Expose JSON-safe metadata to schema tools using graphql-core fields.

    Standard GraphQL introspection exposes the same metadata in descriptions.
    """
    from graphql import GraphQLInputObjectType
    from strawberry.schema.schema_converter import GraphQLCoreConverter

    for gql_type in graphql_schema._schema.type_map.values():
        if not isinstance(gql_type, GraphQLInputObjectType):
            continue
        for field in gql_type.fields.values():
            definition = (field.extensions or {}).get(GraphQLCoreConverter.DEFINITION_BACKREF)
            metadata = getattr(definition, "metadata", {}) or {}
            if "requiredOnCreate" in metadata:
                field.extensions.update(metadata)
