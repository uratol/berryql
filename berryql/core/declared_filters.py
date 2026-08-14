from __future__ import annotations

import inspect
from typing import Any, Mapping

from .errors import InvalidPredicateError
from .utils import coerce_where_value


class DeclaredFilterCompiler:
    """Compile declared relation/root filters with one async contract."""

    def __init__(self, schema: Any):
        self.schema = schema

    async def apply(
        self,
        statement: Any,
        *,
        model_cls: Any,
        berry_type: Any,
        info: Any,
        specs: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> Any:
        for argument_name, raw_value in (values or {}).items():
            if raw_value is None:
                continue
            spec = (specs or {}).get(argument_name)
            if spec is None:
                raise InvalidPredicateError(f"Unknown filter argument: {argument_name}")
            for dependency in getattr(spec, "dependencies", ()):
                await self.schema._policy_engine.require_path(berry_type, info, "filter", str(dependency))
            value = raw_value
            transform = getattr(spec, "transform", None)
            if transform is not None:
                try:
                    value = transform(value)
                    if inspect.isawaitable(value):
                        value = await value
                except Exception as exc:
                    raise InvalidPredicateError(f"Filter transform failed for {argument_name}: {exc}") from exc
            expression = None
            builder = getattr(spec, "builder", None)
            if builder is not None:
                try:
                    expression = builder(model_cls, info, value)
                    if inspect.isawaitable(expression):
                        expression = await expression
                except Exception as exc:
                    raise InvalidPredicateError(f"Filter builder failed for {argument_name}: {exc}") from exc
            elif getattr(spec, "column", None):
                column_name = str(spec.column)
                try:
                    column = model_cls.__table__.c.get(column_name)
                except (AttributeError, TypeError):
                    column = None
                if column is None:
                    raise InvalidPredicateError(f"Unknown filter column: {column_name} for argument {argument_name}")
                operator_name = spec.op or "eq"
                operator = self.schema._predicate_compiler.operators.get(operator_name)
                if operator is None:
                    raise InvalidPredicateError(
                        f"Unknown filter operator: {operator_name} for argument {argument_name}"
                    )
                self.schema._predicate_compiler._validate_operand_shape(column_name, operator_name, value)
                try:
                    if operator_name in {"in", "not_in", "between", "not_between"}:
                        coerced = [coerce_where_value(column, item) for item in value]
                    else:
                        coerced = coerce_where_value(column, value)
                    expression = operator(column, coerced)
                except Exception as exc:
                    raise InvalidPredicateError(f"Filter operation failed for {argument_name}: {exc}") from exc
            else:
                raise InvalidPredicateError(f"Filter '{argument_name}' must declare column or builder")
            if expression is None:
                raise InvalidPredicateError(f"Filter '{argument_name}' produced no predicate")
            statement = statement.where(expression)
        return statement


__all__ = ["DeclaredFilterCompiler"]
