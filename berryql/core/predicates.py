from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import and_

from .errors import AdapterUnsupported, InvalidPredicateError
from .filters import FilterLimits, OperatorRegistry
from .utils import coerce_where_value, to_where_dict


class PredicateError(InvalidPredicateError):
    """Base error for predicate resolution or compilation."""


class UnsupportedPredicateError(PredicateError, AdapterUnsupported):
    """The selected adapter cannot enforce a predicate safely."""


@dataclass(frozen=True)
class Predicate:
    """Marker base for the immutable predicate intermediate representation."""


@dataclass(frozen=True)
class AbsentPredicate(Predicate):
    pass


@dataclass(frozen=True)
class ColumnPredicate(Predicate):
    column: str
    operator: str
    value: Any


@dataclass(frozen=True)
class TrustedExpression(Predicate):
    expression: Any


@dataclass(frozen=True)
class PredicateProvider(Predicate):
    provider: Any


@dataclass(frozen=True)
class Conjunction(Predicate):
    fragments: Tuple[Predicate, ...]


@dataclass
class _PredicateExecutionCache:
    operation: Any
    variable_values: Any
    values: Dict[Any, Any] = field(default_factory=dict)


class _ProviderIdentity:
    """Hash an unhashable provider by identity while retaining the object."""

    __slots__ = ("provider",)

    def __init__(self, provider: Any):
        self.provider = provider

    def __hash__(self) -> int:
        return id(self.provider)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _ProviderIdentity) and self.provider is other.provider


class PredicateCompiler:
    """Resolve and compile caller predicates and trusted scopes uniformly."""

    _CACHE_ATTR = "_berryql_predicate_cache"

    def __init__(self, schema: Any):
        self.schema = schema
        self.operators = OperatorRegistry(getattr(schema, "_operators", None))
        limits = getattr(schema, "filter_limits", None)
        self.limits = limits if isinstance(limits, FilterLimits) else FilterLimits()

    def _execution_cache(self, info: Any) -> Optional[Dict[Any, Any]]:
        context = getattr(info, "context", None)
        if context is None:
            return None
        raw_info = getattr(info, "_raw_info", None) or info
        operation = getattr(raw_info, "operation", None)
        variable_values = getattr(raw_info, "variable_values", None)
        try:
            state = (
                context.get(self._CACHE_ATTR) if isinstance(context, dict) else getattr(context, self._CACHE_ATTR, None)
            )
            if not isinstance(state, _PredicateExecutionCache) or not (
                state.operation is operation and state.variable_values is variable_values
            ):
                state = _PredicateExecutionCache(operation, variable_values)
                if isinstance(context, dict):
                    context[self._CACHE_ATTR] = state
                else:
                    setattr(context, self._CACHE_ATTR, state)
            return state.values
        except Exception:
            return None

    def _provider_cache_key(self, provider: Any, model_cls: Any) -> tuple[Any, ...]:
        try:
            hash(provider)
            provider_key = provider
        except TypeError:
            provider_key = _ProviderIdentity(provider)
        return (self, model_cls, provider_key)

    @staticmethod
    def resolve_graphql_value(info: Any, raw: Any) -> Any:
        if isinstance(raw, Predicate):
            return raw
        if isinstance(raw, (list, tuple)):
            return type(raw)(PredicateCompiler.resolve_graphql_value(info, item) for item in raw)
        value = raw
        if not isinstance(value, (dict, str)) and hasattr(value, "name"):
            name_obj = getattr(value, "name", None)
            variable_name = getattr(name_obj, "value", None) or name_obj
            variable_values = getattr(info, "variable_values", None)
            if variable_values is None:
                raw_info = getattr(info, "_raw_info", None)
                variable_values = getattr(raw_info, "variable_values", None) if raw_info is not None else None
            if isinstance(variable_values, dict) and variable_name in variable_values:
                value = variable_values[variable_name]
        if not isinstance(value, (dict, str)) and hasattr(value, "value"):
            value = getattr(value, "value")
        return value

    def parse(
        self,
        value: Any,
        model_cls: Any,
        *,
        strict: bool = True,
        trusted: bool = True,
    ) -> Predicate:
        if isinstance(value, Predicate):
            return value
        if value is None:
            return AbsentPredicate()
        if callable(value):
            return PredicateProvider(value)
        if isinstance(value, (list, tuple)):
            if not trusted and self.limits.max_depth is not None:
                self._validate_depth(value, self.limits.max_depth)
            return Conjunction(
                tuple(
                    self.parse(
                        part,
                        model_cls,
                        strict=strict,
                        trusted=trusted,
                    )
                    for part in value
                )
            )
        if isinstance(value, (dict, str)):
            if not trusted and self.limits.max_json_length is not None and isinstance(value, (str, bytes)):
                if len(value) > self.limits.max_json_length:
                    raise PredicateError(
                        f"Where JSON exceeds max_json_length={self.limits.max_json_length}"
                    )
            where_dict = to_where_dict(
                value,
                strict=strict,
                model_cls=model_cls,
                auto_camel_case=bool(getattr(self.schema, "_auto_camel_case", False)),
            )
            if not trusted and self.limits.max_depth is not None:
                self._validate_depth(where_dict, self.limits.max_depth)
            fragments = []
            for column_name, operator_map in (where_dict or {}).items():
                if not isinstance(operator_map, dict):
                    if strict:
                        raise PredicateError(f"Where operators for '{column_name}' must be an object")
                    continue
                for operator, operand in operator_map.items():
                    self._validate_operand_shape(str(column_name), str(operator), operand)
                    fragments.append(ColumnPredicate(str(column_name), str(operator), operand))
            if not trusted and self.limits.max_clauses is not None and len(fragments) > self.limits.max_clauses:
                raise PredicateError(
                    f"Where clause count {len(fragments)} exceeds max_clauses={self.limits.max_clauses}"
                )
            return Conjunction(tuple(fragments))
        if not trusted:
            raise PredicateError("Caller where must be a JSON object or JSON string")
        return TrustedExpression(value)

    def _validate_operand_shape(self, column: str, operator: str, operand: Any) -> None:
        if operator in {"between", "not_between"}:
            if not isinstance(operand, (list, tuple)) or len(operand) != 2:
                raise PredicateError(
                    f"Where operator '{operator}' for '{column}' requires exactly two values"
                )
        elif operator in {"in", "not_in"}:
            if not isinstance(operand, (list, tuple, set)) or not operand:
                raise PredicateError(
                    f"Where operator '{operator}' for '{column}' requires a non-empty list"
                )
            if self.limits.max_in_items is not None and len(operand) > self.limits.max_in_items:
                raise PredicateError(
                    f"Where operator '{operator}' for '{column}' exceeds max_in_items={self.limits.max_in_items}"
                )

    @staticmethod
    def _validate_depth(value: Any, maximum: int, depth: int = 0) -> None:
        if depth > maximum:
            raise PredicateError(f"Where nesting depth exceeds max_depth={maximum}")
        if isinstance(value, dict):
            for key, nested in value.items():
                PredicateCompiler._validate_depth(key, maximum, depth + 1)
                PredicateCompiler._validate_depth(nested, maximum, depth + 1)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                PredicateCompiler._validate_depth(nested, maximum, depth + 1)

    async def resolve(
        self,
        value: Any,
        model_cls: Any,
        info: Any,
        *,
        strict: bool = True,
        trusted: bool = True,
    ) -> Predicate:
        value = self.resolve_graphql_value(info, value)
        parsed = self.parse(value, model_cls, strict=strict, trusted=trusted)
        return await self._resolve_predicate(parsed, model_cls, info, strict=strict, trusted=trusted)

    async def _resolve_predicate(
        self,
        predicate: Predicate,
        model_cls: Any,
        info: Any,
        *,
        strict: bool,
        trusted: bool,
    ) -> Predicate:
        if isinstance(predicate, PredicateProvider):
            # The public scope contract is exactly (model_cls, info).  A
            # TypeError raised inside the provider must propagate unchanged and
            # must never trigger a second invocation with different arguments.
            cache = self._execution_cache(info)
            key = self._provider_cache_key(predicate.provider, model_cls)
            if cache is not None and key in cache:
                cached = cache[key]
                if inspect.isawaitable(cached):
                    return await cached
                return cached

            async def _invoke() -> Predicate:
                result = predicate.provider(model_cls, info)
                if inspect.isawaitable(result):
                    result = await result
                return await self.resolve(
                    result,
                    model_cls,
                    info,
                    strict=strict,
                    trusted=trusted,
                )

            if cache is None:
                return await _invoke()
            task = asyncio.create_task(_invoke())
            cache[key] = task
            try:
                resolved = await task
            except BaseException:
                if cache.get(key) is task:
                    cache.pop(key, None)
                raise
            cache[key] = resolved
            return resolved
        if isinstance(predicate, Conjunction):
            resolved = []
            for fragment in predicate.fragments:
                resolved.append(
                    await self._resolve_predicate(
                        fragment,
                        model_cls,
                        info,
                        strict=strict,
                        trusted=trusted,
                    )
                )
            return Conjunction(tuple(resolved))
        return predicate

    def resolve_sync(
        self,
        value: Any,
        model_cls: Any,
        info: Any,
        *,
        strict: bool = True,
        trusted: bool = True,
    ) -> Predicate:
        value = self.resolve_graphql_value(info, value)
        predicate = self.parse(value, model_cls, strict=strict, trusted=trusted)
        if isinstance(predicate, PredicateProvider):
            cache = self._execution_cache(info)
            key = self._provider_cache_key(predicate.provider, model_cls)
            if cache is not None and key in cache:
                cached = cache[key]
                if inspect.isawaitable(cached):
                    raise UnsupportedPredicateError("Predicate provider is still resolving asynchronously")
                return cached
            if inspect.iscoroutinefunction(predicate.provider):
                raise UnsupportedPredicateError("Async predicate provider requires an async compilation path")
            result = predicate.provider(model_cls, info)
            if inspect.isawaitable(result):

                async def _finish_async_result() -> Predicate:
                    awaited = await result
                    return await self.resolve(
                        awaited,
                        model_cls,
                        info,
                        strict=strict,
                        trusted=trusted,
                    )

                try:
                    task = asyncio.get_running_loop().create_task(_finish_async_result())
                except RuntimeError:
                    if inspect.iscoroutine(result):
                        result.close()
                else:
                    if cache is not None:
                        cache[key] = task
                raise UnsupportedPredicateError("Async predicate provider requires an async compilation path")
            resolved = self.resolve_sync(
                result,
                model_cls,
                info,
                strict=strict,
                trusted=trusted,
            )
            if cache is not None:
                cache[key] = resolved
            return resolved
        if isinstance(predicate, Conjunction):
            return Conjunction(
                tuple(
                    self.resolve_sync(
                        fragment,
                        model_cls,
                        info,
                        strict=strict,
                        trusted=trusted,
                    )
                    for fragment in predicate.fragments
                )
            )
        return predicate

    @staticmethod
    def _column(model_cls: Any, name: str, strict: bool) -> Any:
        try:
            column = model_cls.__table__.c.get(name)
        except Exception:
            column = None
        if column is None and strict:
            raise PredicateError(f"Unknown where column: {name}")
        return column

    def compile_sqlalchemy(self, predicate: Predicate, model_cls: Any, *, strict: bool = True) -> Any:
        if isinstance(predicate, AbsentPredicate):
            return None
        if isinstance(predicate, TrustedExpression):
            return predicate.expression
        if isinstance(predicate, PredicateProvider):
            raise PredicateError("Predicate provider was not resolved before compilation")
        if isinstance(predicate, ColumnPredicate):
            column = self._column(model_cls, predicate.column, strict)
            if column is None:
                return None
            operator = self.operators.get(predicate.operator)
            if operator is None:
                if strict:
                    raise PredicateError(f"Unknown where operator: {predicate.operator}")
                return None
            operand = predicate.value
            if predicate.operator in {
                "in",
                "between",
                "not_in",
                "not_between",
            } and isinstance(operand, (list, tuple)):
                operand = [coerce_where_value(column, item) for item in operand]
            else:
                operand = coerce_where_value(column, operand)
            return operator(column, operand)
        if isinstance(predicate, Conjunction):
            expressions = [
                self.compile_sqlalchemy(fragment, model_cls, strict=strict) for fragment in predicate.fragments
            ]
            expressions = [expr for expr in expressions if expr is not None]
            return and_(*expressions) if expressions else None
        raise PredicateError(f"Unknown predicate node {type(predicate).__name__}")

    async def apply(
        self,
        statement: Any,
        model_cls: Any,
        value: Any,
        info: Any,
        *,
        strict: bool = True,
        trusted: bool = True,
    ) -> Any:
        predicate = await self.resolve(
            value,
            model_cls,
            info,
            strict=strict,
            trusted=trusted,
        )
        expression = self.compile_sqlalchemy(predicate, model_cls, strict=strict)
        return statement.where(expression) if expression is not None else statement

    def apply_sync(
        self,
        statement: Any,
        model_cls: Any,
        value: Any,
        info: Any,
        *,
        strict: bool = True,
        trusted: bool = True,
    ) -> Any:
        predicate = self.resolve_sync(
            value,
            model_cls,
            info,
            strict=strict,
            trusted=trusted,
        )
        expression = self.compile_sqlalchemy(predicate, model_cls, strict=strict)
        return statement.where(expression) if expression is not None else statement

    def compile_mssql(
        self,
        predicate: Predicate,
        model_cls: Any,
        adapter: Any,
        *,
        strict: bool = True,
    ) -> list[str]:
        if isinstance(predicate, AbsentPredicate):
            return []
        if isinstance(predicate, PredicateProvider):
            raise PredicateError("Predicate provider was not resolved before compilation")
        if isinstance(predicate, TrustedExpression):
            try:
                from sqlalchemy.dialects import mssql

                return [
                    str(
                        predicate.expression.compile(
                            dialect=mssql.dialect(),
                            compile_kwargs={"literal_binds": True},
                        )
                    )
                ]
            except Exception as exc:
                raise UnsupportedPredicateError("MSSQL adapter cannot safely compile SQLAlchemy predicate") from exc
        if isinstance(predicate, ColumnPredicate):
            # Validate with the SQLAlchemy compiler first, then let the adapter
            # produce its native literal form.
            self.compile_sqlalchemy(predicate, model_cls, strict=strict)
            return adapter.where_from_dict(
                model_cls,
                {predicate.column: {predicate.operator: predicate.value}},
            )
        if isinstance(predicate, Conjunction):
            parts: list[str] = []
            for fragment in predicate.fragments:
                parts.extend(self.compile_mssql(fragment, model_cls, adapter, strict=strict))
            return parts
        raise PredicateError(f"Unknown predicate node {type(predicate).__name__}")

    def compile_mssql_sync(
        self,
        value: Any,
        model_cls: Any,
        adapter: Any,
        info: Any,
        *,
        strict: bool = True,
        trusted: bool = True,
    ) -> list[str]:
        predicate = self.resolve_sync(
            value,
            model_cls,
            info,
            strict=strict,
            trusted=trusted,
        )
        return self.compile_mssql(predicate, model_cls, adapter, strict=strict)


__all__ = [
    "AbsentPredicate",
    "ColumnPredicate",
    "Conjunction",
    "Predicate",
    "PredicateCompiler",
    "PredicateError",
    "PredicateProvider",
    "TrustedExpression",
    "UnsupportedPredicateError",
]
