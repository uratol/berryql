from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple

from .errors import InvalidOrderingError


class OrderingError(InvalidOrderingError):
    """Invalid or unsupported ordering input."""


@dataclass(frozen=True)
class OrderTerm:
    path: Any
    direction: str
    nulls: Optional[str] = None

    def __post_init__(self) -> None:
        if self.nulls not in {None, "first", "last"}:
            raise OrderingError("nulls must be 'first', 'last', or None")


class OrderingCompiler:
    """Normalize, authorize and compile deterministic ordering terms."""

    def __init__(self, schema: Any):
        self.schema = schema

    @staticmethod
    def _value(raw: Any) -> Any:
        return getattr(raw, "value", raw)

    @classmethod
    def direction(cls, value: Any, *, default: str = "asc") -> str:
        raw = default if value is None else cls._value(value)
        normalized = str(raw).strip().lower()
        if normalized not in {"asc", "desc"}:
            raise OrderingError(f"Invalid order direction '{raw}'; expected 'asc' or 'desc'")
        return normalized

    def parse_multi(self, values: Any, *, default_direction: str = "asc", strict: bool = True) -> Tuple[OrderTerm, ...]:
        if values is None:
            return ()
        if not isinstance(values, (list, tuple)):
            if strict:
                raise OrderingError("order_multi must be a list of 'field:direction' terms")
            values = [values]
        terms = []
        flattened = []
        for raw_term in values:
            resolved_term = self._value(raw_term)
            if isinstance(resolved_term, (list, tuple)):
                flattened.extend(resolved_term)
            else:
                flattened.append(resolved_term)
        for index, raw_term in enumerate(flattened):
            value = self._value(raw_term)
            if not isinstance(value, str):
                if strict:
                    raise OrderingError(f"Invalid order_multi term at index {index}: {value!r}")
                value = str(value)
            value = value.strip()
            if not value:
                raise OrderingError(f"Invalid empty order_multi term at index {index}")
            if ":" in value:
                pieces = value.split(":")
                if len(pieces) not in {2, 3} or not pieces[1]:
                    raise OrderingError(
                        f"Invalid order_multi term '{value}'; expected 'field:direction[:nulls_first|nulls_last]'"
                    )
                path, direction = pieces[0], pieces[1]
                nulls = pieces[2] if len(pieces) == 3 else None
                if nulls is not None:
                    normalized_nulls = str(nulls).strip().lower()
                    if normalized_nulls not in {"nulls_first", "nulls_last", "first", "last"}:
                        raise OrderingError(
                            f"Invalid null ordering '{nulls}'; expected nulls_first or nulls_last"
                        )
                    nulls = "first" if normalized_nulls in {"nulls_first", "first"} else "last"
            else:
                if strict:
                    raise OrderingError(f"Invalid order_multi term '{value}'; expected 'field:direction'")
                path, direction = value, default_direction
                nulls = None
            path = path.strip()
            if not path or any(not part for part in path.split(".")):
                raise OrderingError(f"Invalid order path '{path}'")
            terms.append(OrderTerm(path, self.direction(direction, default=default_direction), nulls))
        return tuple(terms)

    def parse(
        self,
        *,
        order_by: Any = None,
        order_dir: Any = None,
        order_multi: Any = None,
        default_direction: str = "asc",
        strict_multi: bool = True,
    ) -> Tuple[OrderTerm, ...]:
        multi = self.parse_multi(
            order_multi,
            default_direction=default_direction,
            strict=strict_multi,
        )
        if multi:
            return multi
        if order_by is None:
            return ()
        if callable(order_by):
            return (OrderTerm(order_by, self.direction(order_dir)),)
        raw_path = self._value(order_by)
        if not isinstance(raw_path, str) or not raw_path.strip():
            # Trusted SQLAlchemy ordering expressions remain additive API
            # compatibility for relation/type defaults.
            if raw_path is not None and not isinstance(raw_path, str):
                return (OrderTerm(raw_path, self.direction(order_dir)),)
            raise OrderingError("order_by must be a non-empty field path")
        path = raw_path.strip()
        if any(not part for part in path.split(".")):
            raise OrderingError(f"Invalid order path '{path}'")
        return (OrderTerm(path, self.direction(order_dir)),)

    async def validate(self, berry_type: Any, info: Any, terms: Iterable[OrderTerm]) -> None:
        for term in terms:
            if not isinstance(term.path, str):
                continue
            normalized = self.schema._normalize_order_path(berry_type, term.path)
            if normalized is None:
                allowed = self.schema._get_allowed_order_fields(berry_type)
                raise OrderingError(f"Invalid order_by '{term.path}'. Allowed: {allowed}")
            await self.schema._policy_engine.require_path(berry_type, info, "order", normalized)

    async def validate_caller(
        self,
        berry_type: Any,
        info: Any,
        *,
        order_by: Any,
        order_dir: Any,
        order_multi: Any,
    ) -> Tuple[OrderTerm, ...]:
        terms = self.parse(
            order_by=order_by,
            order_dir=order_dir,
            order_multi=order_multi,
            strict_multi=True,
        )
        await self.validate(berry_type, info, terms)
        return terms

    @staticmethod
    def _resolve_trusted_expression(path: Any, model_cls: Any, info: Any = None) -> Any:
        if not callable(path):
            return path
        expression = path(model_cls, info)
        if inspect.isawaitable(expression):
            if inspect.iscoroutine(expression):
                expression.close()
            raise OrderingError("Async ordering providers require an async planning path")
        return expression

    def apply_sqlalchemy(
        self,
        statement: Any,
        *,
        model_cls: Any,
        berry_type: Any,
        terms: Iterable[OrderTerm],
        resolve_expression: Callable[[Any, str, dict[str, Any]], tuple[Any, Any]],
        info: Any = None,
        add_pk_tiebreaker: bool = False,
        fallback_pk: bool = False,
        join_cache: Optional[dict[str, Any]] = None,
    ) -> Any:
        resolved_terms = tuple(terms)
        join_cache = join_cache if join_cache is not None else {}
        ordered_paths: list[str] = []
        applied = False
        for term in resolved_terms:
            if isinstance(term.path, str):
                normalized = self.schema._normalize_order_path(berry_type, term.path)
                if normalized is None:
                    allowed = self.schema._get_allowed_order_fields(berry_type)
                    raise OrderingError(f"Invalid order_by '{term.path}'. Allowed: {allowed}")
                statement, expression = resolve_expression(statement, normalized, join_cache)
                ordered_paths.append(normalized)
            else:
                expression = self._resolve_trusted_expression(term.path, model_cls, info)
            if expression is None:
                raise OrderingError(f"Unable to resolve order expression {term.path!r}")
            ordered_expression = expression.desc() if term.direction == "desc" else expression.asc()
            if term.nulls == "first":
                ordered_expression = ordered_expression.nulls_first()
            elif term.nulls == "last":
                ordered_expression = ordered_expression.nulls_last()
            statement = statement.order_by(ordered_expression)
            applied = True

        pk_name = self.schema._get_pk_name(model_cls)
        has_pk = any(path == pk_name for path in ordered_paths)
        if (fallback_pk and not applied) or (add_pk_tiebreaker and not has_pk):
            pk_expression = self.schema._get_pk_column(model_cls)
            statement = statement.order_by(pk_expression.asc())
        return statement

    async def apply_python(
        self,
        items: Iterable[Any],
        terms: Iterable[OrderTerm],
        resolve_value: Callable[[Any, str], Any],
        nulls_first: Optional[Callable[[str], bool]] = None,
    ) -> list[Any]:
        """Apply stable fallback ordering with caller-supplied dialect null rules."""

        ordered = list(items)
        for term in reversed(tuple(terms)):
            if not isinstance(term.path, str):
                raise OrderingError("Python fallback cannot evaluate a SQL ordering expression")
            non_null = []
            null_items = []
            for item in ordered:
                value = resolve_value(item, term.path)
                if inspect.isawaitable(value):
                    value = await value
                if value is None:
                    null_items.append(item)
                else:
                    non_null.append((value, item))
            non_null.sort(key=lambda pair: pair[0], reverse=term.direction == "desc")
            ordered_non_null = [item for _, item in non_null]
            put_nulls_first = (
                term.nulls == "first"
                or (term.nulls is None and nulls_first is not None and nulls_first(term.direction))
            )
            if put_nulls_first:
                ordered = null_items + ordered_non_null
            else:
                ordered = ordered_non_null + null_items
        return ordered

    def normalize_multi_values(self, values: Any) -> list[str]:
        """Compatibility facade used by internal relation configuration paths."""

        return [
            f"{term.path}:{term.direction}"
            + (f":nulls_{term.nulls}" if term.nulls else "")
            for term in self.parse_multi(values, strict=False)
        ]


__all__ = ["OrderTerm", "OrderingCompiler", "OrderingError"]
