"""Normalize database numeric values at the GraphQL output boundary."""

from decimal import Decimal
from inspect import isawaitable
from typing import Any

from strawberry.extensions import FieldExtension
from strawberry.types.base import StrawberryList, StrawberryOptional


def has_float_output(output_type: Any) -> bool:
    """Match Float and nullable/list wrappers without changing other scalars."""
    while isinstance(output_type, (StrawberryOptional, StrawberryList)):
        output_type = output_type.of_type
    return output_type is float


def normalize_float_output(value: Any, output_type: Any) -> Any:
    if value is None:
        return None
    if isinstance(output_type, StrawberryOptional):
        return normalize_float_output(value, output_type.of_type)
    if isinstance(output_type, StrawberryList):
        return [normalize_float_output(item, output_type.of_type) for item in value]
    if output_type is float and isinstance(value, Decimal):
        # GraphQL-core 3.3 no longer coerces Decimal to Float. Keep Decimal in
        # ORM state and let GraphQL reject non-finite values after conversion.
        return float(value)
    return value


class FloatOutputExtension(FieldExtension):
    """Convert Decimal only for fields declared as GraphQL Float."""

    def apply(self, field: Any) -> None:
        self.output_type = field.type

    def resolve(self, next_: Any, source: Any, info: Any, **kwargs: Any) -> Any:
        return normalize_float_output(next_(source, info, **kwargs), self.output_type)

    async def resolve_async(self, next_: Any, source: Any, info: Any, **kwargs: Any) -> Any:
        result = next_(source, info, **kwargs)
        if isawaitable(result):
            result = await result
        return normalize_float_output(result, self.output_type)
