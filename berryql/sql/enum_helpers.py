"""Utilities to declare SQLAlchemy Enum columns safely and concisely.

This module provides:

- ensure_enum_hashable: make a Python Enum class hashable at runtime if needed.
- sa_enum_type: build a configured SQLAlchemy Enum type with safe defaults.
- enum_column: a convenience factory returning a Column with SAEnum attached.

Rationale
---------
Some Python Enum variants (or custom Enums) may end up without a usable
__hash__, which can trip SQLAlchemy's Enum type when it builds internal
lookup dictionaries. To avoid leaking that concern into model code, call
ensure_enum_hashable() before constructing SAEnum—or use enum_column(),
which does this automatically.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Type

import enum as _enum
from sqlalchemy import Column
from sqlalchemy import Enum as SAEnum


def ensure_enum_hashable(enum_cls: Type[_enum.Enum]) -> Type[_enum.Enum]:
    """Ensure the given Enum class has a working __hash__.

    To be maximally defensive across Python/Enum variants, we assign
    Enum.__hash__ directly. This is safe and preserves Enum semantics.
    """
    try:
        enum_cls.__hash__ = _enum.Enum.__hash__  # type: ignore[attr-defined]
    except Exception:
        try:
            # As an extreme fallback, define identity-based hash
            enum_cls.__hash__ = object.__hash__  # type: ignore[assignment]
        except Exception:
            pass
    return enum_cls


def _values_callable_for_storage_values(enum_cls: Type[_enum.Enum]) -> Callable[[Iterable[_enum.Enum]], list]:
    """Build a values_callable that stores enum.value when it's str, else name.

    This keeps DB storage stable and human-readable (e.g., "draft").
    """
    def _values(iterable: Iterable[_enum.Enum]) -> list:
        out = []
        for m in iterable:
            v = m.value
            out.append(v if isinstance(v, str) else m.name)
        return out

    return _values


def sa_enum_type(
    enum_cls: Type[_enum.Enum],
    *,
    native_enum: bool = False,
    validate_strings: bool = True,
    create_constraint: bool = True,
    constraint_name: Optional[str] = None,
) -> SAEnum:
    """Create a configured SQLAlchemy Enum type with safe defaults.

    Ensures enum hashability and uses values_callable to store string values
    (e.g., "draft") when available, falling back to names.
    """
    ensure_enum_hashable(enum_cls)
    return SAEnum(
        enum_cls,
        native_enum=native_enum,
        validate_strings=validate_strings,
        create_constraint=create_constraint,
        constraint_name=constraint_name,
        values_callable=_values_callable_for_storage_values(enum_cls),
    )


def enum_column(
    enum_cls: Type[_enum.Enum],
    *,
    nullable: bool = True,
    default: Optional[_enum.Enum] = None,
    constraint_name: Optional[str] = None,
    native_enum: bool = False,
    validate_strings: bool = True,
    create_constraint: bool = True,
    **column_kwargs,
) -> Column:
    """Convenience factory for a Column with a configured SAEnum.

    Example:

        from berryql.sql.enum_helpers import enum_column

        class Post(Base):
            status = enum_column(PostStatus, nullable=False, default=PostStatus.DRAFT,
                                 constraint_name="ck_post_status")
    """
    # Ensure hashability before constructing SAEnum
    ensure_enum_hashable(enum_cls)
    type_ = sa_enum_type(
        enum_cls,
        native_enum=native_enum,
        validate_strings=validate_strings,
        create_constraint=create_constraint,
        constraint_name=constraint_name,
    )
    return Column(type_, nullable=nullable, default=default, **column_kwargs)
