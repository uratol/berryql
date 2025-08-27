from __future__ import annotations

from typing import Any, Optional
try:
    from sqlalchemy.sql.sqltypes import Enum as SAEnumType  # type: ignore
except Exception:  # pragma: no cover
    SAEnumType = object  # type: ignore


def get_model_enum_cls(model_cls: Any, key: str) -> Optional[type]:
    """Return the Python Enum class for a given model column when column type is SAEnum."""
    if model_cls is None or not key:
        return None
    try:
        col = getattr(getattr(model_cls, '__table__', None).c, key)
    except Exception:
        col = None
    if col is None:
        return None
    try:
        sa_t = getattr(col, 'type', None)
        if isinstance(sa_t, SAEnumType):
            enum_cls = getattr(sa_t, 'enum_class', None)
            if enum_cls is not None:
                # Defensive: make sure enum is hashable for any downstream maps
                try:
                    from berryql.sql.enum_helpers import ensure_enum_hashable
                    ensure_enum_hashable(enum_cls)
                except Exception:
                    pass
            return enum_cls
    except Exception:
        return None
    return None


def coerce_input_to_storage_value(enum_cls: Any, value: Any) -> Any:
    """Coerce incoming GraphQL/mutation value to DB storage value for enum columns.

    Rules:
    - If value is an Enum (any), return its .value (fallback to .name).
    - If value is a string, try NAME then VALUE against enum_cls.
    - Otherwise, return as-is.
    """
    if enum_cls is None:
        return value
    try:
        from enum import Enum as _PyEnum
        if isinstance(value, _PyEnum):
            return getattr(value, 'value', getattr(value, 'name', value))
        if isinstance(value, str):
            # Prefer NAME first for GraphQL inputs, then VALUE
            try:
                return enum_cls[value].value
            except Exception:
                try:
                    return enum_cls(value).value
                except Exception:
                    return value
    except Exception:
        return value
    return value


def coerce_mapping_to_enum(enum_cls: Any, value: Any) -> Any:
    """Coerce DB/raw mapping value to Python Enum instance for output hydration.

    Rules:
    - If already of enum_cls or other Enum, keep or convert to enum_cls if string.
    - If string, try VALUE first (DB stores value), then NAME.
    - Otherwise, return as-is.
    """
    if enum_cls is None:
        return value
    try:
        from enum import Enum as _PyEnum
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, _PyEnum):
            return value
        if isinstance(value, str):
            try:
                return enum_cls(value)
            except Exception:
                try:
                    return enum_cls[value]
                except Exception:
                    return value
    except Exception:
        return value
    return value


def normalize_instance_enums(model_cls: Any, instance: Any) -> None:
    """Normalize enum fields on an ORM instance to storage (enum.value) before flush.

    Only SAEnum-backed columns are considered.
    """
    if model_cls is None or instance is None:
        return
    try:
        cols = getattr(getattr(model_cls, '__table__', None), 'columns', []) or []
    except Exception:
        cols = []
    for _col in cols:
        enum_cls = None
        try:
            sa_t = getattr(_col, 'type', None)
            if isinstance(sa_t, SAEnumType):
                enum_cls = getattr(sa_t, 'enum_class', None)
        except Exception:
            enum_cls = None
        if enum_cls is None:
            continue
        try:
            cur = getattr(instance, _col.name, None)
        except Exception:
            cur = None
        try:
            new_val = coerce_input_to_storage_value(enum_cls, cur)
            if new_val is not cur:
                setattr(instance, _col.name, new_val)
        except Exception:
            continue
