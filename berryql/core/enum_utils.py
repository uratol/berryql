from __future__ import annotations

from typing import Any, Optional


def get_model_enum_cls(model_cls: Any, key: str) -> Optional[type]:
    """Return the Python Enum class for a given model column if declared via Column.info['python_enum']."""
    if model_cls is None or not key:
        return None
    try:
        col = getattr(getattr(model_cls, '__table__', None).c, key)
    except Exception:
        col = None
    if col is None:
        return None
    try:
        info = getattr(col, 'info', {}) or {}
        enum_cls = info.get('python_enum')
    except Exception:
        enum_cls = None
    return enum_cls if enum_cls is not None else None


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
    """Normalize enum fields on an ORM instance to storage (enum.value) before flush."""
    if model_cls is None or instance is None:
        return
    try:
        cols = getattr(getattr(model_cls, '__table__', None), 'columns', []) or []
    except Exception:
        cols = []
    for _col in cols:
        try:
            info = getattr(_col, 'info', {}) or {}
            enum_cls = info.get('python_enum')
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
