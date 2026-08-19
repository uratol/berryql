"""BerryQL public API and lightweight lazy exports.

This __init__ avoids importing heavy submodules at import time to prevent
circular imports when foundational layers (like database models) import
BerryQL utilities (e.g., enum helpers).

Exposes:
- get_active_schema, set_active_schema (without importing registry eagerly)
- Lazy attributes: BerrySchema, BerryType, BerryDomain, PaginationConfig, StrawberryConfig
- Lazy functions: field, relation, aggregate, count, custom, custom_object, domain, mutation
- enum_column (resolved lazily from .sql.enum_helpers)
"""
from __future__ import annotations

from typing import Any

_ACTIVE_SCHEMA: Any = None


def set_active_schema(schema: Any) -> None:
    global _ACTIVE_SCHEMA
    _ACTIVE_SCHEMA = schema


def get_active_schema() -> Any:
    if _ACTIVE_SCHEMA is None:
        raise RuntimeError("Active Berry schema not set. Ensure schema.py initialized it.")
    return _ACTIVE_SCHEMA


def __getattr__(name: str):  # PEP 562 lazy exports
    # Some IDEs/debuggers probe submodules via attribute access (pkg.registry).
    # Return the actual submodule to avoid noisy AttributeError during introspection.
    import importlib as _importlib
    if name == 'registry':
        return _importlib.import_module(__name__ + '.registry')
    if name == 'mutations':
        return _importlib.import_module(__name__ + '.mutations')
    if name in {'BerrySchema', 'BerryType', 'BerryDomain', 'PaginationConfig', 'StrawberryConfig'}:
        _registry = _importlib.import_module(__name__ + '.registry')
        return getattr(_registry, name)
    if name in {'FieldSet', 'FieldPermissions', 'OperationPermissions'}:
        from .core import permissions as _permissions
        return getattr(_permissions, name)
    if name in {'UserFacingError', 'BerryQLError', 'ConfigurationError', 'AuthorizationDenied'}:
        from .core import errors as _errors
        return getattr(_errors, name)
    if name in {'ErrorInterceptor', 'BerryStrawberrySchema'}:
        from .core import error_interceptor as _error_interceptor
        return getattr(_error_interceptor, name)
    if name in {'FilterLimits', 'FilterSpec', 'OperatorRegistry', 'register_operator'}:
        from .core import filters as _filters
        return getattr(_filters, name)
    if name in {'encode_cursor', 'decode_cursor'}:
        from .core import pagination as _pagination
        return getattr(_pagination, name)
    if name in {'MergeOperationContext', 'MergeNodeContext', 'DeferredValidation'}:
        _context = _importlib.import_module(__name__ + '.context')
        return getattr(_context, name)
    if name in {'field', 'relation', 'aggregate', 'count', 'custom', 'custom_object', 'domain', 'mutation', 'hooks', 'scope'}:
        from .core import fields as _fields
        if name not in {'hooks'}:
            return getattr(_fields, name)
        # hooks is provided by registry
        _registry = _importlib.import_module(__name__ + '.registry')
        return getattr(_registry, 'hooks')
    if name == 'enum_column':
        from .sql.enum_helpers import enum_column as _enum_column
        return _enum_column
    raise AttributeError(name)


__all__ = [
    'BerrySchema', 'BerryType', 'BerryDomain', 'PaginationConfig', 'StrawberryConfig',
    'FieldSet', 'FieldPermissions', 'OperationPermissions',
    'FilterLimits', 'FilterSpec', 'OperatorRegistry', 'register_operator',
    'encode_cursor', 'decode_cursor',
    'MergeOperationContext', 'MergeNodeContext', 'DeferredValidation',
    'UserFacingError', 'BerryQLError', 'ConfigurationError', 'AuthorizationDenied',
    'ErrorInterceptor', 'BerryStrawberrySchema',
    'field', 'relation', 'aggregate', 'count', 'custom', 'custom_object', 'domain', 'mutation', 'hooks', 'scope',
    'enum_column',
    'get_active_schema', 'set_active_schema', 'registry', 'mutations',
]

# --- Runtime patch: mark Strawberry mutations with a stable flag ----------------------
# Some Strawberry versions do not reliably expose is_mutation on the public field object.
# We patch strawberry.mutation to set a stable __berry_is_mutation__ attribute on the
# resulting field (and on nested objects when present). This runs as soon as BerryQL is
# imported (schema loads BerryQL before domains are declared), so domain-level
# @strawberry.mutation decorators will carry the marker for precise classification.
try:  # best-effort, no hard dependency on Strawberry internals
    import strawberry as _strawberry  # type: ignore
    _orig_mutation = getattr(_strawberry, 'mutation', None)
    if callable(_orig_mutation):
        def _berry_marked_mutation(*args, **kwargs):  # type: ignore
            f = _orig_mutation(*args, **kwargs)
            try:
                setattr(f, '__berry_is_mutation__', True)
                # Also try to tag nested bits Strawberry may attach
                try:
                    fd = getattr(f, 'field_definition', None)
                    if fd is not None:
                        setattr(fd, '__berry_is_mutation__', True)
                except Exception:
                    pass
                try:
                    br = getattr(f, 'base_resolver', None)
                    if br is not None:
                        setattr(br, '__berry_is_mutation__', True)
                except Exception:
                    pass
                try:
                    rv = getattr(f, 'resolver', None)
                    if rv is not None:
                        setattr(rv, '__berry_is_mutation__', True)
                except Exception:
                    pass
            except Exception:
                pass
            return f
        try:
            setattr(_strawberry, 'mutation', _berry_marked_mutation)
        except Exception:
            pass
except Exception:
    pass
