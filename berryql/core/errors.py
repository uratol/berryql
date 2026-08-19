"""Typed internal errors used across BerryQL's query pipeline.

The classes deliberately inherit from the historical builtin exception types so
existing applications keep their error handling behaviour while internal code
can distinguish configuration, caller input, authorization and adapter gaps.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class BerryQLError(Exception):
    """Base class for BerryQL-owned failures."""


class ConfigurationError(BerryQLError, ValueError):
    """Invalid schema or declaration configuration."""


class InvalidPredicateError(BerryQLError, ValueError):
    """Invalid caller predicate or declared filter."""


class InvalidOrderingError(BerryQLError, ValueError):
    """Invalid caller ordering input."""


class AuthorizationDenied(BerryQLError, PermissionError):
    """An operation was rejected by the effective policy."""


class AdapterUnsupported(BerryQLError, NotImplementedError):
    """An adapter cannot safely implement a requested operation."""


class UserFacingError(BerryQLError):
    """Exception intended to become a clean, client-facing GraphQL error.

    Returned from an error handler registered via
    ``berry_schema.register_error_handler`` / ``@berry_schema.error_handler``,
    it replaces the original (e.g. raw SQL) exception: its message becomes the
    GraphQL error message and its ``extensions`` are attached to the error
    payload (exposed to clients as the ``extensions`` field of the error).

    Example::

        @berry_schema.error_handler(IntegrityError)
        def translate_integrity(exc, context):
            return UserFacingError(
                "This email address is already registered",
                code="EMAIL_TAKEN",
            )
    """

    default_code = "USER_FACING_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        extensions: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        merged: Dict[str, Any] = {"code": code or self.default_code}
        if extensions:
            merged.update(extensions)
        self.extensions = merged


__all__ = [
    "AdapterUnsupported",
    "AuthorizationDenied",
    "BerryQLError",
    "ConfigurationError",
    "InvalidOrderingError",
    "InvalidPredicateError",
    "UserFacingError",
]
