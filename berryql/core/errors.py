"""Typed internal errors used across BerryQL's query pipeline.

The classes deliberately inherit from the historical builtin exception types so
existing applications keep their error handling behaviour while internal code
can distinguish configuration, caller input, authorization and adapter gaps.
"""


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


__all__ = [
    "AdapterUnsupported",
    "AuthorizationDenied",
    "BerryQLError",
    "ConfigurationError",
    "InvalidOrderingError",
    "InvalidPredicateError",
]
