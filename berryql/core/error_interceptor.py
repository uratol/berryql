"""Exception interception and translation for BerryQL schemas.

Library users can register handlers that intercept exceptions raised during
GraphQL execution (queries, mutations and subscriptions) and translate them
into client-friendly GraphQL errors. A typical use case is mapping a
SQLAlchemy ``IntegrityError`` raised by a merge mutation to a clean
"user-facing" message instead of leaking driver internals to API clients.

Handlers are registered on the :class:`~berryql.registry.BerrySchema`::

    berry_schema = BerrySchema()

    @berry_schema.error_handler(IntegrityError)
    def translate_integrity(exc, context):
        return UserFacingError(
            "This email address is already registered",
            code="EMAIL_TAKEN",
        )

    # Catch-all fallback for anything without a specific handler:
    @berry_schema.error_handler
    def translate_unexpected(exc, context):
        if isinstance(exc, SQLAlchemyError):
            return UserFacingError("Database error", code="DB_ERROR")
        return None  # keep the original error

The mechanism is implemented by returning a ``strawberry.Schema`` subclass
(:class:`BerryStrawberrySchema`) from ``BerrySchema.to_strawberry()``; the
subclass post-processes execution results and rewrites matching
``GraphQLError`` entries in place, so it works with any Strawberry-compatible
server (FastAPI, ASGI views, direct ``schema.execute`` calls, ...).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, List, Optional, Tuple

import strawberry
from graphql import GraphQLError

_logger = logging.getLogger("berryql")

# Sentinels
_UNRESOLVED = object()

# A handler may return: None (keep the original error), a replacement message
# (str) or a replacement exception instance (its ``str()`` becomes the GraphQL
# error message; an optional ``extensions`` dict is copied onto the error).
HandlerOutcome = Any


class _Registration:
    """Internal record for one registered error handler."""

    __slots__ = ("handler", "exc_types", "arity")

    def __init__(
        self,
        handler: Callable[..., Any],
        exc_types: Optional[Tuple[type, ...]],
        arity: int,
    ) -> None:
        self.handler = handler
        self.exc_types = exc_types  # None means catch-all
        self.arity = arity  # 1: (exc), 2: (exc, context), 3: (exc, context, gql_error)


def _detect_arity(handler: Callable[..., Any]) -> int:
    """Best-effort detection of how many positional args a handler accepts."""
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):  # builtins / C callables without introspection
        return 1
    count = 0
    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            count += 1
        elif param.kind == param.VAR_POSITIONAL:
            return 3
    if count <= 0:
        return 1
    return min(count, 3)


class ErrorInterceptor:
    """Ordered registry of exception handlers plus result translation logic.

    The first type-specific registration whose exception types match the
    raised exception (via ``isinstance``) handles it; a registration without
    exception types acts as a catch-all fallback that applies only when no
    type-specific registration matched. Within each tier, registration order
    is preserved.
    """

    def __init__(self) -> None:
        self._registrations: List[_Registration] = []

    def __bool__(self) -> bool:
        return bool(self._registrations)

    def __len__(self) -> int:
        return len(self._registrations)

    def register(self, handler: Callable[..., Any], *exc_types: type) -> None:
        """Register ``handler`` for exceptions matching ``exc_types``.

        Without ``exc_types`` the handler becomes a catch-all fallback.
        """
        if not callable(handler):
            raise TypeError("error handler must be callable")
        for exc_type in exc_types:
            if not (isinstance(exc_type, type) and issubclass(exc_type, BaseException)):
                raise TypeError(
                    "error handler exception filters must be exception classes, "
                    f"got {exc_type!r}"
                )
        self._registrations.append(
            _Registration(handler, tuple(exc_types) if exc_types else None, _detect_arity(handler))
        )

    # --- matching / invocation -------------------------------------------------

    def _match(self, exc: BaseException) -> Optional[_Registration]:
        # Type-specific registrations always win over catch-all fallbacks,
        # regardless of registration order; within each tier the earliest
        # registration handles the exception.
        for reg in self._registrations:
            if reg.exc_types is not None and isinstance(exc, reg.exc_types):
                return reg
        for reg in self._registrations:
            if reg.exc_types is None:
                return reg
        return None

    @staticmethod
    def _call(
        reg: _Registration,
        exc: BaseException,
        context: Any,
        gql_error: GraphQLError,
    ) -> Any:
        if reg.arity >= 3:
            return reg.handler(exc, context, gql_error)
        if reg.arity == 2:
            return reg.handler(exc, context)
        return reg.handler(exc)

    # --- public translation entry points ----------------------------------------

    async def translate_result_async(self, result: Any, context: Any = None) -> Any:
        """Translate errors of an execution result in place (async handlers OK)."""
        errors = getattr(result, "errors", None)
        if not errors or not self._registrations:
            return result
        for gql_error in list(errors):
            original = getattr(gql_error, "original_error", None)
            if original is None:
                continue
            reg = self._match(original)
            if reg is None:
                continue
            try:
                outcome = self._call(reg, original, context, gql_error)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
            except BaseException:  # noqa: BLE001 - handler bugs must not break the API
                _logger.exception(
                    "BerryQL error handler %r failed while processing %r; "
                    "keeping the original error",
                    reg.handler,
                    original,
                )
                continue
            _apply_replacement(gql_error, outcome)
        return result

    def translate_result_sync(self, result: Any, context: Any = None) -> Any:
        """Translate errors of an execution result in place (sync execution path)."""
        errors = getattr(result, "errors", None)
        if not errors or not self._registrations:
            return result
        for gql_error in list(errors):
            original = getattr(gql_error, "original_error", None)
            if original is None:
                continue
            reg = self._match(original)
            if reg is None:
                continue
            try:
                outcome = self._call(reg, original, context, gql_error)
                if inspect.isawaitable(outcome):
                    outcome = self._resolve_awaitable_sync(outcome, reg)
                    if outcome is _UNRESOLVED:
                        continue
            except BaseException:  # noqa: BLE001 - handler bugs must not break the API
                _logger.exception(
                    "BerryQL error handler %r failed while processing %r; "
                    "keeping the original error",
                    reg.handler,
                    original,
                )
                continue
            _apply_replacement(gql_error, outcome)
        return result

    @staticmethod
    def _resolve_awaitable_sync(outcome: Any, reg: _Registration) -> Any:
        """Await an async handler result from ``execute_sync`` if possible."""
        try:
            asyncio.get_running_loop()
            _logger.warning(
                "BerryQL async error handler %r cannot be awaited during "
                "synchronous execution (event loop is running); keeping the "
                "original error",
                reg.handler,
            )
            return _UNRESOLVED
        except RuntimeError:
            pass

        async def _await_it():
            return await outcome

        try:
            return asyncio.run(_await_it())
        except Exception:  # noqa: BLE001
            _logger.exception(
                "BerryQL failed to await async error handler %r during "
                "synchronous execution; keeping the original error",
                reg.handler,
            )
            return _UNRESOLVED


def _apply_replacement(gql_error: GraphQLError, outcome: Any) -> None:
    """Rewrite a ``GraphQLError`` in place according to the handler outcome."""
    if outcome is None:
        return
    if isinstance(outcome, str):
        gql_error.message = outcome
        return
    if isinstance(outcome, BaseException):
        try:
            message = str(outcome) or type(outcome).__name__
        except Exception:  # noqa: BLE001 - defensive: broken __str__
            message = type(outcome).__name__
        gql_error.message = message
        extensions = getattr(outcome, "extensions", None)
        gql_error.extensions = dict(extensions) if isinstance(extensions, dict) else {}
        gql_error.original_error = outcome
        return
    # Any other value is coerced to a message.
    try:
        gql_error.message = str(outcome)
    except Exception:  # noqa: BLE001
        pass


def _context_from_call_args(args: tuple, kwargs: dict) -> Any:
    """Extract ``context_value`` from ``execute(query, variable_values, context_value, ...)``."""
    if "context_value" in kwargs:
        return kwargs.get("context_value")
    if len(args) >= 2:  # (variable_values, context_value, root_value, ...)
        return args[1]
    return None


class BerryStrawberrySchema(strawberry.Schema):
    """``strawberry.Schema`` subclass routing execution errors through BerryQL.

    Returned by ``BerrySchema.to_strawberry()``. Overrides the three execution
    entry points (``execute``, ``execute_sync``, ``subscribe``) so that every
    runtime error of an operation passes through the schema's
    :class:`ErrorInterceptor` before the result reaches the client.
    """

    def __init__(
        self,
        *args: Any,
        berry_error_interceptor: Optional[ErrorInterceptor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._berry_error_interceptor = (
            berry_error_interceptor
            if berry_error_interceptor is not None
            else ErrorInterceptor()
        )

    @property
    def berry_error_interceptor(self) -> ErrorInterceptor:
        return self._berry_error_interceptor

    async def execute(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        result = await super().execute(query, *args, **kwargs)
        if self._berry_error_interceptor:
            await self._berry_error_interceptor.translate_result_async(
                result, _context_from_call_args(args, kwargs)
            )
        return result

    def execute_sync(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        result = super().execute_sync(query, *args, **kwargs)
        if self._berry_error_interceptor:
            self._berry_error_interceptor.translate_result_sync(
                result, _context_from_call_args(args, kwargs)
            )
        return result

    async def subscribe(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        stream = await super().subscribe(query, *args, **kwargs)
        interceptor = self._berry_error_interceptor
        if not interceptor:
            return stream
        context = _context_from_call_args(args, kwargs)

        async def _translated_stream():
            async for result in stream:
                await interceptor.translate_result_async(result, context)
                yield result

        return _translated_stream()


__all__ = ["BerryStrawberrySchema", "ErrorInterceptor"]
