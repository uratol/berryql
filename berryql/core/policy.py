from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, get_args, get_origin

from strawberry.extensions.field_extension import FieldExtension

from .naming import from_camel
from .permissions import FieldPermissions
from .utils import to_where_dict


class AuthorizationError(PermissionError):
    """A caller requested a capability that the effective policy denies."""


class PolicyConfigurationError(ValueError):
    """A field policy provider or capability reference is invalid."""


def _is_list_annotation(annotation: Any) -> bool:
    """Return whether an output annotation is a list, unwrapping wrappers."""

    if annotation is None:
        return False
    if isinstance(annotation, str):
        normalized = annotation.replace(" ", "").lower()
        return "list[" in normalized or "typing.list[" in normalized
    if annotation is list:
        return True
    if annotation.__class__.__name__ == "StrawberryList":
        return True
    wrapped = getattr(annotation, "of_type", None)
    if wrapped is not None and wrapped is not annotation:
        return _is_list_annotation(wrapped)
    origin = get_origin(annotation)
    if origin is list:
        return True
    return any(_is_list_annotation(arg) for arg in get_args(annotation))


class SelectGuardExtension(FieldExtension):
    """Final select guard for ordinary Strawberry fields on a Berry type."""

    def __init__(
        self,
        engine: "PolicyEngine",
        berry_type: Any,
        field_name: str,
        annotation: Any = None,
    ) -> None:
        self.engine = engine
        self.berry_type = berry_type
        self.field_name = field_name
        self._is_list = _is_list_annotation(annotation)

    def apply(self, field: Any) -> None:
        self._is_list = self._is_list or _is_list_annotation(getattr(field, "type", None))

    async def resolve_async(self, next_: Any, source: Any, info: Any, **kwargs: Any) -> Any:
        allowed = await self.engine.allows_field(self.berry_type, info, "select", self.field_name)
        if not allowed:
            return [] if self._is_list else None
        result = next_(source, info, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


@dataclass
class _ExecutionPolicyCache:
    operation: Any
    variable_values: Any
    values: Dict[Any, Any] = field(default_factory=dict)


class PolicyEngine:
    """Resolve, cache and enforce BerryQL field/operation policies.

    Providers are request/type policies. They must not depend on a particular
    row or resolver path. The cache stores only the current GraphQL execution
    observed in a context, so reusing a context cannot leak a prior policy and
    the cache cannot grow without bound.
    """

    _CACHE_ATTR = "_berryql_policy_cache"
    _FIELD_CAPABILITIES = {"select", "filter", "order", "create", "update"}
    _OPERATIONS = {"create", "update", "delete", "replace"}

    def __init__(self, schema: Any):
        self.schema = schema

    @staticmethod
    def _context(info: Any) -> Any:
        return getattr(info, "context", None)

    @staticmethod
    def _execution_parts(info: Any) -> tuple[Any, Any]:
        raw_info = getattr(info, "_raw_info", None) or info
        return (
            getattr(raw_info, "operation", None),
            getattr(raw_info, "variable_values", None),
        )

    def _execution_cache(self, info: Any) -> Optional[Dict[Any, Any]]:
        context = self._context(info)
        if context is None:
            return None
        operation, variable_values = self._execution_parts(info)
        try:
            state = (
                context.get(self._CACHE_ATTR) if isinstance(context, dict) else getattr(context, self._CACHE_ATTR, None)
            )
            if not isinstance(state, _ExecutionPolicyCache) or not (
                state.operation is operation and state.variable_values is variable_values
            ):
                state = _ExecutionPolicyCache(operation, variable_values)
                if isinstance(context, dict):
                    context[self._CACHE_ATTR] = state
                else:
                    setattr(context, self._CACHE_ATTR, state)
            return state.values
        except Exception:
            return None

    @staticmethod
    async def _call_provider(provider: Any, berry_type: Any, info: Any) -> FieldPermissions:
        result = provider
        if callable(provider):
            result = provider(berry_type, info)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, FieldPermissions):
            raise PolicyConfigurationError(
                "field_permissions provider must return FieldPermissions, " f"got {type(result).__name__}"
            )
        return result

    async def _resolve_uncached(self, berry_type: Any, info: Any) -> FieldPermissions:
        providers = []
        global_provider = getattr(self.schema, "_field_permissions_provider", None)
        local_provider = getattr(berry_type, "__field_permissions_provider__", None)
        if global_provider is not None:
            providers.append(global_provider)
        if local_provider is not None and local_provider is not global_provider:
            providers.append(local_provider)
        permissions = FieldPermissions.allow_all()
        for provider in providers:
            provided = await self._call_provider(provider, berry_type, info)
            self._validate_permissions(berry_type, provided)
            permissions = permissions.intersection(provided)
        self._validate_permissions(berry_type, permissions)
        return permissions

    @staticmethod
    def _declared_field_names(berry_type: Any) -> set[str]:
        names = set((getattr(berry_type, "__berry_fields__", {}) or {}).keys())
        names.update((getattr(berry_type, "__annotations__", {}) or {}).keys())
        for name, value in vars(berry_type).items():
            if name.startswith("_"):
                continue
            module = str(getattr(value.__class__, "__module__", "") or "")
            if module.startswith("strawberry") or hasattr(value, "base_resolver") or hasattr(value, "resolver"):
                names.add(name)
        return names

    def _validate_permissions(self, berry_type: Any, permissions: FieldPermissions) -> None:
        known = self._declared_field_names(berry_type)
        for capability in self._FIELD_CAPABILITIES:
            mask = getattr(permissions, capability)
            unknown = sorted(name for name in mask.fields if name not in known)
            if unknown:
                raise PolicyConfigurationError(
                    f"Unknown canonical field(s) for {capability} on "
                    f"{getattr(berry_type, '__name__', berry_type)}: {unknown}"
                )

    async def resolve(self, berry_type: Any, info: Any) -> FieldPermissions:
        if berry_type is None:
            return FieldPermissions.allow_all()
        cache = self._execution_cache(info)
        if cache is None:
            return await self._resolve_uncached(berry_type, info)
        key = (self, berry_type)
        cached = cache.get(key)
        if cached is not None:
            if inspect.isawaitable(cached):
                return await cached
            return cached
        task = asyncio.create_task(self._resolve_uncached(berry_type, info))
        cache[key] = task
        try:
            resolved = await task
        except BaseException:
            if cache.get(key) is task:
                cache.pop(key, None)
            raise
        cache[key] = resolved
        return resolved

    def canonical_field_name(self, berry_type: Any, reference: str) -> str:
        fields = getattr(berry_type, "__berry_fields__", {}) or {}
        if reference in fields:
            return reference
        candidate = from_camel(str(reference))
        if candidate in fields:
            return candidate
        for field_name, field_def in fields.items():
            meta = getattr(field_def, "meta", {}) or {}
            if meta.get("column") in {reference, candidate}:
                return field_name
        # Plain Strawberry fields are canonicalized the same way even though
        # they do not have a Berry descriptor.
        annotations = getattr(berry_type, "__annotations__", {}) or {}
        if reference in annotations:
            return str(reference)
        if candidate in annotations or hasattr(berry_type, candidate):
            return candidate
        return candidate

    async def allows_field(self, berry_type: Any, info: Any, capability: str, field_name: str) -> bool:
        if capability not in self._FIELD_CAPABILITIES:
            raise PolicyConfigurationError(f"Unknown field capability '{capability}'")
        permissions = await self.resolve(berry_type, info)
        return self.allows_resolved_field(berry_type, permissions, capability, field_name)

    def allows_resolved_field(
        self,
        berry_type: Any,
        permissions: FieldPermissions,
        capability: str,
        field_name: str,
    ) -> bool:
        if capability not in self._FIELD_CAPABILITIES:
            raise PolicyConfigurationError(f"Unknown field capability '{capability}'")
        canonical = self.canonical_field_name(berry_type, field_name)
        return getattr(permissions, capability).allows(canonical)

    async def require_field(self, berry_type: Any, info: Any, capability: str, field_name: str) -> None:
        canonical = self.canonical_field_name(berry_type, field_name)
        if not await self.allows_field(berry_type, info, capability, canonical):
            type_name = getattr(berry_type, "__name__", berry_type)
            raise AuthorizationError(f"Field '{canonical}' is not allowed for {capability} on {type_name}")

    async def require_path(self, berry_type: Any, info: Any, capability: str, path: str) -> None:
        current_type = berry_type
        parts = str(path).split(".")
        for index, raw_part in enumerate(parts):
            part = raw_part.strip()
            if not part:
                raise PolicyConfigurationError(f"Invalid empty field path in '{path}'")
            field_name = self.canonical_field_name(current_type, part)
            await self.require_field(current_type, info, capability, field_name)
            field_def = (getattr(current_type, "__berry_fields__", {}) or {}).get(field_name)
            if field_def is None or getattr(field_def, "kind", None) != "relation":
                if index != len(parts) - 1:
                    raise PolicyConfigurationError(f"Field path '{path}' crosses non-relation '{field_name}'")
                continue
            target_name = (getattr(field_def, "meta", {}) or {}).get("target")
            target_type = self.schema.types.get(target_name) if target_name else None
            if target_type is None:
                raise PolicyConfigurationError(f"Unknown relation target for '{field_name}' in path '{path}'")
            current_type = target_type

    async def require_operation(self, berry_type: Any, info: Any, operation: str) -> None:
        if operation not in self._OPERATIONS:
            raise PolicyConfigurationError(f"Unknown mutation operation '{operation}'")
        permissions = await self.resolve(berry_type, info)
        if not permissions.operations.allows(operation):
            raise AuthorizationError(
                f"Operation '{operation}' is not allowed on " f"{getattr(berry_type, '__name__', berry_type)}"
            )

    async def sanitize_payload(
        self,
        berry_type: Any,
        info: Any,
        data: Mapping[str, Any],
        *,
        capability: str,
        identity_name: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if capability not in {"create", "update"}:
            raise PolicyConfigurationError("Payload capability must be 'create' or 'update'")
        permissions = await self.resolve(berry_type, info)
        mask = getattr(permissions, capability)
        declared = getattr(berry_type, "__berry_fields__", {}) or {}
        sanitized: Dict[str, Any] = {}
        ignored: Dict[str, Any] = {}
        for input_name, input_value in data.items():
            if (
                input_name in {"_Delete", "_Replace", "_Insert"}
                or input_name == identity_name
                or input_name not in declared
                or mask.allows(input_name)
            ):
                sanitized[input_name] = input_value
            else:
                ignored[input_name] = input_value
        return sanitized, ignored

    async def validate_where_fields(self, berry_type: Any, info: Any, where: Any) -> None:
        if where is None:
            return
        where_dict = to_where_dict(
            where,
            strict=True,
            model_cls=getattr(berry_type, "model", None),
            auto_camel_case=bool(getattr(self.schema, "_auto_camel_case", False)),
        )
        for column_name in (where_dict or {}).keys():
            await self.require_path(berry_type, info, "filter", str(column_name))


__all__ = [
    "AuthorizationError",
    "PolicyConfigurationError",
    "PolicyEngine",
    "SelectGuardExtension",
]
