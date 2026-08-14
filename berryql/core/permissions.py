from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Iterable


def _normalize_names(values: tuple[Any, ...]) -> FrozenSet[str]:
    if len(values) == 1 and not isinstance(values[0], str):
        candidate = values[0]
        if isinstance(candidate, Iterable):
            values = tuple(candidate)
    return frozenset(str(value) for value in values)


@dataclass(frozen=True)
class FieldSet:
    """An allow-list or deny-list of canonical BerryQL field names."""

    mode: str
    fields: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.mode not in {"only", "except"}:
            raise ValueError("FieldSet mode must be 'only' or 'except'")
        object.__setattr__(self, "fields", frozenset(str(name) for name in self.fields))

    @classmethod
    def all(cls) -> "FieldSet":
        return cls("except", frozenset())

    @classmethod
    def none(cls) -> "FieldSet":
        return cls("only", frozenset())

    @classmethod
    def only(cls, *names: Any) -> "FieldSet":
        return cls("only", _normalize_names(names))

    @classmethod
    def all_except(cls, *names: Any) -> "FieldSet":
        return cls("except", _normalize_names(names))

    def allows(self, name: str) -> bool:
        normalized = str(name)
        if self.mode == "only":
            return normalized in self.fields
        return normalized not in self.fields

    def intersection(self, other: "FieldSet") -> "FieldSet":
        if not isinstance(other, FieldSet):
            raise TypeError("Can only intersect FieldSet with FieldSet")
        if self.mode == "only" and other.mode == "only":
            return FieldSet.only(self.fields.intersection(other.fields))
        if self.mode == "only" and other.mode == "except":
            return FieldSet.only(self.fields.difference(other.fields))
        if self.mode == "except" and other.mode == "only":
            return FieldSet.only(other.fields.difference(self.fields))
        return FieldSet.all_except(self.fields.union(other.fields))


@dataclass(frozen=True)
class FieldPermissions:
    """Read and write field masks resolved for one BerryQL type."""

    read: FieldSet = field(default_factory=FieldSet.all)
    write: FieldSet = field(default_factory=FieldSet.all)

    def __post_init__(self) -> None:
        if not isinstance(self.read, FieldSet):
            raise TypeError("FieldPermissions.read must be a FieldSet")
        if not isinstance(self.write, FieldSet):
            raise TypeError("FieldPermissions.write must be a FieldSet")

    @classmethod
    def allow_all(cls) -> "FieldPermissions":
        return cls()

    def intersection(self, other: "FieldPermissions") -> "FieldPermissions":
        if not isinstance(other, FieldPermissions):
            raise TypeError("Can only intersect FieldPermissions with FieldPermissions")
        return FieldPermissions(
            read=self.read.intersection(other.read),
            write=self.write.intersection(other.write),
        )


class FieldPermissionResolver:
    """Resolve and request-cache schema/type field permission providers."""

    _CACHE_ATTR = "_berryql_field_permissions_cache"

    def __init__(self, schema: Any):
        self.schema = schema

    @staticmethod
    def _context_cache(info: Any) -> dict[Any, Any] | None:
        context = getattr(info, "context", None)
        if isinstance(context, dict):
            cache = context.get(FieldPermissionResolver._CACHE_ATTR)
            if cache is None:
                cache = {}
                context[FieldPermissionResolver._CACHE_ATTR] = cache
            return cache
        if context is not None:
            try:
                cache = getattr(context, FieldPermissionResolver._CACHE_ATTR, None)
                if cache is None:
                    cache = {}
                    setattr(context, FieldPermissionResolver._CACHE_ATTR, cache)
                return cache
            except Exception:
                pass
        return None

    @staticmethod
    async def _call_provider(provider: Any, berry_type: Any, info: Any) -> FieldPermissions:
        result = provider
        if callable(provider):
            result = provider(berry_type, info)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, FieldPermissions):
            raise TypeError("field_permissions provider must return FieldPermissions, " f"got {type(result).__name__}")
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
            permissions = permissions.intersection(await self._call_provider(provider, berry_type, info))
        return permissions

    async def resolve(self, berry_type: Any, info: Any) -> FieldPermissions:
        if berry_type is None:
            return FieldPermissions.allow_all()
        cache = self._context_cache(info)
        if cache is None:
            return await self._resolve_uncached(berry_type, info)
        raw_info = getattr(info, "_raw_info", None) or info
        operation = getattr(raw_info, "operation", None)
        variable_values = getattr(raw_info, "variable_values", None)
        execution_token = (id(operation), id(variable_values))
        key = (execution_token, id(self.schema), berry_type)
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
