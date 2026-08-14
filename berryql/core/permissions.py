from __future__ import annotations

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
class OperationPermissions:
    """Capabilities for mutation operations on one BerryQL type."""

    create: bool = True
    update: bool = True
    delete: bool = True
    replace: bool = True

    def __post_init__(self) -> None:
        for name in ("create", "update", "delete", "replace"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"OperationPermissions.{name} must be a bool")

    @classmethod
    def allow_all(cls) -> "OperationPermissions":
        return cls()

    @classmethod
    def deny_all(cls) -> "OperationPermissions":
        return cls(create=False, update=False, delete=False, replace=False)

    def allows(self, operation: str) -> bool:
        if operation not in {"create", "update", "delete", "replace"}:
            raise ValueError(f"Unknown mutation operation '{operation}'")
        return bool(getattr(self, operation))

    def intersection(self, other: "OperationPermissions") -> "OperationPermissions":
        if not isinstance(other, OperationPermissions):
            raise TypeError("Can only intersect OperationPermissions with OperationPermissions")
        return OperationPermissions(
            create=self.create and other.create,
            update=self.update and other.update,
            delete=self.delete and other.delete,
            replace=self.replace and other.replace,
        )


@dataclass(frozen=True)
class FieldPermissions:
    """Capability masks resolved for one BerryQL type.

    ``filter`` and ``order`` are normalized through ``select`` at construction
    time.  A field that cannot be returned therefore cannot be used as a
    caller-controlled filtering or ordering side channel.
    """

    select: FieldSet = field(default_factory=FieldSet.all)
    filter: FieldSet = field(default_factory=FieldSet.all)
    order: FieldSet = field(default_factory=FieldSet.all)
    create: FieldSet = field(default_factory=FieldSet.all)
    update: FieldSet = field(default_factory=FieldSet.all)
    operations: OperationPermissions = field(default_factory=OperationPermissions)

    def __post_init__(self) -> None:
        for name in ("select", "filter", "order", "create", "update"):
            if not isinstance(getattr(self, name), FieldSet):
                raise TypeError(f"FieldPermissions.{name} must be a FieldSet")
        if not isinstance(self.operations, OperationPermissions):
            raise TypeError("FieldPermissions.operations must be an OperationPermissions")
        object.__setattr__(self, "filter", self.select.intersection(self.filter))
        object.__setattr__(self, "order", self.select.intersection(self.order))

    @classmethod
    def allow_all(cls) -> "FieldPermissions":
        return cls()

    def intersection(self, other: "FieldPermissions") -> "FieldPermissions":
        if not isinstance(other, FieldPermissions):
            raise TypeError("Can only intersect FieldPermissions with FieldPermissions")
        return FieldPermissions(
            select=self.select.intersection(other.select),
            filter=self.filter.intersection(other.filter),
            order=self.order.intersection(other.order),
            create=self.create.intersection(other.create),
            update=self.update.intersection(other.update),
            operations=self.operations.intersection(other.operations),
        )


__all__ = ["FieldSet", "FieldPermissions", "OperationPermissions"]
