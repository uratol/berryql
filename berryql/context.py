from __future__ import annotations

import inspect
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Optional, Set, Tuple, Type


class MergeHookContext(dict):
    """Dictionary-compatible type-hook context with attribute access."""

    @property
    def operation(self):
        return self["operation"]

    @property
    def node(self):
        return self["node"]

    @property
    def parent(self):
        return self.get("parent")

    @property
    def relation(self):
        return self.get("relation")

    @property
    def delete(self):
        return bool(self.get("delete"))


@dataclass
class DeferredValidation:
    """One deduplicated validation scheduled for the end of a merge operation."""

    key: Hashable
    callback: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeOperationContext:
    """State shared by every node and root item in one merge mutation."""

    root_model: Type[Any]
    root_berry_type: Type[Any]
    payload_is_list: bool
    session: Any
    state: Dict[str, Any] = field(default_factory=dict)
    touched: Dict[Type[Any], Set[Any]] = field(default_factory=dict)
    deferred_validators: "OrderedDict[Hashable, DeferredValidation]" = field(
        default_factory=OrderedDict
    )

    def defer_validation(
        self,
        key: Hashable,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Schedule validation once for ``key``, preserving first-registration order."""

        hash(key)
        if key in self.deferred_validators:
            return
        if not callable(callback):
            raise TypeError("Deferred validation callback must be callable")
        self.deferred_validators[key] = DeferredValidation(
            key=key,
            callback=callback,
            args=tuple(args),
            kwargs=dict(kwargs),
        )

    def record_touched(self, model: Type[Any], primary_key: Any) -> None:
        """Record a materialized primary key for a model touched by this operation."""

        if primary_key is None:
            return
        self.touched.setdefault(model, set()).add(primary_key)

    async def run_deferred_validators(self, info: Any) -> None:
        """Run scheduled validators in registration order."""

        for validation in self.deferred_validators.values():
            kwargs = dict(validation.kwargs)
            try:
                parameters = inspect.signature(validation.callback).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "info" in parameters and "info" not in kwargs:
                kwargs["info"] = info
            if "operation" in parameters and "operation" not in kwargs:
                kwargs["operation"] = self
            result = validation.callback(*validation.args, **kwargs)
            if inspect.isawaitable(result):
                await result


@dataclass
class MergeNodeContext:
    """Context for one recursively merged object."""

    operation: MergeOperationContext
    parent: Optional["MergeNodeContext"]
    parent_model: Optional[Type[Any]]
    parent_inst: Any
    relation: Optional[str]
    delete: bool = False
    reparent_in_progress: bool = False
