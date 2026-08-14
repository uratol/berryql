from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Optional, Set

from .selection import RelationSelectionExtractor, RootSelectionExtractor
from .utils import normalize_relation_cfg


class ValueSource(str, Enum):
    CALLER = "caller"
    RELATION_DEFAULT = "relation_default"
    TYPE_DEFAULT = "type_default"
    TRUSTED_SCOPE = "trusted_scope"
    ABSENT = "absent"


class PushdownReason(str, Enum):
    NO_TARGET = "no_target"
    TARGET_MODEL_MISSING = "target_model_missing"
    NO_FOREIGN_KEY = "no_foreign_key"
    COMPUTED_SOURCE = "computed_source"
    ADAPTER_UNSUPPORTED = "adapter_unsupported"
    BUILDER_UNSUPPORTED = "builder_unsupported"


@dataclass(frozen=True)
class PushdownDecision:
    pushed: bool
    reason: Optional[PushdownReason] = None
    detail: Optional[str] = None

    @classmethod
    def success(cls) -> "PushdownDecision":
        return cls(True)

    @classmethod
    def fallback(cls, reason: PushdownReason, detail: Optional[str] = None) -> "PushdownDecision":
        return cls(False, reason, detail)


@dataclass(frozen=True)
class HydrationRelationPlan:
    limit: Optional[int]
    offset: Optional[int]
    decision: PushdownDecision
    fk_column_name: Optional[str] = None
    fields: tuple[str, ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        values = {
            "limit": self.limit,
            "offset": self.offset,
            "decision": self.decision,
            "fk_column_name": self.fk_column_name,
            "fields": self.fields,
        }
        return values.get(key, default)


@dataclass(frozen=True)
class RelationPlan(Mapping[str, Any]):
    fields: tuple[str, ...] = ()
    limit: Optional[int] = None
    offset: Optional[int] = None
    after: Optional[str] = None
    order_by: Any = None
    order_dir: Any = None
    order_multi: tuple[Any, ...] = ()
    where: Any = None
    default_where: Any = None
    type_default_where: Any = None
    single: bool = False
    target: Optional[str] = None
    nested: Mapping[str, "RelationPlan"] = field(default_factory=dict)
    fk_column_name: Optional[str] = None
    filter_args: Mapping[str, Any] = field(default_factory=dict)
    arg_specs: Any = None
    order_by_source: ValueSource = ValueSource.ABSENT
    order_dir_source: ValueSource = ValueSource.ABSENT
    order_multi_source: ValueSource = ValueSource.ABSENT
    where_source: ValueSource = ValueSource.ABSENT

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "order_multi", tuple(self.order_multi))
        object.__setattr__(self, "nested", MappingProxyType(dict(self.nested)))
        object.__setattr__(self, "filter_args", MappingProxyType(dict(self.filter_args)))

    def _mapping(self) -> Mapping[str, Any]:
        # Compatibility view for existing SQL builders; service flags are not
        # included. Provenance and pushdown decisions remain typed.
        return {
            "fields": list(self.fields),
            "limit": self.limit,
            "offset": self.offset,
            "after": self.after,
            "order_by": self.order_by,
            "order_dir": self.order_dir,
            "order_multi": list(self.order_multi),
            "where": self.where,
            "default_where": self.default_where,
            "type_default_where": self.type_default_where,
            "single": self.single,
            "target": self.target,
            "nested": self.nested,
            "fk_column_name": self.fk_column_name,
            "filter_args": dict(self.filter_args),
            "arg_specs": self.arg_specs,
            "order_by_source": self.order_by_source,
            "order_dir_source": self.order_dir_source,
            "order_multi_source": self.order_multi_source,
            "where_source": self.where_source,
        }

    def __getitem__(self, key: str) -> Any:
        return self._mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping())

    def __len__(self) -> int:
        return len(self._mapping())

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping().get(key, default)


@dataclass(frozen=True)
class SelectionPlan:
    relations: Mapping[str, RelationPlan]
    scalar_fields: frozenset[str]
    custom_fields: frozenset[str]
    custom_object_fields: frozenset[str]
    aggregate_fields: frozenset[str]
    other_fields: frozenset[str]
    required_fk_parent_cols: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "relations", MappingProxyType(dict(self.relations)))


@dataclass(frozen=True)
class QueryPlan:
    root_field_name: str
    berry_type: Any
    selection: SelectionPlan

    @property
    def requested_relations(self) -> Mapping[str, RelationPlan]:
        return self.selection.relations

    @property
    def requested_scalar_root(self) -> Set[str]:
        return set(self.selection.scalar_fields)

    @property
    def requested_custom_root(self) -> Set[str]:
        return set(self.selection.custom_fields)

    @property
    def requested_custom_obj_root(self) -> Set[str]:
        return set(self.selection.custom_object_fields)

    @property
    def requested_aggregates_root(self) -> Set[str]:
        return set(self.selection.aggregate_fields)

    @property
    def requested_other_root(self) -> Set[str]:
        return set(self.selection.other_fields)

    @property
    def required_fk_parent_cols(self) -> Set[str]:
        return set(self.selection.required_fk_parent_cols)


@dataclass(frozen=True)
class AuthorizedQueryPlan:
    original: QueryPlan
    selection: SelectionPlan
    permissions: Any
    pushdown: Mapping[str, PushdownDecision] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pushdown", MappingProxyType(dict(self.pushdown)))


class QueryAnalyzer:
    """Resolve GraphQL selections and variables once into immutable typed IR."""

    def __init__(self, registry: Any):
        self.registry = registry

    @staticmethod
    def _source(cfg: Mapping[str, Any], name: str, default_present: bool = False) -> ValueSource:
        provenance = cfg.get("_provenance")
        if provenance is not None and getattr(provenance, "is_caller", lambda _name: False)(name):
            return ValueSource.CALLER
        if default_present:
            return ValueSource.RELATION_DEFAULT
        return ValueSource.ABSENT

    def _relation_plan(self, raw: Dict[str, Any]) -> RelationPlan:
        cfg = dict(raw or {})
        normalize_relation_cfg(cfg)
        nested = {
            str(name): self._relation_plan(dict(value or {}))
            for name, value in (cfg.get("nested") or {}).items()
        }
        return RelationPlan(
            fields=tuple(cfg.get("fields") or ()),
            limit=cfg.get("limit"),
            offset=cfg.get("offset"),
            after=cfg.get("after"),
            order_by=cfg.get("order_by"),
            order_dir=cfg.get("order_dir"),
            order_multi=tuple(cfg.get("order_multi") or ()),
            where=cfg.get("where"),
            default_where=cfg.get("default_where"),
            type_default_where=cfg.get("type_default_where"),
            single=bool(cfg.get("single")),
            target=cfg.get("target"),
            nested=nested,
            fk_column_name=cfg.get("fk_column_name"),
            filter_args=cfg.get("filter_args") or {},
            arg_specs=cfg.get("arg_specs"),
            order_by_source=self._source(cfg, "order_by", cfg.get("order_by") is not None),
            order_dir_source=self._source(cfg, "order_dir", cfg.get("order_dir") is not None),
            order_multi_source=self._source(cfg, "order_multi", bool(cfg.get("order_multi"))),
            where_source=ValueSource.CALLER if cfg.get("where") is not None else ValueSource.ABSENT,
        )

    def analyze(self, info: Any, root_field_name: str, btype_cls: Any) -> QueryPlan:
        raw_relations = RelationSelectionExtractor(self.registry).extract(info, root_field_name, btype_cls)
        requested_relations = {
            str(name): self._relation_plan(cfg) for name, cfg in raw_relations.items()
        }
        root_selected = RootSelectionExtractor(self.registry).extract(info, root_field_name, btype_cls)
        required_fk_parent_cols: set[str] = set()
        model_cls = getattr(btype_cls, "model", None)
        if model_cls is not None:
            for rel_name, rel_cfg in requested_relations.items():
                if not rel_cfg.single:
                    continue
                target_btype = self.registry.types.get(rel_cfg.target) if rel_cfg.target else None
                child_model = getattr(target_btype, "model", None)
                parent_fk = rel_cfg.fk_column_name or self.registry._find_parent_fk_column_name(
                    model_cls, child_model, rel_name
                )
                if parent_fk:
                    required_fk_parent_cols.add(parent_fk)
        selection = SelectionPlan(
            relations=requested_relations,
            scalar_fields=frozenset(root_selected.get("scalars", set())),
            custom_fields=frozenset(root_selected.get("custom", set())),
            custom_object_fields=frozenset(root_selected.get("custom_object", set())),
            aggregate_fields=frozenset(root_selected.get("aggregate", set())),
            other_fields=frozenset(root_selected.get("other", set())),
            required_fk_parent_cols=frozenset(required_fk_parent_cols),
        )
        return QueryPlan(root_field_name, btype_cls, selection)


__all__ = [
    "AuthorizedQueryPlan",
    "HydrationRelationPlan",
    "PushdownDecision",
    "PushdownReason",
    "QueryAnalyzer",
    "QueryPlan",
    "RelationPlan",
    "SelectionPlan",
    "ValueSource",
]
