from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Set

from .selection import RelationSelectionExtractor, RootSelectionExtractor
from .utils import normalize_relation_cfg


@dataclass
class QueryPlan:
    requested_relations: Dict[str, Dict[str, Any]]
    requested_scalar_root: Set[str]
    requested_custom_root: Set[str]
    requested_custom_obj_root: Set[str]
    requested_aggregates_root: Set[str]
    required_fk_parent_cols: Set[str]


class QueryAnalyzer:
    """Centralizes analysis of a GraphQL root field selection into a compact plan.

    Responsibilities:
      - Extract requested relations (with nested configs) and root field kinds
      - Normalize relation configs (AST values -> Python types)
      - Compute helper FK parent columns required for single relations
    """

    def __init__(self, registry):
        self.registry = registry

    def analyze(self, info: Any, root_field_name: str, btype_cls: Any) -> QueryPlan:
        # Extract selected relations from both Strawberry's selected_fields and AST
        rel_extractor = RelationSelectionExtractor(self.registry)
        requested_relations = rel_extractor.extract(info, root_field_name, btype_cls)
        # Normalize relation configs in place (where/default_where literals, order_multi, fields, filter_args...)
        for rel_cfg in list(requested_relations.values()):
            try:
                normalize_relation_cfg(rel_cfg)
            except Exception:
                # best-effort
                pass
        # Extract root-level selection kinds
        root_selected = RootSelectionExtractor(self.registry).extract(info, root_field_name, btype_cls)
        requested_scalar_root = set(root_selected.get('scalars', set()))
        requested_custom_root = set(root_selected.get('custom', set()))
        requested_custom_obj_root = set(root_selected.get('custom_object', set()))
        requested_aggregates_root = set(root_selected.get('aggregate', set()))
        # Compute helper FK columns on parent for single relations
        required_fk_parent_cols: set[str] = set()
        try:
            model_cls = getattr(btype_cls, 'model', None)
            if model_cls is None:
                # Lookup model via registry.types mapping
                for _, tb in (self.registry.types or {}).items():
                    if getattr(tb, '__name__', None) == getattr(btype_cls, '__name__', None):
                        model_cls = getattr(tb, 'model', None)
                        break
            if model_cls is not None:
                for rel_name, rel_cfg in list(requested_relations.items()):
                    if not rel_cfg.get('single'):
                        continue
                    try:
                        target_name = rel_cfg.get('target')
                        child_model_cls = (self.registry.types.get(target_name).model if target_name and self.registry.types.get(target_name) else None)
                    except Exception:
                        child_model_cls = None
                    # Respect explicit fk override for single relations
                    try:
                        parent_fk_col_name = rel_cfg.get('fk_column_name') or self.registry._find_parent_fk_column_name(model_cls, child_model_cls, rel_name)  # type: ignore[attr-defined]
                    except Exception:
                        parent_fk_col_name = self.registry._find_parent_fk_column_name(model_cls, child_model_cls, rel_name)  # type: ignore[attr-defined]
                    if parent_fk_col_name is not None:
                        required_fk_parent_cols.add(parent_fk_col_name)
        except Exception:
            required_fk_parent_cols = set()

        return QueryPlan(
            requested_relations=requested_relations,
            requested_scalar_root=requested_scalar_root,
            requested_custom_root=requested_custom_root,
            requested_custom_obj_root=requested_custom_obj_root,
            requested_aggregates_root=requested_aggregates_root,
            required_fk_parent_cols=required_fk_parent_cols,
        )
