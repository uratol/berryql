from __future__ import annotations

from typing import Any, Dict, List
from .enum_utils import get_model_enum_cls, coerce_mapping_to_enum
from datetime import datetime
from .naming import from_camel

# SQLAlchemy DateTime for value coercion
try:
    from sqlalchemy.sql.sqltypes import DateTime  # type: ignore
except Exception:  # pragma: no cover
    DateTime = object  # type: ignore


class Hydrator:
    """Hydration helpers to keep the registry root resolver lean.

    This class performs Python-side hydration from DB row mappings into Strawberry
    instances, including:
      - copying mapped root scalars and helper FK columns
      - attaching ORM model when present
      - reconstructing custom scalars and custom objects from labeled columns
      - hydrating pushed-down relation JSON (including nested relations)
      - populating aggregate caches
    """

    # Cache for datetime column detection to avoid repeated introspection
    # Key: (model_cls_id, column_key), Value: is_datetime_column
    _datetime_column_cache: dict[tuple[int, str], bool] = {}

    def __init__(self, registry):
        self.registry = registry

    # ----- basic helpers -----
    @staticmethod
    def attach_model(inst: Any, sa_row: Any) -> None:
        """Attach underlying ORM model to instance when present in first column."""
        try:
            row0 = sa_row[0]
            if hasattr(getattr(row0, '__class__', object), '__table__'):
                setattr(inst, '_model', row0)
                return
        except Exception:
            pass
        setattr(inst, '_model', None)

    def copy_mapping_fields(self, inst: Any, mapping: Dict[str, Any]) -> None:
        """Best-effort copy of mapping keys as attributes on the instance.

    Enum-aware: when a mapped field corresponds to a model column of SAEnum type
    and the value is a raw string (DB value or NAME), it is converted to the
    Python Enum instance so Strawberry serializes it as ENUM.
        """
        registry = self.registry
        try:
            if registry is None:
                registry = getattr(inst, '__berry_registry__', None)
            if registry is None:
                registry = getattr(getattr(inst, '__class__', object), '__berry_registry__', None)
        except Exception:
            registry = None
        try:
            type_name = getattr(getattr(inst, '__class__', None), '__name__', None)
            btype = registry.types.get(type_name) if (registry and type_name) else None
            model_cls = getattr(btype, 'model', None) if btype is not None else None
        except Exception:
            btype = None
            model_cls = None
        def _coerce_enum(model_cls_local, key: str, value: Any) -> Any:
            enum_cls = get_model_enum_cls(model_cls_local, key)
            return coerce_mapping_to_enum(enum_cls, value)
        # Iterate and set attributes
        try:
            items_iter = mapping.items()
        except Exception:
            try:
                items_iter = dict(mapping).items()
            except Exception:
                items_iter = []  # type: ignore
        for k, v in items_iter:
            try:
                v2 = _coerce_enum(model_cls, str(k), v)
                setattr(inst, str(k), v2)
            except Exception:
                try:
                    setattr(inst, str(k), v)
                except Exception:
                    pass

    def hydrate_base_scalars(
        self,
        inst: Any,
        mapping: Dict[str, Any],
        *,
        requested_scalar_root: set[str] | None,
        requested_relations: Dict[str, Any] | None,
        required_fk_parent_cols: set[str] | None,
    ) -> None:
        # Resolve model class for enum/date coercions
        btype = None
        model_cls = None
        try:
            type_name = getattr(getattr(inst, '__class__', None), '__name__', None)
            btype = self.registry.types.get(type_name) if type_name else None
            model_cls = getattr(btype, 'model', None) if btype is not None else None
        except Exception:
            btype = None
            model_cls = None
        # Helper to resolve underlying source column name from Berry field meta
        def _source_col_for_scalar(field_name: str) -> str | None:
            try:
                fdef = getattr(btype, '__berry_fields__', {}).get(field_name) if btype is not None else None
                if fdef and getattr(fdef, 'kind', None) == 'scalar':
                    meta = (getattr(fdef, 'meta', {}) or {})
                    src = meta.get('column')
                    return str(src) if src else None
            except Exception:
                return None
            return None
        # Helper: decamelize GraphQL field names to python/DB snake_case
        def _decamel(name: str) -> str:
            try:
                if not name or ('_' in name):
                    return name
                return from_camel(str(name))
            except Exception:
                return str(name)
        def _normalize_field_name(field_name: str) -> str:
            # Prefer declared berry field key; else try decamelized variant
            try:
                if btype is not None and field_name in getattr(btype, '__berry_fields__', {}):
                    return field_name
            except Exception:
                pass
            d = _decamel(field_name)
            try:
                if btype is not None and d in getattr(btype, '__berry_fields__', {}):
                    return d
            except Exception:
                pass
            return field_name
        def _coerce_enum_for_model(key: str, value: Any) -> Any:
            enum_cls = get_model_enum_cls(model_cls, key)
            return coerce_mapping_to_enum(enum_cls, value)
        # Only assign requested scalar root fields when specified
        if requested_scalar_root:
            for sf in requested_scalar_root:
                try:
                    sf_eff = _normalize_field_name(str(sf))
                    # Try mapping by effective snake_case key first
                    v = mapping.get(sf_eff, None)
                    # Fallback: honor field alias mapping (meta.column) when projection labeled under source name
                    if v is None:
                        src = _source_col_for_scalar(sf_eff)
                        if src and (src in mapping):
                            v = mapping.get(src)
                    # Additional fallback: original key (in case mapping labeled as camelCase)
                    if v is None and sf != sf_eff:
                        v = mapping.get(sf)
                    # If still None, try reading from attached ORM model when full entity was selected
                    if v is None:
                        try:
                            model_inst = getattr(inst, '_model', None)
                            if model_inst is not None and hasattr(model_inst, sf_eff):
                                v = getattr(model_inst, sf_eff)
                        except Exception:
                            pass
                    # Coerce datetime and enum when needed
                    v = self._coerce_datetime_scalar(model_cls, sf_eff, v)
                    v = _coerce_enum_for_model(sf_eff, v)
                    # Set attribute on the effective python field name
                    setattr(inst, sf_eff, v)
                except Exception:
                    pass
        # Ensure helper cols present (id for relation resolvers + FK helpers)
        try:
            needed: set[str] = set()
            if requested_relations:
                needed.add('id')
            for fk in (required_fk_parent_cols or set()):
                needed.add(fk)
            for name in needed:
                if getattr(inst, name, None) is None:
                    try:
                        v2 = mapping.get(name)
                        if v2 is None:
                            src = _source_col_for_scalar(name)
                            if src and (src in mapping):
                                v2 = mapping.get(src)
                        if v2 is None:
                            try:
                                model_inst2 = getattr(inst, '_model', None)
                                if model_inst2 is not None and hasattr(model_inst2, name):
                                    v2 = getattr(model_inst2, name)
                            except Exception:
                                pass
                        v2 = self._coerce_datetime_scalar(model_cls, name, v2)
                        v2 = _coerce_enum_for_model(name, v2)
                        setattr(inst, name, v2)
                    except Exception:
                        pass
        except Exception:
            pass

    def hydrate_custom_scalars(
        self,
        inst: Any,
        mapping: Dict[str, Any],
        *,
        custom_fields: List[tuple[str, Any]] | None,
    ) -> None:
        if not custom_fields:
            return
        for cf_name, _ in custom_fields:
            try:
                if cf_name in mapping:
                    setattr(inst, cf_name, mapping[cf_name])
            except Exception:
                pass

    def hydrate_custom_objects(
        self,
        inst: Any,
        mapping: Dict[str, Any],
        *,
        custom_object_fields: List[tuple[str, List[str], Any]] | None,
        btype_cls: Any,
    ) -> None:
        """Reconstruct custom object fields from mapping.

        custom_object_fields items are (field_name, column_labels, returns_spec).
        MSSQL path may provide multiple scalar labels; non-MSSQL typically has one JSON label.
        """
        if not custom_object_fields:
            return
        import json as _json
        for cf_name, col_labels, returns_spec in custom_object_fields:
            obj = None
            json_col = col_labels[0] if (col_labels and len(col_labels) == 1) else None
            raw_json = mapping[json_col] if (json_col and (json_col in mapping)) else None
            if raw_json is not None:
                parsed = None
                try:
                    parsed = _json.loads(raw_json) if isinstance(raw_json, (str, bytes)) else raw_json
                except Exception:
                    parsed = None
                data_dict = parsed if isinstance(parsed, dict) else None
                if data_dict is None and json_col and '__' in json_col:
                    # Single labeled scalar case: infer key from label suffix
                    try:
                        key = json_col.split('__', 1)[1]
                    except Exception:
                        key = json_col
                    data_dict = {key: parsed if parsed is not None else raw_json}
            else:
                data_dict = None
                try:
                    if col_labels and len(col_labels) > 1:
                        tmp: dict[str, Any] = {}
                        for lbl in col_labels:
                            if lbl in mapping:
                                try:
                                    key = lbl.split('__', 1)[1]
                                except Exception:
                                    key = lbl
                                tmp[key] = mapping[lbl]
                        data_dict = tmp
                except Exception:
                    data_dict = None
            # Filter and coerce types when returns_spec provided as dict
            if data_dict and isinstance(returns_spec, dict):
                data_dict = {k: data_dict.get(k) for k in returns_spec.keys() if k in data_dict}
                try:
                    for k2, t2 in returns_spec.items():
                        if k2 in data_dict:
                            v = data_dict[k2]
                            if t2 is datetime and isinstance(v, str):
                                try:
                                    data_dict[k2] = datetime.fromisoformat(v)
                                except Exception:
                                    try:
                                        data_dict[k2] = datetime.fromisoformat(v.replace(' ', 'T'))
                                    except Exception:
                                        pass
                except Exception:
                    pass
            # Materialize into nested Strawberry type when available
            if data_dict:
                nested_type_name = f"{getattr(btype_cls, '__name__', 'Type')}_{cf_name}_Type"
                nested_type = self.registry._st_types.get(nested_type_name)
                try:
                    if nested_type is not None:
                        try:
                            obj = nested_type(**data_dict)
                        except Exception:
                            obj = nested_type()
                            for k, v in data_dict.items():
                                try:
                                    setattr(obj, k, v)
                                except Exception:
                                    pass
                    else:
                        obj = data_dict
                except Exception:
                    obj = data_dict
            setattr(inst, f"_{cf_name}_data", obj)
            setattr(inst, f"_{cf_name}_prefetched", obj)
            try:
                setattr(inst, cf_name, obj)
            except Exception:
                pass

    # ----- relations hydration -----
    def _coerce_datetime_scalar(self, target_model_cls: Any, key: str, value: Any) -> Any:
        """Coerce string datetime values to Python datetime objects.
        
        Uses caching to avoid repeated column introspection.
        """
        if target_model_cls is None or not isinstance(value, str):
            return value
        
        # Check cache first
        cache_key = (id(target_model_cls), key)
        if cache_key in self._datetime_column_cache:
            is_datetime_col = self._datetime_column_cache[cache_key]
        else:
            # Cache miss - perform introspection
            is_datetime_col = False
            try:
                col = target_model_cls.__table__.c.get(key)
                if col is not None and isinstance(getattr(col, 'type', None), DateTime):
                    is_datetime_col = True
            except Exception:
                pass
            self._datetime_column_cache[cache_key] = is_datetime_col
        
        if is_datetime_col:
            try:
                return datetime.fromisoformat(value)
            except Exception:
                try:
                    return datetime.fromisoformat(value.replace(' ', 'T'))
                except Exception:
                    return value
        return value

    def hydrate_relations(
        self,
        inst: Any,
        mapping: Dict[str, Any],
        *,
        requested_relations: Dict[str, Any],
        rel_push_status: Dict[str, Dict[str, Any]] | None = None,
    ) -> None:
        import json as _json
        # Helper to coerce enum scalars for target model class
        def _coerce_enum_scalar(target_model_cls: Any, key: str, value: Any) -> Any:
            enum_cls = get_model_enum_cls(target_model_cls, key)
            return coerce_mapping_to_enum(enum_cls, value)
        for rel_name, rel_meta in (requested_relations or {}).items():
            key = f"_pushrel_{rel_name}"
            if key in mapping:
                raw_json = mapping[key]
                is_single = bool(rel_meta.get('single'))
                if raw_json is None:
                    parsed_value = None if is_single else []
                else:
                    try:
                        parsed_value = _json.loads(raw_json) if isinstance(raw_json, (str, bytes)) else raw_json
                    except Exception:
                        parsed_value = (None if is_single else [])
                target_name = rel_meta.get('target')
                target_b = self.registry.types.get(target_name) if target_name else None
                target_st = self.registry._st_types.get(target_name) if target_name else None
                built_value = None if is_single else []
                if target_b and target_b.model and target_st and parsed_value is not None:
                    if is_single:
                        if isinstance(parsed_value, dict):
                            child_inst = target_st()
                            for sf, sdef in target_b.__berry_fields__.items():
                                if sdef.kind == 'scalar':
                                    val = self._coerce_datetime_scalar(target_b.model, sf, parsed_value.get(sf))
                                    val = _coerce_enum_scalar(target_b.model, sf, val)
                                    setattr(child_inst, sf, val)
                            setattr(child_inst, '_model', None)
                            built_value = child_inst
                        else:
                            built_value = None
                    else:
                        tmp_list = []
                        if isinstance(parsed_value, list):
                            for item in parsed_value:
                                if not isinstance(item, dict):
                                    continue
                                child_inst = target_st()
                                for sf, sdef in target_b.__berry_fields__.items():
                                    if sdef.kind == 'scalar':
                                        val = self._coerce_datetime_scalar(target_b.model, sf, item.get(sf))
                                        val = _coerce_enum_scalar(target_b.model, sf, val)
                                        setattr(child_inst, sf, val)
                                setattr(child_inst, '_model', None)
                                # hydrate nested relations recursively if present in item
                                try:
                                    parent_meta = (requested_relations.get(rel_name) or {})
                                    nested_meta_map = (parent_meta.get('nested') or {})
                                    self._hydrate_deeper_nested(child_inst, target_b, item, nested_meta_map)
                                except Exception:
                                    pass
                                tmp_list.append(child_inst)
                        built_value = tmp_list
                else:
                    built_value = parsed_value
                # cache prefetched value unless single None
                if not (bool(is_single) and built_value is None):
                    setattr(inst, f"_{rel_name}_prefetched", built_value)
                # record pushdown meta
                try:
                    meta_map = getattr(inst, '_pushdown_meta', None)
                    if meta_map is None:
                        meta_map = {}
                        setattr(inst, '_pushdown_meta', meta_map)
                    meta_map[rel_name] = {
                        'limit': rel_meta.get('limit'),
                        'offset': rel_meta.get('offset'),
                        'from_pushdown': True,
                        'skip_reason': None,
                    }
                except Exception:
                    pass
            else:
                # Relation requested but not pushed down: record skip reason
                try:
                    meta_map = getattr(inst, '_pushdown_meta', None)
                    if meta_map is None:
                        meta_map = {}
                        setattr(inst, '_pushdown_meta', meta_map)
                    reason_txt = None
                    try:
                        reason_txt = (rel_push_status.get(rel_name) or {}).get('reason') if rel_push_status else None
                    except Exception:
                        reason_txt = None
                    meta_map[rel_name] = {
                        'limit': rel_meta.get('limit'),
                        'offset': rel_meta.get('offset'),
                        'from_pushdown': False,
                        'skip_reason': reason_txt,
                    }
                except Exception:
                    pass

    def _hydrate_deeper_nested(
        self,
        parent_inst: Any,
        parent_b: Any,
        item_dict: Dict[str, Any],
        nested_meta_src_map: Dict[str, Any] | None,
    ) -> None:
        """Hydrate nested relations under a child instance using nested JSON attached under keys."""
        import json as _json
        for nname_i, ndef_i in (getattr(parent_b, '__berry_fields__', {}) or {}).items():
            if getattr(ndef_i, 'kind', None) != 'relation':
                continue
            raw_nested_i = item_dict.get(nname_i, None)
            if raw_nested_i is None:
                continue
            try:
                parsed_nested_i = _json.loads(raw_nested_i) if isinstance(raw_nested_i, (str, bytes)) else raw_nested_i
            except Exception:
                parsed_nested_i = None
            n_target_b = self.registry.types.get(ndef_i.meta.get('target')) if ndef_i.meta.get('target') else None
            n_st = self.registry._st_types.get(ndef_i.meta.get('target')) if ndef_i.meta.get('target') else None
            if not n_target_b or not n_target_b.model or not n_st:
                continue
            if ndef_i.meta.get('single'):
                if isinstance(parsed_nested_i, dict):
                    ni = n_st()
                    for nsf, nsdef in n_target_b.__berry_fields__.items():
                        if nsdef.kind == 'scalar':
                            val = self._coerce_datetime_scalar(n_target_b.model, nsf, parsed_nested_i.get(nsf))
                            # Enum coercion for nested single relation
                            enum_cls = get_model_enum_cls(n_target_b.model, nsf)
                            val = coerce_mapping_to_enum(enum_cls, val)
                            setattr(ni, nsf, val)
                    setattr(ni, '_model', None)
                    setattr(parent_inst, nname_i, ni)
                    setattr(parent_inst, f"_{nname_i}_prefetched", ni)
                else:
                    setattr(parent_inst, nname_i, None)
                    setattr(parent_inst, f"_{nname_i}_prefetched", None)
            else:
                nlist_i: list[Any] = []
                if isinstance(parsed_nested_i, list):
                    for nv_i in parsed_nested_i:
                        if not isinstance(nv_i, dict):
                            continue
                        ni = n_st()
                        for nsf, nsdef in n_target_b.__berry_fields__.items():
                            if nsdef.kind == 'scalar':
                                val = self._coerce_datetime_scalar(n_target_b.model, nsf, nv_i.get(nsf))
                                # Enum coercion for nested list relation
                                enum_cls = get_model_enum_cls(n_target_b.model, nsf)
                                val = coerce_mapping_to_enum(enum_cls, val)
                                setattr(ni, nsf, val)
                        setattr(ni, '_model', None)
                        # recurse deeper if any
                        deeper_meta = (nested_meta_src_map.get(nname_i) or {}).get('nested') if isinstance(nested_meta_src_map, dict) else None
                        try:
                            self._hydrate_deeper_nested(ni, n_target_b, nv_i, deeper_meta or {})
                        except Exception:
                            pass
                        nlist_i.append(ni)
                setattr(parent_inst, nname_i, nlist_i)
                setattr(parent_inst, f"_{nname_i}_prefetched", nlist_i)
            # record nested meta for current level
            try:
                meta_map_nested = getattr(parent_inst, '_pushdown_meta', None)
                if meta_map_nested is None:
                    meta_map_nested = {}
                    setattr(parent_inst, '_pushdown_meta', meta_map_nested)
                src_meta = (nested_meta_src_map.get(nname_i) or {}) if isinstance(nested_meta_src_map, dict) else {}
                meta_map_nested[nname_i] = {
                    'limit': src_meta.get('limit'),
                    'offset': src_meta.get('offset'),
                    'from_pushdown': bool(src_meta.get('from_pushdown', True)),
                    'skip_reason': src_meta.get('skip_reason'),
                }
            except Exception:
                pass

    @staticmethod
    def populate_aggregate_cache(
        inst: Any,
        mapping: Dict[str, Any],
        *,
        count_aggregates: List[tuple[str, Any]] | None,
    ) -> None:
        if not count_aggregates:
            return
        cache = getattr(inst, '_agg_cache', None)
        if cache is None:
            cache = {}
            setattr(inst, '_agg_cache', cache)
        for agg_name, agg_def in count_aggregates:
            try:
                val = mapping.get(agg_name)
            except Exception:
                val = None
            cache_key = (getattr(agg_def, 'meta', {}) or {}).get('cache_key') or ((getattr(agg_def, 'meta', {}) or {}).get('source') + ':count')
            cache[cache_key] = val or 0
