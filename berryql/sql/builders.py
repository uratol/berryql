from __future__ import annotations
from typing import Any, Dict, List, Optional
from sqlalchemy import select, text as _text

# Centralized SQL builders. Keep registry thin and DRY.


class RelationSQLBuilders:
    def __init__(self, registry):
        self.registry = registry

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _resolve_graphql_value(info, raw):
        v = raw
        try:
            if not isinstance(v, (dict, str)) and hasattr(v, 'name'):
                vname = getattr(getattr(v, 'name', None), 'value', None) or getattr(v, 'name', None)
                var_vals = getattr(info, 'variable_values', None)
                if var_vals is None:
                    raw_info = getattr(info, '_raw_info', None)
                    var_vals = getattr(raw_info, 'variable_values', None) if raw_info is not None else None
                if isinstance(var_vals, dict) and vname in var_vals:
                    v = var_vals[vname]
            if not isinstance(v, (dict, str)) and hasattr(v, 'value'):
                v = getattr(v, 'value')
        except Exception:
            pass
        return v

    # --- single relation (object) -------------------------------------------
    def build_single_relation_object(
        self,
        *,
        adapter,
        parent_model_cls,
        child_model_cls,
        rel_name: str,
        projected_columns: list[str],
        parent_fk_col_name: Optional[str],
        json_object_fn,
        json_array_coalesce_fn,  # kept for parity; not used here
        to_where_dict,
        expr_from_where_dict,
        info,
        rel_where: Any,
        rel_default_where: Any,
        filter_args: Dict[str, Any] | None,
        arg_specs: Dict[str, Any] | None,
    ):
        cols = projected_columns or ['id']
        mssql_mode = getattr(adapter, 'name', '') == 'mssql'
        # Determine parent FK name for correlation (parent has FK to child)
        pfk = parent_fk_col_name or f"{rel_name}_id"
        if mssql_mode:
            # Compose WHERE join + relation filters, then let adapter emit FOR JSON PATH
            child_table = child_model_cls.__tablename__
            parent_table = parent_model_cls.__tablename__
            where_parts: list[str] = [
                f"[{child_table}].[{pfk}] = [{parent_table}].[id]"
            ]
            r_where = self._resolve_graphql_value(info, rel_where)
            if r_where is not None and isinstance(r_where, (dict, str)):
                # strict=True must raise for malformed and unknowns
                wdict = to_where_dict(r_where, strict=True)
                if wdict:
                    # validate using SQLA path to raise on unknowns/operators
                    _ = expr_from_where_dict(child_model_cls, wdict, strict=True)
                    where_parts.extend(adapter.where_from_dict(child_model_cls, wdict))
            d_where = self._resolve_graphql_value(info, rel_default_where)
            if d_where is not None and isinstance(d_where, (dict, str)):
                try:
                    dwdict = to_where_dict(d_where, strict=False)
                except Exception:
                    dwdict = None
                if dwdict:
                    where_parts.extend(adapter.where_from_dict(child_model_cls, dwdict))
            where_sql = ' AND '.join(where_parts)
            # Use adapter helper for a single object
            return adapter.build_single_relation_json(
                child_table=child_table,
                projected_columns=cols,
                join_condition=where_sql,
            )
        # Non-MSSQL: correlated select limited to 1, return JSON object or null
        child_id_col = getattr(child_model_cls, 'id')
        parent_fk_col = getattr(parent_model_cls, pfk)
        inner_cols = [getattr(child_model_cls, c) for c in cols]
        inner_sel = (
            select(*inner_cols)
            .select_from(child_model_cls)
            .where(child_id_col == parent_fk_col)
            .correlate(parent_model_cls)
            .limit(1)
        )
        r_where = self._resolve_graphql_value(info, rel_where)
        if r_where is not None and isinstance(r_where, (dict, str)):
            wdict = to_where_dict(r_where, strict=True)
            if wdict:
                expr = expr_from_where_dict(child_model_cls, wdict, strict=True)
                if expr is not None:
                    inner_sel = inner_sel.where(expr)
        d_where = self._resolve_graphql_value(info, rel_default_where)
        if d_where is not None and isinstance(d_where, (dict, str)):
            try:
                dwdict = to_where_dict(d_where, strict=False)
            except Exception:
                dwdict = None
            if dwdict:
                try:
                    expr = expr_from_where_dict(child_model_cls, dwdict, strict=False)
                except Exception:
                    expr = None
                if expr is not None:
                    inner_sel = inner_sel.where(expr)
        limited = inner_sel.subquery()
        args: List[Any] = []
        for c in cols:
            args.extend([_text(f"'{c}'"), getattr(limited.c, c)])
        obj_expr = json_object_fn(*args)
        query = select(obj_expr).select_from(limited).correlate(parent_model_cls)
        try:
            return query.scalar_subquery()
        except Exception:
            return query

    # --- list relation (JSON array) -----------------------------------------
    def build_list_relation_json_adapter(
        self,
        *,
        adapter,
        parent_model_cls,
        child_model_cls,
        requested_scalar: list[str],
        fk_child_to_parent_col,
        rel_cfg: Dict[str, Any],
        json_object_fn,
        json_array_agg_fn,
        json_array_coalesce_fn,
        to_where_dict,
        expr_from_where_dict,
        info,
    ):
        mssql_mode = getattr(adapter, 'name', '') == 'mssql'
        child_table = child_model_cls.__tablename__
        # Add helper FK columns for nested single backrefs (safe no-op if absent)
        requested_scalar_local: list[str] = list(requested_scalar or ['id'])
        try:
            nested = rel_cfg.get('nested') or {}
            target_name = rel_cfg.get('target')
            target_b = self.registry.types.get(target_name) if target_name else None
            if target_b:
                for rname, fdef in (target_b.__berry_fields__ or {}).items():
                    if getattr(fdef, 'kind', None) == 'relation' and (fdef.meta.get('single') or fdef.meta.get('mode') == 'single') and rname in nested:
                        fk_col_name = f"{rname}_id"
                        try:
                            if fk_col_name in child_model_cls.__table__.columns and fk_col_name not in requested_scalar_local:
                                requested_scalar_local.append(fk_col_name)
                        except Exception:
                            pass
        except Exception:
            pass
        if mssql_mode:
            parent_table = parent_model_cls.__tablename__
            # If there are nested relations selected under this list relation, use the
            # MSSQL full nested JSON builder so everything is pushed down in one SELECT.
            nested_cfg_map: Dict[str, Any] = rel_cfg.get('nested') or {}
            if nested_cfg_map:
                nested_specs: list[dict] = []
                def _mk_nested_specs(parent_model_cls_local, parent_target_name: str | None, nested_map: Dict[str, Any]) -> list[dict]:
                    specs: list[dict] = []
                    for nname_i, ncfg_i in (nested_map or {}).items():
                        try:
                            n_target_i = ncfg_i.get('target')
                            n_b_i = self.registry.types.get(n_target_i) if n_target_i else None
                            n_model_i = n_b_i.model if n_b_i and getattr(n_b_i, 'model', None) is not None else None
                        except Exception:
                            n_model_i = None
                        if not n_model_i:
                            continue
                        # Find FK from n_model_i (grandchild) to parent child_model (parent_model_cls_local)
                        g_fk_col_name_i: str | None = None
                        try:
                            for col in n_model_i.__table__.columns:
                                for fk in col.foreign_keys:
                                    if fk.column.table.name == parent_model_cls_local.__table__.name:
                                        g_fk_col_name_i = col.name
                                        break
                                if g_fk_col_name_i is not None:
                                    break
                        except Exception:
                            g_fk_col_name_i = None
                        if not g_fk_col_name_i:
                            continue
                        # Determine scalar fields
                        try:
                            scalars_on_n_i = [sf for sf, sd in (self.registry.types.get(n_target_i).__berry_fields__.items() if n_target_i and self.registry.types.get(n_target_i) else []) if sd.kind == 'scalar']
                        except Exception:
                            scalars_on_n_i = []
                        n_fields_i: list[str] = []
                        try:
                            for sf in (ncfg_i.get('fields') or []):
                                if not scalars_on_n_i or sf in scalars_on_n_i:
                                    n_fields_i.append(sf)
                        except Exception:
                            n_fields_i = []
                        # order_dir default ASC if order_by explicit without dir
                        n_effective_dir_i = ncfg_i.get('order_dir')
                        try:
                            if ncfg_i.get('_has_explicit_order_by') and not ncfg_i.get('_has_explicit_order_dir'):
                                n_effective_dir_i = 'asc'
                        except Exception:
                            pass
                        # Recurse for deeper nests under this node
                        child_specs_i: list[dict] = []
                        try:
                            if ncfg_i.get('nested'):
                                child_specs_i = _mk_nested_specs(n_model_i, n_target_i, ncfg_i.get('nested') or {})
                        except Exception:
                            child_specs_i = []
                        specs.append({
                            'alias': nname_i,
                            'model': n_model_i,
                            'fk_col_name': g_fk_col_name_i,
                            'fields': n_fields_i or None,
                            'where': self._resolve_graphql_value(info, ncfg_i.get('where')),
                            'default_where': ncfg_i.get('default_where'),
                            'order_by': ncfg_i.get('order_by'),
                            'order_dir': n_effective_dir_i,
                            'order_multi': self.registry._normalize_order_multi_values(ncfg_i.get('order_multi') or []),
                            'limit': ncfg_i.get('limit'),
                            'offset': ncfg_i.get('offset'),
                            'nested': child_specs_i or None,
                        })
                    return specs
                # Build nested specs from the immediate nested map
                for nname, ncfg in nested_cfg_map.items():
                    try:
                        n_target = ncfg.get('target')
                        n_b = self.registry.types.get(n_target) if n_target else None
                        n_model = n_b.model if n_b and getattr(n_b, 'model', None) is not None else None
                    except Exception:
                        n_model = None
                    if not n_model:
                        continue
                    # Find FK from grandchild (n_model) to child (child_model_cls)
                    g_fk_col_name: str | None = None
                    try:
                        for col in n_model.__table__.columns:
                            for fk in col.foreign_keys:
                                if fk.column.table.name == child_model_cls.__table__.name:
                                    g_fk_col_name = col.name
                                    break
                            if g_fk_col_name is not None:
                                break
                    except Exception:
                        g_fk_col_name = None
                    if not g_fk_col_name:
                        # can't correlate; skip this nested
                        continue
                    # Determine fields for nested object: only scalar fields
                    try:
                        scalars_on_n = [sf for sf, sd in (self.registry.types.get(n_target).__berry_fields__.items() if n_target and self.registry.types.get(n_target) else []) if sd.kind == 'scalar']
                    except Exception:
                        scalars_on_n = []
                    n_fields = []
                    try:
                        for sf in (ncfg.get('fields') or []):
                            if not scalars_on_n or sf in scalars_on_n:
                                n_fields.append(sf)
                    except Exception:
                        n_fields = []
                    # Determine effective order_dir for nested: ASC when order_by explicit and no explicit dir
                    n_effective_dir = ncfg.get('order_dir')
                    try:
                        if ncfg.get('_has_explicit_order_by') and not ncfg.get('_has_explicit_order_dir'):
                            n_effective_dir = 'asc'
                    except Exception:
                        pass
                    # Build deeper nested children under this nested node
                    nested_children_specs = _mk_nested_specs(n_model, n_target, ncfg.get('nested') or {}) if (ncfg.get('nested') or {}) else []
                    nested_specs.append({
                        'alias': nname,
                        'model': n_model,
                        'fk_col_name': g_fk_col_name,
                        'fields': n_fields or None,
                        'where': self._resolve_graphql_value(info, ncfg.get('where')),
                        'default_where': ncfg.get('default_where'),
                        'order_by': ncfg.get('order_by'),
                        'order_dir': n_effective_dir,
                        'order_multi': self.registry._normalize_order_multi_values(ncfg.get('order_multi') or []),
                        'limit': ncfg.get('limit'),
                        'offset': ncfg.get('offset'),
                        'nested': nested_children_specs or None,
                    })
                # Build full JSON for the list relation, including nested arrays
                # Compute effective top-level order_dir: ASC when order_by explicit without explicit dir
                effective_order_dir = rel_cfg.get('order_dir')
                try:
                    if rel_cfg.get('_has_explicit_order_by') and not rel_cfg.get('_has_explicit_order_dir'):
                        effective_order_dir = 'asc'
                except Exception:
                    pass
                return adapter.build_relation_list_json_full(
                    parent_table=parent_table,
                    child_model=child_model_cls,
                    fk_col_name=fk_child_to_parent_col.name,
                    projected_columns=requested_scalar_local or ['id'],
                    rel_where=self._resolve_graphql_value(info, rel_cfg.get('where')),
                    rel_default_where=rel_cfg.get('default_where'),
                    limit=rel_cfg.get('limit'),
                    offset=rel_cfg.get('offset'),
                    order_by=rel_cfg.get('order_by'),
                    order_dir=effective_order_dir,
                    order_multi=self.registry._normalize_order_multi_values(rel_cfg.get('order_multi') or []),
                    nested=nested_specs,
                )
            # No nested relations: use simple JSON list builder with optional filters/order
            # correlate and compose relation-level where
            where_parts_rel: list[str] = [f"[{child_table}].[{fk_child_to_parent_col.name}] = [{parent_table}].[id]"]
            r_where = self._resolve_graphql_value(info, rel_cfg.get('where'))
            if r_where is not None and isinstance(r_where, (dict, str)):
                wdict_rel = to_where_dict(r_where, strict=True)
                if wdict_rel:
                    # validate to raise on unknowns/operators
                    _ = expr_from_where_dict(child_model_cls, wdict_rel, strict=True)
                    where_parts_rel.extend(adapter.where_from_dict(child_model_cls, wdict_rel))
            d_where = self._resolve_graphql_value(info, rel_cfg.get('default_where'))
            if d_where is not None and isinstance(d_where, (dict, str)):
                try:
                    dwdict_rel = to_where_dict(d_where, strict=False)
                except Exception:
                    dwdict_rel = None
                if dwdict_rel:
                    where_parts_rel.extend(adapter.where_from_dict(child_model_cls, dwdict_rel))
            # Callable default_where: emit a correlated EXISTS against any grandchild that FK's to child
            try:
                d_where_raw = rel_cfg.get('default_where')
                if callable(d_where_raw):
                    # Find a model with a FK -> child_model_cls
                    gc_model = None
                    gc_fk_col = None
                    for tname, tb in (self.registry.types or {}).items():
                        try:
                            m = getattr(tb, 'model', None)
                            if not m or not hasattr(m, '__table__'):
                                continue
                            for col in m.__table__.columns:
                                for fk in col.foreign_keys:
                                    if fk.column.table.name == child_model_cls.__table__.name:
                                        gc_model = m
                                        gc_fk_col = col.name
                                        break
                                if gc_model is not None:
                                    break
                            if gc_model is not None:
                                break
                        except Exception:
                            continue
                    if gc_model is not None and gc_fk_col is not None:
                        gc_table = gc_model.__tablename__
                        where_parts_rel.append(
                            f"EXISTS (SELECT 1 FROM {gc_table} WHERE [{gc_table}].[{gc_fk_col}] = [{child_table}].[id])"
                        )
            except Exception:
                pass
            # filter args best-effort (MSSQL string path)
            try:
                fa = rel_cfg.get('filter_args') or {}
                if fa:
                    expanded: Dict[str, Any] = {}
                    for key, raw in (rel_cfg.get('arg_specs') or {}).items():
                        try:
                            from ..core.filters import normalize_filter_spec as _norm
                            spec = _norm(raw)
                        except Exception:
                            continue
                        if getattr(spec, 'ops', None) and not getattr(spec, 'op', None):
                            for op_name in spec.ops:
                                base = spec.alias or key
                                an = base if base.endswith(f"_{op_name}") else f"{base}_{op_name}"
                                expanded[an] = spec.clone_with(op=op_name, ops=None)
                        else:
                            expanded[spec.alias or key] = spec
                    from ..core.filters import OPERATOR_REGISTRY
                    def _add(col_name: str, op_name: str, value: Any):
                        if op_name in ('like', 'ilike') and isinstance(value, str) and '%' not in value and '_' not in value:
                            value = f"%{value}%"
                        where_parts_rel.extend(adapter.where_from_dict(child_model_cls, {col_name: {op_name: value}}))
                    for arg_name, val in fa.items():
                        f_spec = expanded.get(arg_name)
                        if f_spec and f_spec.transform:
                            try:
                                val = f_spec.transform(val)
                            except Exception:
                                pass
                        if f_spec and f_spec.column:
                            _add(f_spec.column, f_spec.op or 'eq', val)
                            continue
                        try:
                            an = str(arg_name)
                            if '_' in an:
                                base, suffix = an.rsplit('_', 1)
                                if suffix.lower() in OPERATOR_REGISTRY and base in child_model_cls.__table__.columns:
                                    _add(base, suffix.lower(), val)
                        except Exception:
                            pass
            except Exception:
                pass
            where_clause = ' AND '.join(where_parts_rel)
            # Respect explicit order_by without order_dir -> default ASC (override relation default)
            effective_order_dir2 = rel_cfg.get('order_dir')
            try:
                if rel_cfg.get('_has_explicit_order_by') and not rel_cfg.get('_has_explicit_order_dir'):
                    effective_order_dir2 = 'asc'
            except Exception:
                pass
            order_clause = adapter.build_order_clause(
                child_model_cls,
                child_table,
                rel_cfg.get('order_by'),
                effective_order_dir2,
                self.registry._normalize_order_multi_values(rel_cfg.get('order_multi') or []),
            )
            return adapter.build_list_relation_json(
                child_table=child_table,
                projected_columns=requested_scalar_local or ['id'],
                where_condition=where_clause,
                limit=rel_cfg.get('limit'),
                offset=rel_cfg.get('offset'),
                order_by=order_clause,
                nested_subqueries=None,
            )
        # Non-MSSQL: correlated aggregation
        inner_cols = [getattr(child_model_cls, c) for c in (requested_scalar_local or ['id'])]
        inner_sel = (
            select(*inner_cols)
            .select_from(child_model_cls)
            .where(fk_child_to_parent_col == parent_model_cls.id)
            .correlate(parent_model_cls)
        )
        # where/default where
        r_where = self._resolve_graphql_value(info, rel_cfg.get('where'))
        if r_where is not None and isinstance(r_where, (dict, str)):
            wdict = to_where_dict(r_where, strict=True)
            if wdict:
                expr = expr_from_where_dict(child_model_cls, wdict, strict=True)
                if expr is not None:
                    inner_sel = inner_sel.where(expr)
        d_where = self._resolve_graphql_value(info, rel_cfg.get('default_where'))
        if d_where is not None and isinstance(d_where, (dict, str)):
            try:
                wdict = to_where_dict(d_where, strict=False)
            except Exception:
                wdict = None
            if wdict:
                try:
                    expr = expr_from_where_dict(child_model_cls, wdict, strict=False)
                except Exception:
                    expr = None
                if expr is not None:
                    inner_sel = inner_sel.where(expr)
        elif callable(rel_cfg.get('default_where')):
            # Apply callable default where by building a SQLAlchemy expression against the child model
            try:
                expr = rel_cfg.get('default_where')(child_model_cls, info)
            except Exception:
                expr = None
            if expr is not None:
                inner_sel = inner_sel.where(expr)
        # filter args (SQLA expr path)
        fa = rel_cfg.get('filter_args') or {}
        if fa:
            expanded: Dict[str, Any] = {}
            for key, raw in (rel_cfg.get('arg_specs') or {}).items():
                try:
                    from ..core.filters import normalize_filter_spec as _norm
                    spec = _norm(raw)
                except Exception:
                    continue
                if getattr(spec, 'ops', None) and not getattr(spec, 'op', None):
                    for op_name in spec.ops:
                        base = spec.alias or key
                        an = base if base.endswith(f"_{op_name}") else f"{base}_{op_name}"
                        expanded[an] = spec.clone_with(op=op_name, ops=None)
                else:
                    expanded[spec.alias or key] = spec
            from ..core.filters import OPERATOR_REGISTRY
            for arg_name, val in fa.items():
                f_spec = expanded.get(arg_name)
                if not f_spec:
                    continue
                if f_spec.transform:
                    try:
                        val = f_spec.transform(val)
                    except Exception:
                        continue
                expr = None
                if f_spec.builder:
                    try:
                        expr = f_spec.builder(child_model_cls, info, val)
                    except Exception:
                        expr = None
                elif f_spec.column:
                    try:
                        col = child_model_cls.__table__.c.get(f_spec.column)
                    except Exception:
                        col = None
                    if col is None:
                        continue
                    op_fn = OPERATOR_REGISTRY.get(f_spec.op or 'eq')
                    if not op_fn:
                        continue
                    try:
                        expr = op_fn(col, val)
                    except Exception:
                        expr = None
                if expr is not None:
                    inner_sel = inner_sel.where(expr)
        # ordering
        try:
            ordered = False
            allowed = [sf for sf, sd in self.registry.types[rel_cfg.get('target')].__berry_fields__.items() if sd.kind == 'scalar'] if rel_cfg.get('target') in self.registry.types else []
            for spec in self.registry._normalize_order_multi_values(rel_cfg.get('order_multi') or []):
                cn, _, dd = spec.partition(':')
                dd = (dd or 'asc').lower()
                if cn in allowed:
                    col = getattr(child_model_cls, cn, None)
                    if col is not None:
                        inner_sel = inner_sel.order_by(col.desc() if dd == 'desc' else col.asc())
                        ordered = True
            if not ordered and rel_cfg.get('order_by') in allowed:
                ob = rel_cfg.get('order_by')
                dir_v = rel_cfg.get('order_dir')
                try:
                    from ..core.utils import dir_value as _dir_value
                    dd = _dir_value(dir_v)
                except Exception:
                    dd = 'asc'
                col = getattr(child_model_cls, ob, None)
                if col is not None:
                    inner_sel = inner_sel.order_by(col.desc() if dd == 'desc' else col.asc())
                    ordered = True
            if not ordered and 'id' in child_model_cls.__table__.columns:
                inner_sel = inner_sel.order_by(getattr(child_model_cls, 'id').asc())
        except Exception:
            pass
        # pagination
        try:
            if rel_cfg.get('offset') is not None:
                o = rel_cfg.get('offset')
                inner_sel = inner_sel.offset(int(o) if isinstance(o, (int, str)) else o)
            if rel_cfg.get('limit') is not None:
                l = rel_cfg.get('limit')
                inner_sel = inner_sel.limit(int(l) if isinstance(l, (int, str)) else l)
        except Exception:
            pass
        limited_subq = inner_sel.subquery()
        row_json_args: List[Any] = []
        for sf in (requested_scalar_local or ['id']):
            row_json_args.extend([_text(f"'{sf}'"), getattr(limited_subq.c, sf)])
        row_json_expr = json_object_fn(*row_json_args)
        agg_inner_expr = json_array_agg_fn(row_json_expr)
        if agg_inner_expr is None:
            return None
        agg_query = select(json_array_coalesce_fn(agg_inner_expr)).select_from(limited_subq).correlate(parent_model_cls)
        try:
            return agg_query.scalar_subquery()
        except Exception:
            return agg_query
