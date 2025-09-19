from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select, text as _text, literal_column, func

# Centralized SQL builders. Keep registry thin and DRY.


class RelationSQLBuilders:
    def __init__(self, registry):
        self.registry = registry

    # --- helpers -------------------------------------------------------------
    def _pk_col(self, model_cls):
        """Get SQLAlchemy primary key column using registry helper."""
        return self.registry._get_pk_column(model_cls)

    def _pk_name(self, model_cls) -> str:
        """Get primary key column name using registry helper."""
        return self.registry._get_pk_name(model_cls)

    def _project_columns(self, model_cls, columns: list[str] | None) -> list:
        """Map GraphQL scalar names to SQLAlchemy columns with aliasing.

        Rules:
        - If Berry field has meta.column mapping, project that source column and label as GraphQL name.
        - Otherwise, project model attribute with same name.
        - Always ensure the physical PK column is present labeled to its physical name for correlation.
        - Unknown names are skipped.
        """
        cols = list(columns or [])
        out: list = []
        seen_labels: set[str] = set()
        # Fetch Berry field defs if available to consult meta.column
        btype = None
        try:
            # Reverse lookup: model -> btype by scanning registry
            for _name, _bt in (self.registry.types or {}).items():
                if getattr(_bt, 'model', None) is model_cls:
                    btype = _bt
                    break
        except Exception:
            btype = None
        fdefs = getattr(btype, '__berry_fields__', {}) if btype is not None else {}
        for name in cols:
            source_col_name = None
            try:
                fd = fdefs.get(name)
                if fd and getattr(fd, 'kind', None) == 'scalar':
                    source_col_name = (getattr(fd, 'meta', {}) or {}).get('column')
            except Exception:
                source_col_name = None
            sa_col = None
            if source_col_name:
                try:
                    sa_col = getattr(model_cls.__table__.c, source_col_name, None) or model_cls.__table__.c.get(source_col_name)
                except Exception:
                    sa_col = getattr(model_cls, source_col_name, None)
                if sa_col is not None:
                    try:
                        out.append(sa_col.label(name))
                        seen_labels.add(str(name))
                    except Exception:
                        out.append(sa_col)
                        seen_labels.add(getattr(sa_col, 'name', str(name)))
                    continue
            # Fallback: attribute with same name
            sa_col = getattr(model_cls, name, None)
            if sa_col is not None:
                out.append(sa_col)
                seen_labels.add(str(name))
        # Ensure the physical PK is available for correlation under its physical name
        try:
            pk_name = self._pk_name(model_cls)
            if pk_name and pk_name not in seen_labels:
                try:
                    out.append(self._pk_col(model_cls).label(pk_name))
                except Exception:
                    pass
        except Exception:
            pass
        return out

    def _mssql_map_columns_pairs(self, model_cls, columns: list[str] | None) -> list[tuple[str, str]]:
        """Map GraphQL scalar field names to physical column names for MSSQL adapter.

        Returns a list of (source_column_name, alias_name) pairs. Honors field meta['column']
        mapping when present; otherwise uses the same name. Unknown names are skipped.
        """
        cols = list(columns or [])
        out: list[tuple[str, str]] = []
        # lookup btype to access field meta
        btype = None
        try:
            for _name, _bt in (self.registry.types or {}).items():
                if getattr(_bt, 'model', None) is model_cls:
                    btype = _bt
                    break
        except Exception:
            btype = None
        fdefs = getattr(btype, '__berry_fields__', {}) if btype is not None else {}
        for name in cols:
            src = None
            try:
                fd = fdefs.get(name)
                if fd and getattr(fd, 'kind', None) == 'scalar':
                    src = (getattr(fd, 'meta', {}) or {}).get('column')
            except Exception:
                src = None
            if not src:
                # ensure the column exists on the table
                try:
                    if name in model_cls.__table__.columns:
                        src = name
                except Exception:
                    src = name if hasattr(model_cls, name) else None
            if src:
                out.append((src, name))
        return out

    def _prepare_requested_scalar_fields(
        self,
        *,
        target_btype,
        child_model_cls,
        rel_cfg: Dict[str, Any] | None,
        include_helper_fk_for_single_nested: bool = True,
    ) -> List[str]:
        """Return a sanitized list of scalar fields to project for the child model.

        - Honors rel_cfg['fields'] if provided and filters to known scalar fields.
        - Otherwise includes all scalar berry fields on target type.
        - Optionally includes helper FK columns for child's single nested relations that are requested.
        """
        rel_cfg = rel_cfg or {}
        requested: List[str] = list(rel_cfg.get('fields') or [])
        # Normalize requested field names: allow GraphQL camelCase by decamelizing to snake_case
        import re as _re
        def _decamel(name: str) -> str:
            try:
                if not name or ('_' in name):
                    return name
                s1 = _re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                s2 = _re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
                return s2.lower()
            except Exception:
                return name
        try:
            if requested:
                tmp: list[str] = []
                fdefs = getattr(target_btype, '__berry_fields__', {})
                for sf in requested:
                    key = sf
                    fdef = fdefs.get(key)
                    if not fdef:
                        cand = _decamel(str(sf))
                        fdef = fdefs.get(cand)
                        if fdef:
                            key = cand
                    if fdef and getattr(fdef, 'kind', None) == 'scalar':
                        # Exclude write-only helper scalars
                        try:
                            if (getattr(fdef, 'meta', {}) or {}).get('write_only'):
                                continue
                        except Exception:
                            pass
                        if key not in tmp:
                            tmp.append(key)
                requested = tmp
        except Exception:
            pass
        if not requested:
            try:
                for sf, sdef in getattr(target_btype, '__berry_fields__', {}).items():
                    if sdef.kind == 'scalar':
                        # Exclude write-only helper scalars from SQL projections
                        try:
                            if (getattr(sdef, 'meta', {}) or {}).get('write_only'):
                                continue
                        except Exception:
                            pass
                        requested.append(sf)
            except Exception:
                requested = []
        if include_helper_fk_for_single_nested:
            try:
                nested_requested = set((rel_cfg.get('nested') or {}).keys())
                for r2, d2 in getattr(target_btype, '__berry_fields__', {}).items():
                    meta = (getattr(d2, 'meta', {}) or {})
                    if (
                        d2.kind == 'relation'
                        and (meta.get('single') or meta.get('mode') == 'single')
                        and r2 in nested_requested
                    ):
                        fk2 = f"{r2}_id"
                        try:
                            if fk2 in child_model_cls.__table__.columns and fk2 not in requested:
                                requested.append(fk2)
                        except Exception:
                            pass
            except Exception:
                pass
        # Don't auto-append a special id alias; PK is handled internally for correlation
        return requested

    def _build_single_child_select_sqla(
        self,
        *,
        parent_model_cls,
        child_model_cls,
        pfk_col_name: str,
        projected_columns: List[str],
        rel_where: Any,
        rel_default_where: Any,
        filter_args: Dict[str, Any] | None,
        arg_specs: Dict[str, Any] | None,
        to_where_dict,
        expr_from_where_dict,
        info,
    ):
        """Build a correlated SA select for a single relation (parent has FK to child).

    Applies where/default_where, filter args, ordering fallback to id, and limit 1.
    """
        cols = projected_columns or []
        child_id_col = self._pk_col(child_model_cls)
        parent_fk_col = getattr(parent_model_cls, pfk_col_name)
        inner_cols = self._project_columns(child_model_cls, cols)
        sel = (
            select(*inner_cols)
            .select_from(child_model_cls)
            .where(child_id_col == parent_fk_col)
            .correlate(parent_model_cls)
            .limit(1)
        )
        # where/default_where
        sel = self._apply_where_sqla(
            sel,
            child_model_cls,
            rel_where,
            strict=True,
            to_where_dict=to_where_dict,
            expr_from_where_dict=expr_from_where_dict,
            info=info,
        )
        sel = self._apply_where_sqla(
            sel,
            child_model_cls,
            rel_default_where,
            strict=True,
            to_where_dict=to_where_dict,
            expr_from_where_dict=expr_from_where_dict,
            info=info,
        )
        # Apply type-level default scope if provided
        try:
            type_default_where = getattr(self, 'registry', None) and None
        except Exception:
            type_default_where = None
        # In this single-relation builder path we don't have rel_cfg; the caller combines values.
        # filter args
        expanded_specs = self._expand_arg_specs(arg_specs or {})
        sel = self._apply_filter_args_sqla(sel, child_model_cls, info, filter_args or {}, expanded_specs)
        # ensure deterministic ordering when not unique (by PK)
        try:
            sel = sel.order_by(self._pk_col(child_model_cls).asc())
        except Exception:
            pass
        return sel

    def _build_list_child_select_sqla(
        self,
        *,
        parent_model_cls,
        child_model_cls,
        fk_child_to_parent_col,
        projected_columns: List[str],
        rel_cfg: Dict[str, Any],
        to_where_dict,
        expr_from_where_dict,
        info,
    ):
        """Build a correlated SA select for a to-many relation (child has FK to parent).

        Applies relation where/default_where, filter args, ordering (order_multi -> order_by -> id), and pagination.
        """
        inner_cols = self._project_columns(child_model_cls, projected_columns or [])
        sel = (
            select(*inner_cols)
            .select_from(child_model_cls)
            .where(fk_child_to_parent_col == self._pk_col(parent_model_cls))
            .correlate(parent_model_cls)
        )
        # where/default_where
        sel = self._apply_where_sqla(
            sel,
            child_model_cls,
            rel_cfg.get('where'),
            strict=True,
            to_where_dict=to_where_dict,
            expr_from_where_dict=expr_from_where_dict,
            info=info,
        )
        sel = self._apply_where_sqla(
            sel,
            child_model_cls,
            rel_cfg.get('default_where'),
            strict=True,
            to_where_dict=to_where_dict,
            expr_from_where_dict=expr_from_where_dict,
            info=info,
        )
        # Combine with type-level default where when present
        try:
            t_where = rel_cfg.get('type_default_where')
        except Exception:
            t_where = None
        if t_where is not None:
            sel = self._apply_where_sqla(
                sel,
                child_model_cls,
                t_where,
                strict=True,
                to_where_dict=to_where_dict,
                expr_from_where_dict=expr_from_where_dict,
                info=info,
            )
        # filter args
        expanded_specs = self._expand_arg_specs(rel_cfg.get('arg_specs') or {})
        sel = self._apply_filter_args_sqla(sel, child_model_cls, info, rel_cfg.get('filter_args') or {}, expanded_specs)
        # ordering
        try:
            allowed = [
                sf for sf, sd in self.registry.types[rel_cfg.get('target')].__berry_fields__.items() if sd.kind == 'scalar'
            ] if rel_cfg.get('target') in self.registry.types else []
        except Exception:
            allowed = []
        try:
            from ..core.utils import dir_value
        except Exception:
            def _dir_value(x):
                return (x or 'asc')
        sel = self._apply_ordering_sqla(
            sel,
            child_model_cls,
            allowed,
            order_by=rel_cfg.get('order_by'),
            order_dir=rel_cfg.get('order_dir'),
            order_multi=self.registry._normalize_order_multi_values(rel_cfg.get('order_multi') or []),
            dir_value_fn=dir_value,
            default_dir_for_multi='asc',
            fallback_id=True,
        )
        # pagination
        sel = self._apply_pagination_sqla(sel, rel_cfg.get('limit'), rel_cfg.get('offset'))
        return sel
    def _expand_arg_specs(self, arg_specs: Dict[str, Any] | None) -> Dict[str, Any]:
        """Normalize filter arg specs: expand ops to concrete names and honor aliases.

        Returns a dict mapping effective arg name -> spec (possibly cloned with concrete op).
        """
        if not arg_specs:
            return {}
        try:
            from ..core.filters import normalize_filter_spec as _normalize_filter_spec
        except Exception:
            return {}
        expanded: Dict[str, Any] = {}
        for key, raw in (arg_specs or {}).items():
            try:
                spec = _normalize_filter_spec(raw)
            except Exception:
                continue
            if getattr(spec, 'ops', None) and not getattr(spec, 'op', None):
                if spec.ops is not None:
                    for op_name in spec.ops:
                        base = spec.alias or key
                        an = base if str(base).endswith(f"_{op_name}") else f"{base}_{op_name}"
                        try:
                            expanded[an] = spec.clone_with(op=op_name, ops=None)
                        except Exception:
                            continue
            else:
                expanded[spec.alias or key] = spec
        return expanded

    def _apply_filter_args_sqla(self, sel, model_cls, info, filter_args: Dict[str, Any] | None, expanded_specs: Dict[str, Any]):
        """Apply filter_args using SQLAlchemy expressions."""
        if not filter_args:
            return sel
        try:
            from ..core.filters import OPERATOR_REGISTRY
        except Exception:
            OPERATOR_REGISTRY = {}
        for arg_name, val in (filter_args or {}).items():
            f_spec = expanded_specs.get(arg_name)
            if not f_spec:
                continue
            if getattr(f_spec, 'transform', None):
                try:
                    val = f_spec.transform(val)
                except Exception:
                    continue
            expr = None
            if getattr(f_spec, 'builder', None):
                try:
                    expr = f_spec.builder(model_cls, info, val)
                except Exception:
                    expr = None
            elif getattr(f_spec, 'column', None):
                try:
                    col = model_cls.__table__.c.get(f_spec.column)
                except Exception:
                    col = None
                if col is None:
                    continue
                op_fn = OPERATOR_REGISTRY.get(getattr(f_spec, 'op', None) or 'eq')
                if not op_fn:
                    continue
                try:
                    expr = op_fn(col, val)
                except Exception:
                    expr = None
            if expr is not None:
                sel = sel.where(expr)
        return sel

    def _apply_filter_args_mssql(self, where_parts: List[str], model_cls, filter_args: Dict[str, Any] | None, expanded_specs: Dict[str, Any], adapter):
        """Translate filter_args into MSSQL where fragments using adapter.where_from_dict.

        Extends where_parts in-place and returns it.
        """
        if not filter_args:
            return where_parts
        try:
            from ..core.filters import OPERATOR_REGISTRY
        except Exception:
            OPERATOR_REGISTRY = {}

        def _add(col_name: str, op_name: str, value: Any):
            v = value
            if op_name in ('like', 'ilike') and isinstance(v, str) and '%' not in v and '_' not in v:
                v = f"%{v}%"
            try:
                where_parts.extend(adapter.where_from_dict(model_cls, {col_name: {op_name: v}}))
            except Exception:
                pass

        for arg_name, val in (filter_args or {}).items():
            f_spec = expanded_specs.get(arg_name)
            if f_spec:
                if getattr(f_spec, 'transform', None):
                    try:
                        val = f_spec.transform(val)
                    except Exception:
                        pass
                if getattr(f_spec, 'column', None):
                    _add(f_spec.column, getattr(f_spec, 'op', None) or 'eq', val)
                    continue
            # Fallback to suffix operators when spec not found
            try:
                an = str(arg_name)
                if '_' in an:
                    base, suffix = an.rsplit('_', 1)
                    if suffix.lower() in OPERATOR_REGISTRY and base in model_cls.__table__.columns:
                        _add(base, suffix.lower(), val)
            except Exception:
                pass
        return where_parts

    def _apply_where_common(self, sel, model_cls, value, *, strict: bool, to_where_dict, expr_from_where_dict, info):
        """Common implementation to apply dict/str/callable/expr where to a SA selectable.

        Used by both relation builders and root filters to keep behavior consistent.
        """
        v = self._resolve_graphql_value(info, value)
        if v is None:
            return sel
        # If a list/tuple of fragments is passed, apply all in order
        try:
            if isinstance(v, (list, tuple)):
                for part in v:
                    sel = self._apply_where_common(sel, model_cls, part, strict=strict, to_where_dict=to_where_dict, expr_from_where_dict=expr_from_where_dict, info=info)
                return sel
        except Exception:
            pass
        if isinstance(v, (dict, str)):
            # Do not suppress parse errors here; rely on strict flag in helpers to control behavior
            wdict = to_where_dict(v, strict=strict)
            if wdict:
                expr = expr_from_where_dict(model_cls, wdict, strict=strict)
                if expr is not None:
                    sel = sel.where(expr)
            return sel
        # callable or direct expression
        if callable(v):
            rv = v(model_cls, info)
            # If callable returns dict/str, parse to SQL expression
            if isinstance(rv, (dict, str)):
                wdict2 = to_where_dict(rv, strict=strict)
                if wdict2:
                    expr2 = expr_from_where_dict(model_cls, wdict2, strict=strict)
                    if expr2 is not None:
                        sel = sel.where(expr2)
                return sel
            # If callable returns a SQL expression, apply it; ignore None
            if rv is not None:
                sel = sel.where(rv)
            return sel
        # direct SQL expression (non-callable, non-dict/str)
        sel = sel.where(v)
        return sel

    def _apply_where_sqla(self, sel, model_cls, value, *, strict: bool, to_where_dict, expr_from_where_dict, info):
        """Backwards-compatible shim calling the common where applier."""
        return self._apply_where_common(sel, model_cls, value, strict=strict, to_where_dict=to_where_dict, expr_from_where_dict=expr_from_where_dict, info=info)

    def _mssql_where_from_value(self, where_parts: List[str], model_cls, value, *, strict: bool, to_where_dict, expr_from_where_dict, adapter, info) -> List[str]:
        """Append MSSQL WHERE fragments for dict/str values. Validates strict=True via SQLAlchemy path.

        Also supports callables that return dict/str by evaluating them with (model_cls, info).
        Direct SQLAlchemy expressions are not supported in the MSSQL string builder and will be ignored here.
        """
        v = self._resolve_graphql_value(info, value)
        if v is None:
            return where_parts
        # If a list/tuple of fragments is passed, process each
        if isinstance(v, (list, tuple)):
            for part in v:
                where_parts = self._mssql_where_from_value(
                    where_parts,
                    model_cls,
                    part,
                    strict=strict,
                    to_where_dict=to_where_dict,
                    expr_from_where_dict=expr_from_where_dict,
                    adapter=adapter,
                    info=info,
                )
            return where_parts
        # Evaluate callables to a concrete dict/str when possible
        if callable(v):
            rv = v(model_cls, info)
            # Only handle dict/str results here; SQL expressions aren't supported in MSSQL string builder
            if isinstance(rv, (dict, str)):
                return self._mssql_where_from_value(
                    where_parts,
                    model_cls,
                    rv,
                    strict=strict,
                    to_where_dict=to_where_dict,
                    expr_from_where_dict=expr_from_where_dict,
                    adapter=adapter,
                    info=info,
                )
            return where_parts
        if isinstance(v, (dict, str)):
            # Do not wrap in try/except; helper handles strict behavior
            wdict = to_where_dict(v, strict=strict)
            if wdict:
                if strict:
                    # validate to raise on malformed/unknowns
                    _ = expr_from_where_dict(model_cls, wdict, strict=True)
                # Allow adapter to raise if something unexpected occurs; don't silently drop
                where_parts.extend(adapter.where_from_dict(model_cls, wdict))
        return where_parts

    def _json_row_args_from_subq(self, subq, columns: List[str] | None) -> List[Any]:
        args: List[Any] = []
        cols = columns or []
        for c in cols:
            args.extend([_text(f"'{c}'"), getattr(subq.c, c)])
        return args

    @staticmethod
    def _ensure_helper_fk_for_single_nested(requested: List[str], child_model_cls, target_b, nested_map: Dict[str, Any]) -> List[str]:
        out = list(requested or [])
        try:
            for rname, fdef in (getattr(target_b, '__berry_fields__', {}) or {}).items():
                try:
                    is_rel = getattr(fdef, 'kind', None) == 'relation'
                    meta = getattr(fdef, 'meta', {}) or {}
                    if is_rel and (meta.get('single') or meta.get('mode') == 'single') and rname in (nested_map or {}):
                        fk_col_name = f"{rname}_id"
                        try:
                            if fk_col_name in child_model_cls.__table__.columns and fk_col_name not in out:
                                out.append(fk_col_name)
                        except Exception:
                            pass
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def _apply_ordering_sqla(
        self,
        sel,
        model_cls,
        allowed_fields: List[str] | None,
        *,
        order_by: Optional[str],
        order_dir: Optional[str],
        order_multi: List[str] | None,
        dir_value_fn=None,
        default_dir_for_multi: str = 'asc',
        fallback_id: bool = True,
    ):
        ordered = False
        for spec in (order_multi or []):
            cn, _, dd = spec.partition(':')
            dd = (dd or default_dir_for_multi).lower()
            if not allowed_fields or cn in allowed_fields:
                col = getattr(model_cls, cn, None)
                if col is not None:
                    sel = sel.order_by(col.desc() if dd == 'desc' else col.asc())
                    ordered = True
        if not ordered and (not allowed_fields or order_by in (allowed_fields or [])):
            if order_by:
                dd = None
                try:
                    dd = (dir_value_fn(order_dir) if dir_value_fn else order_dir) or 'asc'
                except Exception:
                    dd = (order_dir or 'asc')
                dd = dd.lower()
                col = getattr(model_cls, order_by, None)
                if col is not None:
                    sel = sel.order_by(col.desc() if dd == 'desc' else col.asc())
                    ordered = True
        if fallback_id and not ordered:
            try:
                sel = sel.order_by(self._pk_col(model_cls).asc())
            except Exception:
                pass
        return sel

    @staticmethod
    def _effective_order_dir(cfg: Dict[str, Any] | None) -> Optional[str]:
        if not cfg:
            return None
        try:
            if cfg.get('_has_explicit_order_by') and not cfg.get('_has_explicit_order_dir'):
                return 'asc'
        except Exception:
            pass
        return cfg.get('order_dir')

    @staticmethod
    def _apply_pagination_sqla(sel, limit, offset):
        try:
            if offset is not None:
                o = offset
                sel = sel.offset(int(o) if isinstance(o, (int, str)) else o)
            if limit is not None:
                l = limit
                sel = sel.limit(int(l) if isinstance(l, (int, str)) else l)
        except Exception:
            # Best-effort; ignore pagination errors
            pass
        return sel

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
        type_default_where: Any,
        filter_args: Dict[str, Any] | None,
        arg_specs: Dict[str, Any] | None,
    ):
        # If we don't know how to correlate parent->child (no FK on parent), skip pushdown.
        # Also skip when rel_name suggests a private/virtual relation used only for computed wrappers.
        if not parent_fk_col_name or (isinstance(rel_name, str) and rel_name.startswith('_')):
            return None
        cols = projected_columns or []
        mssql_mode = getattr(adapter, 'name', '') == 'mssql'
        # Determine parent FK name for correlation (parent has FK to child)
        pfk = parent_fk_col_name or f"{rel_name}_id"
        if mssql_mode:
            # Compose WHERE join + relation filters, then let adapter emit FOR JSON PATH
            child_table = child_model_cls  # pass model for schema-aware identifier
            parent_table = parent_model_cls
            # child's PK equals parent's FK
            child_pk_name = self._pk_name(child_model_cls)
            try:
                ct_ident = adapter.table_ident(child_table)
                pt_ident = adapter.table_ident(parent_table)
            except Exception:
                ct_ident = f"[{getattr(child_model_cls,'__tablename__','')}]"
                pt_ident = f"[{getattr(parent_model_cls,'__tablename__','')}]"
            where_parts: list[str] = [f"{ct_ident}.[{child_pk_name}] = {pt_ident}.[{pfk}]"]
            # Apply relation where/default_where fragments
            where_parts = self._mssql_where_from_value(
                where_parts,
                child_model_cls,
                rel_where,
                strict=True,
                to_where_dict=to_where_dict,
                expr_from_where_dict=expr_from_where_dict,
                adapter=adapter,
                info=info,
            )
            where_parts = self._mssql_where_from_value(
                where_parts,
                child_model_cls,
                rel_default_where,
                strict=True,
                to_where_dict=to_where_dict,
                expr_from_where_dict=expr_from_where_dict,
                adapter=adapter,
                info=info,
            )
            if type_default_where is not None:
                where_parts = self._mssql_where_from_value(
                    where_parts,
                    child_model_cls,
                    type_default_where,
                    strict=True,
                    to_where_dict=to_where_dict,
                    expr_from_where_dict=expr_from_where_dict,
                    adapter=adapter,
                    info=info,
                )
            # Apply filter args best-effort on MSSQL single relation as well
            try:
                fa = filter_args or {}
                expanded = self._expand_arg_specs(arg_specs or {})
                where_parts = self._apply_filter_args_mssql(where_parts, child_model_cls, fa, expanded, adapter)
            except Exception:
                pass
            where_sql = ' AND '.join(where_parts)
            # Use adapter helper for a single object
            # On MSSQL, pass (source, alias) pairs so JSON keys remain GraphQL names
            cols_pairs = self._mssql_map_columns_pairs(child_model_cls, cols)
            return adapter.build_single_relation_json(
                child_table=child_table,
                projected_columns=cols_pairs or [(self._pk_name(child_model_cls), self._pk_name(child_model_cls))],
                join_condition=where_sql,
            )
        # Non-MSSQL: correlated select limited to 1, return JSON object or null
        inner_sel = self._build_single_child_select_sqla(
            parent_model_cls=parent_model_cls,
            child_model_cls=child_model_cls,
            pfk_col_name=pfk,
            projected_columns=cols,
            rel_where=rel_where,
            rel_default_where=rel_default_where,
            filter_args=filter_args,
            arg_specs=arg_specs,
            to_where_dict=to_where_dict,
            expr_from_where_dict=expr_from_where_dict,
            info=info,
        )
        # Apply type-level default where after building inner select
        if type_default_where is not None:
            inner_sel = self._apply_where_sqla(
                inner_sel,
                child_model_cls,
                type_default_where,
                strict=True,
                to_where_dict=to_where_dict,
                expr_from_where_dict=expr_from_where_dict,
                info=info,
            )
        limited = inner_sel.subquery()
        obj_expr = json_object_fn(*self._json_row_args_from_subq(limited, cols))
        query = select(obj_expr).select_from(limited).correlate(parent_model_cls)
        try:
            return query.scalar_subquery()
        except Exception:
            return query

    # --- list relation (JSON array) -----------------------------------------
    def build_list_relation_json_recursive(
        self,
        *,
        parent_model_cls,
        parent_btype,
        rel_cfg: Dict[str, Any],
        json_object_fn,
        json_array_agg_fn,
        json_array_coalesce_fn,
        to_where_dict,
        expr_from_where_dict,
        dir_value_fn,
        info,
    ):
        """Build a non-MSSQL JSON array aggregation for a list relation including arbitrarily nested relations.

        Mirrors the previous inline builder from registry._base_impl, but lives here for reuse.
        """
        try:
            target_name_i = rel_cfg.get('target')
            target_b_i = self.registry.types.get(target_name_i)
            if not target_b_i or not target_b_i.model:
                return None
            child_model_cls_i = target_b_i.model
            # FK child -> parent
            fk_col_i = None
            for col in child_model_cls_i.__table__.columns:
                for fk in col.foreign_keys:
                    if fk.column.table.name == parent_model_cls.__table__.name:
                        fk_col_i = col
                        break
                if fk_col_i is not None:
                    break
            if fk_col_i is None:
                return None

            # Recursive builder for nested fields under a given subquery/model
            def _build_nested_fields_for_subq(current_subq, current_model_cls, nested_cfg: Dict[str, Any] | None):
                out_json_args: list[Any] = []
                for nname, ncfg in (nested_cfg or {}).items():
                    try:
                        n_target = ncfg.get('target')
                        nb = self.registry.types.get(n_target)
                        if not nb or not nb.model:
                            continue
                        grand_model = nb.model
                        # FK grandchild -> current_model_cls (respect explicit fk_column_name when provided)
                        g_fk = None
                        try:
                            explicit_child_fk = ncfg.get('fk_column_name')
                        except Exception:
                            explicit_child_fk = None
                        try:
                            if explicit_child_fk and hasattr(grand_model, '__table__'):
                                for c in grand_model.__table__.columns:
                                    if c.name == explicit_child_fk:
                                        g_fk = c
                                        break
                        except Exception:
                            g_fk = None
                        if g_fk is None:
                            for c in grand_model.__table__.columns:
                                for fk in c.foreign_keys:
                                    if fk.column.table.name == current_model_cls.__table__.name:
                                        g_fk = c
                                        break
                                if g_fk is not None:
                                    break
                        if g_fk is None:
                            continue
                        # Projected scalar fields for nested
                        n_cols = list(ncfg.get('fields') or [])
                        if n_cols:
                            try:
                                n_cols = [sf for sf in n_cols if getattr(grand_model.__table__.c, sf, None) is not None or sf in grand_model.__table__.columns]
                            except Exception:
                                pass
                        if not n_cols:
                            for sf2, sd2 in nb.__berry_fields__.items():
                                if sd2.kind == 'scalar':
                                    n_cols.append(sf2)
                        is_single_nested = bool(ncfg.get('single') or False)
                        # Base select for nested correlated to current_subq PK column
                        n_sel = (
                            select(*self._project_columns(grand_model, n_cols or []))
                            .select_from(grand_model)
                            .where(g_fk == getattr(current_subq.c, self._pk_name(current_model_cls)))
                            .correlate(current_subq)
                        )
                        # Apply nested where/default_where
                        rr2 = ncfg.get('where')
                        if rr2 is not None:
                            if isinstance(rr2, (dict, str)):
                                wdict2 = to_where_dict(rr2, strict=True)
                                if wdict2:
                                    expr2 = expr_from_where_dict(grand_model, wdict2, strict=True)
                                    if expr2 is not None:
                                        n_sel = n_sel.where(expr2)
                            else:
                                expr2 = rr2(grand_model, info) if callable(rr2) else rr2
                                if expr2 is not None:
                                    n_sel = n_sel.where(expr2)
                        dr2 = ncfg.get('default_where')
                        if dr2 is not None:
                            if isinstance(dr2, (dict, str)):
                                wdict2 = to_where_dict(dr2, strict=True)
                                if wdict2:
                                    expr2 = expr_from_where_dict(grand_model, wdict2, strict=True)
                                    if expr2 is not None:
                                        n_sel = n_sel.where(expr2)
                            else:
                                expr2 = dr2(grand_model, info) if callable(dr2) else dr2
                                if expr2 is not None:
                                    n_sel = n_sel.where(expr2)
                        # Also apply type-level scope for the nested target
                        try:
                            t_where_n = ncfg.get('type_default_where')
                        except Exception:
                            t_where_n = None
                        if t_where_n is not None:
                            n_sel = self._apply_where_sqla(
                                n_sel,
                                grand_model,
                                t_where_n,
                                strict=True,
                                to_where_dict=to_where_dict,
                                expr_from_where_dict=expr_from_where_dict,
                                info=info,
                            )
                        # Ordering for nested
                        ordered2 = False
                        n_allowed = [sf for sf, sd in nb.__berry_fields__.items() if sd.kind == 'scalar']
                        nmulti_raw = ncfg.get('order_multi') or []
                        nmulti: List[str] = self.registry._normalize_order_multi_values(nmulti_raw)
                        for spec in nmulti:
                            cn, _, dd = spec.partition(':')
                            # Default ASC for multi when direction not specified
                            dd = (dd or 'asc').lower()
                            if cn in n_allowed:
                                col2 = getattr(grand_model, cn, None)
                                if col2 is not None:
                                    n_sel = n_sel.order_by(col2.desc() if dd == 'desc' else col2.asc())
                                    ordered2 = True
                        if not ordered2 and ncfg.get('order_by') in n_allowed:
                            cn = ncfg.get('order_by')
                            # When explicit order_by provided without explicit dir, default to ASC
                            try:
                                eff_dir = self._effective_order_dir(ncfg)
                            except Exception:
                                eff_dir = None
                            dd = (eff_dir or dir_value_fn(ncfg.get('order_dir'))).lower()
                            col2 = getattr(grand_model, cn, None)
                            if col2 is not None:
                                n_sel = n_sel.order_by(col2.desc() if dd == 'desc' else col2.asc())
                                ordered2 = True
                        if not ordered2:
                            try:
                                n_sel = n_sel.order_by(self._pk_col(grand_model).asc())
                            except Exception:
                                pass
                        # Pagination for nested
                        try:
                            if ncfg.get('offset') is not None:
                                n_sel = n_sel.offset(int(ncfg.get('offset')) if isinstance(ncfg.get('offset'), (int, str)) else ncfg.get('offset'))
                            if ncfg.get('limit') is not None:
                                n_sel = n_sel.limit(int(ncfg.get('limit')) if isinstance(ncfg.get('limit'), (int, str)) else ncfg.get('limit'))
                        except Exception:
                            pass
                        if is_single_nested:
                            # Build JSON object of the first matching nested row, including deeper nested fields
                            n_sel_single = n_sel.limit(1)
                            n_subq = n_sel_single.subquery()
                            n_row_args: List[Any] = []
                            use_cols = (n_cols or [self._pk_name(grand_model)])
                            for sf2 in use_cols:
                                n_row_args.extend([_text(f"'{sf2}'"), getattr(n_subq.c, sf2)])
                            # recurse for deeper under this single
                            try:
                                deeper_args = _build_nested_fields_for_subq(n_subq, grand_model, ncfg.get('nested') or {})
                                if deeper_args:
                                    n_row_args.extend(deeper_args)
                            except Exception:
                                pass
                            n_row_json = json_object_fn(*n_row_args)
                            n_single_select = select(n_row_json).select_from(n_subq).correlate(current_subq)
                            try:
                                n_json_scalar = n_single_select.scalar_subquery()
                            except Exception:
                                n_json_scalar = n_single_select
                            out_json_args.extend([_text(f"'{nname}'"), n_json_scalar])
                            try:
                                ncfg['from_pushdown'] = True
                                ncfg['skip_reason'] = None
                            except Exception:
                                pass
                        else:
                            # Build JSON array for nested rows, including deeper nested fields
                            n_subq = n_sel.subquery()
                            n_row_args: List[Any] = []
                            use_cols = (n_cols or [self._pk_name(grand_model)])
                            for sf2 in use_cols:
                                n_row_args.extend([_text(f"'{sf2}'"), getattr(n_subq.c, sf2)])
                            try:
                                deeper_args = _build_nested_fields_for_subq(n_subq, grand_model, ncfg.get('nested') or {})
                                if deeper_args:
                                    n_row_args.extend(deeper_args)
                            except Exception:
                                pass
                            n_row_json = json_object_fn(*n_row_args)
                            n_agg = json_array_agg_fn(n_row_json)
                            if n_agg is None:
                                continue
                            n_json_array = select(json_array_coalesce_fn(n_agg)).select_from(n_subq).correlate(current_subq)
                            try:
                                n_json_scalar = n_json_array.scalar_subquery()
                            except Exception:
                                n_json_scalar = n_json_array
                            out_json_args.extend([_text(f"'{nname}'"), n_json_scalar])
                            try:
                                ncfg['from_pushdown'] = True
                                ncfg['skip_reason'] = None
                            except Exception:
                                pass
                    except Exception:
                        # Skip this nested branch on error
                        try:
                            ncfg['from_pushdown'] = False
                            if not ncfg.get('skip_reason'):
                                ncfg['skip_reason'] = 'builder error'
                        except Exception:
                            pass
                        continue
                return out_json_args

            # Scalars to project for child rows
            requested_scalar_i = list(rel_cfg.get('fields') or [])
            try:
                if requested_scalar_i:
                    tmp: list[str] = []
                    for sf in requested_scalar_i:
                        sdef = target_b_i.__berry_fields__.get(sf)
                        if sdef and sdef.kind == 'scalar':
                            tmp.append(sf)
                    requested_scalar_i = tmp
            except Exception:
                pass
            if not requested_scalar_i:
                for sf, sdef in target_b_i.__berry_fields__.items():
                    if sdef.kind == 'scalar':
                        # Filter out write-only helper scalars
                        try:
                            if (getattr(sdef, 'meta', {}) or {}).get('write_only'):
                                continue
                        except Exception:
                            pass
                        requested_scalar_i.append(sf)
            # Ensure FK helper columns for child's single nested relations are present
            try:
                nested_requested = set((rel_cfg.get('nested') or {}).keys())
                for r2, d2 in target_b_i.__berry_fields__.items():
                    if (
                        d2.kind == 'relation'
                        and (d2.meta.get('single') or d2.meta.get('mode') == 'single')
                        and r2 in nested_requested
                    ):
                        fk2 = f"{r2}_id"
                        if fk2 in child_model_cls_i.__table__.columns and fk2 not in requested_scalar_i:
                            requested_scalar_i.append(fk2)
            except Exception:
                pass
            # Do not auto-append logical 'id'; PK will be included internally when needed
            # Base inner select for child rows (correlated to parent)
            inner_cols_i = self._project_columns(child_model_cls_i, requested_scalar_i) if requested_scalar_i else [self._pk_col(child_model_cls_i)]
            inner_sel_i = select(*inner_cols_i).select_from(child_model_cls_i).where(fk_col_i == self._pk_col(parent_model_cls)).correlate(parent_model_cls)
            # Filters from filter_args
            try:
                filter_args = rel_cfg.get('filter_args') or {}
                if filter_args:
                    target_btype = self.registry.types.get(rel_cfg.get('target'))
                    if target_btype:
                        class_filters = rel_cfg.get('arg_specs') or {}
                        expanded: Dict[str, Any] = {}
                        for key, raw in class_filters.items():
                            try:
                                from ..core.filters import normalize_filter_spec as _normalize_filter_spec
                                spec = _normalize_filter_spec(raw)
                            except Exception:
                                continue
                            if spec.ops and not spec.op:
                                for op_name in spec.ops:
                                    base = spec.alias or key
                                    an = base if base.endswith(f"_{op_name}") else f"{base}_{op_name}"
                                    expanded[an] = spec.clone_with(op=op_name, ops=None)
                            else:
                                expanded[spec.alias or key] = spec
                        from ..core.filters import OPERATOR_REGISTRY
                        for arg_name, val in filter_args.items():
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
                                    expr = f_spec.builder(child_model_cls_i, info, val)
                                except Exception:
                                    expr = None
                            elif f_spec.column:
                                try:
                                    col = child_model_cls_i.__table__.c.get(f_spec.column)
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
                                inner_sel_i = inner_sel_i.where(expr)
            except Exception:
                pass
            # Relation JSON where/default_where for child
            rr = rel_cfg.get('where')
            if rr is not None:
                # Resolve GraphQL VariableNode and StringValueNode-like objects using outer-scope info
                try:
                    if not isinstance(rr, (str, dict)) and hasattr(rr, 'name'):
                        vname = getattr(getattr(rr, 'name', None), 'value', None) or getattr(rr, 'name', None)
                        var_vals = getattr(info, 'variable_values', None)
                        if var_vals is None:
                            raw_info = getattr(info, '_raw_info', None)
                            var_vals = getattr(raw_info, 'variable_values', None) if raw_info is not None else None
                        if isinstance(var_vals, dict) and vname in var_vals:
                            rr = var_vals[vname]
                except Exception:
                    pass
                try:
                    if not isinstance(rr, (dict, str)) and hasattr(rr, 'value'):
                        rr = getattr(rr, 'value')
                except Exception:
                    pass
                if isinstance(rr, (dict, str)):
                    wdict = to_where_dict(rr, strict=True)
                    if wdict:
                        expr = expr_from_where_dict(child_model_cls_i, wdict, strict=True)
                        if expr is not None:
                            inner_sel_i = inner_sel_i.where(expr)
                else:
                    try:
                        expr = rr(child_model_cls_i, info) if callable(rr) else rr
                    except Exception:
                        expr = None
                    if expr is not None:
                        inner_sel_i = inner_sel_i.where(expr)
            dr = rel_cfg.get('default_where')
            if dr is not None:
                if isinstance(dr, (dict, str)):
                    wdict = to_where_dict(dr, strict=True)
                    if wdict:
                        expr = expr_from_where_dict(child_model_cls_i, wdict, strict=True)
                        if expr is not None:
                            inner_sel_i = inner_sel_i.where(expr)
                else:
                    try:
                        expr = dr(child_model_cls_i, info) if callable(dr) else dr
                    except Exception:
                        expr = None
                    if expr is not None:
                        inner_sel_i = inner_sel_i.where(expr)
            # Ordering for child (order_multi -> order_by -> id)
            ordered = False
            # Allowed ordering fields: scalars excluding write-only helpers
            allowed_fields = []
            try:
                for sf, sd in target_b_i.__berry_fields__.items():
                    if sd.kind == 'scalar':
                        try:
                            if (getattr(sd, 'meta', {}) or {}).get('write_only'):
                                continue
                        except Exception:
                            pass
                        allowed_fields.append(sf)
            except Exception:
                allowed_fields = [sf for sf, sd in target_b_i.__berry_fields__.items() if sd.kind == 'scalar']
            nmulti: List[str] = self.registry._normalize_order_multi_values(rel_cfg.get('order_multi') or [])
            for spec in nmulti:
                cn, _, dd = spec.partition(':')
                # Default ASC for multi when direction not specified
                dd = (dd or 'asc').lower()
                if cn in allowed_fields:
                    col = getattr(child_model_cls_i, cn, None)
                    if col is not None:
                        inner_sel_i = inner_sel_i.order_by(col.desc() if dd == 'desc' else col.asc())
                        ordered = True
            if not ordered:
                ob_val = rel_cfg.get('order_by')
                if ob_val is not None:
                    # Support callable or direct SQLAlchemy expression for ordering
                    expr = None
                    if callable(ob_val):
                        try:
                            expr = ob_val(child_model_cls_i, info)
                        except Exception:
                            expr = None
                    elif hasattr(ob_val, 'desc') or hasattr(ob_val, 'asc'):
                        expr = ob_val
                    # Fallback: string column name (allowed_fields guard)
                    if expr is None and ob_val in allowed_fields:
                        expr = getattr(child_model_cls_i, ob_val, None)
                    if expr is not None:
                        try:
                            eff_dir_top = self._effective_order_dir(rel_cfg)
                        except Exception:
                            eff_dir_top = None
                        dd = (eff_dir_top or dir_value_fn(rel_cfg.get('order_dir')) or 'asc')
                        try:
                            inner_sel_i = inner_sel_i.order_by(expr.desc() if str(dd).lower() == 'desc' else expr.asc())
                            ordered = True
                        except Exception:
                            # As a last resort, try plain order_by(expr)
                            try:
                                inner_sel_i = inner_sel_i.order_by(expr)
                                ordered = True
                            except Exception:
                                pass
            if not ordered:
                try:
                    inner_sel_i = inner_sel_i.order_by(self._pk_col(child_model_cls_i).asc())
                except Exception:
                    pass
            # Pagination for child
            try:
                if rel_cfg.get('offset') is not None:
                    inner_sel_i = inner_sel_i.offset(int(rel_cfg.get('offset')) if isinstance(rel_cfg.get('offset'), (int, str)) else rel_cfg.get('offset'))
                if rel_cfg.get('limit') is not None:
                    inner_sel_i = inner_sel_i.limit(int(rel_cfg.get('limit')) if isinstance(rel_cfg.get('limit'), (int, str)) else rel_cfg.get('limit'))
            except Exception:
                pass
            limited_subq_i = inner_sel_i.subquery()
            # Build per-row JSON object including requested scalars and nested relation JSON (recursive)
            row_json_args_i: List[Any] = []
            for sf in (requested_scalar_i if requested_scalar_i else [self._pk_name(child_model_cls_i)]):
                row_json_args_i.extend([_text(f"'{sf}'"), getattr(limited_subq_i.c, sf)])
            try:
                row_json_args_i.extend(_build_nested_fields_for_subq(limited_subq_i, child_model_cls_i, rel_cfg.get('nested') or {}))
            except Exception:
                pass
            row_json_expr_i = json_object_fn(*row_json_args_i)
            agg_inner_expr_i = json_array_agg_fn(row_json_expr_i)
            if agg_inner_expr_i is None:
                return None
            agg_query_i = select(json_array_coalesce_fn(agg_inner_expr_i)).select_from(limited_subq_i).correlate(parent_model_cls)
            try:
                return agg_query_i.scalar_subquery()
            except Exception:
                return agg_query_i
        except Exception:
            return None

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
        # Prepare projected scalar fields consistently across adapters
        try:
            target_b = self.registry.types.get(rel_cfg.get('target')) if rel_cfg.get('target') else None
        except Exception:
            target_b = None
        requested_scalar_local: list[str]
        try:
            if target_b is not None and getattr(target_b, 'model', None) is not None:
                requested_scalar_local = self._prepare_requested_scalar_fields(
                    target_btype=target_b,
                    child_model_cls=child_model_cls,
                    rel_cfg=rel_cfg,
                    include_helper_fk_for_single_nested=True,
                )
            else:
                requested_scalar_local = list(requested_scalar or [])
        except Exception:
            requested_scalar_local = list(requested_scalar or [])
        if mssql_mode:
            parent_table = parent_model_cls
            parent_pk_name = self._pk_name(parent_model_cls)
            # If there are nested relations selected under this list relation, use the
            # MSSQL full nested JSON builder so everything is pushed down in one SELECT.
            nested_cfg_map: Dict[str, Any] = rel_cfg.get('nested') or {}
            if nested_cfg_map:
                nested_specs: list[dict] = []
                # Helper to resolve callable/variable type scope values to concrete dict/str
                def _resolve_type_where_local(raw_val, model_cls_local):
                    v = self._resolve_graphql_value(info, raw_val)
                    try:
                        if callable(v):
                            v = v(model_cls_local, info)
                    except Exception:
                        pass
                    return v
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
                        # Determine whether nested is single or list
                        is_single_i = bool(ncfg_i.get('single'))
                        g_fk_col_name_i: str | None = None
                        child_fk_to_nested_i: str | None = None
                        if is_single_i:
                            # For single nested, we need FK from parent child_model -> nested model
                            try:
                                # Try to infer from naming or metadata
                                rel_name_local = nname_i
                                fk_name_guess = f"{rel_name_local}_id"
                                # Use registry helper to find parent FK column name on parent_model_cls_local
                                pfk_name = self.registry._find_parent_fk_column_name(parent_model_cls_local, n_model_i, rel_name_local)
                                child_fk_to_nested_i = pfk_name or fk_name_guess
                                # Validate that such a column exists on parent_model_cls_local
                                if hasattr(parent_model_cls_local, '__table__'):
                                    if not any(c.name == child_fk_to_nested_i for c in parent_model_cls_local.__table__.columns):
                                        # If not found, skip this nested spec
                                        child_fk_to_nested_i = None
                            except Exception:
                                child_fk_to_nested_i = None
                            if not child_fk_to_nested_i:
                                # can't correlate this single nested; skip it
                                continue
                        else:
                            # For list nested, find FK from nested model -> parent child model
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
                        # Determine scalar fields (map to (src, alias) pairs)
                        try:
                            scalars_on_n_i = [sf for sf, sd in (self.registry.types.get(n_target_i).__berry_fields__.items() if n_target_i and self.registry.types.get(n_target_i) else []) if sd.kind == 'scalar']
                        except Exception:
                            scalars_on_n_i = []
                        n_fields_i_names: list[str] = []
                        try:
                            for sf in (ncfg_i.get('fields') or []):
                                if not scalars_on_n_i or sf in scalars_on_n_i:
                                    n_fields_i_names.append(sf)
                        except Exception:
                            n_fields_i_names = []
                        # map to pairs
                        try:
                            n_fields_i: list[tuple[str, str]] = self._mssql_map_columns_pairs(n_model_i, n_fields_i_names) if n_fields_i_names else []
                        except Exception:
                            n_fields_i = []
                        # order_dir default ASC if order_by explicit without dir
                        n_effective_dir_i = self._effective_order_dir(ncfg_i)
                        # Map order_by and order_multi to physical names
                        def _map_order_name(model_cls_local, name):
                            try:
                                pairs = self._mssql_map_columns_pairs(model_cls_local, [name] if name else [])
                                return pairs[0][0] if pairs else name
                            except Exception:
                                return name
                        n_order_by_mapped = _map_order_name(n_model_i, ncfg_i.get('order_by'))
                        n_order_multi_mapped: list[str] = []
                        try:
                            for spec in (self.registry._normalize_order_multi_values(ncfg_i.get('order_multi') or []) or []):
                                cn, _, dd = str(spec).partition(':')
                                mapped_cn = _map_order_name(n_model_i, cn)
                                n_order_multi_mapped.append(f"{mapped_cn}:{dd}" if dd else mapped_cn)
                        except Exception:
                            n_order_multi_mapped = []
                        # Recurse for deeper nests under this node
                        child_specs_i: list[dict] = []
                        try:
                            if ncfg_i.get('nested'):
                                child_specs_i = _mk_nested_specs(n_model_i, n_target_i, ncfg_i.get('nested') or {})
                        except Exception:
                            child_specs_i = []
                        spec_obj = {
                            'alias': nname_i,
                            'model': n_model_i,
                            'fields': n_fields_i or None,
                            'where': self._resolve_graphql_value(info, ncfg_i.get('where')),
                            'default_where': ncfg_i.get('default_where'),
                            'order_by': n_order_by_mapped,
                            'order_dir': n_effective_dir_i,
                            'order_multi': n_order_multi_mapped,
                            'limit': ncfg_i.get('limit'),
                            'offset': ncfg_i.get('offset'),
                            'nested': child_specs_i or None,
                        }
                        # Preserve resolved type-level default scope for deeper application where supported
                        if 'type_default_where' in ncfg_i:
                            spec_obj['type_default_where'] = _resolve_type_where_local(ncfg_i.get('type_default_where'), n_model_i)
                        if is_single_i:
                            spec_obj['mode'] = 'single'
                            spec_obj['child_fk_name'] = child_fk_to_nested_i
                        else:
                            spec_obj['mode'] = 'list'
                            spec_obj['fk_col_name'] = g_fk_col_name_i
                        specs.append(spec_obj)
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
                    # Determine correlation for nested
                    is_single_nested = bool(ncfg.get('single'))
                    g_fk_col_name: str | None = None
                    child_fk_to_nested: str | None = None
                    if is_single_nested:
                        # Need FK on child -> nested model
                        try:
                            rel_name_local2 = nname
                            pfk_name2 = self.registry._find_parent_fk_column_name(child_model_cls, n_model, rel_name_local2)
                            child_fk_to_nested = pfk_name2 or f"{rel_name_local2}_id"
                            if hasattr(child_model_cls, '__table__') and not any(c.name == child_fk_to_nested for c in child_model_cls.__table__.columns):
                                child_fk_to_nested = None
                        except Exception:
                            child_fk_to_nested = None
                        if not child_fk_to_nested:
                            continue
                    else:
                        # Find FK from grandchild (n_model) to child (child_model_cls), honoring explicit fk_column_name
                        try:
                            explicit_child_fk = ncfg.get('fk_column_name')
                        except Exception:
                            explicit_child_fk = None
                        try:
                            if explicit_child_fk and hasattr(n_model, '__table__'):
                                for col in n_model.__table__.columns:
                                    if col.name == explicit_child_fk:
                                        g_fk_col_name = col.name
                                        break
                        except Exception:
                            g_fk_col_name = None
                        if g_fk_col_name is None:
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
                    # Determine fields for nested object: only scalar fields (map to pairs)
                    try:
                        scalars_on_n = [sf for sf, sd in (self.registry.types.get(n_target).__berry_fields__.items() if n_target and self.registry.types.get(n_target) else []) if sd.kind == 'scalar']
                    except Exception:
                        scalars_on_n = []
                    n_fields_names: list[str] = []
                    try:
                        for sf in (ncfg.get('fields') or []):
                            if not scalars_on_n or sf in scalars_on_n:
                                n_fields_names.append(sf)
                    except Exception:
                        n_fields_names = []
                    try:
                        n_fields = self._mssql_map_columns_pairs(n_model, n_fields_names) if n_fields_names else []
                    except Exception:
                        n_fields = []
                    # Determine effective order_dir for nested: ASC when order_by explicit and no explicit dir
                    n_effective_dir = ncfg.get('order_dir')
                    try:
                        if ncfg.get('_has_explicit_order_by') and not ncfg.get('_has_explicit_order_dir'):
                            n_effective_dir = 'asc'
                    except Exception:
                        pass
                    # Map order_by/order_multi for this nested level
                    def _map_order_name2(model_cls_local, name):
                        try:
                            pairs = self._mssql_map_columns_pairs(model_cls_local, [name] if name else [])
                            return pairs[0][0] if pairs else name
                        except Exception:
                            return name
                    n_order_by_mapped2 = _map_order_name2(n_model, ncfg.get('order_by'))
                    n_order_multi_mapped2: list[str] = []
                    try:
                        for spec in (self.registry._normalize_order_multi_values(ncfg.get('order_multi') or []) or []):
                            cn, _, dd = str(spec).partition(':')
                            mapped_cn = _map_order_name2(n_model, cn)
                            n_order_multi_mapped2.append(f"{mapped_cn}:{dd}" if dd else mapped_cn)
                    except Exception:
                        n_order_multi_mapped2 = []
                    # Build deeper nested children under this nested node
                    nested_children_specs = _mk_nested_specs(n_model, n_target, ncfg.get('nested') or {}) if (ncfg.get('nested') or {}) else []
                    spec_obj2 = {
                        'alias': nname,
                        'model': n_model,
                        'fields': n_fields or None,
                        'where': self._resolve_graphql_value(info, ncfg.get('where')),
                        'default_where': ncfg.get('default_where'),
                        'order_by': n_order_by_mapped2,
                        'order_dir': n_effective_dir,
                        'order_multi': n_order_multi_mapped2,
                        'limit': ncfg.get('limit'),
                        'offset': ncfg.get('offset'),
                        'nested': nested_children_specs or None,
                    }
                    if 'type_default_where' in ncfg:
                        spec_obj2['type_default_where'] = _resolve_type_where_local(ncfg.get('type_default_where'), n_model)
                    if is_single_nested:
                        spec_obj2['mode'] = 'single'
                        spec_obj2['child_fk_name'] = child_fk_to_nested
                    else:
                        spec_obj2['mode'] = 'list'
                        spec_obj2['fk_col_name'] = g_fk_col_name
                    nested_specs.append(spec_obj2)
                # Build full JSON for the list relation, including nested arrays
                # Compute effective top-level order_dir: ASC when order_by explicit without explicit dir
                effective_order_dir = self._effective_order_dir(rel_cfg)
                # Resolve top-level type_default_where (callables/variables) before handing to adapter
                def _resolve_type_where(raw_val, model_cls_local):
                    v = self._resolve_graphql_value(info, raw_val)
                    try:
                        if callable(v):
                            v = v(model_cls_local, info)
                    except Exception:
                        pass
                    return v
                top_type_where = _resolve_type_where(rel_cfg.get('type_default_where'), child_model_cls)
                # Resolve order_by which may be a callable or a SQLAlchemy expression
                order_by_param = None
                from sqlalchemy.sql.elements import ClauseElement  # type: ignore
                try:
                    ob_raw = rel_cfg.get('order_by')
                except Exception:
                    ob_raw = None
                ob_resolved = self._resolve_graphql_value(info, ob_raw)
                if callable(ob_resolved):
                    ob_resolved = ob_resolved(child_model_cls, info)
                # For MSSQL, ensure scalar subqueries return a single row: apply LIMIT 1 when possible
                try:
                    from sqlalchemy.sql.selectable import ScalarSelect, Select  # type: ignore
                    if isinstance(ob_resolved, ScalarSelect):
                        try:
                            ob_resolved = ob_resolved.element.limit(1).scalar_subquery()
                        except Exception:
                            pass
                    elif isinstance(ob_resolved, Select):
                        try:
                            ob_resolved = ob_resolved.limit(1).scalar_subquery()
                        except Exception:
                            pass
                except Exception:
                    pass
                if ob_resolved is not None:
                    # If it's a SQLAlchemy expression, compile it to MSSQL SQL; else map string to physical column
                    try:
                        if isinstance(ob_resolved, ClauseElement):
                            try:
                                from sqlalchemy.dialects import mssql as _sa_mssql  # type: ignore
                                compiled = str(ob_resolved.compile(dialect=_sa_mssql.dialect(), compile_kwargs={"literal_binds": True}))
                            except Exception:
                                # Fallback to generic compile
                                compiled = str(ob_resolved)
                            order_by_param = f"({compiled})"
                        elif isinstance(ob_resolved, str):
                            mapped_pairs = self._mssql_map_columns_pairs(child_model_cls, [ob_resolved]) or [(ob_resolved, ob_resolved)]
                            order_by_param = mapped_pairs[0][0]
                    except Exception:
                        order_by_param = None
                return adapter.build_relation_list_json_full(
                    parent_table=parent_table,
                    parent_pk_name=parent_pk_name,
                    child_model=child_model_cls,
                    fk_col_name=fk_child_to_parent_col.name,
                    projected_columns=self._mssql_map_columns_pairs(child_model_cls, requested_scalar_local) or [(self._pk_name(child_model_cls), self._pk_name(child_model_cls))],
                    rel_where=self._resolve_graphql_value(info, rel_cfg.get('where')),
                    rel_default_where=rel_cfg.get('default_where'),
                    type_default_where=top_type_where,
                    limit=rel_cfg.get('limit'),
                    offset=rel_cfg.get('offset'),
                    order_by=order_by_param,
                    order_dir=effective_order_dir,
                    order_multi=[
                        f"{(self._mssql_map_columns_pairs(child_model_cls, [spec.split(':',1)[0]]) or [(spec.split(':',1)[0], spec.split(':',1)[0])])[0][0]}:{spec.split(':',1)[1]}"
                        if ':' in spec else (self._mssql_map_columns_pairs(child_model_cls, [spec]) or [(spec, spec)])[0][0]
                        for spec in (self.registry._normalize_order_multi_values(rel_cfg.get('order_multi') or []) or [])
                    ],
                    nested=nested_specs,
                )
            # No nested relations: use simple JSON list builder with optional filters/order
            # correlate and compose relation-level where
            parent_pk_name = self._pk_name(parent_model_cls)
            try:
                ct_ident2 = adapter.table_ident(child_model_cls)
                pt_ident2 = adapter.table_ident(parent_table)
            except Exception:
                ct_ident2 = f"[{getattr(child_model_cls,'__tablename__','')}]"
                pt_ident2 = f"[{getattr(parent_model_cls,'__tablename__','')}]"
            where_parts_rel: list[str] = [f"{ct_ident2}.[{fk_child_to_parent_col.name}] = {pt_ident2}.[{parent_pk_name}]"]
            where_parts_rel = self._mssql_where_from_value(
                where_parts_rel,
                child_model_cls,
                rel_cfg.get('where'),
                strict=True,
                to_where_dict=to_where_dict,
                expr_from_where_dict=expr_from_where_dict,
                adapter=adapter,
                info=info,
            )
            where_parts_rel = self._mssql_where_from_value(
                where_parts_rel,
                child_model_cls,
                rel_cfg.get('default_where'),
                strict=True,
                to_where_dict=to_where_dict,
                expr_from_where_dict=expr_from_where_dict,
                adapter=adapter,
                info=info,
            )
            # Apply type-level default where for MSSQL list path
            try:
                t_where2 = rel_cfg.get('type_default_where')
            except Exception:
                t_where2 = None
            if t_where2 is not None:
                where_parts_rel = self._mssql_where_from_value(
                    where_parts_rel,
                    child_model_cls,
                    t_where2,
                    strict=True,
                    to_where_dict=to_where_dict,
                    expr_from_where_dict=expr_from_where_dict,
                    adapter=adapter,
                    info=info,
                )
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
                        child_pk_name2 = self._pk_name(child_model_cls)
                        where_parts_rel.append(
                            f"EXISTS (SELECT 1 FROM {gc_table} WHERE [{gc_table}].[{gc_fk_col}] = [{child_table}].[{child_pk_name2}])"
                        )
            except Exception:
                pass
            # filter args best-effort (MSSQL string path)
            try:
                fa = rel_cfg.get('filter_args') or {}
                expanded = self._expand_arg_specs(rel_cfg.get('arg_specs') or {})
                where_parts_rel = self._apply_filter_args_mssql(where_parts_rel, child_model_cls, fa, expanded, adapter)
            except Exception:
                pass
            where_clause = ' AND '.join(where_parts_rel)
            # Respect explicit order_by without order_dir -> default ASC (override relation default)
            effective_order_dir2 = self._effective_order_dir(rel_cfg)
            # Resolve order_by: allow callable/SQLAlchemy expression for MSSQL simple path as well
            order_clause = None
            try:
                from sqlalchemy.sql.elements import ClauseElement  # type: ignore
            except Exception:
                ClauseElement = tuple()  # type: ignore
            try:
                ob_raw2 = rel_cfg.get('order_by')
            except Exception:
                ob_raw2 = None
            ob_resolved2 = self._resolve_graphql_value(info, ob_raw2)
            try:
                if callable(ob_resolved2):
                    ob_resolved2 = ob_resolved2(child_model_cls, info)
            except Exception:
                pass
            # For MSSQL, ensure scalar subqueries return a single row: apply LIMIT 1 when possible
            try:
                from sqlalchemy.sql.selectable import ScalarSelect, Select  # type: ignore
                if isinstance(ob_resolved2, ScalarSelect):
                    try:
                        ob_resolved2 = ob_resolved2.element.limit(1).scalar_subquery()
                    except Exception:
                        pass
                elif isinstance(ob_resolved2, Select):
                    try:
                        ob_resolved2 = ob_resolved2.limit(1).scalar_subquery()
                    except Exception:
                        pass
            except Exception:
                pass
            if ob_resolved2 is not None and isinstance(ob_resolved2, ClauseElement):
                # Compile the SQLAlchemy expression to MSSQL SQL and combine with order_multi if present
                try:
                    from sqlalchemy.dialects import mssql as _sa_mssql  # type: ignore
                    compiled2 = str(ob_resolved2.compile(dialect=_sa_mssql.dialect(), compile_kwargs={"literal_binds": True}))
                except Exception:
                    compiled2 = str(ob_resolved2)
                expr_part = f"({compiled2}) {'DESC' if (str(effective_order_dir2).lower()=='desc') else 'ASC'}"
                # Build order_multi parts manually to avoid unintended PK fallback
                multi_specs = self.registry._normalize_order_multi_values(rel_cfg.get('order_multi') or []) or []
                multi_parts: list[str] = []
                try:
                    alias_ident3 = adapter.table_ident(child_model_cls)
                except Exception:
                    alias_ident3 = f"[{getattr(child_model_cls,'__tablename__','')}]"
                try:
                    cols = getattr(child_model_cls, '__table__').columns  # type: ignore
                except Exception:
                    cols = {}
                for spec in multi_specs:
                    try:
                        cn, _, dd = str(spec).partition(':')
                        if cn and cn in cols:
                            multi_parts.append(f"{alias_ident3}.[{cn}] {'DESC' if (dd or '').lower()=='desc' else 'ASC'}")
                    except Exception:
                        continue
                order_clause = expr_part if not multi_parts else f"{expr_part}, {', '.join(multi_parts)}"
            else:
                # Fall back to adapter's standard column-based order builder
                order_clause = adapter.build_order_clause(
                    child_model_cls,
                    child_model_cls,
                    ob_resolved2 if isinstance(ob_resolved2, str) else rel_cfg.get('order_by'),
                    effective_order_dir2,
                    self.registry._normalize_order_multi_values(rel_cfg.get('order_multi') or []),
                )
            return adapter.build_list_relation_json(
                child_table=child_model_cls,
                projected_columns=self._mssql_map_columns_pairs(child_model_cls, requested_scalar_local) or [(self._pk_name(child_model_cls), self._pk_name(child_model_cls))],
                where_condition=where_clause,
                limit=rel_cfg.get('limit'),
                offset=rel_cfg.get('offset'),
                order_by=order_clause,
                nested_subqueries=None,
            )
        # Non-MSSQL: correlated aggregation
        inner_sel = self._build_list_child_select_sqla(
            parent_model_cls=parent_model_cls,
            child_model_cls=child_model_cls,
            fk_child_to_parent_col=fk_child_to_parent_col,
            projected_columns=(requested_scalar_local or []),
            rel_cfg=rel_cfg,
            to_where_dict=to_where_dict,
            expr_from_where_dict=expr_from_where_dict,
            info=info,
        )
        limited_subq = inner_sel.subquery()
        row_json_expr = json_object_fn(*self._json_row_args_from_subq(limited_subq, (requested_scalar_local or [self._pk_name(child_model_cls)])))
        agg_inner_expr = json_array_agg_fn(row_json_expr)
        if agg_inner_expr is None:
            return None
        agg_query = select(json_array_coalesce_fn(agg_inner_expr)).select_from(limited_subq).correlate(parent_model_cls)
        try:
            return agg_query.scalar_subquery()
        except Exception:
            return agg_query

    # --- custom field pushdown ------------------------------------------------
    def build_custom_scalar_pushdown(
        self,
        *,
        model_cls,
        field_name: str,
        builder: Any,
    ) -> Any | None:
            """Attempt to build a pushdown SQL expression for a @custom field.

            - Calls builder(model_cls) only; skips if it requires session/info to avoid N+1.
            - If the builder returns a selectable, converts to scalar_subquery when single column.
            - Returns a labeled SQL expression or None if it cannot safely be pushed down.
            """
            if builder is None:
                return None
            import inspect
            try:
                # Only support builder(model_cls) to avoid needing a session in pushdown phase
                if len(inspect.signature(builder).parameters) == 1:
                    expr = builder(model_cls)
                else:
                    return None
            except Exception:
                return None
            if expr is None:
                return None
            try:
                from sqlalchemy.sql import Select as _Select  # type: ignore
            except Exception:
                _Select = None  # type: ignore
            try:
                if _Select is not None and isinstance(expr, _Select):
                    # Use scalar_subquery when single column select
                    try:
                        if len(expr.selected_columns) == 1:  # type: ignore[attr-defined]
                            expr = expr.scalar_subquery()
                    except Exception:
                        pass
                # Ensure label for mapping access
                if hasattr(expr, 'label'):
                    expr = expr.label(field_name)
            except Exception:
                return None
            return expr

    def build_custom_object_pushdown(
        self,
        *,
        model_cls,
        field_name: str,
        builder: Any,
        adapter,
        json_object_fn,
        info,
    ) -> Tuple[List[Any], List[str]] | None:
            """Build pushdown columns for a @custom_object field.

            Returns a tuple: (select_columns_to_add, label_names). On non-MSSQL, it returns a
            single JSON object scalar labeled as one column; on MSSQL, it returns multiple labeled
            scalar columns to reconstruct the object in Python.
            """
            if builder is None:
                return None
            import inspect
            try:
                if len(inspect.signature(builder).parameters) == 1:
                    expr_sel = builder(model_cls)
                else:
                    return None  # skip builders that need session/info
            except Exception:
                return None
            if expr_sel is None:
                return None
            try:
                from sqlalchemy.sql import Select as _Select  # type: ignore
            except Exception:
                _Select = None  # type: ignore
            if _Select is None or not isinstance(expr_sel, _Select):
                return None
            # Gather selected columns and build scalar subqueries per column
            try:
                sel_cols = list(getattr(expr_sel, 'selected_columns', []))  # type: ignore[attr-defined]
            except Exception:
                sel_cols = []
            key_exprs: list[tuple[str, Any]] = []
            agg_pairs: list[tuple[str, Any]] = []  # (key, aggregate/labeled expr)
            for col in sel_cols:
                try:
                    labeled = col
                    col_name = getattr(labeled, 'name', None) or getattr(labeled, 'key', None)
                    if not col_name:
                        col_name = f"{field_name}_{len(key_exprs)}"
                        labeled = col.label(col_name)
                    agg_pairs.append((col_name, labeled))
                    subq = select(labeled)
                    try:
                        for _from in expr_sel.get_final_froms():  # type: ignore[attr-defined]
                            subq = subq.select_from(_from)
                    except Exception:
                        pass
                    for _w in getattr(expr_sel, '_where_criteria', []):  # type: ignore[attr-defined]
                        subq = subq.where(_w)
                    subq_expr = subq.scalar_subquery() if hasattr(subq, 'scalar_subquery') else subq
                    key_exprs.append((col_name, subq_expr))
                except Exception:
                    continue
            if not key_exprs:
                return None
            is_mssql = getattr(adapter, 'name', '') == 'mssql'
            select_columns: list[Any] = []
            labels: list[str] = []
            if is_mssql:
                # Expose each key as a labeled scalar column; registry will reconstruct Python object
                for k, v in key_exprs:
                    lbl = f"_pushcf_{field_name}__{k}"
                    try:
                        select_columns.append(v.label(lbl))
                    except Exception:
                        select_columns.append(literal_column(str(v)).label(lbl))
                    labels.append(lbl)
                return (select_columns, labels)
            # Non-MSSQL: build a single JSON object using aggregated expressions
            json_args: list[Any] = []
            for k, agg_expr in agg_pairs:
                json_args.extend([_text(f"'{k}'"), agg_expr])
            inner = select(json_object_fn(*json_args))
            try:
                for _from in expr_sel.get_final_froms():  # type: ignore[attr-defined]
                    inner = inner.select_from(_from)
            except Exception:
                pass
            for _w in getattr(expr_sel, '_where_criteria', []):  # type: ignore[attr-defined]
                inner = inner.where(_w)
            try:
                json_obj_expr = inner.scalar_subquery()
            except Exception:
                json_obj_expr = inner
            json_label = f"_pushcf_{field_name}"
            select_columns.append(json_obj_expr.label(json_label))
            labels.append(json_label)
            return (select_columns, labels)


# Root-level helpers to keep registry resolver thin
class RootSQLBuilders:
    def __init__(self, registry):
        self.registry = registry

    # --- helpers -------------------------------------------------------------
    def _pk_col(self, model_cls):
        """Get SQLAlchemy primary key column using registry helper (root scope)."""
        return self.registry._get_pk_column(model_cls)

    def _pk_name(self, model_cls) -> str:
        """Get primary key column name using registry helper (root scope)."""
        return self.registry._get_pk_name(model_cls)

    def build_count_aggregates(self, *, model_cls, btype_cls, requested_aggregates: set[str]) -> tuple[list[Any], list[tuple[str, Any]]]:
        cols: list[Any] = []
        meta: list[tuple[str, Any]] = []
        try:
            for cf_name, cf_def in getattr(btype_cls, '__berry_fields__', {}).items():
                try:
                    if cf_def.kind != 'aggregate':
                        continue
                    op = cf_def.meta.get('op')
                    ops = cf_def.meta.get('ops') or []
                    is_count = op == 'count' or 'count' in ops
                    if not is_count:
                        continue
                    if cf_name not in requested_aggregates:
                        continue
                    source_rel = cf_def.meta.get('source')
                    rel_def = getattr(btype_cls, '__berry_fields__', {}).get(source_rel)
                    if not (rel_def and rel_def.kind == 'relation'):
                        continue
                    target_name = rel_def.meta.get('target')
                    target_b = self.registry.types.get(target_name)
                    if not (target_b and target_b.model):
                        continue
                    child_model_cls = target_b.model
                    fk_col = None
                    for col in child_model_cls.__table__.columns:
                        for fk in col.foreign_keys:
                            if fk.column.table.name == model_cls.__table__.name:
                                fk_col = col
                                break
                        if fk_col is not None:
                            break
                    if fk_col is None:
                        continue
                    subq_cnt = select(func.count('*')).select_from(child_model_cls).where(fk_col == self._pk_col(model_cls)).scalar_subquery().label(cf_name)
                    cols.append(subq_cnt)
                    meta.append((cf_name, cf_def))
                except Exception:
                    continue
        except Exception:
            pass
        return cols, meta

    def build_base_root_columns(self, *, model_cls, requested_scalar_root: set[str], requested_relations: Dict[str, Any], required_fk_parent_cols: set[str]) -> list[Any]:
        base_root_cols: list[Any] = []
        effective_root_cols: set[str] = set(requested_scalar_root or set())
        # Ensure helper FK columns required for relation correlation are present
        for fk in (required_fk_parent_cols or set()):
            effective_root_cols.add(fk)
        # Always include PK when any relations are requested so hydrator can resolve relations and ids
        try:
            if requested_relations:
                pk_name = self._pk_name(model_cls)
                if pk_name:
                    effective_root_cols.add(pk_name)
        except Exception:
            pass
        # Access Berry field defs for column mapping
        btype = None
        try:
            for _name, _bt in (self.registry.types or {}).items():
                if getattr(_bt, 'model', None) is model_cls:
                    btype = _bt
                    break
        except Exception:
            btype = None
        fdefs = getattr(btype, '__berry_fields__', {}) if btype is not None else {}
        # Helper: decamelize GraphQL field names to snake_case
        import re as _re
        def _decamel(name: str) -> str:
            try:
                if not name or ('_' in name):
                    return name
                s1 = _re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                s2 = _re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
                return s2.lower()
            except Exception:
                return name
        for sf in effective_root_cols:
            try:
                # Prefer exact model attribute match
                if hasattr(model_cls, sf):
                    col_obj2 = getattr(model_cls, sf)
                    try:
                        base_root_cols.append(col_obj2.label(sf))
                    except Exception:
                        base_root_cols.append(col_obj2)
                    continue
                # Try decamelized name against model attributes/columns
                sfd = _decamel(str(sf)) if isinstance(sf, str) else sf
                if sfd and hasattr(model_cls, sfd):
                    col_obj3 = getattr(model_cls, sfd)
                    # Label with snake_case to keep hydration/enum coercion consistent
                    try:
                        base_root_cols.append(col_obj3.label(sfd))
                    except Exception:
                        base_root_cols.append(col_obj3)
                    continue
                # Try Berry field meta for either key (original or decamelized)
                src = None
                try:
                    fdef = fdefs.get(sf) or (fdefs.get(sfd) if isinstance(sfd, str) else None)
                    src = (getattr(fdef, 'meta', {}) or {}).get('column') if fdef else None
                except Exception:
                    src = None
                if src:
                    sa_col = None
                    try:
                        sa_col = getattr(model_cls.__table__.c, src, None) or model_cls.__table__.c.get(src)
                    except Exception:
                        sa_col = getattr(model_cls, src, None)
                    if sa_col is not None:
                        # Label with the Berry field key when available, prefer snake_case
                        lbl = None
                        try:
                            lbl = (fdef and fdef.name) or (sfd if isinstance(sfd, str) else sf)
                        except Exception:
                            lbl = (sfd if isinstance(sfd, str) else sf)
                        try:
                            base_root_cols.append(sa_col.label(lbl))
                        except Exception:
                            base_root_cols.append(sa_col)
            except Exception:
                pass
        return base_root_cols

    async def apply_root_filters(self, stmt, *, model_cls, btype_cls, info, raw_where, declared_filters: Dict[str, Any], passed_filter_args: Dict[str, Any]):
        from ..core.utils import to_where_dict as _to_where_dict, expr_from_where_dict as _expr_from_where_dict
        import inspect
        where_clauses = []
        # raw where
        if raw_where is not None:
            wdict = raw_where(model_cls, info) if callable(raw_where) else raw_where
            if inspect.isawaitable(wdict):
                wdict = await wdict
            # Apply via common path to preserve identical behavior
            try:
                tmp = select(model_cls)
                tmp = RelationSQLBuilders(self.registry)._apply_where_common(
                    tmp, model_cls, wdict, strict=True,
                    to_where_dict=_to_where_dict,
                    expr_from_where_dict=_expr_from_where_dict,
                    info=info,
                )
                # If string was provided but not JSON object, _apply_where_common would attempt parse and may raise.
                for _w in getattr(tmp, '_where_criteria', []):  # type: ignore[attr-defined]
                    where_clauses.append(_w)
            except Exception as e:
                # Match previous error messages for string inputs
                if isinstance(wdict, str):
                    msg = str(e)
                    if 'Unknown where' in msg or 'operator' in msg or 'column' in msg or 'Invalid' in msg:
                        raise
                    raise ValueError(f"Invalid where JSON: {e}")
                raise
        # type-level scope (BerryType.scope or __type_scope__) must always be enforced at root
        try:
            t_scope = getattr(btype_cls, '__type_scope__', None)
            if t_scope is None:
                t_scope = getattr(btype_cls, 'scope', None)
        except Exception:
            t_scope = None
        if t_scope is not None:
            fragments = t_scope if isinstance(t_scope, (list, tuple)) else [t_scope]
            for frag in fragments:
                # Use common applier to support dict/str/callable/expr uniformly; strict=True to surface issues
                tmp2 = select(model_cls)
                tmp2 = RelationSQLBuilders(self.registry)._apply_where_common(
                    tmp2,
                    model_cls,
                    frag,
                    strict=True,
                    to_where_dict=_to_where_dict,
                    expr_from_where_dict=_expr_from_where_dict,
                    info=info,
                )
                for _w in getattr(tmp2, '_where_criteria', []):  # type: ignore[attr-defined]
                    where_clauses.append(_w)
        # filter args
        from ..core.filters import OPERATOR_REGISTRY
        for arg_name, value in (passed_filter_args or {}).items():
            if value is None:
                continue
            f_spec = declared_filters.get(arg_name)
            if not f_spec:
                raise ValueError(f"Unknown filter argument: {arg_name}")
            if getattr(f_spec, 'transform', None):
                try:
                    value = f_spec.transform(value)
                except Exception as e:
                    raise ValueError(f"Filter transform failed for {arg_name}: {e}")
            expr = None
            if getattr(f_spec, 'builder', None):
                try:
                    expr = f_spec.builder(model_cls, info, value)
                    if inspect.isawaitable(expr):
                        expr = await expr
                except Exception as e:
                    raise ValueError(f"Filter builder failed for {arg_name}: {e}")
            elif getattr(f_spec, 'column', None):
                try:
                    col = model_cls.__table__.c.get(f_spec.column)
                except Exception:
                    col = None
                if col is None:
                    raise ValueError(f"Unknown filter column: {f_spec.column} for argument {arg_name}")
                op_name = f_spec.op or 'eq'
                op_fn = OPERATOR_REGISTRY.get(op_name)
                if not op_fn:
                    raise ValueError(f"Unknown filter operator: {op_name} for argument {arg_name}")
                try:
                    expr = op_fn(col, value)
                except Exception as e:
                    raise ValueError(f"Filter operation failed for {arg_name}: {e}")
            if expr is not None:
                where_clauses.append(expr)
        for wc in where_clauses:
            stmt = stmt.where(wc)
        return stmt

    def apply_ordering(self, stmt, *, model_cls, btype_cls, order_by, order_dir, order_multi):
        from ..core.utils import dir_value as _dir_value
        # multi first
        if order_multi:
            allowed_order_fields = getattr(btype_cls, '__ordering__', None)
            if allowed_order_fields is None:
                allowed_order_fields = [fname for fname, fdef in btype_cls.__berry_fields__.items() if fdef.kind == 'scalar']
            for spec in (order_multi or []):
                try:
                    cn, _, dd = str(spec).partition(':')
                    dd = (dd or 'asc').lower()
                    if cn not in allowed_order_fields:
                        continue
                    col = model_cls.__table__.c.get(cn)
                    if col is None:
                        continue
                    stmt = stmt.order_by(col.desc() if dd=='desc' else col.asc())
                except Exception:
                    pass
            return stmt
        # single order_by
        if order_by:
            allowed_order_fields = getattr(btype_cls, '__ordering__', None)
            if allowed_order_fields is None:
                allowed_order_fields = [fname for fname, fdef in btype_cls.__berry_fields__.items() if fdef.kind == 'scalar']
            if order_by not in allowed_order_fields:
                raise ValueError(f"Invalid order_by '{order_by}'. Allowed: {allowed_order_fields}")
            try:
                col = model_cls.__table__.c.get(order_by)
            except Exception:
                col = None
            if col is None:
                raise ValueError(f"Unknown order_by column: {order_by}")
            dv = _dir_value(order_dir)
            if dv not in ('asc','desc'):
                raise ValueError(f"Invalid order_dir '{order_dir}'. Use asc or desc")
            try:
                stmt = stmt.order_by(col.desc() if dv=='desc' else col.asc())
            except Exception:
                pass
            return stmt
        # defaults
        allowed_order_fields = getattr(btype_cls, '__ordering__', None)
        if allowed_order_fields is None:
            allowed_order_fields = [fname for fname, fdef in btype_cls.__berry_fields__.items() if fdef.kind == 'scalar']
        def_dir = _dir_value(getattr(btype_cls, '__default_order_dir__', None))
        default_multi = getattr(btype_cls, '__default_order_multi__', None) or []
        default_by = getattr(btype_cls, '__default_order_by__', None)
        try:
            if default_multi:
                for spec in default_multi:
                    cn, _, dd = str(spec).partition(':')
                    dd = dd or def_dir
                    if cn in allowed_order_fields:
                        col = model_cls.__table__.c.get(cn)
                        if col is not None:
                            stmt = stmt.order_by(col.desc() if (dd=='desc') else col.asc())
            elif default_by and default_by in allowed_order_fields:
                col = model_cls.__table__.c.get(default_by)
                if col is not None:
                    stmt = stmt.order_by(col.desc() if def_dir=='desc' else col.asc())
        except Exception:
            pass
        return stmt

    def apply_pagination(self, stmt, *, limit, offset):
        if offset is not None:
            try:
                o = int(offset)
            except Exception:
                raise ValueError("offset must be an integer")
            if o < 0:
                raise ValueError("offset must be non-negative")
            if o:
                stmt = stmt.offset(o)
        if limit is not None:
            try:
                l = int(limit)
            except Exception:
                raise ValueError("limit must be an integer")
            if l < 0:
                raise ValueError("limit must be non-negative")
            stmt = stmt.limit(l)
        return stmt
