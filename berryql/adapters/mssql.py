from __future__ import annotations
from typing import Any
from sqlalchemy import text as _text
from .base import BaseAdapter

class MSSQLAdapter(BaseAdapter):
    name = 'mssql'
    # Render a schema-qualified, MSSQL-quoted table identifier like [schema].[table]
    def table_ident(self, model_or_name) -> str:
        try:
            # Accept model class or raw table name
            tbl = getattr(model_or_name, '__table__', None)
            if tbl is not None:
                schema = getattr(tbl, 'schema', None)
                name = getattr(tbl, 'name', None) or getattr(model_or_name, '__tablename__', None)
                if schema:
                    return f"[{schema}].[{name}]"
                return f"[{name}]"
        except Exception:
            pass
        try:
            # If passed a plain string like 'dbo.Users' or 'Users'
            raw = str(model_or_name)
            if '.' in raw:
                s, n = raw.split('.', 1)
                return f"[{s}].[{n}]"
            return f"[{raw}]"
        except Exception:
            return ''
    def json_object(self, *args):
        # Not used directly; registry composes JSON via FOR JSON PATH.
        return None
    def json_array_agg(self, expr):
        # Force registry to use custom MSSQL aggregation path.
        return None
    def json_array_coalesce(self, expr):
        return None
    def supports_relation_pushdown(self) -> bool:
        return True

    # --- MSSQL specific helpers -------------------------------------------------
    def _as_int(self, v) -> int | None:
        if v is None:
            return None
        try:
            if hasattr(v, 'value'):
                return int(getattr(v, 'value'))
            if isinstance(v, (int, float)):
                return int(v)
            s = str(v)
            return int(s)
        except Exception:
            try:
                # Last resort: stringified .value of possible AST node
                return int(str(getattr(v, 'value', v)))
            except Exception:
                raise

    def _as_str(self, v) -> str | None:
        if v is None:
            return None
        try:
            if hasattr(v, 'value'):
                return str(getattr(v, 'value'))
            return str(v)
        except Exception:
            return None

    def _as_list_str(self, v) -> list[str]:
        # Normalize potential GraphQL AST ListValueNode or scalar into list[str]
        if v is None:
            return []
        try:
            if isinstance(v, list):
                return [self._as_str(x) or str(x) for x in v]
            if hasattr(v, 'values'):
                vals = getattr(v, 'values', []) or []
                out = []
                for x in vals:
                    xv = getattr(x, 'value', x)
                    out.append(str(xv))
                return out
            # single spec
            return [self._as_str(v) or str(v)]
        except Exception:
            try:
                return [str(v)]
            except Exception:
                return []

    def _render_literal(self, col, v) -> str:
        """Render a Python value as an MSSQL SQL literal with type awareness."""
        try:
            v2 = v
            try:
                # Try to coerce via column's python type when available
                from sqlalchemy.sql.sqltypes import Integer as _I, Float as _F, Boolean as _B, DateTime as _DT, Numeric as _N
                ctype = getattr(col, 'type', None)
                if isinstance(ctype, _B) and isinstance(v2, str):
                    lv = v2.strip().lower()
                    v2 = True if lv in ('true','t','1','yes','y') else False if lv in ('false','f','0','no','n') else v2
                if isinstance(ctype, _I) and isinstance(v2, str):
                    v2 = int(v2)
                if (_N is not None and isinstance(ctype, _N) or (_F is not None and isinstance(ctype, _F))) and isinstance(v2, str):
                    v2 = float(v2)
                if isinstance(ctype, _DT) and isinstance(v2, str):
                    from datetime import datetime as _dt
                    s = v2.replace('Z', '+00:00') if 'Z' in v2 else v2
                    try:
                        v2 = _dt.fromisoformat(s)
                    except Exception:
                        pass
                # fall through
            except Exception:
                pass
            # Now render
            from sqlalchemy.sql.sqltypes import Boolean as _B, DateTime as _DT
            ctype = getattr(col, 'type', None)
            if isinstance(v2, (int, float)):
                return str(v2)
            if isinstance(ctype, _B) or isinstance(v2, bool):
                return '1' if bool(v2) else '0'
            if isinstance(ctype, _DT):
                try:
                    from datetime import datetime as _dt
                    if isinstance(v2, _dt):
                        iso = v2.replace(tzinfo=None).isoformat(sep='T', timespec='seconds')
                        return f"CONVERT(datetime2, '{iso}', 126)"
                except Exception:
                    pass
            s = str(v2).replace("'", "''")
            return f"'{s}'"
        except Exception:
            s = str(v).replace("'", "''")
            return f"'{s}'"

    def where_from_dict(self, model_cls, where_dict: dict | None) -> list[str]:
        """Translate a simple dict-based where into MSSQL condition strings."""
        parts: list[str] = []
        if not where_dict:
            return parts
        for col_name, op_map in (where_dict or {}).items():
            col = getattr(model_cls.__table__.c, col_name, None)
            if col is None:
                try:
                    col = model_cls.__table__.c.get(col_name)
                except Exception:
                    col = None
            if col is None:
                continue
            tbl_ident = self.table_ident(model_cls)
            for op_name, val in (op_map or {}).items():
                tgt = f"{tbl_ident}.[{col_name}]"
                if op_name == 'eq':
                    parts.append(f"{tgt} = {self._render_literal(col, val)}")
                elif op_name == 'ne':
                    parts.append(f"{tgt} <> {self._render_literal(col, val)}")
                elif op_name == 'lt':
                    parts.append(f"{tgt} < {self._render_literal(col, val)}")
                elif op_name == 'lte':
                    parts.append(f"{tgt} <= {self._render_literal(col, val)}")
                elif op_name == 'gt':
                    parts.append(f"{tgt} > {self._render_literal(col, val)}")
                elif op_name == 'gte':
                    parts.append(f"{tgt} >= {self._render_literal(col, val)}")
                elif op_name == 'like':
                    parts.append(f"{tgt} LIKE {self._render_literal(col, val)}")
                elif op_name == 'ilike':
                    parts.append(f"LOWER({tgt}) LIKE LOWER({self._render_literal(col, val)})")
                elif op_name == 'in' and isinstance(val, (list, tuple)):
                    vals = ', '.join([self._render_literal(col, v) for v in val])
                    parts.append(f"{tgt} IN ({vals})")
                elif op_name == 'between' and isinstance(val, (list, tuple)) and len(val) >= 2:
                    a = self._render_literal(col, val[0])
                    b = self._render_literal(col, val[1])
                    parts.append(f"{tgt} BETWEEN {a} AND {b}")
        return parts

    def build_order_clause(self, model_cls, table_alias: str, order_by: str | None, order_dir: str | None, order_multi: list[str] | None) -> str | None:
        """Public wrapper for MSSQL order clause generation (DRY)."""
        return self._build_order_clause(model_cls, table_alias, order_by, order_dir, order_multi)

    def _build_order_clause(self, model_cls, table_alias: str, order_by: str | None, order_dir: str | None, order_multi: list[str] | None) -> str | None:
        parts: list[str] = []
        multi = self._as_list_str(order_multi)
        ob_str = self._as_str(order_by)
        od_str = (self._as_str(order_dir) or 'asc').lower()
        # Resolve alias as identifier; allow raw alias (like subquery alias) or model
        try:
            alias_ident = self.table_ident(table_alias) if hasattr(table_alias, '__table__') else (f"[{table_alias}]" if table_alias and '[' not in str(table_alias) else str(table_alias))
        except Exception:
            alias_ident = f"[{table_alias}]"
        for spec in multi:
            try:
                cn, _, dd = str(spec).partition(':')
                dd = (dd or od_str).lower()
                if cn in model_cls.__table__.columns:
                    parts.append(f"{alias_ident}.[{cn}] {'DESC' if dd=='desc' else 'ASC'}")
            except Exception:
                continue
        if not parts and ob_str and ob_str in model_cls.__table__.columns:
            dd = od_str
            parts.append(f"{alias_ident}.[{ob_str}] {'DESC' if dd=='desc' else 'ASC'}")
        if not parts:
            try:
                pk_col = next(iter(model_cls.__table__.primary_key.columns)).name
            except Exception:
                pk_col = None
            if pk_col:
                parts.append(f"{alias_ident}.[{pk_col}] ASC")
        return ', '.join(parts) if parts else None

    def build_nested_list_sql(self, *, alias: str, grand_model, child_table: str, g_fk_col_name: str, child_pk_name: str, fields: list[str] | list[tuple[str, str]] | None, where_dict: dict | str | None, default_where: dict | str | None, order_by: str | None, order_dir: str | None, order_multi: list[str] | None, limit: int | None, offset: int | None, extra_where_sql: list[str] | None = None, nested_children: list[dict] | None = None) -> str:
        n_cols = list(fields or [])
        if not n_cols:
            for c in grand_model.__table__.columns:
                n_cols.append(c.name)
        select_bits: list[str] = []
        if n_cols and isinstance(n_cols[0], (list, tuple)):
            for src, alias2 in (n_cols or []):
                select_bits.append(f"{self.table_ident(grand_model)}.[{src}] AS [{alias2}]")
        else:
            for c in (n_cols or []):
                select_bits.append(f"{self.table_ident(grand_model)}.[{c}] AS [{c}]")
        n_col_select = ', '.join(select_bits) if select_bits else '*'
        parent_ident = self.table_ident(grand_model)
        child_ident = self.table_ident(child_table)
        where_parts = [f"{parent_ident}.[{g_fk_col_name}] = {child_ident}.[{child_pk_name}]"]
        # accept both dict and JSON string
        import json as _json
        def _as_dict(maybe):
            # Accept dict or JSON string only; ignore callables/expressions
            if isinstance(maybe, dict):
                return maybe
            if isinstance(maybe, str):
                try:
                    v = _json.loads(maybe)
                    return v if isinstance(v, dict) else None
                except Exception:
                    return None
            return None
        wdict = _as_dict(where_dict)
        if wdict:
            where_parts.extend(self.where_from_dict(grand_model, wdict))
        dwdict = _as_dict(default_where)
        if dwdict:
            where_parts.extend(self.where_from_dict(grand_model, dwdict))
        if extra_where_sql:
            where_parts.extend([str(x) for x in extra_where_sql if x])
        n_where = ' AND '.join(where_parts)
        # Handle nested-of-nested arrays and singles: build JSON subqueries correlated to grand_model -> current child
        nested_cols = ''
        if nested_children:
            n_parts: list[str] = []
            for nn in nested_children:
                try:
                    sub_alias = nn.get('alias')
                    sub_model = nn.get('model')
                    sub_mode = nn.get('mode') or 'list'
                    if sub_mode == 'single':
                        # Single nested object off each grand_model row: join grand_model.child_fk = sub_model.pk
                        child_fk_name = nn.get('child_fk_name')
                        if not child_fk_name:
                            continue
                        try:
                            sub_pk_name = next(iter(sub_model.__table__.primary_key.columns)).name
                        except Exception:
                            sub_pk_name = 'id'
                        # Build where (join + where/default_where)
                        where_parts_s: list[str] = [
                            f"{self.table_ident(grand_model)}.[{child_fk_name}] = {self.table_ident(sub_model)}.[{sub_pk_name}]"
                        ]
                        def _as_dict2(maybe):
                            import json as _json
                            if isinstance(maybe, dict):
                                return maybe
                            if isinstance(maybe, str):
                                try:
                                    v = _json.loads(maybe)
                                    return v if isinstance(v, dict) else None
                                except Exception:
                                    return None
                            return None
                        wdict_s = _as_dict2(nn.get('where'))
                        if wdict_s:
                            where_parts_s.extend(self.where_from_dict(sub_model, wdict_s))
                        dwdict_s = _as_dict2(nn.get('default_where'))
                        if dwdict_s:
                            where_parts_s.extend(self.where_from_dict(sub_model, dwdict_s))
                        join_cond = ' AND '.join(where_parts_s)
                        # Base columns for sub_model
                        cols_pairs = nn.get('fields') or [(c.name, c.name) for c in sub_model.__table__.columns]
                        sub_cols = ', '.join([f"{self.table_ident(sub_model)}.[{src}] AS [{alias}]" for src, alias in cols_pairs])
                        # Include nested under this single (one recursive level: singles and lists)
                        extra_cols = ''
                        sub_children = nn.get('nested') or []
                        if sub_children:
                            sub_parts: list[str] = []
                            for sn in sub_children:
                                s_alias = sn.get('alias')
                                s_model = sn.get('model')
                                s_mode = sn.get('mode') or 'list'
                                if s_mode == 'single':
                                    s_child_fk = sn.get('child_fk_name')
                                    if not s_child_fk:
                                        continue
                                    try:
                                        s_pk = next(iter(s_model.__table__.primary_key.columns)).name
                                    except Exception:
                                        s_pk = 'id'
                                    s_where = [f"{self.table_ident(sub_model)}.[{s_child_fk}] = {self.table_ident(s_model)}.[{s_pk}]"]
                                    wdict_ss = _as_dict2(sn.get('where'))
                                    if wdict_ss:
                                        s_where.extend(self.where_from_dict(s_model, wdict_ss))
                                    dwdict_ss = _as_dict2(sn.get('default_where'))
                                    if dwdict_ss:
                                        s_where.extend(self.where_from_dict(s_model, dwdict_ss))
                                    s_join = ' AND '.join(s_where)
                                    s_cols_pairs = sn.get('fields') or [(c.name, c.name) for c in s_model.__table__.columns]
                                    s_json = self.build_single_relation_json(
                                        child_table=s_model,
                                        projected_columns=s_cols_pairs,
                                        join_condition=s_join,
                                    )
                                    sub_parts.append(f"ISNULL(({str(s_json).strip()}), 'null') AS [{s_alias}]")
                                else:
                                    # list under the single: s_model.fk_to_sub_model = sub_model.pk
                                    s_fk = sn.get('fk_col_name')
                                    if not s_fk:
                                        # Try to infer FK from s_model to sub_model
                                        try:
                                            for c in s_model.__table__.columns:
                                                for fk in c.foreign_keys:
                                                    if fk.column.table.name == sub_model.__table__.name:
                                                        s_fk = c.name
                                                        break
                                                if s_fk:
                                                    break
                                        except Exception:
                                            s_fk = None
                                    if not s_fk:
                                        continue
                                    try:
                                        sub_pk_for_nested = next(iter(sub_model.__table__.primary_key.columns)).name
                                    except Exception:
                                        sub_pk_for_nested = 'id'
                                    nsql2 = self.build_nested_list_sql(
                                        alias=s_alias,
                                        grand_model=s_model,
                                        child_table=sub_model,
                                        g_fk_col_name=s_fk,
                                        child_pk_name=sub_pk_for_nested,
                                        fields=sn.get('fields'),
                                        where_dict=_as_dict2(sn.get('where')),
                                        default_where=_as_dict2(sn.get('default_where')),
                                        order_by=sn.get('order_by'),
                                        order_dir=sn.get('order_dir'),
                                        order_multi=sn.get('order_multi'),
                                        limit=sn.get('limit'),
                                        offset=sn.get('offset'),
                                        extra_where_sql=None,
                                        nested_children=sn.get('nested') or None,
                                    )
                                    sub_parts.append(f"ISNULL(({nsql2}), '[]') AS [{s_alias}]")
                            if sub_parts:
                                extra_cols = ', ' + ', '.join(sub_parts)
                        single_sql = f"ISNULL((SELECT TOP 1 {sub_cols}{extra_cols} FROM {self.table_ident(sub_model)} WHERE {join_cond} FOR JSON PATH, WITHOUT_ARRAY_WRAPPER), 'null')"
                        n_parts.append(f"{single_sql} AS [{sub_alias}]")
                    else:
                        sub_fk = nn.get('fk_col_name')
                        sub_fields = nn.get('fields')
                        sub_where = nn.get('where')
                        sub_default_where = nn.get('default_where')
                        sub_order_by = nn.get('order_by')
                        sub_order_dir = nn.get('order_dir')
                        sub_order_multi = nn.get('order_multi')
                        sub_limit = nn.get('limit')
                        sub_offset = nn.get('offset')
                        # Recurse with PK of current child (grand_model)
                        try:
                            child_pk_recur = next(iter(grand_model.__table__.primary_key.columns)).name
                        except Exception:
                            child_pk_recur = 'id'
                        sub_sql = self.build_nested_list_sql(
                            alias=sub_alias,
                            grand_model=sub_model,
                            child_table=grand_model,
                            g_fk_col_name=sub_fk,
                            child_pk_name=child_pk_recur,
                            fields=sub_fields,
                            where_dict=sub_where,
                            default_where=sub_default_where,
                            order_by=sub_order_by,
                            order_dir=sub_order_dir,
                            order_multi=sub_order_multi,
                            limit=sub_limit,
                            offset=sub_offset,
                            extra_where_sql=None,
                            nested_children=None,
                        )
                        n_parts.append(f"ISNULL(({sub_sql}), '[]') AS [{sub_alias}]")
                except Exception:
                    continue
            if n_parts:
                nested_cols = ', ' + ', '.join(n_parts)
        n_order = self._build_order_clause(grand_model, grand_model.__tablename__, order_by, order_dir, order_multi)
        # Build pagination using ORDER BY ... OFFSET/FETCH to support offset reliably
        pag_clause = ''
        if limit is not None or offset is not None:
            if not n_order:
                # Fallback to PK asc to satisfy OFFSET/FETCH requirement
                try:
                    pk_name_fallback = next(iter(grand_model.__table__.primary_key.columns)).name
                except Exception:
                    pk_name_fallback = 'id'
                n_order = self._build_order_clause(grand_model, grand_model.__tablename__, pk_name_fallback, 'asc', None)
            o = self._as_int(offset) or 0
            pag_clause = f" ORDER BY {n_order} OFFSET {o} ROWS"
            if limit is not None:
                pag_clause += f" FETCH NEXT {self._as_int(limit)} ROWS ONLY"
        elif n_order:
            pag_clause = f" ORDER BY {n_order}"
        return f"SELECT {n_col_select}{nested_cols} FROM {self.table_ident(grand_model)} WHERE {n_where}{pag_clause} FOR JSON PATH"

    def build_relation_list_json_full(self, *, parent_table: str, parent_pk_name: str, child_model, fk_col_name: str, projected_columns: list[str], rel_where: dict | str | None, rel_default_where: dict | str | None, limit: int | None, offset: int | None, order_by: str | None, order_dir: str | None, order_multi: list[str] | None, nested: list[dict] | None) -> Any:
        """Assemble full MSSQL JSON aggregation for a to-many relation with optional nested arrays.

        nested: list of dicts, each with keys: alias, model, fk_col_name, fields, where, default_where, order_by, order_dir, order_multi, limit
        """
        cols = projected_columns or ['id']
        select_bits: list[str] = []
        if cols and isinstance(cols[0], (list, tuple)):
            for src, alias in cols:  # type: ignore
                select_bits.append(f"{self.table_ident(child_model)}.[{src}] AS [{alias}]")
        else:
            for c in cols:  # type: ignore
                select_bits.append(f"{self.table_ident(child_model)}.[{c}] AS [{c}]")
        select_cols = ', '.join(select_bits)
        # Build where: correlate to parent and apply relation-level JSON where
        where_parts_rel: list[str] = [
            f"{self.table_ident(child_model)}.[{fk_col_name}] = {self.table_ident(parent_table)}.[{parent_pk_name}]"
        ]
        import json as _json
        def _as_dict(maybe):
            # Accept dict or JSON string only; ignore callables/expressions
            if isinstance(maybe, dict):
                return maybe
            if isinstance(maybe, str):
                try:
                    v = _json.loads(maybe)
                    return v if isinstance(v, dict) else None
                except Exception:
                    return None
            return None
        wdict = _as_dict(rel_where)
        if wdict:
            where_parts_rel.extend(self.where_from_dict(child_model, wdict))
        dwdict = _as_dict(rel_default_where)
        if dwdict:
            where_parts_rel.extend(self.where_from_dict(child_model, dwdict))
        where_clause = ' AND '.join(where_parts_rel)
        # order clause
        order_clause = self._build_order_clause(child_model, child_model.__tablename__, order_by, order_dir, order_multi)
        # nested
        nested_cols = ''
        if nested:
            nested_parts = []
            for n in nested:
                alias = n.get('alias')
                gm = n.get('model')
                mode = n.get('mode') or 'list'
                if mode == 'single':
                    # Single nested object: join condition is child_model.child_fk_name = gm.pk
                    child_fk_name = n.get('child_fk_name')
                    if not child_fk_name:
                        continue
                    try:
                        gm_pk = next(iter(gm.__table__.primary_key.columns)).name
                    except Exception:
                        gm_pk = 'id'
                    join_cond = f"{self.table_ident(child_model)}.[{child_fk_name}] = {self.table_ident(gm)}.[{gm_pk}]"
                    cols_pairs = n.get('fields') or []
                    # When no explicit fields provided, select all columns from gm
                    if not cols_pairs:
                        cols_pairs = [(c.name, c.name) for c in gm.__table__.columns]
                    # Build column list including nested-of-single projections
                    gm_cols = ', '.join([f"{self.table_ident(gm)}.[{src}] AS [{alias}]" for src, alias in cols_pairs])
                    extra_nested_cols = ''
                    # Process nested children under this single node (if any)
                    sub_nests = n.get('nested') or []
                    if sub_nests:
                        sub_parts = []
                        for sn in sub_nests:
                            s_alias = sn.get('alias')
                            s_model = sn.get('model')
                            s_mode = sn.get('mode') or 'list'
                            if s_mode == 'single':
                                # Correlate gm -> s_model via gm.child_fk_name = s_model.pk
                                s_child_fk = sn.get('child_fk_name')
                                if not s_child_fk:
                                    continue
                                try:
                                    s_pk = next(iter(s_model.__table__.primary_key.columns)).name
                                except Exception:
                                    s_pk = 'id'
                                s_join = f"{self.table_ident(gm)}.[{s_child_fk}] = {self.table_ident(s_model)}.[{s_pk}]"
                                s_cols_pairs = sn.get('fields') or [(c.name, c.name) for c in s_model.__table__.columns]
                                s_sub = self.build_single_relation_json(
                                    child_table=s_model,
                                    projected_columns=s_cols_pairs,
                                    join_condition=s_join,
                                )
                                sub_parts.append(f"ISNULL(({str(s_sub).strip()}), 'null') AS [{s_alias}]")
                            else:
                                # List under gm: s_model.fk_to_gm = gm.pk
                                s_fk = sn.get('fk_col_name')
                                if not s_fk:
                                    continue
                                try:
                                    child_pk_for_nested = gm_pk
                                except Exception:
                                    child_pk_for_nested = 'id'
                                nsql = self.build_nested_list_sql(
                                    alias=s_alias,
                                    grand_model=s_model,
                                    child_table=gm,
                                    g_fk_col_name=s_fk,
                                    child_pk_name=child_pk_for_nested,
                                    fields=sn.get('fields'),
                                    where_dict=_as_dict(sn.get('where')),
                                    default_where=_as_dict(sn.get('default_where')),
                                    order_by=sn.get('order_by'),
                                    order_dir=sn.get('order_dir'),
                                    order_multi=sn.get('order_multi'),
                                    limit=sn.get('limit'),
                                    offset=sn.get('offset'),
                                    extra_where_sql=None,
                                    nested_children=sn.get('nested') or None,
                                )
                                sub_parts.append(f"ISNULL(({nsql}), '[]') AS [{s_alias}]")
                        if sub_parts:
                            extra_nested_cols = ', ' + ', '.join(sub_parts)
                    # Compose final single object with nested columns
                    inner = f"ISNULL((SELECT TOP 1 {gm_cols}{extra_nested_cols} FROM {self.table_ident(gm)} WHERE {join_cond} FOR JSON PATH, WITHOUT_ARRAY_WRAPPER), 'null')"
                    nested_parts.append(f"{inner} AS [{alias}]")
                    continue
                # List nested branch (existing behavior)
                gfk = n.get('fk_col_name')
                if not gfk:
                    # Fallback: find FK on nested model (gm) referencing child_model
                    try:
                        for col in gm.__table__.columns:
                            for fk in col.foreign_keys:
                                if fk.column.table.name == child_model.__table__.name:
                                    gfk = col.name
                                    break
                            if gfk:
                                break
                    except Exception:
                        gfk = None
                # Prepare nested-of-nested specs if any
                n_children = []
                try:
                    for nn in (n.get('nested') or []):
                        n_children.append(nn)
                except Exception:
                    n_children = []
                # Determine PK name for current child table (child_model)
                try:
                    child_pk_for_nested = next(iter(child_model.__table__.primary_key.columns)).name
                except Exception:
                    child_pk_for_nested = 'id'
                nsql = self.build_nested_list_sql(
                    alias=alias,
                    grand_model=gm,
                    child_table=child_model,
                    g_fk_col_name=gfk,
                    child_pk_name=child_pk_for_nested,
                    fields=n.get('fields'),
                    where_dict=_as_dict(n.get('where')),
                    default_where=_as_dict(n.get('default_where')),
                    order_by=n.get('order_by'),
                    order_dir=n.get('order_dir'),
                    order_multi=n.get('order_multi'),
                    limit=n.get('limit'),
                    offset=n.get('offset'),
                    extra_where_sql=None,
                    nested_children=n_children or None,
                )
                nested_parts.append(f"ISNULL(({nsql}), '[]') AS [{alias}]")
            nested_cols = ', ' + ', '.join(nested_parts) if nested_parts else ''
        # Pagination using ORDER BY ... OFFSET/FETCH to support offset reliably
        pag_clause = ''
        if limit is not None or offset is not None:
            # Ensure we have an order clause
            if not order_clause:
                try:
                    pk_fallback = next(iter(child_model.__table__.primary_key.columns)).name
                except Exception:
                    pk_fallback = 'id'
                order_clause = self._build_order_clause(child_model, child_model.__tablename__, pk_fallback, 'asc', None)
            o = self._as_int(offset) or 0
            pag_clause = f" ORDER BY {order_clause} OFFSET {o} ROWS"
            if limit is not None:
                pag_clause += f" FETCH NEXT {self._as_int(limit)} ROWS ONLY"
        elif order_clause:
            pag_clause = f" ORDER BY {order_clause}"
        raw = (
            f"ISNULL((SELECT {select_cols}{nested_cols} FROM {self.table_ident(child_model)} WHERE {where_clause}{pag_clause} FOR JSON PATH),'[]')"
        )
        return _text(raw)

    def build_single_relation_json(self, *, child_table: str, projected_columns: list[str], join_condition: str) -> Any:
        """Build a FOR JSON PATH sub-select for a single related object.

        Parameters:
            child_table: table name of related entity
            projected_columns: list of column names to include
            join_condition: raw SQL condition joining related row(s) to parent
        Returns a TextClause producing a JSON object string or 'null'.
        """
        cols = projected_columns or ['id']
        if cols and isinstance(cols[0], (list, tuple)):
            # respect aliasing if pairs provided
            col_list = ', '.join([f"{self.table_ident(child_table)}.[{src}] AS [{alias}]" for src, alias in cols])  # type: ignore
        else:
            col_list = ', '.join([f"{self.table_ident(child_table)}.[{c}]" for c in cols])  # type: ignore
        raw = (
            f"ISNULL((SELECT TOP 1 {col_list} FROM {self.table_ident(child_table)} WHERE {join_condition} "
            f"FOR JSON PATH, WITHOUT_ARRAY_WRAPPER), 'null')"
        )
        return _text(raw)

    def build_list_relation_json(self, *, child_table: str, projected_columns: list[str], where_condition: str, limit: int | None, offset: int | None, order_by: str | None, nested_subqueries: list[tuple[str, str]] | None = None) -> Any:
        """Build a FOR JSON PATH list aggregation for a to-many relation.

        nested_subqueries: list of (alias, subquery_sql producing JSON array string), each will be selected
        as a scalar column so it becomes a property on each object in the JSON array.
        """
        cols = projected_columns or ['id']
        select_bits: list[str] = []
        if cols and isinstance(cols[0], (list, tuple)):
            for src, alias in cols:  # type: ignore
                select_bits.append(f"{self.table_ident(child_table)}.[{src}] AS [{alias}]")
        else:
            for c in cols:  # type: ignore
                select_bits.append(f"{self.table_ident(child_table)}.[{c}] AS [{c}]")
        select_cols = ', '.join(select_bits)
        nested_cols = ''
        if nested_subqueries:
            # Ensure each nested is wrapped in ISNULL(...) to always return '[]'
            nested_parts = [f"ISNULL(({sql}), '[]') AS [{alias}]" for alias, sql in nested_subqueries]
            nested_cols = (', ' + ', '.join(nested_parts)) if nested_parts else ''
        # Pagination using ORDER BY ... OFFSET/FETCH to support offset
        pag_clause = ''
        if limit is not None or offset is not None:
            ob = order_by
            if not ob:
                # Safety fallback; builders should supply an order when paginating
                ob = "(SELECT NULL)"
            o = self._as_int(offset) or 0
            pag_clause = f" ORDER BY {ob} OFFSET {o} ROWS"
            if limit is not None:
                pag_clause += f" FETCH NEXT {self._as_int(limit)} ROWS ONLY"
        elif order_by:
            pag_clause = f" ORDER BY {order_by}"
        raw = (
            f"ISNULL((SELECT {select_cols}{nested_cols} FROM {self.table_ident(child_table)} WHERE {where_condition}{pag_clause} FOR JSON PATH),'[]')"
        )
        return _text(raw)

