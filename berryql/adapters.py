from __future__ import annotations
from typing import Any
from sqlalchemy import func, text as _text, literal

class BaseAdapter:
    name = 'base'
    def json_object(self, *args):
        raise NotImplementedError
    def json_array_agg(self, expr):
        raise NotImplementedError
    def json_array_coalesce(self, expr):
        raise NotImplementedError
    def supports_relation_pushdown(self) -> bool:
        return True

class SQLiteAdapter(BaseAdapter):
    name = 'sqlite'
    def json_object(self, *args):
        return func.json_object(*args)
    def json_array_agg(self, expr):
        return func.json_group_array(expr)
    def json_array_coalesce(self, expr):
        # Use a bound literal to avoid deprecation warnings for implicit string coercion
        return func.coalesce(expr, literal('[]'))

class PostgresAdapter(BaseAdapter):
    name = 'postgres'
    def json_object(self, *args):
        return func.json_build_object(*args)
    def json_array_agg(self, expr):
        return func.json_agg(expr)
    def json_array_coalesce(self, expr):
        # Prefer a typed literal over raw text for JSON to avoid deprecation warnings
        try:
            from sqlalchemy.dialects.postgresql import JSON as _PG_JSON
            return func.coalesce(expr, literal('[]', type_=_PG_JSON()))
        except Exception:
            # Fallback keeps previous behavior if dialect types aren't available at import time
            return func.coalesce(expr, _text("'[]'::json"))

class MSSQLAdapter(BaseAdapter):
    name = 'mssql'
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
                continue
            for op_name, val in (op_map or {}).items():
                tgt = f"[{model_cls.__tablename__}].[{col_name}]"
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

    def _build_order_clause(self, model_cls, table_alias: str, order_by: str | None, order_dir: str | None, order_multi: list[str] | None) -> str | None:
        parts: list[str] = []
        multi = self._as_list_str(order_multi)
        ob_str = self._as_str(order_by)
        od_str = (self._as_str(order_dir) or 'asc').lower()
        for spec in multi:
            try:
                cn, _, dd = str(spec).partition(':')
                dd = (dd or od_str).lower()
                if cn in model_cls.__table__.columns:
                    parts.append(f"[{table_alias}].[{cn}] {'DESC' if dd=='desc' else 'ASC'}")
            except Exception:
                continue
        if not parts and ob_str and ob_str in model_cls.__table__.columns:
            dd = od_str
            parts.append(f"[{table_alias}].[{ob_str}] {'DESC' if dd=='desc' else 'ASC'}")
        if not parts and 'id' in model_cls.__table__.columns:
            parts.append(f"[{table_alias}].[id] ASC")
        return ', '.join(parts) if parts else None

    def build_nested_list_sql(self, *, alias: str, grand_model, child_table: str, g_fk_col_name: str, fields: list[str] | None, where_dict: dict | str | None, default_where: dict | str | None, order_by: str | None, order_dir: str | None, order_multi: list[str] | None, limit: int | None, offset: int | None, extra_where_sql: list[str] | None = None) -> str:
        n_cols = fields or []
        if not n_cols:
            for c in grand_model.__table__.columns:
                n_cols.append(c.name)
        n_col_select = ', '.join([f"[{grand_model.__tablename__}].[{c}] AS [{c}]" for c in (n_cols or ['id'])])
        where_parts = [f"[{grand_model.__tablename__}].[{g_fk_col_name}] = [{child_table}].[id]"]
        # accept both dict and JSON string
        import json as _json
        def _as_dict(maybe):
            if isinstance(maybe, str):
                try:
                    return _json.loads(maybe)
                except Exception:
                    return None
            return maybe
        if where_dict:
            where_parts.extend(self.where_from_dict(grand_model, _as_dict(where_dict)))
        if default_where:
            where_parts.extend(self.where_from_dict(grand_model, _as_dict(default_where)))
        if extra_where_sql:
            where_parts.extend([str(x) for x in extra_where_sql if x])
        n_where = ' AND '.join(where_parts)
        n_order = self._build_order_clause(grand_model, grand_model.__tablename__, order_by, order_dir, order_multi)
        # Build pagination using ORDER BY ... OFFSET/FETCH to support offset reliably
        pag_clause = ''
        if limit is not None or offset is not None:
            if not n_order:
                # Fallback to id asc to satisfy OFFSET/FETCH requirement
                n_order = self._build_order_clause(grand_model, grand_model.__tablename__, 'id', 'asc', None)
            o = self._as_int(offset) or 0
            pag_clause = f" ORDER BY {n_order} OFFSET {o} ROWS"
            if limit is not None:
                pag_clause += f" FETCH NEXT {self._as_int(limit)} ROWS ONLY"
        elif n_order:
            pag_clause = f" ORDER BY {n_order}"
        return f"SELECT {n_col_select} FROM {grand_model.__tablename__} WHERE {n_where}{pag_clause} FOR JSON PATH"

    def build_relation_list_json_full(self, *, parent_table: str, child_model, fk_col_name: str, projected_columns: list[str], rel_where: dict | str | None, rel_default_where: dict | str | None, limit: int | None, offset: int | None, order_by: str | None, order_dir: str | None, order_multi: list[str] | None, nested: list[dict] | None) -> Any:
        """Assemble full MSSQL JSON aggregation for a to-many relation with optional nested arrays.

        nested: list of dicts, each with keys: alias, model, fk_col_name, fields, where, default_where, order_by, order_dir, order_multi, limit
        """
        cols = projected_columns or ['id']
        select_cols = ', '.join([f"[{child_model.__tablename__}].[{c}] AS [{c}]" for c in cols])
        # Build where: correlate to parent and apply relation-level JSON where
        where_parts_rel: list[str] = [f"[{child_model.__tablename__}].[{fk_col_name}] = [{parent_table}].[id]"]
        import json as _json
        def _as_dict(maybe):
            if isinstance(maybe, str):
                try:
                    return _json.loads(maybe)
                except Exception:
                    return None
            return maybe
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
                gfk = n.get('fk_col_name')
                nsql = self.build_nested_list_sql(
                    alias=alias,
                    grand_model=gm,
                    child_table=child_model.__tablename__,
                    g_fk_col_name=gfk,
                    fields=n.get('fields'),
                    where_dict=_as_dict(n.get('where')),
                    default_where=_as_dict(n.get('default_where')),
                    order_by=n.get('order_by'),
                    order_dir=n.get('order_dir'),
                    order_multi=n.get('order_multi'),
                    limit=n.get('limit'),
                    offset=n.get('offset'),
                    extra_where_sql=None,
                )
                nested_parts.append(f"ISNULL(({nsql}), '[]') AS [{alias}]")
            nested_cols = ', ' + ', '.join(nested_parts) if nested_parts else ''
        # Pagination using ORDER BY ... OFFSET/FETCH to support offset reliably
        pag_clause = ''
        if limit is not None or offset is not None:
            # Ensure we have an order clause
            if not order_clause:
                order_clause = self._build_order_clause(child_model, child_model.__tablename__, 'id', 'asc', None)
            o = self._as_int(offset) or 0
            pag_clause = f" ORDER BY {order_clause} OFFSET {o} ROWS"
            if limit is not None:
                pag_clause += f" FETCH NEXT {self._as_int(limit)} ROWS ONLY"
        elif order_clause:
            pag_clause = f" ORDER BY {order_clause}"
        raw = (
            f"ISNULL((SELECT {select_cols}{nested_cols} FROM {child_model.__tablename__} WHERE {where_clause}{pag_clause} FOR JSON PATH),'[]')"
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
        col_list = ', '.join([f"[{child_table}].[{c}]" for c in cols])
        raw = (
            f"ISNULL((SELECT TOP 1 {col_list} FROM {child_table} WHERE {join_condition} "
            f"FOR JSON PATH, WITHOUT_ARRAY_WRAPPER), 'null')"
        )
        return _text(raw)

    def build_list_relation_json(self, *, child_table: str, projected_columns: list[str], where_condition: str, limit: int | None, offset: int | None, order_by: str | None, nested_subqueries: list[tuple[str, str]] | None = None) -> Any:
        """Build a FOR JSON PATH list aggregation for a to-many relation.

        nested_subqueries: list of (alias, subquery_sql producing JSON array string), each will be selected
        as a scalar column so it becomes a property on each object in the JSON array.
        """
        cols = projected_columns or ['id']
        select_cols = ', '.join([f"[{child_table}].[{c}] AS [{c}]" for c in cols])
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
                ob = f"[{child_table}].[id] ASC"
            o = self._as_int(offset) or 0
            pag_clause = f" ORDER BY {ob} OFFSET {o} ROWS"
            if limit is not None:
                pag_clause += f" FETCH NEXT {self._as_int(limit)} ROWS ONLY"
        elif order_by:
            pag_clause = f" ORDER BY {order_by}"
        raw = (
            f"ISNULL((SELECT {select_cols}{nested_cols} FROM {child_table} WHERE {where_condition}{pag_clause} FOR JSON PATH),'[]')"
        )
        return _text(raw)

def get_adapter(dialect_name: str) -> BaseAdapter:
    dn = (dialect_name or '').lower()
    if dn.startswith('postgres'):
        return PostgresAdapter()
    if dn.startswith('mssql') or 'pyodbc' in dn:
        return MSSQLAdapter()
    return SQLiteAdapter()

__all__ = ['get_adapter','BaseAdapter','SQLiteAdapter','PostgresAdapter','MSSQLAdapter']
