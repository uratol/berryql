from __future__ import annotations
from typing import Any
from sqlalchemy import func, text as _text

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
        return func.coalesce(expr, '[]')

class PostgresAdapter(BaseAdapter):
    name = 'postgres'
    def json_object(self, *args):
        return func.json_build_object(*args)
    def json_array_agg(self, expr):
        return func.json_agg(expr)
    def json_array_coalesce(self, expr):
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

    def build_list_relation_json(self, *, child_table: str, projected_columns: list[str], where_condition: str, limit: int | None, order_by: str | None, nested_subqueries: list[tuple[str, str]] | None = None) -> Any:
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
        top_clause = f"TOP ({int(limit)}) " if limit is not None else ''
        order_clause = f" ORDER BY {order_by}" if order_by else ''
        raw = (
            f"ISNULL((SELECT {top_clause}{select_cols}{nested_cols} FROM {child_table} WHERE {where_condition}{order_clause} FOR JSON PATH),'[]')"
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
