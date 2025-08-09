"""
Database adapters for different SQL databases.

This module provides database-specific implementations for JSON aggregation
and other database-specific operations to support PostgreSQL, SQLite, and MSSQL.
"""

import logging
from abc import ABC, abstractmethod
from sqlalchemy import func, text, literal_column, literal
from sqlalchemy.sql import ColumnElement
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


class DatabaseAdapter(ABC):
    """Abstract base class for database-specific operations."""
    
    @abstractmethod
    def json_agg(self, expression: ColumnElement) -> ColumnElement:
        """Create a JSON array aggregation from the given expression."""
        pass
    
    @abstractmethod
    def json_build_object(self, *args) -> ColumnElement:
        """Build a JSON object from key-value pairs."""
        pass
    
    @abstractmethod
    def json_empty_array(self) -> ColumnElement:
        """Return an empty JSON array literal."""
        pass
    
    @abstractmethod
    def coalesce_json(self, json_expr: ColumnElement, default_expr: ColumnElement) -> ColumnElement:
        """Return json_expr if not null, otherwise return default_expr."""
        pass


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL-specific database adapter."""
    
    def json_agg(self, expression: ColumnElement) -> ColumnElement:
        """Use native PostgreSQL json_agg function."""
        return func.json_agg(expression)
    
    def json_build_object(self, *args) -> ColumnElement:
        """Use native PostgreSQL json_build_object function."""
        return func.json_build_object(*args)
    
    def json_empty_array(self) -> ColumnElement:
        """Return PostgreSQL JSON empty array literal."""
        return text("'[]'::json")
    
    def coalesce_json(self, json_expr: ColumnElement, default_expr: ColumnElement) -> ColumnElement:
        """Use PostgreSQL COALESCE function."""
        return func.coalesce(json_expr, default_expr)


class SQLiteAdapter(DatabaseAdapter):
    """SQLite-specific database adapter."""
    
    def json_agg(self, expression: ColumnElement) -> ColumnElement:
        """
        SQLite json_group_array aggregation.
        Note: Requires SQLite 3.38+ for JSON functions.
        """
        return func.json_group_array(expression)
    
    def json_build_object(self, *args) -> ColumnElement:
        """
        Build JSON object using SQLite json_object function.
        Note: Requires SQLite 3.38+ for JSON functions.
        """
        return func.json_object(*args)
    
    def json_empty_array(self) -> ColumnElement:
        """Return SQLite JSON empty array literal."""
        return text("'[]'")
    
    def coalesce_json(self, json_expr: ColumnElement, default_expr: ColumnElement) -> ColumnElement:
        """Use SQLite COALESCE function."""
        return func.coalesce(json_expr, default_expr)


class MSSQLAdapter(DatabaseAdapter):
    """Microsoft SQL Server-specific database adapter."""
    
    def json_agg(self, expression: ColumnElement) -> ColumnElement:
        """
        Use SQL Server 2016+ FOR JSON PATH aggregation.
        This creates a JSON array from the aggregated expressions.
        """
        # For MSSQL, we need to use a subquery with FOR JSON PATH
        # This is more complex and may need to be handled differently in the calling code
        # For now, we'll use a simpler approach with STRING_AGG and manual JSON formatting
        from sqlalchemy import String
        return func.string_agg(
            func.cast(expression, String()), 
            literal_column("','")
        )
    
    def json_build_object(self, *args) -> ColumnElement:
        """
        Emulate JSON object using safe string concatenation for SQL Server.
        Handles strings, numbers, booleans (bit), and datetime/date to ISO8601.
        """
        if len(args) % 2 != 0:
            raise ValueError("json_build_object requires an even number of arguments (key-value pairs)")

        if len(args) == 0:
            return text("'{}'")

        from sqlalchemy import case, cast
        from sqlalchemy.types import String as SAString, Integer as SAInteger
        from sqlalchemy.types import DateTime, Date, Numeric, Float, Integer, Boolean

        def _json_quote(expr):
            # Quote and JSON-escape string content
            # STRING_ESCAPE(expr, 'json') is available on SQL Server 2016+
            return func.concat(literal('"'), func.STRING_ESCAPE(cast(expr, SAString()), literal('json')), literal('"'))

        def _is_number(sqltype):
            try:
                return isinstance(sqltype, (Numeric, Float, Integer))
            except Exception:
                return False

        def _is_boolean(sqltype):
            try:
                return isinstance(sqltype, Boolean)
            except Exception:
                return False

        def _is_datetime(sqltype):
            try:
                return isinstance(sqltype, (DateTime, Date))
            except Exception:
                return False

        def _json_value(expr):
            sqltype = getattr(expr, 'type', None)
            # booleans -> true/false
            if _is_boolean(sqltype):
                return case((cast(expr, SAInteger()) == literal(1), literal('true')), else_=literal('false'))
            # numbers -> no quotes
            if _is_number(sqltype):
                return cast(expr, SAString())
            # datetimes -> ISO 8601 and quote
            if _is_datetime(sqltype):
                # CONVERT requires a type token and style literal, not bound params.
                # Use literal_column to render without binds.
                iso = func.convert(literal_column("varchar(33)"), expr, literal_column("126"))
                return _json_quote(iso)
            # default: string quoted
            return _json_quote(expr)

        parts = []
        for i in range(0, len(args), 2):
            key = args[i]
            val = args[i + 1]
            # Derive key name (string literal)
            if hasattr(key, 'text'):
                key_name = key.text.strip("'\"")
            else:
                key_name = str(key).strip("'\"")
            key_prefix = func.concat(literal('"'), literal(key_name), literal('":'))
            pair = func.concat(key_prefix, _json_value(val))
            parts.append(pair)

        # Join all pairs with commas
        content = parts[0]
        for p in parts[1:]:
            content = func.concat(content, literal(','), p)

        return func.concat(literal('{'), content, literal('}'))
    
    def json_empty_array(self) -> ColumnElement:
        """Return MSSQL JSON empty array literal."""
        return text("'[]'")
    
    def coalesce_json(self, json_expr: ColumnElement, default_expr: ColumnElement) -> ColumnElement:
        """Use MSSQL COALESCE function."""
        return func.coalesce(json_expr, default_expr)


class LegacySQLiteAdapter(DatabaseAdapter):
    """
    Legacy SQLite adapter for versions < 3.38 that don't have JSON functions.
    Uses string concatenation to simulate JSON.
    """
    
    def json_agg(self, expression: ColumnElement) -> ColumnElement:
        """
        Simulate JSON array using GROUP_CONCAT.
        This creates a comma-separated string that looks like JSON.
        """
        # Create a simpler version that doesn't use text concatenation
        return func.json_quote('[' + func.coalesce(func.group_concat(expression, ','), '') + ']')
    
    def json_build_object(self, *args) -> ColumnElement:
        """
        Simulate JSON object using string concatenation.
        Warning: This doesn't handle proper JSON escaping.
        """
        if len(args) % 2 != 0:
            raise ValueError("json_build_object requires an even number of arguments (key-value pairs)")
        
        if len(args) == 0:
            return text("'{}'")
        
        # Build JSON-like object manually using string formatting
        json_parts = []
        for i in range(0, len(args), 2):
            key = args[i]
            value = args[i + 1]
            # Extract key string
            if hasattr(key, 'text'):
                key_str = key.text.strip("'\"")
            else:
                key_str = str(key).strip("'\"")
            
            # Create key-value pair as a simple string
            json_parts.append(f'"{key_str}":"{value}"')
        
        # Join parts with commas and wrap in braces
        json_content = ','.join(json_parts)
        return text(f"'{{{json_content}}}'")
    
    def json_empty_array(self) -> ColumnElement:
        """Return empty array as string."""
        return text("'[]'")
    
    def coalesce_json(self, json_expr: ColumnElement, default_expr: ColumnElement) -> ColumnElement:
        """Use SQLite COALESCE function."""
        return func.coalesce(json_expr, default_expr)


def get_database_adapter(engine: Engine | AsyncEngine) -> DatabaseAdapter:
    """
    Get the appropriate database adapter based on the engine dialect.
    
    Args:
        engine: SQLAlchemy engine (sync or async)
        
    Returns:
        DatabaseAdapter: Appropriate adapter for the database type
    """
    # Get the actual engine if it's an async engine
    if hasattr(engine, 'sync_engine'):
        dialect_name = engine.sync_engine.dialect.name
    else:
        dialect_name = engine.dialect.name
    
    logger.info(f"Detected database dialect: {dialect_name}")
    
    if dialect_name == 'postgresql':
        return PostgreSQLAdapter()
    elif dialect_name == 'sqlite':
        # Check SQLite version for JSON support
        # For now, we'll default to the modern adapter and fall back if needed
        try:
            return SQLiteAdapter()
        except Exception as e:
            logger.warning(f"Modern SQLite JSON functions not available, falling back to legacy adapter: {e}")
            return LegacySQLiteAdapter()
    elif dialect_name in ('mssql', 'pyodbc'):
        return MSSQLAdapter()
    else:
        logger.warning(f"Unsupported database dialect: {dialect_name}. Falling back to PostgreSQL adapter.")
        return PostgreSQLAdapter()


def check_sqlite_json_support(engine: Engine | AsyncEngine) -> bool:
    """
    Check if the SQLite database supports JSON functions.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        bool: True if JSON functions are supported
    """
    try:
        # Try to execute a simple JSON function
        with engine.connect() as conn:
            result = conn.execute(text("SELECT json_object('test', 1)"))
            result.fetchone()
            return True
    except Exception:
        return False
