"""
Test database adapters.
"""

import pytest
from unittest.mock import Mock
from sqlalchemy import text, func
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine
from berryql.database_adapters import (
    DatabaseAdapter,
    PostgreSQLAdapter,
    SQLiteAdapter,
    MSSQLAdapter,
    LegacySQLiteAdapter,
    get_database_adapter,
    check_sqlite_json_support,
)


class TestDatabaseAdapters:
    """Test database adapter implementations."""
    
    def test_postgresql_adapter(self):
        """Test PostgreSQL adapter functionality."""
        adapter = PostgreSQLAdapter()
        
        # Test json_agg
        result = adapter.json_agg(text("expression"))
        assert result is not None
        
        # Test json_build_object
        result = adapter.json_build_object("key1", text("val1"), "key2", text("val2"))
        assert result is not None
        
        # Test json_empty_array
        result = adapter.json_empty_array()
        assert result is not None
    
    def test_sqlite_adapter(self):
        """Test SQLite adapter functionality."""
        adapter = SQLiteAdapter()
        
        # Test json_agg
        result = adapter.json_agg(text("expression"))
        assert result is not None
        
        # Test json_build_object
        result = adapter.json_build_object("key1", text("val1"), "key2", text("val2"))
        assert result is not None
        
        # Test json_empty_array
        result = adapter.json_empty_array()
        assert result is not None
    
    def test_mssql_adapter(self):
        """Test MSSQL adapter functionality."""
        adapter = MSSQLAdapter()
        
        # Test json_agg (uses string_agg) - just verify it returns something
        result = adapter.json_agg(text("expression"))
        assert result is not None
        
        # Test json_build_object
        result = adapter.json_build_object("key1", text("val1"), "key2", text("val2"))
        assert result is not None
        
        # Test json_empty_array
        result = adapter.json_empty_array()
        assert result is not None
    
    def test_legacy_sqlite_adapter(self):
        """Test Legacy SQLite adapter functionality."""
        adapter = LegacySQLiteAdapter()
        
        # Test json_agg (uses GROUP_CONCAT) - just verify it returns something
        result = adapter.json_agg(text("expression"))
        assert result is not None
        
        # Test json_build_object
        result = adapter.json_build_object("key1", text("val1"), "key2", text("val2"))
        assert result is not None
        
        # Test json_empty_array
        result = adapter.json_empty_array()
        assert result is not None


class TestDatabaseAdapterFactory:
    """Test database adapter factory functions."""
    
    def test_get_postgresql_adapter(self):
        """Test getting PostgreSQL adapter."""
        # Mock PostgreSQL engine
        engine = Mock(spec=Engine)
        engine.dialect = Mock()
        engine.dialect.name = 'postgresql'
        
        adapter = get_database_adapter(engine)
        assert isinstance(adapter, PostgreSQLAdapter)
    
    def test_get_sqlite_adapter(self):
        """Test getting SQLite adapter."""
        # Mock SQLite engine
        engine = Mock(spec=Engine)
        engine.dialect = Mock()
        engine.dialect.name = 'sqlite'
        
        adapter = get_database_adapter(engine)
        assert isinstance(adapter, SQLiteAdapter)
    
    def test_get_mssql_adapter(self):
        """Test getting MSSQL adapter."""
        # Mock MSSQL engine
        engine = Mock(spec=Engine)
        engine.dialect = Mock()
        engine.dialect.name = 'mssql'
        
        adapter = get_database_adapter(engine)
        assert isinstance(adapter, MSSQLAdapter)
    
    def test_get_adapter_async_engine(self):
        """Test getting adapter from async engine."""
        # Mock async engine with sync_engine attribute
        async_engine = Mock(spec=AsyncEngine)
        sync_engine = Mock(spec=Engine)
        sync_engine.dialect = Mock()
        sync_engine.dialect.name = 'postgresql'
        async_engine.sync_engine = sync_engine
        
        adapter = get_database_adapter(async_engine)
        assert isinstance(adapter, PostgreSQLAdapter)
    
    def test_get_adapter_unsupported_dialect(self):
        """Test getting adapter for unsupported dialect falls back to PostgreSQL."""
        # Mock unsupported engine
        engine = Mock(spec=Engine)
        engine.dialect = Mock()
        engine.dialect.name = 'oracle'
        
        adapter = get_database_adapter(engine)
        assert isinstance(adapter, PostgreSQLAdapter)
    
    @pytest.mark.asyncio
    async def test_check_sqlite_json_support_success(self):
        """Test checking SQLite JSON support when available."""
        # Mock engine that supports JSON
        engine = Mock()
        conn_mock = Mock()
        result_mock = Mock()
        result_mock.fetchone.return_value = ('{"test": 1}',)
        conn_mock.execute.return_value = result_mock
        
        # Mock context manager
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=conn_mock)
        context_manager.__exit__ = Mock(return_value=None)
        engine.connect.return_value = context_manager
        
        result = check_sqlite_json_support(engine)
        assert result is True
        conn_mock.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_sqlite_json_support_failure(self):
        """Test checking SQLite JSON support when not available."""
        # Mock engine that doesn't support JSON
        engine = Mock()
        conn_mock = Mock()
        conn_mock.execute.side_effect = Exception("JSON functions not available")
        
        # Mock context manager
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=conn_mock)
        context_manager.__exit__ = Mock(return_value=None)
        engine.connect.return_value = context_manager
        
        result = check_sqlite_json_support(engine)
        assert result is False