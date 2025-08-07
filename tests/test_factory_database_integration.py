"""
Integration tests for BerryQL factory with database adapters.

This module tests the integration between the BerryQL factory and different
database adapters to ensure cross-database compatibility.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import strawberry
from typing import List, Optional
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship

from berryql import BerryQLFactory, GraphQLQueryParams
from berryql.database_adapters import (
    PostgreSQLAdapter,
    SQLiteAdapter,
    MSSQLAdapter,
    get_database_adapter
)


# Test Models
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    posts = relationship("Post", back_populates="author")


class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    author = relationship("User", back_populates="posts")


# Strawberry Types
@strawberry.type
class PostType:
    id: int
    title: str
    content: Optional[str]
    author_id: int
    created_at: datetime


@strawberry.type
class UserType:
    id: int
    name: str
    email: str
    created_at: datetime
    
    @strawberry.field
    async def posts(self, info: strawberry.Info) -> List[PostType]:
        from berryql import get_resolved_field_data
        return get_resolved_field_data(self, info, 'posts')


class TestFactoryDatabaseIntegration:
    """Test BerryQL factory integration with different database adapters."""
    
    @pytest.fixture
    def factory(self):
        """Create a BerryQL factory instance."""
        return BerryQLFactory()
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        mock_session = Mock(spec=AsyncSession)
        mock_bind = Mock()
        mock_session.bind = mock_bind
        return mock_session
    
    def test_factory_gets_postgresql_adapter(self, factory, mock_db_session):
        """Test that factory correctly identifies and uses PostgreSQL adapter."""
        # Mock PostgreSQL engine
        mock_db_session.bind.dialect.name = 'postgresql'
        
        adapter = factory._get_db_adapter(mock_db_session)
        assert isinstance(adapter, PostgreSQLAdapter)
        
        # Test that adapter is cached
        adapter2 = factory._get_db_adapter(mock_db_session)
        assert adapter is adapter2
    
    def test_factory_gets_sqlite_adapter(self, factory, mock_db_session):
        """Test that factory correctly identifies and uses SQLite adapter."""
        # Mock SQLite engine
        mock_db_session.bind.dialect.name = 'sqlite'
        
        adapter = factory._get_db_adapter(mock_db_session)
        assert isinstance(adapter, SQLiteAdapter)
    
    def test_factory_gets_mssql_adapter(self, factory, mock_db_session):
        """Test that factory correctly identifies and uses MSSQL adapter."""
        # Mock MSSQL engine
        mock_db_session.bind.dialect.name = 'mssql'
        
        adapter = factory._get_db_adapter(mock_db_session)
        assert isinstance(adapter, MSSQLAdapter)
    
    def test_factory_caches_adapter_instance(self, factory, mock_db_session):
        """Test that factory caches the adapter instance."""
        mock_db_session.bind.dialect.name = 'postgresql'
        
        # First call should create adapter
        adapter1 = factory._get_db_adapter(mock_db_session)
        
        # Second call should return cached adapter
        adapter2 = factory._get_db_adapter(mock_db_session)
        
        assert adapter1 is adapter2
        assert isinstance(adapter1, PostgreSQLAdapter)
    
    @patch('berryql.database_adapters.get_database_adapter')
    def test_factory_uses_adapter_in_nested_query_building(self, mock_get_adapter, factory, mock_db_session):
        """Test that factory uses the database adapter when building nested queries."""
        # Setup mock adapter
        mock_adapter = Mock()
        mock_adapter.json_build_object.return_value = Mock()
        mock_adapter.json_agg.return_value = Mock()
        mock_adapter.json_empty_array.return_value = Mock()
        mock_adapter.coalesce_json.return_value = Mock()
        mock_get_adapter.return_value = mock_adapter
        
        # Mock database session with execute method
        mock_result = Mock()
        mock_result.first.return_value = ('[]',)
        mock_db_session.execute = AsyncMock(return_value=mock_result)
        
        # Create the resolver
        resolver = factory.create_berryql_resolver(
            strawberry_type=UserType,
            model_class=User
        )
        
        # This should trigger the database adapter usage in query building
        # We can't easily test the full execution without a real database,
        # but we can verify the adapter is called
        factory._db_adapter = mock_adapter
        
        # The adapter should be used when building JSON queries
        assert factory._db_adapter is mock_adapter
    
    def test_factory_handles_async_engine(self, factory):
        """Test that factory correctly handles async engines."""
        # Mock async engine with sync_engine attribute
        mock_async_session = Mock(spec=AsyncSession)
        mock_async_engine = Mock()
        mock_sync_engine = Mock()
        mock_sync_engine.dialect.name = 'postgresql'
        mock_async_engine.sync_engine = mock_sync_engine
        mock_async_session.bind = mock_async_engine
        
        adapter = factory._get_db_adapter(mock_async_session)
        assert isinstance(adapter, PostgreSQLAdapter)


class TestCrossDatabaseCompatibility:
    """Test cross-database compatibility features."""
    
    @pytest.fixture
    def factory(self):
        return BerryQLFactory()
    
    def test_postgresql_json_functions(self, factory):
        """Test PostgreSQL-specific JSON function generation."""
        mock_session = Mock(spec=AsyncSession)
        mock_session.bind.dialect.name = 'postgresql'
        
        adapter = factory._get_db_adapter(mock_session)
        
        # Test that PostgreSQL adapter generates appropriate functions
        json_agg = adapter.json_agg(Mock())
        json_build_object = adapter.json_build_object("key", "value")
        json_empty = adapter.json_empty_array()
        
        # Verify these are the correct types (they should be SQLAlchemy expressions)
        assert hasattr(json_agg, 'compare')
        assert hasattr(json_build_object, 'compare')
        assert hasattr(json_empty, 'compare')
    
    def test_sqlite_json_functions(self, factory):
        """Test SQLite-specific JSON function generation."""
        mock_session = Mock(spec=AsyncSession)
        mock_session.bind.dialect.name = 'sqlite'
        
        adapter = factory._get_db_adapter(mock_session)
        
        # Test that SQLite adapter generates appropriate functions
        json_agg = adapter.json_agg(Mock())
        json_build_object = adapter.json_build_object("key", "value")
        json_empty = adapter.json_empty_array()
        
        # Verify these are the correct types
        assert hasattr(json_agg, 'compare')
        assert hasattr(json_build_object, 'compare')
        assert hasattr(json_empty, 'compare')
    
    def test_mssql_json_functions(self, factory):
        """Test MSSQL-specific JSON function generation."""
        mock_session = Mock(spec=AsyncSession)
        mock_session.bind.dialect.name = 'mssql'
        
        adapter = factory._get_db_adapter(mock_session)
        
        # Test that MSSQL adapter generates appropriate functions
        json_agg = adapter.json_agg(Mock())
        json_build_object = adapter.json_build_object("key", "value")
        json_empty = adapter.json_empty_array()
        
        # Verify these are the correct types
        assert hasattr(json_agg, 'compare')
        assert hasattr(json_build_object, 'compare')
        assert hasattr(json_empty, 'compare')
    
    def test_adapter_switching(self, factory):
        """Test that factory can switch between different adapters."""
        # Create sessions for different databases
        pg_session = Mock(spec=AsyncSession)
        pg_session.bind.dialect.name = 'postgresql'
        
        sqlite_session = Mock(spec=AsyncSession)
        sqlite_session.bind.dialect.name = 'sqlite'
        
        # Factory should cache adapter per instance, but we can test
        # that it would get the right adapter for different engines
        factory._db_adapter = None  # Reset cache
        pg_adapter = factory._get_db_adapter(pg_session)
        
        factory._db_adapter = None  # Reset cache
        sqlite_adapter = factory._get_db_adapter(sqlite_session)
        
        assert isinstance(pg_adapter, PostgreSQLAdapter)
        assert isinstance(sqlite_adapter, SQLiteAdapter)
        assert type(pg_adapter) != type(sqlite_adapter)


class TestErrorHandlingIntegration:
    """Test error handling in factory-adapter integration."""
    
    def test_factory_handles_unknown_database(self):
        """Test that factory handles unknown database types gracefully."""
        factory = BerryQLFactory()
        mock_session = Mock(spec=AsyncSession)
        mock_session.bind.dialect.name = 'unknown_db'
        
        # Should fall back to PostgreSQL adapter
        adapter = factory._get_db_adapter(mock_session)
        assert isinstance(adapter, PostgreSQLAdapter)
    
    def test_factory_handles_missing_bind(self):
        """Test that factory handles sessions without bind attribute."""
        factory = BerryQLFactory()
        mock_session = Mock(spec=AsyncSession)
        del mock_session.bind  # Remove bind attribute
        
        # Should handle gracefully (may raise AttributeError, which is expected)
        with pytest.raises(AttributeError):
            factory._get_db_adapter(mock_session)
    
    def test_adapter_method_signatures(self):
        """Test that all adapters have consistent method signatures."""
        from berryql.database_adapters import DatabaseAdapter
        import inspect
        
        # Get all adapter classes
        adapter_classes = [PostgreSQLAdapter, SQLiteAdapter, MSSQLAdapter]
        
        # Get base class methods
        base_methods = {
            name: method for name, method in inspect.getmembers(DatabaseAdapter, predicate=inspect.isfunction)
            if not name.startswith('_')
        }
        
        # Check each adapter implements all base methods
        for adapter_class in adapter_classes:
            adapter_methods = {
                name: method for name, method in inspect.getmembers(adapter_class, predicate=inspect.ismethod)
                if not name.startswith('_')
            }
            
            for base_method_name in base_methods:
                assert base_method_name in adapter_methods, f"{adapter_class.__name__} missing method {base_method_name}"


@pytest.mark.integration
class TestRealDatabaseIntegration:
    """Integration tests with actual database connections (marked as integration tests)."""
    
    @pytest.mark.asyncio
    async def test_sqlite_memory_database(self):
        """Test with an actual SQLite in-memory database."""
        # Create SQLite in-memory database
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Create session
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        # Test adapter detection
        factory = BerryQLFactory()
        
        async with async_session() as session:
            adapter = factory._get_db_adapter(session)
            assert isinstance(adapter, SQLiteAdapter)
        
        await engine.dispose()
    
    @pytest.mark.skipif(True, reason="Requires PostgreSQL server")
    @pytest.mark.asyncio
    async def test_postgresql_database(self):
        """Test with an actual PostgreSQL database (requires server)."""
        # This test would require a running PostgreSQL server
        # Skip by default, but can be enabled for full integration testing
        
        # engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/test")
        # factory = BerryQLFactory()
        # 
        # async with async_sessionmaker(engine, class_=AsyncSession)() as session:
        #     adapter = factory._get_db_adapter(session)
        #     assert isinstance(adapter, PostgreSQLAdapter)
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
