"""Basic tests for BerryQL package."""

import pytest
from berryql import BerryQLFactory, GraphQLQueryParams


def test_package_imports():
    """Test that all main components can be imported."""
    from berryql import (
        BerryQLFactory,
        GraphQLQueryParams,
        InvalidFieldError,
        QueryFieldAnalyzer,
        query_analyzer,
        get_resolved_field_data,
        ResolvedDataMixin,
        berryql_field,
        custom_field,
        berryql,
        convert_comparison_input,
        convert_where_input,
        convert_order_by_input,
        StringComparisonInput,
        IntComparisonInput,
        FloatComparisonInput,
        DateTimeComparisonInput,
        UUIDComparisonInput,
        BoolComparisonInput,
        OrderByInput,
        PaginationInput,
        ComparisonValue,
    )
    
    # Verify main components are available
    assert BerryQLFactory is not None
    assert GraphQLQueryParams is not None


def test_berryql_factory_creation():
    """Test BerryQL factory can be instantiated."""
    factory = BerryQLFactory()
    assert factory is not None
    assert hasattr(factory, 'create_berryql_resolver')


def test_graphql_query_params():
    """Test GraphQLQueryParams initialization."""
    # Test with no parameters
    params = GraphQLQueryParams()
    assert params.where == {}
    assert params.order_by == []
    assert params.offset is None
    assert params.limit is None
    
    # Test with parameters
    params = GraphQLQueryParams(
        where={'name': 'test'},
        order_by=[{'field': 'id', 'direction': 'asc'}],
        offset=10,
        limit=20
    )
    assert params.where == {'name': 'test'}
    assert params.order_by == [{'field': 'id', 'direction': 'asc'}]
    assert params.offset == 10
    assert params.limit == 20


def test_graphql_query_params_json_where():
    """Test GraphQLQueryParams with JSON string where condition."""
    # Test with JSON string
    params = GraphQLQueryParams(where='{"name": "test"}')
    assert params.where == {'name': 'test'}
    
    # Test with empty string
    params = GraphQLQueryParams(where='')
    assert params.where == {}
    
    # Test with invalid JSON
    params = GraphQLQueryParams(where='invalid json')
    assert params.where == {}


@pytest.mark.asyncio
async def test_factory_create_resolver():
    """Test that factory can create a resolver function."""
    from typing import List
    import strawberry
    from sqlalchemy import Column, Integer, String
    from sqlalchemy.orm import DeclarativeBase
    
    # Create test model
    class Base(DeclarativeBase):
        pass
    
    class TestModel(Base):
        __tablename__ = 'test_table'
        id = Column(Integer, primary_key=True)
        name = Column(String(50))
    
    # Create test Strawberry type
    @strawberry.type
    class TestType:
        id: int
        name: str
    
    # Create factory and resolver
    factory = BerryQLFactory()
    resolver = factory.create_berryql_resolver(
        strawberry_type=TestType,
        model_class=TestModel
    )
    
    # Verify resolver is callable
    assert callable(resolver)
    
    # Verify resolver signature (basic check)
    import inspect
    sig = inspect.signature(resolver)
    assert 'db' in sig.parameters
    assert 'info' in sig.parameters
