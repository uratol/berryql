"""
BerryQL - A powerful GraphQL query optimization library for Strawberry GraphQL and SQLAlchemy.

This library provides advanced query building capabilities that eliminate N+1 problems
and optimize GraphQL queries through intelligent lateral joins and field analysis.
"""

from .factory import BerryQLFactory, GraphQLQueryParams, InvalidFieldError
from .query_analyzer import QueryFieldAnalyzer, query_analyzer
from .resolved_data_helper import (
    get_resolved_field_data,
    ResolvedDataMixin,
    berryql_field,
    custom_field,
    berryql
)
from .input_converter import (
    convert_comparison_input,
    convert_where_input,
    convert_order_by_input
)
from .input_types import (
    StringComparisonInput,
    IntComparisonInput,
    FloatComparisonInput,
    DateTimeComparisonInput,
    UUIDComparisonInput,
    BoolComparisonInput,
    OrderByInput,
    PaginationInput,
    ComparisonValue
)

__version__ = "0.1.0"
__author__ = "BerryQL Contributors"
__license__ = "MIT"

__all__ = [
    # Core factory and configuration
    "BerryQLFactory",
    "GraphQLQueryParams",
    "InvalidFieldError",
    
    # Query analysis
    "QueryFieldAnalyzer", 
    "query_analyzer",
    
    # Resolved data helpers
    "get_resolved_field_data",
    "ResolvedDataMixin",
    "berryql_field",
    "custom_field",
    "berryql",
    
    # Input conversion utilities
    "convert_comparison_input",
    "convert_where_input", 
    "convert_order_by_input",
    
    # Input types for GraphQL
    "StringComparisonInput",
    "IntComparisonInput",
    "FloatComparisonInput", 
    "DateTimeComparisonInput",
    "UUIDComparisonInput",
    "BoolComparisonInput",
    "OrderByInput",
    "PaginationInput",
    "ComparisonValue",
]
