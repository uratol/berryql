"""
Input types for the generic resolver GraphQL fields.

This module defines Strawberry input types for filtering, ordering, and pagination
parameters used by the generic resolver factory. These replace the JSON string
approach with proper GraphQL input types for better type safety and developer experience.
"""

from typing import Optional, List, Union
import strawberry
from datetime import datetime
from uuid import UUID


@strawberry.input
class StringComparisonInput:
    """Input type for string field comparison operations."""
    eq: Optional[str] = None
    ne: Optional[str] = None
    gt: Optional[str] = None
    gte: Optional[str] = None
    lt: Optional[str] = None
    lte: Optional[str] = None
    like: Optional[str] = None
    ilike: Optional[str] = None
    in_: Optional[List[str]] = strawberry.field(name="in", default=None)


@strawberry.input
class IntComparisonInput:
    """Input type for integer field comparison operations."""
    eq: Optional[int] = None
    ne: Optional[int] = None
    gt: Optional[int] = None
    gte: Optional[int] = None
    lt: Optional[int] = None
    lte: Optional[int] = None
    in_: Optional[List[int]] = strawberry.field(name="in", default=None)


@strawberry.input
class FloatComparisonInput:
    """Input type for float field comparison operations."""
    eq: Optional[float] = None
    ne: Optional[float] = None
    gt: Optional[float] = None
    gte: Optional[float] = None
    lt: Optional[float] = None
    lte: Optional[float] = None
    in_: Optional[List[float]] = strawberry.field(name="in", default=None)


@strawberry.input
class DateTimeComparisonInput:
    """Input type for datetime field comparison operations."""
    eq: Optional[datetime] = None
    ne: Optional[datetime] = None
    gt: Optional[datetime] = None
    gte: Optional[datetime] = None
    lt: Optional[datetime] = None
    lte: Optional[datetime] = None
    in_: Optional[List[datetime]] = strawberry.field(name="in", default=None)


@strawberry.input
class UUIDComparisonInput:
    """Input type for UUID field comparison operations."""
    eq: Optional[UUID] = None
    ne: Optional[UUID] = None
    in_: Optional[List[UUID]] = strawberry.field(name="in", default=None)


@strawberry.input
class BoolComparisonInput:
    """Input type for boolean field comparison operations."""
    eq: Optional[bool] = None
    ne: Optional[bool] = None


@strawberry.input
class OrderByInput:
    """Input type for ordering specifications."""
    field: str
    direction: Optional[str] = "asc"  # "asc" or "desc"


@strawberry.input
class PaginationInput:
    """Input type for pagination parameters."""
    offset: Optional[int] = None
    limit: Optional[int] = None


# Generic comparison type that can handle any scalar type
ComparisonValue = Union[
    StringComparisonInput,
    IntComparisonInput, 
    FloatComparisonInput,
    DateTimeComparisonInput,
    UUIDComparisonInput,
    BoolComparisonInput,
    str,  # Direct string value for simple equality
    int,  # Direct int value for simple equality
    float,  # Direct float value for simple equality
    bool,  # Direct bool value for simple equality
    datetime,  # Direct datetime value for simple equality
    UUID,  # Direct UUID value for simple equality
]


# Export all input types
__all__ = [
    'StringComparisonInput',
    'IntComparisonInput',
    'FloatComparisonInput', 
    'DateTimeComparisonInput',
    'UUIDComparisonInput',
    'BoolComparisonInput',
    'OrderByInput',
    'PaginationInput',
    'ComparisonValue'
]
