"""
GraphQL Schema for BerryQL integration tests.

This module contains the GraphQL types and Query schema used in the integration tests.
It demonstrates the enhanced @berryql.field decorator with various parameter mappings.
"""

import strawberry
from typing import List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from berryql import BerryQLFactory, GraphQLQueryParams, berryql
from conftest import User, Post, Comment  # Import models from conftest


# Strawberry GraphQL Types
@strawberry.type
class CommentType:
    id: int
    content: str
    post_id: int
    author_id: int
    created_at: datetime


@strawberry.type
class PostType:
    id: int
    title: str
    content: Optional[str] = None
    author_id: int
    created_at: datetime
    
    @strawberry.field
    @berryql.field
    async def comments(self, info: strawberry.Info) -> List[CommentType]:
        """Get post's comments using pre-resolved data."""
        pass


@strawberry.type
class UserType:
    id: int
    name: str
    email: str
    created_at: datetime
    
    @strawberry.field
    @berryql.field
    async def posts(self, info: strawberry.Info) -> List[PostType]:
        """Get user's posts using pre-resolved data."""
        pass
    
    @strawberry.field
    @berryql.field
    async def comments(self, info: strawberry.Info) -> List[CommentType]:
        """Get user's comments using pre-resolved data."""
        pass


# GraphQL Schema Setup
@strawberry.type
class Query:
    @strawberry.field
    @berryql.field(
        model_class=User,
        name_filter={'name': {'like': lambda value: f'%{value}%'}}
    )
    async def users(
        self,
        info: strawberry.Info,
        db: AsyncSession,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        name_filter: Optional[str] = None
    ) -> List[UserType]:
        """Get users with optional filtering and pagination using @berryql.field decorator."""
        # The decorator handles the resolver creation and execution automatically
        # The BerryQL resolver will process the name_filter parameter based on the mapping
        pass  # Implementation handled by the decorator
    
    @strawberry.field
    @berryql.field(
        model_class=User,
        custom_where={UserType: {'email': {'ne': None}}},  # Only users with email
        custom_order={UserType: ['name', 'created_at desc']}  # Default ordering
    )
    async def active_users(
        self,
        info: strawberry.Info,
        db: AsyncSession,
        params: Optional[GraphQLQueryParams] = None
    ) -> List[UserType]:
        """Get active users with custom default conditions using @berryql.field decorator."""
        # The decorator automatically applies the custom_where and custom_order
        pass  # Implementation handled by the decorator


# Create the schema instance
schema = strawberry.Schema(query=Query)
