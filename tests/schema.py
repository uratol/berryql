"""
GraphQL Schema for BerryQL integration tests.

This module contains the GraphQL types and Query schema used in the integration tests.
It demonstrates the enhanced @berryql.field decorator with various parameter mappings.
"""

import strawberry
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from berryql import GraphQLQueryParams, berryql
from conftest import User, Post
from sqlalchemy import func, select


def build_post_count_query(model_class, requested_fields):
    """Build post count aggregation query."""
    return func.count(Post.id).filter(Post.author_id == model_class.id).label('post_count')


# Strawberry GraphQL Types
@strawberry.type
class PostAggType:
    post_count: int


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
    
    @strawberry.field
    @berryql.field
    async def post_count(self, info: strawberry.Info) -> int:
        """Get user's post count using pre-resolved data."""
        pass
    
    @strawberry.field
    @berryql.field(
        model_class=Post,
        custom_where={PostType: lambda: {'created_at': {'gt': (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()}}},
        custom_order={PostType: ['created_at desc']}
    )
    async def new_posts(self, info: strawberry.Info) -> List[PostType]:
        """Get user's posts created within the last hour, sorted by created_at descending."""
        pass


# GraphQL Schema Setup
@strawberry.type
class Query:
    @strawberry.field
    @berryql.field(
        model_class=User,
        name_filter={'name': {'like': lambda value: f'%{value}%'}},
        custom_fields={
            UserType: {
                'post_count': lambda model_class, requested_fields: func.coalesce(
                    select(func.count(Post.id))
                    .where(Post.author_id == model_class.id)
                    .scalar_subquery(),
                    0
                )
            }
        }
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
        where: Optional[str] = None,  # JSON string for additional where conditions
        order_by: Optional[str] = None,  # JSON string for order conditions
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[UserType]:
        """Get active users with custom default conditions using @berryql.field decorator."""
        # The decorator automatically applies the custom_where and custom_order
        # Additional where/order_by parameters will be processed by the decorator
        pass  # Implementation handled by the decorator


# Create the schema instance
schema = strawberry.Schema(query=Query)
