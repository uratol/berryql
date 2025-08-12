"""
GraphQL Schema for BerryQL integration tests.

This module contains the GraphQL types and Query schema used in the integration tests.
It demonstrates the enhanced @berryql.field decorator with various parameter mappings.
"""

import strawberry
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from berryql import berryql
from .models import User, Post, PostComment
from sqlalchemy import func, select


@strawberry.type
class PostCommentAggType:
    min_created_at: Optional[datetime] = None
    comments_count: int


@strawberry.type
class PostCommentType:
    id: int
    content: str
    rate: int
    post_id: int
    author_id: int
    created_at: datetime
    post: Optional['PostType'] = None
    author: Optional['UserType'] = None

@strawberry.type
class PostType:
    id: int
    title: str
    content: Optional[str] = None
    author_id: int
    created_at: datetime
    author: Optional['UserType'] = None
    
    @strawberry.field
    @berryql.field(
        rate_less_than={'rate': {'lt': lambda value: value}}
    )
    async def post_comments(self, 
                       info: strawberry.Info,
                       order_by: Optional[str] = 'rate',
                       rate_less_than: Optional[int] = None,
                       ) -> List[PostCommentType]:
        """Get post's comments using pre-resolved data."""
        pass
    
    @strawberry.field
    @berryql.custom_field(lambda model_class, info: (
        select(
            berryql.json_object(
                info,
                'min_created_at', func.min(PostComment.created_at),
                'comments_count', func.count(PostComment.id)
            )
        )
    .select_from(PostComment)
    .where(PostComment.post_id == model_class.id)
    ))
    async def post_comments_agg(self, info: strawberry.Info) -> Optional[PostCommentAggType]:
        """Get post's comments aggregation using pre-resolved data."""
        pass

    @strawberry.field
    @berryql.custom_field(lambda model_class, info: (
        select(
            berryql.json_object(
                info,
                'id', PostComment.id,
                'content', PostComment.content,
                'post_id', PostComment.post_id,
                'author_id', PostComment.author_id,
                'created_at', PostComment.created_at
            )
        )
    .select_from(PostComment)
    .where(PostComment.post_id == model_class.id)
    .order_by(PostComment.created_at.desc().nullslast(), PostComment.id.desc())
        .limit(1)
    ))
    async def last_post_comment(self, info: strawberry.Info) -> Optional[PostCommentType]:
        """Get post's last comment using pre-resolved data."""
        pass



@strawberry.type
class PostAggType:
    count: Optional[int] = None


@strawberry.type
class UserType:
    id: int
    name: str
    email: str
    is_admin: bool
    created_at: datetime
    
    @strawberry.field
    @berryql.field(
        content_filter={'content': {'ilike': lambda value: f'%{value}%'}},
    )
    async def posts(self, 
                    info: strawberry.Info,
                    content_filter: Optional[str] = None
                    ) -> List[PostType]:
        """Get user's posts using pre-resolved data with comments aggregation."""
        pass
    
    @strawberry.field
    @berryql.field
    async def post_comments(self, info: strawberry.Info) -> List[PostCommentType]:
        """Get user's comments using pre-resolved data."""
        pass
    
    @strawberry.field
    @berryql.custom_field(lambda model_class, info: (
        select(
            berryql.json_object(
                info,
                'count', func.count(Post.id)
            )
        )
        .select_from(Post)
        .where(Post.author_id == model_class.id)
    ))
    async def post_agg(self, info: strawberry.Info) -> Optional[PostAggType]:
        """Get user's post count using pre-resolved data."""
        pass
    
    @strawberry.field
    @berryql.field(
        model_class=Post,
        custom_where=lambda info=None: {'created_at': {'gt': (datetime.now(timezone.utc) - timedelta(hours=1)).replace(tzinfo=None).isoformat()}},
        custom_order='created_at desc'
    )
    async def new_posts(self, info: strawberry.Info) -> List[PostType]:
        """Get user's posts created within the last hour, sorted by created_at descending."""
        pass


def _get_user_filter(info):
    """Helper function to determine user filtering based on context."""
    if not info or not hasattr(info, 'context'):
        # No context means no access - use impossible condition
        return {'id': -1}
    
    context = info.context
    current_user = context.get('current_user')
    user_id = context.get('user_id')
    
    # Check if user is admin
    if current_user and getattr(current_user, 'is_admin', False):
        # Admins can see all users
        return {}
    
    # Non-admin users can only see themselves
    if user_id:
        return {'id': user_id}
    
    # No user context means no results - use impossible condition
    return {'id': -1}


# GraphQL Schema Setup
@strawberry.type
class Query:
    @strawberry.field
    @berryql.field(
        model_class=User,
        name_filter={'name': {'like': lambda value: f'%{value}%'}},
        custom_where=lambda info=None: _get_user_filter(info)
    )
    async def users(
        self,
        info: strawberry.Info,
        db: AsyncSession,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
    where: Optional[str] = None,
        name_filter: Optional[str] = None
    ) -> List[UserType]:
        """Get users with admin-based filtering and optional name filtering using @berryql.field decorator."""
        # The decorator handles the resolver creation and execution automatically
        # Admins can see all users, non-admins can only see themselves
        # The BerryQL resolver will process the name_filter parameter based on the mapping
        pass  # Implementation handled by the decorator
    
    @strawberry.field
    @berryql.field(
        model_class=User,
        custom_where={'email': {'ne': None}},  # Only users with email
        custom_order=['name', 'created_at desc']  # Default ordering
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

    @strawberry.field
    @berryql.field(
        model_class=User,
        custom_where=lambda info=None: (
            {'id': {'eq': info.context.get('user_id', 0)}} 
            if info and hasattr(info, 'context')
            else {'id': {'eq': None}}  # Return condition that matches no users when no user_id
        )
    )
    async def current_user(
        self, 
        info: strawberry.Info,
        db: AsyncSession
    ) -> Optional[UserType]:
        """Get the current user from context using BerryQL."""
        # Custom logic to prove that custom code executes
        user_id = info.context.get('user_id') if info.context else None
        if user_id == 999:
            raise ValueError("Custom logic executed: User 999 is forbidden!")
        return None  # Falls back to BerryQL resolver

    @strawberry.field
    @berryql.field(
        model_class=User,
        custom_where=lambda info=None: (
            {'id': {'ne': info.context.get('user_id', 0)}} 
            if info and hasattr(info, 'context') and info.context.get('user_id') 
            else {}
        )
    )
    async def other_users(
        self,
        info: strawberry.Info,
        db: AsyncSession,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[UserType]:
        """Get all users except the current user (using custom_where with context)."""
        pass  # Implementation handled by the decorator
    
    @strawberry.field
    @berryql.field(
        model_class=User,
        custom_where=lambda info=None: {
            'id': {
                'in': select(Post.author_id).distinct().scalar_subquery()
            }
        }
    )
    async def bloggers(
        self,
        info: strawberry.Info,
        db: AsyncSession,
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[UserType]:
        """Get users who have at least one post (bloggers)."""
        pass  # Implementation handled by the decorator
    


# Create the schema instance
schema = strawberry.Schema(query=Query)
