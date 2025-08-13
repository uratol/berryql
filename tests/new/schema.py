"""Prototype Berry declarative schema mirroring tests/schema.py using new DSL.

Not functionally equivalent yet: purely structural placeholder to evolve tests against.
"""
from __future__ import annotations
from typing import Optional, List, Any
from datetime import datetime
from sqlalchemy import select, func
import strawberry
from berry.registry import BerrySchema, BerryType, field, relation, aggregate, count, custom, custom_object
from tests.common.models import User, Post, PostComment  # type: ignore

berry_schema = BerrySchema()

@berry_schema.type(model=PostComment)
class PostCommentQL(BerryType):
    id = field()
    content = field()
    rate = field()
    post_id = field()
    author_id = field()
    created_at = field()
    post = relation('PostQL', single=True)
    author = relation('UserQL', single=True)

@berry_schema.type(model=Post)
class PostQL(BerryType):
    # Declare filters for autogeneration (Phase 2 tests)
    __filters__ = {
        'title_ilike': {'column': 'title', 'op': 'ilike', 'transform': lambda v: f"%{v}%"},
    'created_at': {'column': 'created_at', 'ops': ['gt','lt']},  # expands to created_at_gt / created_at_lt
    }
    id = field()
    title = field()
    content = field()
    author_id = field()
    created_at = field()
    author = relation('UserQL', single=True)
    post_comments = relation('PostCommentQL')
    post_comments_agg = count('post_comments')
    last_post_comment = aggregate('post_comments', ops=['last'])
    # Demonstration custom field: total length of all comment contents for the post
    def _comment_text_len_builder(model_cls):
        return (
            select(
                func.coalesce(func.sum(func.length(PostComment.content)), 0).label('comment_text_len')
            )
            .select_from(PostComment)
            .where(PostComment.post_id == model_cls.id)
        )
    comment_text_len = custom(_comment_text_len_builder, returns=int)
    # Multi-column aggregate object (min_created_at, comments_count)
    post_comments_agg_obj = custom_object(
        lambda model_cls: (
            select(
                func.min(PostComment.created_at).label('min_created_at'),
                func.count(PostComment.id).label('comments_count')
            ).select_from(PostComment).where(PostComment.post_id == model_cls.id)
        ),
        returns={'min_created_at': datetime, 'comments_count': int}
    )

@berry_schema.type(model=User)
class UserQL(BerryType):
    __filters__ = {
        'name_ilike': {'column': 'name', 'op': 'ilike', 'transform': lambda v: f"%{v}%"},
        'created_at_between': {'column': 'created_at', 'op': 'between'},
        'is_admin_eq': {'column': 'is_admin', 'op': 'eq'},
    }
    id = field()
    name = field()
    email = field()
    is_admin = field()
    created_at = field()
    posts = relation('PostQL')
    post_comments = relation('PostCommentQL')
    post_agg = count('posts')
    # Object form of post aggregation (mirrors legacy PostAggType { count })
    post_agg_obj = custom_object(
        lambda model_cls: (
            select(
                func.count(Post.id).label('count')
            ).select_from(Post).where(Post.author_id == model_cls.id)
        ),
        returns={'count': int}
    )
    new_posts = relation('PostQL', window='recent')
    other_users = relation('UserQL', mode='exclude_self')
    bloggers = relation('UserQL', mode='has_posts')

# Build the Strawberry schema (prototype)
schema = berry_schema.to_strawberry()
