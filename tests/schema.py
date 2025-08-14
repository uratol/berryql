"""Prototype Berry declarative schema mirroring tests/schema.py using new DSL.

Not functionally equivalent yet: purely structural placeholder to evolve tests against.
"""
from __future__ import annotations
from typing import Optional, List, Any
from datetime import datetime
from sqlalchemy import select, func
import strawberry
from strawberry.types import Info
from berry import BerrySchema, BerryType, field, relation, aggregate, count, custom, custom_object
from tests.models import User, Post, PostComment  # type: ignore

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
    # Regular strawberry field with its own resolver
    @strawberry.field
    def content_preview(self) -> str | None:
        try:
            txt = getattr(self, 'content', None)
            if txt is None:
                m = getattr(self, '_model', None)
                txt = getattr(m, 'content', None) if m is not None else None
            if txt is None:
                return None
            s = str(txt)
            return s if len(s) <= 10 else s[:10] + '...'
        except Exception:
            return None

@berry_schema.type(model=Post)
class PostQL(BerryType):
    id = field()
    title = field()
    content = field()
    author_id = field()
    created_at = field()
    author = relation('UserQL', single=True, arguments={
        'name_ilike': lambda M, info, v: M.name.ilike(f"%{v}%"),
        'created_at_between': lambda M, info, v: (M.created_at.between(v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else None),
        'is_admin_eq': lambda M, info, v: M.is_admin == (bool(v) if isinstance(v, str) else v),
    })
    post_comments = relation('PostCommentQL', order_by='id')
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
    # Private relation (won't be exposed as GraphQL field) to fetch first comment
    _first_comment = relation('PostCommentQL', single=True)
    # Public computed field built on top of the private relation
    @strawberry.field
    async def first_comment_preview(self, info: Info) -> str | None:
        try:
            # Use the private relation resolver directly
            c = await self._first_comment(info)
            if c is None:
                return None
            txt = getattr(c, 'content', None)
            if txt is None:
                return None
            s = str(txt)
            return s if len(s) <= 10 else s[:10] + '...'
        except Exception:
            return None

@berry_schema.type(model=User)
class UserQL(BerryType):
    id = field()
    name = field()
    email = field()
    is_admin = field()
    created_at = field()
    # Regular strawberry field with its own resolver
    @strawberry.field
    def name_upper(self) -> str | None:
        try:
            n = getattr(self, 'name', None)
            if n is None:
                m = getattr(self, '_model', None)
                n = getattr(m, 'name', None) if m is not None else None
            return str(n).upper() if n is not None else None
        except Exception:
            return None
    posts = relation('PostQL', arguments={
        'title_ilike': lambda M, info, v: M.title.ilike(f"%{v}%"),
        'created_at_gt': lambda M, info, v: M.created_at > (datetime.fromisoformat(v) if isinstance(v, str) else v),
        'created_at_lt': lambda M, info, v: M.created_at < (datetime.fromisoformat(v) if isinstance(v, str) else v),
    })
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
    # Relation that returns only this user's posts that have comments
    def _posts_have_comments_where(model_cls, info):
        # Applied in non-pushdown path via callable
        from sqlalchemy import exists, select
        return exists(select(PostComment.id).where(PostComment.post_id == model_cls.id))
    posts_have_comments = relation('PostQL', where=_posts_have_comments_where)
    new_posts = relation('PostQL', window='recent')
    other_users = relation('UserQL', mode='exclude_self')
    bloggers = relation('UserQL', mode='has_posts')

# Declare Query with explicit roots only (no auto-roots)
@berry_schema.query()
class Query:
    # Plural collections
    # Move gating here: if enforce_user_gate and not admin, filter to current user; else no filter.
    def _gate_users(model_cls, info: Info):
        try:
            ctx = info.context or {}
            if not ctx.get('enforce_user_gate'):
                return {}
            cu = ctx.get('current_user')
            uid = ctx.get('user_id')
            if cu and getattr(cu, 'is_admin', False):
                return {}
            if uid:
                return {'id': {'eq': uid}}
            return {'id': {'eq': -1}}
        except Exception:
            return {'id': {'eq': -1}}
    users = relation('UserQL', order_by='id', order_dir='asc', where=_gate_users, arguments={
        'name_ilike': lambda M, info, v: M.name.ilike(f"%{v}%"),
        # Use column-based spec to expose proper GraphQL types (List[DateTime])
        'created_at_between': {
            'column': 'created_at',
            'op': 'between',
            'transform': lambda v: (
                [
                    (datetime.fromisoformat(v[0]) if isinstance(v[0], str) else v[0]),
                    (datetime.fromisoformat(v[1]) if isinstance(v[1], str) else v[1]),
                ] if isinstance(v, (list, tuple)) and len(v) >= 2 else v
            ),
        },
        # Boolean equality with proper GraphQL Boolean type
        'is_admin_eq': {
            'column': 'is_admin',
            'op': 'eq',
        },
    })
    posts = relation('PostQL', order_by='id', order_dir='asc', arguments={
        'title_ilike': lambda M, info, v: M.title.ilike(f"%{v}%"),
        'created_at_gt': lambda M, info, v: M.created_at > (datetime.fromisoformat(v) if isinstance(v, str) else v),
        'created_at_lt': lambda M, info, v: M.created_at < (datetime.fromisoformat(v) if isinstance(v, str) else v),
    })
    # Example: fetch a single user by id using arguments mapping
    userById = relation('UserQL', single=True, arguments={
        'id': {
            'column': 'id',
            'op': 'eq'
        }
    })

    # Regular strawberry field on Query
    @strawberry.field
    def hello(self) -> str:
        return "world"

schema = berry_schema.to_strawberry()
