"""Prototype Berry declarative schema mirroring tests/schema.py using new DSL.

Not functionally equivalent yet: purely structural placeholder to evolve tests against.
"""
from __future__ import annotations
from typing import Optional, List, Any
from datetime import datetime
import strawberry
from .registry import BerrySchema, BerryType, field, relation, aggregate, count
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

@berry_schema.type(model=Post)
class PostQL(BerryType):
    id = field()
    title = field()
    content = field()
    author_id = field()
    created_at = field()
    author = relation('UserQL', single=True)
    post_comments = relation('PostCommentQL')
    post_comments_agg = count('post_comments')
    last_post_comment = aggregate('post_comments', ops=['last'])

@berry_schema.type(model=User)
class UserQL(BerryType):
    id = field()
    name = field()
    email = field()
    is_admin = field()
    created_at = field()
    posts = relation('PostQL')
    post_comments = relation('PostCommentQL')
    post_agg = count('posts')
    new_posts = relation('PostQL', window='recent')
    other_users = relation('UserQL', mode='exclude_self')
    bloggers = relation('UserQL', mode='has_posts')

# Build the Strawberry schema (prototype)
schema = berry_schema.to_strawberry()
