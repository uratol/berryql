"""Prototype Berry declarative schema mirroring tests/schema.py using new DSL.

Not functionally equivalent yet: purely structural placeholder to evolve tests against.
"""
from typing import Optional, List, Any, AsyncGenerator, Annotated
from datetime import datetime
from sqlalchemy import select, func, exists
import strawberry
from strawberry.types import Info
from berryql import BerrySchema, BerryType, BerryDomain, field, relation, count, custom, custom_object, domain, mutation, hooks, scope
from tests.models import User, Post, PostComment, PostCommentLike, View, GenericItem  # type: ignore
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

berry_schema = BerrySchema()

# --- Test-only: pre/post callback hooks & log ---
CALLBACK_EVENTS: list[dict] = []

def _test_pre_upsert(model_cls, info: Info, data: dict | None, ctx: dict | None = None):
    try:
        if not (getattr(info, 'context', None) or {}).get('test_callbacks'):
            return data
    except Exception:
        return data
    d = dict(data or {})
    d['title'] = f"[pre]{d.get('title','')}"
    try:
        CALLBACK_EVENTS.append({'event': 'pre', 'model': getattr(model_cls, '__name__', str(model_cls))})
    except Exception:
        pass
    return d

def _test_post_upsert(model_cls, info: Info, instance: Any, created: bool, ctx: dict | None = None):
    try:
        if (getattr(info, 'context', None) or {}).get('test_callbacks'):
            # mutate title to mark post-callback effect
            try:
                t = getattr(instance, 'title', None)
                if t is not None:
                    setattr(instance, 'title', f"{t}[post]")
            except Exception:
                pass
            try:
                CALLBACK_EVENTS.append({'event': 'post', 'model': getattr(model_cls, '__name__', str(model_cls)), 'created': bool(created)})
            except Exception:
                pass
    except Exception:
        return

# Async variants for callback testing
async def _test_pre_upsert_async(model_cls, info: Info, data: dict | None, ctx: dict | None = None):
    await asyncio.sleep(0)
    try:
        if not (getattr(info, 'context', None) or {}).get('test_callbacks_async'):
            return data
    except Exception:
        return data
    d = dict(data or {})
    d['title'] = f"[apre]{d.get('title','')}"
    try:
        CALLBACK_EVENTS.append({'event': 'apre', 'model': getattr(model_cls, '__name__', str(model_cls))})
    except Exception:
        pass
    return d

async def _test_post_upsert_async(model_cls, info: Info, instance: Any, created: bool, ctx: dict | None = None):
    await asyncio.sleep(0)
    try:
        if (getattr(info, 'context', None) or {}).get('test_callbacks_async'):
            try:
                t = getattr(instance, 'title', None)
                if t is not None:
                    setattr(instance, 'title', f"{t}[apost]")
            except Exception:
                pass
            try:
                CALLBACK_EVENTS.append({'event': 'apost', 'model': getattr(model_cls, '__name__', str(model_cls)), 'created': bool(created)})
            except Exception:
                pass
    except Exception:
        return

# Additional test-only hooks declared via HooksDescriptor (berry_schema.hooks)
# These mark the title with [hpre]/[hpost] and log events 'hpre'/'hpost'.
def _test_pre_upsert_hook(model_cls, info: Info, data: dict | None, ctx: dict | None = None):
    try:
        if not (getattr(info, 'context', None) or {}).get('test_callbacks'):
            return data
    except Exception:
        return data
    d = dict(data or {})
    d['title'] = f"[hpre]{d.get('title','')}"
    try:
        CALLBACK_EVENTS.append({'event': 'hpre', 'model': getattr(model_cls, '__name__', str(model_cls))})
    except Exception:
        pass
    return d

def _test_post_upsert_hook(model_cls, info: Info, instance: Any, created: bool, ctx: dict | None = None):
    try:
        if (getattr(info, 'context', None) or {}).get('test_callbacks'):
            try:
                t = getattr(instance, 'title', None)
                if t is not None:
                    setattr(instance, 'title', f"{t}[hpost]")
            except Exception:
                pass
            try:
                CALLBACK_EVENTS.append({'event': 'hpost', 'model': getattr(model_cls, '__name__', str(model_cls)), 'created': bool(created)})
            except Exception:
                pass
    except Exception:
        return

def _test_post_upsert_hook2(model_cls, info: Info, instance: Any, created: bool, ctx: dict | None = None):
    try:
        if (getattr(info, 'context', None) or {}).get('test_callbacks'):
            try:
                t = getattr(instance, 'title', None)
                if t is not None:
                    setattr(instance, 'title', f"{t}[h2post]")
            except Exception:
                pass
            try:
                CALLBACK_EVENTS.append({'event': 'h2post', 'model': getattr(model_cls, '__name__', str(model_cls)), 'created': bool(created)})
            except Exception:
                pass
    except Exception:
        return

@berry_schema.type(model=PostComment)
class PostCommentQL(BerryType):
    id = field()
    content = field()
    rate = field()
    post_id = field()
    author_id = field()
    # Write-only helper input to set author by email in pre-hooks
    author_email = field(write_only=True, comment="Write-only: resolve to author_id in pre-hook")
    created_at = field()
    post = relation('PostQL', single=True)
    author = relation('UserQL', single=True)
    # Force resolver fallback for nested path by using a callable where
    # (builders mark skip_pushdown when default_where is callable via meta['scope']).
    likes = relation('PostCommentLikeQL', scope=lambda M, info: (M.id > 0))
    like_count = count('likes')
    # Likes made by admin users only
    admin_likes = relation(
        'PostCommentLikeQL',
        # Use JSON where for pushdown-friendly filter: in fixtures, admin user has id=1
        scope='{"user_id": {"eq": 1}}'
    )
    # Polymorphic views for comments
    views = relation('ViewQL', fk_column_name='entity_id', scope='{"entity_type": {"eq": "post_comment"}}')
    # Regular strawberry field with its own resolver (preview of comment content)
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

    # Map write-only author_email to author_id for comments as well
    @berry_schema.pre
    async def _resolve_author_email_comment(model_cls, info: Info, data: dict | None, ctx: dict | None = None):
        try:
            if not isinstance(data, dict):
                return data
            email = data.get('author_email') or data.get('authorEmail')
            if not email:
                return data
            from tests.models import User as _User
            session: AsyncSession | None = info.context.get('db_session') if info and info.context else None
            if session is None:
                return data
            from sqlalchemy import select as _select
            res = await session.execute(_select(_User).where(_User.email == email))
            u = res.scalar_one_or_none()
            if u is not None:
                d = dict(data)
                try:
                    d['author_id'] = int(getattr(u, 'id'))
                except Exception:
                    d['author_id'] = getattr(u, 'id', None)
                d.pop('author_email', None)
                d.pop('authorEmail', None)
                return d
        except Exception:
            return data
        return data
@berry_schema.type(model=PostCommentLike)
class PostCommentLikeQL(BerryType):
    id = field()
    post_comment_id = field()
    user_id = field()
    created_at = field()
    user = relation('UserQL', single=True)
    post = relation('PostQL', single=True)

@berry_schema.type(model=View)
class ViewQL(BerryType):
    id = field()
    entity_type = field()
    entity_id = field()
    user_id = field()
    created_at = field()
    user = relation('UserQL', single=True)
    # Context-aware type-level scope: if context has only_view_user_id, filter Views by that user
    # Otherwise, return None (no-op)
    type_scope = scope(lambda M, info: (
        (lambda uid: ({"user_id": {"eq": int(uid)}} if uid is not None else None))(
            (getattr(info, 'context', None) or {}).get('only_view_user_id')
        )
    ))

@berry_schema.type(model=GenericItem)
class GenericItemQL(BerryType):
    id = field()
    name = field()
    code = field()
    count = field()
    active = field()
    created_at = field()

@berry_schema.type(model=Post)
class PostQL(BerryType):
    id = field()
    title = field()
    content = field()
    author_id = field()
    # Write-only helper input to set author by email in pre-hooks
    author_email = field(write_only=True, comment="Write-only: resolve to author_id in pre-hook")
    created_at = field(read_only=True)
    # Base64-encoded single binary blob across dialects
    binary_blob = field()
    # JSON/JSONB column exposed as Strawberry JSON scalar
    metadata_json = field()
    # New enum field projected from SQLAlchemy Enum
    status = field()
    # Read-only computed scalar mirroring SQLAlchemy column_property on Post
    content_length = field(read_only=True)
    author = relation('UserQL', single=True, arguments={
        'name_ilike': lambda M, info, v: M.name.ilike(f"%{v}%"),
        'created_at_between': lambda M, info, v: (M.created_at.between(v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else None),
        'is_admin_eq': lambda M, info, v: M.is_admin == (bool(v) if isinstance(v, str) else v),
    })
    post_comments = relation('PostCommentQL', order_by='created_at', order_dir='desc')
    # Demonstrate callable order_by using related record column via correlated subquery
    # Order PostComment rows by their author's created_at ascending
    post_comments_ordered_asc = relation(
        'PostCommentQL',
        order_by=lambda M, info: select(User.created_at).where(User.id == M.author_id).scalar_subquery(),
        order_dir='asc'
    )
    post_comments_agg = count('post_comments')
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
    # Polymorphic views for posts
    views = relation('ViewQL', fk_column_name='entity_id', scope='{"entity_type": {"eq": "post"}}')

    # Type-level mutation callbacks (moved from mutation() pre/post params)
    @berry_schema.pre
    def _merge_pre(model_cls, info: Info, data: dict | None, ctx: dict | None = None):
        return _test_pre_upsert(model_cls, info, data, ctx)

    @berry_schema.post
    def _merge_post(model_cls, info: Info, instance: Any, created: bool, ctx: dict | None = None):
        return _test_post_upsert(model_cls, info, instance, created, ctx)

    # Async variants for callback testing
    @berry_schema.pre
    async def _merge_pre_async(model_cls, info: Info, data: dict | None, ctx: dict | None = None):
        return await _test_pre_upsert_async(model_cls, info, data, ctx)

    @berry_schema.post
    async def _merge_post_async(model_cls, info: Info, instance: Any, created: bool, ctx: dict | None = None):
        return await _test_post_upsert_async(model_cls, info, instance, created, ctx)

    # Two ways to declare hooks on a BerryType:
    # 1) Decorators @berry_schema.pre/@berry_schema.post on methods above.
    #    These methods will be auto-registered as merge callbacks.
    # 2) Descriptor-based via berry_schema.hooks(...) which attaches functions (sync or async)
    #    directly without defining methods. This appends to the same callback lists and
    #    runs alongside decorators. Here we register extra test hooks adding [hpre]/[hpost].
    hooks = hooks(pre=_test_pre_upsert_hook, post=[_test_post_upsert_hook, _test_post_upsert_hook2])

    # Resolve write-only author_email to author_id before mutation
    @berry_schema.pre
    async def _resolve_author_email(model_cls, info: Info, data: dict | None, ctx: dict | None = None):
        try:
            if not isinstance(data, dict):
                return data
            email = data.get('author_email') or data.get('authorEmail')
            if not email:
                return data
            from tests.models import User as _User
            session: AsyncSession | None = info.context.get('db_session') if info and info.context else None
            if session is None:
                return data
            from sqlalchemy import select as _select
            res = await session.execute(_select(_User).where(_User.email == email))
            u = res.scalar_one_or_none()
            if u is not None:
                d = dict(data)
                try:
                    d['author_id'] = int(getattr(u, 'id'))
                except Exception:
                    d['author_id'] = getattr(u, 'id', None)
                # Optionally strip helper
                d.pop('author_email', None)
                d.pop('authorEmail', None)
                return d
        except Exception:
            return data
        return data

@berry_schema.type(model=User)
class UserQL(BerryType):
    id = field()
    name = field()
    email = field()
    is_admin = field()
    created_at = field(read_only=True)
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
    posts = relation('PostQL', order_by='created_at', order_dir='desc', arguments={
        'title_ilike': lambda M, info, v: M.title.ilike(f"%{v}%"),
        'created_at_gt': lambda M, info, v: M.created_at > (datetime.fromisoformat(v) if isinstance(v, str) else v),
        'created_at_lt': lambda M, info, v: M.created_at < (datetime.fromisoformat(v) if isinstance(v, str) else v),
    })
    # Test-only: default JSON where to verify SQL-level default WHERE pushdown
    posts_recent = relation('PostQL', scope='{"created_at": {"gt": "1900-01-01T00:00:00"}}')
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
    posts_have_comments = relation('PostQL', scope=_posts_have_comments_where)

# --- Domains: userDomain and blogDomain ---

@berry_schema.domain(name='userDomain')
class UserDomain(BerryDomain):
    # Reuse same relations as flat roots
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
    users = relation('UserQL', order_by='id', order_dir='asc', arguments={
        'name_ilike': lambda M, info, v: M.name.ilike(f"%{v}%"),
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
        'is_admin_eq': {
            'column': 'is_admin',
            'op': 'eq',
        },
    }, scope=_gate_users)
    userById = relation('UserQL', single=True, arguments={
        'id': {
            'column': 'id',
            'op': 'eq'
        }
    })

@berry_schema.domain(name='blogDomain')
class BlogDomain(BerryDomain):
    posts = relation('PostQL', order_by='id', order_dir='asc', arguments={
        'title_ilike': lambda M, info, v: M.title.ilike(f"%{v}%"),
        'created_at_gt': lambda M, info, v: M.created_at > (datetime.fromisoformat(v) if isinstance(v, str) else v),
        'created_at_lt': lambda M, info, v: M.created_at < (datetime.fromisoformat(v) if isinstance(v, str) else v),
    })
    # Regular Strawberry field on the domain container (should be exposed on schema)
    @strawberry.field
    def helloDomain(self) -> str:
        return "hello from blogDomain"
    # Strawberry field with Annotated + lazy return type; returns None for tests
    @strawberry.field
    def samplePostAnnotated(
        self,
        id: int | None = None,
        title: str | None = None,
    ) -> Optional[Annotated['PostQL', strawberry.lazy('tests.schema')]]:
        # This intentionally returns None; the purpose is to ensure the field is exposed
        # and the Annotated lazy type resolves without raising.
        return None
    # A regular public method that should NOT be exposed on Query domain nor Mutation domain
    def should_not_be_on_domain(self) -> str:
        return "nope"
    # Declare merge mutation for posts within the domain)
    merge_posts = mutation('PostQL', comment="Create or update posts (domain)")
    # Scoped domain-level mutation: only author_id == 1 allowed
    merge_posts_scoped = mutation('PostQL', scope='{"author_id": {"eq": 1}}')
    # Async builder for filter args should be awaited in root filters
    async def _created_at_gt_async(M, info, v):
        await asyncio.sleep(0)
        return M.created_at > (datetime.fromisoformat(v) if isinstance(v, str) else v)
    postsAsyncFilter = relation('PostQL', order_by='id', order_dir='asc', arguments={
        'created_at_gt': _created_at_gt_async,
    })
    # Domain-scoped mutation example (must be explicitly decorated to be exposed)
    @strawberry.mutation
    async def create_post_mut(self, info: Info, title: str, content: str, author_id: int) -> Annotated['PostQL', strawberry.lazy('tests.schema')]:
        session: AsyncSession | None = info.context.get('db_session') if info and info.context else None
        if session is None:
            raise ValueError("No db_session in context")
        p = Post(title=title, content=content, author_id=author_id)
        session.add(p)
        await session.flush()
        await session.commit()
        return berry_schema.from_model('PostQL', p)

    # Domain-level subscription example: yields simple integers for tests
    @strawberry.subscription
    async def new_post_event(self, to: int = 1) -> AsyncGenerator[int, None]:
        for i in range(1, max(1, int(to)) + 1):
            yield i
            await asyncio.sleep(0)

# A nested grouping domain that exposes userDomain and blogDomain inside
@berry_schema.domain(name='groupDomain')
class GroupDomain(BerryDomain):
    # Nest other domains inside this domain
    userDomain = domain(UserDomain)
    blogDomain = domain(BlogDomain)

# Domain solely for async callback tests
@berry_schema.domain(name='asyncDomain')
class AsyncDomain(BerryDomain):
    # Explicit async callbacks for merge
    merge_posts = mutation('PostQL', comment="Create or update posts (async domain)")

# Declare Query with explicit roots and grouped domains
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
    users = relation('UserQL', order_by='id', order_dir='asc', scope=_gate_users, arguments={
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
        # Return posts that have at least one comment authored by the given user id
        'commented_by': lambda M, info, v: exists(
            select(PostComment.id).where(
                PostComment.post_id == M.id,
                PostComment.author_id == (int(v) if isinstance(v, str) else v)
            )
        ),
    })
    genericItems = relation('GenericItemQL', order_by='name', order_dir='asc')
    # Async where callable support for roots
    async def _gate_users_async(model_cls, info: Info):
        # tiny await to ensure awaitable path is exercised
        await asyncio.sleep(0)
        try:
            ctx = info.context or {}
        except Exception:
            ctx = {}
        # Example filter: gate to users with id > 1 unless overriden
        if ctx.get('async_gate_return_none'):
            return None
        return {'id': {'gt': 1}}
    usersAsyncGate = relation('UserQL', order_by='id', order_dir='asc', scope=_gate_users_async)
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

    # A regular public method that should NOT be exposed as a Strawberry field
    def should_not_be_exposed(self) -> str:
        return "nope"

    # Expose domains under namespaces
    userDomain = domain(UserDomain)
    blogDomain = domain(BlogDomain)
    groupDomain = domain(GroupDomain)

schema = berry_schema.to_strawberry()

# --- Mutations and Subscriptions for tests ---

@berry_schema.mutation()
class Mutation:
    # Expose a domain group under Mutation as well
    userDomain = domain(UserDomain)
    blogDomain = domain(BlogDomain)
    groupDomain = domain(GroupDomain)
    # Expose async callback domain under Mutation
    asyncDomain = domain(AsyncDomain)

    # Top-level merge for posts at Query level)
    merge_posts = mutation('PostQL', comment="Create or update posts")
    # Single-payload variant: accepts a single PostQLInput instead of a list
    merge_post = mutation('PostQL', single=True, comment="Create or update a single post")
    # Scoped root-level mutation: only author_id == 1 allowed
    merge_posts_scoped = mutation('PostQL', scope='{"author_id": {"eq": 1}}', comment="Create or update posts (only author_id==1)")

    @strawberry.mutation
    async def create_post(self, info: Info, title: str, content: str, author_id: int) -> PostQL:
        session: AsyncSession | None = info.context.get('db_session') if info and info.context else None
        if session is None:
            raise ValueError("No db_session in context")
        p = Post(title=title, content=content, author_id=author_id)
        session.add(p)
        await session.flush()
        await session.commit()
        # Return full PostQL object
        return berry_schema.from_model('PostQL', p)

    # Also support a classic strawberry-decorated mutation returning just the id
    @strawberry.mutation
    async def create_post_id(self, info: Info, title: str, content: str, author_id: int) -> int:
        session: AsyncSession | None = info.context.get('db_session') if info and info.context else None
        if session is None:
            raise ValueError("No db_session in context")
        p = Post(title=title, content=content, author_id=author_id)
        session.add(p)
        await session.flush()
        await session.commit()
        return int(p.id)

    # A regular public method that should NOT be exposed as a Strawberry mutation
    async def should_not_be_mutation(self, info: Info) -> int:
        return 1

@berry_schema.subscription()
class Subscription:
    # Minimal native Strawberry subscription for demonstration/testing
    @strawberry.subscription
    async def tick(self, to: int = 1) -> AsyncGenerator[int, None]:
        # Yield numbers 1..to without any external pub/sub mechanism
        for i in range(1, max(1, int(to)) + 1):
            yield i
            # yield to event loop
            await asyncio.sleep(0)

# Rebuild schema to include query, mutation and subscription
schema = berry_schema.to_strawberry()
