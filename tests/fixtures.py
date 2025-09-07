"""Database fixtures for BerryQL tests (shared)."""

import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User, Post, PostComment, PostCommentLike, View, GenericItem, PostStatus


async def create_sample_users(session: AsyncSession):
    """Create and commit the sample users used across tests and demos."""
    users = [
        User(name="Alice Johnson", email="alice@example.com", is_admin=True),
        User(name="Bob Smith", email="bob@example.com", is_admin=False),
        User(name="Charlie Brown", email="charlie@example.com", is_admin=False),
        User(name="Dave NoPosts", email="dave@example.com", is_admin=False),
    ]
    session.add_all(users)
    await session.flush()
    await session.commit()
    return users


@pytest.fixture(scope="function")
async def sample_users(db_session: AsyncSession):
    return await create_sample_users(db_session)


async def create_sample_posts(session: AsyncSession, users):
    """Create and commit the sample posts with deterministic timestamps."""
    user1, user2, user3, _ = users
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    posts = [
        Post(
            title="First Post",
            content="Hello world!",
            author_id=user1.id,
            created_at=now - timedelta(minutes=60),
            binary_blob=b"a",
            status=PostStatus.PUBLISHED,
            metadata_json={"tags": ["intro", "hello"], "views": 10}
        ),
        Post(
            title="GraphQL is Great",
            content="I love GraphQL!",
            author_id=user1.id,
            created_at=now - timedelta(minutes=45),
            binary_blob=b"x",
            status=PostStatus.PUBLISHED,
            metadata_json={"tags": ["graphql"], "rating": 4.5}
        ),
        Post(
            title="SQLAlchemy Tips",
            content="Some useful tips...",
            author_id=user2.id,
            created_at=now - timedelta(minutes=30),
            binary_blob=None,
            status=PostStatus.DRAFT,
            metadata_json=None
        ),
        Post(
            title="Python Best Practices",
            content="Here are some tips...",
            author_id=user2.id,
            created_at=now - timedelta(minutes=15),
            binary_blob=b"\x00\x01\x02",
            status=PostStatus.PUBLISHED,
            metadata_json={"flags": {"featured": True}, "count": 3}
        ),
        Post(
            title="Getting Started",
            content="A beginner's guide",
            author_id=user3.id,
            created_at=now - timedelta(minutes=5),
            binary_blob=None,
            status=PostStatus.ARCHIVED,
            metadata_json={}
        ),
    ]
    session.add_all(posts)
    await session.flush()
    await session.commit()
    return posts


@pytest.fixture(scope="function")
async def sample_posts(db_session: AsyncSession, sample_users):
    return await create_sample_posts(db_session, sample_users)


async def create_sample_comments(session: AsyncSession, users, posts):
    """Create and commit the sample comments."""
    user1, user2, user3, _ = users
    post1, post2, post3, post4, post5 = posts
    post_comments = [
        PostComment(content="Great post!", post_id=post1.id, author_id=user2.id, rate=2),
        PostComment(content="Thanks for sharing!", post_id=post1.id, author_id=user3.id, rate=1),
        PostComment(content="I agree completely!", post_id=post2.id, author_id=user2.id, rate=3),
        PostComment(content="Very helpful tips", post_id=post3.id, author_id=user1.id, rate=1),
        PostComment(content="Nice work!", post_id=post3.id, author_id=user3.id, rate=2),
        PostComment(content="Looking forward to more", post_id=post4.id, author_id=user1.id, rate=1),
        PostComment(content="This helped me a lot", post_id=post5.id, author_id=user2.id, rate=5),
    ]
    session.add_all(post_comments)
    await session.flush()
    await session.commit()
    return post_comments
async def create_sample_likes(session: AsyncSession, users, comments):
    """Create and commit sample likes for comments."""
    user1, user2, user3, _ = users
    likes = [
        PostCommentLike(post_comment_id=comments[0].id, user_id=user1.id),
        PostCommentLike(post_comment_id=comments[0].id, user_id=user3.id),
        PostCommentLike(post_comment_id=comments[1].id, user_id=user1.id),
        PostCommentLike(post_comment_id=comments[2].id, user_id=user3.id),
    ]
    session.add_all(likes)
    await session.flush()
    await session.commit()
    return likes


@pytest.fixture(scope="function")
async def sample_likes(db_session: AsyncSession, sample_comments, sample_users):
    return await create_sample_likes(db_session, sample_users, sample_comments)



@pytest.fixture(scope="function")
async def sample_comments(db_session: AsyncSession, sample_users, sample_posts):
    return await create_sample_comments(db_session, sample_users, sample_posts)


async def seed_populated_db(session: AsyncSession):
    """Seed users, posts, and comments and return the same structure as populated_db."""
    users = await create_sample_users(session)
    posts = await create_sample_posts(session, users)
    comments = await create_sample_comments(session, users, posts)
    likes = await create_sample_likes(session, users, comments)
    views = await create_sample_views(session, users, posts, comments)
    # Seed Generic items
    import uuid as _uuid
    items = [
        GenericItem(id=_uuid.uuid4(), name='A', code='A1', count=1, active=True),
        GenericItem(id=_uuid.uuid4(), name='B', code='B2', count=2, active=False),
        GenericItem(id=_uuid.uuid4(), name='C', code='C3', count=3, active=True),
    ]
    session.add_all(items)
    await session.flush()
    await session.commit()
    return {
        'users': users,
        'posts': posts,
        'post_comments': comments,
        'post_comment_likes': likes,
        'views': views,
    'generic_items': items,
    }


async def create_sample_views(session: AsyncSession, users, posts, comments):
    """Create and commit sample polymorphic views for posts and comments."""
    user1, user2, user3, _ = users
    post1, post2, post3, post4, post5 = posts
    views = [
        View(entity_type='post', entity_id=post1.id, user_id=user2.id),
        View(entity_type='post', entity_id=post1.id, user_id=user3.id),
        View(entity_type='post', entity_id=post2.id, user_id=user3.id),
        View(entity_type='post_comment', entity_id=comments[0].id, user_id=user1.id),
        View(entity_type='post_comment', entity_id=comments[1].id, user_id=user1.id),
    ]
    session.add_all(views)
    await session.flush()
    await session.commit()
    return views


@pytest.fixture(scope="function")
async def sample_views(db_session: AsyncSession, sample_users, sample_posts, sample_comments):
    return await create_sample_views(db_session, sample_users, sample_posts, sample_comments)


async def create_sample_generic_items(session: AsyncSession):
    import uuid as _uuid
    items = [
        GenericItem(id=_uuid.uuid4(), name='A', code='A1', count=1, active=True),
        GenericItem(id=_uuid.uuid4(), name='B', code='B2', count=2, active=False),
        GenericItem(id=_uuid.uuid4(), name='C', code='C3', count=3, active=True),
    ]
    session.add_all(items)
    await session.flush()
    await session.commit()
    return items


@pytest.fixture(scope="function")
async def sample_generic_items(db_session: AsyncSession):
    return await create_sample_generic_items(db_session)


@pytest.fixture(scope="function")
async def populated_db(sample_users, sample_posts, sample_comments, sample_likes, sample_views, sample_generic_items):
    return {
        'users': sample_users,
        'posts': sample_posts,
        'post_comments': sample_comments,
        'post_comment_likes': sample_likes,
    'views': sample_views,
    'generic_items': sample_generic_items,
    }
