"""Database fixtures for BerryQL tests (shared)."""

import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User, Post, PostComment


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
        Post(title="First Post", content="Hello world!", author_id=user1.id, created_at=now - timedelta(minutes=60)),
        Post(title="GraphQL is Great", content="I love GraphQL!", author_id=user1.id, created_at=now - timedelta(minutes=45)),
        Post(title="SQLAlchemy Tips", content="Some useful tips...", author_id=user2.id, created_at=now - timedelta(minutes=30)),
        Post(title="Python Best Practices", content="Here are some tips...", author_id=user2.id, created_at=now - timedelta(minutes=15)),
        Post(title="Getting Started", content="A beginner's guide", author_id=user3.id, created_at=now - timedelta(minutes=5)),
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


@pytest.fixture(scope="function")
async def sample_comments(db_session: AsyncSession, sample_users, sample_posts):
    return await create_sample_comments(db_session, sample_users, sample_posts)


async def seed_populated_db(session: AsyncSession):
    """Seed users, posts, and comments and return the same structure as populated_db."""
    users = await create_sample_users(session)
    posts = await create_sample_posts(session, users)
    comments = await create_sample_comments(session, users, posts)
    return {
        'users': users,
        'posts': posts,
        'post_comments': comments,
    }


@pytest.fixture(scope="function")
async def populated_db(sample_users, sample_posts, sample_comments):
    return {
        'users': sample_users,
        'posts': sample_posts,
        'post_comments': sample_comments
    }
