"""Database fixtures for BerryQL tests."""

import pytest
from typing import AsyncGenerator
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User, Post, PostComment


@pytest.fixture(scope="function")
async def sample_users(db_session: AsyncSession):
    """Create sample users for testing for each test function."""
    users = [
        User(name="Alice Johnson", email="alice@example.com", is_admin=True),
        User(name="Bob Smith", email="bob@example.com", is_admin=False),
        User(name="Charlie Brown", email="charlie@example.com", is_admin=False),
        # User with no posts
        User(name="Dave NoPosts", email="dave@example.com", is_admin=False),
    ]
    
    db_session.add_all(users)
    # Flush to persist and populate primary keys without issuing per-row refreshes
    await db_session.flush()
    # Commit once after IDs are populated
    await db_session.commit()
    
    return users


@pytest.fixture(scope="function")
async def sample_posts(db_session: AsyncSession, sample_users):
    """Create sample posts for testing for each test function."""
    # Use the first three users to create posts; the 4th (Dave) has no posts
    user1 = sample_users[0]
    user2 = sample_users[1]
    user3 = sample_users[2]
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    posts = [
        # Stagger Alice's posts to have deterministic created_at ordering
        Post(
            title="First Post",
            content="Hello world!",
            author_id=user1.id,
            created_at=now - timedelta(minutes=45),
        ),
        Post(
            title="GraphQL is Great",
            content="I love GraphQL!",
            author_id=user1.id,
            created_at=now - timedelta(minutes=15),
        ),
        Post(title="SQLAlchemy Tips", content="Some useful tips...", author_id=user2.id),
        Post(title="Python Best Practices", content="Here are some tips...", author_id=user2.id),
        Post(title="Getting Started", content="A beginner's guide", author_id=user3.id),
    ]
    
    db_session.add_all(posts)
    # Flush to persist and populate primary keys without issuing per-row refreshes
    await db_session.flush()
    # Commit once after IDs are populated
    await db_session.commit()
    
    return posts


@pytest.fixture(scope="function")
async def sample_comments(db_session: AsyncSession, sample_users, sample_posts):
    """Create sample comments for testing for each test function."""
    # Use the first three users for comments
    user1 = sample_users[0]
    user2 = sample_users[1]
    user3 = sample_users[2]
    post1, post2, post3, post4, post5 = sample_posts
    
    post_comments = [
        # Assign rates so that default ordering by rate asc is deterministic
    PostComment(content="Great post!", post_id=post1.id, author_id=user2.id, rate=2),
    PostComment(content="Thanks for sharing!", post_id=post1.id, author_id=user3.id, rate=1),
    PostComment(content="I agree completely!", post_id=post2.id, author_id=user2.id, rate=3),
    PostComment(content="Very helpful tips", post_id=post3.id, author_id=user1.id, rate=1),
    PostComment(content="Nice work!", post_id=post3.id, author_id=user3.id, rate=2),
    PostComment(content="Looking forward to more", post_id=post4.id, author_id=user1.id, rate=1),
    PostComment(content="This helped me a lot", post_id=post5.id, author_id=user2.id, rate=5),
    ]
    
    db_session.add_all(post_comments)
    # Flush to persist and populate primary keys without issuing per-row refreshes
    await db_session.flush()
    # Commit once after IDs are populated
    await db_session.commit()
    
    return post_comments


@pytest.fixture(scope="function")
async def populated_db(sample_users, sample_posts, sample_comments):
    """Fixture that ensures database is populated with sample data for each test function."""
    return {
        'users': sample_users,
        'posts': sample_posts,
        'post_comments': sample_comments
    }
