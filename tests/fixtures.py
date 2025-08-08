"""Database fixtures for BerryQL tests."""

import pytest
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User, Post, Comment


@pytest.fixture(scope="function")
async def sample_users(db_session: AsyncSession):
    """Create sample users for testing for each test function."""
    users = [
        User(name="Alice Johnson", email="alice@example.com", is_admin=True),
        User(name="Bob Smith", email="bob@example.com", is_admin=False),
        User(name="Charlie Brown", email="charlie@example.com", is_admin=False),
    ]
    
    db_session.add_all(users)
    await db_session.commit()
    
    # Refresh to get IDs
    for user in users:
        await db_session.refresh(user)
    
    return users


@pytest.fixture(scope="function")
async def sample_posts(db_session: AsyncSession, sample_users):
    """Create sample posts for testing for each test function."""
    user1, user2, user3 = sample_users
    
    posts = [
        Post(title="First Post", content="Hello world!", author_id=user1.id),
        Post(title="GraphQL is Great", content="I love GraphQL!", author_id=user1.id),
        Post(title="SQLAlchemy Tips", content="Some useful tips...", author_id=user2.id),
        Post(title="Python Best Practices", content="Here are some tips...", author_id=user2.id),
        Post(title="Getting Started", content="A beginner's guide", author_id=user3.id),
    ]
    
    db_session.add_all(posts)
    await db_session.commit()
    
    # Refresh to get IDs
    for post in posts:
        await db_session.refresh(post)
    
    return posts


@pytest.fixture(scope="function")
async def sample_comments(db_session: AsyncSession, sample_users, sample_posts):
    """Create sample comments for testing for each test function."""
    user1, user2, user3 = sample_users
    post1, post2, post3, post4, post5 = sample_posts
    
    comments = [
        Comment(content="Great post!", post_id=post1.id, author_id=user2.id),
        Comment(content="Thanks for sharing!", post_id=post1.id, author_id=user3.id),
        Comment(content="I agree completely!", post_id=post2.id, author_id=user2.id),
        Comment(content="Very helpful tips", post_id=post3.id, author_id=user1.id),
        Comment(content="Nice work!", post_id=post3.id, author_id=user3.id),
        Comment(content="Looking forward to more", post_id=post4.id, author_id=user1.id),
        Comment(content="This helped me a lot", post_id=post5.id, author_id=user2.id),
    ]
    
    db_session.add_all(comments)
    await db_session.commit()
    
    # Refresh to get IDs
    for comment in comments:
        await db_session.refresh(comment)
    
    return comments


@pytest.fixture(scope="function")
async def populated_db(sample_users, sample_posts, sample_comments):
    """Fixture that ensures database is populated with sample data for each test function."""
    return {
        'users': sample_users,
        'posts': sample_posts,
        'comments': sample_comments
    }
