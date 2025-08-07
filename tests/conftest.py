"""Test configuration for BerryQL."""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship
import tempfile
import os


class Base(DeclarativeBase):
    """Base class for test models."""
    pass


# Test models for integration tests
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationship to posts
    posts = relationship("Post", back_populates="author")


class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationship to user
    author = relationship("User", back_populates="posts")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def engine():
    """Create a test database engine for the entire test session."""
    # Use in-memory SQLite for tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=True,  # Enable SQL logging for debugging
        future=True
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    await engine.dispose()


@pytest.fixture(scope="session")
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a single database session for all tests."""
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture(scope="session")
async def sample_users(db_session):
    """Create sample users for testing once for all tests."""
    users = [
        User(name="Alice Johnson", email="alice@example.com"),
        User(name="Bob Smith", email="bob@example.com"),
        User(name="Charlie Brown", email="charlie@example.com"),
    ]
    
    db_session.add_all(users)
    await db_session.commit()
    
    # Refresh to get IDs
    for user in users:
        await db_session.refresh(user)
    
    return users


@pytest.fixture(scope="session")
async def sample_posts(db_session, sample_users):
    """Create sample posts for testing once for all tests."""
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


@pytest.fixture(scope="session")
async def populated_db(sample_users, sample_posts):
    """Fixture that ensures database is populated with sample data once for all tests."""
    return {
        'users': sample_users,
        'posts': sample_posts
    }
