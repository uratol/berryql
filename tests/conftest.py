"""Test configuration and fixtures for BerryQL."""

from dotenv import load_dotenv
import pytest
import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import AsyncGenerator
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship
import tempfile

# Try to load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def event_loop_policy():
    """Set event loop policy for Windows compatibility."""
    if sys.platform.startswith("win"):
        # Use SelectorEventLoop instead of ProactorEventLoop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    yield

class Base(DeclarativeBase):
    """Base class for test models."""
    pass


# Test models for integration tests
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    # Relationship to posts
    posts = relationship("Post", back_populates="author")
    # Relationship to comments
    comments = relationship("Comment", back_populates="author")


class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    # Relationship to user
    author = relationship("User", back_populates="posts")
    # Relationship to comments
    comments = relationship("Comment", back_populates="post")


class Comment(Base):
    __tablename__ = 'comments'
    
    id = Column(Integer, primary_key=True)
    content = Column(String(1000), nullable=False)
    post_id = Column(Integer, ForeignKey('posts.id'), nullable=False)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    # Relationships
    post = relationship("Post", back_populates="comments")
    author = relationship("User", back_populates="comments")


@pytest.fixture(scope="function")
async def engine():
    """Create a test database engine for each test function."""
    # Check for TEST_DATABASE_URL environment variable
    test_db_url = os.getenv('TEST_DATABASE_URL')
    
    if test_db_url:
        # Try to use the provided database URL
        try:
            if test_db_url.startswith("postgresql"):
                # PostgreSQL with asyncpg - use minimal connection args to avoid event loop issues
                engine = create_async_engine(
                    test_db_url,
                    echo=False,
                    future=True,
                    # Minimal configuration to avoid connection pool issues in tests
                    pool_size=1,
                    max_overflow=0,
                    pool_pre_ping=False,
                    pool_recycle=-1
                )
            else:
                # Other databases 
                engine = create_async_engine(
                    test_db_url,
                    echo=False,
                    future=True,
                    pool_size=5,
                    max_overflow=10,
                    pool_recycle=3600,
                    pool_pre_ping=True
                )
            
            # Test if we can create the engine (this will fail if driver is missing)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            is_external_db = True
            print(f"Using external database: {test_db_url}")
            
        except Exception as e:
            print(f"Failed to connect to external database ({test_db_url}): {e}")
            print("Falling back to SQLite in-memory database")
            # Fall back to SQLite
            engine = create_async_engine(
                "sqlite+aiosqlite:///:memory:",
                echo=True,  # Enable SQL logging for debugging with SQLite
                future=True
            )
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            is_external_db = False
    else:
        # Use in-memory SQLite for tests
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=True,  # Enable SQL logging for debugging with SQLite
            future=True
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        is_external_db = False
        print("Using SQLite in-memory database")
    
    yield engine
    
    # Clean up: for external databases, drop all tables
    if is_external_db:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        except Exception as e:
            print(f"Failed to clean up external database: {e}")
    
    await engine.dispose()


@pytest.fixture(scope="function")
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a new database session for each test function."""
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture(scope="function")
async def sample_users(db_session):
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
async def sample_posts(db_session, sample_users):
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
async def sample_comments(db_session, sample_users, sample_posts):
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
