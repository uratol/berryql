"""Test configuration and fixtures for BerryQL."""

from dotenv import load_dotenv
import pytest
import asyncio
import os
import sys
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from tests.common.models import Base

# Try to load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def event_loop_policy():
    """Set event loop policy for Windows compatibility."""
    if sys.platform.startswith("win"):
        # Use SelectorEventLoop instead of ProactorEventLoop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    yield


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
            
            # Ensure a clean slate before tests: drop then create all tables
            # This also validates the connection/driver early.
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
            
            is_external_db = True
            print(f"Using external database: {test_db_url}")
            
        except Exception as e:
            print(f"Failed to connect to external database ({test_db_url}): {e}")
            raise  # Re-raise the exception to fail the test
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


# Import fixtures from fixtures module
from tests.common.fixtures import sample_users, sample_posts, sample_comments, populated_db
