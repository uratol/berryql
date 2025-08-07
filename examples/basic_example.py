"""
Basic example of using BerryQL with Strawberry GraphQL and SQLAlchemy.

This example demonstrates:
- Setting up models and GraphQL types
- Creating optimized resolvers with BerryQL
- Using query parameters for filtering and pagination
- Handling relationships with automatic N+1 elimination
"""

import asyncio
import strawberry
from typing import List, Optional
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship

from berryql import BerryQLFactory, GraphQLQueryParams, get_resolved_field_data, berryql


# SQLAlchemy Models
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to posts
    posts = relationship("Post", back_populates="author")


class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to user
    author = relationship("User", back_populates="posts")


# Strawberry GraphQL Types
@strawberry.type
class PostType:
    id: int
    title: str
    content: Optional[str]
    author_id: int
    created_at: datetime


@strawberry.type
class UserType:
    id: int
    name: str
    email: str
    created_at: datetime
    
    @strawberry.field
    @berryql.field
    async def posts(self, info: strawberry.Info) -> List[PostType]:
        """Get user's posts using pre-resolved data."""
        pass


# GraphQL Schema Setup
@strawberry.type
class Query:
    @strawberry.field
    async def users(
        self,
        info: strawberry.Info,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        name_filter: Optional[str] = strawberry.field(name="nameFilter", default=None)
    ) -> List[UserType]:
        """Get users with optional filtering and pagination."""
        # Build query parameters
        where_conditions = {}
        if name_filter:
            where_conditions['name'] = {'like': f'%{name_filter}%'}
        
        params = GraphQLQueryParams(
            where=where_conditions,
            limit=limit,
            offset=offset,
            order_by=[{'field': 'created_at', 'direction': 'desc'}]
        )
        
        # Use the optimized resolver
        return await user_resolver(db=get_db_session(), info=info, params=params)
    
    @strawberry.field
    async def posts(
        self,
        info: strawberry.Info,
        limit: Optional[int] = None,
        author_id: Optional[int] = None
    ) -> List[PostType]:
        """Get posts with optional filtering."""
        where_conditions = {}
        if author_id:
            where_conditions['author_id'] = {'eq': author_id}
        
        params = GraphQLQueryParams(
            where=where_conditions,
            limit=limit,
            order_by=[{'field': 'created_at', 'direction': 'desc'}]
        )
        
        return await post_resolver(db=get_db_session(), info=info, params=params)


# Create BerryQL Factory and Resolvers
factory = BerryQLFactory()

# Create optimized resolvers
user_resolver = factory.create_berryql_resolver(
    strawberry_type=UserType,
    model_class=User
)

post_resolver = factory.create_berryql_resolver(
    strawberry_type=PostType,
    model_class=Post
)

# Database setup (for demo purposes)
async_engine = None
async_session = None


async def setup_database():
    """Setup test database with sample data."""
    global async_engine, async_session
    
    # Create async engine (using SQLite for demo)
    async_engine = create_async_engine(
        "sqlite+aiosqlite:///./example.db",
        echo=True  # Enable SQL logging
    )
    
    # Create session factory
    async_session = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    # Drop and recreate tables
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    # Add sample data
    async with async_session() as session:
        # Create users
        user1 = User(name="Alice Johnson", email="alice@example.com")
        user2 = User(name="Bob Smith", email="bob@example.com")
        user3 = User(name="Charlie Brown", email="charlie@example.com")
        
        session.add_all([user1, user2, user3])
        await session.commit()
        
        # Create posts
        posts = [
            Post(title="First Post", content="Hello world!", author_id=user1.id),
            Post(title="GraphQL is Great", content="I love GraphQL!", author_id=user1.id),
            Post(title="SQLAlchemy Tips", content="Some useful tips...", author_id=user2.id),
            Post(title="Python Best Practices", content="Here are some tips...", author_id=user2.id),
            Post(title="Getting Started", content="A beginner's guide", author_id=user3.id),
        ]
        
        session.add_all(posts)
        await session.commit()


def get_db_session():
    """Get database session for resolvers."""
    # In a real application, this would be properly injected
    # This is simplified for demo purposes
    return async_session()


# Create GraphQL schema
schema = strawberry.Schema(query=Query)


async def main():
    """Main demo function."""
    # Setup database
    await setup_database()
    
    # Example GraphQL query
    query = """
    query {
        users(limit: 2, nameFilter: "Alice") {
            id
            name
            email
            posts {
                id
                title
                content
            }
        }
    }
    """
    
    # Execute query
    result = await schema.execute(query)
    
    if result.errors:
        print("Errors:", result.errors)
    else:
        print("Result:", result.data)
    
    # Cleanup
    if async_engine:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
