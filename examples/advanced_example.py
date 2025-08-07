"""
Advanced example showing BerryQL's custom fields and complex filtering.

This example demonstrates:
- Custom fields with computed values
- Advanced filtering with GraphQL input types
- Complex relationships and nested queries
- Using decorators for resolved data access
"""

import asyncio
import strawberry
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship

from berryql import (
    BerryQLFactory, 
    GraphQLQueryParams, 
    custom_field, 
    berryql_field,
    convert_where_input
)
from berryql.input_types import StringComparisonInput, IntComparisonInput, DateTimeComparisonInput


# SQLAlchemy Models
class Base(DeclarativeBase):
    pass


class Category(Base):
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    
    # Relationship to posts
    posts = relationship("Post", back_populates="category")


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
    category_id = Column(Integer, ForeignKey('categories.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    view_count = Column(Integer, default=0)
    
    # Relationships
    author = relationship("User", back_populates="posts")
    category = relationship("Category", back_populates="posts")


# GraphQL Input Types for Advanced Filtering
@strawberry.input
class PostWhereInput:
    """Advanced filtering input for posts."""
    title: Optional[StringComparisonInput] = None
    content: Optional[StringComparisonInput] = None
    author_id: Optional[IntComparisonInput] = None
    category_id: Optional[IntComparisonInput] = None
    created_at: Optional[DateTimeComparisonInput] = None
    view_count: Optional[IntComparisonInput] = None


@strawberry.input
class UserWhereInput:
    """Advanced filtering input for users."""
    name: Optional[StringComparisonInput] = None
    email: Optional[StringComparisonInput] = None
    created_at: Optional[DateTimeComparisonInput] = None


# Custom Field Query Builders
def build_post_count_query(model_class, requested_fields):
    """Build query for post count custom field."""
    return func.count(Post.id).filter(Post.author_id == model_class.id).label('post_count')


def build_recent_post_count_query(model_class, requested_fields):
    """Build query for recent post count (last 30 days)."""
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    return func.count(Post.id).filter(
        Post.author_id == model_class.id,
        Post.created_at >= thirty_days_ago
    ).label('recent_post_count')


def build_avg_views_query(model_class, requested_fields):
    """Build query for average views per post."""
    return func.avg(Post.view_count).filter(Post.author_id == model_class.id).label('avg_views')


# Strawberry GraphQL Types
@strawberry.type
class CategoryType:
    id: int
    name: str
    description: Optional[str]
    
    @strawberry.field
    @berryql_field
    async def posts(self, info: strawberry.Info) -> List["PostType"]:
        """Get category posts using resolved data."""
        pass  # Implementation handled by decorator


@strawberry.type
class PostType:
    id: int
    title: str
    content: Optional[str]
    author_id: int
    category_id: Optional[int]
    created_at: datetime
    view_count: int
    
    @strawberry.field
    @berryql_field
    async def author(self, info: strawberry.Info) -> Optional["UserType"]:
        """Get post author using resolved data."""
        pass
    
    @strawberry.field
    @berryql_field
    async def category(self, info: strawberry.Info) -> Optional[CategoryType]:
        """Get post category using resolved data."""
        pass


@strawberry.type
class UserType:
    id: int
    name: str
    email: str
    created_at: datetime
    
    @strawberry.field
    @berryql_field
    async def posts(self, info: strawberry.Info) -> List[PostType]:
        """Get user's posts using resolved data."""
        pass
    
    @strawberry.field
    @custom_field
    def post_count(self) -> int:
        """Total number of posts by this user."""
        pass  # Implementation provided by custom field query builder
    
    @strawberry.field
    @custom_field
    def recent_post_count(self) -> int:
        """Number of posts in the last 30 days."""
        pass
    
    @strawberry.field
    @custom_field
    def avg_views(self) -> Optional[float]:
        """Average views per post."""
        pass


# GraphQL Schema
@strawberry.type
class Query:
    @strawberry.field
    async def users(
        self,
        info: strawberry.Info,
        where: Optional[UserWhereInput] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[UserType]:
        """Get users with advanced filtering."""
        # Convert GraphQL input to berryql format
        where_dict = convert_where_input(where.__dict__ if where else {})
        
        params = GraphQLQueryParams(
            where=where_dict,
            limit=limit,
            offset=offset,
            order_by=[{'field': 'created_at', 'direction': 'desc'}]
        )
        
        return await user_resolver(db=get_db_session(), info=info, params=params)
    
    @strawberry.field
    async def posts(
        self,
        info: strawberry.Info,
        where: Optional[PostWhereInput] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[PostType]:
        """Get posts with advanced filtering."""
        where_dict = convert_where_input(where.__dict__ if where else {})
        
        params = GraphQLQueryParams(
            where=where_dict,
            limit=limit,
            offset=offset,
            order_by=[{'field': 'created_at', 'direction': 'desc'}]
        )
        
        return await post_resolver(db=get_db_session(), info=info, params=params)
    
    @strawberry.field
    async def categories(
        self,
        info: strawberry.Info,
        limit: Optional[int] = None
    ) -> List[CategoryType]:
        """Get all categories."""
        params = GraphQLQueryParams(
            limit=limit,
            order_by=[{'field': 'name', 'direction': 'asc'}]
        )
        
        return await category_resolver(db=get_db_session(), info=info, params=params)


# Create BerryQL Factory and Resolvers
factory = BerryQLFactory()

# Create optimized resolvers with custom fields
user_resolver = factory.create_berryql_resolver(
    strawberry_type=UserType,
    model_class=User,
    custom_fields={
        UserType: {
            'post_count': build_post_count_query,
            'recent_post_count': build_recent_post_count_query,
            'avg_views': build_avg_views_query
        }
    }
)

post_resolver = factory.create_berryql_resolver(
    strawberry_type=PostType,
    model_class=Post
)

category_resolver = factory.create_berryql_resolver(
    strawberry_type=CategoryType,
    model_class=Category
)

# Database setup
async_engine = None
async_session = None


async def setup_database():
    """Setup test database with sample data."""
    global async_engine, async_session
    
    async_engine = create_async_engine(
        "sqlite+aiosqlite:///./advanced_example.db",
        echo=True
    )
    
    async_session = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Add sample data
    async with async_session() as session:
        # Create categories
        tech_cat = Category(name="Technology", description="Tech-related posts")
        lifestyle_cat = Category(name="Lifestyle", description="Lifestyle posts")
        
        session.add_all([tech_cat, lifestyle_cat])
        await session.commit()
        
        # Create users
        users = [
            User(name="Alice Johnson", email="alice@example.com"),
            User(name="Bob Smith", email="bob@example.com"),
            User(name="Charlie Brown", email="charlie@example.com"),
            User(name="Diana Prince", email="diana@example.com"),
        ]
        
        session.add_all(users)
        await session.commit()
        
        # Create posts with realistic data
        posts = [
            Post(title="Introduction to GraphQL", content="GraphQL is amazing...", 
                 author_id=users[0].id, category_id=tech_cat.id, view_count=150),
            Post(title="Python Best Practices", content="Here are some Python tips...", 
                 author_id=users[0].id, category_id=tech_cat.id, view_count=200),
            Post(title="Work-Life Balance", content="How to maintain balance...", 
                 author_id=users[1].id, category_id=lifestyle_cat.id, view_count=75),
            Post(title="Database Optimization", content="SQL optimization techniques...", 
                 author_id=users[1].id, category_id=tech_cat.id, view_count=300),
            Post(title="Morning Routines", content="Start your day right...", 
                 author_id=users[2].id, category_id=lifestyle_cat.id, view_count=125),
            Post(title="Advanced SQLAlchemy", content="Deep dive into SQLAlchemy...", 
                 author_id=users[3].id, category_id=tech_cat.id, view_count=450),
        ]
        
        session.add_all(posts)
        await session.commit()


def get_db_session():
    """Get database session for resolvers."""
    return async_session()


# Create GraphQL schema
schema = strawberry.Schema(query=Query)


async def main():
    """Main demo function showing advanced features."""
    await setup_database()
    
    # Example 1: Basic query with relationships
    print("=== Example 1: Basic Query with Relationships ===")
    query1 = """
    query {
        users(limit: 2) {
            id
            name
            email
            post_count
            recent_post_count
            avg_views
            posts {
                id
                title
                view_count
                category {
                    name
                }
            }
        }
    }
    """
    
    result1 = await schema.execute(query1)
    print("Result:", result1.data)
    
    # Example 2: Advanced filtering
    print("\n=== Example 2: Advanced Filtering ===")
    query2 = """
    query {
        posts(where: {
            title: { like: "%GraphQL%" },
            view_count: { gte: 100 }
        }) {
            id
            title
            view_count
            author {
                name
                post_count
            }
        }
    }
    """
    
    result2 = await schema.execute(query2)
    print("Result:", result2.data)
    
    # Example 3: Categories with nested posts
    print("\n=== Example 3: Categories with Posts ===")
    query3 = """
    query {
        categories {
            id
            name
            posts {
                id
                title
                author {
                    name
                }
            }
        }
    }
    """
    
    result3 = await schema.execute(query3)
    print("Result:", result3.data)
    
    if async_engine:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
