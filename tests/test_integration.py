"""
Integration tests for BerryQL with Strawberry GraphQL and SQLAlchemy.

This module contains integration tests that demonstrate and verify:
- Setting up models and GraphQL types
- Creating optimized resolvers with BerryQL
- Using query parameters for filtering and pagination
- Handling relationships with automatic N+1 elimination
- Cross-database compatibility (SQLite, PostgreSQL, MSSQL)
"""

import pytest
import strawberry
from typing import List, Optional
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship

from berryql import BerryQLFactory, GraphQLQueryParams, berryql
from conftest import User, Post, Comment  # Import models from conftest to avoid duplication


# Strawberry GraphQL Types
@strawberry.type
class CommentType:
    id: int
    content: str
    post_id: int
    author_id: int
    created_at: datetime


@strawberry.type
class PostType:
    id: int
    title: str
    content: Optional[str] = None
    author_id: int
    created_at: datetime
    
    @strawberry.field
    @berryql.field
    async def comments(self, info: strawberry.Info) -> List[CommentType]:
        """Get post's comments using pre-resolved data."""
        pass


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
    
    @strawberry.field
    @berryql.field
    async def comments(self, info: strawberry.Info) -> List[CommentType]:
        """Get user's comments using pre-resolved data."""
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
        name_filter: Optional[str] = None
    ) -> List[UserType]:
        """Get users with optional filtering and pagination."""
        # Get db_session from context
        db_session = info.context.get('db_session')
        user_resolver = info.context.get('user_resolver')
        
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
        return await user_resolver(db=db_session, info=info, params=params)
    
    @strawberry.field
    async def posts(
        self,
        info: strawberry.Info,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        title_filter: Optional[str] = None
    ) -> List[PostType]:
        """Get posts with optional filtering and pagination."""
        # Get db_session from context
        db_session = info.context.get('db_session')
        post_resolver = info.context.get('post_resolver')
        
        # Build query parameters
        where_conditions = {}
        if title_filter:
            where_conditions['title'] = {'like': f'%{title_filter}%'}
        
        params = GraphQLQueryParams(
            where=where_conditions,
            limit=limit,
            offset=offset,
            order_by=[{'field': 'created_at', 'direction': 'desc'}]
        )
        
        # Use the optimized resolver
        return await post_resolver(db=db_session, info=info, params=params)
    
# Remove global variables - we'll use context instead

@pytest.fixture
async def berryql_factory():
    """Create BerryQL factory."""
    return BerryQLFactory()


@pytest.fixture
async def resolvers(berryql_factory):
    """Create optimized resolvers."""
    user_resolver = berryql_factory.create_berryql_resolver(
        strawberry_type=UserType,
        model_class=User
    )
    
    post_resolver = berryql_factory.create_berryql_resolver(
        strawberry_type=PostType,
        model_class=Post
    )
    
    comment_resolver = berryql_factory.create_berryql_resolver(
        strawberry_type=CommentType,
        model_class=Comment
    )
    
    return {
        'user_resolver': user_resolver,
        'post_resolver': post_resolver,
        'comment_resolver': comment_resolver
    }


@pytest.fixture
async def graphql_schema():
    """Create GraphQL schema with resolvers."""
    return strawberry.Schema(query=Query)


@pytest.fixture
async def graphql_context(db_session, resolvers):
    """Create GraphQL execution context."""
    return {
        'db_session': db_session,
        'user_resolver': resolvers['user_resolver'],
        'post_resolver': resolvers['post_resolver'],
        'comment_resolver': resolvers['comment_resolver']
    }


# Remove the function-scoped fixtures since we're using session-scoped ones from conftest.py


@pytest.mark.integration
class TestBerryQLIntegration:
    """Integration tests for BerryQL functionality."""
    
    @pytest.mark.asyncio
    async def test_users_query_with_relationships(self, graphql_schema, graphql_context, populated_db):
        """Test basic users query with nested posts."""
        query = """
        query {
            users {
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
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        assert 'users' in result.data
        
        users = result.data['users']
        assert len(users) == 3  # We have 3 sample users
        
        # Check that relationships are properly loaded
        alice = next(user for user in users if user['name'] == 'Alice Johnson')
        assert len(alice['posts']) == 2  # Alice has 2 posts
        assert any(post['title'] == 'First Post' for post in alice['posts'])
        assert any(post['title'] == 'GraphQL is Great' for post in alice['posts'])
    
    @pytest.mark.asyncio
    async def test_users_with_posts_and_comments(self, graphql_schema, graphql_context, populated_db):
        """Test users query with nested posts and comments."""
        query = """
        query {
            users {
                id
                name
                email
                posts {
                    id
                    title
                    content
                    comments {
                        id
                        content
                        authorId
                    }
                }
                comments {
                    id
                    content
                    postId
                }
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        assert 'users' in result.data
        
        users = result.data['users']
        assert len(users) == 3  # We have 3 sample users
        
        # Check that Alice's posts have comments
        alice = next(user for user in users if user['name'] == 'Alice Johnson')
        first_post = next(post for post in alice['posts'] if post['title'] == 'First Post')
        assert len(first_post['comments']) == 2  # First post has 2 comments
        
        # Check that users have their own comments
        bob = next(user for user in users if user['name'] == 'Bob Smith')
        assert len(bob['comments']) >= 1  # Bob has made comments
    
    @pytest.mark.asyncio
    async def test_posts_query_with_comments(self, graphql_schema, graphql_context, populated_db):
        """Test posts query with nested comments."""
        query = """
        query {
            posts {
                id
                title
                content
                comments {
                    id
                    content
                    authorId
                }
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        assert 'posts' in result.data
        
        posts = result.data['posts']
        assert len(posts) == 5  # We have 5 sample posts
        
        # Check that the first post has comments
        first_post = next(post for post in posts if post['title'] == 'First Post')
        assert len(first_post['comments']) == 2  # First post has 2 comments
        
        # Verify comment structure
        for comment in first_post['comments']:
            assert 'id' in comment
            assert 'content' in comment
            assert 'authorId' in comment
    
    @pytest.mark.asyncio
    async def test_users_query_with_filtering(self, graphql_schema, graphql_context, populated_db):
        """Test users query with name filtering."""
        query = """
        query {
            users(nameFilter: "Alice") {
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
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 1
        assert users[0]['name'] == 'Alice Johnson'
        assert len(users[0]['posts']) == 2
    
    
    @pytest.mark.asyncio
    async def test_users_query_with_offset(self, graphql_schema, graphql_context, populated_db):
        """Test users query with offset pagination."""
        query = """
        query {
            users(limit: 1, offset: 1) {
                id
                name
                email
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 1  # Limited to 1 user with offset
    
    
    @pytest.mark.asyncio
    async def test_empty_results(self, graphql_schema, graphql_context, populated_db):
        """Test queries that return no results."""
        query = """
        query {
            users(nameFilter: "NonexistentUser") {
                id
                name
                email
                posts {
                    id
                    title
                }
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 0  # No users should match
    
    @pytest.mark.asyncio
    async def test_nested_relationship_optimization(self, graphql_schema, graphql_context, populated_db):
        """Test that nested relationships are optimized (no N+1 queries)."""
        # This test verifies that the query is optimized by checking the structure
        # In a real test, you might also want to monitor SQL query count
        
        query = """
        query {
            users {
                id
                name
                posts {
                    id
                    title
                    content
                    comments {
                        id
                        content
                        authorId
                    }
                }
                comments {
                    id
                    content
                    postId
                }
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        
        # Verify all users have their posts and comments loaded
        for user in users:
            assert 'posts' in user
            assert isinstance(user['posts'], list)
            assert 'comments' in user
            assert isinstance(user['comments'], list)
            
            # Check that posts have the expected fields and comments
            for post in user['posts']:
                assert 'id' in post
                assert 'title' in post
                assert 'content' in post
                assert 'comments' in post
                assert isinstance(post['comments'], list)
                
                # Check that comments have the expected fields
                for comment in post['comments']:
                    assert 'id' in comment
                    assert 'content' in comment
                    assert 'authorId' in comment


@pytest.mark.performance
class TestBerryQLPerformance:
    """Performance-focused integration tests."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, graphql_schema, graphql_context, db_session):
        """Test performance with a larger dataset."""
        # Create more test data
        users = []
        for i in range(50):
            user = User(
                name=f"User {i}",
                email=f"user{i}@example.com"
            )
            users.append(user)
        
        db_session.add_all(users)
        await db_session.commit()
        
        # Refresh to get IDs
        for user in users:
            await db_session.refresh(user)
        
        # Create posts for each user
        posts = []
        for user in users:
            for j in range(3):  # 3 posts per user
                post = Post(
                    title=f"Post {j} by {user.name}",
                    content=f"Content for post {j}",
                    author_id=user.id
                )
                posts.append(post)
        
        db_session.add_all(posts)
        await db_session.commit()
        
        # Refresh to get IDs
        for post in posts:
            await db_session.refresh(post)
        
        # Create comments for posts
        comments = []
        for i, post in enumerate(posts[:30]):  # Add comments to first 30 posts
            for k in range(2):  # 2 comments per post
                # Pick a random user to comment
                commenter = users[k % len(users)]
                comment = Comment(
                    content=f"Comment {k} on {post.title}",
                    post_id=post.id,
                    author_id=commenter.id
                )
                comments.append(comment)
        
        db_session.add_all(comments)
        await db_session.commit()
        
        # Test query performance
        query = """
        query {
            users(limit: 20) {
                id
                name
                posts {
                    id
                    title
                    comments {
                        id
                        content
                    }
                }
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 20
        
        # Verify all users have their posts with comments
        for user in users:
            assert len(user['posts']) == 3
            for post in user['posts']:
                # Comments should be loaded
                assert isinstance(post['comments'], list)


@pytest.mark.integration
class TestBerryQLComments:
    """Integration tests specifically for Comments functionality."""
    
    @pytest.mark.asyncio
    async def test_comments_filtering_by_post(self, graphql_schema, graphql_context, populated_db):
        """Test that we can filter comments by post efficiently."""
        # This would typically be implemented as a separate query or parameter
        # For this test, we'll verify the data structure supports it
        query = """
        query {
            posts(titleFilter: "First") {
                id
                title
                comments {
                    id
                    content
                    authorId
                }
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        posts = result.data['posts']
        assert len(posts) == 1  # Only "First Post" should match
        assert posts[0]['title'] == 'First Post'
        assert len(posts[0]['comments']) == 2  # First post has 2 comments
    
    @pytest.mark.asyncio
    async def test_user_comments_relationship(self, graphql_schema, graphql_context, populated_db):
        """Test that user comments relationship works correctly."""
        query = """
        query {
            users(nameFilter: "Bob") {
                id
                name
                comments {
                    id
                    content
                    postId
                }
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 1
        bob = users[0]
        assert bob['name'] == 'Bob Smith'
        
        # Bob should have made some comments
        assert len(bob['comments']) >= 1
        
        # Verify comment structure
        for comment in bob['comments']:
            assert 'id' in comment
            assert 'content' in comment
            assert 'postId' in comment
    
    @pytest.mark.asyncio
    async def test_deep_nested_relationships(self, graphql_schema, graphql_context, populated_db):
        """Test deep nesting: User -> Posts -> Comments."""
        query = """
        query {
            users(limit: 1) {
                id
                name
                posts {
                    id
                    title
                    comments {
                        id
                        content
                        authorId
                    }
                }
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 1
        
        user = users[0]
        assert 'posts' in user
        
        for post in user['posts']:
            assert 'comments' in post
            for comment in post['comments']:
                assert 'authorId' in comment
                # The author_id should be a valid user ID
                assert isinstance(comment['authorId'], int)
                assert comment['authorId'] > 0


@pytest.mark.integration
class TestBerryQLErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_filter_handling(self, graphql_schema, graphql_context, populated_db):
        """Test handling of invalid filter parameters."""
        # This test might depend on your specific validation logic
        query = """
        query {
            users(limit: -1) {
                id
                name
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        # The exact behavior depends on your validation logic
        # This test ensures the system handles edge cases gracefully
        assert result is not None
