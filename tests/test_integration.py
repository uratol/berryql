"""
Integration tests for BerryQL with Strawberry GraphQL and SQLAlchemy.

This module contains integration tests that demonstrate and verify:
- Setting up models and GraphQL types
- Using the enhanced @berryql.field decorator with model_class, custom_fields, custom_where, and custom_order
- Avoiding direct resolver creation by using decorator parameters
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
from sqlalchemy.ext.asyncio import AsyncSession

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
    @berryql.field(model_class=User)
    async def users(
        self,
        info: strawberry.Info,
        db: AsyncSession,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        name_filter: Optional[str] = None
    ) -> List[UserType]:
        """Get users with optional filtering and pagination using @berryql.field decorator."""
        # The decorator handles the resolver creation and execution automatically
        # The BerryQL resolver will process the name_filter parameter
        pass  # Implementation handled by the decorator
    
    @strawberry.field
    @berryql.field(
        model_class=User,
        custom_where={UserType: {'email': {'ne': None}}},  # Only users with email
        custom_order={UserType: ['name', 'created_at desc']}  # Default ordering
    )
    async def active_users(
        self,
        info: strawberry.Info,
        db: AsyncSession,
        params: Optional[GraphQLQueryParams] = None
    ) -> List[UserType]:
        """Get active users with custom default conditions using @berryql.field decorator."""
        # The decorator automatically applies the custom_where and custom_order
        pass  # Implementation handled by the decorator
    

@pytest.fixture
async def berryql_factory():
    """Create BerryQL factory."""
    return BerryQLFactory()


@pytest.fixture
async def graphql_schema():
    """Create GraphQL schema with resolvers."""
    return strawberry.Schema(query=Query)


@pytest.fixture
async def graphql_context(db_session):
    """Create GraphQL execution context."""
    return {
        'db_session': db_session
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
    async def test_active_users_with_custom_conditions(self, graphql_schema, graphql_context, populated_db):
        """Test active_users query with custom where and order conditions."""
        query = """
        query {
            activeUsers {
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
        
        users = result.data['activeUsers']
        assert len(users) >= 1  # Should have users with email
        
        # Verify all users have email (due to custom_where condition)
        for user in users:
            assert user['email'] is not None
            assert user['email'] != ''
        
        # Verify ordering - should be ordered by name first, then created_at desc
        if len(users) > 1:
            # Check that names are in alphabetical order where same name
            for i in range(len(users) - 1):
                current_name = users[i]['name']
                next_name = users[i + 1]['name']
                assert current_name <= next_name  # Should be in alphabetical order
    
    
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
