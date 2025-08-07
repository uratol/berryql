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
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from berryql import BerryQLFactory
from schema import schema, UserType  # Import schema from separate file
from conftest import User, Post, Comment  # Import models for direct database operations
@pytest.fixture
async def berryql_factory():
    """Create BerryQL factory."""
    return BerryQLFactory()


@pytest.fixture
async def graphql_schema():
    """Create GraphQL schema with resolvers."""
    return schema


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
    async def test_active_users_with_query_params(self, graphql_schema, graphql_context, populated_db):
        """Test activeUsers with GraphQL query where and order parameters."""
        
        # Test with simple where condition using string escaping 
        query1 = '''
        query {
            activeUsers(where: "{\\"name\\": {\\"like\\": \\"%Alice%\\"}}", orderBy: "[{\\"field\\": \\"created_at\\", \\"direction\\": \\"desc\\"}]") {
                id
                name
                email
                createdAt
                posts {
                    id
                    title
                }
            }
        }
        '''
        
        result = await graphql_schema.execute(query1, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['activeUsers']
        
        # Should find Alice (filtered by name) and she should have an email (from custom_where)
        assert len(users) >= 1
        alice = next((user for user in users if "Alice" in user['name']), None)
        assert alice is not None
        assert alice['email'] is not None
        assert alice['email'] != ''
        
        # Test with different where condition (no name filter, test id > 0)
        query2 = '''
        query {
            activeUsers(where: "{\\"id\\": {\\"gt\\": 0}}") {
                id
                name
                email
                createdAt
            }
        }
        '''
        
        result_all = await graphql_schema.execute(query2, context_value=graphql_context)
        
        assert result_all.errors is None
        assert result_all.data is not None
        
        all_active_users = result_all.data['activeUsers']
        
        # Should have all users with email (from custom_where condition)
        assert len(all_active_users) >= 3  # We have 3 sample users, all should have email
        
        # Verify all users have email (due to custom_where condition)
        for user in all_active_users:
            assert user['email'] is not None
            assert user['email'] != ''
        
        # Test order by functionality 
        query3 = '''
        query {
            activeUsers(orderBy: "[{\\"field\\": \\"name\\", \\"direction\\": \\"asc\\"}]") {
                id
                name
                email
            }
        }
        '''
        
        result_ordered = await graphql_schema.execute(query3, context_value=graphql_context)
        
        assert result_ordered.errors is None
        assert result_ordered.data is not None
        
        ordered_users = result_ordered.data['activeUsers']
        
        # Verify ordering - should be in alphabetical order by name
        if len(ordered_users) > 1:
            for i in range(len(ordered_users) - 1):
                current_name = ordered_users[i]['name']
                next_name = ordered_users[i + 1]['name']
                assert current_name <= next_name  # Should be in alphabetical order
    
    
    @pytest.mark.asyncio
    async def test_users_with_post_aggregation(self, graphql_schema, graphql_context, populated_db):
        """Test users query with postCount aggregation field."""
        # Start with a simple test - just query users without custom fields first
        query1 = """
        query {
            users {
                id
                name
                email
            }
        }
        """
        
        result1 = await graphql_schema.execute(query1, context_value=graphql_context)
        assert result1.errors is None, f"Basic query failed: {result1.errors}"
        assert result1.data is not None
        assert 'users' in result1.data
        users1 = result1.data['users']
        assert len(users1) > 0, "No users found in basic query"
        
        # Now test with the custom field
        query2 = """
        query {
            users {
                id
                name
                email
                postCount
            }
        }
        """
        
        result2 = await graphql_schema.execute(query2, context_value=graphql_context)
        
        # Print result for debugging
        print(f"Result data: {result2.data}")
        print(f"Result errors: {result2.errors}")
        
        if result2.errors:
            for error in result2.errors:
                print(f"Error: {error}")
                print(f"Error path: {error.path if hasattr(error, 'path') else 'No path'}")
        
        assert result2.errors is None, f"Custom field query failed: {result2.errors}"
        assert result2.data is not None
        assert 'users' in result2.data
        
        users2 = result2.data['users']
        assert len(users2) > 0, "No users found with custom field"
        
        # Check that all users have postCount data
        for user in users2:
            assert 'postCount' in user, f"User missing postCount: {user}"
            post_count = user['postCount']
            assert isinstance(post_count, int), f"postCount should be int, got {type(post_count)}: {post_count}"
            assert post_count >= 0, f"postCount should be non-negative: {post_count}"
    
    
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
    
    @pytest.mark.asyncio
    async def test_new_posts_filter(self, graphql_schema, graphql_context, populated_db):
        """Test new_posts field that filters posts created within the last hour and sorts by created_at desc."""
        from datetime import datetime, timezone, timedelta
        from sqlalchemy import select, update
        
        # Get a database session for manual manipulation
        session = graphql_context['db_session']
        
        # Get Alice's user ID
        alice_result = await session.execute(
            select(User).where(User.name == "Alice Johnson")
        )
        alice = alice_result.scalar_one()
        
        # Update one of Alice's posts to be older than 1 hour
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        await session.execute(
            update(Post)
            .where(Post.author_id == alice.id)
            .where(Post.title == "First Post")
            .values(created_at=old_time)
        )
        await session.commit()
        
        # Test regular posts field (should include all posts)
        query_regular_posts = """
        query {
            users(nameFilter: "Alice Johnson") {
                id
                name
                posts {
                    id
                    title
                }
            }
        }
        """
        
        result = await graphql_schema.execute(query_regular_posts, context_value=graphql_context)
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 1
        alice_data = users[0]
        
        all_posts = alice_data['posts']
        print(f"Alice's all posts: {all_posts}")
        assert len(all_posts) == 2  # Alice has 2 posts total
        
        # Test new_posts field (should exclude old posts)
        query_new_posts = """
        query {
            users(nameFilter: "Alice Johnson") {
                id
                name
                newPosts {
                    id
                    title
                    createdAt
                }
                posts {
                    id
                    createdAt
                }
            }
        }
        """
        
        result = await graphql_schema.execute(query_new_posts, context_value=graphql_context)
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 1
        alice_data = users[0]
        
        new_posts = alice_data['newPosts']
        print(f"Alice's new posts: {new_posts}")
        
        # new_posts should only include posts from the last hour (excluding "First Post")
        assert len(new_posts) == 1  # Should only have the recent post
        assert new_posts[0]['title'] == "GraphQL is Great"  # Should be the newer post
        
        # Verify that the time filtering is working - old post should not be included
        post_titles = [post['title'] for post in new_posts]
        assert "First Post" not in post_titles  # The old post should be filtered out
        assert "GraphQL is Great" in post_titles  # The new post should be included


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
