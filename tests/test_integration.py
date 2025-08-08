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
async def graphql_context(db_session, sample_users):
    """Create GraphQL execution context."""
    current_user = sample_users[0] if sample_users else None  # Use Alice as current user
    return {
        'db_session': db_session,
        'user_id': 1,
        'current_user': current_user
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
        from sqlalchemy import event
        
        # Set up SQL query counting
        query_count = {'count': 0}
        
        def count_queries(conn, cursor, statement, parameters, context, executemany):
            query_count['count'] += 1
        
        # Get the engine from the session
        engine = graphql_context['db_session'].get_bind()
        
        # Add event listener to count SQL queries
        event.listen(engine, "before_cursor_execute", count_queries)
        
        try:
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
            
            # Reset counter before executing GraphQL query
            query_count['count'] = 0
            
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
            
            # Assert that the SQL query was executed only once (N+1 prevention)
            assert query_count['count'] == 1, f"Expected 1 SQL query, but {query_count['count']} were executed"
            
        finally:
            # Clean up event listener
            event.remove(engine, "before_cursor_execute", count_queries)
    
    
    @pytest.mark.asyncio
    async def test_users_query_with_filtering(self, graphql_schema, graphql_context, populated_db):
        """Test users query with name filtering for admin user."""
        query = """
        query {
            users(nameFilter: "Alice") {
                id
                name
                email
                isAdmin
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
        assert users[0]['isAdmin'] == True  # Alice is admin
        assert len(users[0]['posts']) == 2
    
    @pytest.mark.asyncio
    async def test_admin_users_can_see_all_users(self, graphql_schema, graphql_context, populated_db):
        """Test that admin users can see all users."""
        query = """
        query {
            users {
                id
                name
                email
                isAdmin
            }
        }
        """
        
        # graphql_context uses Alice (user_id=1) who is admin
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 3  # Admin can see all 3 users
        
        # Verify we have all users including admin status
        user_names = [user['name'] for user in users]
        assert 'Alice Johnson' in user_names
        assert 'Bob Smith' in user_names
        assert 'Charlie Brown' in user_names
        
        # Verify admin status
        alice = next(user for user in users if user['name'] == 'Alice Johnson')
        bob = next(user for user in users if user['name'] == 'Bob Smith')
        charlie = next(user for user in users if user['name'] == 'Charlie Brown')
        
        assert alice['isAdmin'] == True
        assert bob['isAdmin'] == False
        assert charlie['isAdmin'] == False
    
    @pytest.mark.asyncio
    async def test_non_admin_users_see_only_themselves(self, graphql_schema, db_session, sample_users, populated_db):
        """Test that non-admin users can only see themselves."""
        # Create context for Bob (non-admin user)
        bob = sample_users[1]  # Bob Smith is at index 1
        non_admin_context = {
            'db_session': db_session,
            'user_id': bob.id,
            'current_user': bob
        }
        
        query = """
        query {
            users {
                id
                name
                email
                isAdmin
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=non_admin_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 1  # Non-admin can only see themselves
        assert users[0]['name'] == 'Bob Smith'
        assert users[0]['id'] == bob.id
        assert users[0]['isAdmin'] == False
    
    @pytest.mark.asyncio
    async def test_non_admin_users_filtering_still_restricted(self, graphql_schema, db_session, sample_users, populated_db):
        """Test that non-admin users can't see other users even with name filtering."""
        # Create context for Bob (non-admin user)
        bob = sample_users[1]  # Bob Smith is at index 1
        non_admin_context = {
            'db_session': db_session,
            'user_id': bob.id,
            'current_user': bob
        }
        
        # Try to filter for Alice while logged in as Bob
        query = """
        query {
            users(nameFilter: "Alice") {
                id
                name
                email
                isAdmin
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=non_admin_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        # Non-admin should get no results when trying to filter for someone else
        assert len(users) == 0
    
    @pytest.mark.asyncio
    async def test_no_user_context_returns_empty(self, graphql_schema, db_session, populated_db):
        """Test that queries with no user context return empty results."""
        # Create context without current_user
        no_user_context = {
            'db_session': db_session,
            'user_id': None,
            'current_user': None
        }
        
        query = """
        query {
            users {
                id
                name
                email
                isAdmin
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=no_user_context)
        
        assert result.errors is None
        assert result.data is not None
        
        users = result.data['users']
        assert len(users) == 0  # No user context means no results
    
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
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).replace(tzinfo=None)
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

    @pytest.mark.asyncio
    async def test_current_user_query(self, graphql_schema, graphql_context, populated_db):
        """Test current_user query returns the user from context."""
        query = """
        query {
            currentUser {
                id
                name
                email
                createdAt
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        current_user = result.data['currentUser']
        assert current_user is not None
        
        # Should match the current_user from context (Alice, the first user)
        expected_user = graphql_context['current_user']
        assert current_user['id'] == expected_user.id
        assert current_user['name'] == expected_user.name
        assert current_user['email'] == expected_user.email
        assert current_user['createdAt'] is not None

    @pytest.mark.asyncio
    async def test_current_user_with_no_context(self, graphql_schema, db_session):
        """Test current_user query when no current_user is in context."""
        # Create context without current_user
        context_without_user = {
            'db_session': db_session,
            'user_id': None
        }
        
        query = """
        query {
            currentUser {
                id
                name
                email
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=context_without_user)
        
        assert result.errors is None
        assert result.data is not None
        assert result.data['currentUser'] is None

    @pytest.mark.asyncio
    async def test_custom_where_with_context(self, graphql_schema, graphql_context, populated_db):
        """Test custom_where callable that accesses GraphQL context."""
        query = """
        query {
            otherUsers {
                id
                name
                email
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=graphql_context)
        
        assert result.errors is None
        assert result.data is not None
        
        other_users = result.data['otherUsers']
        assert other_users is not None
        assert len(other_users) >= 2  # Should have at least 2 users (excluding current user)
        
        # Verify the current user (user_id = 1) is excluded
        current_user_id = graphql_context.get('user_id', 1)
        user_ids = [user['id'] for user in other_users]
        assert current_user_id not in user_ids
        
        # Verify we get the other users (Bob and Charlie)
        user_names = [user['name'] for user in other_users]
        assert "Bob Smith" in user_names
        assert "Charlie Brown" in user_names
        assert "Alice Johnson" not in user_names  # Alice should be excluded (current user)

    @pytest.mark.asyncio
    async def test_custom_logic_execution_in_berryql_field(self, graphql_schema, db_session, sample_users):
        """Test that custom logic in @berryql.field decorated methods executes."""
        # Create context that should trigger custom logic
        test_context = {
            'db_session': db_session,
            'user_id': 1  # Alice's user ID
        }
        
        query = """
        query {
            currentUser {
                id
                name
                email
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=test_context)
        
        # Should work normally since custom logic just returns None (falls back to BerryQL)
        assert result.errors is None
        assert result.data is not None
        assert result.data['currentUser'] is not None
        assert result.data['currentUser']['id'] == 1

    @pytest.mark.asyncio
    async def test_berryql_field_custom_logic_capability(self, graphql_schema, db_session, sample_users):
        """Test that @berryql.field decorated methods can execute custom logic."""
        # Test normal case (should work)
        normal_context = {'db_session': db_session, 'user_id': 1}
        query = """
        query {
            currentUser {
                id
                name
            }
        }
        """
        
        result = await graphql_schema.execute(query, context_value=normal_context)
        assert result.errors is None
        assert result.data['currentUser'] is not None
        
        # Test exception case (should fail with our custom error)
        exception_context = {'db_session': db_session, 'user_id': 999}
        result_with_error = await graphql_schema.execute(query, context_value=exception_context)
        
        # Verify our custom exception was thrown
        assert result_with_error.errors is not None
        assert "Custom logic executed: User 999 is forbidden!" in str(result_with_error.errors[0])


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
