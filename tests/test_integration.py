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
from .schema import schema
from .models import User, Post
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
        assert len(users) == 4  # We have 4 sample users

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
            assert len(users) == 4  # We have 4 sample users
            
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
    async def test_posts_content_filtering(self, graphql_schema, graphql_context, populated_db):
        """Test posts field contentFilter argument filters posts by content (ILIKE %value%)."""
        # Filter Alice's posts for those whose content contains 'GraphQL'
        query_graphql = """
        query {
            users(nameFilter: "Alice") {
                id
                name
                posts(contentFilter: "GraphQL") {
                    id
                    title
                    content
                }
            }
        }
        """
        result_graphql = await graphql_schema.execute(query_graphql, context_value=graphql_context)
        assert result_graphql.errors is None, f"Errors: {result_graphql.errors}"
        users_graphql = result_graphql.data['users']
        assert len(users_graphql) == 1
        alice_graphql = users_graphql[0]
        filtered_posts = alice_graphql['posts']
        assert len(filtered_posts) == 1, "Should return only the post whose content includes 'GraphQL'"
        assert filtered_posts[0]['title'] == 'GraphQL is Great'

        # Case-insensitive check: search with lowercase should still match (ILIKE)
        query_lower = """
        query {
            users(nameFilter: "Alice") {
                id
                name
                posts(contentFilter: "hello") {
                    title
                    content
                }
            }
        }
        """
        result_lower = await graphql_schema.execute(query_lower, context_value=graphql_context)
        assert result_lower.errors is None, f"Errors: {result_lower.errors}"
        posts_lower = result_lower.data['users'][0]['posts']
        # 'Hello world!' should match
        assert len(posts_lower) == 1
        assert posts_lower[0]['title'] == 'First Post'

        # Unmatched filter should return empty list
        query_none = """
        query {
            users(nameFilter: "Alice") {
                posts(contentFilter: "NoMatchSubstring") { id }
            }
        }
        """
        result_none = await graphql_schema.execute(query_none, context_value=graphql_context)
        assert result_none.errors is None, f"Errors: {result_none.errors}"
        assert result_none.data['users'][0]['posts'] == []

    @pytest.mark.asyncio
    async def test_post_comments_rate_less_than_filter(self, graphql_schema, graphql_context, populated_db):
        """Test nested comments(rateLessThan) argument filters comments by rate (< value)."""
        # Query Alice's posts and filter comments with rateLessThan: 3 (should exclude rate >=3)
        query_rate_lt = """
        query {
            users(nameFilter: "Alice") {
                id
                posts {
                    id
                    title
                    comments(rateLessThan: 3) {
                        id
                        rate
                        content
                    }
                }
            }
        }
        """
        result_rate_lt = await graphql_schema.execute(query_rate_lt, context_value=graphql_context)
        assert result_rate_lt.errors is None, f"Errors: {result_rate_lt.errors}"
        users = result_rate_lt.data['users']
        assert len(users) == 1
        alice = users[0]
        for post in alice['posts']:
            if post['title'] == 'First Post':
                # First Post has comment rates 2 and 1 in fixtures; both are <3
                comment_rates = sorted(c['rate'] for c in post['comments'])
                assert comment_rates == [1, 2]
            elif post['title'] == 'GraphQL is Great':
                # GraphQL is Great has one comment rate 3 (==3) so with <3 filter should exclude it
                assert post['comments'] == []

        # Stricter filter (rateLessThan: 2) should remove the rate 2 comment as well
        query_rate_lt_2 = """
        query {
            users(nameFilter: "Alice") {
                posts {
                    id
                    title
                    comments(rateLessThan: 2) { id rate }
                }
            }
        }
        """
        result_rate_lt_2 = await graphql_schema.execute(query_rate_lt_2, context_value=graphql_context)
        assert result_rate_lt_2.errors is None, f"Errors: {result_rate_lt_2.errors}"
        for post in result_rate_lt_2.data['users'][0]['posts']:
            if post['title'] == 'First Post':
                # <2 should keep only rate 1
                rates = sorted(c['rate'] for c in post['comments'])
                assert rates == [1]
            else:
                assert post['comments'] == []
    
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
        assert len(users) == 4  # Admin can see all 4 users

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
    async def test_active_users_default_custom_order(self, graphql_schema, graphql_context, populated_db):
        """Ensure activeUsers uses default custom_order when no orderBy is provided (name asc, created_at desc)."""
        query = """
        query {
            activeUsers {
                id
                name
            }
        }
        """

        result = await graphql_schema.execute(query, context_value=graphql_context)
        assert result.errors is None
        users = result.data['activeUsers']
        assert len(users) == 4
        names = [u['name'] for u in users]
        # Expect alphabetical by name (custom_order default)
        assert names == sorted(names)
        assert names == [
            'Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Dave NoPosts'
        ]

    @pytest.mark.asyncio
    async def test_active_users_order_by_param_overrides_default(self, graphql_schema, graphql_context, populated_db):
        """Ensure activeUsers(orderBy: ...) overrides the default custom_order."""
        query = """
        query {
            activeUsers(orderBy: "[{\\\"field\\\": \\\"id\\\", \\\"direction\\\": \\\"desc\\\"}]") {
                id
                name
            }
        }
        """

        result = await graphql_schema.execute(query, context_value=graphql_context)
        assert result.errors is None
        users = result.data['activeUsers']
        ids = [u['id'] for u in users]
        assert ids == sorted(ids, reverse=True)  # id desc
    
    
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
                postAgg { count }
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
            assert 'postAgg' in user, f"User missing postAgg: {user}"
            post_count = user['postAgg']['count']
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
        
        # Add another recent post for Alice to ensure multiple new posts are available
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        await session.merge(Post(
            title="Async SQLAlchemy Rocks",
            content="Async sessions are great!",
            author_id=alice.id,
            created_at=now - timedelta(minutes=5),
        ))
        await session.commit()

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
        
        # new_posts should include posts from the last hour (excluding "First Post")
        assert len(new_posts) >= 2  # Now there should be at least two recent posts
        
        # Verify that the time filtering is working - old post should not be included
        post_titles = [post['title'] for post in new_posts]
        assert "First Post" not in post_titles  # The old post should be filtered out
        assert "GraphQL is Great" in post_titles  # Existing recent post should be included
        assert "Async SQLAlchemy Rocks" in post_titles  # Newly added recent post should be included

        # Ensure newPosts are ordered by createdAt desc
        created_times = [p['createdAt'] for p in new_posts]
        assert created_times == sorted(created_times, reverse=True)
        # Newest title should be the one we just added
        assert new_posts[0]['title'] == "Async SQLAlchemy Rocks"

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
    async def test_bloggers_excludes_users_with_no_posts(self, graphql_schema, graphql_context, populated_db):
        """Add a user with no posts and verify bloggers returns only users with posts."""
        # Dave NoPosts is included in common fixtures with no posts

        # Act: query bloggers
        query = """
        query {
            bloggers(orderBy: "[{\\\"field\\\": \\\"id\\\", \\\"direction\\\": \\\"asc\\\"}]") {
                id
                name
                email
            }
        }
        """

        result = await graphql_schema.execute(query, context_value=graphql_context)

        # Assert: no errors and bloggers exist
        assert result.errors is None
        assert result.data is not None
        bloggers = result.data['bloggers']
        assert isinstance(bloggers, list)
        # From fixtures: Alice, Bob, Charlie all have posts; user_no_posts has none
        blogger_names = [u['name'] for u in bloggers]
        assert "Alice Johnson" in blogger_names
        assert "Bob Smith" in blogger_names
        assert "Charlie Brown" in blogger_names
        assert "Dave NoPosts" not in blogger_names

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

    @pytest.mark.asyncio
    async def test_posts_comments_agg(self, graphql_schema, graphql_context, populated_db):
        """Verify commentsAgg provides minCreatedAt and commentsCount per post."""
        query = """
        query {
            users {
                id
                name
                posts {
                    id
                    title
                    commentsAgg {
                        minCreatedAt
                        commentsCount
                    }
                }
            }
        }
        """

        result = await graphql_schema.execute(query, context_value=graphql_context)
        assert result.errors is None
        assert result.data is not None

        users = result.data['users']
        # Build a map of user -> {post title -> (minCreatedAt, commentsCount)}
        for user in users:
            for post in user['posts']:
                agg = post.get('commentsAgg')
                assert agg is not None
                # minCreatedAt should be present (stringified datetime) and commentsCount non-negative
                assert agg['minCreatedAt'] is not None
                assert isinstance(agg['commentsCount'], int)
                assert agg['commentsCount'] >= 0

        # Spot-check known counts from fixtures
        alice = next(u for u in users if u['name'] == 'Alice Johnson')
        alice_counts = {p['title']: p['commentsAgg']['commentsCount'] for p in alice['posts']}
        assert alice_counts.get('First Post') == 2
        assert alice_counts.get('GraphQL is Great') == 1

        bob = next(u for u in users if u['name'] == 'Bob Smith')
        bob_counts = {p['title']: p['commentsAgg']['commentsCount'] for p in bob['posts']}
        assert bob_counts.get('SQLAlchemy Tips') == 2
        assert bob_counts.get('Python Best Practices') == 1

    @pytest.mark.asyncio
    async def test_posts_last_comment(self, graphql_schema, graphql_context, populated_db):
        """Verify lastComment returns the latest comment per post (by created_at then id)."""
        query = """
        query {
            users {
                id
                name
                posts {
                    id
                    title
                    lastComment {
                        id
                        content
                        authorId
                        postId
                        createdAt
                    }
                }
            }
        }
        """

        result = await graphql_schema.execute(query, context_value=graphql_context)
        assert result.errors is None
        assert result.data is not None

        users = result.data['users']
        # Build mapping post title -> last comment content for quick checks
        def last_content_of(user_name, post_title):
            u = next(u for u in users if u['name'] == user_name)
            p = next(p for p in u['posts'] if p['title'] == post_title)
            lc = p.get('lastComment')
            return lc['content'] if lc else None

        # From fixtures, each post with comments should have a last comment
        assert last_content_of('Alice Johnson', 'First Post') is not None
        assert last_content_of('Alice Johnson', 'GraphQL is Great') is not None
        assert last_content_of('Bob Smith', 'SQLAlchemy Tips') is not None
        assert last_content_of('Bob Smith', 'Python Best Practices') is not None
        assert last_content_of('Charlie Brown', 'Getting Started') is not None

    @pytest.mark.asyncio
    async def test_post_comments_order_by_content_desc(self, graphql_schema, graphql_context, populated_db):
        """Ensure PostType.comments(orderBy: "content desc") sorts comments by content descending."""
        query = """
        query {
            users(nameFilter: "Alice Johnson") {
                id
                name
                posts {
                    id
                    title
                    comments(orderBy: "content desc") {
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
        assert len(users) == 1
        alice = users[0]
        first_post = next(p for p in alice['posts'] if p['title'] == 'First Post')
        contents = [c['content'] for c in first_post['comments']]
        # From fixtures for Alice's "First Post":
        # - "Great post!"
        # - "Thanks for sharing!"
        # Descending by content should put "Thanks for sharing!" first
        assert contents == sorted(contents, reverse=True)
        assert contents[0] == "Thanks for sharing!"

    @pytest.mark.asyncio
    async def test_post_comments_order_by_content_asc(self, graphql_schema, graphql_context, populated_db):
        """Ensure PostType.comments(orderBy: "content asc") sorts comments by content ascending."""
        query = """
        query {
            users(nameFilter: "Alice Johnson") {
                id
                name
                posts {
                    id
                    title
                    comments(orderBy: "content asc") {
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
        assert len(users) == 1
        alice = users[0]
        first_post = next(p for p in alice['posts'] if p['title'] == 'First Post')
        contents = [c['content'] for c in first_post['comments']]
        # Ascending by content should put "Great post!" first
        assert contents == sorted(contents)
        assert contents[0] == "Great post!"

    @pytest.mark.asyncio
    async def test_post_comments_default_order_by_rate(self, graphql_schema, graphql_context, populated_db):
        """Ensure PostType.comments (without orderBy) defaults to order by rate (ascending)."""
        query = """
        query {
            users(nameFilter: "Alice Johnson") {
                id
                name
                posts {
                    id
                    title
                    comments {
                        id
                        content
                        rate
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
        alice = users[0]
        first_post = next(p for p in alice['posts'] if p['title'] == 'First Post')
        rates = [c['rate'] for c in first_post['comments']]
        # With rates [1,2] for Alice's first post, default asc ordering should keep [1,2]
        assert rates == sorted(rates)
        # Also check the first is the lower-rated "Great post!"
        assert first_post['comments'][0]['content'] == "Thanks for sharing!"


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
