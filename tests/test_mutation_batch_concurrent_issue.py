"""Test to reproduce and fix concurrent operations error in batch mutations.

This test file focuses on reproducing the specific error:
"This session is provisioning a new connection; concurrent operations are not permitted"
or "Session is already flushing"
(SQLAlchemy error code: isce)

THE ROOT CAUSE:
GraphQL executes multiple mutations in a SINGLE batch request IN PARALLEL.
When you have:
  mutation {
    m1: merge_posts(...) { ... }
    m2: merge_posts(...) { ... }
  }

GraphQL resolver execution happens CONCURRENTLY for m1, m2.
Each mutation's pre-hook tries to query the database using the SAME session.
Result: "concurrent operations are not permitted" or "Session is already flushing" error.

POTENTIAL FIX:
In berryql/mutations.py:
1. Add a session-level lock/semaphore to serialize database access
2. Use asyncio.Lock to ensure only one mutation uses session at a time
3. Or use connection pooling with multiple connections
"""
import pytest
from tests.schema import schema, CALLBACK_EVENTS


pytestmark = pytest.mark.asyncio


async def test_sequential_mutations_with_hooks_same_session(db_session, populated_db):
    """Reproduce: Two sequential mutations in one batch, each with hooks querying the DB.
    
    The error happens because:
    - mutation1 starts, triggers pre-hook that queries users table
    - mutation1 flushes to DB (session busy)
    - GraphQL starts resolving mutation2 before mutation1 fully commits
    - mutation2 pre-hook tries to query users table with same session
    - Session throws: "concurrent operations are not permitted"
    
    This is NOT about parallel execution - it's about sequential mutations
    sharing the same session when one is still in a pending state.
    """
    CALLBACK_EVENTS.clear()
    
    mutation = """
    mutation TwoMutationsInBatch {
      first: merge_posts(payload: [{
        title: "First Mutation",
        content: "Content 1",
        author_email: "alice@example.com"
      }]) {
        id
        title
        author_id
      }
      second: merge_posts(payload: [{
        title: "Second Mutation",
        content: "Content 2",
        author_email: "bob@example.com"
      }]) {
        id
        title
        author_id
      }
    }
    """
    
    # The author_email field triggers _resolve_author_email pre-hook
    # which executes: await session.execute(select(User).where(User.email == email))
    # If this happens while the session is busy from the first mutation,
    # we get the concurrent operations error
    
    res = await schema.execute(mutation, context_value={"db_session": db_session})
    
    if res.errors:
        error_str = str(res.errors[0]) if res.errors else ""
        
        # Check if this is the concurrent operations error
        is_concurrent_error = any([
            "concurrent operations are not permitted" in error_str.lower(),
            "provisioning a new connection" in error_str.lower(),
            "connection is busy" in error_str.lower(),
            "isce" in error_str.lower(),
        ])
        
        if is_concurrent_error:
            # This is the bug we're trying to fix
            pytest.fail(
                f"CONCURRENT OPERATIONS ERROR REPRODUCED:\n"
                f"{error_str[:500]}\n\n"
                f"This error occurs because both mutations use the same session,\n"
                f"and the second mutation's hook tries to query while the first\n"
                f"mutation hasn't finished committing.\n\n"
                f"FIX NEEDED IN: berryql/mutations.py - ensure hooks properly await\n"
                f"session operations or use separate connection contexts."
            )
        else:
            # Some other error
            pytest.fail(f"Unexpected error (not concurrent ops): {res.errors}")
    
    # If we get here, the mutations succeeded
    assert res.data is not None
    assert res.data["first"][0]["author_id"] == 1  # alice
    assert res.data["second"][0]["author_id"] == 2  # bob


async def test_batch_mutations_with_enabled_callbacks(db_session, populated_db):
    """Test with test_callbacks enabled - this adds MORE hooks that query the DB."""
    CALLBACK_EVENTS.clear()
    
    mutation = """
    mutation BatchWithCallbacks {
      m1: merge_posts(payload: [{
        title: "With Callbacks 1",
        content: "C1",
        author_email: "alice@example.com"
      }]) {
        id
        title
      }
      m2: merge_posts(payload: [{
        title: "With Callbacks 2",
        content: "C2",
        author_email: "bob@example.com"
      }]) {
        id
        title
      }
    }
    """
    
    # Enable test_callbacks which adds pre/post hooks that modify data
    # and log events - this increases the chance of concurrent operations
    res = await schema.execute(
        mutation,
        context_value={"db_session": db_session, "test_callbacks": True}
    )
    
    if res.errors:
        error_str = str(res.errors[0]) if res.errors else ""
        if any(x in error_str.lower() for x in ["concurrent", "provisioning", "connection is busy", "isce"]):
            pytest.fail(f"CONCURRENT OPS ERROR WITH CALLBACKS:\n{error_str[:500]}")
        else:
            # May be an expected error from callbacks
            pass
    
    if res.data:
        # Verify callbacks ran (they should add [hpre][pre] and [post][hpost] to titles)
        if res.data.get("m1"):
            title = res.data["m1"][0].get("title", "")
            # Check if any hook markers are present
            assert any(marker in title for marker in ["[pre]", "[hpre]", "[post]", "[hpost]"])


async def test_direct_session_concurrent_usage(db_session, populated_db):
    """Minimal test: directly use session twice without awaiting first operation.
    
    This simulates what might happen in the hooks during batch mutations.
    """
    from tests.models import User
    from sqlalchemy import select
    
    # This should NOT cause an error if done sequentially
    result1 = await db_session.execute(select(User).where(User.email == "alice@example.com"))
    user1 = result1.scalar_one_or_none()
    assert user1 is not None
    
    result2 = await db_session.execute(select(User).where(User.email == "bob@example.com"))
    user2 = result2.scalar_one_or_none()
    assert user2 is not None
    
    # But if we try to execute both "at once" (without await), we'd get the error
    # This test verifies our fix ensures proper awaiting


async def test_batch_mutations_three_in_sequence(db_session, populated_db):
    """Three mutations to increase probability of concurrent operations error."""
    mutation = """
    mutation ThreeMutations {
      m1: merge_posts(payload: [{title: "M1", content: "C1", author_email: "alice@example.com"}]) { id author_id }
      m2: merge_posts(payload: [{title: "M2", content: "C2", author_email: "bob@example.com"}]) { id author_id }
      m3: merge_posts(payload: [{title: "M3", content: "C3", author_email: "charlie@example.com"}]) { id author_id }
    }
    """
    
    res = await schema.execute(mutation, context_value={"db_session": db_session})
    
    if res.errors:
        error_str = str(res.errors[0])
        if any(x in error_str.lower() for x in ["concurrent", "provisioning", "isce"]):
            pytest.fail(f"CONCURRENT OPS ERROR (3 mutations):\n{error_str[:500]}")
        else:
            pytest.fail(f"Unexpected error: {res.errors}")
    
    assert res.data is not None
    assert res.data["m1"][0]["author_id"] == 1
    assert res.data["m2"][0]["author_id"] == 2
    assert res.data["m3"][0]["author_id"] == 3


async def test_batch_mutations_with_nested_relations(db_session, populated_db):
    """Batch mutations that also query nested relations - more DB operations."""
    mutation = """
    mutation BatchWithRelations {
      first: merge_posts(payload: [{
        title: "Post with nested",
        content: "Content",
        author_email: "alice@example.com"
      }]) {
        id
        title
        author {
          id
          name
          posts {
            id
            title
          }
        }
      }
      second: merge_posts(payload: [{
        title: "Another nested",
        content: "Content 2",
        author_email: "bob@example.com"
      }]) {
        id
        title
        author {
          id
          name
          post_comments {
            id
            content
          }
        }
      }
    }
    """
    
    res = await schema.execute(mutation, context_value={"db_session": db_session})
    
    if res.errors:
        error_str = str(res.errors[0])
        if any(x in error_str.lower() for x in ["concurrent", "provisioning", "isce"]):
            pytest.fail(
                f"CONCURRENT OPS ERROR WITH NESTED RELATIONS:\n{error_str[:500]}\n\n"
                f"Nested relation queries may be executing while mutation is still pending."
            )
        else:
            pytest.fail(f"Unexpected error: {res.errors}")
    
    assert res.data is not None
    assert res.data["first"][0]["author"]["name"] == "Alice Johnson"
    assert res.data["second"][0]["author"]["name"] == "Bob Smith"


async def test_batch_mutations_with_updates(db_session, populated_db):
    """Test batch mutations that UPDATE existing records (pass id).
    
    This is different from creates - updates need to:
    1. Query existing record by id
    2. Apply changes
    3. Flush changes
    
    With parallel execution, all these database operations happen concurrently.
    """
    # Get existing post IDs from populated_db
    posts = populated_db['posts']
    post_ids = [p.id for p in posts[:3]]  # Use first 3 posts
    
    mutation = f"""
    mutation BatchUpdates {{
      u1: merge_posts(payload: [{{
        id: {post_ids[0]},
        title: "Updated Title 1",
        author_email: "alice@example.com"
      }}]) {{
        id
        title
        author_id
      }}
      u2: merge_posts(payload: [{{
        id: {post_ids[1]},
        title: "Updated Title 2",
        author_email: "bob@example.com"
      }}]) {{
        id
        title
        author_id
      }}
      u3: merge_posts(payload: [{{
        id: {post_ids[2]},
        title: "Updated Title 3",
        author_email: "charlie@example.com"
      }}]) {{
        id
        title
        author_id
      }}
    }}
    """
    
    res = await schema.execute(mutation, context_value={"db_session": db_session})
    
    if res.errors:
        error_str = str(res.errors[0])
        if any(x in error_str.lower() for x in ["concurrent", "provisioning", "isce", "connection is busy"]):
            pytest.fail(
                f"✓✓✓ REPRODUCED WITH UPDATE OPERATIONS ✓✓✓\n\n"
                f"{error_str[:600]}\n\n"
                f"Updates trigger concurrent session access:\n"
                f"- Each mutation queries for existing record by ID\n"
                f"- All queries happen in parallel using same session\n"
                f"- Result: concurrent operations error"
            )
        else:
            pytest.fail(f"Unexpected error with updates: {res.errors}")
    
    # Verify updates succeeded
    assert res.data is not None
    assert res.data["u1"][0]["title"] == "Updated Title 1"
    assert res.data["u2"][0]["title"] == "Updated Title 2"
    assert res.data["u3"][0]["title"] == "Updated Title 3"


async def test_mixed_creates_and_updates(db_session, populated_db):
    """Mix of CREATE and UPDATE operations in parallel - realistic scenario."""
    posts = populated_db['posts']
    existing_post_id = posts[0].id
    
    mutation = f"""
    mutation MixedOperations {{
      create1: merge_posts(payload: [{{title: "New Post 1", content: "C1", author_email: "alice@example.com"}}]) {{ id title }}
      update1: merge_posts(payload: [{{id: {existing_post_id}, title: "Updated Post", author_email: "bob@example.com"}}]) {{ id title }}
      create2: merge_posts(payload: [{{title: "New Post 2", content: "C2", author_email: "charlie@example.com"}}]) {{ id title }}
      create3: merge_posts(payload: [{{title: "New Post 3", content: "C3", author_email: "alice@example.com"}}]) {{ id title }}
      create4: merge_posts(payload: [{{title: "New Post 4", content: "C4", author_email: "bob@example.com"}}]) {{ id title }}
      create5: merge_posts(payload: [{{title: "New Post 5", content: "C5", author_email: "charlie@example.com"}}]) {{ id title }}
    }}
    """
    
    res = await schema.execute(mutation, context_value={"db_session": db_session})
    
    if res.errors:
        error_str = str(res.errors[0])
        if any(x in error_str.lower() for x in ["concurrent", "provisioning", "isce"]):
            pytest.fail(
                f"✓ REPRODUCED WITH MIXED CREATE/UPDATE ✓\n\n"
                f"{error_str[:600]}"
            )
        else:
            pytest.fail(f"Unexpected error: {res.errors}")
    
    assert res.data is not None
    assert res.data["create1"][0]["title"] == "New Post 1"
    assert res.data["update1"][0]["title"] == "Updated Post"
    assert res.data["create2"][0]["title"] == "New Post 2"


async def test_domain_mutations_parallel(db_session, populated_db):
    """Test parallel mutations nested under SINGLE domain - key pattern for concurrent errors.
    
    This pattern is critical: multiple mutations within the SAME domain block.
    When you have:
      mutation {
        blogDomain {
          mergePost1: merge_posts(...) { ... }
          mergePost2: merge_posts(...) { ... }
        }
      }
    
    Both mutations execute in parallel within the same domain scope,
    both use the same session, and both trigger pre-hooks that query the DB.
    This is a VERY common pattern that triggers concurrent operations errors.
    """
    # Get existing post IDs for updates
    posts = populated_db['posts']
    post_id_1 = posts[0].id
    post_id_2 = posts[1].id
    
    mutation = f"""
    mutation DomainParallelMutations {{
      blogDomain {{
        mergePost1: merge_posts(payload: [{{
          id: {post_id_1},
          title: "Updated Domain Post 1",
          author_email: "alice@example.com"
        }}]) {{
          id
          title
          author_id
        }}
        mergePost2: merge_posts(payload: [{{
          id: {post_id_2},
          title: "Updated Domain Post 2",
          author_email: "bob@example.com"
        }}]) {{
          id
          title
          author_id
        }}
      }}
    }}
    """
    
    res = await schema.execute(mutation, context_value={"db_session": db_session})
    
    if res.errors:
        error_str = str(res.errors[0])
        if any(x in error_str.lower() for x in ["concurrent", "provisioning", "isce", "connection is busy", "already flushing"]):
            pytest.fail(
                f"✓✓✓ REPRODUCED WITH PARALLEL MUTATIONS IN SINGLE DOMAIN ✓✓✓\n\n"
                f"{error_str[:600]}\n\n"
                f"CRITICAL PATTERN: Multiple mutations within the SAME domain block.\n"
                f"Both mergePost1 and mergePost2 execute in parallel,\n"
                f"both access the same session concurrently through their pre-hooks.\n"
                f"This is a very common usage pattern that triggers the error."
            )
        else:
            pytest.fail(f"Unexpected error with domain mutations: {res.errors}")
    
    assert res.data is not None
    assert res.data["blogDomain"]["mergePost1"] is not None
    assert res.data["blogDomain"]["mergePost2"] is not None
    assert res.data["blogDomain"]["mergePost1"][0]["title"] == "Updated Domain Post 1"
    assert res.data["blogDomain"]["mergePost2"][0]["title"] == "Updated Domain Post 2"
    assert res.data["blogDomain"]["mergePost1"][0]["author_id"] == 1  # alice
    assert res.data["blogDomain"]["mergePost2"][0]["author_id"] == 2  # bob


async def test_mixed_root_and_domain_mutations(db_session, populated_db):
    """Mix of root-level and domain-nested mutations in parallel."""
    mutation = """
    mutation MixedRootAndDomain {
      root1: merge_posts(payload: [{
        title: "Root Post 1",
        content: "C1",
        author_email: "alice@example.com"
      }]) {
        id
        title
      }
      domain1: blogDomain {
        merge_posts(payload: [{
          title: "Domain Post 1",
          content: "C2",
          author_email: "bob@example.com"
        }]) {
          id
          title
        }
      }
      root2: merge_posts(payload: [{
        title: "Root Post 2",
        content: "C3",
        author_email: "charlie@example.com"
      }]) {
        id
        title
      }
      domain2: blogDomain {
        merge_posts(payload: [{
          title: "Domain Post 2",
          content: "C4",
          author_email: "alice@example.com"
        }]) {
          id
          title
        }
      }
      root3: merge_posts(payload: [{
        title: "Root Post 3",
        content: "C5",
        author_email: "bob@example.com"
      }]) {
        id
        title
      }
    }
    """
    
    res = await schema.execute(mutation, context_value={"db_session": db_session})
    
    if res.errors:
        error_str = str(res.errors[0])
        if any(x in error_str.lower() for x in ["concurrent", "provisioning", "isce"]):
            pytest.fail(
                f"✓ REPRODUCED WITH MIXED ROOT/DOMAIN ✓\n\n"
                f"{error_str[:600]}\n\n"
                f"Mixing root and domain mutations doesn't help -\n"
                f"they all share the same session."
            )
        else:
            pytest.fail(f"Unexpected error: {res.errors}")
    
    assert res.data is not None
    assert res.data["root1"][0]["title"] == "Root Post 1"
    assert res.data["domain1"]["merge_posts"][0]["title"] == "Domain Post 1"
    assert res.data["root2"][0]["title"] == "Root Post 2"
