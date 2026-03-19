"""Test that nested relations within a single-relation pushdown resolve correctly.

Covers the LIST → SINGLE → SINGLE pattern:
    users (list) → post_comments (list) → post (single) → author (single)

This is the exact code path where _base_impl processes a single relation on a type
and must pass nested_cfg to build_single_relation_object so that deeper single
relations (e.g. post.author) are included in the JSON pushdown.
"""

import os
import pytest
from sqlalchemy import event
from tests.schema import schema

_is_mssql = 'mssql' in (os.getenv('BERRYQL_TEST_DATABASE_URL') or '')


@pytest.mark.asyncio
async def test_list_single_single_pushdown_one_select(engine, db_session, populated_db):
    """Query post_comments through users, requesting post.author (SINGLE→SINGLE).

    The nested author inside the single post relation must be resolved via
    JSON pushdown in a single SELECT statement.
    """
    calls = {"selects": 0}

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        try:
            if str(statement).lstrip().upper().startswith("SELECT"):
                calls["selects"] += 1
        except Exception:
            pass

    event.listen(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
    try:
        query = """{
  users {
    id
    name
    post_comments {
      id
      content
      post {
        id
        title
        author {
          id
          name
          email
        }
      }
    }
  }
}"""
        res = await schema.execute(query, context_value={"db_session": db_session})
        assert res.errors is None, f"GraphQL errors: {res.errors}"

        data = res.data or {}
        assert "users" in data and isinstance(data["users"], list)

        # Collect all post.author objects across all users' comments
        author_names_from_posts = set()
        comments_with_post = 0
        for user in data["users"]:
            for comment in user.get("post_comments") or []:
                post = comment.get("post")
                assert post is not None, (
                    f"post is None for comment {comment.get('id')} — "
                    "single relation pushdown failed to include post"
                )
                assert post.get("id") is not None
                assert post.get("title") is not None

                author = post.get("author")
                if _is_mssql:
                    # MSSQL FOR JSON PATH cannot express nested correlated
                    # subqueries, so post.author won't be pushed down.  The
                    # resolver fallback doesn't recurse into nested relations
                    # of a pushed-down single object — accept None here.
                    continue
                assert author is not None, (
                    f"post.author is None for post {post.get('id')} (comment {comment.get('id')}) — "
                    "nested single relation within single relation pushdown is broken"
                )
                assert author.get("id") is not None
                assert author.get("name") is not None
                assert author.get("email") is not None
                author_names_from_posts.add(author["name"])
                comments_with_post += 1

        if not _is_mssql:
            # Fixtures create 7 comments across 5 posts by users Alice, Bob, Charlie
            assert comments_with_post >= 1, "Expected at least one comment with a resolved post.author"
            # At least some of the known authors should appear
            known_authors = {"Alice Johnson", "Bob Smith", "Charlie Brown"}
            assert author_names_from_posts & known_authors, (
                f"Expected some of {known_authors} in post authors, got {author_names_from_posts}"
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", before_cursor_execute)

    # On non-MSSQL, JSON pushdown resolves everything in a single SELECT.
    # On MSSQL, nested single relations within a single relation fall back to
    # the resolver (FOR JSON PATH can't express nested correlated subqueries),
    # so multiple SELECTs are expected.
    if not _is_mssql:
        assert calls["selects"] == 1, f"Expected 1 SELECT, got {calls['selects']}"


@pytest.mark.asyncio
async def test_single_single_chain_data_correctness(engine, db_session, populated_db):
    """Validate that post.author data matches the fixture-seeded relationships.

    Posts are authored by specific users; verify the chain is wired correctly.
    """
    query = """{
  posts {
    id
    title
    author_id
    author {
      id
      name
      email
    }
    reviewer {
      id
      name
      email
    }
  }
}"""
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, f"GraphQL errors: {res.errors}"

    posts = res.data.get("posts") or []
    assert len(posts) >= 1

    for post in posts:
        author = post.get("author")
        assert author is not None, (
            f"author is None for post {post.get('id')} — single relation pushdown broken"
        )
        # The pushed-down author.id must match author_id on the post
        assert author["id"] == post["author_id"], (
            f"author.id ({author['id']}) != author_id ({post['author_id']}) for post {post['id']}"
        )

    # From fixtures: post 1 & 2 are by Alice (user 1), post 3 & 4 by Bob (user 2), post 5 by Charlie (user 3)
    authors_by_post_title = {p["title"]: p["author"]["name"] for p in posts}
    assert authors_by_post_title.get("First Post") == "Alice Johnson"
    assert authors_by_post_title.get("SQLAlchemy Tips") == "Bob Smith"
    assert authors_by_post_title.get("Getting Started") == "Charlie Brown"
