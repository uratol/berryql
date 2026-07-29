"""Tests for nested relation re-parenting during merge mutations.

Scenario under test (from the user request):
    merge_posts(payload: [{ id: 10, post_comments: [{ id: 123, content: "bar", post_id: 11 }] }])

A child record referenced inside a parent's nested relation list may declare a foreign key
pointing to a DIFFERENT parent than the one it is nested under. In that case the input value
wins: the child is edited AND re-parented onto the post identified by its own FK
(``post_id=11`` in the example).

Additionally, the security scope that applies to the POSTS (e.g. ``merge_posts_scoped``
with ``scope={"author_id": {"eq": 1}}``) MUST be enforced against BOTH:
  * the origin parent (the post the comment currently belongs to) and
  * the target parent (the post the comment is being moved to)
"""

import pytest

from tests.schema import schema


async def _fetch_comment_post_id(db_session, comment_id):
    """Read the persisted post_id/content of a PostComment by scanning posts' comments.

    ``post_comments`` is not a top-level Query root, so we traverse the ``posts``
    relation and locate the comment by id.
    """
    q = (
        """
        query {
          posts {
            id
            post_comments { id content post_id }
          }
        }
        """
    )
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    for post in res.data["posts"]:
        for c in post["post_comments"]:
            if int(c["id"]) == int(comment_id):
                return int(c["post_id"]), c["content"]
    raise AssertionError(f"comment {comment_id} not found in any post")


@pytest.mark.asyncio
async def test_nested_comment_reparent_unscoped(db_session, populated_db):
    """Basic re-parenting: comment currently on post1, nested under post1, with
    input ``post_id=post2.id`` -> comment is moved to post2."""
    post1 = populated_db["posts"][0]
    post2 = populated_db["posts"][1]
    # comment currently belonging to post1
    comment = populated_db["post_comments"][0]
    assert int(comment.post_id) == int(post1.id)

    mutation = (
        """
        mutation Upsert($payload: [PostQLInput!]!) {
          merge_posts(payload: $payload) {
            id
            post_comments { id content post_id }
          }
        }
        """
    )
    variables = {
        "payload": [
            {
                "id": int(post1.id),
                "post_comments": [
                    {
                        "id": int(comment.id),
                        "content": "reparented",
                        "post_id": int(post2.id),
                    }
                ],
            }
        ]
    }
    res = await schema.execute(
        mutation,
        variable_values=variables,
        context_value={"db_session": db_session},
    )
    assert res.errors is None, res.errors

    # Verify the comment now lives on post2
    new_post_id, new_content = await _fetch_comment_post_id(db_session, comment.id)
    assert new_post_id == int(post2.id), f"expected comment moved to post {post2.id}, got {new_post_id}"
    assert new_content == "reparented"


@pytest.mark.asyncio
async def test_nested_comment_reparent_scoped_target_allowed(db_session, populated_db):
    """Scoped merge (author_id==1): comment on post1 (author=user1), nested under post1,
    re-parented onto post2 (also author=user1, in scope) -> allowed."""
    post1 = populated_db["posts"][0]
    post2 = populated_db["posts"][1]
    comment = populated_db["post_comments"][0]
    # Sanity: both posts belong to the in-scope author (user1, id=1)
    assert int(post1.author_id) == 1
    assert int(post2.author_id) == 1

    mutation = (
        """
        mutation Upsert($payload: [PostQLInput!]!) {
          merge_posts_scoped(payload: $payload) {
            id
            post_comments { id content post_id }
          }
        }
        """
    )
    variables = {
        "payload": [
            {
                "id": int(post1.id),
                "post_comments": [
                    {
                        "id": int(comment.id),
                        "content": "moved-in-scope",
                        "post_id": int(post2.id),
                    }
                ],
            }
        ]
    }
    res = await schema.execute(
        mutation,
        variable_values=variables,
        context_value={"db_session": db_session},
    )
    assert res.errors is None, res.errors

    new_post_id, new_content = await _fetch_comment_post_id(db_session, comment.id)
    assert new_post_id == int(post2.id), f"expected comment moved to post {post2.id}, got {new_post_id}"
    assert new_content == "moved-in-scope"


@pytest.mark.asyncio
async def test_nested_comment_reparent_scoped_target_blocked(db_session, populated_db):
    """Scoped merge (author_id==1): comment on post1 (in scope), nested under post1,
    re-parented onto post3 (author=user2, OUT of scope) -> must be rejected."""
    post1 = populated_db["posts"][0]
    post3 = populated_db["posts"][2]
    comment = populated_db["post_comments"][0]
    assert int(post1.author_id) == 1
    assert int(post3.author_id) == 2  # out of scope
    post1_id = int(post1.id)
    post3_id = int(post3.id)
    comment_id = int(comment.id)

    mutation = (
        """
        mutation Upsert($payload: [PostQLInput!]!) {
          merge_posts_scoped(payload: $payload) {
            id
            post_comments { id content post_id }
          }
        }
        """
    )
    variables = {
        "payload": [
            {
                "id": post1_id,
                "post_comments": [
                    {
                        "id": comment_id,
                        "content": "should-not-move",
                        "post_id": post3_id,
                    }
                ],
            }
        ]
    }
    res = await schema.execute(
        mutation,
        variable_values=variables,
        context_value={"db_session": db_session},
    )
    assert res.errors is not None, "expected scope violation when re-parenting to out-of-scope post"
    msg = str(res.errors[0])
    assert "out of scope" in msg.lower(), msg

    # Confirm the comment did NOT move (still on post1, content unchanged)
    new_post_id, new_content = await _fetch_comment_post_id(db_session, comment_id)
    assert new_post_id == post1_id, f"comment must remain on post {post1_id}, got {new_post_id}"
    assert new_content != "should-not-move"


@pytest.mark.asyncio
async def test_nested_comment_reparent_cross_origin_allowed(db_session, populated_db):
    """Re-parent where the child currently belongs to a DIFFERENT parent than the nesting
    parent. Both origin (post2) and target (post1) are in scope (author=user1) -> allowed,
    even though the child is nested under post1 while currently living on post2."""
    post1 = populated_db["posts"][0]
    post2 = populated_db["posts"][1]
    # comment currently belonging to post2 (a different parent than the nesting one)
    comment = populated_db["post_comments"][2]  # c3 lives on post2
    assert int(comment.post_id) == int(post2.id)
    assert int(post1.author_id) == 1 and int(post2.author_id) == 1  # both in scope

    mutation = (
        """
        mutation Upsert($payload: [PostQLInput!]!) {
          merge_posts_scoped(payload: $payload) {
            id
            post_comments { id content post_id }
          }
        }
        """
    )
    variables = {
        "payload": [
            {
                "id": int(post1.id),
                "post_comments": [
                    {
                        "id": int(comment.id),
                        "content": "cross-origin-move",
                        "post_id": int(post1.id),
                    }
                ],
            }
        ]
    }
    res = await schema.execute(
        mutation,
        variable_values=variables,
        context_value={"db_session": db_session},
    )
    assert res.errors is None, res.errors

    new_post_id, new_content = await _fetch_comment_post_id(db_session, comment.id)
    assert new_post_id == int(post1.id), f"expected comment moved to post {post1.id}, got {new_post_id}"
    assert new_content == "cross-origin-move"


@pytest.mark.asyncio
async def test_nested_comment_reparent_origin_out_of_scope_blocked(db_session, populated_db):
    """Re-parent blocked when the ORIGIN parent (where the child currently lives) is out of
    scope. Comment currently on post3 (author=user2, out of scope), nested under post1, with
    target post2 (in scope). Must be rejected because origin is out of scope."""
    post1 = populated_db["posts"][0]
    post2 = populated_db["posts"][1]
    post3 = populated_db["posts"][2]
    # comment currently belonging to post3 (out of scope)
    comment = populated_db["post_comments"][3]  # c4 lives on post3
    assert int(comment.post_id) == int(post3.id)
    assert int(post3.author_id) == 2  # origin out of scope
    assert int(post2.author_id) == 1  # target in scope
    post1_id = int(post1.id)
    post2_id = int(post2.id)
    post3_id = int(post3.id)
    comment_id = int(comment.id)

    mutation = (
        """
        mutation Upsert($payload: [PostQLInput!]!) {
          merge_posts_scoped(payload: $payload) {
            id
            post_comments { id content post_id }
          }
        }
        """
    )
    variables = {
        "payload": [
            {
                "id": post1_id,
                "post_comments": [
                    {
                        "id": comment_id,
                        "content": "origin-oob",
                        "post_id": post2_id,
                    }
                ],
            }
        ]
    }
    res = await schema.execute(
        mutation,
        variable_values=variables,
        context_value={"db_session": db_session},
    )
    assert res.errors is not None, "expected scope violation when origin parent is out of scope"
    msg = str(res.errors[0])
    assert "out of scope" in msg.lower(), msg

    # Comment must remain on post3, unchanged
    new_post_id, new_content = await _fetch_comment_post_id(db_session, comment_id)
    assert new_post_id == post3_id, f"comment must remain on post {post3_id}, got {new_post_id}"
    assert new_content != "origin-oob"
