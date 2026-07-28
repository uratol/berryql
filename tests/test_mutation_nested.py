import pytest

from tests.schema import schema
from tests.models import Post


@pytest.mark.asyncio
async def test_upsert_post_with_nested_comments_and_likes(db_session, populated_db):
  # Use top-level auto-generated mutation merge_posts with nested relations
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]
    u3 = populated_db['users'][2]

    mutation = (
        """
  mutation Upsert($payload: [PostQLInput!]!) {
          merge_posts(payload: $payload) {
            id
            title
            author_id
            post_comments(order_by: \"id\", order_dir: desc) {
              id
              content
              author_id
              likes(order_by: \"id\") { id user_id }
            }
          }
        }
        """
    )
    variables = {
    "payload": [{
            "title": "Nested Create",
            "content": "Body",
            "author_id": int(u1.id),
            "post_comments": [
                {"content": "c1", "author_id": int(u2.id)},
                {
                    "content": "c2",
                    "author_id": int(u3.id),
                    # nested likes for this comment
                    "likes": [
                        {"user_id": int(u1.id)},
                        {"user_id": int(u2.id)}
                    ]
                },
            ],
    }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    post_list = res.data["merge_posts"]
    assert isinstance(post_list, list)
    assert len(post_list) == 1
    post = post_list[0]
    assert post["title"] == "Nested Create"
    assert int(post["author_id"]) == int(u1.id)
    pcs = post["post_comments"]
    assert isinstance(pcs, list) and len(pcs) == 2
    # Find the one with likes and validate
    pc_with_likes = next((x for x in pcs if x.get("content") == "c2"), None)
    assert pc_with_likes is not None
    likes = pc_with_likes["likes"]
    assert isinstance(likes, list) and len(likes) == 2

    # Query back to ensure persistence and FK wiring
    q = """
    query {
      posts(where: "{\\\"id\\\": {\\\"eq\\\": %d}}") {
        id
        post_comments { id content likes { id user_id } }
      }
    }
    """ % int(post["id"])
    res2 = await schema.execute(q, context_value={"db_session": db_session})
    assert res2.errors is None, res2.errors
    posts = res2.data["posts"]
    assert len(posts) == 1
    assert len(posts[0]["post_comments"]) == 2
    # the comment with likes should still have 2 likes
    pc2 = next((x for x in posts[0]["post_comments"] if x.get("content") == "c2"), None)
    assert pc2 is not None
    assert len(pc2["likes"]) == 2


@pytest.mark.asyncio
async def test_nested_comment_merge_coerces_iso_datetime_strings(db_session, populated_db):
    u1 = populated_db["users"][0]
    u2 = populated_db["users"][1]
    created_iso = "2026-03-19T12:34:56"
    updated_iso = "2026-03-20T08:09:10"

    create_mutation = """
        mutation Upsert($payload: PostQLInput!) {
            merge_post(payload: $payload) {
                id
                post_comments(order_by: "id") {
                    id
                    content
                    created_at
                }
            }
        }
    """
    create_variables = {
        "payload": {
            "title": "Datetime Nested Create",
            "content": "Body",
            "author_id": int(u1.id),
            "post_comments": [
                {
                    "content": "dt comment",
                    "author_id": int(u2.id),
                    "created_at": created_iso,
                }
            ],
        }
    }
    res1 = await schema.execute(create_mutation, variable_values=create_variables, context_value={"db_session": db_session})
    assert res1.errors is None, res1.errors
    post = res1.data["merge_post"]
    comment = next((row for row in post["post_comments"] if row["content"] == "dt comment"), None)
    assert comment is not None
    assert str(comment["created_at"]).startswith(created_iso)

    post_id = int(post["id"])
    comment_id = int(comment["id"])

    update_mutation = """
        mutation Upsert($payload: PostQLInput!) {
            merge_post(payload: $payload) {
                id
                post_comments(order_by: "id") {
                    id
                    content
                    created_at
                }
            }
        }
    """
    update_variables = {
        "payload": {
            "id": post_id,
            "post_comments": [
                {
                    "id": comment_id,
                    "content": "dt comment updated",
                    "created_at": updated_iso,
                }
            ],
        }
    }
    res2 = await schema.execute(update_mutation, variable_values=update_variables, context_value={"db_session": db_session})
    assert res2.errors is None, res2.errors
    updated_comment = next((row for row in res2.data["merge_post"]["post_comments"] if int(row["id"]) == comment_id), None)
    assert updated_comment is not None
    assert updated_comment["content"] == "dt comment updated"
    assert str(updated_comment["created_at"]).startswith(updated_iso)

    query = """
    query {
      posts(where: "{\\\"id\\\": {\\\"eq\\\": %d}}") {
        id
        post_comments(order_by: \"id\") {
          id
          content
          created_at
        }
      }
    }
    """ % post_id
    res3 = await schema.execute(query, context_value={"db_session": db_session})
    assert res3.errors is None, res3.errors
    persisted_comment = next(
        (row for row in res3.data["posts"][0]["post_comments"] if int(row["id"]) == comment_id),
        None,
    )
    assert persisted_comment is not None
    assert persisted_comment["content"] == "dt comment updated"
    assert str(persisted_comment["created_at"]).startswith(updated_iso)


@pytest.mark.asyncio
async def test_mutation_domain_upsert_posts(db_session, populated_db):
    # Upsert via domain-scoped mutation under blogDomain
    u1 = populated_db['users'][0]
    m = (
        """
    mutation($payload: [PostQLInput!]!) {
          blogDomain { 
            merge_posts(payload: $payload) {
              id
              title
            }
          }
        }
        """
    )
    variables = {
    "payload": [{
            "title": "Domain Upsert",
            "content": "From domain",
            "author_id": int(u1.id),
    }]
    }
    res = await schema.execute(m, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data_list = res.data["blogDomain"]["merge_posts"]
    assert isinstance(data_list, list)
    assert len(data_list) == 1
    data = data_list[0]
    assert data["title"] == "Domain Upsert"


@pytest.mark.asyncio
async def test_replace_relation_deletes_unlisted_children(db_session, populated_db):
    """_Replace should upsert listed children and delete the rest."""
    from tests.models import PostComment
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]
    u3 = populated_db['users'][2]

    # Seed a post with 3 comments
    post = Post(title="Replace Target", content="c", author_id=int(u1.id))
    db_session.add(post)
    await db_session.flush()
    c1 = PostComment(content="keep-me", post_id=post.id, author_id=int(u2.id))
    c2 = PostComment(content="update-me", post_id=post.id, author_id=int(u2.id))
    c3 = PostComment(content="delete-me", post_id=post.id, author_id=int(u3.id))
    db_session.add_all([c1, c2, c3])
    await db_session.flush()
    await db_session.commit()

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) {
            id
            post_comments(order_by: "id") { id content }
        }
    }
    """
    variables = {
        "payload": [{
            "id": int(post.id),
            "post_comments": [
                {"id": int(c2.id), "content": "updated"},   # update existing
                {"content": "new-one", "author_id": int(u2.id)},  # insert new
                # c1 (keep-me) is intentionally omitted but kept because it has no PK in payload? No:
                # kept set is rows WITH a pk in payload. c1 has no pk -> would be deleted.
            ],
            "_Replace": ["post_comments"],
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    pcs = res.data["merge_posts"][0]["post_comments"]
    contents = sorted(pc["content"] for pc in pcs)
    # c2 updated, new-one inserted; c1 and c3 deleted
    assert contents == ["new-one", "updated"], contents

    # Query back to confirm persistence
    q = """
    query {
        posts(where: "{\\\"id\\\": {\\\"eq\\\": %d}}") {
            post_comments(order_by: "id") { id content }
        }
    }
    """ % int(post.id)
    res2 = await schema.execute(q, context_value={"db_session": db_session})
    assert res2.errors is None, res2.errors
    persisted = sorted(pc["content"] for pc in res2.data["posts"][0]["post_comments"])
    assert persisted == ["new-one", "updated"], persisted


@pytest.mark.asyncio
async def test_replace_relation_keeps_rows_listed_by_pk(db_session, populated_db):
    """Rows whose PK is present in the payload should be kept (even if not updated)."""
    from tests.models import PostComment
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]

    post = Post(title="Keep By PK", content="c", author_id=int(u1.id))
    db_session.add(post)
    await db_session.flush()
    c1 = PostComment(content="keep-untouched", post_id=post.id, author_id=int(u2.id))
    c2 = PostComment(content="will-delete", post_id=post.id, author_id=int(u2.id))
    db_session.add_all([c1, c2])
    await db_session.flush()
    await db_session.commit()

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) {
            id
            post_comments(order_by: "id") { id content }
        }
    }
    """
    # List c1 by PK only (no content change) -> kept; c2 omitted -> deleted
    variables = {
        "payload": [{
            "id": int(post.id),
            "post_comments": [
                {"id": int(c1.id)},  # keep, untouched
            ],
            "_Replace": ["post_comments"],
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    pcs = res.data["merge_posts"][0]["post_comments"]
    assert len(pcs) == 1
    assert int(pcs[0]["id"]) == int(c1.id)
    assert pcs[0]["content"] == "keep-untouched"


@pytest.mark.asyncio
async def test_replace_relation_deletes_all_when_no_items(db_session, populated_db):
    """_Replace with an absent/empty relation list should delete ALL children of that relation."""
    from tests.models import PostComment
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]

    post = Post(title="Clear All", content="c", author_id=int(u1.id))
    db_session.add(post)
    await db_session.flush()
    c1 = PostComment(content="a", post_id=post.id, author_id=int(u2.id))
    c2 = PostComment(content="b", post_id=post.id, author_id=int(u2.id))
    db_session.add_all([c1, c2])
    await db_session.flush()
    await db_session.commit()

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) {
            id
            post_comments { id }
        }
    }
    """
    # No post_comments provided, but _Replace names it -> all comments deleted
    variables = {
        "payload": [{
            "id": int(post.id),
            "_Replace": ["post_comments"],
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    pcs = res.data["merge_posts"][0]["post_comments"]
    assert pcs == [], pcs


@pytest.mark.asyncio
async def test_replace_relation_cascades_to_grandchildren(db_session, populated_db):
    """_Replace deleting a comment should also delete its likes (grandchildren)."""
    from tests.models import PostComment, PostCommentLike
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]

    post = Post(title="Cascade Replace", content="c", author_id=int(u1.id))
    db_session.add(post)
    await db_session.flush()
    c1 = PostComment(content="survives", post_id=post.id, author_id=int(u2.id))
    c2 = PostComment(content="dies", post_id=post.id, author_id=int(u2.id))
    db_session.add_all([c1, c2])
    await db_session.flush()
    like1 = PostCommentLike(post_comment_id=c1.id, user_id=int(u1.id))
    like2 = PostCommentLike(post_comment_id=c2.id, user_id=int(u1.id))
    db_session.add_all([like1, like2])
    await db_session.flush()
    await db_session.commit()

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) {
            id
            post_comments(order_by: "id") { id content }
        }
    }
    """
    variables = {
        "payload": [{
            "id": int(post.id),
            "post_comments": [
                {"id": int(c1.id)},  # keep
            ],
            "_Replace": ["post_comments"],
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors

    # Verify the like tied to the deleted comment is gone
    from sqlalchemy import select
    remaining_likes = (await db_session.execute(
        select(PostCommentLike).where(PostCommentLike.post_comment_id == int(c2.id))
    )).scalars().all()
    assert remaining_likes == [], "grandchild likes should be cascade-deleted"
    # And the surviving comment's like remains
    surviving_likes = (await db_session.execute(
        select(PostCommentLike).where(PostCommentLike.post_comment_id == int(c1.id))
    )).scalars().all()
    assert len(surviving_likes) == 1


@pytest.mark.asyncio
async def test_replace_unknown_relation_name_errors(db_session, populated_db):
    """_Replace referencing an unknown relation should raise a validation error."""
    u1 = populated_db['users'][0]

    post = Post(title="Bad Rel", content="c", author_id=int(u1.id))
    db_session.add(post)
    await db_session.flush()
    await db_session.commit()

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) { id }
    }
    """
    variables = {
        "payload": [{
            "id": int(post.id),
            "_Replace": ["does_not_exist"],
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is not None
    assert "does_not_exist" in str(res.errors[0])


@pytest.mark.asyncio
async def test_replace_single_relation_errors(db_session, populated_db):
    """_Replace cannot target a single relation (reviewer)."""
    u1 = populated_db['users'][0]

    post = Post(title="Single Rel", content="c", author_id=int(u1.id))
    db_session.add(post)
    await db_session.flush()
    await db_session.commit()

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) { id }
    }
    """
    variables = {
        "payload": [{
            "id": int(post.id),
            "_Replace": ["reviewer"],  # single relation -> error
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is not None
    assert "reviewer" in str(res.errors[0])


@pytest.mark.asyncio
async def test_replace_does_not_affect_other_relations(db_session, populated_db):
    """_Replace on post_comments should not touch other relations (e.g. none specified)."""
    from tests.models import PostComment
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]

    post = Post(title="Scoped Replace", content="c", author_id=int(u1.id))
    db_session.add(post)
    await db_session.flush()
    c1 = PostComment(content="kept", post_id=post.id, author_id=int(u2.id))
    c2 = PostComment(content="gone", post_id=post.id, author_id=int(u2.id))
    db_session.add_all([c1, c2])
    await db_session.flush()
    await db_session.commit()

    # Without _Replace, both comments must survive even though only c1 is listed
    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) {
            id
            post_comments(order_by: "id") { id content }
        }
    }
    """
    variables = {
        "payload": [{
            "id": int(post.id),
            "post_comments": [
                {"id": int(c1.id), "content": "kept-updated"},
            ],
            # NOTE: no _Replace here
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    pcs = res.data["merge_posts"][0]["post_comments"]
    contents = sorted(pc["content"] for pc in pcs)
    # Both survive: c1 updated, c2 untouched (merge semantics)
    assert contents == ["gone", "kept-updated"], contents


@pytest.mark.asyncio
async def test_insert_flag_inserts_with_provided_pk(db_session, populated_db):
    """_Insert: true should insert a NEW row using the provided PK value, not update."""
    from tests.models import Post
    from sqlalchemy import select
    u1 = populated_db['users'][0]

    # Seed an existing post with a known PK
    existing = Post(title="Original", content="orig", author_id=int(u1.id))
    db_session.add(existing)
    await db_session.flush()
    await db_session.commit()
    existing_pk = int(existing.id)

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) {
            id
            title
            content
        }
    }
    """
    # Pass a DIFFERENT (non-existing) PK with _Insert: true -> new row inserted with that exact PK
    new_pk = existing_pk + 1000
    variables = {
        "payload": [{
            "id": new_pk,
            "title": "Inserted With PK",
            "content": "new body",
            "author_id": int(u1.id),
            "_Insert": True,
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    created = res.data["merge_posts"][0]
    # The new row carries the exact provided PK
    assert int(created["id"]) == new_pk, "expected the provided PK to be used in the insert"
    assert created["title"] == "Inserted With PK"

    # The original row must be untouched (no update happened)
    orig = await db_session.get(Post, existing_pk)
    assert orig is not None
    assert orig.title == "Original"
    assert orig.content == "orig"

    # Confirm both rows exist
    rows = (await db_session.execute(
        select(Post).where(Post.id.in_([existing_pk, new_pk])).order_by(Post.id)
    )).scalars().all()
    assert {int(r.id) for r in rows} == {existing_pk, new_pk}


@pytest.mark.asyncio
async def test_insert_flag_with_existing_pk_raises(db_session, populated_db):
    """_Insert: true with an ALREADY-EXISTING PK should error (duplicate key)."""
    from tests.models import Post
    u1 = populated_db['users'][0]

    existing = Post(title="Original", content="orig", author_id=int(u1.id))
    db_session.add(existing)
    await db_session.flush()
    await db_session.commit()
    existing_pk = int(existing.id)

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) { id }
    }
    """
    # Pass the SAME existing PK with _Insert: true -> DB-level duplicate-key error
    variables = {
        "payload": [{
            "id": existing_pk,
            "title": "Dup",
            "author_id": int(u1.id),
            "_Insert": True,
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is not None, "expected a duplicate-key error"


@pytest.mark.asyncio
async def test_insert_flag_without_pk_behaves_like_create(db_session, populated_db):
    """_Insert: true without a PK behaves like a normal create (auto PK)."""
    from tests.models import Post
    u1 = populated_db['users'][0]

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) {
            id
            title
        }
    }
    """
    variables = {
        "payload": [{
            "title": "No PK Insert",
            "content": "body",
            "author_id": int(u1.id),
            "_Insert": True,
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    created = res.data["merge_posts"][0]
    assert created["title"] == "No PK Insert"
    assert int(created["id"]) > 0


@pytest.mark.asyncio
async def test_insert_flag_false_falls_back_to_update(db_session, populated_db):
    """_Insert: false (or absent) should still update an existing row matched by PK."""
    from tests.models import Post
    u1 = populated_db['users'][0]

    existing = Post(title="Before", content="c", author_id=int(u1.id))
    db_session.add(existing)
    await db_session.flush()
    await db_session.commit()
    existing_pk = int(existing.id)

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) { id title }
    }
    """
    # Explicit _Insert: false -> normal update semantics
    variables = {
        "payload": [{
            "id": existing_pk,
            "title": "After",
            "_Insert": False,
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    updated = res.data["merge_posts"][0]
    # Same PK, title updated -> it was an update, not an insert
    assert int(updated["id"]) == existing_pk
    assert updated["title"] == "After"

    # Ensure no extra row was created
    from sqlalchemy import select, func
    count = (await db_session.execute(select(func.count()).select_from(Post).where(Post.id == existing_pk))).scalar()
    assert count == 1


@pytest.mark.asyncio
async def test_insert_flag_on_nested_child_uses_provided_pk(db_session, populated_db):
    """_Insert: true on a nested child inserts a new child row with the provided PK."""
    from tests.models import Post, PostComment
    from sqlalchemy import select
    u1 = populated_db['users'][0]
    u2 = populated_db['users'][1]

    post = Post(title="Parent", content="c", author_id=int(u1.id))
    db_session.add(post)
    await db_session.flush()
    # Seed an existing comment under this post
    existing_comment = PostComment(content="orig comment", post_id=post.id, author_id=int(u2.id))
    db_session.add(existing_comment)
    await db_session.flush()
    await db_session.commit()
    existing_comment_pk = int(existing_comment.id)

    mutation = """
    mutation Upsert($payload: [PostQLInput!]!) {
        merge_posts(payload: $payload) {
            id
            post_comments(order_by: "id") { id content }
        }
    }
    """
    # Nested child uses a FRESH PK with _Insert: true -> new child row with that PK
    new_child_pk = existing_comment_pk + 1000
    variables = {
        "payload": [{
            "id": int(post.id),
            "post_comments": [
                {"id": new_child_pk, "content": "new comment", "author_id": int(u2.id), "_Insert": True},
            ],
        }]
    }
    res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    pcs = res.data["merge_posts"][0]["post_comments"]
    # Now there should be TWO comments: the original untouched + the newly inserted one
    assert len(pcs) == 2, pcs
    new_pc = next((pc for pc in pcs if pc["content"] == "new comment"), None)
    assert new_pc is not None
    # The new child carries the exact provided PK
    assert int(new_pc["id"]) == new_child_pk, "expected the provided child PK to be used in the insert"
    # Original comment untouched
    orig_pc = next((pc for pc in pcs if int(pc["id"]) == existing_comment_pk), None)
    assert orig_pc is not None
    assert orig_pc["content"] == "orig comment"

    # Confirm two distinct comment rows in DB
    rows = (await db_session.execute(
        select(PostComment).where(PostComment.post_id == int(post.id)).order_by(PostComment.id)
    )).scalars().all()
    assert len(rows) == 2
