from __future__ import annotations

from collections import OrderedDict

import pytest
from sqlalchemy import func, select

from berryql import (
    BerryType,
    MergeNodeContext,
    MergeOperationContext,
    field,
    relation,
)
from berryql.mutations import _ordered_relation_entries
from tests.models import Post, PostComment
from tests.schema import PostCommentQL, PostQL, UserQL, berry_schema, schema


def _relation_names(berry_type, values):
    return [
        name
        for name, _, _ in _ordered_relation_entries(berry_type, values)
    ]


def test_declaration_indices_are_local_and_define_inheritance_order():
    descriptor_created_early = relation("TargetQL")

    class BaseQL(BerryType):
        base_first = relation("TargetQL")
        moved_by_override = relation("TargetQL")

    class ChildQL(BaseQL):
        child_first = descriptor_created_early
        moved_by_override = relation("TargetQL")
        child_last = relation("TargetQL")

    class UnrelatedQL(BerryType):
        unrelated = relation("TargetQL")

    assert list(BaseQL.__berry_fields__) == [
        "base_first",
        "moved_by_override",
    ]
    assert list(ChildQL.__berry_fields__) == [
        "base_first",
        "child_first",
        "moved_by_override",
        "child_last",
    ]
    assert [
        definition.meta["declaration_index"]
        for definition in ChildQL.__berry_fields__.values()
    ] == [0, 1, 2, 3]
    assert UnrelatedQL.__berry_fields__["unrelated"].meta[
        "declaration_index"
    ] == 0


def test_payload_order_never_overrides_relation_declaration_order():
    class ForwardQL(BerryType):
        scalar = field()
        shots = relation("ShotQL")
        beats = relation("BeatQL")

    class ReverseQL(BerryType):
        beats = relation("BeatQL")
        shots = relation("ShotQL")

    beats_first = OrderedDict([("beats", []), ("shots", [])])
    shots_first = OrderedDict([("shots", []), ("beats", [])])

    assert _relation_names(ForwardQL, beats_first) == ["shots", "beats"]
    assert _relation_names(ForwardQL, shots_first) == ["shots", "beats"]
    assert _relation_names(ReverseQL, shots_first) == ["beats", "shots"]


@pytest.mark.asyncio
async def test_strawberry_input_order_does_not_control_relation_merge_order(
    db_session, populated_db
):
    relation_events = []

    def capture_post_pre(model_cls, info, data, ctx):
        relation_events.append("posts")
        assert isinstance(ctx["operation"], MergeOperationContext)
        assert isinstance(ctx["node"], MergeNodeContext)
        return data

    def capture_comment_pre(model_cls, info, data, ctx):
        relation_events.append("post_comments")
        return data

    original_post = tuple(getattr(PostQL, "__merge_pre_cbs__", ()) or ())
    original_comment = tuple(
        getattr(PostCommentQL, "__merge_pre_cbs__", ()) or ()
    )
    PostQL.__merge_pre_cbs__ = original_post + (capture_post_pre,)
    PostCommentQL.__merge_pre_cbs__ = original_comment + (
        capture_comment_pre,
    )
    try:
        user_id = populated_db["users"][0].id
        comment_author_id = user_id
        comment_post_id = populated_db["posts"][0].id
        mutation = """
            mutation($payload: [UserQLInput!]!) {
              merge_users(payload: $payload) { id }
            }
        """
        payloads = [
            OrderedDict(
                [
                    ("id", user_id),
                    (
                        "post_comments",
                        [
                            {
                                "content": "order-comment-one",
                                "rate": 1,
                                "author_id": comment_author_id,
                                "post_id": comment_post_id,
                            }
                        ],
                    ),
                    (
                        "posts",
                        [{"title": "order-post-one", "content": "one"}],
                    ),
                ]
            ),
            OrderedDict(
                [
                    ("id", user_id),
                    (
                        "posts",
                        [{"title": "order-post-two", "content": "two"}],
                    ),
                    (
                        "post_comments",
                        [
                            {
                                "content": "order-comment-two",
                                "rate": 2,
                                "author_id": comment_author_id,
                                "post_id": comment_post_id,
                            }
                        ],
                    ),
                ]
            ),
        ]
        for payload in payloads:
            relation_events.clear()
            result = await schema.execute(
                mutation,
                variable_values={"payload": [payload]},
                context_value={"db_session": db_session},
            )
            assert result.errors is None, result.errors
            # UserQL declares posts before post_comments.
            assert relation_events == ["posts", "post_comments"]
    finally:
        PostQL.__merge_pre_cbs__ = original_post
        PostCommentQL.__merge_pre_cbs__ = original_comment


@pytest.mark.asyncio
async def test_list_lifecycle_deferred_validation_and_shared_context(
    db_session, populated_db
):
    events = []
    operation_ids = []
    validation_calls = 0

    async def validator(info, operation, title_prefix):
        nonlocal validation_calls
        validation_calls += 1
        operation_ids.append(id(operation))
        count = await operation.session.scalar(
            select(func.count(Post.id)).where(
                Post.title.like(f"{title_prefix}%")
            )
        )
        events.append(("validate", count))

    def post_hook(model_cls, info, instance, created, ctx):
        operation = ctx.operation
        operation_ids.append(id(operation))
        events.append(("node", instance.title))
        operation.defer_validation(
            ("batch-post-check",),
            validator,
            title_prefix="operation-test-",
        )

    def before_merge(info, operation, payload):
        operation_ids.append(id(operation))
        events.append(("before_merge", len(payload)))

    def before_commit(info, operation, result):
        operation_ids.append(id(operation))
        events.append(("before_commit", len(result)))
        assert len(operation.touched[Post]) == 2

    def after_commit(info, operation, result):
        operation_ids.append(id(operation))
        events.append(("after_commit", len(result)))

    original_post = tuple(getattr(PostQL, "__merge_post_cbs__", ()) or ())
    original_lengths = {
        phase: len(callbacks)
        for phase, callbacks in berry_schema._merge_hooks.items()
    }
    PostQL.__merge_post_cbs__ = original_post + (post_hook,)
    berry_schema.merge_hooks(
        before_merge=before_merge,
        before_commit=before_commit,
        after_commit=after_commit,
    )
    try:
        author_id = populated_db["users"][0].id
        result = await schema.execute(
            """
            mutation($payload: [PostQLInput!]!) {
              merge_posts(payload: $payload) { id title }
            }
            """,
            variable_values={
                "payload": [
                    {
                        "title": "operation-test-one",
                        "content": "one",
                        "author_id": author_id,
                    },
                    {
                        "title": "operation-test-two",
                        "content": "two",
                        "author_id": author_id,
                    },
                ]
            },
            context_value={"db_session": db_session},
        )
        assert result.errors is None, result.errors
        assert validation_calls == 1
        assert len(set(operation_ids)) == 1
        assert events == [
            ("before_merge", 2),
            ("node", "operation-test-one"),
            ("node", "operation-test-two"),
            ("validate", 2),
            ("before_commit", 2),
            ("after_commit", 2),
        ]
    finally:
        PostQL.__merge_post_cbs__ = original_post
        for phase, length in original_lengths.items():
            del berry_schema._merge_hooks[phase][length:]


@pytest.mark.asyncio
async def test_deep_nodes_share_operation_context(db_session, populated_db):
    seen = []

    def capture(model_cls, info, data, ctx):
        seen.append((model_cls, ctx["operation"], ctx["node"]))
        return data

    original_user = tuple(getattr(UserQL, "__merge_pre_cbs__", ()) or ())
    original_post = tuple(getattr(PostQL, "__merge_pre_cbs__", ()) or ())
    UserQL.__merge_pre_cbs__ = original_user + (capture,)
    PostQL.__merge_pre_cbs__ = original_post + (capture,)
    try:
        user_id = populated_db["users"][0].id
        result = await schema.execute(
            """
            mutation($payload: [UserQLInput!]!) {
              merge_users(payload: $payload) { id }
            }
            """,
            variable_values={
                "payload": [
                    {
                        "id": user_id,
                        "posts": [
                            {"title": "deep-one", "content": "one"},
                            {"title": "deep-two", "content": "two"},
                        ],
                    }
                ]
            },
            context_value={"db_session": db_session},
        )
        assert result.errors is None, result.errors
        assert len(seen) == 3
        operations = [operation for _, operation, _ in seen]
        assert all(operation is operations[0] for operation in operations)
        root_node = seen[0][2]
        assert root_node.parent is None
        assert all(node.parent is root_node for _, _, node in seen[1:])
    finally:
        UserQL.__merge_pre_cbs__ = original_user
        PostQL.__merge_pre_cbs__ = original_post


@pytest.mark.asyncio
async def test_explicit_pk_from_earlier_sibling_is_available_to_later_sibling(
    db_session, populated_db
):
    user_id = populated_db["users"][0].id
    explicit_post_id = 90_001
    explicit_comment_id = 90_002

    result = await schema.execute(
        """
        mutation($payload: [UserQLInput!]!) {
          merge_users(payload: $payload) {
            id
            posts { id title }
            post_comments { id post_id }
          }
        }
        """,
        variable_values={
            "payload": [
                {
                    "id": user_id,
                    # UserQL declares posts before post_comments.
                    "posts": [
                        {
                            "id": explicit_post_id,
                            "_Insert": True,
                            "title": "client-identified-post",
                            "content": "body",
                        }
                    ],
                    "post_comments": [
                        {
                            "id": explicit_comment_id,
                            "_Insert": True,
                            "content": "references-earlier-sibling",
                            "rate": 1,
                            "author_id": user_id,
                            "post_id": explicit_post_id,
                        }
                    ],
                }
            ]
        },
        context_value={"db_session": db_session},
    )
    assert result.errors is None, result.errors
    created_post = await db_session.get(Post, explicit_post_id)
    created_comment = await db_session.get(PostComment, explicit_comment_id)
    assert created_post is not None
    assert created_post.title == "client-identified-post"
    assert created_comment is not None
    assert created_comment.post_id == explicit_post_id


@pytest.mark.asyncio
async def test_before_commit_failure_rolls_back_and_on_error_cannot_mask_it(
    db_session, populated_db, caplog
):
    events = []
    rollback_calls = 0
    original_rollback = db_session.rollback

    async def tracked_rollback():
        nonlocal rollback_calls
        rollback_calls += 1
        await original_rollback()

    def before_commit(info, operation, result):
        raise RuntimeError("original-before-commit-error")

    def after_commit(info, operation, result):
        events.append("after_commit")

    def failing_on_error(info, operation, exception):
        events.append(("on_error", str(exception), operation.session.in_transaction()))
        raise RuntimeError("secondary-on-error-error")

    original_lengths = {
        phase: len(callbacks)
        for phase, callbacks in berry_schema._merge_hooks.items()
    }
    db_session.rollback = tracked_rollback
    berry_schema.merge_hooks(
        before_commit=before_commit,
        after_commit=after_commit,
        on_error=failing_on_error,
    )
    try:
        author_id = populated_db["users"][0].id
        result = await schema.execute(
            """
            mutation($payload: PostQLInput!) {
              merge_post(payload: $payload) { id }
            }
            """,
            variable_values={
                "payload": {
                    "title": "must-roll-back",
                    "content": "body",
                    "author_id": author_id,
                }
            },
            context_value={"db_session": db_session},
        )
        assert result.errors
        assert "original-before-commit-error" in str(result.errors[0])
        assert "secondary-on-error-error" not in str(result.errors[0])
        assert rollback_calls == 1
        assert events == [
            ("on_error", "original-before-commit-error", False)
        ]
        assert not db_session.in_transaction()
        count = await db_session.scalar(
            select(func.count(Post.id)).where(Post.title == "must-roll-back")
        )
        assert count == 0
    finally:
        db_session.rollback = original_rollback
        for phase, length in original_lengths.items():
            del berry_schema._merge_hooks[phase][length:]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_phase", ["before_merge", "type_pre", "type_post", "validator"]
)
async def test_each_precommit_failure_phase_rolls_back(
    db_session, populated_db, failure_phase
):
    rollback_calls = 0
    error_events = []
    original_rollback = db_session.rollback
    title = f"failure-phase-{failure_phase}"

    async def tracked_rollback():
        nonlocal rollback_calls
        rollback_calls += 1
        await original_rollback()

    def fail(message):
        raise RuntimeError(message)

    def before_merge(info, operation, payload):
        if failure_phase == "before_merge":
            fail("failed-before-merge")

    def type_pre(model_cls, info, data, ctx):
        if failure_phase == "type_pre":
            fail("failed-type-pre")
        return data

    def type_post(model_cls, info, instance, created, ctx):
        if failure_phase == "type_post":
            fail("failed-type-post")
        if failure_phase == "validator":
            ctx["operation"].defer_validation(
                ("failing-validator",),
                lambda: fail("failed-validator"),
            )

    def after_commit(info, operation, result):
        error_events.append("after_commit")

    def on_error(info, operation, exception):
        error_events.append(
            ("on_error", str(exception), operation.session.in_transaction())
        )

    original_pre = tuple(getattr(PostQL, "__merge_pre_cbs__", ()) or ())
    original_post = tuple(getattr(PostQL, "__merge_post_cbs__", ()) or ())
    original_lengths = {
        phase: len(callbacks)
        for phase, callbacks in berry_schema._merge_hooks.items()
    }
    PostQL.__merge_pre_cbs__ = original_pre + (type_pre,)
    PostQL.__merge_post_cbs__ = original_post + (type_post,)
    db_session.rollback = tracked_rollback
    berry_schema.merge_hooks(
        before_merge=before_merge,
        after_commit=after_commit,
        on_error=on_error,
    )
    try:
        author_id = populated_db["users"][0].id
        result = await schema.execute(
            """
            mutation($payload: PostQLInput!) {
              merge_post(payload: $payload) { id }
            }
            """,
            variable_values={
                "payload": {
                    "title": title,
                    "content": "body",
                    "author_id": author_id,
                }
            },
            context_value={"db_session": db_session},
        )
        assert result.errors
        assert f"failed-{failure_phase.replace('_', '-')}" in str(
            result.errors[0]
        )
        assert rollback_calls == 1
        assert error_events == [
            (
                "on_error",
                f"failed-{failure_phase.replace('_', '-')}",
                False,
            )
        ]
        count = await db_session.scalar(
            select(func.count(Post.id)).where(Post.title == title)
        )
        assert count == 0
    finally:
        db_session.rollback = original_rollback
        PostQL.__merge_pre_cbs__ = original_pre
        PostQL.__merge_post_cbs__ = original_post
        for phase, length in original_lengths.items():
            del berry_schema._merge_hooks[phase][length:]
