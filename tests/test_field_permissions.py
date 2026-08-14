from __future__ import annotations

import asyncio
import logging
from collections import Counter

import pytest
import strawberry
from sqlalchemy import event, func, select

from berryql import (
    BerrySchema,
    BerryType,
    FieldPermissions,
    FieldSet,
    OperationPermissions,
    count,
    custom,
    field,
    mutation,
    relation,
)
from tests.models import Post, PostComment, User


async def _global_permissions(type_, info):
    context = info.context or {}
    calls = context.setdefault("permission_calls", Counter())
    calls[f"global:{type_.__name__}"] += 1
    await asyncio.sleep(0)
    return (context.get("field_permissions") or {}).get(type_.__name__, FieldPermissions.allow_all())


async def _post_local_permissions(type_, info):
    context = info.context or {}
    calls = context.setdefault("permission_calls", Counter())
    calls[f"local:{type_.__name__}"] += 1
    await asyncio.sleep(0)
    return context.get("post_local_permissions", FieldPermissions.allow_all())


permission_schema = BerrySchema(field_permissions=_global_permissions)


@permission_schema.type(model=User)
class PermissionUserQL(BerryType):
    id = field()
    name = field()
    email = field()


@permission_schema.type(model=PostComment)
class PermissionCommentQL(BerryType):
    id = field()
    content = field()
    rate = field()
    post_id = field()
    author_id = field()

    @permission_schema.pre
    def _capture_sanitized_payload(model_cls, info, data, context=None):
        (info.context or {}).setdefault("pre_payloads", []).append((model_cls.__name__, set((data or {}).keys())))
        return data


@permission_schema.type(
    model=Post,
    field_permissions=_post_local_permissions,
)
class PermissionPostQL(BerryType):
    id = field()
    title = field()
    content = field()
    author_id = field()
    author = relation("PermissionUserQL", single=True)
    post_comments = relation(
        "PermissionCommentQL",
        order_by="id",
        arguments={
            "content_eq": {"column": "content", "op": "eq"},
        },
    )
    comment_count = count("post_comments")
    content_size = custom(
        lambda model: select(func.length(model.content).label("content_size")),
        returns=int,
    )

    @strawberry.field
    def plain_note(self) -> str | None:
        return "plain"

    @strawberry.field
    def plain_tags(self) -> list[str] | None:
        return ["plain"]

    @strawberry.field
    def plain_required(self) -> str:
        return "plain"

    @permission_schema.pre
    def _capture_sanitized_payload(model_cls, info, data, context=None):
        (info.context or {}).setdefault("pre_payloads", []).append((model_cls.__name__, set((data or {}).keys())))
        return data


@permission_schema.query()
class PermissionQuery:
    posts = relation("PermissionPostQL", order_by="id")


@permission_schema.mutation()
class PermissionMutation:
    merge_post = mutation("PermissionPostQL", single=True)


permission_graphql_schema = permission_schema.to_strawberry()


def _context(db_session, **values):
    return {"db_session": db_session, **values}


def _permissions(*, select=None, filter=None, order=None, create=None, update=None, operations=None):
    return FieldPermissions(
        select=select or FieldSet.all(),
        filter=filter or FieldSet.all(),
        order=order or FieldSet.all(),
        create=create or FieldSet.all(),
        update=update or FieldSet.all(),
        operations=operations or OperationPermissions(),
    )


def test_field_set_constructors_and_intersection():
    assert FieldSet.all().allows("anything")
    assert not FieldSet.none().allows("anything")
    assert FieldSet.only("id", "title").allows("title")
    assert not FieldSet.only(["id", "title"]).allows("content")
    assert FieldSet.all_except("secret").allows("title")
    assert not FieldSet.all_except({"secret"}).allows("secret")

    combined = FieldSet.only("id", "title").intersection(FieldSet.all_except("title"))
    assert combined.allows("id")
    assert not combined.allows("title")
    assert not combined.allows("content")

    static_schema = BerrySchema(field_permissions=FieldPermissions(select=FieldSet.only("id")))
    assert isinstance(static_schema._field_permissions_provider, FieldPermissions)


def test_field_and_operation_permission_intersections_and_legacy_rejection():
    left = FieldPermissions(
        select=FieldSet.only("id", "title", "content"),
        filter=FieldSet.only("id", "content"),
        order=FieldSet.all_except("content"),
        create=FieldSet.only("title", "content"),
        update=FieldSet.only("content"),
        operations=OperationPermissions(
            create=True, update=True, delete=False, replace=True
        ),
    )
    right = FieldPermissions(
        select=FieldSet.all_except("content"),
        filter=FieldSet.all(),
        order=FieldSet.only("id", "content"),
        create=FieldSet.only("title"),
        update=FieldSet.all_except("content"),
        operations=OperationPermissions(
            create=True, update=False, delete=True, replace=False
        ),
    )

    combined = left.intersection(right)
    assert combined.select.allows("id")
    assert not combined.select.allows("content")
    assert combined.filter.allows("id")
    assert not combined.filter.allows("content")
    assert combined.order.allows("id")
    assert not combined.order.allows("content")
    assert combined.create.allows("title")
    assert not combined.create.allows("content")
    assert not combined.update.allows("content")
    assert combined.operations == OperationPermissions(
        create=True, update=False, delete=False, replace=False
    )

    with pytest.raises(TypeError, match="read"):
        FieldPermissions(read=FieldSet.all())
    with pytest.raises(TypeError, match="write"):
        FieldPermissions(write=FieldSet.all())


@pytest.mark.asyncio
async def test_plain_strawberry_scalar_and_list_use_final_select_guard(
    db_session, populated_db
):
    context = _context(
        db_session,
        field_permissions={
            "PermissionPostQL": _permissions(select=FieldSet.only("id"))
        },
    )
    result = await permission_graphql_schema.execute(
        "query { posts(limit: 1) { id plain_note plain_tags } }",
        context_value=context,
    )

    assert result.errors is None
    assert result.data["posts"] == [
        {"id": populated_db["posts"][0].id, "plain_note": None, "plain_tags": []}
    ]


@pytest.mark.asyncio
async def test_denied_non_null_plain_strawberry_field_uses_graphql_null_bubbling(
    db_session, populated_db
):
    result = await permission_graphql_schema.execute(
        "query { posts(limit: 1) { id plain_required } }",
        context_value=_context(
            db_session,
            field_permissions={
                "PermissionPostQL": _permissions(select=FieldSet.only("id"))
            },
        ),
    )

    assert result.errors
    assert "Cannot return null for non-nullable field" in str(result.errors[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capability", "query"),
    [
        (
            "filter",
            'query { posts(where: "{\\"content\\": {\\"eq\\": \\"Hello world!\\"}}") { id content } }',
        ),
        ("order", 'query { posts(order_by: "content") { id content } }'),
    ],
)
async def test_filter_and_order_capabilities_are_independent(
    db_session, populated_db, capability, query
):
    masks = {capability: FieldSet.all_except("content")}
    context = _context(
        db_session,
        field_permissions={"PermissionPostQL": _permissions(**masks)},
    )
    result = await permission_graphql_schema.execute(query, context_value=context)

    assert result.errors
    assert f"not allowed for {capability}" in str(result.errors[0])


@pytest.mark.asyncio
async def test_read_allow_list_returns_null_and_empty_relations_without_extra_sql(db_session, populated_db, engine):
    statements = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", _capture)
    try:
        context = _context(
            db_session,
            field_permissions={"PermissionPostQL": _permissions(select=FieldSet.only("id", "title"))},
        )
        result = await permission_graphql_schema.execute(
            """
            query {
              posts {
                id
                title
                content
                author { id name }
                post_comments { id content }
                comment_count
                content_size
              }
            }
            """,
            context_value=context,
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _capture)

    assert result.errors is None
    assert len(result.data["posts"]) == 5
    for post in result.data["posts"]:
        assert post["id"] is not None
        assert post["title"] is not None
        assert post["content"] is None
        assert post["author"] is None
        assert post["post_comments"] == []
        assert post["comment_count"] is None
        assert post["content_size"] is None
    assert len(statements) == 1
    assert "post_comments" not in statements[0].lower()


@pytest.mark.asyncio
async def test_read_deny_list_and_nested_target_allow_list(db_session, populated_db):
    context = _context(
        db_session,
        field_permissions={
            "PermissionPostQL": _permissions(select=FieldSet.all_except("content")),
            "PermissionCommentQL": _permissions(select=FieldSet.only("id")),
        },
    )
    result = await permission_graphql_schema.execute(
        """
        query {
          posts(where: "{\\"id\\": {\\"eq\\": 1}}") {
            id
            content
            post_comments { id content rate }
          }
        }
        """,
        context_value=context,
    )

    assert result.errors is None
    post = result.data["posts"][0]
    assert post["id"] == 1
    assert post["content"] is None
    assert len(post["post_comments"]) == 2
    assert all(comment["id"] is not None for comment in post["post_comments"])
    assert all(comment["content"] is None for comment in post["post_comments"])
    assert all(comment["rate"] is None for comment in post["post_comments"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        'query { posts(where: "{\\"content\\": {\\"eq\\": \\"Hello world!\\"}}") { id } }',
        'query { posts(order_by: "content") { id } }',
        'query { posts(where: "{\\"id\\": {\\"eq\\": 1}}") { post_comments(content_eq: "Great post!") { id } } }',
    ],
)
async def test_unreadable_fields_cannot_be_used_for_where_order_or_filter_args(db_session, populated_db, query):
    context = _context(
        db_session,
        field_permissions={
            "PermissionPostQL": _permissions(select=FieldSet.only("id", "post_comments")),
            "PermissionCommentQL": _permissions(select=FieldSet.only("id")),
        },
    )
    result = await permission_graphql_schema.execute(query, context_value=context)

    assert result.errors
    assert "not allowed for" in str(result.errors[0])


@pytest.mark.asyncio
async def test_global_and_type_permissions_intersect_and_resolve_once_per_request(db_session, populated_db):
    context = _context(
        db_session,
        field_permissions={"PermissionPostQL": _permissions(select=FieldSet.only("id", "title"))},
        post_local_permissions=_permissions(select=FieldSet.all_except("title")),
    )
    result = await permission_graphql_schema.execute(
        "query { posts { id title content } }",
        context_value=context,
    )

    assert result.errors is None
    assert len(result.data["posts"]) == 5
    assert all(post["id"] is not None for post in result.data["posts"])
    assert all(post["title"] is None for post in result.data["posts"])
    assert all(post["content"] is None for post in result.data["posts"])
    assert context["permission_calls"]["global:PermissionPostQL"] == 1
    assert context["permission_calls"]["local:PermissionPostQL"] == 1


@pytest.mark.asyncio
async def test_permission_provider_failure_fails_closed(db_session, populated_db):
    async def _broken_provider(type_, info):
        raise RuntimeError("ACL unavailable")

    broken_schema = BerrySchema(field_permissions=_broken_provider)

    @broken_schema.type(model=Post)
    class BrokenPermissionPostQL(BerryType):
        id = field()

    @broken_schema.query()
    class BrokenPermissionQuery:
        posts = relation("BrokenPermissionPostQL")

    graphql_schema = broken_schema.to_strawberry()
    result = await graphql_schema.execute(
        "query { posts { id } }",
        context_value={"db_session": db_session},
    )

    assert result.data is None
    assert result.errors
    assert "ACL unavailable" in str(result.errors[0])


@pytest.mark.asyncio
async def test_denied_scalar_write_is_ignored_before_hooks_and_warned_if_changed(db_session, populated_db, caplog):
    post = populated_db["posts"][0]
    old_content = post.content
    context = _context(
        db_session,
        field_permissions={"PermissionPostQL": _permissions(update=FieldSet.only("title"))},
    )

    with caplog.at_level(logging.WARNING, logger="berryql"):
        result = await permission_graphql_schema.execute(
            """
            mutation($payload: PermissionPostQLInput!) {
              merge_post(payload: $payload) { id title content }
            }
            """,
            variable_values={
                "payload": {
                    "id": post.id,
                    "title": "Allowed title",
                    "content": "Blocked content",
                }
            },
            context_value=context,
        )

    assert result.errors is None
    assert result.data["merge_post"]["title"] == "Allowed title"
    assert result.data["merge_post"]["content"] == old_content
    assert post.title == "Allowed title"
    assert post.content == old_content
    post_pre_payloads = [keys for model_name, keys in context["pre_payloads"] if model_name == "Post"]
    assert post_pre_payloads
    assert "content" not in post_pre_payloads[0]
    assert "id" in post_pre_payloads[0]
    assert "title" in post_pre_payloads[0]
    assert "Ignored unauthorized write fields" in caplog.text
    assert "content" in caplog.text


@pytest.mark.asyncio
async def test_same_or_unloaded_denied_value_does_not_warn(db_session, populated_db, caplog, engine):
    post = populated_db["posts"][0]
    context = _context(
        db_session,
        field_permissions={"PermissionPostQL": _permissions(update=FieldSet.none())},
    )
    with caplog.at_level(logging.WARNING, logger="berryql"):
        same_result = await permission_graphql_schema.execute(
            """
            mutation($payload: PermissionPostQLInput!) {
              merge_post(payload: $payload) { id }
            }
            """,
            variable_values={"payload": {"id": post.id, "content": post.content}},
            context_value=context,
        )
    assert same_result.errors is None
    assert "Ignored unauthorized write fields" not in caplog.text

    caplog.clear()
    db_session.expire(post, ["content"])
    statements = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    context = _context(
        db_session,
        field_permissions={"PermissionPostQL": _permissions(update=FieldSet.none())},
    )
    event.listen(engine.sync_engine, "before_cursor_execute", _capture)
    try:
        with caplog.at_level(logging.WARNING, logger="berryql"):
            unloaded_result = await permission_graphql_schema.execute(
                """
                mutation($payload: PermissionPostQLInput!) {
                  merge_post(payload: $payload) { id }
                }
                """,
                variable_values={"payload": {"id": post.id, "content": "blocked"}},
                context_value=context,
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _capture)
    assert unloaded_result.errors is None
    assert "Ignored unauthorized write fields" not in caplog.text
    select_statements = [statement for statement in statements if statement.lstrip().upper().startswith("SELECT")]
    # The merge's normal post-flush refresh is the only SELECT. Comparing the
    # denied expired attribute must not cause a separate lazy-load query.
    assert len(select_statements) == 1


@pytest.mark.asyncio
async def test_denied_relation_write_ignores_entire_nested_payload(db_session, populated_db, caplog):
    post = populated_db["posts"][0]
    comment = populated_db["post_comments"][0]
    old_content = comment.content
    old_rate = comment.rate
    context = _context(
        db_session,
        field_permissions={
            "PermissionPostQL": _permissions(update=FieldSet.only("title")),
            "PermissionCommentQL": _permissions(update=FieldSet.all()),
        },
    )
    with caplog.at_level(logging.WARNING, logger="berryql"):
        result = await permission_graphql_schema.execute(
            """
            mutation($payload: PermissionPostQLInput!) {
              merge_post(payload: $payload) { id }
            }
            """,
            variable_values={
                "payload": {
                    "id": post.id,
                    "post_comments": [{"id": comment.id, "content": "blocked", "rate": 99}],
                }
            },
            context_value=context,
        )

    assert result.errors is None
    assert comment.content == old_content
    assert comment.rate == old_rate
    assert not any(model_name == "PostComment" for model_name, _ in context["pre_payloads"])
    assert "Ignored unauthorized write fields" not in caplog.text


@pytest.mark.asyncio
async def test_nested_write_applies_target_permissions_recursively(db_session, populated_db, caplog):
    post = populated_db["posts"][0]
    comment = populated_db["post_comments"][0]
    old_content = comment.content
    context = _context(
        db_session,
        field_permissions={
            "PermissionPostQL": _permissions(update=FieldSet.only("post_comments")),
            "PermissionCommentQL": _permissions(update=FieldSet.only("rate")),
        },
    )
    with caplog.at_level(logging.WARNING, logger="berryql"):
        result = await permission_graphql_schema.execute(
            """
            mutation($payload: PermissionPostQLInput!) {
              merge_post(payload: $payload) { id }
            }
            """,
            variable_values={
                "payload": {
                    "id": post.id,
                    "post_comments": [{"id": comment.id, "content": "blocked", "rate": 77}],
                }
            },
            context_value=context,
        )

    assert result.errors is None
    assert comment.content == old_content
    assert comment.rate == 77
    comment_pre_payloads = [keys for model_name, keys in context["pre_payloads"] if model_name == "PostComment"]
    assert comment_pre_payloads
    assert "content" not in comment_pre_payloads[0]
    assert "rate" in comment_pre_payloads[0]
    assert "PermissionCommentQL" in caplog.text
    assert "content" in caplog.text


@pytest.mark.asyncio
async def test_denied_field_on_create_is_ignored_without_warning(db_session, populated_db, caplog):
    author = populated_db["users"][0]
    context = _context(
        db_session,
        field_permissions={"PermissionPostQL": _permissions(create=FieldSet.only("title", "author_id"))},
    )
    with caplog.at_level(logging.WARNING, logger="berryql"):
        result = await permission_graphql_schema.execute(
            """
            mutation($payload: PermissionPostQLInput!) {
              merge_post(payload: $payload) { id title content }
            }
            """,
            variable_values={
                "payload": {
                    "title": "Created with permissions",
                    "content": "must be ignored",
                    "author_id": author.id,
                }
            },
            context_value=context,
        )

    assert result.errors is None
    assert result.data["merge_post"]["title"] == "Created with permissions"
    assert result.data["merge_post"]["content"] is None
    assert "Ignored unauthorized write fields" not in caplog.text


@pytest.mark.asyncio
async def test_mutation_response_uses_read_not_write_permissions(db_session, populated_db):
    post = populated_db["posts"][0]
    context = _context(
        db_session,
        field_permissions={
            "PermissionPostQL": _permissions(
                select=FieldSet.all_except("content"),
                update=FieldSet.only("content"),
            )
        },
    )
    result = await permission_graphql_schema.execute(
        """
        mutation($payload: PermissionPostQLInput!) {
          merge_post(payload: $payload) { id content }
        }
        """,
        variable_values={"payload": {"id": post.id, "content": "stored but hidden"}},
        context_value=context,
    )

    assert result.errors is None
    assert post.content == "stored but hidden"
    assert result.data["merge_post"]["content"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("include_relation_payload", [False, True])
async def test_denied_replace_relation_is_fully_ignored_before_hooks(
    db_session, populated_db, include_relation_payload
):
    post = populated_db["posts"][0]
    post_id = post.id
    comments = [
        comment
        for comment in populated_db["post_comments"]
        if comment.post_id == post_id
    ]
    original = [(comment.id, comment.content, comment.rate) for comment in comments]
    payload = {"id": post_id, "_Replace": ["post_comments"]}
    if include_relation_payload:
        payload["post_comments"] = [
            {"id": comments[0].id, "content": "must not be visible"}
        ]
    context = _context(
        db_session,
        field_permissions={
            "PermissionPostQL": _permissions(
                update=FieldSet.all_except("post_comments")
            )
        },
    )

    result = await permission_graphql_schema.execute(
        """
        mutation($payload: PermissionPostQLInput!) {
          merge_post(payload: $payload) { id }
        }
        """,
        variable_values={"payload": payload},
        context_value=context,
    )

    assert result.errors is None
    rows = (
        await db_session.execute(
            select(PostComment)
            .where(PostComment.post_id == post_id)
            .order_by(PostComment.id)
        )
    ).scalars().all()
    assert [(row.id, row.content, row.rate) for row in rows] == original
    post_payloads = [
        keys for model_name, keys in context["pre_payloads"] if model_name == "Post"
    ]
    assert post_payloads
    assert "post_comments" not in post_payloads[0]
    assert "_Replace" not in post_payloads[0]
    assert not any(
        model_name == "PostComment" for model_name, _ in context["pre_payloads"]
    )


@pytest.mark.asyncio
async def test_replace_requires_child_delete_capability_and_rolls_back(
    db_session, populated_db
):
    post = populated_db["posts"][0]
    post_id = post.id
    comments = [
        comment
        for comment in populated_db["post_comments"]
        if comment.post_id == post_id
    ]
    original_ids = [comment.id for comment in comments]
    context = _context(
        db_session,
        field_permissions={
            "PermissionPostQL": _permissions(
                update=FieldSet.only("post_comments")
            ),
            "PermissionCommentQL": _permissions(
                operations=OperationPermissions(delete=False)
            ),
        },
    )

    result = await permission_graphql_schema.execute(
        """
        mutation($payload: PermissionPostQLInput!) {
          merge_post(payload: $payload) { id }
        }
        """,
        variable_values={
            "payload": {
                "id": post_id,
                "post_comments": [{"id": comments[0].id}],
                "_Replace": ["post_comments"],
            }
        },
        context_value=context,
    )

    assert result.errors
    assert "Operation 'delete' is not allowed" in str(result.errors[0])
    remaining_ids = (
        await db_session.execute(
            select(PostComment.id)
            .where(PostComment.post_id == post_id)
            .order_by(PostComment.id)
        )
    ).scalars().all()
    assert remaining_ids == original_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_replace_uses_resolved_scope_and_preserves_out_of_scope_rows(
    db_session, populated_db, monkeypatch, is_async
):
    post = populated_db["posts"][0]
    comments = [
        comment
        for comment in populated_db["post_comments"]
        if comment.post_id == post.id
    ]

    if is_async:
        async def scope(model_cls, info):
            await asyncio.sleep(0)
            return {"rate": {"gte": 2}}
    else:
        def scope(model_cls, info):
            return {"rate": {"gte": 2}}

    monkeypatch.setitem(
        PermissionPostQL.__berry_fields__["post_comments"].meta,
        "scope",
        scope,
    )
    result = await permission_graphql_schema.execute(
        """
        mutation($payload: PermissionPostQLInput!) {
          merge_post(payload: $payload) { id }
        }
        """,
        variable_values={
            "payload": {"id": post.id, "_Replace": ["post_comments"]}
        },
        context_value=_context(db_session),
    )

    assert result.errors is None
    remaining = (
        await db_session.execute(
            select(PostComment.id, PostComment.rate)
            .where(PostComment.post_id == post.id)
            .order_by(PostComment.id)
        )
    ).all()
    assert remaining == [(comments[1].id, 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_throwing_replace_scope_rolls_back_entire_merge(
    db_session, populated_db, monkeypatch, is_async
):
    post = populated_db["posts"][0]
    old_title = post.title
    original_ids = [
        comment.id
        for comment in populated_db["post_comments"]
        if comment.post_id == post.id
    ]

    if is_async:
        async def scope(model_cls, info):
            await asyncio.sleep(0)
            raise RuntimeError("replace scope failed")
    else:
        def scope(model_cls, info):
            raise RuntimeError("replace scope failed")

    monkeypatch.setitem(
        PermissionPostQL.__berry_fields__["post_comments"].meta,
        "scope",
        scope,
    )
    result = await permission_graphql_schema.execute(
        """
        mutation($payload: PermissionPostQLInput!) {
          merge_post(payload: $payload) { id }
        }
        """,
        variable_values={
            "payload": {
                "id": post.id,
                "title": "must roll back",
                "_Replace": ["post_comments"],
            }
        },
        context_value=_context(db_session),
    )

    assert result.errors
    assert "replace scope failed" in str(result.errors[0])
    await db_session.refresh(post)
    assert post.title == old_title
    remaining_ids = (
        await db_session.execute(
            select(PostComment.id)
            .where(PostComment.post_id == post.id)
            .order_by(PostComment.id)
        )
    ).scalars().all()
    assert remaining_ids == original_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "operations", "expected_operation"),
    [
        (
            {"id": 99999, "title": "blocked", "author_id": 1, "_Insert": True},
            OperationPermissions(create=False),
            "create",
        ),
        (
            {"id": 1, "title": "blocked", "_Insert": False},
            OperationPermissions(update=False),
            "update",
        ),
        (
            {"id": 1, "_Delete": True},
            OperationPermissions(delete=False),
            "delete",
        ),
        (
            {"id": 1, "_Replace": ["post_comments"]},
            OperationPermissions(replace=False),
            "replace",
        ),
    ],
)
async def test_operation_capabilities_are_enforced_independently(
    db_session, populated_db, payload, operations, expected_operation
):
    context = _context(
        db_session,
        field_permissions={
            "PermissionPostQL": _permissions(operations=operations)
        },
    )
    result = await permission_graphql_schema.execute(
        """
        mutation($payload: PermissionPostQLInput!) {
          merge_post(payload: $payload) { id }
        }
        """,
        variable_values={"payload": payload},
        context_value=context,
    )

    assert result.errors
    assert f"Operation '{expected_operation}' is not allowed" in str(
        result.errors[0]
    )
