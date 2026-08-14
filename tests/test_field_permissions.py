from __future__ import annotations

import asyncio
import logging
from collections import Counter

import pytest
from sqlalchemy import event, func, select

from berryql import (
    BerrySchema,
    BerryType,
    FieldPermissions,
    FieldSet,
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


def _permissions(*, read=None, write=None):
    return FieldPermissions(
        read=read or FieldSet.all(),
        write=write or FieldSet.all(),
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

    static_schema = BerrySchema(field_permissions=FieldPermissions(read=FieldSet.only("id")))
    assert isinstance(static_schema._field_permissions_provider, FieldPermissions)


@pytest.mark.asyncio
async def test_read_allow_list_returns_null_and_empty_relations_without_extra_sql(db_session, populated_db, engine):
    statements = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", _capture)
    try:
        context = _context(
            db_session,
            field_permissions={"PermissionPostQL": _permissions(read=FieldSet.only("id", "title"))},
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
            "PermissionPostQL": _permissions(read=FieldSet.all_except("content")),
            "PermissionCommentQL": _permissions(read=FieldSet.only("id")),
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
            "PermissionPostQL": _permissions(read=FieldSet.only("id", "post_comments")),
            "PermissionCommentQL": _permissions(read=FieldSet.only("id")),
        },
    )
    result = await permission_graphql_schema.execute(query, context_value=context)

    assert result.errors
    assert "not readable" in str(result.errors[0])


@pytest.mark.asyncio
async def test_global_and_type_permissions_intersect_and_resolve_once_per_request(db_session, populated_db):
    context = _context(
        db_session,
        field_permissions={"PermissionPostQL": _permissions(read=FieldSet.only("id", "title"))},
        post_local_permissions=_permissions(read=FieldSet.all_except("title")),
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
        field_permissions={"PermissionPostQL": _permissions(write=FieldSet.only("title"))},
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
        field_permissions={"PermissionPostQL": _permissions(write=FieldSet.none())},
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
        field_permissions={"PermissionPostQL": _permissions(write=FieldSet.none())},
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
            "PermissionPostQL": _permissions(write=FieldSet.only("title")),
            "PermissionCommentQL": _permissions(write=FieldSet.all()),
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
            "PermissionPostQL": _permissions(write=FieldSet.only("post_comments")),
            "PermissionCommentQL": _permissions(write=FieldSet.only("rate")),
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
        field_permissions={"PermissionPostQL": _permissions(write=FieldSet.only("title", "author_id"))},
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
                read=FieldSet.all_except("content"),
                write=FieldSet.only("content"),
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
