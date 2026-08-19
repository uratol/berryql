"""Tests for BerryQL exception interception (translating exceptions into user errors).

Covers the schema-level error interceptor:
- SQL exceptions (e.g. SQLAlchemy IntegrityError) translated to client errors;
- type-specific handlers, catch-all fallbacks and precedence;
- sync/async handlers, handler arity variants, context passing;
- original error preserved when handler returns None or itself fails;
- validation errors (no original exception) left untouched;
- registration before/after schema build; execute_sync path.
"""
import pytest
import strawberry
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from berryql import BerrySchema, BerryType, UserFacingError, field, relation, mutation
from tests.fixtures import *  # noqa: F401,F403
from tests.models import User

pytestmark = pytest.mark.asyncio


def _make_schema():
    """Build an isolated BerrySchema exposing users + merge_users + a failing field."""
    berry = BerrySchema()

    @berry.type(model=User)
    class ErrUserQL(BerryType):
        id = field()
        name = field()
        email = field()

    @berry.query()
    class Query:
        users = relation('ErrUserQL', order_by='id', order_dir='asc')

        @strawberry.field
        def boom(self) -> int:
            raise SQLAlchemyError("synthetic SQL failure")

    @berry.mutation()
    class Mutation:
        merge_users = mutation('ErrUserQL', comment="Create or update users")

    return berry


DUP_EMAIL_MUTATION = """
mutation {
  merge_users(payload: [{name: "Dup", email: "alice@example.com"}]) { id }
}
"""


async def test_sql_integrity_error_translated_to_user_error(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()

    @berry.error_handler(IntegrityError)
    def translate(exc, context):
        return UserFacingError("This email address is already registered", code="EMAIL_TAKEN")

    res = await schema.execute(DUP_EMAIL_MUTATION, context_value={"db_session": db_session})
    assert res.errors, "expected mutation to fail on unique constraint"
    err = res.errors[0]
    assert err.message == "This email address is already registered"
    assert err.extensions == {"code": "EMAIL_TAKEN"}
    # The formatted GraphQL payload exposes the translated message + extensions
    formatted = err.formatted
    assert formatted["message"] == "This email address is already registered"
    assert formatted["extensions"] == {"code": "EMAIL_TAKEN"}
    # Original exception is preserved for server-side introspection
    assert isinstance(err.original_error, UserFacingError)


async def test_without_handler_original_error_surfaces(db_session, populated_db):
    schema = _make_schema().to_strawberry()

    res = await schema.execute(DUP_EMAIL_MUTATION, context_value={"db_session": db_session})
    assert res.errors
    # No handler registered: raw driver error surfaces untouched
    assert isinstance(res.errors[0].original_error, IntegrityError)


async def test_catch_all_fallback_applies_when_no_type_match(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()
    called = []

    @berry.error_handler(KeyError)
    def never_called(exc):
        called.append("specific")
        return "should not happen"

    @berry.error_handler
    def fallback(exc, context):
        called.append("fallback")
        return "service temporarily unavailable"

    res = await schema.execute(DUP_EMAIL_MUTATION, context_value={"db_session": db_session})
    assert res.errors
    assert res.errors[0].message == "service temporarily unavailable"
    assert called == ["fallback"]


async def test_specific_handler_takes_precedence_over_catch_all(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()

    berry.register_error_handler(lambda exc: "generic fallback")
    berry.register_error_handler(
        lambda exc: UserFacingError("Email already in use", code="EMAIL_TAKEN"),
        IntegrityError,
    )

    res = await schema.execute(DUP_EMAIL_MUTATION, context_value={"db_session": db_session})
    assert res.errors
    assert res.errors[0].message == "Email already in use"
    assert res.errors[0].extensions == {"code": "EMAIL_TAKEN"}


async def test_handler_returning_none_keeps_original_error(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()

    berry.register_error_handler(lambda exc: None, SQLAlchemyError)

    res = await schema.execute(DUP_EMAIL_MUTATION, context_value={"db_session": db_session})
    assert res.errors
    assert isinstance(res.errors[0].original_error, IntegrityError)
    # Message still the raw driver message (not translated, no extensions added)
    assert res.errors[0].message == str(res.errors[0].original_error)


async def test_async_handler_supported(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()

    @berry.error_handler(IntegrityError)
    async def translate(exc, context):
        return UserFacingError("duplicate detected", code="DUP")

    res = await schema.execute(DUP_EMAIL_MUTATION, context_value={"db_session": db_session})
    assert res.errors
    assert res.errors[0].message == "duplicate detected"
    assert res.errors[0].extensions == {"code": "DUP"}


async def test_failing_handler_does_not_mask_original_error(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()

    @berry.error_handler(IntegrityError)
    def broken(exc, context):
        raise RuntimeError("handler bug")

    res = await schema.execute(DUP_EMAIL_MUTATION, context_value={"db_session": db_session})
    assert res.errors
    # Original raw error preserved; handler failure logged, not propagated
    assert isinstance(res.errors[0].original_error, IntegrityError)


async def test_handler_receives_context_and_gql_error(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()
    seen = {}

    @berry.error_handler(SQLAlchemyError)
    def record(exc, context, gql_error):
        seen["exc_type"] = type(exc)
        seen["context"] = context
        seen["message"] = gql_error.message
        return "translated"

    res = await schema.execute(
        DUP_EMAIL_MUTATION,
        context_value={"db_session": db_session, "marker": 42},
    )
    assert res.errors
    assert res.errors[0].message == "translated"
    assert seen["exc_type"] is IntegrityError
    assert seen["context"]["marker"] == 42
    assert "db_session" in seen["context"]
    assert seen["message"]  # original GraphQL message passed through


async def test_query_path_exception_translated(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()

    @berry.error_handler(SQLAlchemyError)
    def translate(exc, context):
        return "database is temporarily unavailable"

    res = await schema.execute(
        "query { boom }",
        context_value={"db_session": db_session},
    )
    assert res.errors
    assert res.errors[0].message == "database is temporarily unavailable"


async def test_validation_errors_untouched(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()

    berry.register_error_handler(lambda exc: "translated")

    res = await schema.execute(
        "query { nope }",
        context_value={"db_session": db_session},
    )
    assert res.errors
    # Validation errors carry no original exception -> not translated
    assert res.errors[0].original_error is None
    assert "Cannot query field" in res.errors[0].message


async def test_handler_registered_after_schema_build(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()  # build first

    # Register after to_strawberry() — the shared interceptor picks it up
    berry.register_error_handler(
        lambda exc: UserFacingError("late registration works", code="LATE"),
        IntegrityError,
    )

    res = await schema.execute(DUP_EMAIL_MUTATION, context_value={"db_session": db_session})
    assert res.errors
    assert res.errors[0].message == "late registration works"


async def test_execute_sync_translates_with_sync_handler(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()

    @berry.error_handler(SQLAlchemyError)
    def translate(exc):
        return "sync translation works"

    res = schema.execute_sync(
        "query { boom }",
        context_value={"db_session": db_session},
    )
    assert res.errors
    assert res.errors[0].message == "sync translation works"


async def test_string_return_replaces_message_only(db_session, populated_db):
    berry = _make_schema()
    schema = berry.to_strawberry()

    berry.register_error_handler(lambda exc, ctx: "plain message replacement", SQLAlchemyError)

    res = await schema.execute(DUP_EMAIL_MUTATION, context_value={"db_session": db_session})
    assert res.errors
    assert res.errors[0].message == "plain message replacement"
    # No extensions added for a plain string outcome
    assert not res.errors[0].extensions


async def test_invalid_registration_arguments_rejected():
    berry = _make_schema()
    with pytest.raises(TypeError):
        berry.register_error_handler("not callable")
    with pytest.raises(TypeError):
        berry.register_error_handler(lambda exc: None, "not-an-exception-class")


async def test_schema_build_untouched_without_handlers(db_session, populated_db):
    # Sanity: schema without any handler behaves exactly like before
    schema = _make_schema().to_strawberry()
    res = await schema.execute(
        "query { users { id email } }",
        context_value={"db_session": db_session},
    )
    assert res.errors is None, res.errors
    emails = {u["email"] for u in res.data["users"]}
    assert "alice@example.com" in emails
