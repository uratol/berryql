import pytest

from tests.schema import berry_schema, schema


def _restore_hooks(lengths):
    for phase, length in lengths.items():
        del berry_schema._merge_hooks[phase][length:]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document, field_name",
    [
        (
            """
            mutation {
              create_post_id(title: "broken", content: "body", author_id: 1)
            }
            """,
            "create_post_id",
        ),
        (
            """
            mutation {
              blogDomain {
                create_post_mut(title: "broken", content: "body", author_id: 1) {
                  id
                }
              }
            }
            """,
            "create_post_mut",
        ),
    ],
)
async def test_custom_mutation_failure_runs_on_error(document, field_name):
    events = []

    async def on_error(info, operation, exception):
        events.append((info.field_name, operation, str(exception)))

    original_lengths = {
        phase: len(callbacks)
        for phase, callbacks in berry_schema._merge_hooks.items()
    }
    berry_schema.merge_hooks(on_error=on_error)
    try:
        result = await schema.execute(document, context_value={})

        assert result.errors
        assert "No db_session in context" in str(result.errors[0])
        assert events == [(field_name, None, "No db_session in context")]
    finally:
        _restore_hooks(original_lengths)


@pytest.mark.asyncio
async def test_custom_mutation_on_error_failure_does_not_mask_original(caplog):
    def failing_on_error(info, operation, exception):
        raise RuntimeError("secondary-hook-error")

    original_lengths = {
        phase: len(callbacks)
        for phase, callbacks in berry_schema._merge_hooks.items()
    }
    berry_schema.merge_hooks(on_error=failing_on_error)
    try:
        result = await schema.execute(
            """
            mutation {
              create_post_id(title: "broken", content: "body", author_id: 1)
            }
            """,
            context_value={},
        )

        assert result.errors
        assert "No db_session in context" in str(result.errors[0])
        assert "secondary-hook-error" not in str(result.errors[0])
        assert "BerryQL on_error hook failed" in caplog.text
    finally:
        _restore_hooks(original_lengths)
