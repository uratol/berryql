import pytest

from tests.schema import berry_schema


@pytest.mark.asyncio
async def test_domain_subscription_naming_camel(db_session, populated_db):
    try:
        from strawberry.schema.config import StrawberryConfig
    except Exception:
        pytest.skip("StrawberryConfig not available in this Strawberry version")

    schema = berry_schema.to_strawberry(strawberry_config=StrawberryConfig(auto_camel_case=True))
    # Expect camelCase naming: blogDomain + NewPostEvent
    sub = await schema.subscribe("subscription { blogDomainNewPostEvent(to: 1) }")
    it = sub if hasattr(sub, "__anext__") else sub.__aiter__()
    event = await it.__anext__()
    assert getattr(event, "errors", None) is None
    assert event.data["blogDomainNewPostEvent"] == 1


@pytest.mark.asyncio
async def test_domain_subscription_naming_snake(db_session, populated_db):
    try:
        from strawberry.schema.config import StrawberryConfig
    except Exception:
        pytest.skip("StrawberryConfig not available in this Strawberry version")

    schema = berry_schema.to_strawberry(strawberry_config=StrawberryConfig(auto_camel_case=False))
    # Expect snake_case naming: blogDomain_new_post_event
    sub = await schema.subscribe("subscription { blogDomain_new_post_event(to: 1) }")
    it = sub if hasattr(sub, "__anext__") else sub.__aiter__()
    event = await it.__anext__()
    assert getattr(event, "errors", None) is None
    assert event.data["blogDomain_new_post_event"] == 1
