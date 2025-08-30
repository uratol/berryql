import pytest

from tests.schema import schema


@pytest.mark.asyncio
async def test_domain_mutation_not_exposed_on_query_blogDomain(populated_db):  # noqa: F811
    # Introspect the schema to ensure create_post_mut is not a field on the Query blogDomain container
    sdl = str(schema)
    # The query container type for BlogDomain is BlogDomainType
    assert 'type BlogDomainType' in sdl
    # Ensure the mutation create_post_mut is NOT listed under BlogDomainType fields
    blog_block_start = sdl.find('type BlogDomainType')
    assert blog_block_start != -1
    blog_block_end = sdl.find('\n}', blog_block_start)
    block = sdl[blog_block_start:blog_block_end]
    assert 'create_post_mut' not in block, block

    # But the Mutation root should include create_post_mut under blogDomain
    assert 'type Mutation' in sdl
    # Ensure the mutation name appears somewhere under Mutation (as nested field path blogDomain { create_post_mut })
    # We can't trivially parse nested selection in sdl string, but at least ensure the name exists in schema string
    assert 'create_post_mut' in sdl
