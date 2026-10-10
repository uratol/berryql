"""Query and Domain relations post-process one hydrated, paginated value."""
import pytest
from sqlalchemy import select, func
from berryql import BerrySchema, BerryType, BerryDomain, field, relation, domain
from tests.fixtures import *  # noqa: F401,F403
from tests.models import User


def build_schema(callback, *, single=False, returns=None, scope=None):
    berry = BerrySchema()

    @berry.type(model=User)
    class CallbackUserQL(BerryType):
        id = field()
        name = field()

    @berry.domain(name='people')
    class People(BerryDomain):
        users = relation(CallbackUserQL, single=single, order_by='id', scope=scope,
                         post_process=callback, returns=returns)

    @berry.query()
    class Query:
        users = relation(CallbackUserQL, single=single, order_by='id', scope=scope,
                         post_process=callback, returns=returns)
        people = domain(People)

    return berry.to_strawberry()


@pytest.mark.asyncio
@pytest.mark.parametrize('in_domain', [False, True])
@pytest.mark.parametrize('asynchronous', [False, True])
async def test_root_callback_runs_once_on_hydrated_offset_page(db_session, sample_users, in_domain, asynchronous):
    calls = []
    def update(rows, info):
        calls.append([row.id for row in rows])
        for row in rows:
            row.name = 'processed ' + row.name
        return rows
    async def update_async(rows, info):
        # The root SQL cursor and session lock have been released before this read.
        assert await info.context['db_session'].scalar(select(func.count(User.id))) == 4
        return update(rows, info)
    schema = build_schema(update_async if asynchronous else update)
    selection = 'users(offset:1,limit:2){id name}'
    query = '{people{' + selection + '}}' if in_domain else '{' + selection + '}'
    result = await schema.execute(query, context_value={'db_session': db_session})
    assert not result.errors
    actual = result.data['people']['users'] if in_domain else result.data['users']
    expected = sorted(sample_users, key=lambda row: row.id)[1:3]
    assert calls == [[row.id for row in expected]]
    assert actual == [{'id': row.id, 'name': 'processed ' + row.name} for row in expected]


@pytest.mark.asyncio
@pytest.mark.parametrize('in_domain', [False, True])
@pytest.mark.parametrize('single', [False, True])
@pytest.mark.parametrize('empty', [False, True])
async def test_root_returns_override_and_empty_values(db_session, sample_users, in_domain, single, empty):
    calls = []
    def names(value, info):
        calls.append(value)
        if single:
            return str(value.id) if value else None
        return [str(row.id) for row in value]
    scope = (lambda M, info: M.id < 0) if empty else None
    schema = build_schema(names, single=single, returns=str, scope=scope)
    query = '{people{users}}' if in_domain else '{users}'
    result = await schema.execute(query, context_value={'db_session': db_session})
    assert not result.errors
    actual = result.data['people']['users'] if in_domain else result.data['users']
    expected = [str(row.id) for row in sorted(sample_users, key=lambda row: row.id)]
    assert len(calls) == 1
    assert actual == ((None if empty else expected[0]) if single else ([] if empty else expected))
    if empty:
        assert calls[0] == (None if single else [])


@pytest.mark.asyncio
@pytest.mark.parametrize('in_domain', [False, True])
async def test_callback_errors_are_not_silently_replaced_with_raw_rows(db_session, sample_users, in_domain):
    def fail(rows, info):
        raise ValueError('Cannot build collection metadata')
    schema = build_schema(fail)
    query = '{people{users{id}}}' if in_domain else '{users{id}}'
    result = await schema.execute(query, context_value={'db_session': db_session})
    assert result.errors and 'Cannot build collection metadata' in result.errors[0].message


@pytest.mark.asyncio
@pytest.mark.parametrize('in_domain', [False, True])
async def test_callback_keeps_primary_key_when_client_does_not_select_it(db_session, sample_users, in_domain):
    calls = []
    def label(rows, info):
        calls.extend(row.id for row in rows)
        for row in rows:
            row.name = f'User {row.id}'
        return rows
    schema = build_schema(label)
    query = '{people{users(limit:1){name}}}' if in_domain else '{users(limit:1){name}}'
    result = await schema.execute(query, context_value={'db_session': db_session})
    assert not result.errors
    actual = result.data['people']['users'] if in_domain else result.data['users']
    ident = min(user.id for user in sample_users)
    assert calls == [ident] and actual == [{'name': f'User {ident}'}]
