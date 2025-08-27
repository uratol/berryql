import asyncio, sys
sys.path.append(r'd:\Work\berryql')
from tests.schema import schema
from tests.conftest import engine as engine_fixture
from tests.fixtures import seed_populated_db
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

async def main():
    # Use the pytest fixture factory to create an engine
    eng_gen = engine_fixture()
    engine = await eng_gen.__anext__()
    try:
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session() as session:
            await seed_populated_db(session)
            q = "query { posts { id status } }"
            res = await schema.execute(q, context_value={"db_session": session})
            print("errors:", res.errors)
            rows = res.data['posts']
            for r in rows:
                s = r['status']
                print("row:", r['id'], type(s), s, getattr(s, 'name', None), getattr(s, 'value', None))
    finally:
        # finalize the engine fixture
        try:
            await eng_gen.aclose()
        except Exception:
            pass

if __name__ == '__main__':
    asyncio.run(main())
