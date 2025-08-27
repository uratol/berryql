import sys, asyncio
sys.path.append(r'd:\Work\berryql')
from tests.schema import schema
from tests.models import Base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

async def main():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=True, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        # seed minimal users to satisfy FK
        await session.execute(text("INSERT INTO users (name, email, is_admin, created_at) VALUES ('A','a@x',1, NULL)"))
        await session.commit()
        mutation = (
            "mutation Upsert($payload: PostQLInput!) {\n"
            "  merge_post(payload: $payload) { id title status author_id }\n"
            "}"
        )
        variables = {"payload": {"title": "Enum Post", "content": "C2", "author_id": 1, "status": "PUBLISHED"}}
        res = await schema.execute(mutation, variable_values=variables, context_value={"db_session": session})
        print("errors:", res.errors)
        print("data:", res.data)
    await engine.dispose()

if __name__ == '__main__':
    asyncio.run(main())
