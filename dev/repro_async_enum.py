from sqlalchemy.ext.asyncio import create_async_engine
import sys
import os
import asyncio

# Ensure we can import the workspace 'tests' package when invoked directly
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from tests.models import Base

async def main():
    eng = create_async_engine('sqlite+aiosqlite:///:memory:')
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print('CREATE_ALL_OK')
    except Exception as e:
        print('CREATE_ALL_ERR:', repr(e))
        raise
    finally:
        await eng.dispose()

if __name__ == '__main__':
    asyncio.run(main())
