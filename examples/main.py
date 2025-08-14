"""FastAPI app exposing the Strawberry GraphQL schema with GraphiQL playground.

Run this file to start a local server and open http://127.0.0.1:8000/graphql

Environment variables:
  BERRYQL_TEST_DATABASE_URL  optional SQLAlchemy async URL, e.g. postgresql+asyncpg://user:pass@localhost/db
                     defaults to sqlite+aiosqlite:///./berryql_demo.db
  DEMO_SEED          set to '0' to skip demo data seeding (default '1')
    SQL_ECHO           set to '1' (default) to log SQL, '0' to disable; uses 'debug' to include params
"""
from __future__ import annotations

import os
import sys
import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy import event

from strawberry.fastapi import GraphQLRouter

# Reuse test models and the already-built Strawberry schema from tests
from tests.models import Base, User, Post, PostComment  # type: ignore
from tests.schema import schema  # Strawberry schema built from berryql DSL
from tests.fixtures import seed_populated_db  # reuse test seeding logic

app = FastAPI(title="BerryQL GraphQL Playground")


# Engine + session factory stored on app.state
async def _init_db(app: FastAPI) -> None:
    # Use in-memory SQLite by default; allow override via BERRYQL_TEST_DATABASE_URL
    env_db_url = os.getenv("BERRYQL_TEST_DATABASE_URL")
    # Toggle SQL echo via env var (1=on, 0=off). When on, use 'debug' for params.
    echo_enabled = os.getenv("SQL_ECHO", "1") != "0"
    echo_value = "debug" if echo_enabled else False
    if env_db_url:
        engine: AsyncEngine = create_async_engine(env_db_url, echo=echo_value, future=True)
    else:
        # Shared in-memory DB across connections using StaticPool
        engine: AsyncEngine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=echo_value,
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    app.state.engine = engine
    app.state.async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Ensure logger level so SQL shows up under Uvicorn
    if echo_enabled:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.INFO)

        # Optional: add cursor-level hooks for timing/visibility
        try:
            @event.listens_for(engine.sync_engine, "before_cursor_execute")
            def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
                logging.getLogger("sqlalchemy.engine").info("SQL: %s | params=%s", statement, parameters)

            @event.listens_for(engine.sync_engine, "after_cursor_execute")
            def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
                if context and getattr(context, "_query_start_time", None):
                    dur_ms = (context._execution_endtime - context._query_start_time) * 1000  # type: ignore[attr-defined]
                    logging.getLogger("sqlalchemy.engine").info("Done in %.2f ms", dur_ms)
        except Exception:
            # If event wiring fails (shouldn't), continue with echo logging
            pass

    # Create schema (tables)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed demo data on first run (in-memory DB is empty every boot)
    if os.getenv("DEMO_SEED", "1") != "0":
        async with app.state.async_session() as session:
            # Only seed if empty
            from sqlalchemy import select

            users_count = (await session.execute(select(User).limit(1))).first()
            if not users_count:
                await seed_populated_db(session)


async def _seed_demo(session: AsyncSession) -> None:
    # Kept for backward compatibility; now delegates to shared test seeding
    await seed_populated_db(session)


# Create/cleanup one DB session per request
@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    async_session: async_sessionmaker[AsyncSession] = app.state.async_session
    async with async_session() as session:
        request.state.db_session = session
        response = await call_next(request)
        return response


@app.on_event("startup")
async def on_startup() -> None:
    # Windows: use SelectorEventLoop for broad driver compatibility
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    await _init_db(app)


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/graphql")


async def get_context(request: Request):
    # You can add auth-derived info here, e.g., current_user or user_id
    return {
        "db_session": getattr(request.state, "db_session", None),
        "current_user": None,
    }


graphql_router = GraphQLRouter(
    schema,
    graphiql=True,  # enables GraphiQL playground UI
    context_getter=get_context,
)

app.include_router(graphql_router, prefix="/graphql")


if __name__ == "__main__":
    # Local dev runner
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
