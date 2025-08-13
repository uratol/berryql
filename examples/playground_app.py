"""FastAPI app exposing the Strawberry GraphQL schema with GraphiQL playground.

Run this file to start a local server and open http://127.0.0.1:8000/graphql

Environment variables:
  TEST_DATABASE_URL  optional SQLAlchemy async URL, e.g. postgresql+asyncpg://user:pass@localhost/db
                     defaults to sqlite+aiosqlite:///./berryql_demo.db
  DEMO_SEED          set to '0' to skip demo data seeding (default '1')
"""
from __future__ import annotations

import os
import sys
import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from strawberry.fastapi import GraphQLRouter

# Reuse test models and the already-built Strawberry schema from tests
from tests.models import Base, User, Post, PostComment  # type: ignore
from tests.schema import schema  # Strawberry schema built from Berry DSL

app = FastAPI(title="BerryQL GraphQL Playground")


# Engine + session factory stored on app.state
async def _init_db(app: FastAPI) -> None:
    # Use in-memory SQLite by default; allow override via TEST_DATABASE_URL
    env_db_url = os.getenv("TEST_DATABASE_URL")
    if env_db_url:
        engine: AsyncEngine = create_async_engine(env_db_url, echo=False, future=True)
    else:
        # Shared in-memory DB across connections using StaticPool
        engine: AsyncEngine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    app.state.engine = engine
    app.state.async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

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
                await _seed_demo(session)


async def _seed_demo(session: AsyncSession) -> None:
    """Seed data mirroring tests/fixtures.py for consistency."""
    from datetime import datetime, timezone, timedelta

    # Users
    users = [
        User(name="Alice Johnson", email="alice@example.com", is_admin=True),
        User(name="Bob Smith", email="bob@example.com", is_admin=False),
        User(name="Charlie Brown", email="charlie@example.com", is_admin=False),
        User(name="Dave NoPosts", email="dave@example.com", is_admin=False),
    ]
    session.add_all(users)
    await session.flush()
    u1, u2, u3, _ = users

    # Posts with deterministic created_at
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    posts = [
        Post(title="First Post", content="Hello world!", author_id=u1.id, created_at=now - timedelta(minutes=60)),
        Post(title="GraphQL is Great", content="I love GraphQL!", author_id=u1.id, created_at=now - timedelta(minutes=45)),
        Post(title="SQLAlchemy Tips", content="Some useful tips...", author_id=u2.id, created_at=now - timedelta(minutes=30)),
        Post(title="Python Best Practices", content="Here are some tips...", author_id=u2.id, created_at=now - timedelta(minutes=15)),
        Post(title="Getting Started", content="A beginner's guide", author_id=u3.id, created_at=now - timedelta(minutes=5)),
    ]
    session.add_all(posts)
    await session.flush()
    p1, p2, p3, p4, p5 = posts

    # Comments
    post_comments = [
        PostComment(content="Great post!", post_id=p1.id, author_id=u2.id, rate=2),
        PostComment(content="Thanks for sharing!", post_id=p1.id, author_id=u3.id, rate=1),
        PostComment(content="I agree completely!", post_id=p2.id, author_id=u2.id, rate=3),
        PostComment(content="Very helpful tips", post_id=p3.id, author_id=u1.id, rate=1),
        PostComment(content="Nice work!", post_id=p3.id, author_id=u3.id, rate=2),
        PostComment(content="Looking forward to more", post_id=p4.id, author_id=u1.id, rate=1),
        PostComment(content="This helped me a lot", post_id=p5.id, author_id=u2.id, rate=5),
    ]
    session.add_all(post_comments)
    await session.commit()


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
