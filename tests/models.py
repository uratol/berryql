"""Database models for BerryQL tests (shared)."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.types import TypeDecorator
from sqlalchemy.dialects import postgresql


class BinaryArray(TypeDecorator):
    """Portable array-of-bytes column.

    - On PostgreSQL: uses BYTEA[] (ARRAY(BYTEA)).
    - On other DBs (e.g., SQLite for tests): stores as JSON array of base64 strings.

    GraphQL exposure treats this as List[str] (base64) on Postgres; on SQLite it
    will appear as a string-typed scalar in the schema but our tests skip unless
    Postgres is used.
    """

    cache_ok = True
    impl = JSON  # default/fallback representation

    def load_dialect_impl(self, dialect):  # pragma: no cover - exercised via DB
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(postgresql.ARRAY(postgresql.BYTEA))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):  # pragma: no cover - trivial
        if value is None:
            return None
        # Expect list-like of bytes or base64 strings
        if dialect.name == 'postgresql':
            # Accept bytes/bytearray or base64 strings; convert strings back to bytes
            out = []
            import base64
            for v in value:
                if v is None:
                    out.append(None)
                    continue
                if isinstance(v, (bytes, bytearray, memoryview)):
                    out.append(bytes(v))
                else:
                    # assume base64 string
                    s = str(v)
                    out.append(base64.b64decode(s.encode('ascii')))
            return out
        # Non-Postgres: store as base64 strings JSON
        import base64
        out: list[str | None] = []
        for v in value:
            if v is None:
                out.append(None)
                continue
            if isinstance(v, (bytes, bytearray, memoryview)):
                out.append(base64.b64encode(bytes(v)).decode('ascii'))
            else:
                out.append(str(v))
        return out

    def process_result_value(self, value, dialect):  # pragma: no cover - trivial
        if value is None:
            return None
        import base64
        import json as _json
        if dialect.name == 'postgresql':
            # Convert bytes to base64 strings for consistent GraphQL representation
            out: list[str | None] = []
            for v in value:
                if v is None:
                    out.append(None)
                else:
                    bb = bytes(v) if isinstance(v, (bytearray, memoryview)) else (v if isinstance(v, bytes) else bytes(v))
                    out.append(base64.b64encode(bb).decode('ascii'))
            return out
        # Non-Postgres: ensure list[str] even if driver returns JSON string
        if isinstance(value, str):
            try:
                parsed = _json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            # Fallback: wrap string
            return [value]
        return value


class Base(DeclarativeBase):
    """Base class for test models."""
    pass


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    posts = relationship("Post", back_populates="author")
    post_comments = relationship("PostComment", back_populates="author")


class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    # New: array of binary blobs
    binary_blobs = Column(BinaryArray(), nullable=True)
    
    author = relationship("User", back_populates="posts")
    post_comments = relationship("PostComment", back_populates="post")


class PostComment(Base):
    __tablename__ = 'post_comments'
    
    id = Column(Integer, primary_key=True)
    content = Column(String(1000), nullable=False)
    rate = Column(Integer, nullable=False, default=0)
    post_id = Column(Integer, ForeignKey('posts.id'), nullable=False)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    post = relationship("Post", back_populates="post_comments")
    author = relationship("User", back_populates="post_comments")
    likes = relationship("PostCommentLike", back_populates="comment")


class PostCommentLike(Base):
    __tablename__ = 'post_comment_likes'
    
    id = Column(Integer, primary_key=True)
    post_comment_id = Column(Integer, ForeignKey('post_comments.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    comment = relationship("PostComment", back_populates="likes")
    user = relationship("User")
