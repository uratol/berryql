"""Database models for BerryQL tests (shared)."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, JSON, func, CheckConstraint
from sqlalchemy import Enum as SAEnum
import enum
from sqlalchemy.orm import DeclarativeBase, relationship, column_property
from sqlalchemy.types import TypeDecorator, LargeBinary
from sqlalchemy.dialects import postgresql, mssql

class BinaryBlob(TypeDecorator):
    """Single binary blob column normalized to base64 string for GraphQL.

    - On PostgreSQL: BYTEA
    - On MSSQL: VARBINARY(MAX)
    - On others: LargeBinary/BLOB
    """

    cache_ok = True
    impl = LargeBinary

    def load_dialect_impl(self, dialect):  # pragma: no cover - exercised via DB
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(postgresql.BYTEA())
        if dialect.name == 'mssql':
            try:
                return dialect.type_descriptor(mssql.VARBINARY(None))  # MAX
            except Exception:
                return dialect.type_descriptor(mssql.VARBINARY('max'))
        if dialect.name == 'sqlite':
            # Store as TEXT (base64) so SQLite JSON functions can include it
            from sqlalchemy import String as _String
            return dialect.type_descriptor(_String())
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):  # pragma: no cover - trivial
        if value is None:
            return None
        import base64
        if isinstance(value, (bytes, bytearray, memoryview)):
            if dialect.name == 'sqlite':
                # Persist base64 string
                return base64.b64encode(bytes(value)).decode('ascii')
            return bytes(value)
        # Accept base64 string input
        s = str(value)
        try:
            raw = base64.b64decode(s.encode('ascii'))
            if dialect.name == 'sqlite':
                return s  # already base64 text
            return raw
        except Exception:
            # Last resort: store UTF-8 bytes of the string (or plain text for sqlite)
            return s if dialect.name == 'sqlite' else s.encode('utf-8')

    def process_result_value(self, value, dialect):  # pragma: no cover - trivial
        if value is None:
            return None
        import base64
        if dialect.name == 'sqlite':
            # Stored as base64 text already
            return str(value)
        bb = bytes(value) if isinstance(value, (bytearray, memoryview)) else (value if isinstance(value, bytes) else bytes(value))
        return base64.b64encode(bb).decode('ascii')


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


class PostStatus(enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class Post(Base):
    __tablename__ = 'posts'
    # SQLAlchemy Enum with explicit CHECK constraint name for all dialects
    __table_args__ = ()
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    # New: single binary blob (base64 in GraphQL)
    binary_blob = Column(BinaryBlob(), nullable=True)
    # Enum with helper: ensures hashability and consistent storage; emits named CHECK
    from berryql.sql.enum_helpers import enum_column
    status = enum_column(
        PostStatus,
        nullable=False,
        default=PostStatus.DRAFT,
        constraint_name="ck_post_status",
    )
    # Computed (read-only) column: length of content; uses SQL function for cross-dialect support
    content_length = column_property(func.length(content))
    
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


class View(Base):
    __tablename__ = 'views'

    id = Column(Integer, primary_key=True)
    entity_type = Column(String(50), nullable=False)  # 'post' or 'post_comment'
    entity_id = Column(Integer, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    user = relationship("User")
