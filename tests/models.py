"""Database models for BerryQL tests (shared)."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, JSON, func, CheckConstraint
from sqlalchemy import Enum as SAEnum
import enum
from sqlalchemy.orm import DeclarativeBase, relationship, column_property
from sqlalchemy.types import TypeDecorator, LargeBinary
from sqlalchemy.dialects import postgresql, mssql
from sqlalchemy import Uuid as SA_Uuid
from berryql import enum_column

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
    """Application users (docstring)"""
    __tablename__ = 'users'
    __table_args__ = {'comment': 'Application users'}
    
    id = Column(Integer, primary_key=True, comment='User primary key')
    name = Column(String(100), nullable=False, comment='Public display name')
    email = Column(String(255), unique=True, nullable=False, comment='Unique login email')
    is_admin = Column(Boolean, default=False, nullable=False, comment='Administrative flag')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), comment='Creation timestamp (UTC)')
    
    posts = relationship("Post", back_populates="author")
    post_comments = relationship("PostComment", back_populates="author")


class PostStatus(enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class Post(Base):
    """Blog posts (docstring)"""
    __tablename__ = 'posts'
    # SQLAlchemy Enum with explicit CHECK constraint name for all dialects
    __table_args__ = {'comment': 'Blog posts'}
    
    id = Column(Integer, primary_key=True, comment='Post primary key')
    title = Column(String(200), nullable=False, comment='Post title')
    content = Column(String(5000), comment='Post body text')
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False, comment='Author FK to users.id')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), comment='Creation timestamp (UTC)')
    # New: single binary blob (base64 in GraphQL)
    binary_blob = Column(BinaryBlob(), nullable=True)
    # New: JSON/JSONB metadata column (JSONB on PostgreSQL, JSON elsewhere)
    metadata_json = Column(
        JSON().with_variant(postgresql.JSONB(), 'postgresql'),
        nullable=True,
        comment='Arbitrary metadata (JSON/JSONB)'
    )
    # Enum with helper: ensures hashability and consistent storage; emits named CHECK
    status = enum_column(
        PostStatus,
        nullable=False,
        default=PostStatus.DRAFT,
        constraint_name="ck_post_status",
    )
    # Hidden from Berry schema on purpose: used to verify pre-hook can write
    # to model columns that are not exposed as Berry fields
    internal_note = Column(String(255), nullable=True, comment='Internal note (not exposed)')
    # Computed (read-only) column: length of content; uses SQL function for cross-dialect support
    content_length = column_property(func.length(content))
    
    author = relationship("User", back_populates="posts")
    post_comments = relationship("PostComment", back_populates="post")


class PostComment(Base):
    __tablename__ = 'post_comments'
    __table_args__ = {'comment': 'User comments on posts'}
    
    id = Column(Integer, primary_key=True, comment='Comment primary key')
    content = Column(String(1000), nullable=False, comment='Comment text')
    rate = Column(Integer, nullable=False, default=0, comment='Simple rating value')
    post_id = Column(Integer, ForeignKey('posts.id'), nullable=False, comment='FK to posts.id')
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False, comment='FK to users.id')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), comment='Creation timestamp (UTC)')
    
    post = relationship("Post", back_populates="post_comments")
    author = relationship("User", back_populates="post_comments")
    likes = relationship("PostCommentLike", back_populates="comment")


class PostCommentLike(Base):
    __tablename__ = 'post_comment_likes'
    __table_args__ = {'comment': 'Likes for comments'}
    
    id = Column(Integer, primary_key=True, comment='Like primary key')
    post_comment_id = Column(Integer, ForeignKey('post_comments.id'), nullable=False, comment='FK to post_comments.id')
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, comment='FK to users.id')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), comment='Creation timestamp (UTC)')

    comment = relationship("PostComment", back_populates="likes")
    user = relationship("User")


class View(Base):
    __tablename__ = 'views'
    __table_args__ = {'comment': 'Polymorphic views on posts and comments'}

    id = Column(Integer, primary_key=True, comment='View primary key')
    entity_type = Column(String(50), nullable=False, comment="Entity type: 'post' or 'post_comment'")
    entity_id = Column(Integer, nullable=False, comment='Polymorphic entity id')
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, comment='FK to users.id')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), comment='Creation timestamp (UTC)')

    user = relationship("User")


class GenericItem(Base):
    """Generic test entity with UUID id and assorted typed columns for WHERE tests."""
    __tablename__ = 'generic_items'
    __table_args__ = {'comment': 'Generic items with various column types for testing'}

    id = Column(SA_Uuid(as_uuid=True), primary_key=True, comment='UUID primary key')
    name = Column(String(100), nullable=False, comment='Item name')
    code = Column(String(50), nullable=False, default='', comment='Short code')
    count = Column(Integer, nullable=False, default=0, comment='Numeric counter')
    active = Column(Boolean, nullable=False, default=True, comment='Active flag')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), comment='Creation timestamp (UTC)')
