"""Database models for BerryQL tests."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import DeclarativeBase, relationship


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
    
    # Relationship to posts
    posts = relationship("Post", back_populates="author")
    # Relationship to comments (original name kept for BerryQL); GraphQL field renamed separately
    post_comments = relationship("PostComment", back_populates="author")


class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    # Relationship to user
    author = relationship("User", back_populates="posts")
    # Relationship to comments (original name kept for BerryQL); GraphQL field renamed separately
    post_comments = relationship("PostComment", back_populates="post")


class PostComment(Base):
    __tablename__ = 'post_comments'
    
    id = Column(Integer, primary_key=True)
    content = Column(String(1000), nullable=False)
    # Simple rating field used for default ordering in tests
    rate = Column(Integer, nullable=False, default=0)
    post_id = Column(Integer, ForeignKey('posts.id'), nullable=False)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    # Relationships
    post = relationship("Post", back_populates="post_comments")
    author = relationship("User", back_populates="post_comments")

