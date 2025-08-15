"""Database models for BerryQL tests (shared)."""

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
    
    posts = relationship("Post", back_populates="author")
    post_comments = relationship("PostComment", back_populates="author")


class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
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
