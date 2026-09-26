"""SQLAlchemy declarative base for future database models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base; no sensitive data models exist yet."""