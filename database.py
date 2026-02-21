"""
Database models for HR Analytics tracking.
Tracks queries, escalations, and FAQ patterns.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean,
    Text, JSON, create_engine, event
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/hr_analytics.db")


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    """Logs every employee query for analytics."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    employee_id = Column(String(100), nullable=False, index=True)
    department = Column(String(100), nullable=True)
    role = Column(String(50), nullable=True)
    query_text = Column(Text, nullable=False)
    query_category = Column(String(100), nullable=True, index=True)  # leave, reimbursement, insurance, etc.
    query_intent = Column(String(100), nullable=True)
    response_text = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    escalated = Column(Boolean, default=False)
    escalation_reason = Column(Text, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    sources_used = Column(JSON, nullable=True)  # list of document sources
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    satisfied = Column(Boolean, nullable=True)  # user feedback
    feedback_text = Column(Text, nullable=True)


class EscalationLog(Base):
    """Tracks escalations to HR team."""
    __tablename__ = "escalation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_log_id = Column(Integer, nullable=False)
    employee_id = Column(String(100), nullable=False)
    department = Column(String(100), nullable=True)
    escalation_type = Column(String(50), nullable=False)  # complex, sensitive, policy_gap
    escalation_reason = Column(Text, nullable=False)
    assigned_to = Column(String(100), nullable=True)  # HR person assigned
    status = Column(String(30), default="pending")  # pending, in_progress, resolved
    priority = Column(String(20), default="medium")  # low, medium, high, critical
    resolution_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class FAQPattern(Base):
    """Tracks frequently asked questions for HR insights."""
    __tablename__ = "faq_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(100), nullable=False, index=True)
    question_pattern = Column(Text, nullable=False)
    frequency = Column(Integer, default=1)
    department_distribution = Column(JSON, nullable=True)  # {dept: count}
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    avg_confidence = Column(Float, nullable=True)
    escalation_rate = Column(Float, default=0.0)


class Document(Base):
    """Tracks ingested HR documents."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, unique=True)
    document_type = Column(String(100), nullable=True)  # policy, handbook, form, etc.
    category = Column(String(100), nullable=True)
    page_count = Column(Integer, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    ingestion_date = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON, nullable=True)


class Employee(Base):
    """Employee directory for RBAC."""
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(100), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), nullable=True)
    department = Column(String(100), nullable=True)
    role = Column(String(50), nullable=False, default="employee")  # employee, manager, hr_admin, hr_manager, executive
    manager_id = Column(String(100), nullable=True)
    join_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Async engine setup
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database initialized successfully")


async def get_db():
    """Dependency to get DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
