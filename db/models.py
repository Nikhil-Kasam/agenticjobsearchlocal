"""
db/models.py — SQLAlchemy async ORM models for the job agent pipeline.

Tables:
  jobs              — Complete job lifecycle (search → evaluate → apply)
  application_attempts — Browser session tracking per job
"""

import enum
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text,
    ForeignKey, Enum as SAEnum, JSON
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    FOUND = "found"
    EVALUATED = "evaluated"
    PENDING_REVIEW = "pending_review"   # Cover letter ready, awaiting user approval
    APPROVED = "approved"               # User clicked Approve in dashboard
    SUBMITTED = "submitted"             # Application submitted via browser
    SKIPPED = "skipped"                 # User clicked Skip
    FAILED = "failed"                   # Evaluation or apply error
    DOMAIN_BLOCKED = "domain_blocked"   # URL not in whitelist


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)  # UUID
    title = Column(String, nullable=False)
    company = Column(String, nullable=False)
    url = Column(String, nullable=False, unique=True)
    description = Column(Text)
    source = Column(String)
    scraped_at = Column(DateTime, default=func.now())

    # Evaluation results (set by Celery evaluate worker)
    match_score = Column(Integer, default=-1)
    top_matching_skills = Column(JSON)    # list[str]
    missing_skills = Column(JSON)         # list[str]
    recommendation = Column(String)       # "apply" | "skip" | "borderline"
    eval_reasoning = Column(Text)

    # Application (set by Celery apply worker)
    cover_letter = Column(Text)
    cover_letter_subject = Column(String)

    # LangGraph thread id for crash recovery
    langgraph_thread_id = Column(String)

    status = Column(SAEnum(JobStatus), default=JobStatus.FOUND, index=True)
    error_message = Column(Text)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    attempts = relationship("ApplicationAttempt", back_populates="job")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "url": self.url,
            "description": self.description,
            "source": self.source,
            "scraped_at": self.scraped_at.isoformat() if self.scraped_at else None,
            "match_score": self.match_score,
            "top_matching_skills": self.top_matching_skills or [],
            "missing_skills": self.missing_skills or [],
            "recommendation": self.recommendation,
            "eval_reasoning": self.eval_reasoning,
            "cover_letter": self.cover_letter,
            "cover_letter_subject": self.cover_letter_subject,
            "status": self.status.value if self.status else None,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ApplicationAttempt(Base):
    __tablename__ = "application_attempts"

    id = Column(String, primary_key=True)  # UUID
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, index=True)
    langgraph_thread_id = Column(String)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    result = Column(String)  # "SUBMITTED" | "FAILED" | "SKIPPED"
    error = Column(Text)

    job = relationship("Job", back_populates="attempts")
