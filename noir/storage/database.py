"""Database schema and ORM models for Project NOIR using SQLAlchemy."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from noir.core.logging import get_logger

logger = get_logger("database")

Base = declarative_base()


class ExperimentModel(Base):
    __tablename__ = "experiments"

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    parent_id = Column(String(64), ForeignKey("experiments.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(32), default="CREATED")  # CREATED, RUNNING, PAUSED, COMPLETED, FAILED
    config_json = Column(Text, nullable=False, default="{}")
    best_metric = Column(Float, nullable=True)
    total_steps = Column(Integer, default=0)
    current_epoch = Column(Integer, default=0)
    description = Column(Text, nullable=True)

    # Relationships
    children = relationship("ExperimentModel", backref="parent", remote_side=[id])
    metrics = relationship("MetricModel", back_populates="experiment", cascade="all, delete-orphan")
    emotional_states = relationship("EmotionalStateModel", back_populates="experiment", cascade="all, delete-orphan")
    strategies = relationship("StrategyModel", back_populates="experiment", cascade="all, delete-orphan")
    memories = relationship("MemoryRecordModel", back_populates="experiment", cascade="all, delete-orphan")


class MetricModel(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), ForeignKey("experiments.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False, index=True)
    epoch = Column(Integer, default=0)
    metric_name = Column(String(64), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(Float, default=datetime.utcnow().timestamp)

    experiment = relationship("ExperimentModel", back_populates="metrics")

    __table_args__ = (
        Index("idx_metric_exp_name_step", "experiment_id", "metric_name", "step"),
    )


class EmotionalStateModel(Base):
    __tablename__ = "emotional_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), ForeignKey("experiments.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    frustration = Column(Float, nullable=False)
    anticipation = Column(Float, nullable=False)
    satisfaction = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)
    curiosity = Column(Float, nullable=False)
    caution = Column(Float, nullable=False)
    persistence = Column(Float, nullable=False)
    timestamp = Column(Float, default=datetime.utcnow().timestamp)

    experiment = relationship("ExperimentModel", back_populates="emotional_states")


class StrategyModel(Base):
    __tablename__ = "strategies"

    id = Column(String(64), primary_key=True)
    experiment_id = Column(String(64), ForeignKey("experiments.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    hypothesis = Column(Text, nullable=False)
    proposal = Column(Text, nullable=False)
    status = Column(String(32), default="PROPOSED")  # PROPOSED, ACCEPTED, REJECTED, EXECUTED
    metrics_summary = Column(Text, nullable=True)
    timestamp = Column(Float, default=datetime.utcnow().timestamp)

    experiment = relationship("ExperimentModel", back_populates="strategies")


class MemoryRecordModel(Base):
    __tablename__ = "memory_records"

    id = Column(String(64), primary_key=True)
    experiment_id = Column(String(64), ForeignKey("experiments.id"), nullable=False, index=True)
    memory_type = Column(String(32), nullable=False)  # SHORT_TERM, EPISODIC, SEMANTIC
    key = Column(String(255), nullable=False)
    content_json = Column(Text, nullable=False)
    importance = Column(Float, default=1.0)
    timestamp = Column(Float, default=datetime.utcnow().timestamp)

    experiment = relationship("ExperimentModel", back_populates="memories")


class DatabaseManager:
    """Manages SQLite database connections and table creation."""

    def __init__(self, db_path: str | Path = "noir.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{self.db_path.as_posix()}"
        self.engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
        self.SessionFactory = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
        self.create_tables()

    def create_tables(self) -> None:
        """Create all relational tables if they do not exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables initialized at %s", self.db_path)

    def get_session(self) -> Session:
        """Provide a new SQLAlchemy session."""
        return self.SessionFactory()
