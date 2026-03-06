from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from slurminator.models import NotificationHandle, SlurmJob


class Base(DeclarativeBase):
    pass


class JobWatch(Base):
    __tablename__ = "job_watches"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_name: Mapped[str] = mapped_column(String(128), nullable=False)
    job_name: Mapped[str] = mapped_column(String(256), nullable=False)
    node_list: Mapped[str] = mapped_column(Text, default="", nullable=False)
    gpu_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    idle_since_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    warned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    warning_channel_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    warning_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    last_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    last_snapshot_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)


class WatchStore:
    def __init__(self, database_url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(database_url, future=True)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def get_watch(self, job_id: str) -> JobWatch | None:
        async with self.session_factory() as session:
            return await session.get(JobWatch, job_id)

    async def list_open_watches(self) -> list[JobWatch]:
        async with self.session_factory() as session:
            result = await session.scalars(
                select(JobWatch).where(JobWatch.resolved_at.is_(None))
            )
            return list(result)

    async def list_warned_open_watches(self) -> list[JobWatch]:
        async with self.session_factory() as session:
            result = await session.scalars(
                select(JobWatch).where(
                    JobWatch.resolved_at.is_(None),
                    JobWatch.warned_at.is_not(None),
                    JobWatch.warning_channel_id.is_not(None),
                    JobWatch.warning_message_id.is_not(None),
                )
            )
            return list(result)

    async def touch_running_job(
        self,
        job: SlurmJob,
        observed_at: datetime,
        *,
        summary: str,
        snapshot_json: str,
    ) -> JobWatch:
        async with self.session_factory.begin() as session:
            watch = await session.get(JobWatch, job.job_id)
            if watch is None:
                watch = JobWatch(
                    job_id=job.job_id,
                    user_name=job.user_name,
                    job_name=job.job_name,
                    node_list=job.node_list,
                    gpu_count=job.gpu_count,
                    first_seen_at=observed_at,
                    last_seen_at=observed_at,
                    last_summary=summary,
                    last_snapshot_json=snapshot_json,
                )
                session.add(watch)
                return watch

            if watch.resolved_at is not None:
                watch.first_seen_at = observed_at
                watch.idle_since_at = None
                watch.warned_at = None
                watch.warning_channel_id = None
                watch.warning_message_id = None
                watch.resolution = None
                watch.resolution_note = None
                watch.resolved_at = None

            watch.user_name = job.user_name
            watch.job_name = job.job_name
            watch.node_list = job.node_list
            watch.gpu_count = job.gpu_count
            watch.last_seen_at = observed_at
            watch.last_summary = summary
            watch.last_snapshot_json = snapshot_json
            return watch

    async def mark_idle(
        self,
        job_id: str,
        observed_at: datetime,
        *,
        summary: str,
        snapshot_json: str,
    ) -> JobWatch | None:
        async with self.session_factory.begin() as session:
            watch = await session.get(JobWatch, job_id)
            if watch is None:
                return None
            if watch.idle_since_at is None:
                watch.idle_since_at = observed_at
            watch.last_seen_at = observed_at
            watch.last_summary = summary
            watch.last_snapshot_json = snapshot_json
            return watch

    async def clear_idle(
        self,
        job_id: str,
        observed_at: datetime,
        *,
        summary: str,
        snapshot_json: str,
    ) -> JobWatch | None:
        async with self.session_factory.begin() as session:
            watch = await session.get(JobWatch, job_id)
            if watch is None:
                return None
            watch.idle_since_at = None
            watch.last_seen_at = observed_at
            watch.last_summary = summary
            watch.last_snapshot_json = snapshot_json
            return watch

    async def record_warning(
        self,
        job_id: str,
        warned_at: datetime,
        *,
        summary: str,
        snapshot_json: str,
        handle: NotificationHandle | None = None,
    ) -> JobWatch | None:
        async with self.session_factory.begin() as session:
            watch = await session.get(JobWatch, job_id)
            if watch is None:
                return None
            watch.warned_at = warned_at
            watch.last_seen_at = warned_at
            watch.last_summary = summary
            watch.last_snapshot_json = snapshot_json
            watch.warning_channel_id = handle.channel_id if handle else None
            watch.warning_message_id = handle.message_id if handle else None
            return watch

    async def clear_warning(
        self,
        job_id: str,
        observed_at: datetime,
        *,
        note: str,
        clear_idle: bool = False,
    ) -> JobWatch | None:
        async with self.session_factory.begin() as session:
            watch = await session.get(JobWatch, job_id)
            if watch is None:
                return None
            watch.warned_at = None
            watch.warning_channel_id = None
            watch.warning_message_id = None
            if clear_idle:
                watch.idle_since_at = None
            watch.last_seen_at = observed_at
            watch.last_summary = note
            return watch

    async def mark_resolved(
        self,
        job_id: str,
        resolved_at: datetime,
        *,
        resolution: str,
        note: str,
    ) -> JobWatch | None:
        async with self.session_factory.begin() as session:
            watch = await session.get(JobWatch, job_id)
            if watch is None:
                return None
            watch.last_seen_at = resolved_at
            watch.last_summary = note
            watch.resolved_at = resolved_at
            watch.resolution = resolution
            watch.resolution_note = note
            return watch
