from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(slots=True)
class SlurmJob:
    job_id: str
    user_name: str
    job_name: str
    state: str
    node_list: str
    run_time: str | None
    command: str | None
    work_dir: str | None
    gpu_count: int
    raw_fields: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GpuSample:
    index: int
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float

    @property
    def memory_utilization_pct(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "index": self.index,
            "utilization_pct": self.utilization_pct,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "memory_utilization_pct": self.memory_utilization_pct,
        }


@dataclass(slots=True)
class NodeSample:
    node_name: str
    gpus: list[GpuSample] = field(default_factory=list)
    error: str | None = None

    @property
    def max_gpu_utilization(self) -> float | None:
        if not self.gpus:
            return None
        return max(gpu.utilization_pct for gpu in self.gpus)

    @property
    def max_memory_utilization(self) -> float | None:
        if not self.gpus:
            return None
        return max(gpu.memory_utilization_pct for gpu in self.gpus)

    def to_dict(self) -> dict[str, object]:
        return {
            "node_name": self.node_name,
            "error": self.error,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "max_gpu_utilization": self.max_gpu_utilization,
            "max_memory_utilization": self.max_memory_utilization,
        }


@dataclass(slots=True)
class JobEvaluation:
    job: SlurmJob
    nodes: list[str]
    samples: list[NodeSample]
    observed_at: datetime
    status: Literal["idle", "active", "unknown"]
    summary: str
    max_gpu_utilization: float | None = None
    max_memory_utilization: float | None = None

    def to_snapshot(self) -> dict[str, object]:
        return {
            "job_id": self.job.job_id,
            "job_name": self.job.job_name,
            "user_name": self.job.user_name,
            "status": self.status,
            "summary": self.summary,
            "observed_at": self.observed_at.isoformat(),
            "nodes": self.nodes,
            "max_gpu_utilization": self.max_gpu_utilization,
            "max_memory_utilization": self.max_memory_utilization,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(slots=True)
class NotificationHandle:
    channel_id: str | None = None
    message_id: str | None = None


@dataclass(slots=True)
class TerminationResult:
    success: bool
    user_message: str
