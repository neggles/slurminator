from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from slurminator.config import Settings
from slurminator.db import JobWatch, WatchStore
from slurminator.identity import IdentityDirectory
from slurminator.models import JobEvaluation, NotificationHandle, SlurmJob, TerminationResult
from slurminator.probe import GpuNodeProber
from slurminator.slurm import SlurmClient, SlurmCommandError
from slurminator.util import format_duration, utcnow

if TYPE_CHECKING:
    from slurminator.notifier import Notifier

logger = logging.getLogger(__name__)


class MonitorService:
    def __init__(
        self,
        settings: Settings,
        store: WatchStore,
        slurm: SlurmClient,
        prober: GpuNodeProber,
    ) -> None:
        self.settings = settings
        self.store = store
        self.slurm = slurm
        self.prober = prober
        self.identities = IdentityDirectory.from_path(settings.user_map_path)
        self.notifier: Notifier | None = None

    def set_notifier(self, notifier: "Notifier | None") -> None:
        self.notifier = notifier

    async def initialize(self) -> None:
        await self.store.initialize()

    async def run_forever(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Monitoring loop failed")
            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def inspect_once(self) -> list[JobEvaluation]:
        return await self._collect_evaluations()

    async def poll_once(self) -> None:
        evaluations = await self._collect_evaluations()
        observed_job_ids = {evaluation.job.job_id for evaluation in evaluations}

        for evaluation in evaluations:
            await self._apply_evaluation(evaluation)

        await self._resolve_missing_jobs(observed_job_ids)

    async def authorize_manual_kill(
        self,
        job_id: str,
        *,
        actor_user_id: int,
        actor_role_ids: set[int],
    ) -> tuple[bool, str]:
        watch = await self.store.get_watch(job_id)
        if watch is None or watch.resolved_at is not None:
            return False, f"Job `{job_id}` is no longer being tracked."

        if actor_role_ids.intersection(self.settings.discord_admin_role_ids):
            return True, "Moderator override accepted."

        owner_ids = self.identities.discord_user_ids(watch.user_name)
        if actor_user_id in owner_ids:
            return True, "Owner match accepted."

        if owner_ids:
            return (
                False,
                "Only the mapped job owner or a configured moderator role can terminate this job.",
            )

        return (
            False,
            f"No Discord identity mapping is configured for Slurm user `{watch.user_name}`.",
        )

    async def manual_terminate(self, job_id: str, *, actor: str) -> TerminationResult:
        watch = await self.store.get_watch(job_id)
        if watch is None or watch.resolved_at is not None:
            return TerminationResult(False, f"Job `{job_id}` is no longer being tracked.")

        if self.settings.dry_run:
            return TerminationResult(
                False,
                f"Dry run is enabled, so Slurminator did not cancel `{job_id}`.",
            )

        note = f"Cancelled by {actor}."
        try:
            await self.slurm.cancel_job(job_id)
        except SlurmCommandError as exc:
            logger.warning("Manual termination failed for %s: %s", job_id, exc)
            return TerminationResult(False, f"Failed to cancel `{job_id}`: {exc}")

        await self._close_warning_notification(watch, note=note)

        await self.store.mark_resolved(
            job_id,
            utcnow(),
            resolution="manual-kill",
            note=note,
        )
        return TerminationResult(True, f"Cancelled `{job_id}`.")

    async def _collect_evaluations(self) -> list[JobEvaluation]:
        jobs = await self.slurm.list_running_jobs()
        if self.settings.only_gpu_jobs:
            jobs = [job for job in jobs if job.gpu_count > 0]

        evaluations: list[JobEvaluation] = []
        for job in jobs:
            evaluations.append(await self._evaluate_job(job))
        return evaluations

    async def _evaluate_job(self, job: SlurmJob) -> JobEvaluation:
        observed_at = utcnow()
        nodes = await self.slurm.expand_hostlist(job.node_list)
        if not nodes:
            return JobEvaluation(
                job=job,
                nodes=[],
                samples=[],
                observed_at=observed_at,
                status="unknown",
                summary="Skipped enforcement because the job did not expose an allocated node list.",
            )

        samples = await self.prober.probe_nodes(nodes)
        failed_nodes = [sample.node_name for sample in samples if sample.error]
        if failed_nodes:
            joined = ", ".join(failed_nodes)
            return JobEvaluation(
                job=job,
                nodes=nodes,
                samples=samples,
                observed_at=observed_at,
                status="unknown",
                summary=f"Skipped enforcement because GPU telemetry was unavailable on {joined}.",
            )

        if any(not sample.gpus for sample in samples):
            return JobEvaluation(
                job=job,
                nodes=nodes,
                samples=samples,
                observed_at=observed_at,
                status="unknown",
                summary="Skipped enforcement because one or more nodes reported no GPUs.",
            )

        max_gpu_utilization = max(sample.max_gpu_utilization or 0.0 for sample in samples)
        max_memory_utilization = max(sample.max_memory_utilization or 0.0 for sample in samples)
        idle = (
            max_gpu_utilization < self.settings.gpu_utilization_threshold_percent
            and max_memory_utilization < self.settings.gpu_memory_threshold_percent
        )
        status = "idle" if idle else "active"
        summary = (
            f"Observed max GPU util {max_gpu_utilization:.1f}% and max memory util "
            f"{max_memory_utilization:.1f}% across {len(nodes)} node(s)."
        )

        return JobEvaluation(
            job=job,
            nodes=nodes,
            samples=samples,
            observed_at=observed_at,
            status=status,
            summary=summary,
            max_gpu_utilization=max_gpu_utilization,
            max_memory_utilization=max_memory_utilization,
        )

    async def _apply_evaluation(self, evaluation: JobEvaluation) -> None:
        snapshot_json = json.dumps(evaluation.to_snapshot(), sort_keys=True)
        watch = await self.store.touch_running_job(
            evaluation.job,
            evaluation.observed_at,
            summary=evaluation.summary,
            snapshot_json=snapshot_json,
        )

        if evaluation.status == "unknown":
            logger.info(
                "Job %s is currently not enforceable: %s",
                evaluation.job.job_id,
                evaluation.summary,
            )
            return

        if evaluation.status == "active":
            await self._handle_active_job(watch, evaluation, snapshot_json)
            return

        await self._handle_idle_job(watch, evaluation, snapshot_json)

    async def _handle_active_job(
        self,
        watch: JobWatch,
        evaluation: JobEvaluation,
        snapshot_json: str,
    ) -> None:
        note = "GPU activity resumed, so the alert was closed."
        if watch.warned_at is not None:
            await self._close_warning_notification(watch, note=note)
            await self.store.clear_warning(
                watch.job_id,
                evaluation.observed_at,
                note=note,
                clear_idle=True,
            )
            return

        await self.store.clear_idle(
            watch.job_id,
            evaluation.observed_at,
            summary=evaluation.summary,
            snapshot_json=snapshot_json,
        )

    async def _handle_idle_job(
        self,
        watch: JobWatch,
        evaluation: JobEvaluation,
        snapshot_json: str,
    ) -> None:
        watch = await self.store.mark_idle(
            watch.job_id,
            evaluation.observed_at,
            summary=evaluation.summary,
            snapshot_json=snapshot_json,
        )
        if watch is None or watch.idle_since_at is None:
            return

        idle_for = evaluation.observed_at - watch.idle_since_at
        if watch.warned_at is None:
            if idle_for < timedelta(seconds=self.settings.idle_warning_after_seconds):
                logger.info(
                    "Job %s is idle but below warning threshold (%s < %s).",
                    watch.job_id,
                    format_duration(idle_for),
                    format_duration(self.settings.idle_warning_after_seconds),
                )
                return

            handle: NotificationHandle | None = None
            warning_delivered = True
            if self.notifier is not None:
                kill_deadline = evaluation.observed_at + timedelta(
                    seconds=self.settings.idle_kill_grace_seconds
                )
                try:
                    handle = await self.notifier.send_warning(
                        watch,
                        evaluation,
                        kill_deadline=kill_deadline,
                    )
                except Exception:
                    warning_delivered = False
                    logger.exception("Failed to send warning for job %s", watch.job_id)

            if not warning_delivered:
                return

            await self.store.record_warning(
                watch.job_id,
                evaluation.observed_at,
                summary=evaluation.summary,
                snapshot_json=snapshot_json,
                handle=handle,
            )
            logger.warning(
                "Warned on idle job %s after %s.",
                watch.job_id,
                format_duration(idle_for),
            )
            return

        warned_for = evaluation.observed_at - watch.warned_at
        if warned_for < timedelta(seconds=self.settings.idle_kill_grace_seconds):
            logger.info(
                "Job %s is still in the warning grace period (%s < %s).",
                watch.job_id,
                format_duration(warned_for),
                format_duration(self.settings.idle_kill_grace_seconds),
            )
            return

        await self._auto_kill(watch, evaluation)

    async def _auto_kill(self, watch: JobWatch, evaluation: JobEvaluation) -> None:
        note = (
            "Warning grace period expired with no GPU activity, so Slurminator cancelled the job."
        )
        if self.settings.dry_run:
            logger.warning("Dry run enabled; would have auto-cancelled job %s.", watch.job_id)
            return

        try:
            await self.slurm.cancel_job(watch.job_id)
        except SlurmCommandError as exc:
            logger.warning("Auto-cancel failed for %s: %s", watch.job_id, exc)
            return

        await self._close_warning_notification(watch, note=note)

        await self.store.mark_resolved(
            watch.job_id,
            evaluation.observed_at,
            resolution="auto-kill",
            note=note,
        )
        logger.warning("Auto-cancelled idle job %s.", watch.job_id)

    async def _resolve_missing_jobs(self, observed_job_ids: set[str]) -> None:
        open_watches = await self.store.list_open_watches()
        now = utcnow()
        for watch in open_watches:
            if watch.job_id in observed_job_ids:
                continue

            note = "Job is no longer running in Slurm."
            await self._close_warning_notification(watch, note=note)

            await self.store.mark_resolved(
                watch.job_id,
                now,
                resolution="gone",
                note=note,
            )

    async def _close_warning_notification(self, watch: JobWatch, *, note: str) -> None:
        if self.notifier is None or watch.warned_at is None:
            return
        try:
            await self.notifier.close_warning(watch, note=note)
        except Exception:
            logger.exception("Failed to close warning for job %s", watch.job_id)
