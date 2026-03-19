from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from slurminator.config import Settings
from slurminator.db import JobWatch, WatchStore
from slurminator.models import JobEvaluation, SlurmJob, UserHistorySnapshot, WarningContext
from slurminator.notifier import DiscordNotifier, StdoutNotifier
from slurminator.probe import GpuNodeProber
from slurminator.service import MonitorService
from slurminator.slurm import SlurmClient
from slurminator.warning_text import WarningMessageComposer

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def build_service(settings: Settings) -> MonitorService:
    return MonitorService(
        settings=settings,
        store=WatchStore(settings.database_url),
        slurm=SlurmClient(settings),
        prober=GpuNodeProber(settings),
    )


@app.command()
def run() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    asyncio.run(_run(settings))


@app.command()
def once() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    asyncio.run(_once(settings))


@app.command()
def kill(
    job_id: str,
    actor: str = typer.Option("CLI", help="Name recorded in the termination note."),
) -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    success = asyncio.run(_kill(settings, job_id=job_id, actor=actor))
    if not success:
        raise typer.Exit(code=1)


@app.command("preview-warning")
def preview_warning(
    user_name: str = typer.Option("example-user"),
    job_id: str = typer.Option("123456"),
    job_name: str = typer.Option("train-gpt"),
    gpu_count: int = typer.Option(4, min=1),
    idle_seconds: int = typer.Option(5400, min=1),
    warning_count: int = typer.Option(0, min=0),
    auto_kill_count: int = typer.Option(0, min=0),
    manual_kill_count: int = typer.Option(0, min=0),
    total_idle_cost_usd: float = typer.Option(0.0, min=0.0),
) -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    asyncio.run(
        _preview_warning(
            settings,
            user_name=user_name,
            job_id=job_id,
            job_name=job_name,
            gpu_count=gpu_count,
            idle_seconds=idle_seconds,
            warning_count=warning_count,
            auto_kill_count=auto_kill_count,
            manual_kill_count=manual_kill_count,
            total_idle_cost_usd=total_idle_cost_usd,
        )
    )


async def _run(settings: Settings) -> None:
    service = build_service(settings)
    if settings.notifier == "discord":
        notifier = DiscordNotifier(settings, service, service.identities)
        service.set_notifier(notifier)
        await notifier.start(settings.require_discord_token())
        return

    service.set_notifier(StdoutNotifier())
    await service.initialize()
    await service.run_forever()


async def _once(settings: Settings) -> None:
    service = build_service(settings)
    await service.initialize()
    evaluations = await service.inspect_once()

    table = Table(title="Slurminator inspection")
    table.add_column("Job")
    table.add_column("User")
    table.add_column("Nodes")
    table.add_column("GPUs")
    table.add_column("Status")
    table.add_column("Observed")

    for evaluation in evaluations:
        status_style = {
            "idle": "bold red",
            "active": "green",
            "unknown": "yellow",
        }[evaluation.status]
        table.add_row(
            evaluation.job.job_id,
            evaluation.job.user_name,
            ",".join(evaluation.nodes) or "-",
            str(evaluation.job.gpu_count),
            f"[{status_style}]{evaluation.status}[/{status_style}]",
            evaluation.summary,
        )

    if not evaluations:
        console.print("No matching running jobs found.")
        return

    console.print(table)


async def _kill(settings: Settings, *, job_id: str, actor: str) -> bool:
    service = build_service(settings)
    await service.initialize()
    result = await service.manual_terminate(job_id, actor=actor)
    console.print(result.user_message)
    return result.success


async def _preview_warning(
    settings: Settings,
    *,
    user_name: str,
    job_id: str,
    job_name: str,
    gpu_count: int,
    idle_seconds: int,
    warning_count: int,
    auto_kill_count: int,
    manual_kill_count: int,
    total_idle_cost_usd: float,
) -> None:
    now = datetime.now(timezone.utc)
    watch = JobWatch(
        job_id=job_id,
        user_name=user_name,
        job_name=job_name,
        node_list="gpu-a001",
        gpu_count=gpu_count,
        first_seen_at=now - timedelta(seconds=idle_seconds + 300),
        last_seen_at=now,
        idle_since_at=now - timedelta(seconds=idle_seconds),
        warned_at=None,
        warning_channel_id=None,
        warning_message_id=None,
        last_summary="Preview warning generated from CLI.",
        last_snapshot_json="{}",
        resolved_at=None,
        resolution=None,
        resolution_note=None,
    )
    evaluation = JobEvaluation(
        job=SlurmJob(
            job_id=job_id,
            user_name=user_name,
            job_name=job_name,
            state="RUNNING",
            node_list=watch.node_list,
            run_time=None,
            command=None,
            work_dir=None,
            gpu_count=gpu_count,
        ),
        nodes=[watch.node_list],
        samples=[],
        observed_at=now,
        status="idle",
        summary="No GPU activity observed in preview sample.",
    )
    current_idle_gpu_hours = gpu_count * idle_seconds / 3600.0
    current_idle_cost_usd = current_idle_gpu_hours * settings.gpu_hourly_cost_usd
    context = WarningContext(
        current_idle_seconds=float(idle_seconds),
        current_idle_gpu_hours=current_idle_gpu_hours,
        current_idle_cost_usd=current_idle_cost_usd,
        gpu_hourly_cost_usd=settings.gpu_hourly_cost_usd,
        history=UserHistorySnapshot(
            user_name=user_name,
            warning_count=warning_count,
            auto_kill_count=auto_kill_count,
            manual_kill_count=manual_kill_count,
            total_idle_cost_usd=total_idle_cost_usd,
        ),
    )

    composer = WarningMessageComposer(settings)
    context = composer.enrich_context(context)
    context.custom_intro = await composer.compose_intro(watch, evaluation, context)

    console.print("Warning preview")
    console.print(f"Mode: {settings.warning_message_mode}")
    console.print(f"Persona/severity: {context.persona_preset}/{context.severity_band}")
    console.print(f"Reasons: {', '.join(context.severity_reasons)}")
    console.print(context.custom_intro or context.fallback_intro)


def main() -> None:
    app()
