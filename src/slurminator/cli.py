from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from slurminator.config import Settings
from slurminator.db import WatchStore
from slurminator.notifier import DiscordNotifier, StdoutNotifier
from slurminator.probe import GpuNodeProber
from slurminator.service import MonitorService
from slurminator.slurm import SlurmClient

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


def main() -> None:
    app()
