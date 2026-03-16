from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

import discord

from slurminator.config import Settings
from slurminator.db import JobWatch
from slurminator.identity import IdentityDirectory
from slurminator.models import JobEvaluation, NotificationHandle, WarningContext
from slurminator.util import format_currency, format_duration, format_hours

if TYPE_CHECKING:
    from slurminator.service import MonitorService

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    async def send_warning(
        self,
        watch: JobWatch,
        evaluation: JobEvaluation,
        *,
        warning_context: WarningContext,
        kill_deadline: datetime,
    ) -> NotificationHandle | None: ...

    async def close_warning(self, watch: JobWatch, *, note: str) -> None: ...


class StdoutNotifier:
    async def send_warning(
        self,
        watch: JobWatch,
        evaluation: JobEvaluation,
        *,
        warning_context: WarningContext,
        kill_deadline: datetime,
    ) -> NotificationHandle:
        logger.warning(
            "WARNING job=%s user=%s name=%s message=%r kill_deadline=%s",
            watch.job_id,
            watch.user_name,
            watch.job_name,
            self._build_warning_message(watch, evaluation, warning_context, kill_deadline),
            kill_deadline.isoformat(),
        )
        return NotificationHandle()

    async def close_warning(self, watch: JobWatch, *, note: str) -> None:
        logger.info("ALERT CLOSED job=%s note=%s", watch.job_id, note)

    def _build_warning_message(
        self,
        watch: JobWatch,
        evaluation: JobEvaluation,
        warning_context: WarningContext,
        kill_deadline: datetime,
    ) -> str:
        idle_for = (
            evaluation.observed_at - watch.idle_since_at
            if watch.idle_since_at is not None
            else None
        )
        lines = [warning_context.custom_intro or "Slurminator found an idle GPU job."]
        lines.append(f"Job: `{watch.job_id}` (`{watch.job_name}`)")
        if idle_for is not None:
            lines.append(f"Idle for: {format_duration(idle_for)}")
        lines.append(
            "Current idle waste: "
            f"{format_hours(warning_context.current_idle_gpu_hours)} GPU-hours"
            + (
                f" (~{format_currency(warning_context.current_idle_cost_usd)})"
                if warning_context.current_idle_cost_usd > 0
                else ""
            )
        )
        if warning_context.history.warning_count:
            lines.append(_build_history_line(warning_context))
        lines.append(f"Kill deadline: {kill_deadline.isoformat()}")
        return "\n".join(lines)


class KillJobButton(discord.ui.Button["KillJobView"]):
    def __init__(self, service: "MonitorService", job_id: str) -> None:
        super().__init__(
            label="Terminate job",
            style=discord.ButtonStyle.danger,
            custom_id=f"slurminator:kill:{job_id}",
        )
        self.service = service
        self.job_id = job_id

    async def callback(self, interaction: discord.Interaction) -> None:
        role_ids = {
            role.id
            for role in getattr(interaction.user, "roles", [])
            if isinstance(role, discord.Role)
        }
        allowed, message = await self.service.authorize_manual_kill(
            self.job_id,
            actor_user_id=interaction.user.id,
            actor_role_ids=role_ids,
        )
        if not allowed:
            await interaction.response.send_message(message, ephemeral=True)
            return

        actor_name = getattr(interaction.user, "display_name", interaction.user.name)
        result = await self.service.manual_terminate(self.job_id, actor=actor_name)
        await interaction.response.send_message(result.user_message, ephemeral=True)


class KillJobView(discord.ui.View):
    def __init__(self, service: "MonitorService", job_id: str) -> None:
        super().__init__(timeout=None)
        self.add_item(KillJobButton(service, job_id))


class DiscordNotifier(discord.Client):
    def __init__(
        self,
        settings: Settings,
        service: "MonitorService",
        identities: IdentityDirectory,
    ) -> None:
        intents = discord.Intents(guilds=True)
        super().__init__(intents=intents)
        self.settings = settings
        self.service = service
        self.identities = identities
        self._monitor_task: asyncio.Task[None] | None = None

    async def setup_hook(self) -> None:
        await self.service.initialize()
        await self._restore_views()
        self._monitor_task = asyncio.create_task(self._run_monitor_loop())

    async def close(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task
        await super().close()

    async def on_ready(self) -> None:
        logger.info("Discord client connected as %s", self.user)

    async def send_warning(
        self,
        watch: JobWatch,
        evaluation: JobEvaluation,
        *,
        warning_context: WarningContext,
        kill_deadline: datetime,
    ) -> NotificationHandle:
        channel = await self._get_channel(self.settings.discord_channel_id)
        view = KillJobView(self.service, watch.job_id)
        message = await channel.send(
            content=self._build_warning_message(
                watch,
                evaluation,
                warning_context,
                kill_deadline,
            ),
            view=view,
        )
        return NotificationHandle(
            channel_id=str(channel.id),
            message_id=str(message.id),
        )

    async def close_warning(self, watch: JobWatch, *, note: str) -> None:
        message = await self._fetch_warning_message(watch)
        if message is None:
            return
        await message.edit(content=self._build_closed_message(watch, note), view=None)

    async def _run_monitor_loop(self) -> None:
        await self.wait_until_ready()
        await self.service.run_forever()

    async def _restore_views(self) -> None:
        warned_watches = await self.service.store.list_warned_open_watches()
        for watch in warned_watches:
            if watch.warning_message_id is None:
                continue
            self.add_view(KillJobView(self.service, watch.job_id), message_id=int(watch.warning_message_id))

    async def _get_channel(
        self,
        channel_id: int | None,
    ) -> discord.TextChannel | discord.Thread:
        if channel_id is None:
            msg = "SLURMINATOR_DISCORD_CHANNEL_ID is required when notifier=discord"
            raise RuntimeError(msg)

        channel = self.get_channel(channel_id)
        if channel is None:
            channel = await self.fetch_channel(channel_id)

        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            return channel

        msg = f"Discord channel {channel_id} is not a text channel or thread"
        raise RuntimeError(msg)

    async def _fetch_warning_message(self, watch: JobWatch) -> discord.Message | None:
        if watch.warning_channel_id is None or watch.warning_message_id is None:
            return None

        try:
            channel = await self._get_channel(int(watch.warning_channel_id))
            return await channel.fetch_message(int(watch.warning_message_id))
        except (discord.NotFound, discord.Forbidden):
            logger.warning("Could not fetch Discord warning message for job %s", watch.job_id)
            return None

    def _build_warning_message(
        self,
        watch: JobWatch,
        evaluation: JobEvaluation,
        warning_context: WarningContext,
        kill_deadline: datetime,
    ) -> str:
        mentions = self.identities.discord_mentions(watch.user_name)
        owner_reference = " ".join(mentions) if mentions else f"`{watch.user_name}`"
        idle_for = (
            evaluation.observed_at - watch.idle_since_at
            if watch.idle_since_at is not None
            else None
        )
        deadline_relative = discord.utils.format_dt(kill_deadline, style="R")
        deadline_absolute = discord.utils.format_dt(kill_deadline, style="f")

        lines = [
            (
                f"{owner_reference} {warning_context.custom_intro}"
                if warning_context.custom_intro is not None
                else f"{owner_reference} Slurminator found an idle GPU job."
            ),
            f"Job: `{watch.job_id}` (`{watch.job_name}`)",
            f"Nodes: `{watch.node_list}` | GPUs requested: `{watch.gpu_count}`",
            f"Observed: {evaluation.summary}",
        ]
        if idle_for is not None:
            lines.append(f"Idle for: {format_duration(idle_for)}")
        lines.append(
            "Current idle waste: "
            f"{format_hours(warning_context.current_idle_gpu_hours)} GPU-hours"
            + (
                f" (~{format_currency(warning_context.current_idle_cost_usd)})"
                if warning_context.current_idle_cost_usd > 0
                else ""
            )
        )
        if warning_context.gpu_hourly_cost_usd > 0:
            lines.append(
                "Cost model: "
                f"{format_currency(warning_context.gpu_hourly_cost_usd)}/GPU-hour"
            )
        if warning_context.history.warning_count:
            lines.append(_build_history_line(warning_context))
        lines.append(
            "Press **Terminate job** before "
            f"{deadline_relative} ({deadline_absolute}) or Slurminator will cancel it."
        )
        return "\n".join(lines)

    def _build_closed_message(self, watch: JobWatch, note: str) -> str:
        return "\n".join(
            [
                f"Slurminator closed the alert for job `{watch.job_id}` (`{watch.job_name}`).",
                note,
            ]
        )


def _build_history_line(warning_context: WarningContext) -> str:
    history = warning_context.history
    parts = [
        f"History: {history.warning_count} prior warning(s)",
        f"{format_hours(history.total_idle_gpu_hours)} idle GPU-hours",
    ]
    if history.total_idle_cost_usd > 0:
        parts.append(f"~{format_currency(history.total_idle_cost_usd)} estimated waste")
    if history.auto_kill_count:
        parts.append(f"{history.auto_kill_count} auto-kill(s)")
    if history.manual_kill_count:
        parts.append(f"{history.manual_kill_count} manual kill(s)")
    return " | ".join(parts)
