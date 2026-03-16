from __future__ import annotations

import asyncio
import json
import logging
import os

from openai import OpenAI

from slurminator.config import Settings
from slurminator.db import JobWatch
from slurminator.models import JobEvaluation, WarningContext

logger = logging.getLogger(__name__)


class WarningMessageComposer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.warning_message_mode == "openai"
        self._client: OpenAI | None = None
        self._warned_missing_key = False

        if not self.enabled:
            return

        api_key = settings.openai_api_key_value()
        if api_key is None and os.getenv("OPENAI_API_KEY") is None:
            self.enabled = False
            self._warned_missing_key = True
            logger.warning(
                "warning_message_mode=openai but no OpenAI API key was configured; "
                "falling back to static messages."
            )
            return

        client_kwargs: dict[str, object] = {"timeout": settings.openai_timeout_seconds}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = OpenAI(**client_kwargs)

    async def compose_intro(
        self,
        watch: JobWatch,
        evaluation: JobEvaluation,
        context: WarningContext,
    ) -> str | None:
        if not self.enabled or self._client is None:
            return None

        try:
            return await asyncio.to_thread(
                self._compose_intro_sync,
                watch,
                evaluation,
                context,
            )
        except Exception:
            logger.exception("OpenAI warning generation failed for job %s", watch.job_id)
            return None

    def _compose_intro_sync(
        self,
        watch: JobWatch,
        evaluation: JobEvaluation,
        context: WarningContext,
    ) -> str | None:
        if self._client is None:
            return None

        response = self._client.responses.create(
            model=self.settings.openai_model,
            instructions=self._instructions(),
            input=json.dumps(
                {
                    "user_name": watch.user_name,
                    "job_id": watch.job_id,
                    "job_name": watch.job_name,
                    "gpu_count": watch.gpu_count,
                    "node_list": watch.node_list,
                    "current_idle_seconds": round(context.current_idle_seconds, 1),
                    "current_idle_gpu_hours": round(context.current_idle_gpu_hours, 3),
                    "current_idle_cost_usd": round(context.current_idle_cost_usd, 2),
                    "prior_warning_count": context.history.warning_count,
                    "prior_auto_kill_count": context.history.auto_kill_count,
                    "prior_manual_kill_count": context.history.manual_kill_count,
                    "prior_resolved_without_kill_count": (
                        context.history.resolved_without_kill_count
                    ),
                    "prior_total_idle_gpu_hours": round(
                        context.history.total_idle_gpu_hours,
                        3,
                    ),
                    "prior_total_idle_cost_usd": round(
                        context.history.total_idle_cost_usd,
                        2,
                    ),
                    "telemetry_summary": evaluation.summary,
                    "recent_incidents": [
                        {
                            "job_id": incident.job_id,
                            "job_name": incident.job_name,
                            "resolution": incident.resolution,
                            "idle_gpu_hours": round(incident.idle_gpu_hours, 3),
                            "estimated_cost_usd": round(
                                incident.estimated_cost_usd,
                                2,
                            ),
                            "resolved_at": incident.resolved_at.isoformat(),
                        }
                        for incident in context.history.recent_incidents
                    ],
                },
                sort_keys=True,
            ),
        )
        output_text = response.output_text.strip()
        if not output_text:
            return None

        normalized = " ".join(output_text.split())
        return normalized[:320].rstrip()

    def _instructions(self) -> str:
        return (
            "You write short public warnings for a shared GPU cluster. "
            f"Tone: {self.settings.openai_warning_style}. "
            "Use only the facts provided. "
            "Write one or two sentences, maximum 320 characters. "
            "Reference repeat-offender history and estimated waste when the facts support it. "
            "Do not invent policies, deadlines, telemetry, or consequences. "
            "Do not mention buttons, timestamps, percentages, node names, or raw JSON keys. "
            "Do not use profanity, emojis, hashtags, or quotation marks. "
            "Return only the warning text."
        )
