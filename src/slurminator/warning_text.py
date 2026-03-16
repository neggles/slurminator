from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

from slurminator.config import Settings
from slurminator.db import JobWatch
from slurminator.models import JobEvaluation, UserHistorySnapshot, WarningContext

logger = logging.getLogger(__name__)

WarningSeverity = Literal["gentle", "pointed", "savage"]


@dataclass(frozen=True, slots=True)
class WarningTone:
    severity_band: WarningSeverity
    style_prompt: str
    fallback_intro: str
    reasons: list[str]


_PERSONA_STYLES: dict[str, dict[WarningSeverity, tuple[str, str]]] = {
    "snarky": {
        "gentle": (
            "dryly amused, playful, and lightly teasing",
            "Slurminator noticed your GPUs appear to be enjoying a funded sabbatical.",
        ),
        "pointed": (
            "wry, publicly disappointed, and sharper without becoming abusive",
            "Slurminator noticed your GPUs have been taking an expensive coffee break.",
        ),
        "savage": (
            "cutting, deadpan, and openly unimpressed while staying professional",
            "Slurminator noticed your GPUs are once again delivering a breathtaking performance of nothing.",
        ),
    },
    "bureaucratic": {
        "gentle": (
            "formal, restrained, and mildly disappointed",
            "Slurminator is issuing a courteous notice that this GPU allocation appears idle.",
        ),
        "pointed": (
            "formal, firm, and visibly less patient",
            "Slurminator is issuing a firmer notice that this GPU allocation is idling at shared expense.",
        ),
        "savage": (
            "clinical, icy, and bureaucratically devastating",
            "Slurminator is issuing a recurring idle-allocation notice because this GPU reservation is again producing no useful work.",
        ),
    },
    "bardic": {
        "gentle": (
            "dramatic, witty, and theatrical without being verbose",
            "Slurminator finds your GPUs upon the stage, yet hears not a single line delivered.",
        ),
        "pointed": (
            "theatrical, biting, and publicly embarrassed for the offender",
            "Slurminator finds your GPUs holding the stage in silence while the meter keeps perfect time.",
        ),
        "savage": (
            "grandly theatrical, mercilessly mocking, and still concise",
            "Slurminator finds your GPUs center stage once more, committing fully to the role of expensive scenery.",
        ),
    },
}


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

    def enrich_context(self, context: WarningContext) -> WarningContext:
        tone = self._select_tone(context)
        context.persona_preset = self.settings.warning_persona_preset
        context.severity_band = tone.severity_band
        context.severity_reasons = tone.reasons
        context.fallback_intro = tone.fallback_intro
        return context

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
            instructions=self._instructions(context),
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
                    "warning_persona_preset": context.persona_preset,
                    "warning_severity_band": context.severity_band,
                    "severity_reasons": context.severity_reasons,
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

    def _instructions(self, context: WarningContext) -> str:
        persona_preset = self.settings.warning_persona_preset
        style_prompt, _ = _PERSONA_STYLES[persona_preset][context.severity_band]
        reason_text = (
            "; ".join(context.severity_reasons)
            if context.severity_reasons
            else "first recorded incident"
        )
        return (
            "You write short public warnings for a shared GPU cluster. "
            f"Persona preset: {persona_preset}. "
            f"Severity band: {context.severity_band}. "
            f"Voice guidance: {style_prompt}. "
            f"Extra style guidance: {self.settings.openai_warning_style}. "
            f"Escalation reason: {reason_text}. "
            "Use only the facts provided. "
            "Write one or two sentences, maximum 320 characters. "
            "Reference repeat-offender history and estimated waste when the facts support it. "
            "Do not invent policies, deadlines, telemetry, or consequences. "
            "Do not mention buttons, timestamps, percentages, node names, or raw JSON keys. "
            "Do not use profanity, emojis, hashtags, or quotation marks. "
            "Return only the warning text."
        )

    def _select_tone(self, context: WarningContext) -> WarningTone:
        severity_band: WarningSeverity = "gentle"
        reasons = self._severity_reasons(context.history, context)

        if self._should_use_savage(context.history, context):
            severity_band = "savage"
        elif self._should_use_pointed(context.history, context):
            severity_band = "pointed"

        persona_preset = self.settings.warning_persona_preset
        style_prompt, fallback_intro = _PERSONA_STYLES[persona_preset][severity_band]
        return WarningTone(
            severity_band=severity_band,
            style_prompt=style_prompt,
            fallback_intro=fallback_intro,
            reasons=reasons,
        )

    def _should_use_pointed(
        self,
        history: UserHistorySnapshot,
        context: WarningContext,
    ) -> bool:
        return any(
            (
                history.warning_count >= self.settings.warning_pointed_after_warnings,
                history.manual_kill_count > 0,
                history.auto_kill_count > 0,
                context.current_idle_cost_usd >= self.settings.warning_pointed_after_cost_usd,
                history.total_idle_cost_usd >= self.settings.warning_pointed_after_cost_usd,
            )
        )

    def _should_use_savage(
        self,
        history: UserHistorySnapshot,
        context: WarningContext,
    ) -> bool:
        return any(
            (
                history.warning_count >= self.settings.warning_savage_after_warnings,
                history.auto_kill_count > 0,
                history.manual_kill_count >= 2,
                context.current_idle_cost_usd >= self.settings.warning_savage_after_cost_usd,
                history.total_idle_cost_usd >= self.settings.warning_savage_after_cost_usd,
            )
        )

    def _severity_reasons(
        self,
        history: UserHistorySnapshot,
        context: WarningContext,
    ) -> list[str]:
        reasons: list[str] = []
        if history.warning_count:
            reasons.append(f"{history.warning_count} prior warning(s)")
        if history.auto_kill_count:
            reasons.append(f"{history.auto_kill_count} prior auto-kill(s)")
        if history.manual_kill_count:
            reasons.append(f"{history.manual_kill_count} prior manual kill(s)")
        if context.current_idle_cost_usd >= self.settings.warning_pointed_after_cost_usd:
            reasons.append(
                f"current idle cost already at ${context.current_idle_cost_usd:.2f}"
            )
        if history.total_idle_cost_usd >= self.settings.warning_pointed_after_cost_usd:
            reasons.append(
                f"historical idle cost totals ${history.total_idle_cost_usd:.2f}"
            )
        if not reasons:
            reasons.append("first recorded incident")
        return reasons
