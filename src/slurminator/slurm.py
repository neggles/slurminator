from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from slurminator.config import Settings
from slurminator.models import SlurmJob

logger = logging.getLogger(__name__)

_KEY_VALUE_PATTERN = re.compile(r"([A-Za-z0-9_]+)=")
_GPU_COUNT_PATTERNS = (
    re.compile(r"(?:gres/)?gpu(?::[^=,:]+)?=(\d+)"),
    re.compile(r"(?:gres:)?gpu(?::[^=,:]+)?:(\d+)"),
)
_NULL_VALUES = {"", "(null)", "N/A", "Unknown", "None"}


class SlurmCommandError(RuntimeError):
    pass


def parse_key_value_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    matches = list(_KEY_VALUE_PATTERN.finditer(line))
    for index, match in enumerate(matches):
        key = match.group(1)
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        fields[key] = line[value_start:value_end].strip()
    return fields


def extract_gpu_count(fields: dict[str, str]) -> int:
    max_gpu_count = 0
    combined_values = " ".join(value for value in fields.values() if value not in _NULL_VALUES)
    for pattern in _GPU_COUNT_PATTERNS:
        for match in pattern.finditer(combined_values):
            max_gpu_count = max(max_gpu_count, int(match.group(1)))
    return max_gpu_count


class SlurmClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _binary(self, name: str) -> str:
        if self.settings.slurm_bin_dir is None:
            return name
        return str(Path(self.settings.slurm_bin_dir) / name)

    async def _run(self, *args: str, timeout: int = 30) -> str:
        logger.debug("Running Slurm command: %s", " ".join(args))
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            msg = f"Timed out running {' '.join(args)}"
            raise SlurmCommandError(msg) from exc

        if process.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()
            msg = f"Slurm command failed ({process.returncode}): {' '.join(args)}"
            if stderr_text:
                msg = f"{msg}: {stderr_text}"
            raise SlurmCommandError(msg)

        return stdout.decode(errors="replace").strip()

    async def list_running_jobs(self) -> list[SlurmJob]:
        output = await self._run(self._binary("scontrol"), "show", "job", "-o")
        jobs: list[SlurmJob] = []

        for line in output.splitlines():
            fields = parse_key_value_line(line)
            if fields.get("JobState") != "RUNNING":
                continue

            node_list = fields.get("NodeList", "")
            if node_list in _NULL_VALUES:
                continue

            gpu_count = extract_gpu_count(fields)
            job_id = fields.get("JobId")
            user_name = fields.get("UserId", "").split("(")[0]
            job_name = fields.get("JobName", job_id or "unknown")
            if not job_id or not user_name:
                continue

            jobs.append(
                SlurmJob(
                    job_id=job_id,
                    user_name=user_name,
                    job_name=job_name,
                    state=fields.get("JobState", ""),
                    node_list=node_list,
                    run_time=fields.get("RunTime"),
                    command=fields.get("Command"),
                    work_dir=fields.get("WorkDir"),
                    gpu_count=gpu_count,
                    raw_fields=fields,
                )
            )

        logger.info("Found %s running Slurm jobs (%s GPU jobs)", len(jobs), sum(job.gpu_count > 0 for job in jobs))
        return jobs

    async def expand_hostlist(self, node_list: str) -> list[str]:
        if node_list in _NULL_VALUES:
            return []
        output = await self._run(self._binary("scontrol"), "show", "hostnames", node_list)
        return [line.strip() for line in output.splitlines() if line.strip()]

    async def cancel_job(self, job_id: str) -> None:
        await self._run(self._binary("scancel"), job_id, timeout=15)
