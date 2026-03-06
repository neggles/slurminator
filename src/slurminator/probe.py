from __future__ import annotations

import asyncio
import shlex

from slurminator.config import Settings
from slurminator.models import GpuSample, NodeSample


class GpuNodeProber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def probe_nodes(self, nodes: list[str]) -> list[NodeSample]:
        if not nodes:
            return []
        return await asyncio.gather(*(self._probe_node(node) for node in nodes))

    async def _probe_node(self, node_name: str) -> NodeSample:
        if self.settings.node_probe_mode == "local":
            command = ["bash", "-lc", self.settings.probe_command]
        else:
            remote_command = shlex.join(["bash", "-lc", self.settings.probe_command])
            command = [
                self.settings.ssh_binary,
                *self.settings.ssh_extra_args,
                node_name,
                remote_command,
            ]

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.settings.node_probe_timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return NodeSample(node_name=node_name, error="probe timed out")

        if process.returncode != 0:
            message = stderr.decode(errors="replace").strip() or "probe failed"
            return NodeSample(node_name=node_name, error=message)

        lines = [line.strip() for line in stdout.decode(errors="replace").splitlines() if line.strip()]
        if not lines:
            return NodeSample(node_name=node_name, error="probe returned no GPU data")

        gpus: list[GpuSample] = []
        for line in lines:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                return NodeSample(node_name=node_name, error=f"unexpected probe output: {line}")

            try:
                gpus.append(
                    GpuSample(
                        index=int(parts[0]),
                        utilization_pct=float(parts[1]),
                        memory_used_mb=float(parts[2]),
                        memory_total_mb=float(parts[3]),
                    )
                )
            except ValueError:
                return NodeSample(node_name=node_name, error=f"unexpected probe output: {line}")

        return NodeSample(node_name=node_name, gpus=gpus)
