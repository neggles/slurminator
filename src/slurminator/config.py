from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SLURMINATOR_",
        extra="ignore",
        validate_default=True,
    )

    database_url: str = "sqlite+aiosqlite:///data/slurminator.db"
    notifier: Literal["stdout", "discord"] = "stdout"
    only_gpu_jobs: bool = True
    dry_run: bool = False
    log_level: str = "INFO"

    poll_interval_seconds: int = 60
    idle_warning_after_seconds: int = 1800
    idle_kill_grace_seconds: int = 1800
    gpu_utilization_threshold_percent: float = 5.0
    gpu_memory_threshold_percent: float = 10.0
    gpu_hourly_cost_usd: float = 0.0
    warning_message_mode: Literal["static", "openai"] = "static"

    node_probe_mode: Literal["ssh", "local"] = "ssh"
    node_probe_timeout_seconds: int = 20
    probe_command: str = (
        "nvidia-smi "
        "--query-gpu=index,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits"
    )
    ssh_binary: str = "ssh"
    ssh_extra_args: list[str] = Field(
        default_factory=lambda: ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
    )

    slurm_bin_dir: Path | None = None

    discord_token: SecretStr | None = None
    discord_channel_id: int | None = None
    discord_admin_role_ids: list[int] = Field(default_factory=list)

    user_map_path: Path | None = None

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"
    openai_timeout_seconds: int = 20
    openai_warning_style: str = "dryly funny, concise, and not profane"

    @field_validator("ssh_extra_args", mode="before")
    @classmethod
    def _parse_string_list(cls, value: object) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @field_validator("discord_admin_role_ids", mode="before")
    @classmethod
    def _parse_int_list(cls, value: object) -> list[int]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return [int(item) for item in value]
        return [int(item.strip()) for item in str(value).split(",") if item.strip()]

    @field_validator("user_map_path", "slurm_bin_dir", mode="before")
    @classmethod
    def _empty_path_to_none(cls, value: object) -> object:
        if value in (None, ""):
            return None
        return value

    def require_discord_token(self) -> str:
        if self.discord_token is None:
            msg = "SLURMINATOR_DISCORD_TOKEN is required when notifier=discord"
            raise RuntimeError(msg)
        return self.discord_token.get_secret_value()

    def openai_api_key_value(self) -> str | None:
        if self.openai_api_key is None:
            return None
        return self.openai_api_key.get_secret_value()
