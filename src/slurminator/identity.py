from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IdentityRecord:
    discord_user_ids: tuple[int, ...] = ()


class IdentityDirectory:
    def __init__(self, records: dict[str, IdentityRecord] | None = None) -> None:
        self._records = records or {}

    @classmethod
    def from_path(cls, path: Path | None) -> "IdentityDirectory":
        if path is None:
            return cls()
        if not path.exists():
            return cls()

        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in user map {path}: {exc}"
            raise ValueError(msg) from exc

        records: dict[str, IdentityRecord] = {}
        for slurm_user, raw_value in payload.items():
            discord_user_ids: list[int]
            if isinstance(raw_value, list):
                discord_user_ids = [int(item) for item in raw_value]
            elif isinstance(raw_value, dict):
                raw_ids = raw_value.get("discord_user_ids", [])
                if not isinstance(raw_ids, list):
                    msg = f"discord_user_ids for {slurm_user} must be a list"
                    raise ValueError(msg)
                discord_user_ids = [int(item) for item in raw_ids]
            else:
                msg = f"User map entry for {slurm_user} must be a list or object"
                raise ValueError(msg)

            records[str(slurm_user)] = IdentityRecord(
                discord_user_ids=tuple(discord_user_ids)
            )

        return cls(records)

    def discord_user_ids(self, slurm_user: str) -> set[int]:
        record = self._records.get(slurm_user)
        if record is None:
            return set()
        return set(record.discord_user_ids)

    def discord_mentions(self, slurm_user: str) -> list[str]:
        return [f"<@{user_id}>" for user_id in sorted(self.discord_user_ids(slurm_user))]
