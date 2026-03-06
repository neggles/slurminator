# slurminator

a bot that watches slurm jobs and semipublicly shames misbehaving users in discord/slack

Right now the scaffold supports the full Discord path and leaves the notifier layer open for a future Slack adapter.

## What it does

- polls running Slurm jobs
- filters to GPU jobs by default
- probes allocated nodes with `nvidia-smi`
- tracks how long a job has stayed below configurable GPU util and GPU memory thresholds
- posts a public Discord warning with a kill button after a configurable idle window
- auto-cancels the job after a second configurable grace period
- stores watch state in SQLite so restarts do not lose outstanding warnings

## Assumptions in this first pass

- the job effectively owns the GPUs on the allocated node(s)
- the bot can reach compute nodes either directly over `ssh` or by running locally on the node
- Discord user IDs are mapped to Slurm usernames through a small JSON file for button authorization
- if telemetry is missing from any allocated node, the bot backs off instead of shaming someone on incomplete data

## Configuration

All runtime settings are environment variables with the `SLURMINATOR_` prefix.

Important ones:

- `SLURMINATOR_NOTIFIER=discord` to enable the Discord bot
- `SLURMINATOR_DISCORD_TOKEN` and `SLURMINATOR_DISCORD_CHANNEL_ID`
- `SLURMINATOR_IDLE_WARNING_AFTER_SECONDS`
- `SLURMINATOR_IDLE_KILL_GRACE_SECONDS`
- `SLURMINATOR_GPU_UTILIZATION_THRESHOLD_PERCENT`
- `SLURMINATOR_GPU_MEMORY_THRESHOLD_PERCENT`
- `SLURMINATOR_NODE_PROBE_MODE=ssh` or `local`
- `SLURMINATOR_USER_MAP_PATH=$PWD/data/user-map.example.json`

`data/user-map.example.json` shows the expected username-to-Discord-ID format.

## Commands

Inspect the cluster once:

```bash
uv run slurminator once
```

Run the continuous monitor:

```bash
uv run slurminator run
```

Cancel a tracked job from the CLI:

```bash
uv run slurminator kill 123456
```
