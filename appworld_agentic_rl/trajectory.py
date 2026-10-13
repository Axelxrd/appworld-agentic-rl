"""Trajectory records for AppWorld agent-tool rollouts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "appworld-agentic-rl-trajectory-v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TrajectoryStep:
    """One agent-selected tool step visible to SFT/RL data builders."""

    step_index: int
    tool_name: str
    arguments: dict[str, Any]
    observation: Any
    error: str | None = None
    state_change: bool = False
    terminated: bool = False
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrajectoryRecord:
    """Serializable rollout container produced by AppWorldAdapter."""

    task_id: str
    split: str | None
    experiment_name: str
    initial_observation: dict[str, Any]
    steps: list[TrajectoryStep] = field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    terminal_reward: float | None = None
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)
        if step.terminated:
            self.finished_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["steps"] = [step.to_dict() for step in self.steps]
        return output
