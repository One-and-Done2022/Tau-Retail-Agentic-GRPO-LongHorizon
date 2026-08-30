"""Small, dependency-free helpers for response-relative tool-call spans."""
from __future__ import annotations

from typing import Any, Sequence


def build_tool_step_record(
    span: tuple[int, int],
    tool_names: Sequence[str],
    tool_rewards: Sequence[float],
    metadata: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        raise ValueError(f"Invalid assistant tool-call span: {span}")
    if not (len(tool_names) == len(tool_rewards) == len(metadata)):
        raise ValueError("Tool names, rewards, and metadata must have identical lengths")
    if not tool_names:
        raise ValueError("A tool-step record requires at least one executed tool")

    return {
        "start": start,
        "end": end,
        "reward": float(sum(float(value) for value in tool_rewards)),
        "tools": list(tool_names),
        "components": [item.get("cs_step_credit") for item in metadata],
    }
