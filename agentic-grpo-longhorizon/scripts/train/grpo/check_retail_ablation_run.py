#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+) - (.*)")
REQUIRED_METRICS = (
    "actor/pg_loss",
    "actor/grad_norm",
    "critic/rewards/mean",
    "critic/rewards/min",
    "critic/rewards/max",
    "critic/advantages/min",
    "critic/advantages/max",
    "critic/score/all_zero_frac",
    "critic/score/all_one_frac",
    "training/global_step",
)


def parse_steps(log_path: Path) -> dict[int, dict[str, float]]:
    clean = ANSI_RE.sub("", log_path.read_text(errors="replace")).replace("\r", "\n")
    steps: dict[int, dict[str, float]] = {}
    for line in clean.splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        metrics: dict[str, float] = {}
        for item in match.group(2).split(" - "):
            if ":" not in item:
                continue
            key, value = item.rsplit(":", 1)
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
        steps[step] = metrics
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    output = run_dir / "gate_result.json"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing gate result: {output}")

    failures: list[str] = []
    log_path = run_dir / "training.log"
    if not log_path.is_file():
        failures.append("missing training.log")
        steps = {}
    else:
        steps = parse_steps(log_path)
    if sorted(steps) != [1, 2, 3]:
        failures.append(f"expected metric steps [1, 2, 3], got {sorted(steps)}")

    selected_metrics: dict[str, dict[str, float]] = {}
    for step in (1, 2, 3):
        metrics = steps.get(step, {})
        missing = [key for key in REQUIRED_METRICS if key not in metrics]
        if missing:
            failures.append(f"step {step} missing metrics: {missing}")
            continue
        invalid = [key for key in REQUIRED_METRICS if not math.isfinite(metrics[key])]
        if invalid:
            failures.append(f"step {step} non-finite metrics: {invalid}")
        if metrics["actor/grad_norm"] <= 0:
            failures.append(f"step {step} grad_norm is not positive")
        if int(metrics["training/global_step"]) != step:
            failures.append(f"step {step} global_step mismatch")
        selected_metrics[str(step)] = {key: metrics[key] for key in REQUIRED_METRICS}

    rollout_files = [run_dir / "rollouts" / f"{step}.jsonl" for step in (1, 2, 3)]
    for path in rollout_files:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty rollout: {path}")

    actor_dir = run_dir / "checkpoints/global_step_3/actor"
    checkpoint_files = [actor_dir / "fsdp_config.json", actor_dir / "huggingface/config.json"]
    for rank in (0, 1):
        checkpoint_files.extend(
            [
                actor_dir / f"model_world_size_2_rank_{rank}.pt",
                actor_dir / f"optim_world_size_2_rank_{rank}.pt",
                actor_dir / f"extra_state_world_size_2_rank_{rank}.pt",
            ]
        )
    for path in checkpoint_files:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty checkpoint file: {path}")
    latest = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    if not latest.is_file() or latest.read_text().strip() != "3":
        failures.append("latest checkpoint pointer is not 3")

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if not manifest:
        failures.append("missing run_manifest.json")

    result = {
        "passed": not failures,
        "label": "fixed-budget-engineering-pilot",
        "method": manifest.get("method"),
        "run_id": manifest.get("run_id"),
        "steps": selected_metrics,
        "rollout_files": [str(path) for path in rollout_files],
        "checkpoint_actor_dir": str(actor_dir),
        "checkpoint_files": {str(path): path.stat().st_size for path in checkpoint_files if path.is_file()},
        "failures": failures,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
