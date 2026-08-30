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
    "prompt_length/clip_ratio",
    "response/aborted_ratio",
    "training/global_step",
)
CHECKPOINT_STEPS = (4, 8)
SYSTEM_ERROR_PATTERNS = (
    "APIConnectionError",
    "APITimeoutError",
    "env.step(RESPOND) failed",
    "EngineDeadError",
    "torch.OutOfMemoryError",
    "CUDA out of memory",
)


def parse_steps(log_path: Path) -> dict[int, dict[str, float]]:
    clean = ANSI_RE.sub("", log_path.read_text(errors="replace")).replace("\r", "\n")
    steps: dict[int, dict[str, float]] = {}
    for line in clean.splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        metrics: dict[str, float] = {}
        for item in match.group(2).split(" - "):
            if ":" not in item:
                continue
            key, value = item.rsplit(":", 1)
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
        steps[int(match.group(1))] = metrics
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-name", default="gate_result.json")
    args = parser.parse_args()
    run_dir = args.run_dir
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".json"):
        raise SystemExit("output-name must be a JSON basename")
    output = run_dir / args.output_name
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing gate result: {output}")

    failures: list[str] = []
    log_path = run_dir / "training.log"
    steps = parse_steps(log_path) if log_path.is_file() else {}
    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    for pattern in SYSTEM_ERROR_PATTERNS:
        count = log_text.count(pattern)
        if count:
            failures.append(f"training log contains {count} system-error marker(s): {pattern}")
    expected_steps = list(range(1, 9))
    if sorted(steps) != expected_steps:
        failures.append(f"expected metric steps {expected_steps}, got {sorted(steps)}")

    selected_metrics: dict[str, dict[str, float]] = {}
    for step in expected_steps:
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
        if metrics["prompt_length/clip_ratio"] != 0:
            failures.append(f"step {step} prompt was clipped")
        if metrics["response/aborted_ratio"] != 0:
            failures.append(f"step {step} has aborted responses")
        selected_metrics[str(step)] = {key: metrics[key] for key in REQUIRED_METRICS}

    rollout_files = [run_dir / "rollouts" / f"{step}.jsonl" for step in expected_steps]
    for path in rollout_files:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty rollout: {path}")

    checkpoint_files: dict[str, int] = {}
    for checkpoint_step in CHECKPOINT_STEPS:
        actor_dir = run_dir / f"checkpoints/global_step_{checkpoint_step}/actor"
        required = [actor_dir / "fsdp_config.json", actor_dir / "huggingface/config.json"]
        for rank in (0, 1):
            required.extend(
                [
                    actor_dir / f"model_world_size_2_rank_{rank}.pt",
                    actor_dir / f"optim_world_size_2_rank_{rank}.pt",
                    actor_dir / f"extra_state_world_size_2_rank_{rank}.pt",
                ]
            )
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing or empty checkpoint file: {path}")
            elif path.is_file():
                checkpoint_files[str(path)] = path.stat().st_size

    latest = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    if not latest.is_file() or latest.read_text().strip() != "8":
        failures.append("latest checkpoint pointer is not 8")

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if not manifest:
        failures.append("missing run_manifest.json")

    result = {
        "passed": not failures,
        "label": "confirmatory-fixed-budget-engineering-study",
        "method": manifest.get("method"),
        "run_id": manifest.get("run_id"),
        "steps": selected_metrics,
        "rollout_files": [str(path) for path in rollout_files],
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_files": checkpoint_files,
        "failures": failures,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
