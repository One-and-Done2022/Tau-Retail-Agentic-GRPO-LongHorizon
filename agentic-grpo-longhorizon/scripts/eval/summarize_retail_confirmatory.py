#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


METHODS = ("vanilla", "state", "constraint", "cs")
CANDIDATES = (
    "sft",
    "vanilla_step_4",
    "vanilla_step_8",
    "state_step_4",
    "state_step_8",
    "constraint_step_4",
    "constraint_step_8",
    "cs_step_4",
    "cs_step_8",
)
SPLITS = ("seen", "unseen_dev")
EVAL_METRICS = (
    "pass_hat_1",
    "pass_at_1",
    "pass_hat_4",
    "error_rate",
    "constraint_violation_rate",
    "high_risk_tool_error_rate",
    "tool_error_rate",
    "error_recovery_rate",
    "avg_turns",
    "avg_tool_calls",
    "avg_reasoning_tokens_per_turn",
    "max_reasoning_tokens_single_turn",
    "avg_latency_seconds",
)
TRAIN_METRICS = (
    "actor/pg_loss",
    "actor/grad_norm",
    "critic/rewards/mean",
    "critic/rewards/min",
    "critic/rewards/max",
    "critic/advantages/min",
    "critic/advantages/max",
    "critic/score/std",
    "critic/score/all_zero_frac",
    "critic/score/all_one_frac",
    "prompt_length/clip_ratio",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "response/aborted_ratio",
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+) - (.*)")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing required JSON: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"Missing required hash input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"Expected finite numeric {label}, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise SystemExit(f"Expected finite numeric {label}, got {value!r}")
    return result


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def task_bootstrap_ci(
    task_values: dict[int, float], *, seed: int, replicates: int
) -> dict[str, float | int]:
    if not task_values:
        raise SystemExit("Task bootstrap requires at least one task")
    if replicates < 100:
        raise SystemExit("Task bootstrap requires at least 100 replicates")
    values = [task_values[task_id] for task_id in sorted(task_values)]
    rng = random.Random(seed)
    samples = [
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(replicates)
    ]
    return {
        "estimate": sum(values) / len(values),
        "lower_95": quantile(samples, 0.025),
        "upper_95": quantile(samples, 0.975),
        "unit": "task",
        "num_tasks": len(values),
        "replicates": replicates,
        "seed": seed,
    }


def classify_error(error: object) -> str | None:
    if not error:
        return None
    normalized = str(error).lower()
    if "loop detected" in normalized and "tool call" in normalized:
        return "repeated_tool_call_loop"
    if "context_length_exceeded" in normalized or "maximum context length" in normalized:
        return "context_length_exceeded"
    if "out of memory" in normalized or "cuda oom" in normalized:
        return "out_of_memory"
    return "other"


def parse_training_steps(log_path: Path) -> dict[int, dict[str, float]]:
    if not log_path.is_file():
        raise SystemExit(f"Missing training log: {log_path}")
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
            key, raw_value = item.rsplit(":", 1)
            try:
                metrics[key] = float(raw_value)
            except ValueError:
                continue
        steps[int(match.group(1))] = metrics
    return steps


def summarize_training(
    project_dir: Path,
    experiment_root: Path,
    manifest_path: Path,
    method: str,
    run_id: str,
) -> dict:
    run_dir = experiment_root / "train" / method / run_id
    if read_json(run_dir / "frozen_manifest.json") != read_json(manifest_path):
        raise SystemExit(f"Frozen training manifest mismatch: {run_dir}")
    gate_path = run_dir / "gate_result.json"
    gate = read_json(gate_path)
    if not gate.get("passed") or gate.get("method") != method or gate.get("run_id") != run_id:
        raise SystemExit(f"Training gate did not pass for {method}: {gate_path}")
    if sorted(gate.get("steps", {})) != [str(step) for step in range(1, 9)]:
        raise SystemExit(f"Training gate does not contain steps 1-8: {gate_path}")

    parsed = parse_training_steps(run_dir / "training.log")
    if sorted(parsed) != list(range(1, 9)):
        raise SystemExit(f"Training log does not contain exactly steps 1-8 for {method}")
    curve: list[dict] = []
    for step in range(1, 9):
        missing = [key for key in TRAIN_METRICS if key not in parsed[step]]
        if missing:
            raise SystemExit(f"Training step {method}/{step} is missing metrics: {missing}")
        metrics = {key: finite(parsed[step][key], f"{method}/{step}:{key}") for key in TRAIN_METRICS}
        curve.append({"step": step, "metrics": metrics})

    checkpoint_evidence = {}
    for step in (4, 8):
        merged_dir = run_dir / f"hf_step_{step}"
        reload_log = run_dir / f"hf_reload_step_{step}.log"
        has_weights = (merged_dir / "model.safetensors").is_file() or (
            merged_dir / "model.safetensors.index.json"
        ).is_file()
        if not has_weights or not (merged_dir / "config.json").is_file():
            raise SystemExit(f"Merged checkpoint is incomplete: {merged_dir}")
        if not reload_log.is_file() or "HF_RELOAD" not in reload_log.read_text(errors="replace"):
            raise SystemExit(f"HF reload evidence is incomplete: {reload_log}")
        checkpoint_evidence[str(step)] = {
            "merged_checkpoint": str(merged_dir.relative_to(project_dir)),
            "hf_reload_log": str(reload_log.relative_to(project_dir)),
        }

    log_text = (run_dir / "training.log").read_text(errors="replace").lower()
    metric_rows = [point["metrics"] for point in curve]
    saturation = {
        "all_zero_steps": sum(row["critic/score/all_zero_frac"] == 1.0 for row in metric_rows),
        "all_one_steps": sum(row["critic/score/all_one_frac"] == 1.0 for row in metric_rows),
        "mixed_reward_steps": sum(
            row["critic/score/all_zero_frac"] < 1.0 and row["critic/score/all_one_frac"] < 1.0
            for row in metric_rows
        ),
        "reward_mean": sum(row["critic/rewards/mean"] for row in metric_rows) / len(metric_rows),
        "pg_loss_abs_mean": sum(abs(row["actor/pg_loss"]) for row in metric_rows) / len(metric_rows),
        "grad_norm_mean": sum(row["actor/grad_norm"] for row in metric_rows) / len(metric_rows),
        "grad_norm_min": min(row["actor/grad_norm"] for row in metric_rows),
        "response_clip_max": max(row["response_length/clip_ratio"] for row in metric_rows),
        "response_clipped_steps": [
            point["step"] for point in curve if point["metrics"]["response_length/clip_ratio"] > 0
        ],
        "prompt_clip_max": max(row["prompt_length/clip_ratio"] for row in metric_rows),
        "aborted_ratio_max": max(row["response/aborted_ratio"] for row in metric_rows),
    }
    return {
        "source": str(gate_path.relative_to(project_dir)),
        "passed": True,
        "curve": curve,
        "saturation": saturation,
        "checkpoint_evidence": checkpoint_evidence,
        "log_anomalies": {
            "context_error_mentions": log_text.count("context_length_exceeded")
            + log_text.count("maximum context length"),
            "oom_mentions": log_text.count("out of memory") + log_text.count("cuda oom"),
            "dataloader_worker_killed_mentions": len(
                re.findall(r"dataloader worker .*? killed", log_text)
            ),
        },
    }


def expected_eval_identity(manifest: dict, split: str) -> tuple[list[int], int, int, int]:
    evaluation = manifest["evaluation"]
    split_manifest_key = "seen" if split == "seen" else "unseen"
    task_ids = [int(value) for value in evaluation[split_manifest_key]["task_ids"]]
    return (
        task_ids,
        int(evaluation["samples_per_task"]),
        int(evaluation["policy_seed_base"]),
        int(evaluation["user_seed_base"]),
    )


def summarize_evaluation(
    report: dict,
    source: Path,
    manifest: dict,
    split: str,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict:
    task_ids, samples_per_task, policy_seed_base, user_seed_base = expected_eval_identity(
        manifest, split
    )
    if report.get("env_name") != "retail":
        raise SystemExit(f"Expected retail report: {source}")
    if int(report.get("num_tasks", -1)) != len(task_ids):
        raise SystemExit(f"Task count mismatch: {source}")
    if int(report.get("num_samples_per_task", -1)) != samples_per_task:
        raise SystemExit(f"Sample count mismatch: {source}")
    if report.get("policy_seed_base") != policy_seed_base:
        raise SystemExit(f"Policy seed base mismatch: {source}")
    if report.get("user_seed_base") != user_seed_base:
        raise SystemExit(f"User seed base mismatch: {source}")
    if report.get("seed_rule") != "base + task_id * 1000 + sample_id":
        raise SystemExit(f"Seed rule mismatch: {source}")

    task_rows = report.get("per_task_results", [])
    if [int(row.get("task_id", -1)) for row in task_rows] != task_ids:
        raise SystemExit(f"Task ID/order mismatch: {source}")

    pairs: dict[tuple[int, int], dict] = {}
    task_success: dict[int, float] = {}
    errors: Counter[str] = Counter()
    contaminated = 0
    constraint_violations: Counter[str] = Counter()
    for task_row in task_rows:
        task_id = int(task_row["task_id"])
        trajectories = task_row.get("trajectories", [])
        if len(trajectories) != samples_per_task:
            raise SystemExit(f"Trajectory count mismatch for task {task_id}: {source}")
        sample_ids = [int(trajectory.get("sample_id", -1)) for trajectory in trajectories]
        if sample_ids != list(range(samples_per_task)):
            raise SystemExit(f"Sample identity/order mismatch for task {task_id}: {source}")
        successes = []
        for trajectory in trajectories:
            sample_id = int(trajectory["sample_id"])
            expected_policy_seed = policy_seed_base + task_id * 1000 + sample_id
            expected_user_seed = user_seed_base + task_id * 1000 + sample_id
            if trajectory.get("policy_seed") != expected_policy_seed:
                raise SystemExit(f"Policy seed mismatch for {(task_id, sample_id)}: {source}")
            if trajectory.get("user_seed") != expected_user_seed:
                raise SystemExit(f"User seed mismatch for {(task_id, sample_id)}: {source}")
            key = (task_id, sample_id)
            if key in pairs:
                raise SystemExit(f"Duplicate trajectory identity {key}: {source}")
            error_category = classify_error(trajectory.get("error"))
            if error_category:
                errors[error_category] += 1
            is_contaminated = trajectory.get("was_contaminated_from_turn") is not None
            contaminated += is_contaminated
            constraint_violations.update(trajectory.get("constraint_violations") or {})
            success = bool(trajectory.get("success"))
            successes.append(float(success))
            tokens = trajectory.get("per_turn_assistant_content_tokens") or []
            pairs[key] = {
                "task_id": task_id,
                "sample_id": sample_id,
                "policy_seed": expected_policy_seed,
                "user_seed": expected_user_seed,
                "success": success,
                "reward": finite(trajectory.get("reward"), f"{source}:{key}:reward"),
                "error_class": error_category,
                "contaminated": is_contaminated,
                "constraint_violation_count": int(trajectory.get("constraint_violation_count", 0)),
                "high_risk_tool_calls": int(trajectory.get("high_risk_tool_calls", 0)),
                "high_risk_tool_errors": int(trajectory.get("high_risk_tool_errors", 0)),
                "tool_error_count": int(trajectory.get("tool_error_count", 0)),
                "recovered_tool_errors": int(trajectory.get("recovered_tool_errors", 0)),
                "turns": int(trajectory.get("num_turns", 0)),
                "tool_calls": int(trajectory.get("num_tool_calls", 0)),
                "assistant_content_tokens": sum(int(value) for value in tokens),
                "latency_seconds": finite(
                    trajectory.get("latency_seconds"), f"{source}:{key}:latency_seconds"
                ),
            }
        task_success[task_id] = sum(successes) / len(successes)

    metrics = {key: finite(report.get(key), f"{source}:{key}") for key in EVAL_METRICS}
    if not math.isclose(
        metrics["pass_hat_1"], sum(task_success.values()) / len(task_success), abs_tol=1e-9
    ):
        raise SystemExit(f"pass^1 does not match trajectory outcomes: {source}")
    system_errors = sum(
        count for category, count in errors.items() if category != "repeated_tool_call_loop"
    )
    return {
        "source": str(source),
        "trajectory_count": len(pairs),
        "success_count": sum(row["success"] for row in pairs.values()),
        "metrics": metrics,
        "pass_hat_1_task_bootstrap_95ci": task_bootstrap_ci(
            task_success, seed=bootstrap_seed, replicates=bootstrap_replicates
        ),
        "task_success": {str(key): value for key, value in sorted(task_success.items())},
        "pairs": {f"{task_id}:{sample_id}": row for (task_id, sample_id), row in sorted(pairs.items())},
        "error_count": sum(errors.values()),
        "error_classes": dict(sorted(errors.items())),
        "system_error_count": system_errors,
        "contaminated_trajectory_count": contaminated,
        "contamination_rate": contaminated / len(pairs),
        "constraint_violations_by_rule": dict(sorted(constraint_violations.items())),
    }


def paired_delta(
    parent: dict,
    candidate: dict,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict:
    parent_pairs = parent["pairs"]
    candidate_pairs = candidate["pairs"]
    if set(parent_pairs) != set(candidate_pairs):
        raise SystemExit("Paired comparison has mismatched task/sample identities")
    pair_rows = []
    task_values: dict[int, list[float]] = {}
    for key in sorted(parent_pairs, key=lambda value: tuple(map(int, value.split(":")))):
        left = parent_pairs[key]
        right = candidate_pairs[key]
        if left["policy_seed"] != right["policy_seed"] or left["user_seed"] != right["user_seed"]:
            raise SystemExit(f"Paired comparison has mismatched seeds: {key}")
        value = float(right["success"]) - float(left["success"])
        task_id = int(right["task_id"])
        task_values.setdefault(task_id, []).append(value)
        pair_rows.append(
            {
                "task_id": task_id,
                "sample_id": int(right["sample_id"]),
                "success_delta": value,
            }
        )
    task_deltas = {task_id: sum(values) / len(values) for task_id, values in task_values.items()}
    ci = task_bootstrap_ci(task_deltas, seed=bootstrap_seed, replicates=bootstrap_replicates)
    return {
        "success_delta": ci["estimate"],
        "task_bootstrap_95ci": ci,
        "task_deltas": {str(key): value for key, value in sorted(task_deltas.items())},
        "pair_deltas": pair_rows,
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_report(summary: dict) -> str:
    lines = [
        f"# {summary['label']}",
        "",
        "## 结论",
        "",
        "本报告完成同一 SFT 起点、固定 8-update 预算下的 Vanilla、State、Constraint 与 CS-GRPO "
        "多 checkpoint 工程对照。结果是本地确认性工程证据，不是官方 τ-bench 性能或泛化结论。",
        "",
        "## 冻结协议",
        "",
        "- 训练：retail train tasks 0–7，四方法各 8 updates，group size 4，checkpoint 4/8。",
        "- 评测：seen=train 0–7；unseen=全部 20 个 dev tasks；每任务 4 个配对 policy/user seeds。",
        "- 所有方法保持相同模型、用户模拟器、temperature、top_p、turn/context 与成功判定。",
        "- pilot test 结果未用于选择本轮 dev task；报告不包含 `raw_messages` 或隐藏目标。",
        "",
        "## Checkpoint curves",
        "",
        "| 方法/checkpoint | split | pass^1 [task-bootstrap 95% CI] | pass@1 | pass^4 | error | violation | high-risk error | tool error | recovery | turns | tools | tokens/turn | latency(s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATES:
        for split in SPLITS:
            result = summary["evaluation"][candidate][split]
            metrics = result["metrics"]
            ci = result["pass_hat_1_task_bootstrap_95ci"]
            lines.append(
                f"| {candidate} | {split} | {fmt(metrics['pass_hat_1'])} "
                f"[{fmt(ci['lower_95'])}, {fmt(ci['upper_95'])}] | "
                f"{fmt(metrics['pass_at_1'])} | {fmt(metrics['pass_hat_4'])} | "
                f"{fmt(metrics['error_rate'])} | "
                f"{fmt(metrics['constraint_violation_rate'])} | "
                f"{fmt(metrics['high_risk_tool_error_rate'])} | "
                f"{fmt(metrics['tool_error_rate'])} | {fmt(metrics['error_recovery_rate'])} | "
                f"{metrics['avg_turns']:.2f} | "
                f"{metrics['avg_tool_calls']:.2f} | {metrics['avg_reasoning_tokens_per_turn']:.2f} | "
                f"{metrics['avg_latency_seconds']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## 与 SFT 的配对 success delta",
            "",
            "| checkpoint | split | Δ pass^1 [task-bootstrap 95% CI] |",
            "| --- | --- | ---: |",
        ]
    )
    for candidate in CANDIDATES[1:]:
        for split in SPLITS:
            delta = summary["paired_against_sft"][candidate][split]
            ci = delta["task_bootstrap_95ci"]
            lines.append(
                f"| {candidate} | {split} | {delta['success_delta']:+.3f} "
                f"[{ci['lower_95']:+.3f}, {ci['upper_95']:+.3f}] |"
            )

    lines.extend(
        [
            "",
            "## Checkpoint 8 − checkpoint 4 配对 success delta",
            "",
            "| 方法 | split | Δ pass^1 [task-bootstrap 95% CI] |",
            "| --- | --- | ---: |",
        ]
    )
    for method in METHODS:
        for split in SPLITS:
            delta = summary["checkpoint_step_8_minus_step_4"][method][split]
            ci = delta["task_bootstrap_95ci"]
            lines.append(
                f"| {method} | {split} | {delta['success_delta']:+.3f} "
                f"[{ci['lower_95']:+.3f}, {ci['upper_95']:+.3f}] |"
            )

    lines.extend(
        [
            "",
            "## 训练饱和与数值门禁",
            "",
            "| 方法 | gate | reward mean | all-zero steps | all-one steps | mixed steps | |pg loss| mean | grad mean/min | response clip max (steps) | prompt clip max | aborted max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        result = summary["training"][method]
        sat = result["saturation"]
        clipped_steps = ",".join(map(str, sat["response_clipped_steps"])) or "—"
        lines.append(
            f"| {method} | pass | {fmt(sat['reward_mean'])} | {sat['all_zero_steps']}/8 | "
            f"{sat['all_one_steps']}/8 | {sat['mixed_reward_steps']}/8 | "
            f"{sat['pg_loss_abs_mean']:.3e} | "
            f"{sat['grad_norm_mean']:.3e}/{sat['grad_norm_min']:.3e} | "
            f"{fmt(sat['response_clip_max'])} ({clipped_steps}) | "
            f"{fmt(sat['prompt_clip_max'])} | {fmt(sat['aborted_ratio_max'])} |"
        )

    totals = summary["totals"]
    error_classes = ", ".join(
        f"{name}={count}" for name, count in totals["error_classes"].items()
    ) or "none"
    training_anomalies = "; ".join(
        f"{method}: "
        f"context={result['log_anomalies']['context_error_mentions']}, "
        f"oom={result['log_anomalies']['oom_mentions']}, "
        f"dataloader-killed={result['log_anomalies']['dataloader_worker_killed_mentions']}"
        for method, result in summary["training"].items()
    )
    lines.extend(
        [
            "",
            "## 错误、污染与边界",
            "",
            f"- 共 {totals['trajectory_count']} 条轨迹；success={totals['success_count']}；"
            f"error={totals['error_count']}；system error={totals['system_error_count']}。",
            f"- 评测错误分类：{error_classes}。",
            f"- 污染轨迹 {totals['contaminated_trajectory_count']}/{totals['trajectory_count']} "
            f"({pct(totals['contamination_rate'])})；错误分类见 `summary.json`。",
            f"- 训练日志异常计数：{training_anomalies}。训练完成后的清理期异常与训练期硬错误分开保留。",
            "- task/sample 级配对差值、每任务差值、checkpoint 4→8 差值、全部 8-step 训练曲线"
            "和约束规则计数保存在 `summary.json`；未复制原始对话。",
            "- 置信区间以 task 为重采样单位，反映本固定 task suite 的不确定性；不应外推为官方 benchmark 置信区间。",
            "",
            "## 复现入口",
            "",
            "```bash",
            summary["reproduce_command"],
            "```",
            "",
            "输出采用 append-only 策略；目标 `summary.json` 或 `report.md` 已存在时拒绝覆盖。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the frozen retail confirmatory matrix")
    parser.add_argument("--run-id", default="run_001")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments/retail_confirmatory"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/eval/retail/confirmatory_manifest.json"),
    )
    parser.add_argument(
        "--seen-config",
        type=Path,
        default=Path("configs/eval/retail/confirmatory_seen.yaml"),
    )
    parser.add_argument(
        "--unseen-config",
        type=Path,
        default=Path("configs/eval/retail/confirmatory_unseen_dev.yaml"),
    )
    parser.add_argument(
        "--eval-matrix-script",
        type=Path,
        default=Path("scripts/eval/run_retail_confirmatory_eval_matrix.sh"),
    )
    parser.add_argument("--eval-orchestration-name")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()

    if not re.fullmatch(r"run_[0-9]{3}", args.run_id):
        raise SystemExit(f"run_id must match run_NNN, got: {args.run_id}")
    orchestration_name = args.eval_orchestration_name or args.run_id
    if not re.fullmatch(r"[A-Za-z0-9._-]+", orchestration_name):
        raise SystemExit(f"Invalid eval orchestration name: {orchestration_name}")
    project_dir = args.project_dir.resolve()
    experiment_root = (project_dir / args.experiment_root).resolve()
    manifest_path = (project_dir / args.manifest).resolve()
    manifest = read_json(manifest_path)
    if not str(manifest.get("experiment_label", "")).startswith("retail-confirmatory-"):
        raise SystemExit(f"Unexpected confirmatory manifest label: {manifest_path}")
    output_dir = (args.output_dir or experiment_root / "final_summary").resolve()
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite append-only output: {existing}")

    eval_orchestration = experiment_root / "eval_orchestration" / orchestration_name
    if not (eval_orchestration / "COMPLETED").is_file():
        raise SystemExit(f"Evaluation matrix is not complete: {eval_orchestration}")

    training = {
        method: summarize_training(
            project_dir, experiment_root, manifest_path, method, args.run_id
        )
        for method in METHODS
    }
    evaluation: dict[str, dict[str, dict]] = {}
    total_errors: Counter[str] = Counter()
    total_trajectories = total_success = total_contaminated = total_system_errors = 0
    for candidate_index, candidate in enumerate(CANDIDATES):
        evaluation[candidate] = {}
        candidate_dir = experiment_root / "eval" / candidate / args.run_id
        if not (candidate_dir / "COMPLETED").is_file():
            raise SystemExit(f"Evaluation candidate is not complete: {candidate_dir}")
        if read_json(candidate_dir / "frozen_manifest.json") != manifest:
            raise SystemExit(f"Frozen evaluation manifest mismatch: {candidate_dir}")
        candidate_gate = read_json(candidate_dir / "gate_result.json")
        if (
            not candidate_gate.get("passed")
            or candidate_gate.get("candidate") != candidate
            or candidate_gate.get("run_id") != args.run_id
        ):
            raise SystemExit(f"Evaluation candidate gate did not pass: {candidate_dir}")
        for split_index, split in enumerate(SPLITS):
            path = candidate_dir / split / "eval_report.json"
            result = summarize_evaluation(
                read_json(path),
                path.relative_to(project_dir),
                manifest,
                split,
                bootstrap_seed=20260825 + candidate_index * 10 + split_index,
                bootstrap_replicates=args.bootstrap_replicates,
            )
            evaluation[candidate][split] = result
            total_errors.update(result["error_classes"])
            total_trajectories += result["trajectory_count"]
            total_success += result["success_count"]
            total_contaminated += result["contaminated_trajectory_count"]
            total_system_errors += result["system_error_count"]
    if total_trajectories != 1008:
        raise SystemExit(f"Expected 1008 confirmatory trajectories, got {total_trajectories}")

    paired_against_sft: dict[str, dict[str, dict]] = {}
    for candidate_index, candidate in enumerate(CANDIDATES[1:], start=1):
        paired_against_sft[candidate] = {}
        for split_index, split in enumerate(SPLITS):
            paired_against_sft[candidate][split] = paired_delta(
                evaluation["sft"][split],
                evaluation[candidate][split],
                bootstrap_seed=40260825 + candidate_index * 10 + split_index,
                bootstrap_replicates=args.bootstrap_replicates,
            )

    checkpoint_deltas: dict[str, dict[str, dict]] = {}
    checkpoint_curves: dict[str, dict[str, list[dict]]] = {}
    for method_index, method in enumerate(METHODS):
        checkpoint_deltas[method] = {}
        checkpoint_curves[method] = {}
        for split_index, split in enumerate(SPLITS):
            checkpoint_deltas[method][split] = paired_delta(
                evaluation[f"{method}_step_4"][split],
                evaluation[f"{method}_step_8"][split],
                bootstrap_seed=50260825 + method_index * 10 + split_index,
                bootstrap_replicates=args.bootstrap_replicates,
            )
            checkpoint_curves[method][split] = [
                {
                    "step": step,
                    "candidate": candidate,
                    "metrics": evaluation[candidate][split]["metrics"],
                    "pass_hat_1_task_bootstrap_95ci": evaluation[candidate][split][
                        "pass_hat_1_task_bootstrap_95ci"
                    ],
                }
                for step, candidate in (
                    (0, "sft"),
                    (4, f"{method}_step_4"),
                    (8, f"{method}_step_8"),
                )
            ]

    summary = {
        "schema_version": 1,
        "label": manifest["experiment_label"],
        "interpretation": manifest["interpretation"],
        "run_id": args.run_id,
        "eval_orchestration_name": orchestration_name,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_manifest": str(manifest_path.relative_to(project_dir)),
        "reproduce_command": (
            "python scripts/eval/summarize_retail_confirmatory.py "
            f"--run-id {args.run_id} "
            f"--experiment-root {args.experiment_root} "
            f"--manifest {args.manifest} "
            f"--seen-config {args.seen_config} "
            f"--unseen-config {args.unseen_config} "
            f"--eval-matrix-script {args.eval_matrix_script} "
            f"--eval-orchestration-name {orchestration_name}"
        ),
        "source_hashes": {
            str(path.relative_to(project_dir)): sha256_file(path)
            for path in (
                manifest_path,
                project_dir / "configs/train/grpo/retail_confirmatory_4090.yaml",
                project_dir / args.seen_config,
                project_dir / args.unseen_config,
                project_dir / manifest["training"]["train_parquet"],
                project_dir / "src/evaluation/pass_k_eval.py",
                project_dir / "scripts/eval/eval_sft.py",
                project_dir / "scripts/eval/check_retail_confirmatory_eval.py",
                project_dir / "scripts/eval/run_retail_confirmatory_eval.sh",
                project_dir / args.eval_matrix_script,
                Path(__file__).resolve(),
            )
        },
        "protocol": manifest,
        "bootstrap": {
            "unit": "task",
            "replicates": args.bootstrap_replicates,
            "interval": "percentile 95%",
            "deterministic_seed_family": "20260825/40260825/50260825 plus candidate and split offsets",
        },
        "training": training,
        "evaluation": evaluation,
        "checkpoint_curves": checkpoint_curves,
        "paired_against_sft": paired_against_sft,
        "checkpoint_step_8_minus_step_4": checkpoint_deltas,
        "totals": {
            "trajectory_count": total_trajectories,
            "success_count": total_success,
            "error_count": sum(total_errors.values()),
            "error_classes": dict(sorted(total_errors.items())),
            "system_error_count": total_system_errors,
            "contaminated_trajectory_count": total_contaminated,
            "contamination_rate": total_contaminated / total_trajectories,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    report_path.write_text(render_report(summary))
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()
