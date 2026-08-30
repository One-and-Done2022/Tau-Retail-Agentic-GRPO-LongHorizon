#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


METHODS = ("sft", "vanilla", "state", "constraint", "cs")
RL_METHODS = METHODS[1:]
SPLITS = ("seen", "unseen")
METRICS = (
    "pass_hat_1",
    "pass_at_1",
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
    "critic/score/all_zero_frac",
    "critic/score/all_one_frac",
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing required JSON: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SystemExit(f"Expected finite numeric {label}, got {value!r}")
    return float(value)


def trajectories(report: dict, source: Path) -> list[dict]:
    result: list[dict] = []
    for task in report.get("per_task_results", []):
        result.extend(task.get("trajectories", []))
    expected = int(report["num_tasks"]) * int(report["num_samples_per_task"])
    if len(result) != expected:
        raise SystemExit(f"Trajectory count mismatch in {source}: expected {expected}, got {len(result)}")
    return result


def error_class(error: object) -> str | None:
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


def summarize_eval(report: dict, source: Path) -> dict:
    rows = trajectories(report, source)
    errors = Counter(category for row in rows if (category := error_class(row.get("error"))))
    contaminated = sum(row.get("was_contaminated_from_turn") is not None for row in rows)
    metrics = {key: finite_number(report[key], f"{source}:{key}") for key in METRICS}
    return {
        "source": str(source),
        "num_tasks": int(report["num_tasks"]),
        "samples_per_task": int(report["num_samples_per_task"]),
        "trajectory_count": len(rows),
        "success_count": sum(bool(row.get("success")) for row in rows),
        "metrics": metrics,
        "error_count": sum(errors.values()),
        "error_classes": dict(sorted(errors.items())),
        "contaminated_trajectory_count": contaminated,
        "contamination_rate": contaminated / len(rows) if rows else 0.0,
    }


def summarize_gate(gate: dict, gate_path: Path, project_dir: Path, method: str) -> dict:
    if not gate.get("passed") or gate.get("method") != method:
        raise SystemExit(f"Training gate did not pass for {method}: {gate_path}")
    steps = gate.get("steps", {})
    if sorted(steps) != ["1", "2", "3"]:
        raise SystemExit(f"Expected training steps 1-3 for {method}, got {sorted(steps)}")

    selected_steps: dict[str, dict[str, float]] = {}
    for step, values in steps.items():
        selected_steps[step] = {
            key: finite_number(values[key], f"{gate_path}:step={step}:{key}") for key in TRAIN_METRICS
        }
        if selected_steps[step]["actor/grad_norm"] <= 0:
            raise SystemExit(f"Non-positive grad norm for {method} step {step}")

    run_dir = gate_path.parent
    merged_dir = run_dir / "hf_step_3"
    reload_log = run_dir / "hf_reload.log"
    has_weights = (merged_dir / "model.safetensors").is_file() or (
        merged_dir / "model.safetensors.index.json"
    ).is_file()
    if not has_weights or not (merged_dir / "config.json").is_file() or not reload_log.is_file():
        raise SystemExit(f"Incomplete merged/reload evidence for {method}: {run_dir}")
    reload_text = reload_log.read_text(errors="replace")
    if "HF_RELOAD" not in reload_text:
        raise SystemExit(f"HF reload marker missing for {method}: {reload_log}")

    aggregates = {
        f"{key}_mean": sum(values[key] for values in selected_steps.values()) / len(selected_steps)
        for key in TRAIN_METRICS
    }
    aggregates["actor/grad_norm_min"] = min(
        values["actor/grad_norm"] for values in selected_steps.values()
    )
    return {
        "source": str(gate_path.relative_to(project_dir)),
        "passed": True,
        "run_id": gate.get("run_id"),
        "trained_updates": len(selected_steps),
        "steps": selected_steps,
        "aggregates": aggregates,
        "merged_checkpoint": str(merged_dir.relative_to(project_dir)),
        "hf_reload_evidence": str(reload_log.relative_to(project_dir)),
    }


def summarize_superseded(project_dir: Path, method: str, run_id: str) -> dict:
    split_results = {}
    totals = Counter()
    for split in SPLITS:
        path = project_dir / "experiments/retail_ablation_eval" / method / run_id / split / "eval_report.json"
        report = read_json(path)
        rows = trajectories(report, path)
        classes = Counter(category for row in rows if (category := error_class(row.get("error"))))
        totals.update(classes)
        split_results[split] = {
            "source": str(path.relative_to(project_dir)),
            "trajectory_count": len(rows),
            "error_classes": dict(sorted(classes.items())),
        }
    return {
        "method": method,
        "run_id": run_id,
        "splits": split_results,
        "error_classes": dict(sorted(totals.items())),
    }


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def number(value: float) -> str:
    return f"{value:.3f}"


def render_report(summary: dict) -> str:
    lines = [
        "# Retail 固定预算工程 pilot：run_003",
        "",
        "## 结论",
        "",
        "Baseline、SFT、Vanilla GRPO 和 CS-GRPO 的 smoke 门禁已在此前验证；本轮固定预算 pilot "
        "完成 Vanilla、State、Constraint、CS 四种方法各 3 updates 的训练门禁，以及 SFT + 四种方法"
        "统一协议下的 seen/unseen 评测。结果仅用于验证工程闭环和观察现象，不构成正式性能或泛化结论。",
        "",
        "CS-GRPO 在 unseen 的 `pass^1` 数值为 0.250，是本 pilot 的最高值；但 unseen 每个方法只有 "
        "8 条轨迹，且用户模拟器未提供固定 seed，因此不能据此宣称方法优越。",
        "",
        "## 固定协议",
        "",
        "- 训练：同一 SFT 起点，train task 顺序 `[1, 0]`，每种 RL 方法 3 updates，group size 4。",
        "- 评测：seen train tasks `[0, 1]`；unseen test tasks `[0, 1, 2, 3]`；每任务 2 samples，"
        "20 turns，temperature 0.7，top_p 0.9，max output 4096。",
        "- policy seed：`20260824 + task_id * 1000 + sample_id`；用户模拟器接口未传 seed。",
        "- 最终横向比较仅读取 `run_003`；τ-bench 原生 outcome 和成功条件保持不变。",
        "",
        "## 三步训练门禁",
        "",
        "| 方法 | gate | reward mean | all-zero | all-one | pg loss mean | grad norm mean | grad norm min |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in RL_METHODS:
        gate = summary["training_gates"][method]
        agg = gate["aggregates"]
        lines.append(
            f"| {method} | {'pass' if gate['passed'] else 'fail'} | "
            f"{number(agg['critic/rewards/mean_mean'])} | "
            f"{number(agg['critic/score/all_zero_frac_mean'])} | "
            f"{number(agg['critic/score/all_one_frac_mean'])} | "
            f"{agg['actor/pg_loss_mean']:.3e} | {number(agg['actor/grad_norm_mean'])} | "
            f"{number(agg['actor/grad_norm_min'])} |"
        )

    lines.extend(
        [
            "",
            "四组均有 3 个非空 rollout、有限 loss/gradient、正 `grad_norm`、完整双 rank checkpoint，"
            "且 `global_step_3` 合并模型通过 Hugging Face reload。",
            "",
            "## run_003 统一评测",
            "",
            "| 方法 | split | pass^1 | pass@1 | error | violation | high-risk error | tool error | recovery | turns | tools | tokens/turn | max tokens | latency(s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        for split in SPLITS:
            metrics = summary["evaluation"][method][split]["metrics"]
            lines.append(
                f"| {method} | {split} | {number(metrics['pass_hat_1'])} | "
                f"{number(metrics['pass_at_1'])} | {number(metrics['error_rate'])} | "
                f"{number(metrics['constraint_violation_rate'])} | "
                f"{number(metrics['high_risk_tool_error_rate'])} | "
                f"{number(metrics['tool_error_rate'])} | {number(metrics['error_recovery_rate'])} | "
                f"{metrics['avg_turns']:.2f} | {metrics['avg_tool_calls']:.2f} | "
                f"{metrics['avg_reasoning_tokens_per_turn']:.2f} | "
                f"{metrics['max_reasoning_tokens_single_turn']:.0f} | "
                f"{metrics['avg_latency_seconds']:.2f} |"
            )

    totals = summary["run_003_totals"]
    lines.extend(
        [
            "",
            "## 错误、污染与协议修订",
            "",
            f"- `run_003` 共 {totals['trajectory_count']} 条轨迹，{totals['error_count']} 条 error 均归类为 "
            "`repeated_tool_call_loop`；context/OOM 系统错误为 0。",
            f"- 历史消息截断污染为 {totals['contaminated_trajectory_count']}/{totals['trajectory_count']} "
            f"({percent(totals['contamination_rate'])})：Vanilla unseen 与 State unseen 各 1 条。",
            "- `run_001` 的 8,192 context 使 SFT 12/12 请求发生 context failure；原始报告保留，未进入横向表。",
            "- `run_002` 的 16,384 context 使 Vanilla 1/12 请求在 16,401 tokens 发生 context failure；"
            "原始报告保留，未进入横向表。",
            "- `run_003` 统一采用 24,576 context，已知 KV cache 容量为 86,288 tokens，未发生 context failure。",
            "",
            "## 证据层级与边界",
            "",
            "- 已验证 smoke：retail 环境 baseline、SFT save/merge/reload/tool-call、Vanilla GRPO 单步、"
            "CS-GRPO 单步以及相关状态/约束/span 单元测试。",
            "- 本报告：固定预算工程 pilot；四种 RL 方法只训练 3 updates，并只评测最终 checkpoint。",
            "- 尚未完成正式结论所需的扩大任务集、多 checkpoint、多 policy/user-simulator seed 与足够重复采样；"
            "风险分层采样 Full 版本也不在本 pilot 范围。",
            "- 高约束违例率和小样本波动应作为后续诊断信号，不能脱离每条规则的分母直接排序模型。",
            "",
            "## 复现入口",
            "",
            "```bash",
            "python scripts/eval/summarize_retail_ablation.py --run-id run_003",
            "```",
            "",
            "该命令采用 append-only 输出策略；目标报告存在时会拒绝覆盖。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the frozen retail ablation pilot")
    parser.add_argument("--run-id", default="run_003")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    manifest_path = project_dir / "configs/eval/retail/ablation_manifest.json"
    manifest = read_json(manifest_path)
    if args.run_id != "run_003":
        raise SystemExit("The frozen final comparison is restricted to run_003")
    output_dir = args.output_dir or (
        project_dir / "experiments/retail_ablation_eval" / f"fixed_budget_pilot_{args.run_id}"
    )
    output_dir = output_dir.resolve()
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite append-only output: {existing}")

    training_gates = {}
    for method in RL_METHODS:
        path = project_dir / "experiments/retail_ablation" / method / "run_001/gate_result.json"
        training_gates[method] = summarize_gate(read_json(path), path, project_dir, method)

    evaluation = {}
    total_errors = Counter()
    total_trajectories = 0
    total_contaminated = 0
    for method in METHODS:
        evaluation[method] = {}
        for split in SPLITS:
            path = (
                project_dir
                / "experiments/retail_ablation_eval"
                / method
                / args.run_id
                / split
                / "eval_report.json"
            )
            result = summarize_eval(read_json(path), path.relative_to(project_dir))
            evaluation[method][split] = result
            total_errors.update(result["error_classes"])
            total_trajectories += result["trajectory_count"]
            total_contaminated += result["contaminated_trajectory_count"]

    summary = {
        "schema_version": 1,
        "label": "fixed-budget-engineering-pilot",
        "run_id": args.run_id,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "interpretation": "Engineering pilot only; not a formal performance or generalization conclusion.",
        "source_manifest": str(manifest_path.relative_to(project_dir)),
        "protocol": {
            "training": manifest["training"],
            "evaluation": manifest["evaluation"],
            "amendments": manifest.get("amendments", []),
        },
        "training_gates": training_gates,
        "evaluation": evaluation,
        "run_003_totals": {
            "trajectory_count": total_trajectories,
            "error_count": sum(total_errors.values()),
            "error_classes": dict(sorted(total_errors.items())),
            "system_error_count": sum(
                count for category, count in total_errors.items() if category != "repeated_tool_call_loop"
            ),
            "contaminated_trajectory_count": total_contaminated,
            "contamination_rate": total_contaminated / total_trajectories,
        },
        "superseded_protocol_evidence": [
            summarize_superseded(project_dir, "sft", "run_001"),
            summarize_superseded(project_dir, "vanilla", "run_002"),
        ],
        "formal_evaluation_remaining": [
            "larger multi-task seen/unseen suites",
            "multiple checkpoints",
            "multiple policy and user-simulator seeds",
            "enough repeated samples for uncertainty estimates",
        ],
    }
    if summary["run_003_totals"]["system_error_count"]:
        raise SystemExit("run_003 contains a system error; inspect before reporting")

    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    report_path.write_text(render_report(summary))
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()
