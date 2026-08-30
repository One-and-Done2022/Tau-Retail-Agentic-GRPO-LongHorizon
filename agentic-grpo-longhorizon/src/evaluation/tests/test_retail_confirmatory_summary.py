from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/eval/summarize_retail_confirmatory.py"
SPEC = importlib.util.spec_from_file_location("retail_confirmatory_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


def _manifest() -> dict:
    return {
        "evaluation": {
            "seen": {"task_ids": [0, 1]},
            "unseen": {"task_ids": [2, 3]},
            "samples_per_task": 2,
            "policy_seed_base": 100,
            "user_seed_base": 200,
        }
    }


def _trajectory(task_id: int, sample_id: int, success: bool) -> dict:
    return {
        "task_id": task_id,
        "sample_id": sample_id,
        "policy_seed": 100 + task_id * 1000 + sample_id,
        "user_seed": 200 + task_id * 1000 + sample_id,
        "success": success,
        "reward": float(success),
        "error": None,
        "was_contaminated_from_turn": None,
        "constraint_violation_count": 0,
        "constraint_violations": {},
        "high_risk_tool_calls": 1,
        "high_risk_tool_errors": 0,
        "tool_error_count": 0,
        "recovered_tool_errors": 0,
        "num_turns": 3,
        "num_tool_calls": 1,
        "per_turn_assistant_content_tokens": [4, 5],
        "latency_seconds": 1.0,
        "raw_messages": [{"role": "system", "content": "must not be copied"}],
    }


def _report(outcomes: dict[int, list[bool]]) -> dict:
    task_rows = []
    for task_id, values in outcomes.items():
        trajectories = [
            _trajectory(task_id, sample_id, success)
            for sample_id, success in enumerate(values)
        ]
        task_rows.append(
            {
                "task_id": task_id,
                "success_count": sum(values),
                "total_samples": len(values),
                "pass^1": sum(values) / len(values),
                "trajectories": trajectories,
            }
        )
    pass_hat_1 = sum(sum(values) / len(values) for values in outcomes.values()) / len(outcomes)
    return {
        "env_name": "retail",
        "num_tasks": len(outcomes),
        "num_samples_per_task": 2,
        "policy_seed_base": 100,
        "user_seed_base": 200,
        "seed_rule": "base + task_id * 1000 + sample_id",
        "pass_hat_1": pass_hat_1,
        "pass_at_1": sum(any(values) for values in outcomes.values()) / len(outcomes),
        "pass_hat_4": 0.0,
        "error_rate": 0.0,
        "constraint_violation_rate": 0.0,
        "high_risk_tool_error_rate": 0.0,
        "tool_error_rate": 0.0,
        "error_recovery_rate": 0.0,
        "avg_turns": 3.0,
        "avg_tool_calls": 1.0,
        "avg_reasoning_tokens_per_turn": 4.5,
        "max_reasoning_tokens_single_turn": 5,
        "avg_latency_seconds": 1.0,
        "per_task_results": task_rows,
    }


def test_summary_preserves_pair_identity_without_raw_messages():
    result = SUMMARY.summarize_evaluation(
        _report({0: [False, True], 1: [True, True]}),
        Path("synthetic/eval_report.json"),
        _manifest(),
        "seen",
        bootstrap_seed=7,
        bootstrap_replicates=100,
    )

    assert result["trajectory_count"] == 4
    assert list(result["pairs"]) == ["0:0", "0:1", "1:0", "1:1"]
    assert result["pairs"]["1:1"]["policy_seed"] == 1101
    assert "raw_messages" not in str(result)


def test_paired_delta_uses_task_as_bootstrap_unit():
    parent = SUMMARY.summarize_evaluation(
        _report({0: [False, False], 1: [True, True]}),
        Path("parent/eval_report.json"),
        _manifest(),
        "seen",
        bootstrap_seed=7,
        bootstrap_replicates=100,
    )
    candidate = SUMMARY.summarize_evaluation(
        _report({0: [True, False], 1: [True, False]}),
        Path("candidate/eval_report.json"),
        _manifest(),
        "seen",
        bootstrap_seed=8,
        bootstrap_replicates=100,
    )

    delta = SUMMARY.paired_delta(
        parent, candidate, bootstrap_seed=9, bootstrap_replicates=100
    )

    assert delta["success_delta"] == pytest.approx(0.0)
    assert delta["task_deltas"] == {"0": 0.5, "1": -0.5}
    assert delta["task_bootstrap_95ci"]["unit"] == "task"


def test_summary_rejects_wrong_policy_seed():
    report = _report({0: [False, True], 1: [True, True]})
    report["per_task_results"][0]["trajectories"][0]["policy_seed"] = 999

    with pytest.raises(SystemExit, match="Policy seed mismatch"):
        SUMMARY.summarize_evaluation(
            report,
            Path("synthetic/eval_report.json"),
            _manifest(),
            "seen",
            bootstrap_seed=7,
            bootstrap_replicates=100,
        )
