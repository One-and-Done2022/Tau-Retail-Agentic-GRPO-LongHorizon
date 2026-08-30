from __future__ import annotations

from dataclasses import replace

import pytest

from tau_bench.envs import get_env

from src.envs.tau_bench_wrapper import TauBenchWrapper, TrajectoryResult
from src.evaluation.pass_k_eval import run_eval


class FakeWrapper:
    env_name = "retail"

    def __init__(self):
        self.user_seeds = []

    def get_num_tasks(self) -> int:
        return 6

    def run_single_task(
        self, task_idx: int, policy, max_turns: int, user_seed=None
    ) -> TrajectoryResult:
        del policy, max_turns
        self.user_seeds.append((task_idx, user_seed))
        base = TrajectoryResult(
            task_id=task_idx,
            success=task_idx == 4,
            reward=float(task_idx == 4),
            num_turns=task_idx,
            num_tool_calls=2,
            raw_messages=[{"role": "assistant", "content": "ok"}],
            constraint_violation_count=1,
            high_risk_tool_calls=2,
            high_risk_tool_errors=1,
            tool_error_count=1,
            recovered_tool_errors=1,
            latency_seconds=float(task_idx),
            user_seed=user_seed,
        )
        return replace(base)


def test_run_eval_accepts_explicit_non_contiguous_task_ids(tmp_path):
    policy_seeds = []
    wrapper = FakeWrapper()

    def policy_factory(task_idx=None, sample_idx=None):
        policy_seeds.append((task_idx, sample_idx))
        return object()

    report = run_eval(
        wrapper=wrapper,
        policy_factory=policy_factory,
        task_ids=[4, 1],
        num_samples_per_task=2,
        num_workers=1,
        output_dir=tmp_path,
        policy_seed_base=100,
        user_seed_base=200,
    )

    assert report.num_tasks == 2
    assert [row["task_id"] for row in report.per_task_results] == [4, 1]
    assert report.pass_hat_1 == pytest.approx(0.5)
    assert report.constraint_violation_rate == pytest.approx(0.5)
    assert report.high_risk_tool_error_rate == pytest.approx(0.5)
    assert report.tool_error_rate == pytest.approx(0.5)
    assert report.error_recovery_rate == pytest.approx(1.0)
    assert report.avg_latency_seconds == pytest.approx(2.5)
    assert report.avg_reasoning_tokens_per_turn == pytest.approx(2.0)
    assert report.max_reasoning_tokens_single_turn == 2
    assert policy_seeds[:4] == [(4, 0), (4, 1), (1, 0), (1, 1)]
    assert policy_seeds[-1] == (None, None)  # tokenizer probe
    assert wrapper.user_seeds == [(4, 4200), (4, 4201), (1, 1200), (1, 1201)]
    assert report.policy_seed_base == 100
    assert report.user_seed_base == 200
    for row in report.per_task_results:
        for sample_id, trajectory in enumerate(row["trajectories"]):
            assert trajectory["sample_id"] == sample_id
            assert trajectory["policy_seed"] == 100 + row["task_id"] * 1000 + sample_id
            assert trajectory["user_seed"] == 200 + row["task_id"] * 1000 + sample_id
    assert (tmp_path / "eval_report.json").is_file()


@pytest.mark.parametrize("task_ids", [[], [1, 1], [-1], [6]])
def test_run_eval_rejects_invalid_explicit_task_ids(tmp_path, task_ids):
    with pytest.raises(ValueError):
        run_eval(
            wrapper=FakeWrapper(),
            policy_factory=lambda: object(),
            task_ids=task_ids,
            num_samples_per_task=1,
            num_workers=1,
            output_dir=tmp_path,
        )


class DeterministicUser:
    def reset(self, instruction=None):
        return "Please return my delivered order."

    def step(self, content):
        return "###STOP###"

    def get_total_cost(self):
        return 0.0


class ErrorThenRecoveryPolicy:
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.turn = 0

    def set_tools(self, tools):
        self.tools = tools

    def __call__(self, messages):
        del messages
        if self.turn == 0:
            name = "return_delivered_order_items"
            arguments = (
                '{"order_id":"%s","item_ids":["not_an_order_item"],'
                '"payment_method_id":"not_owned"}' % self.order_id
            )
        else:
            name = "get_order_details"
            arguments = '{"order_id":"%s"}' % self.order_id
        self.turn += 1
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{self.turn}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }


def test_retail_trajectory_records_constraint_error_and_recovery():
    env = get_env("retail", "human", "dummy", "train", task_index=0)
    env.user = DeterministicUser()
    order_id = env.task.actions[0].kwargs["order_id"]
    wrapper = TauBenchWrapper(env_name="retail", user_strategy="human", task_split="train")
    wrapper._make_env = lambda _, user_seed=None: env

    trajectory = wrapper.run_single_task(
        0, ErrorThenRecoveryPolicy(order_id), max_turns=2
    )

    assert trajectory.high_risk_tool_calls == 1
    assert trajectory.constraint_violation_count == 1
    assert trajectory.constraint_violations
    assert trajectory.high_risk_tool_errors == 1
    assert trajectory.tool_error_count == 1
    assert trajectory.recovered_tool_errors == 1
    assert trajectory.num_tool_calls == 2
    assert trajectory.latency_seconds > 0
