from __future__ import annotations

import copy
import asyncio
from pathlib import Path

import pytest

from tau_bench.envs import get_env

from src.envs.retail_cs_grpo import (
    RetailProgressTracker,
    check_retail_constraints,
    compute_step_credit,
    has_explicit_confirmation,
    observe_retail_tool_result,
    redundant_retail_write,
)
from src.envs.tau_bench_context import CURRENT_TAU_STATE
from src.envs.tau_bench_interaction import TauBenchInteraction
from verl.tools.utils.tool_registry import initialize_tools_from_config


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _train_env(task_id: int = 0):
    return get_env("retail", "human", "dummy", "train", task_index=task_id)


def _valid_state(env) -> dict:
    order_id = env.task.actions[0].kwargs["order_id"]
    return {
        "authenticated_user_id": env.task.user_id,
        "read_order_ids": [order_id],
        "read_user_ids": [env.task.user_id],
        "successful_writes": [],
        "last_user_message": "Yes, that's correct. Please proceed.",
    }


def test_goal_tracker_no_change_progress_and_regression():
    env = _train_env(0)
    tracker = RetailProgressTracker.from_env(env, task_split="train")
    initial_data = copy.deepcopy(env.data)
    before = tracker.snapshot(env.data)

    # A read/no-op leaves hidden goal progress unchanged.
    no_change = tracker.snapshot(env.data)
    assert no_change.progress == before.progress
    assert no_change.digest == before.digest

    action = env.task.actions[0]
    observation = env.tools_map[action.name].invoke(data=env.data, **action.kwargs)
    assert not observation.startswith("Error:")
    after = tracker.snapshot(env.data)
    assert after.progress > before.progress
    assert after.progress == 1.0

    # Restoring the pre-action records is a measurable regression.
    env.data = initial_data
    regressed = tracker.snapshot(env.data)
    assert regressed.progress - after.progress < 0


def test_goal_tracker_rejects_non_train_split():
    env = get_env("retail", "human", "dummy", "dev", task_index=0)
    with pytest.raises(ValueError, match="restricted"):
        RetailProgressTracker.from_env(env, task_split="dev")


def test_goal_tracker_matches_native_noop_semantics_for_failed_target_action():
    env = _train_env(24)
    tracker = RetailProgressTracker.from_env(env, task_split="train")
    assert tracker.goal_action_errors
    assert "insufficient gift card balance" in tracker.goal_action_errors[0]
    assert tracker.total_goal_leaves > 0  # a later target action can still mutate state


def test_step_credit_error_and_redundant_penalties():
    env = _train_env(0)
    tracker = RetailProgressTracker.from_env(env, task_split="train")
    snapshot = tracker.snapshot(env.data)
    config = {
        "progress_weight": 1.0,
        "constraint_weight": 0.0,
        "error_penalty": 0.75,
        "redundant_write_penalty": 0.5,
    }

    error_credit = compute_step_credit(
        tracker, snapshot, snapshot, (), tool_error=True, redundant_write=False, config=config
    )
    assert error_credit.progress_delta == 0.0
    assert error_credit.reward == pytest.approx(-0.75)

    repeated_credit = compute_step_credit(
        tracker, snapshot, snapshot, (), tool_error=False, redundant_write=True, config=config
    )
    assert repeated_credit.reward == pytest.approx(-0.5)


def test_valid_return_constraints_and_observation_evidence():
    env = _train_env(0)
    action = env.task.actions[0]
    state = _valid_state(env)
    assert check_retail_constraints(action.name, action.kwargs, env.data, state) == ()

    evidence_state = {
        "authenticated_user_id": None,
        "read_order_ids": [],
        "read_user_ids": [],
        "successful_writes": [],
    }
    observe_retail_tool_result(
        "find_user_id_by_email", {"email": "x@example.com"}, env.task.user_id, evidence_state
    )
    observe_retail_tool_result(
        "get_order_details", {"order_id": action.kwargs["order_id"]}, "{}", evidence_state
    )
    observe_retail_tool_result(
        "get_user_details", {"user_id": env.task.user_id}, "{}", evidence_state
    )
    assert evidence_state["authenticated_user_id"] == env.task.user_id
    assert evidence_state["read_order_ids"] == [action.kwargs["order_id"]]
    assert evidence_state["read_user_ids"] == [env.task.user_id]


def test_constraint_violations_are_generic_and_pre_action():
    env = _train_env(0)
    action = env.task.actions[0]
    bad = {
        "authenticated_user_id": "some_other_user_1",
        "read_order_ids": [],
        "read_user_ids": [],
        "successful_writes": [],
        "last_user_message": "No, do not proceed yet.",
    }
    params = dict(action.kwargs)
    params["item_ids"] = ["not_in_this_order"]
    params["payment_method_id"] = "credit_card_not_owned"
    codes = {item.code for item in check_retail_constraints(action.name, params, env.data, bad)}
    assert {
        "authenticated_user_not_order_owner",
        "order_not_read",
        "user_not_read",
        "order_item_mismatch",
        "payment_method_not_owned",
        "missing_explicit_confirmation",
    } <= codes

    order = env.data["orders"][action.kwargs["order_id"]]
    order["status"] = "pending"
    codes = {item.code for item in check_retail_constraints(action.name, action.kwargs, env.data, _valid_state(env))}
    assert "order_not_delivered" in codes


def test_repeated_successful_write_is_detected():
    env = _train_env(0)
    action = env.task.actions[0]
    state = _valid_state(env)
    observe_retail_tool_result(action.name, action.kwargs, "{}", state)
    assert redundant_retail_write(action.name, action.kwargs, state)
    codes = {item.code for item in check_retail_constraints(action.name, action.kwargs, env.data, state)}
    assert "repeated_one_shot_write" in codes


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Yes, please proceed.", True),
        ("I confirm that is correct.", True),
        ("Please return my order.", False),
        ("No, do not proceed.", False),
        ("Yes, wait—not yet.", False),
        ("", False),
    ],
)
def test_explicit_confirmation(message: str, expected: bool):
    assert has_explicit_confirmation(message) is expected


def test_interaction_and_tool_emit_serializable_step_credit(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "initial request")

    async def run():
        interaction = TauBenchInteraction(
            {
                "env_name": "retail",
                "user_strategy": "human",
                "user_model": "dummy",
                "task_split": "train",
                "reward_mode": "binary",
                "step_credit": {
                    "enabled": True,
                    "progress_weight": 1.0,
                    "constraint_weight": 1.0,
                    "error_penalty": 1.0,
                    "redundant_write_penalty": 1.0,
                },
            }
        )
        await interaction.start_interaction("integration", task_id=0)
        state = CURRENT_TAU_STATE.get()
        env = interaction._instance_dict["integration"]["env"]
        action = env.task.actions[0]
        state.update(_valid_state(env))

        config_path = _PROJECT_ROOT / "configs/tool_config/tau_bench_retail_tools.yaml"
        tools = {tool.name: tool for tool in initialize_tools_from_config(str(config_path))}
        tool = tools[action.name]
        instance_id, _ = await tool.create()
        response, reward, metadata = await tool.execute(instance_id, action.kwargs)
        await interaction.finalize_interaction("integration")
        return response, reward, metadata

    response, reward, metadata = asyncio.run(run())
    assert not response.text.startswith("Error:")
    assert reward == pytest.approx(1.0)
    assert metadata["cs_step_credit"]["progress_delta"] == pytest.approx(1.0)
    assert metadata["cs_step_credit"]["violations"] == []


def test_interaction_derives_deterministic_user_seed(monkeypatch):
    import tau_bench.envs as env_module

    captured = []
    real_get_env = env_module.get_env

    def capture_get_env(**kwargs):
        captured.append(kwargs.get("user_seed"))
        return real_get_env(**kwargs)

    monkeypatch.setattr(env_module, "get_env", capture_get_env)
    monkeypatch.setattr("builtins.input", lambda _: "initial request")

    async def run():
        interaction = TauBenchInteraction(
            {
                "env_name": "retail",
                "user_strategy": "human",
                "user_model": "dummy",
                "task_split": "train",
                "user_seed": 7000,
            }
        )
        await interaction.start_interaction("seeded", task_id=2)
        await interaction.finalize_interaction("seeded")

    asyncio.run(run())
    assert captured == [9000]
