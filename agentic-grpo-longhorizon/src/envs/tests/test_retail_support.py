from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau_bench.envs import get_env
from tau_bench.types import Action

from src.envs.tau_bench_tools import verify_tool_classes_match_env
from src.envs.tau_bench_wrapper import TauBenchWrapper


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PARQUET_SCRIPT = _PROJECT_ROOT / "scripts/train/grpo/build_grpo_parquet.py"
_SPEC = importlib.util.spec_from_file_location("build_grpo_parquet", _PARQUET_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_PARQUET_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PARQUET_MODULE)
build_rows = _PARQUET_MODULE.build_rows


class DeterministicUser:
    def reset(self, instruction=None):
        self.instruction = instruction
        return "I need help with my retail order."

    def step(self, content):
        return "###STOP###"

    def get_total_cost(self):
        return 0.0


class CapturePolicy:
    def set_tools(self, tools):
        self.tools = tools

    def __call__(self, messages):
        self.messages = messages
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "capture",
                "type": "function",
                "function": {
                    "name": "transfer_to_human_agents",
                    "arguments": '{"summary":"prompt capture"}',
                },
            }],
        }


def test_retail_splits_and_tool_classes_match():
    expected_counts = {"train": 500, "dev": 20, "test": 115}
    for split, expected in expected_counts.items():
        env = get_env("retail", "human", "dummy", split, task_index=0)
        assert len(env.tasks) == expected

    assert len(verify_tool_classes_match_env("retail")) == 16

    with pytest.raises(ValueError, match="Only the retail environment"):
        get_env("unsupported", "human", "dummy", "test", task_index=0)


def test_retail_reset_tool_state_and_native_reward():
    env = get_env("retail", "human", "dummy", "test", task_index=0)
    env.user = DeterministicUser()

    reset = env.reset(task_index=0)
    assert reset.observation == "I need help with my retail order."
    initial_hash = env.get_data_hash()

    for action in env.task.actions:
        response = env.step(action)
        assert not str(response.observation).startswith("Error:")

    response = env.step(
        Action(
            name="transfer_to_human_agents",
            kwargs={"summary": "deterministic retail baseline verification"},
        )
    )
    assert response.done
    assert response.reward == 1.0
    assert env.get_data_hash() != initial_hash


def test_retail_grpo_rows_record_environment_identity():
    rows = build_rows(
        [3],
        split="seen",
        env_name="retail",
        task_split="train",
        interaction_name="tau_bench_retail_train",
    )
    assert rows[0]["data_source"] == "tau_bench_retail_train"
    assert rows[0]["extra_info"]["env_name"] == "retail"
    assert rows[0]["extra_info"]["task_split"] == "train"
    assert "authenticate the user identity" in rows[0]["prompt"][0]["content"]
    assert "product orders and exchanges" in rows[0]["prompt"][0]["content"]


def test_sft_wrapper_includes_official_retail_policy():
    env = get_env("retail", "human", "dummy", "test", task_index=0)
    env.user = DeterministicUser()
    wrapper = TauBenchWrapper(env_name="retail", user_strategy="human")
    wrapper._make_env = lambda _, user_seed=None: env
    policy = CapturePolicy()

    wrapper.run_single_task(0, policy, max_turns=1)

    system_prompt = policy.messages[0]["content"]
    assert "authenticate the user identity" in system_prompt
    assert "Current Date Context" in system_prompt


def test_local_openai_user_simulator_bypasses_litellm(monkeypatch):
    import tau_bench.envs.user as user_module

    requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="hello"))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(user_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        user_module,
        "completion",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("LiteLLM must not be called")),
    )

    user = user_module.LLMUserSimulationEnv(
        model="local-model", provider="openai", api_base="http://localhost:8011/v1", seed=1234
    )
    assert user.reset("retail instruction") == "hello"
    assert user.get_total_cost() == 0.0
    assert requests and all(request["seed"] == 1234 for request in requests)
