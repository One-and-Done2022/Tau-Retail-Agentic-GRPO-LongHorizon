"""veRL interaction adapter for τ-bench retail."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from verl.interactions.base import BaseInteraction

from src.envs.tau_bench_context import (
    CURRENT_ASSISTANT_CONTENT,
    CURRENT_TAU_ENV,
    CURRENT_TAU_STATE,
    make_initial_state,
)

logger = logging.getLogger(__name__)

FORBIDDEN_TEMPLATE_TOKENS = ["</tool_response>", "<tool_response>"]


def record_assistant_content(content: str) -> None:
    """Record the current assistant turn for the following tool call."""
    CURRENT_ASSISTANT_CONTENT.set(content)


def _has_forbidden_token(content: str) -> bool:
    return bool(content) and any(token in content for token in FORBIDDEN_TEMPLATE_TOKENS)


def _extract_latest_assistant_content(messages: list[dict]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message.get("content", "") or ""
    return ""


class TauBenchInteraction(BaseInteraction):
    """Drive one isolated Retail environment per rollout."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.env_name = config.get("env_name", "retail")
        if self.env_name != "retail":
            raise ValueError(f"Only the retail environment is supported, got: {self.env_name}")

        self.user_strategy = config.get("user_strategy", "llm")
        self.user_model = config.get("user_model", "Qwen/Qwen2.5-72B-Instruct-AWQ")
        self.user_provider = config.get("user_provider", "openai")
        self.user_base_url = config.get("user_base_url", "http://localhost:8001/v1")
        self.task_split = config.get("task_split", "train")
        self.max_turns = int(config.get("max_turns", 30))
        self.user_seed: Optional[int] = config.get("user_seed")

        reward_mode = config.get("reward_mode", "binary")
        if reward_mode != "binary":
            raise ValueError(f"Only binary Retail outcome reward is supported, got: {reward_mode}")
        self.reward_mode = reward_mode

        self.step_credit_config: dict[str, Any] = dict(config.get("step_credit", {}) or {})
        self.step_credit_enabled = bool(self.step_credit_config.get("enabled", False))
        if self.step_credit_enabled and self.task_split != "train":
            raise ValueError("Goal-derived step credit may only be enabled for task_split='train'")

        self._instance_dict: dict[str, dict] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        task_id: int = 0,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid.uuid4())

        from tau_bench.envs import get_env

        task_id_int = int(task_id)
        user_seed = None if self.user_seed is None else int(self.user_seed) + task_id_int * 1000
        env = get_env(
            env_name="retail",
            user_strategy=self.user_strategy,
            user_model=self.user_model,
            user_provider=self.user_provider,
            user_api_base=self.user_base_url,
            task_split=self.task_split,
            task_index=task_id_int,
            user_seed=user_seed,
        )
        env.reset(task_index=task_id_int)

        state = make_initial_state(task_id_int)
        if self.step_credit_enabled:
            from src.envs.retail_cs_grpo import RetailProgressTracker

            state["cs_tracker"] = RetailProgressTracker.from_env(env, task_split=self.task_split)
            state["cs_config"] = dict(self.step_credit_config)

        CURRENT_TAU_ENV.set(env)
        CURRENT_TAU_STATE.set(state)
        self._instance_dict[instance_id] = {"env": env, "state": state}
        logger.debug(
            "[start_interaction] instance=%s task_id=%s env_id=%s",
            instance_id[:8],
            task_id_int,
            id(env),
        )
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            raise RuntimeError(
                "TauBenchInteraction.generate_response called without a matching "
                f"start_interaction: {instance_id}"
            )

        env = entry["env"]
        state = entry["state"]
        CURRENT_TAU_ENV.set(env)
        CURRENT_TAU_STATE.set(state)

        assistant_content = _extract_latest_assistant_content(messages)
        if _has_forbidden_token(assistant_content):
            state["contaminated"] = True
            state["done"] = True
            return (
                True,
                "",
                0.0,
                {
                    "contaminated": True,
                    "reason": "forbidden_template_token",
                    "total_reward": state["total_reward"],
                    "num_turns": state["num_user_turns"] + state["num_tool_calls"],
                    "task_id": state["task_id"],
                },
            )

        from tau_bench.types import Action, RESPOND_ACTION_NAME

        try:
            step_res = env.step(
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": assistant_content})
            )
        except Exception as exc:
            logger.warning(
                "[generate_response] env.step(RESPOND) failed for task %s: %s: %s",
                state["task_id"],
                type(exc).__name__,
                exc,
            )
            state["done"] = True
            return (
                True,
                "",
                0.0,
                {
                    "error": "respond_exception",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "task_id": state["task_id"],
                },
            )

        inc_reward = float(getattr(step_res, "reward", 0.0))
        is_done = bool(getattr(step_res, "done", False))
        user_reply = str(getattr(step_res, "observation", ""))
        state["last_user_message"] = user_reply
        state["total_reward"] += inc_reward
        state["num_user_turns"] += 1
        total_turns = state["num_user_turns"] + state["num_tool_calls"]

        if is_done or total_turns >= self.max_turns:
            state["done"] = True
            final_score = 1.0 if state["total_reward"] >= 1.0 else 0.0
            return (
                True,
                "",
                final_score,
                {
                    "total_reward": state["total_reward"],
                    "num_turns": total_turns,
                    "num_tool_calls": state["num_tool_calls"],
                    "num_user_turns": state["num_user_turns"],
                    "task_id": state["task_id"],
                    "reason": "done" if is_done else "max_turns",
                    "reward_mode": self.reward_mode,
                    "transferred_to_human": state.get("transferred_to_human", False),
                },
            )

        return (
            False,
            user_reply,
            0.0,
            {
                "turn": total_turns,
                "num_tool_calls": state["num_tool_calls"],
                "task_id": state["task_id"],
            },
        )

    async def calculate_score(self, instance_id: str, **kwargs) -> dict[str, float]:
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            return {"score": 0.0, "outcome_score": 0.0, "process_score": 0.0}
        outcome = 1.0 if entry["state"]["total_reward"] >= 1.0 else 0.0
        return {"score": outcome, "outcome_score": outcome, "process_score": 0.0}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
