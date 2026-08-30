"""τ-bench retail 环境封装，为评测和 GRPO rollout 提供统一 API。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
import json
import os
from collections import Counter
import time

# τ-bench 原生 API
from tau_bench.envs import get_env
from tau_bench.types import EnvRunResult, Action, RESPOND_ACTION_NAME


@dataclass
class TrajectoryStep:
    """一轮交互的完整记录"""
    turn_idx: int
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_name: Optional[str] = None


@dataclass
class TrajectoryResult:
    """一条完整轨迹的结果,用于评测和 RL 训练"""
    task_id: int
    success: bool           # outcome reward (0/1)
    reward: float           # τ-bench 原生 reward
    num_turns: int
    num_tool_calls: int
    steps: list[TrajectoryStep] = field(default_factory=list)
    raw_messages: list[dict] = field(default_factory=list)  # OpenAI 格式
    error: Optional[str] = None  # 如果 trajectory 异常中止
    # [污染标记] 截断发生时的 turn 索引；None 表示未被截断污染
    was_contaminated_from_turn: Optional[int] = None
    constraint_violation_count: int = 0
    constraint_violations: dict[str, int] = field(default_factory=dict)
    high_risk_tool_calls: int = 0
    high_risk_tool_errors: int = 0
    tool_error_count: int = 0
    recovered_tool_errors: int = 0
    latency_seconds: float = 0.0
    user_seed: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "reward": self.reward,
            "num_turns": self.num_turns,
            "num_tool_calls": self.num_tool_calls,
            "raw_messages": self.raw_messages,
            "error": self.error,
            "was_contaminated_from_turn": self.was_contaminated_from_turn,
            "constraint_violation_count": self.constraint_violation_count,
            "constraint_violations": self.constraint_violations,
            "high_risk_tool_calls": self.high_risk_tool_calls,
            "high_risk_tool_errors": self.high_risk_tool_errors,
            "tool_error_count": self.tool_error_count,
            "recovered_tool_errors": self.recovered_tool_errors,
            "latency_seconds": self.latency_seconds,
            "user_seed": self.user_seed,
        }


class TauBenchWrapper:
    """
    对 τ-bench 的薄封装,提供两个关键能力:
    1. run_single_task: 给定 policy 和 task_id,跑一条轨迹
    2. batch_eval: 批量评测,用于 baseline 测试
    
    policy 需要实现 __call__(messages: list[dict]) -> dict 接口,
    返回 OpenAI 格式的 assistant message (可能包含 tool_calls).
    """

    def __init__(
        self,
        env_name: str = "retail",
        user_strategy: str = "llm",       # τ-bench 默认用 llm simulator
        user_model: str = "gpt-4o",       # 后面会改成本地 Qwen-72B
        user_provider: str = "openai",    # 或 "anthropic" / "local"
        user_base_url: Optional[str] = None,  # user simulator 请求的 vLLM 地址
        task_split: str = "test",
        task_index: Optional[int] = None,
        user_seed: Optional[int] = None,
    ):
        if env_name != "retail":
            raise ValueError(f"Only the retail environment is supported, got: {env_name}")
        self.env_name = env_name
        self.user_strategy = user_strategy
        self.user_model = user_model
        self.user_provider = user_provider
        self.user_base_url = user_base_url
        self.task_split = task_split
        self.task_index = task_index
        self.user_seed = user_seed

    def _make_env(self, task_idx: int, user_seed: Optional[int] = None):
        """为每个 task 创建一个独立的 env 实例(τ-bench 的设计)"""
        return get_env(
            env_name=self.env_name,
            user_strategy=self.user_strategy,
            user_model=self.user_model,
            user_provider=self.user_provider,
            user_api_base=self.user_base_url,
            task_split=self.task_split,
            task_index=task_idx,
            user_seed=self.user_seed if user_seed is None else user_seed,
        )

    def get_num_tasks(self) -> int:
        # Counting tasks must not initialize an LLM user simulator or make an API request.
        env = get_env(
            env_name=self.env_name,
            user_strategy="human",
            user_model="dummy",
            user_provider=None,
            task_split=self.task_split,
            task_index=0,
        )
        return len(env.tasks)

    def run_single_task(
        self,
        task_idx: int,
        policy,
        max_turns: int = 30,
        user_seed: Optional[int] = None,
    ) -> TrajectoryResult:
        effective_user_seed = self.user_seed if user_seed is None else user_seed
        env = self._make_env(task_idx, user_seed=effective_user_seed)
        # 把环境可用工具注册给 policy，模型才知道能调哪些工具
        if hasattr(policy, "set_tools"):
            policy.set_tools(env.tools_info)
        obs_res = env.reset(task_index=task_idx)

        # [Fix: policy 状态泄漏] 每个 task 开始前重置截断标记
        if hasattr(policy, "was_truncated"):
            policy.was_truncated = False

        # tau-bench reset 返回 EnvResetResponse(observation=str, info=EnvInfo)
        # 初始 observation 就是用户的第一条消息
        # 
        date_context = (
            "# Current Date Context\n"
            "The current date is 2024-05-15 (Wednesday). "
            "When users mention dates without specifying the year, "
            "always assume they refer to 2024. "
            "All product orders and exchanges should use 2024 dates unless explicitly stated otherwise."
        )
        # Match tau-bench's official ToolCallingAgent: the environment policy is the
        # authoritative system prompt. Date grounding is an additional local context.
        system_content = f"{env.wiki.rstrip()}\n\n{date_context}"
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(obs_res.observation)},
        ]
        steps: list[TrajectoryStep] = []
        total_reward = 0.0
        done = False
        error_msg = None
        turn_idx = 0
        tool_call_count = 0
        recent_tool_calls: list[tuple[str, str]] = []
        was_contaminated_from_turn: Optional[int] = None
        started_at = time.perf_counter()
        from src.envs.tau_bench_context import make_initial_state

        audit_state = make_initial_state(task_idx)
        audit_state["last_user_message"] = str(obs_res.observation)
        violation_counts: Counter[str] = Counter()
        constraint_violation_count = 0
        high_risk_tool_calls = 0
        high_risk_tool_errors = 0
        tool_error_count = 0
        recovered_tool_errors = 0
        pending_tool_errors = 0
        try:
            while not done and turn_idx < max_turns:
                assistant_msg = policy(messages)
                messages.append(assistant_msg)
                
                # [污染检测] policy 一旦截断，标记当前 turn 为污染起点
                if hasattr(policy, "was_truncated") and policy.was_truncated and was_contaminated_from_turn is None:
                    was_contaminated_from_turn = turn_idx
                
                steps.append(TrajectoryStep(
                    turn_idx=turn_idx,
                    role="assistant",
                    content=assistant_msg.get("content", "") or "",
                    tool_calls=assistant_msg.get("tool_calls"),
                ))

                # 判定是否有 tool_calls
                tcs = assistant_msg.get("tool_calls")
                if tcs:
                    # [Loop detection] 检测完全重复的工具调用
                    for tc in tcs:
                        call_sig = (tc["function"]["name"], tc["function"]["arguments"])
                        recent_tool_calls.append(call_sig)
                    if len(recent_tool_calls) > 20:
                        recent_tool_calls = recent_tool_calls[-20:]
                    call_counts = Counter(recent_tool_calls)
                    if any(c >= 3 for c in call_counts.values()):
                        error_msg = "Loop detected: same tool call repeated 3+ times"
                        break  # break while

                    for tc in tcs:
                        tool_call_count += 1
                        # tau-bench 的 step 需要 Action 对象，不是原始 dict
                        action = Action(
                            name=tc["function"]["name"],
                            kwargs=json.loads(tc["function"]["arguments"]),
                        )
                        is_high_risk = False
                        if audit_state is not None:
                            from src.envs.retail_cs_grpo import RETAIL_WRITE_TOOLS, check_retail_constraints

                            is_high_risk = action.name in RETAIL_WRITE_TOOLS
                            if is_high_risk:
                                high_risk_tool_calls += 1
                                violations = check_retail_constraints(
                                    action.name, action.kwargs, env.data, audit_state
                                )
                                if violations:
                                    constraint_violation_count += 1
                                    violation_counts.update(item.code for item in violations)
                        tool_result = env.step(action)
                        obs_content = tool_result.observation if hasattr(tool_result, 'observation') else str(tool_result)
                        tool_failed = str(obs_content).startswith("Error:")
                        if tool_failed:
                            tool_error_count += 1
                            pending_tool_errors += 1
                            if is_high_risk:
                                high_risk_tool_errors += 1
                        elif pending_tool_errors:
                            recovered_tool_errors += pending_tool_errors
                            pending_tool_errors = 0
                        if audit_state is not None:
                            from src.envs.retail_cs_grpo import observe_retail_tool_result

                            observe_retail_tool_result(action.name, action.kwargs, str(obs_content), audit_state)
                        
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.get("id", f"call_{tool_call_count}"),
                            "name": tc["function"]["name"],
                            "content": str(obs_content),
                        }
                        messages.append(tool_msg)
                        steps.append(TrajectoryStep(
                            turn_idx=turn_idx, role="tool",
                            content=tool_msg["content"], tool_name=tc["function"]["name"],
                        ))
                        total_reward += getattr(tool_result, 'reward', 0.0)
                        if getattr(tool_result, 'done', False):
                            done = True
                            break
                else:
                    # tau-bench 没有 step_user 方法，用户交互也是通过 step(Action(name="respond", ...))
                    action = Action(
                        name=RESPOND_ACTION_NAME,
                        kwargs={"content": assistant_msg.get("content", "")},
                    )
                    user_obs = env.step(action)
                    if getattr(user_obs, 'done', False):
                        done = True
                        total_reward += getattr(user_obs, 'reward', 0.0)
                    else:
                        obs_str = getattr(user_obs, 'observation', str(user_obs))
                        user_msg = {"role": "user", "content": obs_str}
                        if audit_state is not None:
                            audit_state["last_user_message"] = obs_str
                        messages.append(user_msg)
                        steps.append(TrajectoryStep(
                            turn_idx=turn_idx, role="user", content=obs_str,
                        ))
                turn_idx += 1
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        success = total_reward >= 1.0
        return TrajectoryResult(
            task_id=task_idx, success=success, reward=total_reward,
            num_turns=turn_idx, num_tool_calls=tool_call_count,
            steps=steps, raw_messages=messages, error=error_msg,
            was_contaminated_from_turn=was_contaminated_from_turn,
            constraint_violation_count=constraint_violation_count,
            constraint_violations=dict(violation_counts),
            high_risk_tool_calls=high_risk_tool_calls,
            high_risk_tool_errors=high_risk_tool_errors,
            tool_error_count=tool_error_count,
            recovered_tool_errors=recovered_tool_errors,
            latency_seconds=time.perf_counter() - started_at,
            user_seed=effective_user_seed,
        )
