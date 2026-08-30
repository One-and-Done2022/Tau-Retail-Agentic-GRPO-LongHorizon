"""
TauBenchToolBase + τ-bench retail tool 的静态子类。

设计要点(见 design doc §3.1,已根据 review 修订):
- 16 个独立子类,veRL 按 class_name 实例化时一个 schema 对应一个类
- 静态定义在本文件中,避免 cloudpickle 序列化动态类的坑(Ray/FSDP 多进程场景)
- 启动时有一致性校验: verify_tool_classes_match_env() 检查静态类名与 env.tools_info 完全对齐
- Tool 本身 stateless, env 通过 contextvar 从当前 asyncio task 获取

关键修订(相对初稿):
1. 动态 type() 生成 → 静态 class 定义(pickle-safe)
2. "no env in context" 从返回错误字符串 → raise RuntimeError(fail loud)
3. Tool error 格式 "[TOOL_ERROR] XXX" → "Error: XXX"(对齐 τ-bench 原生)
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from src.envs.tau_bench_context import CURRENT_TAU_ENV, CURRENT_TAU_STATE, CURRENT_ASSISTANT_CONTENT

logger = logging.getLogger(__name__)


# ============================================================================
# 基类
# ============================================================================

class TauBenchToolBase(BaseTool):
    """
    τ-bench tool 的基类。14 个子类都继承它,不重写任何方法。
    self.name 由 BaseTool.__init__ 从 tool_schema.function.name 设置,
    Tool.execute 用 self.name 构造 τ-bench Action。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """veRL 每次 tool call 都会新建一个 instance_id,create 是 no-op。"""
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """
        1. 从 contextvar 拿当前 trajectory 的 env(Interaction.start_interaction 设置)
        2. 调 env.step(Action(tool_name, parameters))
        3. observation 回传给模型,inc_reward 累计到 state

        默认 step-level reward 为 0.0；仅当 retail interaction 显式开启
        step_credit 时返回环境侧信用信号。终局 outcome 仍由 Interaction 提供。
        """
        env = CURRENT_TAU_ENV.get()
        state = CURRENT_TAU_STATE.get()

        # 【修订 1】Fail loud: 宁可整个 batch 崩,也不让带毒 trajectory 进 GRPO
        if env is None or state is None:
            raise RuntimeError(
                f"[CRITICAL] TauBench env/state missing from contextvar for tool "
                f"'{self.name}'. This means Interaction.start_interaction did not "
                f"run in the same asyncio task as this Tool.execute call. "
                f"Check veRL ToolAgentLoop wiring (expected: single asyncio task "
                f"per trajectory, contextvar fork on asyncio.create_task)."
            )

        from tau_bench.types import Action

        cs_tracker = state.get("cs_tracker")
        cs_config = state.get("cs_config", {})
        before_snapshot = None
        violations = ()
        redundant_write = False
        if cs_tracker is not None:
            from src.envs.retail_cs_grpo import check_retail_constraints, redundant_retail_write

            before_snapshot = cs_tracker.snapshot(env.data)
            violations = check_retail_constraints(
                self.name,
                parameters,
                env.data,
                state,
                rule_weights=cs_config.get("rule_weights"),
            )
            redundant_write = redundant_retail_write(self.name, parameters, state)

        try:
            action = Action(name=self.name, kwargs=parameters)
            step_res = env.step(action)
        except Exception as e:
            # env.step 自身异常(参数 schema 不匹配 / env 已 done 被再 step / 等)
            # 【修订 3】Error 格式对齐 τ-bench 原生: 一句自然语言,无前缀
            # (τ-bench 自己的 tool error 样例: "Unknown action update_reservation_insurance")
            err_msg = f"Error: {type(e).__name__}: {e}"
            logger.warning(f"[Tool.execute] {self.name} raised: {err_msg}")
            step_reward = 0.0
            metadata = {"error": "env_step_exception", "tool": self.name, "detail": str(e)}
            if cs_tracker is not None:
                from src.envs.retail_cs_grpo import compute_step_credit

                after_snapshot = cs_tracker.snapshot(env.data)
                credit = compute_step_credit(
                    cs_tracker,
                    before_snapshot,
                    after_snapshot,
                    violations,
                    tool_error=True,
                    redundant_write=redundant_write,
                    config=cs_config,
                )
                step_reward = credit.reward
                metadata["cs_step_credit"] = credit.to_dict()
            return (
                ToolResponse(text=err_msg),
                step_reward,
                metadata,
            )

        obs = str(getattr(step_res, "observation", ""))
        inc_reward = float(getattr(step_res, "reward", 0.0))
        is_done = bool(getattr(step_res, "done", False))

        state["total_reward"] += inc_reward
        state["num_tool_calls"] += 1
        if is_done:
            state["done"] = True

        # 记录动作历史，供重复写操作检测使用。
        import json as _json
        assistant_content = CURRENT_ASSISTANT_CONTENT.get()
        state["action_history"].append({
            "tool": self.name,
            "parameters": parameters,
            "param_str": _json.dumps(parameters, sort_keys=True, ensure_ascii=False).lower(),
            "inc_reward": inc_reward,
            "done": is_done,
            "is_error": bool(obs and obs.startswith("Error:")),
            "content": assistant_content or "",
        })
        if self.name == "transfer_to_human_agents":
            state["transferred_to_human"] = True

        step_reward = 0.0
        metadata = {"inc_reward": inc_reward, "done": is_done, "tool": self.name}
        if cs_tracker is not None:
            from src.envs.retail_cs_grpo import compute_step_credit, observe_retail_tool_result

            after_snapshot = cs_tracker.snapshot(env.data)
            tool_error = bool(obs.startswith("Error:"))
            credit = compute_step_credit(
                cs_tracker,
                before_snapshot,
                after_snapshot,
                violations,
                tool_error=tool_error,
                redundant_write=redundant_write,
                config=cs_config,
            )
            step_reward = credit.reward
            metadata["cs_step_credit"] = credit.to_dict()
            observe_retail_tool_result(self.name, parameters, obs, state)

        return (
            ToolResponse(text=obs),
            step_reward,
            metadata,
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass


# ============================================================================
# Retail 静态工具子类（一个 schema 一个类）
#
# 为什么要静态定义而不是 type() 动态生成:
#   - cloudpickle(Ray/FSDP 跨进程传对象时用的)对 "动态 type() 创建且挂到模块"的类
#     序列化是 best-effort,在某些 Python / cloudpickle 版本组合下会 AttributeError
#   - 14 行样板代码换 100% pickle 可靠性,划算
#
# 若未来 τ-bench 加/减 tool,启动时 verify_tool_classes_match_env() 会报错,
# 提示手工更新本文件。
# ============================================================================

class TauBench_calculate_Tool(TauBenchToolBase): pass
class TauBench_get_user_details_Tool(TauBenchToolBase): pass
class TauBench_think_Tool(TauBenchToolBase): pass
class TauBench_transfer_to_human_agents_Tool(TauBenchToolBase): pass

class TauBench_cancel_pending_order_Tool(TauBenchToolBase): pass
class TauBench_exchange_delivered_order_items_Tool(TauBenchToolBase): pass
class TauBench_find_user_id_by_email_Tool(TauBenchToolBase): pass
class TauBench_find_user_id_by_name_zip_Tool(TauBenchToolBase): pass
class TauBench_get_order_details_Tool(TauBenchToolBase): pass
class TauBench_get_product_details_Tool(TauBenchToolBase): pass
class TauBench_list_all_product_types_Tool(TauBenchToolBase): pass
class TauBench_modify_pending_order_address_Tool(TauBenchToolBase): pass
class TauBench_modify_pending_order_items_Tool(TauBenchToolBase): pass
class TauBench_modify_pending_order_payment_Tool(TauBenchToolBase): pass
class TauBench_modify_user_address_Tool(TauBenchToolBase): pass
class TauBench_return_delivered_order_items_Tool(TauBenchToolBase): pass


# ============================================================================
# 一致性校验
# ============================================================================

RETAIL_TOOL_NAMES = [
    "calculate",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "think",
    "transfer_to_human_agents",
]

def get_tool_class_by_name(tool_name: str) -> type[TauBenchToolBase]:
    """根据 tool_name 查对应的类。scripts/train/grpo/gen_tool_config.py 用。"""
    cls_name = f"TauBench_{tool_name}_Tool"
    module = sys.modules[__name__]
    if not hasattr(module, cls_name):
        raise KeyError(
            f"No tool class {cls_name} defined in {__name__}. "
            f"If τ-bench added a new tool '{tool_name}', add a static class here."
        )
    return getattr(module, cls_name)


def verify_tool_classes_match_env(
    env_name: str = "retail",
    task_index: int = 0,
) -> list[str]:
    """
    验证指定环境的静态类和 env.tools_info 一一对应,不多不少。
    应在 scripts/train/grpo/gen_tool_config.py 启动时调用,不一致立刻报错。

    Returns:
        env.tools_info 里的 tool 名列表(调用方可用来生成 yaml)。
    Raises:
        AssertionError: 静态类名和 env tool 名不一致时。
    """
    if env_name != "retail":
        raise ValueError(f"Only the retail environment is supported, got: {env_name}")

    from tau_bench.envs import get_env

    env = get_env(
        env_name=env_name,
        user_strategy="human",
        user_model="dummy",
        user_provider=None,
        task_split="test",
        task_index=task_index,
    )
    env_tool_names = sorted(t["function"]["name"] for t in env.tools_info)
    static_tool_names = sorted(RETAIL_TOOL_NAMES)

    missing_in_static = set(env_tool_names) - set(static_tool_names)
    extra_in_static = set(static_tool_names) - set(env_tool_names)

    errors = []
    if missing_in_static:
        errors.append(
            f"Tools in env but no static class: {sorted(missing_in_static)}"
        )
    if extra_in_static:
        errors.append(
            f"Static classes but not in env: {sorted(extra_in_static)}"
        )

    if errors:
        raise AssertionError(
            "Static tool classes out of sync with τ-bench env:\n  " +
            "\n  ".join(errors) +
            f"\nUpdate RETAIL_TOOL_NAMES and static class definitions in "
            f"{__file__}."
        )

    # 还要验证每个静态类都是 TauBenchToolBase 的子类
    for tn in static_tool_names:
        cls = get_tool_class_by_name(tn)
        assert issubclass(cls, TauBenchToolBase), f"{cls} is not TauBenchToolBase subclass"

    logger.info(
        f"Tool class <-> env consistency check PASSED: {len(env_tool_names)} tools matched."
    )
    return env_tool_names
