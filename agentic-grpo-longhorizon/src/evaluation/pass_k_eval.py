"""
pass^k 评测
对每个 task 独立采样 k 次,计算 pass^k (至少一次成功的比例)
同时统计 turn efficiency 和 tool call accuracy
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import json
import inspect
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.envs.tau_bench_wrapper import TauBenchWrapper, TrajectoryResult


def _get_tokenizer(policy_factory):
    """尝试从 policy 获取 model_name 并加载 tokenizer；失败则返回 None。"""
    try:
        policy = policy_factory()
        tokenizer_name = (
            getattr(policy, "tokenizer_name_or_path", None)
            or getattr(policy, "model_name", None)
        )
        if tokenizer_name:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            print(f"[pass_k_eval] Loaded tokenizer from {tokenizer_name}")
            return tok
    except Exception as e:
        print(f"[pass_k_eval] Failed to load tokenizer: {e}, falling back to char length")
    return None


def _make_token_counter(tokenizer):
    """返回一个 callable: text -> token count"""
    if tokenizer is not None:
        def _count(text: str) -> int:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                return len(text)
        return _count
    return len


@dataclass
class EvalReport:
    env_name: str
    num_tasks: int
    num_samples_per_task: int
    pass_at_1: float           # 任意一次成功
    pass_hat_1: float          # pass^1: 平均成功率
    pass_hat_4: float          # pass^4: 连续 4 次都成功的比例(稳定性)
    pass_hat_8: float
    avg_turns: float
    avg_tool_calls: float
    error_rate: float          # trajectory 异常中止的比例
    constraint_violation_rate: float
    high_risk_tool_error_rate: float
    tool_error_rate: float
    error_recovery_rate: float
    avg_latency_seconds: float
    avg_reasoning_tokens_per_turn: float
    max_reasoning_tokens_single_turn: int
    policy_seed_base: Optional[int]
    user_seed_base: Optional[int]
    seed_rule: Optional[str]
    per_task_results: list[dict]


def run_eval(
    wrapper: TauBenchWrapper,
    policy_factory,                    # callable -> policy instance (thread-safe)
    num_tasks: Optional[int] = None,
    num_samples_per_task: int = 4,
    max_turns: int = 30,
    num_workers: int = 4,
    output_dir: str = "experiments/baseline_airline_7B_user",
    task_ids: Optional[list[int]] = None,
    policy_seed_base: Optional[int] = None,
    user_seed_base: Optional[int] = None,
) -> EvalReport:
    """
    policy_factory: 每个 worker 线程自己 new 一个 policy,避免并发问题
    """
    if task_ids is None:
        if num_tasks is None:
            num_tasks = wrapper.get_num_tasks()
        task_ids = list(range(num_tasks))
    else:
        task_ids = list(task_ids)
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("task_ids must be a non-empty list of unique task IDs")
        available = wrapper.get_num_tasks()
        if min(task_ids) < 0 or max(task_ids) >= available:
            raise ValueError(f"task_ids out of bounds for {available} tasks: {task_ids}")
        num_tasks = len(task_ids)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 每个 (task_idx, sample_idx) 是一个独立 job
    jobs = [(t, s) for t in task_ids for s in range(num_samples_per_task)]
    results: dict[int, list[tuple[int, TrajectoryResult]]] = {t: [] for t in task_ids}
    
    def _run_one(task_idx: int, sample_idx: int) -> tuple[int, int, TrajectoryResult]:
        try:
            inspect.signature(policy_factory).bind(task_idx, sample_idx)
        except TypeError:
            policy = policy_factory()
        else:
            policy = policy_factory(task_idx, sample_idx)
        user_seed = (
            None
            if user_seed_base is None
            else int(user_seed_base) + int(task_idx) * 1000 + int(sample_idx)
        )
        traj = wrapper.run_single_task(
            task_idx, policy, max_turns=max_turns, user_seed=user_seed
        )
        return task_idx, sample_idx, traj
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_run_one, t, s) for t, s in jobs]
        for fut in tqdm(as_completed(futures), total=len(jobs), desc="Eval"):
            task_idx, sample_idx, traj = fut.result()
            results[task_idx].append((sample_idx, traj))
    
    # 计算 pass^k
    # pass^k 定义: 对同一个 task 采样 n 次,估计"连续 k 次都成功"的概率
    # 用 unbiased estimator (HumanEval 里的 pass@k 公式)
    import numpy as np
    
    def pass_at_k(n: int, c: int, k: int) -> float:
        """n: 总采样数, c: 成功数, k: pass^k 的 k"""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    # 尝试加载 tokenizer（用于精确统计 assistant content tokens）
    tokenizer = _get_tokenizer(policy_factory)
    count_tokens = _make_token_counter(tokenizer)

    per_task = []
    pass_1_list, pass_4_list, pass_8_list = [], [], []
    pass_at_1_list = []  # 任意一次成功
    all_turns, all_tool_calls, all_errors, all_latencies = [], [], [], []
    total_write_calls = total_violating_writes = total_high_risk_errors = 0
    total_tool_errors = total_tool_calls = total_recovered_errors = 0
    all_reasoning_tokens = []

    for t in task_ids:
        sampled_trajs = sorted(results[t], key=lambda item: item[0])
        trajs = [traj for _, traj in sampled_trajs]
        n = len(trajs)
        c = sum(1 for tr in trajs if tr.success)

        p1 = pass_at_k(n, c, 1)
        p4 = pass_at_k(n, c, 4) if n >= 4 else None
        p8 = pass_at_k(n, c, 8) if n >= 8 else None

        pass_1_list.append(p1)
        pass_at_1_list.append(1.0 if c > 0 else 0.0)
        if p4 is not None: pass_4_list.append(p4)
        if p8 is not None: pass_8_list.append(p8)

        traj_dicts = []
        for sample_id, tr in sampled_trajs:
            all_turns.append(tr.num_turns)
            all_tool_calls.append(tr.num_tool_calls)
            all_errors.append(1.0 if tr.error else 0.0)
            all_latencies.append(tr.latency_seconds)
            total_write_calls += tr.high_risk_tool_calls
            total_violating_writes += tr.constraint_violation_count
            total_high_risk_errors += tr.high_risk_tool_errors
            total_tool_errors += tr.tool_error_count
            total_tool_calls += tr.num_tool_calls
            total_recovered_errors += tr.recovered_tool_errors

            # 计算每轮 assistant turn 的 content token 数
            per_turn_assistant_content_tokens = []
            for msg in tr.raw_messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "") or ""
                    per_turn_assistant_content_tokens.append(count_tokens(content))
            all_reasoning_tokens.extend(per_turn_assistant_content_tokens)

            traj_dict = tr.to_dict()
            traj_dict["sample_id"] = sample_id
            traj_dict["policy_seed"] = (
                None
                if policy_seed_base is None
                else int(policy_seed_base) + int(t) * 1000 + int(sample_id)
            )
            traj_dict["per_turn_assistant_content_tokens"] = per_turn_assistant_content_tokens
            traj_dicts.append(traj_dict)

        per_task.append({
            "task_id": t,
            "success_count": c,
            "total_samples": n,
            "pass^1": p1,
            "avg_turns": np.mean([tr.num_turns for tr in trajs]),
            "trajectories": traj_dicts,
        })
    
    report = EvalReport(
        env_name=wrapper.env_name,
        num_tasks=num_tasks,
        num_samples_per_task=num_samples_per_task,
        pass_at_1=float(np.mean(pass_at_1_list)),
        pass_hat_1=float(np.mean(pass_1_list)),
        pass_hat_4=float(np.mean(pass_4_list)) if pass_4_list else 0.0,
        pass_hat_8=float(np.mean(pass_8_list)) if pass_8_list else 0.0,
        avg_turns=float(np.mean(all_turns)),
        avg_tool_calls=float(np.mean(all_tool_calls)),
        error_rate=float(np.mean(all_errors)),
        constraint_violation_rate=total_violating_writes / max(total_write_calls, 1),
        high_risk_tool_error_rate=total_high_risk_errors / max(total_write_calls, 1),
        tool_error_rate=total_tool_errors / max(total_tool_calls, 1),
        error_recovery_rate=total_recovered_errors / max(total_tool_errors, 1),
        avg_latency_seconds=float(np.mean(all_latencies)),
        avg_reasoning_tokens_per_turn=(
            float(np.mean(all_reasoning_tokens)) if all_reasoning_tokens else 0.0
        ),
        max_reasoning_tokens_single_turn=(
            int(max(all_reasoning_tokens)) if all_reasoning_tokens else 0
        ),
        policy_seed_base=policy_seed_base,
        user_seed_base=user_seed_base,
        seed_rule=(
            "base + task_id * 1000 + sample_id"
            if policy_seed_base is not None or user_seed_base is not None
            else None
        ),
        per_task_results=per_task,
    )
    
    # 保存
    with open(output_dir / "eval_report.json", "w") as f:
        #json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)
    
    # 打印摘要
    print(f"\n=== Eval Report: {wrapper.env_name} ===")
    print(f"Tasks: {num_tasks} × Samples: {num_samples_per_task}")
    print(f"pass@1 (any success): {report.pass_at_1:.3f}")
    print(f"pass^1 (avg success): {report.pass_hat_1:.3f}")
    print(f"pass^4 (stability):   {report.pass_hat_4:.3f}")
    print(f"pass^8 (stability):   {report.pass_hat_8:.3f}")
    print(f"Avg turns:       {report.avg_turns:.2f}")
    print(f"Avg tool calls:  {report.avg_tool_calls:.2f}")
    print(f"Error rate:      {report.error_rate:.3f}")
    print(f"Constraint violation rate: {report.constraint_violation_rate:.3f}")
    print(f"High-risk tool error rate: {report.high_risk_tool_error_rate:.3f}")
    print(f"Tool error recovery rate:  {report.error_recovery_rate:.3f}")
    print(f"Avg assistant content tokens/turn: {report.avg_reasoning_tokens_per_turn:.2f}")
    
    return report
