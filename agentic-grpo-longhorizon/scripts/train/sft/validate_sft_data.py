"""Validate successful tau-bench trajectories before SFT training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.sft_dataset import build_supervised_example


class ReplayUser:
    def reset(self, instruction=None):
        return "replay"

    def step(self, content):
        return "continue"

    def get_total_cost(self):
        return 0.0


def _message_hash(messages: list[dict]) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def replay_reward(env_name: str, task_split: str, task_id: int, messages: list[dict]) -> float:
    from tau_bench.envs import get_env
    from tau_bench.types import Action, RESPOND_ACTION_NAME

    env = get_env(env_name, "human", "dummy", task_split, task_index=task_id)
    env.user = ReplayUser()
    env.reset(task_index=task_id)
    response = None
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            for tool_call in tool_calls:
                function = tool_call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                response = env.step(Action(name=function["name"], kwargs=arguments))
                if response.done:
                    return float(response.reward)
        elif message.get("content"):
            response = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]}))

    if response is None or not response.done:
        response = env.step(
            Action(name="transfer_to_human_agents", kwargs={"summary": "offline SFT replay"})
        )
    return float(response.reward)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--env", choices=("retail",), default="retail")
    parser.add_argument("--task-split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--split-json")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    seen_ids = unseen_ids = None
    if args.split_json:
        split = json.loads(Path(args.split_json).read_text())
        seen_ids = set(split["seen_task_ids"])
        unseen_ids = set(split["unseen_task_ids"])
        if seen_ids & unseen_ids:
            raise ValueError("seen/unseen task IDs overlap")

    hashes: set[str] = set()
    task_ids: set[int] = set()
    lengths: list[int] = []
    label_lengths: list[int] = []
    duplicate_count = replay_failures = render_failures = contaminated_count = 0
    for index, row in enumerate(rows):
        if not row.get("success") or float(row.get("reward", 0.0)) < 1.0:
            raise ValueError(f"row {index} is not a successful native-reward trajectory")
        if row.get("was_contaminated_from_turn") is not None or row.get("was_contaminated"):
            contaminated_count += 1
            continue
        task_id = int(row["task_id"])
        task_ids.add(task_id)
        if seen_ids is not None and (task_id not in seen_ids or task_id in unseen_ids):
            raise ValueError(f"training row {index} task_id={task_id} violates seen/unseen split")
        messages = row["messages"]
        digest = _message_hash(messages)
        if digest in hashes:
            duplicate_count += 1
        hashes.add(digest)
        replayed = replay_reward(args.env, args.task_split, task_id, messages)
        if replayed < 1.0:
            replay_failures += 1
        example = build_supervised_example(messages, tokenizer, max_length=args.max_length)
        if example is None:
            render_failures += 1
        else:
            lengths.append(example["n_total_tokens"])
            label_lengths.append(example["n_label_tokens"])

    report = {
        "env": args.env,
        "task_split": args.task_split,
        "rows": len(rows),
        "unique_tasks": len(task_ids),
        "duplicate_trajectories": duplicate_count,
        "contaminated_trajectories": contaminated_count,
        "replay_failures": replay_failures,
        "render_failures": render_failures,
        "max_tokens": max(lengths, default=0),
        "mean_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "mean_label_tokens": sum(label_lengths) / len(label_lengths) if label_lengths else 0.0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if contaminated_count or replay_failures or render_failures or not rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
