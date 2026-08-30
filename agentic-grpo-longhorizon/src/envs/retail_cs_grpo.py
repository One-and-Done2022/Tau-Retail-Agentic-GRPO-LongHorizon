"""Environment-side state and constraint credit for tau-bench retail.

The hidden task actions are used only to build a progress evaluator for retail train
rollouts.  Neither goal values nor constraint diagnostics are added to model messages.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


RETAIL_WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
)
ORDER_WRITE_TOOLS = RETAIL_WRITE_TOOLS - {"modify_user_address"}
PENDING_WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
    }
)
DELIVERED_WRITE_TOOLS = frozenset(
    {"exchange_delivered_order_items", "return_delivered_order_items"}
)
ITEM_WRITE_TOOLS = frozenset(
    {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
        "return_delivered_order_items",
    }
)
VARIANT_WRITE_TOOLS = frozenset(
    {"exchange_delivered_order_items", "modify_pending_order_items"}
)
PAYMENT_WRITE_TOOLS = frozenset(
    {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
    }
)

_MISSING = object()
_AFFIRMATIVE_RE = re.compile(
    r"\b(yes|yep|yeah|confirm(?:ed)?|proceed|go ahead|please do|sounds good|that's correct|that is correct)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(no|nope|not|never|stop|don't|do not|cancel that|wait)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetailStateSnapshot:
    matched_goal_leaves: int
    total_goal_leaves: int
    digest: str

    @property
    def progress(self) -> float:
        if self.total_goal_leaves == 0:
            return 0.0
        return self.matched_goal_leaves / self.total_goal_leaves


@dataclass(frozen=True)
class ConstraintViolation:
    code: str
    cost: float = 1.0


@dataclass(frozen=True)
class RetailStepCredit:
    reward: float
    progress_delta: float
    constraint_cost: float
    tool_error: bool
    redundant_write: bool
    violations: tuple[ConstraintViolation, ...]
    before: RetailStateSnapshot
    after: RetailStateSnapshot

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["violations"] = [asdict(item) for item in self.violations]
        return value


def has_explicit_confirmation(message: str | None) -> bool:
    """Return true only for an affirmative latest user turn without negation."""
    if not message or _NEGATION_RE.search(message):
        return False
    return bool(_AFFIRMATIVE_RE.search(message))


def _iter_leaf_differences(before: Any, target: Any, path: tuple[Any, ...] = ()) -> Iterable[tuple[Any, ...]]:
    if isinstance(before, dict) and isinstance(target, dict):
        for key in sorted(set(before) | set(target)):
            yield from _iter_leaf_differences(before.get(key, _MISSING), target.get(key, _MISSING), path + (key,))
        return
    if isinstance(before, list) and isinstance(target, list):
        for index in range(max(len(before), len(target))):
            left = before[index] if index < len(before) else _MISSING
            right = target[index] if index < len(target) else _MISSING
            yield from _iter_leaf_differences(left, right, path + (index,))
        return
    if before != target:
        yield path


def _lookup(root: Any, path: tuple[Any, ...]) -> Any:
    value = root
    for part in path:
        if isinstance(part, int):
            if not isinstance(value, list) or part >= len(value):
                return _MISSING
            value = value[part]
        else:
            if not isinstance(value, dict) or part not in value:
                return _MISSING
            value = value[part]
    return value


def _jsonable(value: Any) -> Any:
    if value is _MISSING:
        return {"__missing__": True}
    return value


class RetailProgressTracker:
    """Measure current state against the isolated goal-state delta."""

    def __init__(
        self,
        goal_values: Mapping[tuple[Any, ...], Any],
        goal_action_errors: Iterable[str] = (),
    ):
        self._goal_values = {path: copy.deepcopy(value) for path, value in goal_values.items()}
        self.goal_action_errors = tuple(goal_action_errors)

    @classmethod
    def from_env(cls, env: Any, task_split: str) -> "RetailProgressTracker":
        if task_split != "train":
            raise ValueError("Retail goal-derived step credit is restricted to task_split='train'")

        initial_data = copy.deepcopy(env.data)
        goal_data = copy.deepcopy(env.data)
        goal_action_errors = []
        for action in env.task.actions:
            if action.name == "respond" or action.name in env.terminate_tools:
                continue
            tool = env.tools_map.get(action.name)
            if tool is None:
                raise ValueError(f"Goal action uses unknown retail tool: {action.name}")
            observation = tool.invoke(data=goal_data, **action.kwargs)
            if isinstance(observation, str) and observation.startswith("Error:"):
                # Match tau-bench's native evaluator exactly: failed target actions
                # are retained as no-op transitions while later actions still run.
                goal_action_errors.append(f"{action.name}: {observation}")

        paths = tuple(_iter_leaf_differences(initial_data, goal_data))
        return cls(
            {path: _lookup(goal_data, path) for path in paths},
            goal_action_errors=goal_action_errors,
        )

    @property
    def total_goal_leaves(self) -> int:
        return len(self._goal_values)

    def snapshot(self, data: Mapping[str, Any]) -> RetailStateSnapshot:
        values = [_lookup(data, path) for path in self._goal_values]
        matched = sum(value == target for value, target in zip(values, self._goal_values.values()))
        encoded = json.dumps([_jsonable(value) for value in values], sort_keys=True, ensure_ascii=False, default=str)
        return RetailStateSnapshot(
            matched_goal_leaves=matched,
            total_goal_leaves=len(values),
            digest=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        )


def _successful_writes(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(state.get("successful_writes", []))


def _is_redundant_write(tool_name: str, parameters: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    signature = json.dumps(parameters, sort_keys=True, ensure_ascii=False, default=str)
    return any(item.get("tool") == tool_name and item.get("signature") == signature for item in _successful_writes(state))


def _payment_diff(tool_name: str, parameters: Mapping[str, Any], order: Mapping[str, Any], data: Mapping[str, Any]) -> float | None:
    if tool_name == "modify_pending_order_payment":
        history = order.get("payment_history", [])
        return float(history[0]["amount"]) if history else None
    if tool_name not in VARIANT_WRITE_TOOLS:
        return None
    old_ids = parameters.get("item_ids", [])
    new_ids = parameters.get("new_item_ids", [])
    if len(old_ids) != len(new_ids):
        return None
    diff = 0.0
    for old_id, new_id in zip(old_ids, new_ids):
        old_item = next((item for item in order.get("items", []) if item.get("item_id") == old_id), None)
        if old_item is None:
            return None
        variant = data.get("products", {}).get(old_item.get("product_id"), {}).get("variants", {}).get(new_id)
        if variant is None:
            return None
        diff += float(variant["price"]) - float(old_item["price"])
    return round(diff, 2)


def check_retail_constraints(
    tool_name: str,
    parameters: Mapping[str, Any],
    data: Mapping[str, Any],
    state: Mapping[str, Any],
    rule_weights: Mapping[str, float] | None = None,
) -> tuple[ConstraintViolation, ...]:
    """Check generic policy/schema/state preconditions for a retail write."""
    if tool_name not in RETAIL_WRITE_TOOLS:
        return ()

    weights = rule_weights or {}
    violations: list[ConstraintViolation] = []
    seen: set[str] = set()

    def add(code: str) -> None:
        if code not in seen:
            seen.add(code)
            violations.append(ConstraintViolation(code=code, cost=float(weights.get(code, 1.0))))

    authenticated_user_id = state.get("authenticated_user_id")
    if not authenticated_user_id:
        add("identity_not_authenticated")
    if not has_explicit_confirmation(state.get("last_user_message")):
        add("missing_explicit_confirmation")

    order = None
    order_id = parameters.get("order_id")
    owner_id = None
    if tool_name in ORDER_WRITE_TOOLS:
        order = data.get("orders", {}).get(order_id)
        if order is None:
            add("order_not_found")
        else:
            owner_id = order.get("user_id")
            if authenticated_user_id and authenticated_user_id != owner_id:
                add("authenticated_user_not_order_owner")
            if order_id not in state.get("read_order_ids", []):
                add("order_not_read")
            if owner_id not in state.get("read_user_ids", []):
                add("user_not_read")
            if tool_name in PENDING_WRITE_TOOLS and order.get("status") != "pending":
                add("order_not_pending")
            if tool_name in DELIVERED_WRITE_TOOLS and order.get("status") != "delivered":
                add("order_not_delivered")
    else:
        owner_id = parameters.get("user_id")
        if owner_id not in data.get("users", {}):
            add("user_not_found")
        if authenticated_user_id and authenticated_user_id != owner_id:
            add("authenticated_user_mismatch")
        if owner_id not in state.get("read_user_ids", []):
            add("user_not_read")

    if _is_redundant_write(tool_name, parameters, state):
        add("repeated_one_shot_write")
    elif order_id is not None and any(
        item.get("tool") == tool_name and item.get("order_id") == order_id for item in _successful_writes(state)
    ):
        add("repeated_one_shot_write")

    if order is not None and tool_name in ITEM_WRITE_TOOLS:
        requested = list(parameters.get("item_ids", []))
        available = Counter(item.get("item_id") for item in order.get("items", []))
        if not requested or any(count > available[item_id] for item_id, count in Counter(requested).items()):
            add("order_item_mismatch")

    if order is not None and tool_name in VARIANT_WRITE_TOOLS:
        old_ids = list(parameters.get("item_ids", []))
        new_ids = list(parameters.get("new_item_ids", []))
        if len(old_ids) != len(new_ids):
            add("variant_count_mismatch")
        else:
            for old_id, new_id in zip(old_ids, new_ids):
                old_item = next((item for item in order.get("items", []) if item.get("item_id") == old_id), None)
                if old_item is None:
                    continue
                variant = data.get("products", {}).get(old_item.get("product_id"), {}).get("variants", {}).get(new_id)
                if variant is None:
                    add("variant_product_mismatch")
                elif not variant.get("available", False):
                    add("variant_unavailable")

    payment = None
    payment_id = parameters.get("payment_method_id")
    if order is not None and tool_name in PAYMENT_WRITE_TOOLS:
        payment = data.get("users", {}).get(owner_id, {}).get("payment_methods", {}).get(payment_id)
        if payment is None:
            add("payment_method_not_owned")
        if tool_name == "return_delivered_order_items" and payment is not None:
            history = order.get("payment_history", [])
            original_id = history[0].get("payment_method_id") if history else None
            if payment.get("source") != "gift_card" and payment_id != original_id:
                add("invalid_return_refund_method")

    if order is not None and payment is not None and payment.get("source") == "gift_card":
        required = _payment_diff(tool_name, parameters, order, data)
        if required is not None and required > 0 and float(payment.get("balance", 0.0)) < required:
            add("insufficient_gift_card_balance")

    return tuple(violations)


def compute_step_credit(
    tracker: RetailProgressTracker,
    before: RetailStateSnapshot,
    after: RetailStateSnapshot,
    violations: tuple[ConstraintViolation, ...],
    tool_error: bool,
    redundant_write: bool,
    config: Mapping[str, Any],
) -> RetailStepCredit:
    progress_delta = after.progress - before.progress
    constraint_cost = sum(item.cost for item in violations)
    reward = (
        float(config.get("progress_weight", 1.0)) * progress_delta
        - float(config.get("constraint_weight", 1.0)) * constraint_cost
        - float(config.get("error_penalty", 1.0)) * float(tool_error)
        - float(config.get("redundant_write_penalty", 1.0)) * float(redundant_write)
    )
    return RetailStepCredit(
        reward=float(reward),
        progress_delta=float(progress_delta),
        constraint_cost=float(constraint_cost),
        tool_error=bool(tool_error),
        redundant_write=bool(redundant_write),
        violations=violations,
        before=before,
        after=after,
    )


def observe_retail_tool_result(
    tool_name: str,
    parameters: Mapping[str, Any],
    observation: str,
    state: dict[str, Any],
) -> None:
    """Update only policy-visible evidence accumulated from successful tool calls."""
    if observation.startswith("Error:"):
        return
    if tool_name in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
        state["authenticated_user_id"] = observation.strip()
    elif tool_name == "get_order_details":
        order_id = parameters.get("order_id")
        if order_id is not None and order_id not in state["read_order_ids"]:
            state["read_order_ids"].append(order_id)
    elif tool_name == "get_user_details":
        user_id = parameters.get("user_id")
        if user_id is not None and user_id not in state["read_user_ids"]:
            state["read_user_ids"].append(user_id)

    if tool_name in RETAIL_WRITE_TOOLS:
        state["successful_writes"].append(
            {
                "tool": tool_name,
                "order_id": parameters.get("order_id"),
                "signature": json.dumps(parameters, sort_keys=True, ensure_ascii=False, default=str),
            }
        )


def redundant_retail_write(tool_name: str, parameters: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    return tool_name in RETAIL_WRITE_TOOLS and _is_redundant_write(tool_name, parameters, state)
