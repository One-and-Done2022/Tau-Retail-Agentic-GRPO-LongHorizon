from __future__ import annotations

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import compute_grpo_cs_advantage, compute_grpo_outcome_advantage


def _batch():
    mask = torch.tensor(
        [
            [1, 1, 0, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    rewards = torch.zeros_like(mask, dtype=torch.float32)
    rewards[1, 6] = 1.0
    uid = np.array(["same-prompt", "same-prompt"], dtype=object)
    records = np.empty(2, dtype=object)
    records[0] = [
        {"start": 0, "end": 2, "reward": 0.0},
        {"start": 4, "end": 6, "reward": -1.0},
    ]
    records[1] = [
        {"start": 0, "end": 2, "reward": 2.0},
        {"start": 4, "end": 6, "reward": 1.0},
    ]
    return rewards, mask, uid, records


def test_cs_advantage_changes_only_matching_tool_spans():
    rewards, mask, uid, records = _batch()
    vanilla, _ = compute_grpo_outcome_advantage(rewards, mask, uid)
    cs, returns = compute_grpo_cs_advantage(
        rewards,
        mask,
        uid,
        step_records=records,
        config={"cs_grpo": {"beta": 0.5}},
    )

    assert torch.equal(cs[:, 6], vanilla[:, 6])  # ordinary assistant token
    assert torch.equal(cs[:, 2:4], torch.zeros_like(cs[:, 2:4]))  # tool observation
    assert torch.equal(cs[:, 7:], torch.zeros_like(cs[:, 7:]))  # padding
    assert not torch.equal(cs[:, 0:2], vanilla[:, 0:2])
    assert not torch.equal(cs[:, 4:6], vanilla[:, 4:6])
    assert torch.equal(returns, cs)


def test_beta_zero_is_exact_vanilla_and_needs_no_metadata():
    rewards, mask, uid, _ = _batch()
    vanilla_adv, vanilla_returns = compute_grpo_outcome_advantage(rewards, mask, uid)
    cs_adv, cs_returns = compute_grpo_cs_advantage(
        rewards,
        mask,
        uid,
        step_records=None,
        config={"cs_grpo": {"beta": 0.0}},
    )
    assert torch.equal(cs_adv, vanilla_adv)
    assert torch.equal(cs_returns, vanilla_returns)


def test_singleton_step_ordinal_adds_zero():
    rewards, mask, uid, records = _batch()
    records[1] = records[1][:1]
    vanilla, _ = compute_grpo_outcome_advantage(rewards, mask, uid)
    cs, _ = compute_grpo_cs_advantage(
        rewards,
        mask,
        uid,
        step_records=records,
        config={"cs_grpo": {"beta": 1.0}},
    )
    assert torch.equal(cs[0, 4:6], vanilla[0, 4:6])


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([[{"start": -1, "end": 1, "reward": 0.0}], []], "Out-of-bounds"),
        ([[{"start": 0, "end": 10, "reward": 0.0}], []], "Out-of-bounds"),
        ([[{"start": 1, "end": 3, "reward": 0.0}], []], "masked observation"),
        (
            [[{"start": 0, "end": 2, "reward": 0.0}, {"start": 1, "end": 2, "reward": 1.0}], []],
            "Overlapping",
        ),
    ],
)
def test_invalid_spans_fail_loudly(records, message):
    rewards, mask, uid, _ = _batch()
    with pytest.raises(ValueError, match=message):
        compute_grpo_cs_advantage(
            rewards,
            mask,
            uid,
            step_records=np.array(records, dtype=object),
            config={"cs_grpo": {"beta": 1.0}},
        )


def test_zero_variance_step_group_adds_zero():
    rewards, mask, uid, records = _batch()
    for row in range(2):
        records[row] = [{"start": 0, "end": 2, "reward": 1.0}]
    vanilla, _ = compute_grpo_outcome_advantage(rewards, mask, uid)
    cs, _ = compute_grpo_cs_advantage(
        rewards,
        mask,
        uid,
        step_records=records,
        config={"cs_grpo": {"beta": 1.0}},
    )
    assert torch.allclose(cs, vanilla)
