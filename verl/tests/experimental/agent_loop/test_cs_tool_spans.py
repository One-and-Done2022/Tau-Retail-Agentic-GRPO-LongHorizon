from __future__ import annotations

import pytest

from verl.experimental.agent_loop.tool_step_spans import build_tool_step_record


def test_tool_step_record_keeps_response_relative_span_after_observation_tokens():
    record = build_tool_step_record(
        (7, 11),
        ["get_order_details"],
        [0.25],
        [{"cs_step_credit": {"progress_delta": 0.25}}],
    )
    assert record == {
        "start": 7,
        "end": 11,
        "reward": 0.25,
        "tools": ["get_order_details"],
        "components": [{"progress_delta": 0.25}],
    }


def test_parallel_calls_share_one_assistant_span_and_aggregate_reward():
    record = build_tool_step_record(
        (3, 8),
        ["tool_a", "tool_b"],
        [0.5, -0.25],
        [{}, {}],
    )
    assert record["start"] == 3
    assert record["end"] == 8
    assert record["reward"] == pytest.approx(0.25)


@pytest.mark.parametrize("span", [(-1, 2), (2, 2), (4, 3)])
def test_invalid_capture_span_fails(span):
    with pytest.raises(ValueError, match="Invalid"):
        build_tool_step_record(span, ["tool"], [0.0], [{}])


def test_capture_lengths_must_align():
    with pytest.raises(ValueError, match="identical lengths"):
        build_tool_step_record((0, 1), ["a", "b"], [0.0], [{}])
