#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_retail_confirmatory import read_json, summarize_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate one retail confirmatory evaluation candidate")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--experiment-root", default="experiments/retail_confirmatory")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/eval/retail/confirmatory_manifest.json"),
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    run_dir = args.run_dir.resolve()
    output = run_dir / "gate_result.json"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing gate result: {output}")
    try:
        relative_run_dir = run_dir.relative_to(project_dir)
    except ValueError as exc:
        raise SystemExit(f"Run directory must be inside the project: {run_dir}") from exc
    expected_root = Path(args.experiment_root).parts
    if (
        len(relative_run_dir.parts) < len(expected_root) + 3
        or relative_run_dir.parts[: len(expected_root)] != expected_root
    ):
        raise SystemExit(f"Unexpected confirmatory run directory: {relative_run_dir}")

    candidate = run_dir.parent.name
    run_id = run_dir.name
    manifest_path = project_dir / args.manifest
    manifest = read_json(manifest_path)
    if read_json(run_dir / "frozen_manifest.json") != manifest:
        raise SystemExit(f"Frozen evaluation manifest mismatch: {run_dir}")

    splits = {}
    for split_index, split in enumerate(("seen", "unseen_dev")):
        report_path = run_dir / split / "eval_report.json"
        summary = summarize_evaluation(
            read_json(report_path),
            report_path.relative_to(project_dir),
            manifest,
            split,
            bootstrap_seed=60260825 + split_index,
            bootstrap_replicates=100,
        )
        splits[split] = {
            "source": summary["source"],
            "trajectory_count": summary["trajectory_count"],
            "success_count": summary["success_count"],
            "metrics": summary["metrics"],
            "error_classes": summary["error_classes"],
            "system_error_count": summary["system_error_count"],
            "contaminated_trajectory_count": summary["contaminated_trajectory_count"],
        }

    result = {
        "passed": True,
        "label": "retail-confirmatory-fixed-budget-engineering-study",
        "candidate": candidate,
        "run_id": run_id,
        "trajectory_count": sum(split["trajectory_count"] for split in splits.values()),
        "splits": splits,
        "failures": [],
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
