#!/usr/bin/env python3
"""Orchestrate NexusBench natural-scene construction.

Main flow:
1) HVSG construction (includes part-mask/part-graph stages)
2) Part-level QA generation
3) Final `Natural_Scenes.json` building
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NexusBench construction orchestrator")
    p.add_argument(
        "--step",
        choices=["all", "hvsg", "part_graph", "qa_generate", "build_json"],
        default="all",
        help="Run all or one step",
    )

    p.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("draft"),
        help="Root containing NexusBench-pipeline and natural_vidor legacy code",
    )

    p.add_argument(
        "--qa-input-dir",
        type=Path,
        default=Path("NexusBench/construction/qa_pair/generate_qa_piar/output_json"),
        help="Per-video QA JSON directory for final merge",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("draft/huggingface/qae_triplet/Natural_Scenes.json"),
        help="Final Natural_Scenes output path",
    )
    p.add_argument(
        "--video-root",
        type=Path,
        default=Path("draft/huggingface/video/natural"),
        help="Directory containing source videos (repo-relative by default)",
    )
    p.add_argument(
        "--video-url-prefix",
        type=str,
        default="/NexusBench/video/natural",
        help="URL or relative prefix for video_url field",
    )
    p.add_argument("--time-unit", choices=["frame", "second"], default="frame")

    # Optional passthrough script paths for old steps
    p.add_argument(
        "--part-graph-runner",
        type=Path,
        default=Path("run_hvsg_local.sh"),
        help="Local shell runner for HVSG construction",
    )
    p.add_argument(
        "--qa-generator",
        type=Path,
        default=Path("NexusBench/construction/qa_pair/generate_qa_piar/generate_qa_batch_ns.py"),
        help="Legacy script for part-level QA generation",
    )

    return p.parse_args()


def resolve_repo_path(path_like: Path) -> Path:
    if path_like.is_absolute():
        return path_like
    return REPO_ROOT / path_like


def step_hvsg(args: argparse.Namespace) -> None:
    runner = resolve_repo_path(args.part_graph_runner)
    if not runner.exists():
        print(f"[WARN] HVSG runner not found: {runner}")
        return
    run(["bash", str(runner), "all"])


def step_qa_generate(args: argparse.Namespace) -> None:
    script = resolve_repo_path(args.qa_generator)
    if not script.exists():
        print(f"[WARN] QA generator not found: {script}")
        return
    run([sys.executable, str(script)])


def step_build_json(args: argparse.Namespace) -> None:
    script = HERE / "build_natural_scenes.py"
    qa_input_dir = resolve_repo_path(args.qa_input_dir)
    output_json = resolve_repo_path(args.output_json)
    video_root = resolve_repo_path(args.video_root)
    run(
        [
            sys.executable,
            str(script),
            "--input-dir",
            str(qa_input_dir),
            "--output-json",
            str(output_json),
            "--video-root",
            str(video_root),
            "--video-url-prefix",
            args.video_url_prefix,
            "--time-unit",
            args.time_unit,
        ]
    )


def main() -> None:
    args = parse_args()
    if args.step in {"all", "hvsg", "part_graph"}:
        step_hvsg(args)
    if args.step in {"all", "qa_generate"}:
        step_qa_generate(args)
    if args.step in {"all", "build_json"}:
        step_build_json(args)


if __name__ == "__main__":
    main()
