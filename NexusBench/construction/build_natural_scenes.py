#!/usr/bin/env python3
"""Build NexusBench Natural_Scenes.json from per-video QA JSON files.

Input QA format per file:
[
  {"question": "...", "answer": "...", "type": "Type 4: ..."},
  ...
]

Output format (list):
{
  "sample_id": int,
  "video_id": str,
  "video_url": str,
  "question": str,
  "answer_complete": str,
  "answer_summary": str,
  "answer_evidence": str,
  "evidence": {"temporal": {"<T1>": ["00:00.00", "00:01.50"]}, "spatial": {...}},
  "type": str (optional if missing in input)
}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

TIME_PATTERN = re.compile(r"<(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)>")
SPATIAL_PATTERN = re.compile(r"\[([^\[\]]+?)\s+(\d+)\]")


def sec_to_mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mm = int(seconds // 60)
    ss = seconds % 60
    return f"{mm:02d}:{ss:05.2f}"


def frame_to_mmss(frame_value: str, fps: float) -> str:
    frame = float(frame_value)
    # Existing scripts use (frame - 1) / fps
    return sec_to_mmss((frame - 1) / fps)


def get_video_fps(video_file: Path, default_fps: float) -> float:
    if cv2 is None or not video_file.exists():
        return default_fps
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return default_fps
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 0 else default_fps


def split_answer(answer: str) -> Tuple[str, str]:
    if "Evidence:" in answer:
        left, right = answer.split("Evidence:", 1)
        return left.strip(), right.strip()
    if "<" in answer:
        idx = answer.find("<")
        return answer[:idx].strip(), answer[idx:].strip()
    return answer.strip(), ""


def parse_temporal(answer_complete: str, fps: float, as_frames: bool) -> Dict[str, List[str]]:
    temporal: Dict[str, List[str]] = {}
    for i, m in enumerate(TIME_PATTERN.finditer(answer_complete), start=1):
        s_raw, e_raw = m.group(1), m.group(2)
        if as_frames:
            s, e = frame_to_mmss(s_raw, fps), frame_to_mmss(e_raw, fps)
        else:
            s, e = sec_to_mmss(float(s_raw)), sec_to_mmss(float(e_raw))
        temporal[f"<T{i}>"] = [s, e]
    return temporal


def parse_spatial(answer_complete: str) -> Dict[str, str]:
    spatial: Dict[str, str] = {}
    for obj_name, obj_id in SPATIAL_PATTERN.findall(answer_complete):
        spatial[obj_name.strip()] = obj_id
    return spatial


def rewrite_time_spans(answer_text: str, fps: float, as_frames: bool) -> str:
    def repl(match: re.Match[str]) -> str:
        s_raw, e_raw = match.group(1), match.group(2)
        if as_frames:
            s, e = frame_to_mmss(s_raw, fps), frame_to_mmss(e_raw, fps)
        else:
            s, e = sec_to_mmss(float(s_raw)), sec_to_mmss(float(e_raw))
        return f"<{s}-{e}>"

    return TIME_PATTERN.sub(repl, answer_text)


def iter_qa_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        yield path


def build_dataset(
    input_dir: Path,
    output_json: Path,
    video_root: Path,
    video_url_prefix: str,
    default_fps: float,
    as_frames: bool,
) -> int:
    rows: List[dict] = []
    sample_id = 1

    for qa_file in iter_qa_files(input_dir):
        video_id = qa_file.stem
        video_file = video_root / f"{video_id}.mp4"
        fps = get_video_fps(video_file, default_fps)

        data = json.loads(qa_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue

        for item in data:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            qa_type = item.get("type")
            if not q or not a:
                continue

            answer_complete = rewrite_time_spans(a, fps=fps, as_frames=as_frames)
            answer_summary, answer_evidence = split_answer(answer_complete)

            row = {
                "sample_id": sample_id,
                "video_id": video_id,
                "video_url": f"{video_url_prefix.rstrip('/')}/{video_id}.mp4",
                "question": q,
                "answer_complete": answer_complete,
                "answer_summary": answer_summary,
                "answer_evidence": answer_evidence,
                "evidence": {
                    "temporal": parse_temporal(a, fps=fps, as_frames=as_frames),
                    "spatial": parse_spatial(a),
                },
            }
            if qa_type is not None:
                row["type"] = qa_type

            rows.append(row)
            sample_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Natural_Scenes style JSON from per-video QA files")
    p.add_argument("--input-dir", type=Path, required=True, help="Folder containing per-video QA JSON files")
    p.add_argument("--output-json", type=Path, required=True, help="Output Natural_Scenes.json path")
    p.add_argument("--video-root", type=Path, required=True, help="Directory containing <video_id>.mp4")
    p.add_argument("--video-url-prefix", type=str, required=True, help="Prefix used to build video_url")
    p.add_argument("--default-fps", type=float, default=30.0, help="Fallback FPS when video cannot be opened")
    p.add_argument(
        "--time-unit",
        choices=["frame", "second"],
        default="frame",
        help="Interpret <a-b> in answer as frame index or second value",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    num = build_dataset(
        input_dir=args.input_dir,
        output_json=args.output_json,
        video_root=args.video_root,
        video_url_prefix=args.video_url_prefix,
        default_fps=args.default_fps,
        as_frames=(args.time_unit == "frame"),
    )
    print(f"Built {num} samples -> {args.output_json}")


if __name__ == "__main__":
    main()
