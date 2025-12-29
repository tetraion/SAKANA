#!/usr/bin/env python3
"""
Silence detector: find silence intervals to cut.

ffmpeg の silencedetect フィルタを使って、無音区間の開始・終了を抽出し、
JSON で出力する簡易ツール。カット候補の一覧を作る用途を想定。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from typing import List, Sequence

# 調整ポイント
SILENCE_THRESHOLD_DB = -35  # これより小さい音を無音とみなす
SILENCE_MIN_DURATION = 0.8  # 秒。ここより短い無音は無視
DEFAULT_FPS = 30.0


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def detect_silence(input_path: str) -> List[dict]:
    """
    ffmpeg silencedetect で無音区間を抽出し、{start, end, duration} のリストを返す。
    """
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-af",
        f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB:d={SILENCE_MIN_DURATION}",
        "-f",
        "null",
        "-",
    ]
    result = _run(cmd)

    # silencedetect 出力をパース
    # 例: [silencedetect @ ...] silence_start: 12.3456
    #     [silencedetect @ ...] silence_end: 20.000 | silence_duration: 7.6544
    starts = []
    intervals: List[dict] = []
    start_re = re.compile(r"silence_start:\s*([0-9.]+)")
    end_re = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")

    for line in (result.stderr or "").splitlines():
        m_start = start_re.search(line)
        if m_start:
            starts.append(float(m_start.group(1)))
            continue
        m_end = end_re.search(line)
        if m_end and starts:
            start_val = starts.pop(0)
            end_val = float(m_end.group(1))
            dur = float(m_end.group(2))
            intervals.append({"start": start_val, "end": end_val, "duration": dur})

    return intervals


def _seconds_to_tc(seconds: float, fps: float) -> str:
    """秒を HH:MM:SS:FF のタイムコードに変換（フレームは四捨五入）。"""
    total_frames = int(math.floor(seconds * fps + 0.5))
    frames = total_frames % int(fps)
    secs = total_frames // int(fps)
    s = secs % 60
    m = (secs // 60) % 60
    h = secs // 3600
    return f"{h:02d}:{m:02d}:{s:02d}:{frames:02d}"


def _tc_to_seconds(tc: str, fps: float) -> float:
    """HH:MM:SS:FF を秒に変換。"""
    parts = tc.split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid timecode: {tc}")
    h, m, s, f = (int(p) for p in parts)
    return (h * 3600) + (m * 60) + s + (f / fps)


def silences_to_markers_rows(silences: List[dict], fps: float) -> List[List[str]]:
    """
    silenceリストをDaVinci Resolve向けマーカー行（Name, Start TC, End TC, Color, Notes）に変換。
    """
    rows: List[List[str]] = []
    for idx, item in enumerate(silences, start=1):
        start = float(item["start"])
        end = float(item["end"])
        dur = end - start
        start_tc = _seconds_to_tc(start, fps)
        end_tc = _seconds_to_tc(end, fps)
        name = f"Silence {idx}"
        note = f"duration={dur:.2f}s"
        color = "Blue"
        rows.append([name, start_tc, end_tc, color, note])
    return rows


def silences_to_edl_lines(
    silences: List[dict],
    fps: float,
    title: str,
    tc_base: str,
    offset_seconds: float,
) -> List[str]:
    """
    ResolveのTimeline Marker EDL形式に合わせて出力。
    例: イベント行 + `|C:ResolveColorBlue |M:マーカー名 |D:1` の行。
    """
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    base_seconds = _tc_to_seconds(tc_base, fps)

    for idx, item in enumerate(silences, start=1):
        start = float(item["start"]) + offset_seconds
        end = float(item["end"])
        start_tc = _seconds_to_tc(base_seconds + start, fps)
        end_tc = _seconds_to_tc(base_seconds + start + (1.0 / fps), fps)
        event = f"{idx:03d}  001      V     C        {start_tc} {end_tc} {start_tc} {end_tc}  "
        color = "ResolveColorBlue"
        name = f"Silence {idx}"
        duration_frames = max(1, int(round((end - start) * fps)))
        marker_line = f" |C:{color} |M:{name} |D:{duration_frames}"
        lines.append(event)
        lines.append(marker_line)
        lines.append("")

    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect silence intervals and output EDL for Resolve")
    parser.add_argument("input", help="入力の動画/音声ファイル")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"タイムコード計算に使うFPS（デフォルト: {DEFAULT_FPS}）",
    )
    parser.add_argument(
        "--edl",
        help="EDL出力パス（省略時は <input>_silence.edl）",
    )
    parser.add_argument(
        "--tc-base",
        default="00:00:00:00",
        help="EDLの基準タイムコード（デフォルト: 00:00:00:00）",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="EDL出力時の秒オフセット（タイムラインに置いたIn点補正用）",
    )
    parser.add_argument(
        "--json",
        help="JSON出力パス（指定時のみ出力）",
    )
    args = parser.parse_args(argv)

    try:
        intervals = detect_silence(args.input)
    except subprocess.CalledProcessError as exc:
        print(f"✗ ffmpeg failed: {exc.stderr or exc.stdout or exc}")
        return 1

    base_without_ext, _ = os.path.splitext(args.input)
    edl_out = args.edl or f"{base_without_ext}_silence.edl"

    edl_lines = silences_to_edl_lines(
        intervals,
        fps=args.fps,
        title=os.path.basename(base_without_ext),
        tc_base=args.tc_base,
        offset_seconds=args.offset,
    )
    with open(edl_out, "w", encoding="utf-8") as f:
        f.write("\n".join(edl_lines) + "\n")
    print(f"✓ Written EDL to {edl_out} (events={len(intervals)}, fps={args.fps})")

    if args.json:
        json_out = args.json
        payload = {
            "input": args.input,
            "fps": args.fps,
            "threshold_db": SILENCE_THRESHOLD_DB,
            "min_duration": SILENCE_MIN_DURATION,
            "silences": intervals,
        }
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        print(f"✓ Written JSON to {json_out} (events={len(intervals)}, fps={args.fps})")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
