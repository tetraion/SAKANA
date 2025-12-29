#!/usr/bin/env python3
"""
Create a cut EDL from an SRT subtitle file by removing gaps.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Tuple


def _srt_timestamp_to_seconds(ts: str) -> float:
    # Format: HH:MM:SS,mmm
    parts = ts.strip().replace(",", ".").split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid SRT timestamp: {ts}")
    h, m, s = parts
    return (int(h) * 3600) + (int(m) * 60) + float(s)


def _seconds_to_tc(seconds: float, fps: float) -> str:
    total_frames = int(seconds * fps + 0.5)
    frames = total_frames % int(fps)
    secs = total_frames // int(fps)
    s = secs % 60
    m = (secs // 60) % 60
    h = secs // 3600
    return f"{h:02d}:{m:02d}:{s:02d}:{frames:02d}"


def _tc_to_seconds(tc: str, fps: float) -> float:
    parts = tc.split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid timecode: {tc}")
    h, m, s, f = (int(p) for p in parts)
    return (h * 3600) + (m * 60) + s + (f / fps)


def _parse_srt(path: str) -> List[dict]:
    segments: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # index line
        if line.isdigit() and i + 1 < len(lines):
            time_line = lines[i + 1].strip()
            if "-->" in time_line:
                start_s, end_s = [s.strip() for s in time_line.split("-->")]
                start = _srt_timestamp_to_seconds(start_s)
                end = _srt_timestamp_to_seconds(end_s)
                text_lines: List[str] = []
                i += 2
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i])
                    i += 1
                text = "\n".join(text_lines).strip()
                if end > start and text:
                    segments.append({"start": start, "end": end, "text": text})
                continue
        i += 1

    return segments


def _merge_segments(segments: Iterable[Tuple[float, float]], gap_threshold: float) -> List[Tuple[float, float]]:
    sorted_segments = sorted(segments, key=lambda x: x[0])
    if not sorted_segments:
        return []
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = sorted_segments[0]
    for start, end in sorted_segments[1:]:
        gap = start - cur_end
        if gap <= gap_threshold:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _build_gaps(merged: List[Tuple[float, float]], gap_threshold: float) -> List[Tuple[float, float]]:
    gaps: List[Tuple[float, float]] = []
    for (prev_start, prev_end), (next_start, next_end) in zip(merged, merged[1:]):
        gap = next_start - prev_end
        if gap > gap_threshold:
            gaps.append((prev_end, next_start))
    return gaps


def _map_time(t: float, gaps: List[Tuple[float, float]]) -> float:
    removed = 0.0
    for gap_start, gap_end in gaps:
        if t >= gap_end:
            removed += gap_end - gap_start
            continue
        if t > gap_start:
            return gap_start - removed
        break
    return t - removed


def _write_srt(segments: List[dict], output_path: str, gaps: List[Tuple[float, float]]) -> None:
    lines = []
    idx = 1
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        text = str(seg["text"]).strip()
        if not text or end <= start:
            continue
        new_start = _map_time(start, gaps)
        new_end = _map_time(end, gaps)
        if new_end <= new_start:
            continue
        lines.append(str(idx))
        lines.append(f"{_format_srt_ts(new_start)} --> {_format_srt_ts(new_end)}")
        lines.append(text)
        lines.append("")
        idx += 1
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _format_srt_ts(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_edl(
    segments: List[Tuple[float, float]],
    fps: float,
    title: str,
    src_tc_base: str,
    rec_tc_base: str,
    reel: str,
) -> List[str]:
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    src_base_sec = _tc_to_seconds(src_tc_base, fps)
    rec_base_sec = _tc_to_seconds(rec_tc_base, fps)

    record_cursor = 0.0
    for idx, (start, end) in enumerate(segments, start=1):
        duration = end - start
        if duration <= 0:
            continue
        src_in = _seconds_to_tc(src_base_sec + start, fps)
        src_out = _seconds_to_tc(src_base_sec + end, fps)
        rec_in = _seconds_to_tc(rec_base_sec + record_cursor, fps)
        rec_out = _seconds_to_tc(rec_base_sec + record_cursor + duration, fps)
        line = f"{idx:03d}  {reel}      V     C        {src_in} {src_out} {rec_in} {rec_out}"
        lines.append(line)
        lines.append("")
        record_cursor += duration

    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a cut EDL from SRT by removing gaps")
    parser.add_argument("srt", help="入力SRTパス")
    parser.add_argument("edl", help="出力EDLパス")
    parser.add_argument("--fps", type=float, default=30.0, help="フレームレート")
    parser.add_argument("--gap", type=float, default=0.5, help="この秒数以上のギャップをカット")
    parser.add_argument("--src-tc-base", default="00:00:00:00", help="ソースTC基準")
    parser.add_argument("--rec-tc-base", default="00:00:00:00", help="レコードTC基準")
    parser.add_argument("--reel", default="001", help="EDLのリール名")
    parser.add_argument("--srt-out", help="カット後の字幕SRT出力パス（省略時はEDLと同名で作成）")
    args = parser.parse_args(argv)

    segments = _parse_srt(args.srt)
    if not segments:
        print("No subtitle segments found.")
        return 1

    merged = _merge_segments([(s["start"], s["end"]) for s in segments], gap_threshold=args.gap)
    gaps = _build_gaps(merged, gap_threshold=args.gap)
    title = os.path.splitext(os.path.basename(args.srt))[0]
    edl_lines = _segments_to_edl(
        merged,
        fps=args.fps,
        title=title,
        src_tc_base=args.src_tc_base,
        rec_tc_base=args.rec_tc_base,
        reel=args.reel,
    )

    with open(args.edl, "w", encoding="utf-8") as f:
        f.write("\n".join(edl_lines) + "\n")

    srt_out = args.srt_out
    if srt_out is None:
        base = os.path.splitext(args.edl)[0]
        srt_out = f"{base}_cut.srt"
    _write_srt(segments, srt_out, gaps)
    print(f"✓ Written SRT to {srt_out}")

    print(f"✓ Written EDL to {args.edl} (segments={len(merged)}, gap>={args.gap}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
