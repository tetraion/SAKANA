#!/usr/bin/env python3
"""
Cut a video by keeping only SRT cue intervals (remove gaps).
"""

from __future__ import annotations

import argparse
import json
import os
import math
import re
import subprocess
from dataclasses import dataclass
from typing import List, Sequence


TIME_RE = re.compile(r"(?P<start>[\d:,]+)\s+-->\s+(?P<end>[\d:,]+)")


@dataclass
class Cue:
    start: float
    end: float
    text: str | None = None


def _run(cmd: Sequence[str]) -> None:
    subprocess.run(cmd, check=True)


def _time_to_seconds(t: str) -> float:
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _seconds_to_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    if s >= 60:
        m += 1
        s -= 60
    if m >= 60:
        h += 1
        m -= 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _seconds_to_frame(seconds: float, fps: float, mode: str) -> int:
    value = seconds * fps
    if mode == "floor":
        return int(math.floor(value))
    if mode == "ceil":
        return int(math.ceil(value))
    return int(round(value))


def _frames_to_tc(total_frames: int, fps: float) -> str:
    frames = total_frames % int(fps)
    secs = total_frames // int(fps)
    s = secs % 60
    m = (secs // 60) % 60
    h = secs // 3600
    return f"{h:02d}:{m:02d}:{s:02d}:{frames:02d}"


def _snap_to_fps(sec: float, fps: float, mode: str) -> float:
    frames = _seconds_to_frame(sec, fps, mode)
    return frames / fps


def _probe_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _parse_srt(path: str) -> List[Cue]:
    content = open(path, "r", encoding="utf-8").read()
    blocks = content.strip().split("\n\n")
    cues: List[Cue] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        m = TIME_RE.search(lines[1])
        if not m:
            continue
        start = _time_to_seconds(m.group("start"))
        end = _time_to_seconds(m.group("end"))
        if end > start:
            text = "\n".join(lines[2:]).strip()
            cues.append(Cue(start=start, end=end, text=text))
    return cues


def _load_words(path: str) -> List[tuple[float, float]]:
    data = json.loads(open(path, "r", encoding="utf-8").read())
    segments = None
    if isinstance(data, dict):
        if "segments" in data:
            segments = data["segments"]
        elif "transcription" in data:
            segments = data["transcription"]
    if segments is None:
        raise ValueError("Unsupported JSON format: expected top-level 'segments' or 'transcription'")
    words: List[tuple[float, float]] = []
    for seg in segments:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append((float(w["start"]), float(w["end"])))
    if not words:
        raise ValueError("No words found in JSON (missing 'words' array?)")
    return words


def _segments_from_words(words: List[tuple[float, float]], gap: float) -> List[Cue]:
    words = sorted(words)
    segments: List[Cue] = []
    start, end = words[0]
    for w_start, w_end in words[1:]:
        if w_start - end <= gap:
            end = max(end, w_end)
        else:
            segments.append(Cue(start=start, end=end))
            start, end = w_start, w_end
    segments.append(Cue(start=start, end=end))
    return segments


def _merge_close(cues: List[Cue], gap: float) -> List[Cue]:
    if not cues:
        return []
    cues = sorted(cues, key=lambda c: (c.start, c.end))
    merged: List[Cue] = [cues[0]]
    for cue in cues[1:]:
        prev = merged[-1]
        if cue.start <= prev.end + gap:
            prev.end = max(prev.end, cue.end)
        else:
            merged.append(Cue(start=cue.start, end=cue.end, text=cue.text))
    return merged


def _format_srt(cues: List[Cue], snap_fps: float | None = None) -> str:
    lines: List[str] = []
    for idx, cue in enumerate(cues, start=1):
        start = cue.start
        end = cue.end
        if snap_fps:
            start = _snap_to_fps(start, snap_fps, "floor")
            end = _snap_to_fps(end, snap_fps, "ceil")
        lines.append(str(idx))
        lines.append(f"{_seconds_to_time(start)} --> {_seconds_to_time(end)}")
        lines.append(cue.text or "")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_filter(cues: List[Cue]) -> str:
    parts: List[str] = []
    for idx, cue in enumerate(cues):
        start = f"{cue.start:.3f}"
        end = f"{cue.end:.3f}"
        parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{idx}]")
        parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{idx}]")
    concat_inputs = "".join([f"[v{idx}][a{idx}]" for idx in range(len(cues))])
    parts.append(f"{concat_inputs}concat=n={len(cues)}:v=1:a=1[outv][outa]")
    return ";".join(parts)


def _expand_cues(cues: List[Cue], pad: float) -> List[Cue]:
    if pad <= 0:
        return cues
    expanded: List[Cue] = []
    for cue in cues:
        expanded.append(
            Cue(start=max(0.0, cue.start - pad), end=cue.end + pad, text=cue.text)
        )
    return expanded


def _clamp_cues(cues: List[Cue], duration: float) -> List[Cue]:
    clamped: List[Cue] = []
    for cue in cues:
        start = max(0.0, cue.start)
        end = min(duration, cue.end)
        if end <= start:
            continue
        clamped.append(Cue(start=start, end=end, text=cue.text))
    return clamped


def _build_timeline_map(segments: List[Cue]) -> List[tuple[float, float, float]]:
    mapped: List[tuple[float, float, float]] = []
    cursor = 0.0
    for seg in segments:
        mapped.append((seg.start, seg.end, cursor))
        cursor += seg.end - seg.start
    return mapped


def _write_edl(
    segments: List[Cue],
    input_path: str,
    output_path: str,
    fps: float,
    reel_name: str | None,
) -> None:
    base = os.path.splitext(os.path.basename(input_path))[0]
    reel = reel_name or base
    lines = [f"TITLE: {base}", "FCM: NON-DROP FRAME", ""]
    cursor_seconds = 0.0
    duration = _probe_duration(input_path)
    max_frame = max(0, int(math.floor(duration * fps)) - 1)
    for idx, seg in enumerate(segments, start=1):
        src_in_f = _seconds_to_frame(seg.start, fps, "round")
        src_out_f = _seconds_to_frame(seg.end, fps, "round")
        if src_in_f > max_frame:
            continue
        if src_out_f > max_frame:
            src_out_f = max_frame
        if src_out_f <= src_in_f:
            src_out_f = src_in_f + 1
        rec_in_f = _seconds_to_frame(cursor_seconds, fps, "round")
        # Keep record length aligned to the (possibly clamped) source length.
        src_len_frames = max(1, src_out_f - src_in_f)
        cursor_seconds += src_len_frames / fps
        rec_out_f = _seconds_to_frame(cursor_seconds, fps, "round")
        if rec_out_f <= rec_in_f:
            rec_out_f = rec_in_f + 1
        src_in = _frames_to_tc(src_in_f, fps)
        src_out = _frames_to_tc(src_out_f, fps)
        rec_in = _frames_to_tc(rec_in_f, fps)
        rec_out = _frames_to_tc(rec_out_f, fps)
        line = f"{idx:03d}  {reel:<8} V     C        {src_in} {src_out} {rec_in} {rec_out}"
        lines.append(line)
        lines.append(f"* FROM CLIP NAME: {os.path.basename(input_path)}")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _map_cues_to_timeline(
    cues: List[Cue],
    segments: List[Cue],
    drop_unmapped: bool = False,
) -> List[Cue]:
    mapped = _build_timeline_map(segments)
    out: List[Cue] = []
    for cue in cues:
        overlaps: List[tuple[float, float]] = []
        for seg_start, seg_end, seg_offset in mapped:
            if cue.end <= seg_start or cue.start >= seg_end:
                continue
            start = max(cue.start, seg_start)
            end = min(cue.end, seg_end)
            new_start = seg_offset + (start - seg_start)
            new_end = seg_offset + (end - seg_start)
            overlaps.append((new_start, new_end))
        if overlaps:
            out.append(Cue(start=overlaps[0][0], end=overlaps[-1][1], text=cue.text))
        else:
            if drop_unmapped:
                continue
            nearest = min(mapped, key=lambda seg: abs(cue.start - seg[0]))
            seg_start, _, seg_offset = nearest
            duration = max(0.0, cue.end - cue.start)
            new_start = seg_offset + max(0.0, cue.start - seg_start)
            out.append(Cue(start=new_start, end=new_start + duration, text=cue.text))
    return out


def cut_video(
    srt_path: str,
    input_path: str,
    output_path: str,
    srt_out: str | None,
    edl_out: str | None,
    fps: float,
    reel_name: str | None,
    snap_srt: bool,
    merge_gap: float,
    pad: float,
    crf: int,
    preset: str,
    render_video: bool,
    words_json: str | None,
    words_gap: float | None,
) -> None:
    cues = _parse_srt(srt_path)
    duration = _probe_duration(input_path)
    if words_json:
        words = _load_words(words_json)
        base_segments = _segments_from_words(words, gap=words_gap if words_gap is not None else merge_gap)
    else:
        base_segments = cues
    padded = _expand_cues(base_segments, pad=pad)
    padded = _clamp_cues(padded, duration)
    segments = _merge_close(padded, gap=merge_gap)
    segments = _clamp_cues(segments, duration)
    if not segments:
        raise SystemExit("✗ No valid cues found in SRT.")

    if render_video:
        filter_complex = _build_filter(segments)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_path,
        ]
        _run(cmd)
    elif not (srt_out or edl_out):
        raise SystemExit("✗ --no-video requires --srt-out and/or --edl.")

    if srt_out:
        mapped = _map_cues_to_timeline(cues, segments, drop_unmapped=bool(words_json))
        with open(srt_out, "w", encoding="utf-8") as fh:
            fh.write(_format_srt(mapped, snap_fps=fps if snap_srt else None))
    if edl_out:
        _write_edl(segments, input_path=input_path, output_path=edl_out, fps=fps, reel_name=reel_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut video by keeping SRT cue intervals")
    parser.add_argument("srt", help="Input SRT path")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--srt-out", help="Output SRT aligned to cut video")
    parser.add_argument("--edl", help="Output EDL path (CMX3600)")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for EDL timecode")
    parser.add_argument("--reel-name", help="Reel name override for EDL")
    parser.add_argument("--snap-srt", action="store_true", help="Snap SRT times to FPS frame boundaries")
    parser.add_argument("--merge-gap", type=float, default=0.15, help="Merge gaps shorter than this (seconds)")
    parser.add_argument("--pad", type=float, default=0.0, help="Extend each cue by this many seconds on both ends")
    parser.add_argument("--crf", type=int, default=20, help="x264 CRF (lower is higher quality)")
    parser.add_argument("--preset", default="veryfast", help="x264 preset")
    parser.add_argument("--no-video", action="store_true", help="Skip mp4 output (EDL/SRT only)")
    parser.add_argument("--words-json", help="Use word timestamps JSON to build cut segments")
    parser.add_argument("--words-gap", type=float, help="Gap threshold for word-based segments (seconds)")
    args = parser.parse_args()

    if not os.path.exists(args.srt):
        raise SystemExit(f"✗ SRT not found: {args.srt}")
    if not os.path.exists(args.input):
        raise SystemExit(f"✗ Input not found: {args.input}")

    cut_video(
        args.srt,
        args.input,
        args.output,
        srt_out=args.srt_out,
        edl_out=args.edl,
        fps=args.fps,
        reel_name=args.reel_name,
        snap_srt=args.snap_srt,
        merge_gap=args.merge_gap,
        pad=args.pad,
        crf=args.crf,
        preset=args.preset,
        render_video=not args.no_video,
        words_json=args.words_json,
        words_gap=args.words_gap,
    )
    if args.no_video:
        print("✓ Cut outputs saved (EDL/SRT only)")
    else:
        print(f"✓ Cut video saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
