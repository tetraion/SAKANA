#!/usr/bin/env python3
"""
SRT de-duplicate/merge helper.

同じテキストが連続している字幕をまとめ、番号を振り直します。
・開始/終了が前後していても、テキストが同じなら結合
・小さな隙間（デフォルト 0.2 秒）までは連続扱いとして結合
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List


TIME_RE = re.compile(r"(?P<start>[\d:,]+)\s+-->\s+(?P<end>[\d:,]+)")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def time_to_seconds(t: str) -> float:
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def seconds_to_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    # 1000 に丸め上がるケースを補正
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


def parse_srt(text: str) -> List[Cue]:
    blocks = text.strip().split("\n\n")
    cues: List[Cue] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # 1行目: index（無視）
        # 2行目: time
        m = TIME_RE.search(lines[1])
        if not m:
            continue
        start = time_to_seconds(m.group("start"))
        end = time_to_seconds(m.group("end"))
        body = "\n".join(lines[2:]).strip()
        if not body:
            continue
        cues.append(Cue(start=min(start, end), end=max(start, end), text=body))
    # 開始時刻でソート
    cues.sort(key=lambda c: (c.start, c.end))
    return cues


def merge_cues(cues: List[Cue], max_gap: float = 0.2) -> List[Cue]:
    """同一テキストかつ時間が重なる/ほぼ連続の場合は結合する。"""
    merged: List[Cue] = []
    for cue in cues:
        if merged:
            prev = merged[-1]
            if (
                prev.text == cue.text
                and cue.start <= prev.end + max_gap
            ):
                prev.start = min(prev.start, cue.start)
                prev.end = max(prev.end, cue.end)
                continue
        merged.append(Cue(start=cue.start, end=cue.end, text=cue.text))
    return merged


def format_srt(cues: List[Cue]) -> str:
    lines: List[str] = []
    for idx, cue in enumerate(cues, start=1):
        lines.append(str(idx))
        lines.append(f"{seconds_to_time(cue.start)} --> {seconds_to_time(cue.end)}")
        lines.append(cue.text)
        lines.append("")  # blank line
    return "\n".join(lines).strip() + "\n"


def dedup_srt(input_path: str, output_path: str, max_gap: float) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    cues = parse_srt(content)
    merged = merge_cues(cues, max_gap=max_gap)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(format_srt(merged))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge duplicate/overlapping SRT cues")
    parser.add_argument("input", help="入力SRT")
    parser.add_argument("output", help="出力SRT")
    parser.add_argument(
        "--gap",
        type=float,
        default=0.2,
        help="同一テキストでこの秒数以内に連続/重複していたら結合（デフォルト0.2秒）",
    )
    args = parser.parse_args(argv)

    dedup_srt(args.input, args.output, max_gap=args.gap)
    print(f"✓ Deduped SRT saved to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
