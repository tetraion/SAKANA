#!/usr/bin/env python3
"""
Trim SRT cue boundaries using word-level timestamps.

This keeps the SRT text as-is, but sets each cue's start/end to the
first/last matched word timestamps in a word-level transcript.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


TIME_RE = re.compile(r"(?P<start>[\d:,]+)\s+-->\s+(?P<end>[\d:,]+)")


@dataclass
class Cue:
    start: float
    end: float
    text: str


@dataclass
class Word:
    start: float
    end: float
    text: str


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
        body = "\n".join(lines[2:]).strip()
        if not body:
            continue
        cues.append(Cue(start=min(start, end), end=max(start, end), text=body))
    return cues


def _format_srt(cues: List[Cue]) -> str:
    lines: List[str] = []
    for idx, cue in enumerate(cues, start=1):
        lines.append(str(idx))
        lines.append(f"{_seconds_to_time(cue.start)} --> {_seconds_to_time(cue.end)}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[。、，,.!?？！「」『』（）()\\[\\]{}<>《》【】〜~・…ー-]", "", text)
    return text


def _load_words(path: str) -> List[Word]:
    data = json.loads(open(path, "r", encoding="utf-8").read())
    words: List[Word] = []

    segments = None
    if isinstance(data, dict):
        if "segments" in data:
            segments = data["segments"]
        elif "transcription" in data:
            segments = data["transcription"]
    if segments is None:
        raise ValueError("Unsupported JSON format: expected top-level 'segments' or 'transcription'")

    for seg in segments:
        for w in seg.get("words", []):
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            words.append(Word(start=float(w["start"]), end=float(w["end"]), text=text))

    if not words:
        raise ValueError("No words found in JSON (missing 'words' array?)")
    return words


def _build_char_index(words: List[Word]) -> Dict[str, List[int] | str]:
    full_text = ""
    char_to_word: List[int] = []
    for idx, word in enumerate(words):
        norm = _normalize_text(word.text)
        if not norm:
            continue
        full_text += norm
        char_to_word.extend([idx] * len(norm))
    return {"text": full_text, "map": char_to_word}


def _build_srt_index(cues: List[Cue]) -> Tuple[str, List[Tuple[int, int]]]:
    full_text = ""
    ranges: List[Tuple[int, int]] = []
    for cue in cues:
        norm = _normalize_text(cue.text)
        start = len(full_text)
        full_text += norm
        end = len(full_text)
        ranges.append((start, end))
        full_text += "|"  # separator not expected in ASR text
    return full_text, ranges


def _trim_by_words(
    cues: List[Cue],
    words: List[Word],
    pad: float,
    min_cue: float,
) -> tuple[List[Cue], int]:
    asr_index = _build_char_index(words)
    asr_text = asr_index["text"]
    asr_map: List[int] = asr_index["map"]  # type: ignore[assignment]

    srt_text, srt_ranges = _build_srt_index(cues)
    matcher = SequenceMatcher(None, srt_text, asr_text)

    srt_to_asr: List[Optional[int]] = [None] * len(srt_text)
    for i, j, size in matcher.get_matching_blocks():
        for k in range(size):
            if i + k < len(srt_to_asr):
                srt_to_asr[i + k] = j + k

    misses = 0
    trimmed: List[Cue] = []

    for cue, (start_pos, end_pos) in zip(cues, srt_ranges):
        if end_pos <= start_pos:
            trimmed.append(cue)
            continue

        mapped = [idx for idx in (srt_to_asr[start_pos:end_pos]) if idx is not None]
        if not mapped:
            misses += 1
            trimmed.append(cue)
            continue

        start_idx = asr_map[min(mapped)]
        end_idx = asr_map[max(mapped)]
        new_start = max(0.0, words[start_idx].start - pad)
        new_end = words[end_idx].end + pad
        if new_end - new_start < min_cue:
            trimmed.append(cue)
        else:
            trimmed.append(Cue(start=new_start, end=new_end, text=cue.text))

    return trimmed, misses


def main() -> int:
    parser = argparse.ArgumentParser(description="Trim SRT cue boundaries using word timestamps")
    parser.add_argument("srt", help="Input SRT path")
    parser.add_argument("words_json", help="Word-level timestamps JSON (segments[].words[])")
    parser.add_argument("output", help="Output SRT path")
    parser.add_argument("--pad", type=float, default=0.0, help="Seconds to pad at both ends")
    parser.add_argument("--min-cue", type=float, default=0.2, help="Minimum cue duration after trimming")
    parser.add_argument(
        "--no-guard-next",
        action="store_true",
        help="Disable guard that reverts trims crossing the next cue start",
    )
    args = parser.parse_args()

    cues = _parse_srt(args.srt)
    words = _load_words(args.words_json)
    trimmed, misses = _trim_by_words(cues, words, pad=args.pad, min_cue=args.min_cue)
    if not args.no_guard_next:
        for i in range(len(trimmed) - 1):
            if trimmed[i].start > cues[i + 1].start:
                trimmed[i] = cues[i]

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(_format_srt(trimmed))

    print(f"✓ Trimmed SRT saved to {args.output} (missed={misses})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
