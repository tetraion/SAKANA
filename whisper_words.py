#!/usr/bin/env python3
"""
Generate word-level timestamps JSON using faster-whisper.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def transcribe_words(
    input_path: str,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
    beam_size: int,
) -> Dict[str, Any]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dep check
        raise SystemExit("faster-whisper が入っていません。`pip install faster-whisper` を実行してください。") from exc

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        input_path,
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
    )

    out_segments: List[Dict[str, Any]] = []
    for seg in segments:
        words = []
        for w in seg.words or []:
            words.append(
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word,
                }
            )
        out_segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
                "words": words,
            }
        )

    return {
        "language": info.language,
        "duration": info.duration,
        "segments": out_segments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate word timestamps JSON with faster-whisper")
    parser.add_argument("input", help="Input audio/video path")
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("--model", default="large", help="Whisper model name (default: large)")
    parser.add_argument("--language", default="ja", help="Language code (default: ja)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--compute-type", default="int8", help="Compute type (e.g., int8, int8_float16)")
    parser.add_argument("--beam", type=int, default=5, help="Beam size")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"✗ Input not found: {args.input}")

    result = transcribe_words(
        args.input,
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam,
    )

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=True, indent=2)
    print(f"✓ Written words JSON to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
