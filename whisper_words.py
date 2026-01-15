#!/usr/bin/env python3
"""
Generate word-level timestamps JSON using faster-whisper.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def transcribe_words_local(
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


def _parse_chunk_timestamp(chunk: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    ts = chunk.get("timestamp")
    if ts is None:
        ts = chunk.get("timestamps")
    if isinstance(ts, (list, tuple)) and len(ts) == 2:
        return float(ts[0]), float(ts[1])
    if "start" in chunk and "end" in chunk:
        return float(chunk["start"]), float(chunk["end"])
    return None


def transcribe_words_remote(
    input_path: str,
    base_url: str,
    model_name: str,
    language: Optional[str],
) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dep check
        raise SystemExit("requests が入っていません。`pip install requests` を実行してください。") from exc

    url = base_url.rstrip("/") + "/asr"
    params: Dict[str, str] = {"model": model_name, "return_timestamps": "word"}
    if language:
        params["language"] = language

    with open(input_path, "rb") as fh:
        resp = requests.post(
            url,
            params=params,
            files={"file": (os.path.basename(input_path), fh, "audio/wav")},
            timeout=600,
        )
    if resp.status_code != 200:
        raise SystemExit(f"✗ Remote ASR failed ({resp.status_code}): {resp.text}")

    payload = resp.json()
    result = payload.get("result", {})
    chunks = result.get("chunks") or []
    words: List[Dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        ts = _parse_chunk_timestamp(chunk)
        if not ts:
            continue
        text = str(chunk.get("text") or chunk.get("word") or "").strip()
        if not text:
            continue
        words.append({"start": ts[0], "end": ts[1], "word": text})

    if not words:
        raise SystemExit("✗ Remote ASR returned no word timestamps")

    start = min(w["start"] for w in words)
    end = max(w["end"] for w in words)
    return {
        "language": result.get("language", language or "unknown"),
        "duration": result.get("duration"),
        "segments": [{"start": start, "end": end, "text": result.get("text", ""), "words": words}],
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
    parser.add_argument("--remote-url", help="Windows HF Server base URL (e.g., http://<IP>:8000)")
    parser.add_argument("--remote-model", default="large-v3", help="Remote ASR model name")
    parser.add_argument("--remote-language", help="Remote ASR language code (default: --language)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"✗ Input not found: {args.input}")

    if args.remote_url:
        result = transcribe_words_remote(
            args.input,
            base_url=args.remote_url,
            model_name=args.remote_model,
            language=args.remote_language or args.language,
        )
    else:
        result = transcribe_words_local(
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
