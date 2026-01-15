#!/usr/bin/env python3
"""
MLX Whisper transcription helper (Apple Silicon).

動画/音声から音声を抽出し、mlx-whisper の large モデルで文字起こしして SRT を出力する。
KIRINUKI の Step1 実装を簡略化し、このディレクトリで完結するようにした版。
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import tempfile
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_LANGUAGE = "ja"


def _run(cmd: Sequence[str]) -> None:
    subprocess.run(cmd, check=True)


def extract_audio(input_path: str, wav_path: str) -> None:
    """ffmpeg で 16kHz モノラルの wav を切り出す。"""
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            wav_path,
        ]
    )


def format_timestamp_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments: Iterable[dict], output_path: str, offset_seconds: float) -> None:
    lines: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start_sec = max(0.0, float(seg["start"]) + offset_seconds)
        end_sec = max(0.0, float(seg["end"]) + offset_seconds)
        start = format_timestamp_srt(start_sec)
        end = format_timestamp_srt(end_sec)
        text = str(seg["text"]).strip()
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _parse_chunk_timestamp(chunk: dict) -> Optional[Tuple[float, float]]:
    ts = chunk.get("timestamp")
    if ts is None:
        ts = chunk.get("timestamps")
    if isinstance(ts, (list, tuple)) and len(ts) == 2:
        return float(ts[0]), float(ts[1])
    if "start" in chunk and "end" in chunk:
        return float(chunk["start"]), float(chunk["end"])
    return None


def transcribe_remote(input_path: str, base_url: str, model: str, language: Optional[str]):
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dep check
        raise SystemExit("requests が入っていません。`pip install requests` を実行してください。") from exc

    url = base_url.rstrip("/") + "/asr"
    params = {"model": model, "return_timestamps": "true"}
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
    segments = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        ts = _parse_chunk_timestamp(chunk)
        if not ts:
            continue
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        segments.append({"start": ts[0], "end": ts[1], "text": text})

    if not segments:
        raise SystemExit("✗ Remote ASR returned no timestamped segments")
    return result, segments


def transcribe_with_mlx(audio_path: str, model: str, language: str, verbose: bool = True):
    """mlx-whisper で文字起こし。"""
    try:
        import mlx_whisper  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dep check
        raise SystemExit("mlx-whisper が入っていません。`pip install mlx-whisper` を実行してください。") from exc

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model,
        language=language,
        verbose=verbose,
        fp16=True,
    )
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in result.get("segments", [])
    ]
    return result, segments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe with MLX Whisper (large model)")
    parser.add_argument("input", help="入力の動画/音声ファイル")
    parser.add_argument("output", help="出力先 SRT パス")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="mlx-whisper モデル（デフォルト: large v3）")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="言語コード（デフォルト: ja）")
    parser.add_argument("--no-verbose", action="store_true", help="mlx-whisper の詳細ログを抑制")
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="字幕タイムコードに加える秒オフセット（デフォルト: 0）",
    )
    parser.add_argument("--remote-url", help="Windows HF Server base URL (e.g., http://<IP>:8000)")
    parser.add_argument("--remote-model", default="openai/whisper-large-v3", help="Remote ASR model name")
    parser.add_argument("--remote-language", help="Remote ASR language code (default: --language)")
    # カットEDL生成は srt_to_edl.py 側で完結させる
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"✗ Input not found: {args.input}")
        return 1

    use_remote = args.remote_url is not None
    if not use_remote:
        # Apple Silicon 以外はパフォーマンス出ないため警告のみ
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            print("⚠️  MLX Whisper は Apple Silicon 最適化です。この環境では遅い/動かない可能性があります。")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        if use_remote:
            print(f"Transcribing with remote ASR ({args.remote_model})...")
            try:
                result, segments = transcribe_remote(
                    args.input,
                    base_url=args.remote_url,
                    model=args.remote_model,
                    language=args.remote_language or args.language,
                )
            except SystemExit as exc:
                print(f"✗ {exc}")
                return 1
            except Exception as exc:
                print(f"✗ Remote transcription failed: {exc}")
                return 1
        else:
            print("Extracting audio...")
            try:
                extract_audio(args.input, wav_path)
            except subprocess.CalledProcessError as exc:
                print(f"✗ Failed to extract audio: {exc}")
                return 1

            print(f"Transcribing with MLX Whisper ({args.model})...")
            try:
                result, segments = transcribe_with_mlx(
                    wav_path,
                    model=args.model,
                    language=args.language,
                    verbose=not args.no_verbose,
                )
            except SystemExit as exc:
                print(f"✗ {exc}")
                return 1
            except Exception as exc:
                print(f"✗ Transcription failed: {exc}")
                return 1

        print("Writing SRT...")
        generate_srt(segments, args.output, offset_seconds=args.offset)

        print("✓ Done")
        print(f"  SRT: {args.output}")
        print(f"  Language: {result.get('language', 'unknown')}")
        print(f"  Segments: {len(segments)}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
