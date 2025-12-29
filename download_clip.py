#!/usr/bin/env python3
"""
Sakanaction clip downloader (standalone)

KIRINUKI の Step0 で使っている処理を、このディレクトリだけで完結するように
切り出したシンプル版。YouTube の URL と開始/終了時刻を受け取り、指定区間を
ダウンロードして保存する。
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# 安全側に少し前後を含めるバッファ（fastモード時のみ有効）
DEFAULT_BUFFER_SECONDS = 3.0


def parse_time(time_str: str) -> float:
    """`hh:mm:ss`, `mm:ss`, または秒数の文字列を秒に変換。"""
    parts = str(time_str).split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 1:
        hours = 0
        minutes = 0
        seconds = parts[0]
    else:
        raise ValueError(f"Invalid time string: {time_str}")
    return hours * 3600 + minutes * 60 + seconds


def format_time(seconds: float, include_ms: bool = False) -> str:
    """秒を `hh:mm:ss` 形式に整形（yt-dlp/ffmpegの入力用）。"""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds % 60
    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{remaining:06.3f}"
    return f"{hours:02d}:{minutes:02d}:{int(remaining):02d}"


def download_clip(
    video_url: str,
    start_time: str,
    end_time: Optional[str],
    output_path: str,
    *,
    # KIRINUKI と同様に、扱いやすい H.264/mp4 を優先（1080p上限）
    video_format: str = "bv*[height<=1080][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[height<=1080][ext=mp4][vcodec^=avc1]",
    buffer_seconds: float = DEFAULT_BUFFER_SECONDS,
) -> bool:
    """
    YouTube から指定区間をダウンロードして保存する。

    Args:
        video_url: 対象の YouTube URL
        start_time: 開始時刻（hh:mm:ss, mm:ss, もしくは秒）
        end_time: 終了時刻。None の場合は動画末尾まで。
        output_path: 保存するパス（拡張子は自動判定されるが、明示推奨）
        video_format: yt-dlp の -f 指定（デフォルトは H.264/mp4 優先）
        buffer_seconds: 前後に足すバッファ（キーフレームのズレ対策）
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    return _download_with_sections(
        video_url,
        start_time,
        end_time,
        output_path,
        video_format,
        buffer_seconds,
    )


def _download_with_sections(
    video_url: str,
    start_time: str,
    end_time: Optional[str],
    output_path: str,
    video_format: str,
    buffer_seconds: float,
) -> bool:
    """yt-dlp の --download-sections で指定区間だけを取得。"""
    start_sec = parse_time(start_time)
    start_for_dl = max(0.0, start_sec - max(0.0, buffer_seconds))
    start_label = format_time(start_for_dl)

    if end_time:
        end_sec = parse_time(end_time)
        end_for_dl = end_sec + max(0.0, buffer_seconds)
        section = f"*{start_label}-{format_time(end_for_dl)}"
    else:
        section = f"*{start_label}-inf"

    base_path = os.path.splitext(output_path)[0]
    js_runtime_args = []
    node_path = shutil.which("node")
    if node_path:
        js_runtime_args = ["--js-runtimes", f"node={node_path}"]

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        video_format,
        "--download-sections",
        section,
        "-o",
        base_path,
        "--force-keyframes-at-cuts",
        "--force-overwrites",
        *js_runtime_args,
        video_url,
    ]

    possible_outputs = [
        output_path,
        base_path,
        f"{base_path}.webm",
        f"{base_path}.mp4",
        f"{base_path}.mkv",
        f"{base_path}.part",
    ]
    for path in possible_outputs:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    try:
        print(f"Downloading section: {section}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        error_msg = exc.stderr or exc.stdout or str(exc)
        print(f"✗ Download failed: {error_msg}")
        return False

    found = None
    for path in possible_outputs:
        if os.path.exists(path):
            found = path
            break
    if not found:
        candidates = glob.glob(f"{base_path}*")
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            found = candidates[0]

    if not found:
        print("✗ Output file not found after yt-dlp finished")
        if result.stdout:
            print("  yt-dlp stdout:", result.stdout.strip())
        if result.stderr:
            print("  yt-dlp stderr:", result.stderr.strip())
        return False

    if found != output_path:
        os.rename(found, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Downloaded clip saved to {output_path} ({size_mb:.2f} MB)")
    return True


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sakanaction clip downloader (standalone)")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("start", help="Start time (hh:mm:ss or mm:ss or seconds)")
    parser.add_argument("output", help="Output file path (e.g. data/clip.webm)")
    parser.add_argument(
        "-e",
        "--end",
        help="End time (optional; if omitted, downloads until the end)",
    )
    parser.add_argument(
        "-b",
        "--buffer",
        type=float,
        default=DEFAULT_BUFFER_SECONDS,
        help="Seconds to pad before/after clip in fast mode",
    )
    parser.add_argument(
        "--format",
        default="bv*[height<=1080]+ba/b[height<=1080]/best",
        help="yt-dlp format string (default keeps up to 1080p)",
    )

    args = parser.parse_args(argv)

    ok = download_clip(
        args.url,
        args.start,
        args.end,
        args.output,
        video_format=args.format,
        buffer_seconds=args.buffer,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
