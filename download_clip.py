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
import tempfile
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
    mode: str = "precise",
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

    mode = mode.lower()
    if mode not in ("fast", "precise"):
        raise ValueError(f"Invalid mode: {mode} (choose 'fast' or 'precise')")

    if mode == "fast":
        return _download_full_then_stream_copy(
            video_url,
            start_time,
            end_time,
            output_path,
            video_format,
        )

    return _download_with_sections(
        video_url,
        start_time,
        end_time,
        output_path,
        video_format,
        buffer_seconds,
        mode,
    )


def _download_with_sections(
    video_url: str,
    start_time: str,
    end_time: Optional[str],
    output_path: str,
    video_format: str,
    buffer_seconds: float,
    mode: str,
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
        "--force-overwrites",
        *js_runtime_args,
        video_url,
    ]
    if mode == "precise":
        cmd.insert(cmd.index("--force-overwrites"), "--force-keyframes-at-cuts")

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

    if mode == "precise" and output_path.lower().endswith(".mp4"):
        _ensure_h264_mp4(output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Downloaded clip saved to {output_path} ({size_mb:.2f} MB)")
    return True


def _download_full_then_stream_copy(
    video_url: str,
    start_time: str,
    end_time: Optional[str],
    output_path: str,
    video_format: str,
) -> bool:
    """動画全体をダウンロードして、ストリームコピーで切り抜く（再エンコードなし）。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        full_base = os.path.join(tmpdir, "full_video")
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            video_format,
            "-o",
            full_base,
            "--force-overwrites",
            video_url,
        ]
        try:
            print("Downloading full video...")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr or exc.stdout or str(exc)
            print(f"✗ Full download failed: {error_msg}")
            return False

        candidates = glob.glob(f"{full_base}*")
        if not candidates:
            print("✗ Full download output not found")
            return False
        candidates.sort(key=os.path.getmtime, reverse=True)
        full_path = candidates[0]

        print("Clipping with stream copy...")
        return _clip_stream_copy(full_path, start_time, end_time, output_path)


def _clip_stream_copy(
    input_path: str,
    start_time: str,
    end_time: Optional[str],
    output_path: str,
) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        start_time,
        "-i",
        input_path,
    ]
    if end_time:
        cmd.extend(["-to", end_time])
    cmd.extend(["-c", "copy", output_path])
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"✗ Stream copy failed: {exc}")
        return False
    if not os.path.exists(output_path):
        print("✗ Stream copy output not found")
        return False
    return True


def _probe_codec(path: str, stream_selector: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                stream_selector,
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nk=1:nw=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    codec = result.stdout.strip()
    return codec or None


def _ensure_h264_mp4(path: str) -> None:
    vcodec = _probe_codec(path, "v:0")
    acodec = _probe_codec(path, "a:0")
    if vcodec == "h264" and acodec == "aac":
        return

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                tmp_path,
            ],
            check=True,
        )
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
    parser.add_argument(
        "--mode",
        choices=["fast", "precise"],
        default="precise",
        help="precise: keyframe cut + h264 mp4 ensure, fast: full download + stream copy",
    )

    args = parser.parse_args(argv)

    ok = download_clip(
        args.url,
        args.start,
        args.end,
        args.output,
        video_format=args.format,
        buffer_seconds=args.buffer,
        mode=args.mode,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
