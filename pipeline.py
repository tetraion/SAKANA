#!/usr/bin/env python3
"""
End-to-end pipeline:
- Download clip
- Reduce BGM/noise
- Transcribe (SRT)
- Create cut EDL + cut SRT from subtitle gaps
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from typing import Optional

from download_clip import parse_time


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _default_name_from_url(url: str) -> str:
    tail = url.split("/")[-1]
    if "v=" in tail:
        return tail.split("v=")[-1].split("&")[0]
    return tail.split("?")[0] or "clip"


def _read_segments(path: str) -> list[tuple[str, Optional[str], Optional[str]]]:
    segments: list[tuple[str, Optional[str], Optional[str]]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 1:
                start, end, name = parts[0], None, None
            elif len(parts) == 2:
                start, end, name = parts[0], parts[1], None
            else:
                start, end = parts[0], parts[1]
                name = " ".join(parts[2:])
            segments.append((start, end, name))
    return segments


def _slugify(name: str) -> str:
    cleaned = []
    for ch in name.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in ("-", "_"):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("-")
    return "".join(cleaned).strip("-")


def _download_full_video(url: str, output_path: str, fmt: str) -> str:
    base_path = os.path.splitext(output_path)[0]
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        fmt,
        "-o",
        base_path,
        "--merge-output-format",
        "mp4",
        "--force-overwrites",
        url,
    ]
    _run(cmd)

    candidates = glob.glob(f"{base_path}*")
    if not candidates:
        raise FileNotFoundError("Full download output not found")
    candidates.sort(key=os.path.getmtime, reverse=True)
    found = candidates[0]
    if found != output_path:
        os.replace(found, output_path)
    return output_path


def _clip_from_full(
    full_path: str,
    start: str,
    end: Optional[str],
    output_path: str,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        full_path,
        "-ss",
        start,
    ]
    if end:
        duration = parse_time(end) - parse_time(start)
        if duration <= 0:
            raise ValueError(f"Invalid segment: end <= start ({start} - {end})")
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_path,
        ]
    )
    _run(cmd)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download -> Enhance -> Transcribe -> (Optional) Cut EDL pipeline")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("start", nargs="?", help="Start time (hh:mm:ss or mm:ss or seconds)")
    parser.add_argument("end", nargs="?", help="End time (optional; if omitted, runs to end)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--name", help="Base name for outputs (default from URL)")
    parser.add_argument("--buffer", type=float, default=3.0, help="Download buffer seconds")
    parser.add_argument(
        "--format",
        default="bv*[height<=1080][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[height<=1080][ext=mp4][vcodec^=avc1]",
        help="yt-dlp format",
    )
    parser.add_argument(
        "--download-mode",
        choices=["fast", "precise"],
        default="precise",
        help="Download clip mode: fast (full download + stream copy), precise (keyframe cut + h264 mp4 ensure)",
    )
    parser.add_argument("--enhance-method", choices=["noisereduce", "deepfilternet"], default="noisereduce")
    parser.add_argument("--language", default="ja", help="Whisper language code")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for EDL")
    parser.add_argument("--gap", type=float, default=0.5, help="Gap threshold for cuts (seconds)")
    parser.add_argument("--cut", action="store_true", help="Create cut EDL and cut SRT")
    parser.add_argument("--src-tc-base", default="00:00:00:00", help="EDL source TC base")
    parser.add_argument("--rec-tc-base", default="00:00:00:00", help="EDL record TC base")
    parser.add_argument("--segments", help="Segments list file (start end [name])")
    args = parser.parse_args(argv)

    if not args.segments and not args.start:
        parser.error("start time is required unless --segments is provided")

    base_name = args.name or _default_name_from_url(args.url)

    if args.segments:
        output_dir = os.path.join(args.output_dir, base_name)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if args.segments:
        segments = _read_segments(args.segments)
        if not segments:
            raise SystemExit("No segments found in segments file")

        full_path = os.path.join(output_dir, f"{base_name}_full.mp4")
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            print("Full download exists, skipping download.")
        else:
            print("Downloading full video once...")
            _download_full_video(args.url, full_path, args.format)

        for idx, (start, end, name) in enumerate(segments, start=1):
            label = f"{idx:02d}"
            suffix = f"_{_slugify(name)}" if name else ""
            clip_base = f"{base_name}_{label}{suffix}"

            downloaded = os.path.join(output_dir, f"{clip_base}_src.mp4")
            enhanced = os.path.join(output_dir, f"{clip_base}_enh.mp4")
            srt_out = os.path.join(output_dir, f"{clip_base}.srt")
            edl_out = os.path.join(output_dir, f"{clip_base}_cut.edl")

            if os.path.exists(downloaded) and os.path.getsize(downloaded) > 0:
                print(f"[{label}] Clip exists, skipping cut.")
            else:
                print(f"[{label}] Cutting clip from full video...")
                _clip_from_full(full_path, start, end, downloaded)

            print(f"[{label}] Enhancing speech (BGM reduction)...")
            enh_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "speech_enhance.py"),
                downloaded,
                enhanced,
                "--method",
                args.enhance_method,
            ]
            _run(enh_cmd)

            if os.path.exists(srt_out) and os.path.getsize(srt_out) > 0:
                print(f"[{label}] Transcribing to SRT... (skipped, file exists)")
            else:
                print(f"[{label}] Transcribing to SRT...")
                tr_cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "transcribe_mlx.py"),
                    enhanced,
                    srt_out,
                    "--language",
                    args.language,
                    "--model",
                    args.model,
                ]
                _run(tr_cmd)

            print(f"[{label}] De-duplicating SRT...")
            dedup_tmp = f"{srt_out}.dedup"
            dedup_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "srt_dedup.py"),
                srt_out,
                dedup_tmp,
            ]
            _run(dedup_cmd)
            os.replace(dedup_tmp, srt_out)

            if args.cut:
                print(f"[{label}] Creating cut EDL and cut SRT...")
                cut_cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "srt_to_edl.py"),
                    srt_out,
                    edl_out,
                    "--fps",
                    str(args.fps),
                    "--gap",
                    str(args.gap),
                    "--src-tc-base",
                    args.src_tc_base,
                    "--rec-tc-base",
                    args.rec_tc_base,
                ]
                _run(cut_cmd)

            print(f"[{label}] ✓ Done: {clip_base}")

        print("✓ Pipeline complete (multi-segment)")
        print(f"  Output dir: {output_dir}")
        return 0

    downloaded = os.path.join(output_dir, f"{base_name}_src.mp4")
    enhanced = os.path.join(output_dir, f"{base_name}_enh.mp4")
    srt_out = os.path.join(output_dir, f"{base_name}.srt")
    edl_out = os.path.join(output_dir, f"{base_name}_cut.edl")

    total_steps = 5 if args.cut else 4
    step = 1

    def _step(msg: str) -> None:
        nonlocal step
        print(f"[{step}/{total_steps}] {msg}")
        step += 1

    if os.path.exists(downloaded) and os.path.getsize(downloaded) > 0:
        _step("Downloading clip... (skipped, file exists)")
    else:
        _step("Downloading clip...")
        dl_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "download_clip.py"),
            args.url,
            args.start,
            downloaded,
            "--buffer",
            str(args.buffer),
            "--format",
            args.format,
            "--mode",
            args.download_mode,
        ]
        if args.end:
            dl_cmd.extend(["--end", args.end])
        _run(dl_cmd)

    _step("Enhancing speech (BGM reduction)...")
    enh_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "speech_enhance.py"),
        downloaded,
        enhanced,
        "--method",
        args.enhance_method,
    ]
    _run(enh_cmd)

    if os.path.exists(srt_out) and os.path.getsize(srt_out) > 0:
        _step("Transcribing to SRT... (skipped, file exists)")
    else:
        _step("Transcribing to SRT...")
        tr_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "transcribe_mlx.py"),
            enhanced,
            srt_out,
            "--language",
            args.language,
            "--model",
            args.model,
        ]
        _run(tr_cmd)

    _step("De-duplicating SRT...")
    dedup_tmp = f"{srt_out}.dedup"
    dedup_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "srt_dedup.py"),
        srt_out,
        dedup_tmp,
    ]
    _run(dedup_cmd)
    os.replace(dedup_tmp, srt_out)

    if args.cut:
        _step("Creating cut EDL and cut SRT...")
        cut_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "srt_to_edl.py"),
            srt_out,
            edl_out,
            "--fps",
            str(args.fps),
            "--gap",
            str(args.gap),
            "--src-tc-base",
            args.src_tc_base,
            "--rec-tc-base",
            args.rec_tc_base,
        ]
        _run(cut_cmd)

    print("✓ Pipeline complete")
    print(f"  Downloaded: {downloaded}")
    print(f"  Enhanced:   {enhanced}")
    print(f"  SRT:        {srt_out}")
    if args.cut:
        print(f"  Cut EDL:    {edl_out}")
        print(f"  Cut SRT:    {os.path.splitext(edl_out)[0]}_cut.srt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
