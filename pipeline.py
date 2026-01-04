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
import os
import subprocess
import sys
from typing import Optional


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _default_name_from_url(url: str) -> str:
    tail = url.split("/")[-1]
    if "v=" in tail:
        return tail.split("v=")[-1].split("&")[0]
    return tail.split("?")[0] or "clip"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download -> Enhance -> Transcribe -> (Optional) Cut EDL pipeline")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("start", help="Start time (hh:mm:ss or mm:ss or seconds)")
    parser.add_argument("end", nargs="?", help="End time (optional; if omitted, runs to end)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--name", help="Base name for outputs (default from URL)")
    parser.add_argument("--buffer", type=float, default=3.0, help="Download buffer seconds")
    parser.add_argument(
        "--format",
        default="bv*[height<=1080][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[height<=1080][ext=mp4][vcodec^=avc1]",
        help="yt-dlp format",
    )
    parser.add_argument("--enhance-method", choices=["noisereduce", "deepfilternet"], default="noisereduce")
    parser.add_argument("--language", default="ja", help="Whisper language code")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for EDL")
    parser.add_argument("--gap", type=float, default=0.5, help="Gap threshold for cuts (seconds)")
    parser.add_argument("--cut", action="store_true", help="Create cut EDL and cut SRT")
    parser.add_argument("--src-tc-base", default="00:00:00:00", help="EDL source TC base")
    parser.add_argument("--rec-tc-base", default="00:00:00:00", help="EDL record TC base")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = args.name or _default_name_from_url(args.url)

    downloaded = os.path.join(args.output_dir, f"{base_name}_src.mp4")
    enhanced = os.path.join(args.output_dir, f"{base_name}_enh.mp4")
    srt_out = os.path.join(args.output_dir, f"{base_name}.srt")
    edl_out = os.path.join(args.output_dir, f"{base_name}_cut.edl")

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
