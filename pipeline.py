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


def _read_segments(path: str) -> tuple[Optional[str], list[tuple[str, Optional[str], Optional[str], dict[str, str]]]]:
    segments: list[tuple[str, Optional[str], Optional[str], dict[str, str]]] = []
    url: Optional[str] = None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.lower().startswith("url="):
                url = line.split("=", 1)[-1].strip()
                continue
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 1:
                start, end, name, opts = parts[0], None, None, {}
            elif len(parts) == 2:
                start, end, name, opts = parts[0], parts[1], None, {}
            else:
                start, end = parts[0], parts[1]
                name_tokens: list[str] = []
                opts: dict[str, str] = {}
                options_started = False
                for token in parts[2:]:
                    if ("=" in token) or token.startswith("--"):
                        options_started = True
                    if options_started:
                        keyval = token.lstrip("-")
                        if "=" in keyval:
                            key, value = keyval.split("=", 1)
                            opts[key] = value
                        else:
                            opts[keyval] = "true"
                    else:
                        name_tokens.append(token)
                name = " ".join(name_tokens) if name_tokens else None
            segments.append((start, end, name, opts))
    return url, segments


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


def _ensure_cfr(input_path: str, output_path: str, fps: int = 30) -> None:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-r",
        str(fps),
        "-vsync",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        output_path,
    ]
    _run(cmd)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download -> Enhance -> Transcribe -> (Optional) Cut EDL pipeline")
    parser.add_argument("url", nargs="?", help="YouTube URL (optional if provided in segments.txt)")
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
    parser.add_argument("--gap", type=float, default=0.5, help="Gap threshold for cuts (seconds)")
    parser.add_argument("--words-model", default="large", help="Whisper model for word timestamps")
    parser.add_argument("--words-language", default="ja", help="Language code for word timestamps")
    parser.add_argument("--words-device", default="cpu", help="Device for word timestamps (cpu/cuda/mps)")
    parser.add_argument("--words-compute-type", default="int8", help="Compute type for word timestamps")
    parser.add_argument("--words-beam", type=int, default=5, help="Beam size for word timestamps")
    parser.add_argument("--cut-pad", type=float, default=0.12, help="Pad seconds for cut video cues")
    parser.add_argument("--cut-merge-gap", type=float, default=0.3, help="Merge gaps shorter than this in cut video")
    parser.add_argument("--cut-crf", type=int, default=20, help="CRF for cut video")
    parser.add_argument("--cut-preset", default="veryfast", help="x264 preset for cut video")
    parser.add_argument("--segments", help="Segments list file (start end [name])")
    args = parser.parse_args(argv)

    if not args.segments and not args.start:
        parser.error("start time is required unless --segments is provided")

    url = args.url
    if args.segments:
        seg_url, segments = _read_segments(args.segments)
        if not url:
            url = seg_url
        if not url:
            raise SystemExit("URL is required (pass as arg or set url=... in segments.txt)")
        base_name = args.name or _default_name_from_url(url)
        output_dir = os.path.join(args.output_dir, base_name)
    else:
        if not url:
            raise SystemExit("URL is required.")
        base_name = args.name or _default_name_from_url(url)
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if args.segments:
        if not segments:
            raise SystemExit("No segments found in segments file")

        full_path = os.path.join(output_dir, f"{base_name}_full.mp4")
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            print("Full download exists, skipping download.")
        else:
            print("Downloading full video once...")
            _download_full_video(url, full_path, args.format)

        for idx, (start, end, name, seg_opts) in enumerate(segments, start=1):
            label = f"{idx:02d}"
            suffix = f"_{_slugify(name)}" if name else ""
            clip_base = f"{base_name}_{label}{suffix}"

            downloaded = os.path.join(output_dir, f"{clip_base}_src.mp4")
            enhanced = os.path.join(output_dir, f"{clip_base}_enh.mp4")
            srt_out = os.path.join(output_dir, f"{clip_base}.srt")
            words_json = os.path.join(output_dir, f"{clip_base}_words.json")
            words_trimmed = os.path.join(output_dir, f"{clip_base}_words_trimmed.srt")
            cut_video = os.path.join(output_dir, f"{clip_base}_cut.mp4")
            cut_srt = os.path.join(output_dir, f"{clip_base}_cut.srt")
            cut_edl = os.path.join(output_dir, f"{clip_base}_cut.edl")
            cfr_src = os.path.join(output_dir, f"{clip_base}_src_cfr.mp4")

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

            do_post = False
            if seg_opts.get("trim_words"):
                do_post = seg_opts.get("trim_words", "").lower() in ("1", "true", "yes", "on")

            if do_post:
                if os.path.exists(words_json) and os.path.getsize(words_json) > 0:
                    print(f"[{label}] Word timestamps... (skipped, file exists)")
                else:
                    print(f"[{label}] Word timestamps...")
                    ww_cmd = [
                        sys.executable,
                        os.path.join(os.path.dirname(__file__), "whisper_words.py"),
                        enhanced,
                        words_json,
                        "--model",
                        seg_opts.get("words_model", args.words_model),
                        "--language",
                        seg_opts.get("words_language", args.words_language),
                        "--device",
                        seg_opts.get("words_device", args.words_device),
                        "--compute-type",
                        seg_opts.get("words_compute_type", args.words_compute_type),
                        "--beam",
                        str(int(seg_opts.get("words_beam", args.words_beam))),
                    ]
                    _run(ww_cmd)

                print(f"[{label}] Trimming SRT by words...")
                trim_cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "srt_trim_words.py"),
                    srt_out,
                    words_json,
                    words_trimmed,
                ]
                _run(trim_cmd)

            do_cut_video = False
            if seg_opts.get("cut_video"):
                do_cut_video = seg_opts.get("cut_video", "").lower() in ("1", "true", "yes", "on")

            if do_cut_video:
                use_edl = seg_opts.get("edl", "").lower() in ("1", "true", "yes", "on")
                cut_input = downloaded
                if use_edl:
                    print(f"[{label}] CFR source for EDL...")
                    _ensure_cfr(downloaded, cfr_src)
                    cut_input = cfr_src
                print(f"[{label}] Cutting video by SRT cues...")
                pad = float(seg_opts.get("pad", args.cut_pad))
                merge_gap = float(seg_opts.get("merge_gap", args.cut_merge_gap))
                crf = int(seg_opts.get("crf", args.cut_crf))
                preset = seg_opts.get("preset", args.cut_preset)
                srt_input = words_trimmed if os.path.exists(words_trimmed) else srt_out
                cut_cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "srt_cut_video.py"),
                    srt_input,
                    cut_input,
                    cut_video,
                    "--srt-out",
                    cut_srt,
                    "--merge-gap",
                    str(merge_gap),
                    "--pad",
                    str(pad),
                    "--crf",
                    str(crf),
                    "--preset",
                    preset,
                ]
                if use_edl:
                    cut_cmd.extend(["--edl", cut_edl, "--snap-srt"])
                _run(cut_cmd)

            print(f"[{label}] ✓ Done: {clip_base}")

        print("✓ Pipeline complete (multi-segment)")
        print(f"  Output dir: {output_dir}")
        return 0

    downloaded = os.path.join(output_dir, f"{base_name}_src.mp4")
    enhanced = os.path.join(output_dir, f"{base_name}_enh.mp4")
    srt_out = os.path.join(output_dir, f"{base_name}.srt")
    words_json = os.path.join(output_dir, f"{base_name}_words.json")
    words_trimmed = os.path.join(output_dir, f"{base_name}_words_trimmed.srt")
    cut_video = os.path.join(output_dir, f"{base_name}_cut.mp4")
    cut_srt = os.path.join(output_dir, f"{base_name}_cut.srt")
    cut_edl = os.path.join(output_dir, f"{base_name}_cut.edl")

    total_steps = 4
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
            url,
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

    print("✓ Pipeline complete")
    print(f"  Downloaded: {downloaded}")
    print(f"  Enhanced:   {enhanced}")
    print(f"  SRT:        {srt_out}")
    # post-process is driven by segments.txt options only

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
