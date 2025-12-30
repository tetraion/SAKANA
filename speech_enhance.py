#!/usr/bin/env python3
"""
Speech enhancement helper.

動画/音声ファイルから音声を抽出し、DeepFilterNet でノイズ・BGM を低減した上で
映像に戻す簡易パイプライン。DeepFilterNet が入っていない場合は、実行前に
`pip install deepfilternet` を行ってください。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# 調整用パラメータ（ここだけ変更すれば効き方を変えられる）
# 値を大きくするとどうなるかを初心者向けに簡潔に記載
NOISEREDUCE_PARAMS = {
    # どれくらい減衰させるか（0.0〜1.0、デフォルト=1.0）
    # 大きくするとBGMを強く削るが、声も痩せやすくなる
    "prop_decrease": 1.0,
    # 定常ノイズ前提にするか（True だと話者への影響が少ないがBGM除去は弱め）
    "stationary": False,
    # 時間・周波数解像度まわり（必要に応じて上書き）
    # n_fft を大きくすると周波数分解能が上がり細かく削れるが計算が重く、時間ぼやけする
    "n_fft": None,
    "hop_length": None,
    "win_length": None,
    # マスク平滑化の強さ
    # 値を大きくすると滑らかに抑制するが、除去が甘くなることも
    "time_mask_smooth_ms": None,
    "freq_mask_smooth_hz": None,
    # 閾値の厳しさ（上げるほど強く削る）
    # 大きくするとBGM除去が強まるが、声のディテールも削れやすい
    "thresh_n_mult_nonstationary": None,
    "n_std_thresh_stationary": None,
}

# DeepFilterNet に追加したい CLI 引数があればここに書く（例: ["--model", "dns48"]）
DEEPFILTERNET_ARGS: list[str] = []

def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _probe_codec(path: str, stream_selector: str) -> str | None:
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


def _extract_audio(input_path: str, wav_path: str) -> None:
    # 16-bit PCM / 48kHz に揃える
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
            "48000",
            wav_path,
        ]
    )


def _enhance_with_deepfilternet(in_wav: str, out_wav: str) -> None:
    """
    DeepFilterNet CLI でノイズ抑圧を行う。
    パッケージが無い場合は ImportError で案内する。
    """
    try:
        import deepfilternet  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(
            "DeepFilterNet がインストールされていません。先に `pip install deepfilternet` を実行してください。"
        ) from exc

    # DeepFilterNet の公式 CLI
    cmd = [
        sys.executable,
        "-m",
        "deepfilternet.cli.enhance",
        "--input",
        in_wav,
        "--output",
        out_wav,
    ]
    if DEEPFILTERNET_ARGS:
        cmd.extend(DEEPFILTERNET_ARGS)
    _run(cmd)


def _enhance_with_noisereduce(in_wav: str, out_wav: str) -> None:
    """
    noisereduce（スペクトルゲート系）でノイズ/BGM を低減。
    必要に応じて reduce_noise のパラメータ（prop_decrease, stationary, n_fft など）を
    ここで直接渡すことで効き方を調整できます。
    """
    try:
        import soundfile as sf
        import noisereduce as nr
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(
            "noisereduce/soundfile がインストールされていません。`pip install noisereduce soundfile` を実行してください。"
        ) from exc

    data, rate = sf.read(in_wav)
    # モノラル化（多チャネルの場合は平均）
    if data.ndim > 1:
        data_mono = data.mean(axis=1)
    else:
        data_mono = data

    reduced = nr.reduce_noise(y=data_mono, sr=rate, **{k: v for k, v in NOISEREDUCE_PARAMS.items() if v is not None})
    sf.write(out_wav, reduced, rate)


def _mux_audio_video(video_path: str, audio_path: str, output_path: str) -> None:
    vcodec = _probe_codec(video_path, "v:0")
    if vcodec == "h264":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-shortest",
            output_path,
        ]
    else:
        # Ensure QuickTime-friendly MP4 even if the source video is VP9/AV1.
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-shortest",
            output_path,
        ]
    _run(cmd)


def enhance(
    input_path: str,
    output_path: str,
    *,
    audio_only: bool = False,
    method: str = "noisereduce",
) -> str:
    """
    音声を強調したファイルを生成する。

    Args:
        input_path: 入力の動画/音声ファイル
        output_path: 書き出し先（拡張子で動画/音声を判定）
        audio_only: True の場合は音声ファイルのみ出力し、映像へのマージを行わない
        method: "noisereduce"（依存が軽め）または "deepfilternet"

    Returns:
        出力ファイルのパス
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_wav = os.path.join(tmpdir, "raw.wav")
        enhanced_wav = os.path.join(tmpdir, "enhanced.wav")

        print("Extracting audio...")
        _extract_audio(input_path, raw_wav)

        if method == "deepfilternet":
            print("Enhancing with DeepFilterNet...")
            _enhance_with_deepfilternet(raw_wav, enhanced_wav)
        else:
            print("Enhancing with noisereduce (spectral gating)...")
            _enhance_with_noisereduce(raw_wav, enhanced_wav)

        if audio_only or output_path.lower().endswith((".wav", ".flac", ".mp3", ".m4a")):
            # 音声だけ欲しい場合はそのまま保存
            final_audio = output_path
            _run(["ffmpeg", "-y", "-i", enhanced_wav, final_audio])
            return final_audio

        print("Muxing enhanced audio back into video...")
        _mux_audio_video(input_path, enhanced_wav, output_path)
        return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Speech enhancement via DeepFilterNet + ffmpeg mux")
    parser.add_argument("input", help="入力の動画/音声ファイル")
    parser.add_argument("output", help="出力先ファイル（動画でも音声でも可）")
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="映像へ戻さず、音声ファイルだけ出力する",
    )
    parser.add_argument(
        "--method",
        choices=["noisereduce", "deepfilternet"],
        default="noisereduce",
        help="使用する強調手法。デフォルトは依存の軽い noisereduce。",
    )
    args = parser.parse_args(argv)

    try:
        out = enhance(args.input, args.output, audio_only=args.audio_only, method=args.method)
        print(f"✓ Done: {out}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI entry
        print(f"✗ Failed: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
