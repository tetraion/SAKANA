# SAKANA – Clip Downloader (Standalone)

サカナクションの切り抜き用に、動画のダウンロード部分だけをこのフォルダ内で完結する形でまとめました。上位の `KIRINUKI` フォルダには一切手を触れていません。

## 必要なもの
- Python 3.9+
- `yt-dlp` と `ffmpeg` が PATH にあること  
  例: `brew install yt-dlp ffmpeg`

## 使い方
指定区間だけをダウンロードして保存します。高速な範囲ダウンロード（再エンコードなし）のみを提供します。

```bash
python download_clip.py "https://www.youtube.com/watch?v=XXXXX" 00:01:23 output/clip.webm \
  --end 00:02:34 \
  --buffer 3.0           # 前後バッファ秒数
  --format "bv*[height<=1080]+ba/best"
```

- yt-dlp の `--download-sections` でコピーするだけ（再エンコードなし）。キーフレームずれに備えてバッファを付与。
- `--end` を省略すると末尾まで。

## 補足
- コードは `KIRINUKI/kirinuki_processor/steps/step0_download_clip.py` の挙動を参考に、依存を最小化するよう再構成しています。
- 依存を追加したくない場合は、既存の Python 環境に `yt-dlp` と `ffmpeg` が入っていれば実行できます。

## スピーチ強調（ノイズ/BGM低減）
スピーチ強調を追加しました。バックのBGM/歌を抑えて話者を聞きやすくします。

```bash
# 依存（軽めのスペクトルゲート版）
pip install noisereduce soundfile

# DeepFilterNet 版を使いたい場合（要 Rust/Cargo）
# pip install deepfilternet

# 音声を強調して映像に戻す（デフォルト: noisereduce）
python speech_enhance.py input_video.mp4 output_video_enh.mp4

# DeepFilterNet を明示指定
python speech_enhance.py input_video.mp4 output_video_enh.mp4 --method deepfilternet

# 音声だけ欲しい場合（WAVなど）
python speech_enhance.py input_video.mp4 enhanced.wav --audio-only
```

仕組み: `ffmpeg` で音声抽出 → noisereduce（または DeepFilterNet）でノイズ/BGM低減 → 必要なら映像と再結合。DeepFilterNet が無い場合は案内が出ます。

## Whisper 文字起こし（MLX / Apple Silicon 向け）
- 依存: `pip install mlx-whisper`（Apple Silicon 最適化。arm64 macOS 推奨）
- 実行:
  ```bash
  .venv/bin/python transcribe_mlx.py input_video.mp4 output/subs.srt \
    --language ja \
    --model mlx-community/whisper-large-v3-mlx
  ```
- 中でやっていること:
  1. ffmpeg で 16kHz モノラル wav を抽出
  2. mlx-whisper で文字起こし（デフォルト large v3）
  3. SRT を生成

※ Apple Silicon 以外では遅い/動かない可能性があります。mlx-whisper が無い場合はエラーで案内を出します。

## 複数区間のパイプライン（segments.txt）
同一動画から複数区間を切り抜いて順番に処理します。動画は一度だけ取得し、各区間を正確に切り出します。

`segments.txt` 例:
```
# url=YOUTUBE_URL
# start end name(optional)
00:01:23 00:02:34 intro
00:10:00 00:11:10 chorus
00:20:00
```

実行:
```bash
python pipeline.py "https://www.youtube.com/watch?v=XXXXX" --segments segments.txt
```

出力は `output/<video_id>/` 配下にまとめられます（`<video_id>_full.mp4`, `<video_id>_01_src.mp4` など）。

### segments.txt でのオプション指定
区間ごとに `key=value` を追加できます（例: `pad=0.12 merge_gap=0.3 cut_video=1 edl=1`）。
```
# start end name(optional) [options...]
// options: trim_words=1 cut_video=1 edl=1 pad=0.12 merge_gap=0.3 words_model=large words_language=ja words_device=cpu words_compute_type=int8 words_beam=5 cut_crf=20 cut_preset=veryfast
00:16:44 00:25:19 intro pad=0.12 merge_gap=0.3 trim_words=1 cut_video=1 edl=1
```

## 単語タイムスタンプで字幕の開始/終了を詰める
「えーと」等のフィラーが音声にはあるが字幕に無い場合、単語タイムスタンプを使って
字幕の開始/終了を話された単語に合わせて詰めます。

### 1) 単語タイムスタンプJSONを作る（faster-whisper）
```bash
.venv/bin/python whisper_words.py output/ui1MX5M_2D0/ui1MX5M_2D0_01_enh.mp4 \
  output/ui1MX5M_2D0/ui1MX5M_2D0_01_words.json \
  --model small --language ja
```

### 2) SRTを詰める
```bash
# words JSON は segments[].words[] を含む形式を想定
python srt_trim_words.py output/ui1MX5M_2D0/ui1MX5M_2D0_01.srt \
  output/ui1MX5M_2D0/ui1MX5M_2D0_01_words.json \
  output/ui1MX5M_2D0/ui1MX5M_2D0_01_words_trimmed.srt
```

## SRTの空白区間をカットして動画を短くする
SRTのキュー間の「字幕が出ていない区間」を切り落として動画を短くします。

```bash
python srt_cut_video.py output/ui1MX5M_2D0/ui1MX5M_2D0_01_words_trimmed.srt \
  output/ui1MX5M_2D0/ui1MX5M_2D0_01_src.mp4 \
  output/ui1MX5M_2D0/ui1MX5M_2D0_01_cut.mp4
```

EDLも出す場合:
```bash
python srt_cut_video.py output/ui1MX5M_2D0/ui1MX5M_2D0_01_words_trimmed.srt \
  output/ui1MX5M_2D0/ui1MX5M_2D0_01_src.mp4 \
  output/ui1MX5M_2D0/ui1MX5M_2D0_01_cut.mp4 \
  --edl output/ui1MX5M_2D0/ui1MX5M_2D0_01_cut.edl --fps 30
```
