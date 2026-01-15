# CodeX Integration Guide (HF Server)

This document defines the minimal contract for CodeX to call the Windows HF Server.

## Base

- Base URL: `http://<WINDOWS_IP>:8000`
- All responses are JSON.
- Errors return: `{"detail": "<message>"}` with non-200 status.

## Endpoints

### 1) POST /infer

Run any Hugging Face pipeline.

Request JSON:
```json
{
  "task": "string (required)",
  "model": "string (required)",
  "inputs": "any (required)",
  "params": "object (optional, default {})"
}
```

Response JSON:
```json
{
  "result": "any"
}
```

Notes:
- `task` and `model` are passed to `transformers.pipeline(task, model, device=0)`.
- `inputs` can be string, list, dict, etc. supported by the pipeline.
- `params` is forwarded as keyword args to the pipeline.

Example:
```python
import requests
payload = {
    "task": "text-classification",
    "model": "distilbert-base-uncased-finetuned-sst-2-english",
    "inputs": "I love this product!",
    "params": {}
}
r = requests.post("http://<WINDOWS_IP>:8000/infer", json=payload)
print(r.json())
```

### 2) POST /asr

Whisper (ASR) with optional timestamps and language.

Request:
- Multipart form field: `file` (audio file)
- Query params:
  - `model` (optional, default `openai/whisper-small`)
  - `return_timestamps` (optional: `true` or `word`)
  - `language` (optional: e.g. `ja`, `en`)

Response JSON:
```json
{
  "result": {
    "text": "string",
    "chunks": "optional (if return_timestamps=word)"
  }
}
```

Example:
```python
import requests
with open("sample.wav", "rb") as f:
    r = requests.post(
        "http://<WINDOWS_IP>:8000/asr?model=openai/whisper-small&return_timestamps=word&language=ja",
        files={"file": ("sample.wav", f, "audio/wav")},
    )
print(r.json())
```

## Operational Notes

- CUDA required. Server will refuse to start if CUDA is unavailable.
- Pipelines are cached in-process by `(task, model)` key.
- Models download on first use and are cached under `E:\hf_cache`.
- `ffmpeg` must be available on PATH for `/asr` to load audio files.
