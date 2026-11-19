# transcribe

Local speech-to-text with timestamps. Takes an MP3 (or any common audio or video file) and writes an `.srt` subtitle file.

## About

Runs Whisper `large-v3-turbo` through [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on an RTX 3060 Ti. Everything runs locally: the internet is only used once, to download the ~1.6 GB model. It detects about 99 languages automatically.

## Usage

```bash
transcribe yourfile.mp3
```

Segments print as they're transcribed:

```
[00:00:01,240 -> 00:00:04,880] Hello there...
```

The `.srt` is saved next to the input with the same name (`interview.mp3` → `interview.srt`). An existing file with that name is overwritten.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install faster-whisper "av<19" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
chmod +x transcribe.py
echo "alias transcribe='/root/code/transcribe/transcribe.py'" >> ~/.bashrc
```

Requires an NVIDIA driver. The script uses the venv's Python directly and sets up the CUDA library paths itself, so you don't need to activate the venv or export anything.

## Customizing

Edit `transcribe.py`:

| Want | Change |
|---|---|
| Force a language | `language="es"` in `model.transcribe(...)` |
| Better accuracy or translation | model `"large-v3"` (~4.5 GB VRAM) with `task="translate"` |
| Less VRAM | `compute_type="int8_float16"` |

## Troubleshooting

- **cuBLAS / cuDNN errors:** reinstall them with `.venv/bin/pip install --force-reinstall nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`.
- **`unexpected keyword argument 'metadata_errors'`:** PyAV 19+ breaks faster-whisper 1.2.1, so keep it pinned to `av<19`.
- **CUDA out of memory:** close other GPU apps or use `int8_float16`.
