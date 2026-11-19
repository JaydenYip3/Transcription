# transcribe

Local speech-to-text with timestamps, running on the GPU (RTX 3060 Ti, 8 GB).
Takes an MP3 (or any common audio/video file) and writes an `.srt` subtitle file.

- **Model:** Whisper `large-v3-turbo` via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Runs fully locally.** Audio never leaves the machine. The internet is only
  used once, to download the model (~1.6 GB).
- **Languages:** ~99, detected automatically.

## Usage

```bash
transcribe yourfile.mp3
```

Works from any folder (the `transcribe` alias is defined in `~/.bashrc`).
Without the alias:

```bash
/root/code/transcribe/transcribe.py yourfile.mp3
```

## Output

1. **Terminal:** each segment is printed as it's transcribed:
   `[00:00:01,240 -> 00:00:04,880] Hello there...`
2. **`.srt` file** next to the input, with the same name:
   `~/Music/interview.mp3` → `~/Music/interview.srt`.
   An existing `.srt` with that name is overwritten.

The `.srt` is plain text; open it in any editor or load it as subtitles in VLC.

## How it works

- **First line** (`#!/root/code/transcribe/.venv/bin/python`) runs the script
  with the project's virtualenv, so no `source .venv/bin/activate` is needed.
- **CUDA libraries:** cuBLAS and cuDNN come from pip packages inside `.venv`.
  If their folders aren't on `LD_LIBRARY_PATH`, the script adds them and
  restarts itself, so no manual `export` is needed.
- **Model cache:** the script loads the model offline from
  `~/.cache/huggingface/hub/`. If it isn't cached, it downloads it once.

## Setup (already done)

For reference, in case it needs to be rebuilt:

```bash
cd /root/code/transcribe
python3 -m venv .venv
.venv/bin/pip install faster-whisper "av<19" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
chmod +x transcribe.py
echo "alias transcribe='/root/code/transcribe/transcribe.py'" >> ~/.bashrc
```

Requires an NVIDIA driver (check with `nvidia-smi`).

## Customizing

Edit `transcribe.py`:

| Want | Change |
|---|---|
| Force a language | `model.transcribe(audio, language="es", word_timestamps=True)` |
| Best accuracy / translate to English | Model `"large-v3"` (~4.5 GB VRAM), and `task="translate"` for translation. `large-v3-turbo` translates poorly. |
| Less VRAM | `compute_type="int8_float16"` |

## Troubleshooting

- **`transcribe: command not found`:** open a new terminal, or run `source ~/.bashrc`.
- **`libcublas.so.12` / cuDNN errors:** reinstall the libraries:
  `.venv/bin/pip install --force-reinstall nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`
- **`open() got an unexpected keyword argument 'metadata_errors'`:** PyAV 19+
  removed that option, but faster-whisper 1.2.1 still passes it. Keep PyAV below 19:
  `.venv/bin/pip install "av<19"` (18.1.0 is installed and works).
- **CUDA out of memory:** close other GPU apps, or use `compute_type="int8_float16"`.
