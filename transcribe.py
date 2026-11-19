#!/root/code/transcribe/.venv/bin/python
import os
import sys

# Make the venv's CUDA libraries (cuBLAS, cuDNN) findable without a manual
# export: if they aren't on LD_LIBRARY_PATH yet, add them and restart this script.
import nvidia.cublas.lib
import nvidia.cudnn.lib

cuda_dirs = [*nvidia.cublas.lib.__path__, *nvidia.cudnn.lib.__path__]
current = os.environ.get("LD_LIBRARY_PATH", "")
if not all(d in current.split(":") for d in cuda_dirs):
    os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_dirs + ([current] if current else []))
    os.execv(sys.executable, [sys.executable] + sys.argv)

from faster_whisper import WhisperModel

if len(sys.argv) < 2:
    sys.exit("usage: transcribe.py <audio file>")

audio = sys.argv[1]

# Use the cached model offline; download it only if it isn't cached yet.
try:
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16", local_files_only=True)
except Exception:
    print("Model not cached yet, downloading (~1.6 GB, one time only)...")
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

segments, info = model.transcribe(audio, word_timestamps=True)

print(f"Detected language: {info.language} ({info.language_probability:.0%})")


def ts(t):
    h, m = divmod(int(t), 3600)
    m, s = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02},{int((t % 1) * 1000):03}"


with open(audio.rsplit(".", 1)[0] + ".srt", "w", encoding="utf-8") as f:
    for i, s in enumerate(segments, 1):
        print(f"[{ts(s.start)} -> {ts(s.end)}] {s.text.strip()}")
        f.write(f"{i}\n{ts(s.start)} --> {ts(s.end)}\n{s.text.strip()}\n\n")
