#!/usr/bin/env python
"""Reproduce: load parakeet on thread A, transcribe on thread B (fresh thread), then C."""
import os, sys, threading, wave, tempfile, traceback, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from parakeet_mlx import from_pretrained
import mlx.core as mx

fd, wav = tempfile.mkstemp(suffix=".wav"); os.close(fd)
with wave.open(wav, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00"*32000)

model = {}
def load():
    print("[A] default_stream", mx.default_stream(mx.cpu), "gpu", mx.default_stream(mx.gpu), flush=True)
    model["m"] = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    print("[A] loaded", flush=True)
def run(tag):
    try:
        print(f"[{tag}] default_stream cpu", mx.default_stream(mx.cpu), flush=True)
        r = model["m"].transcribe(wav)
        print(f"[{tag}] OK text={r.text!r}", flush=True)
    except Exception as e:
        print(f"[{tag}] FAIL {e!r}", flush=True)

t = threading.Thread(target=load); t.start(); t.join()
for tag in ("B", "C", "D"):
    t = threading.Thread(target=run, args=(tag,)); t.start(); t.join()
print("main-thread:"); run("MAIN")
os.unlink(wav)
