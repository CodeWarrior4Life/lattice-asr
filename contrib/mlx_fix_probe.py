#!/usr/bin/env python
import os, threading, wave, tempfile
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from parakeet_mlx import from_pretrained
import mlx.core as mx
from mlx.utils import tree_flatten

fd, wav = tempfile.mkstemp(suffix=".wav"); os.close(fd)
with wave.open(wav, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00"*32000)

def run(m, tag):
    try:
        r = m.transcribe(wav); print(f"[{tag}] OK {r.text!r}", flush=True)
    except Exception as e:
        print(f"[{tag}] FAIL {e!r}", flush=True)

# Variant 1: load on A, FULL eval of params on A, run on fresh thread B
box = {}
def load_eval():
    m = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    mx.eval([v for _, v in tree_flatten(m.parameters())])
    run(m, "V1-A-after-eval")  # a first pass on A too (warm)
    box["m"] = m
t = threading.Thread(target=load_eval); t.start(); t.join()
t = threading.Thread(target=run, args=(box["m"], "V1-B-fresh-thread")); t.start(); t.join()

# Variant 2: single dedicated executor thread for load + every transcribe
ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
m2 = ex.submit(from_pretrained, "mlx-community/parakeet-tdt-0.6b-v3").result()
for i in range(3):
    ex.submit(run, m2, f"V2-dedicated-{i}").result()
# and prove m2 still fails off-thread (control)
t = threading.Thread(target=run, args=(m2, "V2-control-other-thread")); t.start(); t.join()
os.unlink(wav)
