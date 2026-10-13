"""Warm CUDA baseline on the 6 fixtures: same alignment config as the Rust port.

dither=0, no VAD, language=auto, itn=on. One model load, a warmup generate
(first-call cuBLAS/kernel init is not part of any measured fixture), then 3
repeats per fixture with `torch.cuda.synchronize()` around each and the median
reported. Serial on purpose: one GPU, one process.

usage: python bench_cuda_warm.py
"""
import json
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"D:\SenseVoiceSmall"
FIX = {
    "15s_en": 15.00,
    "30s_zh": 30.09,
    "90s_en": 90.00,
    "90s_ja": 89.33,
    "180s_en": 176.31,
    "180s_zh": 180.00,
}
REPEATS = 3

import torch
from funasr import AutoModel

print(f"torch {torch.__version__} | cuda {torch.version.cuda}", flush=True)
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} | {p.multi_processor_count} SM | {p.total_memory // (1024**2)} MiB", flush=True)

t0 = time.perf_counter()
m = AutoModel(
    model=rf"{ROOT}\model\SenseVoiceSmall",
    disable_update=True,
    log_level="ERROR",
    device="cuda:0",
)
m.kwargs["frontend"].dither = 0.0
print(f"model loaded in {time.perf_counter() - t0:.1f}s", flush=True)


def gen(path):
    torch.cuda.synchronize()
    t = time.perf_counter()
    m.generate(input=path, language="auto", use_itn=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t


# Warmup: CUDA context, cuBLAS handles, kernel JIT/caches. Never measured.
w = gen(rf"{ROOT}\fixtures\15s_en.wav")
print(f"warmup: {w:.2f}s (discarded)\n", flush=True)

rows = []
print(f"{'fixture':<9} {'audio_s':>8} {'median_s':>9} {'min_s':>8} {'RTFx':>7}")
for name, audio_s in FIX.items():
    path = rf"{ROOT}\fixtures\{name}.wav"
    ts = [gen(path) for _ in range(REPEATS)]
    med = statistics.median(ts)
    rtfx = audio_s / med
    rows.append({"fixture": name, "audio_s": audio_s, "median_s": round(med, 4),
                 "min_s": round(min(ts), 4), "rtfx": round(rtfx, 2)})
    print(f"{name:<9} {audio_s:>8.2f} {med:>9.3f} {min(ts):>8.3f} {rtfx:>7.1f}", flush=True)

out = rf"{ROOT}\_scratch\bench_cuda_warm.json"
json.dump({"device": "cuda:0", "repeats": REPEATS, "rows": rows}, open(out, "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nsaved {out}", flush=True)
