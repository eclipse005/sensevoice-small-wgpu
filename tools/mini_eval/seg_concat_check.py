"""拼接路径 vs 逐段路径：逐行 CTC argmax 是否完全一致。

SVS_DUMP_FRAMES=<path> 会导出本次 encode 的逐行 argmax；拼接模式一次导出全部行，
逐段模式每段导出一次。本脚本按段的行区间切分拼接受后与逐段结果比较。
"""
import json, os, subprocess, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
EXE = r"D:\SenseVoiceSmall\SenseVoiceSmall-wgpu\target\release\sensevoice.exe"
WAV = r"D:\SenseVoiceSmall\fixtures\15s_en.wav"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_frames")
os.makedirs(OUT, exist_ok=True)
CASES = {
    "2段": [[0, 7000], [8000, 15000]],
    "3段": [[0, 5000], [6000, 11000], [12000, 15000]],
    "5段": [[0, 3000], [3500, 6000], [7000, 9000], [10000, 12000], [13000, 15000]],
}
def run(segs, mode, tag):
    env = dict(os.environ)
    env["SVS_SEG_CONCAT"] = mode
    env["SVS_DUMP_FRAMES"] = os.path.join(OUT, f"{tag}.bin")
    args = [EXE, "--wav", WAV, "--device", "P104", "--raw", "--vad-segments", json.dumps(segs)]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return r.stdout.strip(), os.path.join(OUT, f"{tag}.bin")
ok = True
for name, segs in CASES.items():
    text_all, f_all = run(segs, "1", f"{name}_all")
    frames_all = np.fromfile(f_all, dtype=np.int32)
    off = 0; total_bad = 0; detail = []
    for i, s in enumerate(segs):
        text_one, f_one = run([s], "0", f"{name}_seg{i}")
        fr = np.fromfile(f_one, dtype=np.int32)
        got = frames_all[off:off + fr.size]
        bad = int((got != fr).sum())
        detail.append(f"段{i}: {bad}/{fr.size}")
        total_bad += bad
        off += fr.size
    ok &= (total_bad == 0)
    print(f"{name}: 逐行 argmax 不一致总数 = {total_bad}  ({', '.join(detail)})")
print("所有用例逐行一致:", ok)
