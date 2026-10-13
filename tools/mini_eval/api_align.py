import sys, json, time, torch, torch.nn as nn
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, 'D:/SenseVoiceSmall/_scratch/mini_eval')
import dither0  # 强制 dither=0
from funasr import AutoModel

class StubVad(nn.Module):
    """只用切点：完全替代 VAD 模型，返回给定 segments。"""
    def __init__(self, segs):
        super().__init__()
        self.segs = segs
        self.dummy = nn.Parameter(torch.zeros(1))
    def inference(self, data_in=None, key=None, **kw):
        return ([{"key": k, "value": [list(s) for s in self.segs]} for k in key], {})

CASES = {
    "180s_en": {
        "fixed10": [[i*10000, min((i+1)*10000, 180000)] for i in range(18)],
        "gaps": [[1000,9000],[13000,26000],[31000,48000],[55000,72000],[80000,95000],[100000,118000],[125000,140000],[150000,175000]],
    },
    "90s_ja": {
        "fixed10": [[i*10000, min((i+1)*10000, 90000)] for i in range(9)],
        "gaps": [[500,7000],[15000,31000],[40000,58000],[65000,88000]],
    },
}
m = AutoModel(model=r"D:\SenseVoiceSmall\model\SenseVoiceSmall", disable_update=True, log_level="ERROR", device="cuda:0")
out = {}
for wav, sets in CASES.items():
    path = rf"D:\SenseVoiceSmall\fixtures\{wav}.wav"
    for name, segs in sets.items():
        for merged in (False, True):
            m.vad_model = StubVad(segs); m.vad_kwargs = {}
            t0 = time.perf_counter()
            r = m.inference_with_vad(path, input_len=None, language="auto", use_itn=True,
                                     merge_vad=merged, merge_length_s=15, batch_size_s=300)
            wall = time.perf_counter() - t0
            key = f"{wav}|{name}|merge={int(merged)}"
            out[key] = {"text": r[0]["text"], "wall": wall}
            print(f"{key}: wall={wall:.3f}s chars={len(r[0]['text'])}")
json.dump(out, open("api_py.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved api_py.json")
