import sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, 'D:/SenseVoiceSmall/_scratch/mini_eval')
import dither0
from common import clips
device, out_path = sys.argv[1], sys.argv[2]
cs = clips()
from funasr import AutoModel
m = AutoModel(model=r"D:\SenseVoiceSmall\model\SenseVoiceSmall", disable_update=True, log_level="ERROR", device=device)
m.kwargs["frontend"].dither = 0.0
m.generate(input=cs[0]['wav'], language=cs[0]['lang'], use_itn=True, batch_size_s=300, output_dir=None)
out = {}
for i, c in enumerate(cs):
    t0 = time.perf_counter()
    r = m.generate(input=c['wav'], language=c['lang'], use_itn=True, batch_size_s=300, output_dir=None)
    wall = time.perf_counter() - t0
    out[c['id']] = {'lang': c['lang'], 'dur': c['dur'], 'wall': wall, 'rtfx': c['dur']/wall,
                    'text': (r[0]['text'] if r else ''), 'ref': c['ref']}
    json.dump(out, open(out_path,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[{i+1}/{len(cs)}] {c['id']} lang={c['lang']} wall={wall:.2f}", flush=True)
