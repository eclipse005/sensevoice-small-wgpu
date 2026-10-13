import sys, json, time, subprocess
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, 'D:/SenseVoiceSmall/_scratch/mini_eval')
from common import clips
device, out_path = sys.argv[1], sys.argv[2]
EXE = r'D:\SenseVoiceSmall\SenseVoiceSmall-wgpu\target\release\sensevoice.exe'
cs = clips(); out = {}
for i, c in enumerate(cs):
    t0 = time.perf_counter()
    p = subprocess.run([EXE,'--wav',c['wav'],'--device',device,'--raw','--lang',c['lang']],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    wall = time.perf_counter()-t0
    rtfx = None
    for line in (p.stderr or '').splitlines():
        if line.startswith('rtfx: '): rtfx = float(line.split()[1])
    out[c['id']] = {'lang': c['lang'], 'dur': c['dur'], 'wall': wall, 'rtfx': rtfx, 'text': (p.stdout or '').strip()}
    json.dump(out, open(out_path,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[{i+1}/{len(cs)}] {c['id']} lang={c['lang']} wall={wall:.2f} rtfx={rtfx}", flush=True)
