import json, subprocess, sys, time
sys.stdout.reconfigure(encoding="utf-8")
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
EXE = r"D:\SenseVoiceSmall\SenseVoiceSmall-wgpu\target\release\sensevoice.exe"
py = json.load(open("api_py.json", encoding="utf-8"))
rows = []
for wav, sets in CASES.items():
    for name, segs in sets.items():
        for merged in (False, True):
            key = f"{wav}|{name}|merge={int(merged)}"
            args = [EXE, "--wav", rf"D:\SenseVoiceSmall\fixtures\{wav}.wav", "--device", "P104", "--raw",
                    "--vad-segments", json.dumps(segs)]
            if merged: args += ["--merge-ms", "15000"]
            t0 = time.perf_counter()
            p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
            wall = time.perf_counter() - t0
            rtfx = None
            for line in (p.stderr or "").splitlines():
                if line.startswith("rtfx: "): rtfx = float(line.split()[1])
            same = (p.stdout or "").strip() == py[key]["text"].strip()
            rows.append((key, same, len(py[key]["text"]), rtfx, wall, py[key]["wall"]))
            print(f"{key}: 文本一致={same} chars py={len(py[key]['text'])} rs={len((p.stdout or '').strip())} rtfx={rtfx}")
json.dump(rows, open("api_cmp.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
ok = sum(1 for r in rows if r[1])
print(f"\n*** API 对齐: {ok}/{len(rows)} 文本完全一致 ***")
