import json, sys, os, re, statistics, itertools, unicodedata
sys.stdout.reconfigure(encoding="utf-8")
TAG = re.compile(r"<\|[^|]*\|>")
def norm(t):
    t = TAG.sub("", t or "")
    t = unicodedata.normalize("NFKC", t).strip()
    return re.sub(r"\s+", " ", t)
FILES = {"py_cuda":"out_py_cuda.json","py_cpu":"out_py_cpu.json",
         "rust_wgpu":"out_rust_wgpu.json","rust_cpu":"out_rust_cpu.json"}
data = {k: json.load(open(v, encoding="utf-8")) for k, v in FILES.items() if os.path.exists(v)}
backs = [k for k in ["py_cuda","py_cpu","rust_wgpu","rust_cpu"] if k in data]
ids = sorted(set.intersection(*[set(d) for d in data.values()]))
MAN = json.load(open('D:/mini_asr_data/manifest.json', encoding='utf-8'))
REF = {c['id']: c['ref'] for c in MAN['clips']}
for b in backs:
    for i in data[b]:
        data[b][i].setdefault('ref', REF.get(i, ''))
print(f"# 共同片段 {len(ids)} 条，后端: {backs}\n")
langs = sorted({data[backs[0]][i]['lang'] for i in ids})
# ---- 转录一致率（两两，去除富文本标签后精确比较）----
print("## 1. 转录一致率（去标签后精确相等）")
hdr = "| 语种 | " + " | ".join(f"{a} vs {b}" for a, b in itertools.combinations(backs, 2)) + " |"
print(hdr); print("|" + "---|" * (len(list(itertools.combinations(backs, 2))) + 1))
def agree(a, b, sub):
    n = sum(1 for i in sub if norm(data[a][i]['text']) == norm(data[b][i]['text']))
    return f"{n}/{len(sub)}" if sub else "-"
for lg in langs + ["=全部="]:
    sub = [i for i in ids if lg == "=全部=" or data[backs[0]][i]['lang'] == lg]
    print(f"| {lg} | " + " | ".join(agree(a, b, sub) for a, b in itertools.combinations(backs, 2)) + " |")
print()
# ---- RTFx ----
print("## 2. RTFx（各后端自身口径：py=dur/端到端generate；rust=CLI rtfx，均不含模型加载）")
print("| 语种 | n | " + " | ".join(backs) + " |")
print("|" + "---|" * (len(backs) + 2))
def med(b, sub):
    v = [data[b][i]['rtfx'] for i in sub if data[b][i].get('rtfx')]
    return f"{statistics.median(v):.1f}" if v else "-"
for lg in langs + ["=全部="]:
    sub = [i for i in ids if lg == "=全部=" or data[backs[0]][i]['lang'] == lg]
    print(f"| {lg} | {len(sub)} | " + " | ".join(med(b, sub) for b in backs) + " |")
print()
# ---- 与 ref 的 CER/WER ----
try:
    import jiwer
    print("## 3. 与人工 reference 的 CER/WER（去标签、NFKC、空白归一）")
    print("| 语种 | " + " | ".join(backs) + " |")
    print("|" + "---|" * (len(backs) + 1))
    for lg in langs:
        sub = [i for i in ids if data[backs[0]][i]['lang'] == lg]
        row = []
        for b in backs:
            r = []; h = []
            for i in sub:
                r.append(norm(data[b][i].get('ref', ''))); h.append(norm(data[b][i]['text']))
            if not any(r): row.append("-"); continue
            use_cer = lg in ('zh','ja','yue')
            f = jiwer.cer if use_cer else jiwer.wer
            try: row.append(f"{f(r, h)*100:.2f}")
            except Exception as e: row.append("err")
        print(f"| {lg} | " + " | ".join(row) + " |")
    print()
except ImportError:
    print("(jiwer 未安装，跳过 §3)\n")
# ---- 不一致样例 ----
print("## 4. 不一致明细（基准 py_cuda vs 其余；去标签）")
for b in backs:
    if b == 'py_cuda': continue
    bad = [i for i in ids if norm(data['py_cuda'][i]['text']) != norm(data[b][i]['text'])]
    print(f"\n### py_cuda vs {b}: {len(bad)}/{len(ids)} 不一致")
    for i in bad[:8]:
        print(f"- {i} [{data['py_cuda'][i]['lang']}]")
        print(f"    py_cuda : {norm(data['py_cuda'][i]['text'])[:150]}")
        print(f"    {b:8s}: {norm(data[b][i]['text'])[:150]}")
print("## 5. 不一致条的字符级差异（py_cuda 为基准；对 hyp 之间算 CER，越低表示差别越小）")
try:
    import jiwer
    for b in backs:
        if b == 'py_cuda': continue
        bad = [i for i in ids if norm(data['py_cuda'][i]['text']) != norm(data[b][i]['text'])]
        if not bad: continue
        cers = []
        for i in bad:
            a, c = norm(data['py_cuda'][i]['text']), norm(data[b][i]['text'])
            try: cers.append(jiwer.cer([a], [c]))
            except Exception: pass
        tot = sum(len(norm(data['py_cuda'][i]['text'])) for i in bad)
        print(f"- py_cuda vs {b}: {len(bad)}/{len(ids)} 不一致，平均 CER {100*sum(cers)/len(cers):.2f}%（这些条合计 {tot} 字符）")
except ImportError:
    pass
print()
print("## 6. 富文本标签一致率（语言/情感/事件/ITN 标签）")
tag = re.compile(r"<\|([^|]*)\|>")
for b in backs:
    if b == 'py_cuda': continue
    n = sum(1 for i in ids if tag.findall(data['py_cuda'][i]['text']) == tag.findall(data[b][i]['text']))
    print(f"- py_cuda vs {b}: 标签序列一致 {n}/{len(ids)}")
json.dump({'ids': ids, 'backs': backs}, open('compare_meta.json','w',encoding='utf-8'), ensure_ascii=False)
