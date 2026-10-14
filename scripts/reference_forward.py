# Reference SenseVoiceSmall forward (dither=0) plus weight/vocab export.
# Official FunASR WavFrontend defaults dither to 1.0, which makes runs
# non-deterministic. This script is the alignment target: same math, dither 0.
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi

sys.stdout.reconfigure(encoding="utf-8")
torch.set_grad_enabled(False)

MODEL = Path(r"D:\SenseVoiceSmall\model\SenseVoiceSmall")
ROOT = Path(r"D:\SenseVoiceSmall\SenseVoiceSmall-wgpu")
LID = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
EMO_UNK = 25009
BLANK = 0

EMO_DICT = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}
EVENT_DICT = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    # Official FunASR `event_dict` really maps Cough to 🤧 (same as Sneeze);
    # `emoji_dict` says 😷 but its values are never used by the renderer.
    "<|Cough|>": "🤧",
}
LANG_DICT = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}
EMOJI_DICT = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}
EMO_SET = {"😊", "😔", "😡", "😰", "🤢", "😮"}
EVENT_SET = {"🎼", "👏", "😀", "😭", "🤧", "😷"}


def load_cmvn(path):
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.split()]
    means, vars_ = None, None
    for i, line in enumerate(lines):
        parts = line.split()
        if parts[0] == "<AddShift>":
            nxt = lines[i + 1].split()
            if nxt and nxt[0] == "<LearnRateCoef>":
                means = [float(x) for x in nxt[3:-1]]
        elif parts[0] == "<Rescale>":
            nxt = lines[i + 1].split()
            if nxt and nxt[0] == "<LearnRateCoef>":
                vars_ = [float(x) for x in nxt[3:-1]]
    return torch.tensor([means, vars_], dtype=torch.float32)


def apply_lfr(inputs, lfr_m, lfr_n):
    import numpy as np

    t = inputs.shape[0]
    t_lfr = int(np.ceil(t / lfr_n))
    left = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left, inputs))
    t = inputs.shape[0]
    feat_dim = inputs.shape[-1]
    strides = (lfr_n * feat_dim, 1)
    sizes = (t_lfr, lfr_m * feat_dim)
    last_idx = (t - lfr_m) // lfr_n + 1
    num_padding = lfr_m - (t - last_idx * lfr_n)
    if num_padding > 0:
        num_padding = (2 * lfr_m - 2 * t + (t_lfr - 1 + last_idx) * lfr_n) / 2 * (t_lfr - last_idx)
        inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_padding))
    return inputs.as_strided(sizes, strides).clone()


def apply_cmvn(inputs, cmvn):
    dim = inputs.shape[-1]
    inputs = inputs + cmvn[0, :dim]
    inputs = inputs * cmvn[1, :dim]
    return inputs


def fbank_lfr_cmvn(wav_1d, cmvn):
    wave = wav_1d.unsqueeze(0) * (1 << 15)
    mat = kaldi.fbank(
        wave,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=16000.0,
        snip_edges=True,
    )
    n_fbank = mat.shape[0]
    trace("fbank", mat)
    mat = apply_lfr(mat, 7, 6)
    trace("lfr", mat)
    mat = apply_cmvn(mat, cmvn)
    trace("cmvn", mat)
    return mat, n_fbank


def posenc(x):
    b, t, depth = x.shape
    positions = torch.arange(1, t + 1, dtype=x.dtype)
    half = depth / 2
    log_inc = torch.log(torch.tensor(10000.0, dtype=x.dtype)) / (half - 1)
    inv = torch.exp(torch.arange(half, dtype=x.dtype) * (-log_inc))
    scaled = positions[:, None] * inv[None, :]
    enc = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
    return x + enc.unsqueeze(0)


def trace(name, x):
    if __import__("os").environ.get("SVS_TRACE") == "1":
        flat = x.detach().double().reshape(-1)
        print(
            f"trace {name} sum={flat.sum().item():.6f} max={flat.abs().max().item():.6f} n={flat.numel()}",
            flush=True,
        )


def sanm(x, w):
    qkv = F.linear(x, w["qkv_w"], w["qkv_b"])
    q, k, v = qkv.chunk(3, dim=-1)
    mem = v.transpose(1, 2)
    mem = F.pad(mem, (5, 5))
    mem = F.conv1d(mem, w["fsmn"], bias=None, stride=1, padding=0, groups=v.shape[-1])
    mem = mem.transpose(1, 2) + v
    b, t, _ = q.shape
    h, dk = 4, 128
    qh = q.view(b, t, h, dk).transpose(1, 2) * (dk ** -0.5)
    kh = k.view(b, t, h, dk).transpose(1, 2)
    vh = v.view(b, t, h, dk).transpose(1, 2)
    attn = torch.softmax(qh @ kh.transpose(-2, -1), dim=-1)
    ctx = (attn @ vh).transpose(1, 2).contiguous().view(b, t, h * dk)
    return F.linear(ctx, w["out_w"], w["out_b"]) + mem


def layer(x, w, residual):
    h = F.layer_norm(x, (x.shape[-1],), w["n1_w"], w["n1_b"], 1e-5)
    y = sanm(h, w)
    x = (x + y) if residual else y
    h = F.layer_norm(x, (x.shape[-1],), w["n2_w"], w["n2_b"], 1e-5)
    ff = F.linear(F.relu(F.linear(h, w["w1_w"], w["w1_b"])), w["w2_w"], w["w2_b"])
    return x + ff


def pack_layer(sd, prefix):
    fsmn = sd[f"{prefix}.self_attn.fsmn_block.weight"]
    return {
        "n1_w": sd[f"{prefix}.norm1.weight"],
        "n1_b": sd[f"{prefix}.norm1.bias"],
        "qkv_w": sd[f"{prefix}.self_attn.linear_q_k_v.weight"],
        "qkv_b": sd[f"{prefix}.self_attn.linear_q_k_v.bias"],
        "fsmn": fsmn,
        "out_w": sd[f"{prefix}.self_attn.linear_out.weight"],
        "out_b": sd[f"{prefix}.self_attn.linear_out.bias"],
        "n2_w": sd[f"{prefix}.norm2.weight"],
        "n2_b": sd[f"{prefix}.norm2.bias"],
        "w1_w": sd[f"{prefix}.feed_forward.w_1.weight"],
        "w1_b": sd[f"{prefix}.feed_forward.w_1.bias"],
        "w2_w": sd[f"{prefix}.feed_forward.w_2.weight"],
        "w2_b": sd[f"{prefix}.feed_forward.w_2.bias"],
    }


def encode(feats, sd):
    x = feats.unsqueeze(0) * (512 ** 0.5)
    x = posenc(x)
    trace("input", x)
    layers = [pack_layer(sd, "encoder.encoders0.0")]
    i = 0
    while f"encoder.encoders.{i}.norm1.weight" in sd:
        layers.append(pack_layer(sd, f"encoder.encoders.{i}"))
        i += 1
    tp = []
    i = 0
    while f"encoder.tp_encoders.{i}.norm1.weight" in sd:
        tp.append(pack_layer(sd, f"encoder.tp_encoders.{i}"))
        i += 1
    for li, w in enumerate(layers):
        x = layer(x, w, residual=(li != 0))
        trace(f"L{li}", x)
    x = F.layer_norm(x, (512,), sd["encoder.after_norm.weight"], sd["encoder.after_norm.bias"], 1e-5)
    trace("after_norm", x)
    for li, w in enumerate(tp):
        x = layer(x, w, residual=True)
        trace(f"TP{li}", x)
    x = F.layer_norm(x, (512,), sd["encoder.tp_norm.weight"], sd["encoder.tp_norm.bias"], 1e-5)
    trace("tp_norm", x)
    return x


def greedy(logits):
    ids = logits.argmax(dim=-1)
    out = []
    prev = None
    for i in ids.tolist():
        if i != prev:
            if i != BLANK:
                out.append(i)
            prev = i
    return out


def manual_decode(sp, ids):
    raw = bytearray()
    for i in ids:
        if sp.is_control(i):
            continue
        p = sp.id_to_piece(i)
        if len(p) == 6 and p.startswith("<0x") and p.endswith(">"):
            raw.append(int(p[3:5], 16))
        else:
            raw.extend(p.replace("\u2581", " ").encode("utf-8"))
    return raw.decode("utf-8")


def format_str_v2(s):
    counts = {}
    for sptk in EMOJI_DICT:
        counts[sptk] = s.count(sptk)
        s = s.replace(sptk, "")
    emo = "<|NEUTRAL|>"
    for e in EMO_DICT:
        if counts[e] > counts[emo]:
            emo = e
    for e in EVENT_DICT:
        if counts[e] > 0:
            s = EVENT_DICT[e] + s
    s = s + EMO_DICT[emo]
    for emoji in sorted(EMO_SET | EVENT_SET):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    return s.strip()


def rich_post(s):
    def get_emo(text):
        return text[-1] if text and text[-1] in EMO_SET else None

    def get_event(text):
        return text[0] if text and text[0] in EVENT_SET else None

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in LANG_DICT:
        s = s.replace(lang, "<|lang|>")
    parts = [format_str_v2(p).strip(" ") for p in s.split("<|lang|>")]
    new_s = " " + parts[0]
    cur_event = get_event(new_s)
    for i in range(1, len(parts)):
        if len(parts[i]) == 0:
            continue
        if get_event(parts[i]) == cur_event and get_event(parts[i]) is not None:
            parts[i] = parts[i][1:]
        if len(parts[i]) == 0:
            continue
        cur_event = get_event(parts[i])
        if get_emo(parts[i]) is not None and get_emo(parts[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += parts[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def export_weights(sd):
    from safetensors.torch import save_file

    out = MODEL / "model.safetensors"
    if out.exists() and out.stat().st_size > 0:
        print(f"weights exist {out}", flush=True)
        return
    tensors = {k: v.detach().cpu().contiguous().float() for k, v in sd.items()}
    save_file(tensors, str(out))
    print(f"wrote {out}", flush=True)


def export_tokens():
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL / "chn_jpn_yue_eng_ko_spectok.bpe.model"))
    n = sp.get_piece_size()
    pieces = [sp.id_to_piece(i) for i in range(n)]
    control = [i for i in range(n) if sp.is_control(i)]
    payload = {"pieces": pieces, "control_ids": control}
    path = MODEL / "tokens.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path} vocab={n} control={control[:12]}", flush=True)
    return sp


def load_wav(path):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav[0].float()


def run_one(wav_path, sd, cmvn, sp, lang, use_itn):
    wav = load_wav(wav_path)
    feats, n_fbank = fbank_lfr_cmvn(wav, cmvn)
    lid = LID[lang]
    itn = 14 if use_itn else 15
    embed = sd["embed.weight"]
    queries = embed[torch.tensor([lid, 1, 2, itn])]
    feats_q = torch.cat([queries, feats], dim=0)
    hidden = encode(feats_q, sd)[0]
    logits = F.linear(hidden, sd["ctc.ctc_lo.weight"], sd["ctc.ctc_lo.bias"])
    trace("logits", logits)
    ids = greedy(logits)
    raw_sp = sp.decode(ids)
    raw_manual = manual_decode(sp, ids)
    if raw_sp != raw_manual:
        print("DECODE MISMATCH", flush=True)
        print("sp    ", raw_sp[:200], flush=True)
        print("manual", raw_manual[:200], flush=True)
        raise SystemExit(1)
    text = rich_post(raw_sp)
    meta = {
        "wav": str(wav_path),
        "language": lang,
        "use_itn": use_itn,
        "audio_samples": int(wav.numel()),
        "n_fbank": int(n_fbank),
        "n_lfr": int(feats.shape[0]),
        "n_input": int(feats_q.shape[0]),
        "feat_sum": float(feats.sum()),
        "text": text,
        "raw": raw_sp,
        "token_ids": ids,
    }
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--wav", action="append", default=[])
    args = ap.parse_args()
    print("loading model.pt", flush=True)
    sd = torch.load(MODEL / "model.pt", map_location="cpu", weights_only=False)
    export_weights(sd)
    sp = export_tokens()
    if args.export_only:
        return
    cmvn = load_cmvn(MODEL / "am.mvn")
    wavs = args.wav or [
        r"D:\SenseVoiceSmall\fixtures\15s_en.wav",
        r"D:\SenseVoiceSmall\fixtures\30s_zh.wav",
        r"D:\SenseVoiceSmall\fixtures\90s_en.wav",
        r"D:\SenseVoiceSmall\fixtures\90s_ja.wav",
        r"D:\SenseVoiceSmall\fixtures\180s_en.wav",
        r"D:\SenseVoiceSmall\fixtures\180s_zh.wav",
    ]
    gold = ROOT / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    for wav in wavs:
        print(f"forward {wav}", flush=True)
        import time

        t0 = time.perf_counter()
        meta = run_one(wav, sd, cmvn, sp, "auto", True)
        dt = time.perf_counter() - t0
        audio_s = meta["audio_samples"] / 16000
        meta["python_elapsed_s"] = dt
        meta["python_rtfx"] = audio_s / dt
        name = Path(wav).stem
        (gold / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"{name} lfr={meta['n_lfr']} feat_sum={meta['feat_sum']:.4f} "
            f"rtfx={meta['python_rtfx']:.2f} text={meta['text'][:120]}",
            flush=True,
        )
        print(f"raw={meta['raw'][:180]}", flush=True)


if __name__ == "__main__":
    main()
