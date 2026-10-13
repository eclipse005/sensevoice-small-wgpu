# SenseVoiceSmall wgpu

**SenseVoiceSmall speech recognition in Rust with wgpu.**

**English** · [简体中文](README.zh-CN.md)

A lightweight, cross-platform Rust implementation of [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice), using [wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration.

The goal is simple: run SenseVoiceSmall **locally and natively** without Python or vendor-specific GPU runtimes — and stay **token-for-token aligned** with the official graph.

### Features

* 🦀 Pure Rust
* 🎮 GPU acceleration with wgpu
* 🌍 Cross-platform GPU support (Vulkan / DX12 / Metal)
* 🖥️ Windows / macOS / Linux
* ⚡ CPU fallback (the alignment oracle)
* 📦 Offline local inference
* 🎙️ SenseVoiceSmall — `zh` / `en` / `yue` / `ja` / `ko`
* 🔤 Language forcing, ITN on/off, emotion / event tags
* ✂️ **VAD-agnostic** — you supply the cut points, the engine does the rest
* 🧩 CLI + Rust library
* 🔢 f32 everywhere — no f16 / bf16, no quantisation
* 🎯 Aligned with the official FunASR graph at `dither = 0`

### Install

As a Cargo dependency:

```toml
[dependencies]
sensevoice-small-wgpu = { git = "https://github.com/eclipse005/sensevoice-small-wgpu.git" }
```

Or build the CLI from source:

```bash
git clone https://github.com/eclipse005/sensevoice-small-wgpu.git
cd sensevoice-small-wgpu
cargo build --release        # target/release/sensevoice
```

### Model download

Weights are **not** included in this repository. Download the official checkpoint (rights remain with the original authors) and pass the directory to `--model` unchanged:

- [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) (Hugging Face)
- [iic/SenseVoiceSmall](https://modelscope.cn/models/iic/SenseVoiceSmall) (ModelScope)

The directory must contain `model.pt`, `am.mvn` and `chn_jpn_yue_eng_ko_spectok.bpe.model`.

**No VAD model is bundled.** This crate answers to the same structure as FunASR — the VAD is a separate model and the ASR only wants its cut points. Get segments from any VAD (including FunASR's `fsmn-vad`, e.g. with `max_single_segment_time = 30000`) and pass them as `[start_ms, end_ms]` pairs.

### Quick Start

```bash
sensevoice --model ./SenseVoiceSmall --wav ./audio.wav
```

The run prints the device in use, timings and RTFx on stderr, and the transcript on stdout.

| Option | Description |
|--------|-------------|
| `--model <dir>` | Model directory (default `D:\SenseVoiceSmall\model\SenseVoiceSmall`) |
| `--wav <file>` | wav / mp3 / flac; non-16 kHz is resampled internally |
| `--lang <name>` | `auto` (default) / `zh` / `en` / `yue` / `ja` / `ko` / `nospeech` |
| `--itn` / `--no-itn` | Inverse text normalisation, **on by default** (punctuation + number formatting) |
| `--vad-segments <json>` | Cut points: `[[start_ms,end_ms],…]` or `[{"start_ms":…,"end_ms":…}]`; `@file` reads them from a file |
| `--merge-ms <n>` | `0` (default) keeps the cuts exactly as given. `15000` reproduces FunASR `merge_vad` with `merge_length_s=15` (the cursor starts at 0, so leading silence is pulled in) |
| `--ban-emo-unk` | Mask the CTC `emo_unk` token (id 25009) |
| `--raw` | Print the untagged token text (`<\|en\|><\|NEUTRAL\|>…`) instead of the cleaned transcript |
| `--check-gold <file>` | Compare token ids / text / raw against a reference JSON; exit code 1 on mismatch |
| `--device <spec>` | `auto` (default), `cpu`, `vulkan[:i]`, `dx12`, `metal`, `#n`, or a name substring such as `P104` |
| `--list-devices` | List the devices usable on this machine |

Transcribing with explicit cut points:

```bash
sensevoice --model ./SenseVoiceSmall --wav ./long.wav --vad-segments '[[1690,2280],[4750,6060],[7840,12100]]'
sensevoice --model ./SenseVoiceSmall --wav ./long.wav --vad-segments @cuts.json --merge-ms 15000
```

### Library

```rust
use sensevoice_small_wgpu::{load_audio, Model, Options, Segment};
use std::path::Path;

let samples = load_audio(Path::new("audio.wav"))?;              // 16 kHz mono f32
let mut model = Model::load(Path::new("SenseVoiceSmall"), "P104")?;
model.reserve_samples(samples.len())?;                          // pre-allocate, outside the clock

let opt = Options {
    language: "auto".into(),
    use_itn: true,
    ban_emo_unk: false,
    // Exactly what a VAD produces. Empty = transcribe the whole clip.
    segments: cuts.iter()
        .map(|(s, e)| Segment { start_ms: *s, end_ms: *e })
        .collect(),
    // 0.0 keeps the cuts; 15000.0 runs the official merge_vad (merge_length_s = 15).
    merge_ms: 0.0,
};

let out = model.transcribe(&samples, &opt)?;
println!("{}", out.text);            // out.raw keeps the <|en|><|NEUTRAL|>… tags
for seg in &out.segments {
    println!("{:>8.0}–{:<8.0} {}", seg.start_ms, seg.end_ms, seg.raw);
}
```

A model is loaded once and reused across calls. `Transcript` also carries `language`, `emotion`,
`event`, `itn`, per-segment `token_ids`, and a `Timing { frontend_ms, encode_ms, decode_ms, audio_s, rtfx }`.

Segment semantics match FunASR `inference_with_vad` exactly: `start = int(ms * 16)` (truncated),
`end = min(int(ms * 16), len)`, the audio **after the last cut is not transcribed**, and the
per-segment texts are joined with single spaces.

### Performance

RTFx (audio seconds per wall second), no model load, medians of 5, on a P104-100 / Vulkan box.
Whole-clip transcription:

| Clip | FunASR CPU | **this CPU** | FunASR CUDA (warm) | **this GPU** |
|---|---|---|---|---|
| 15s_en | 30.30 | **70.3** | 193.1 | **201.0** |
| 30s_zh | 31.67 | **95.4** | 239.8 | **246.5** |
| 90s_en | 25.39 | **88.5** | 192.8 | **219.6** |
| 90s_ja | 24.77 | **90.6** | 213.4 | **217.9** |
| 180s_en | 17.57 | **65.7** | 146.1 | **176.0** |
| 180s_zh | 15.66 | **65.0** | 145.6 | **169.3** |

With real VAD cut points (official `fsmn-vad`, `max_single_segment_time = 30000`):

| Clip | official full pipeline | **this GPU** |
|---|---|---|
| 180s_en (50 cuts) | 85 | **190.3** |
| 180s_zh (58 cuts) | 89 | **143.3** |
| 90s_ja (16 cuts) | 57 | **228.2** |
| 90s_en (16 cuts) | 48 | **204.3** |
| 180s_en + `merge_length_s=15` | 118 | **193.9** |
| 180s_zh + `merge_length_s=15` | 101 | **202.9** |

Memory: peak process RAM 969–979 MiB against CUDA's 1.33–1.58 GB (lower on all six clips).
Peak device VRAM is lower on five of six; the 15 s clip is +4.7 %.

### Fidelity

* **f32 only.** GPU weights and activations are f32, accumulation stays f32. The CPU path is the
  reference; the GPU path tracks it token-for-token.
* **`dither = 0` is the target.** FunASR's `WavFrontend` defaults to `dither = 1.0`, which adds
  noise per frame and makes the reference *non-reproducible* — two consecutive `generate()` calls
  on the same audio return different text. Comparisons must force `dither = 0`
  (see `tools/mini_eval/dither0.py`), or the "reference" differs run to run.
* **Evidence** (all re-runnable, see `docs/evidence/README.md`): 6 acceptance clips match FunASR's
  token ids on both CPU and GPU; a 80-clip FLEURS subset (en / zh / ja / yue) is **80/80
  character-identical** to FunASR under matched `dither = 0` conditions; the VAD path is **8/8**
  against `inference_with_vad`; the CPU stage trace is byte-identical over 78 stages.
* Multi-segment runs concatenate the cuts into a single encoder pass with a block-diagonal
  attention/FSMN mask; per-row argmax is bit-identical to running the segments one at a time.

### Environment switches

Every optimisation has a same-binary rollback (see `src/tuning.rs`):

| Variable | Effect when set to `0` |
|---|---|
| `SVS_TILE_SMALL` | use the 128×128 GEMM tile everywhere |
| `SVS_TILE_WIDE` | keep N=512 projections on the 64×64 tile |
| `SVS_FUSE_RELU` / `SVS_FUSE_ADD` / `SVS_FUSE_MERGE` / `SVS_FUSE_SPLIT` | disable the respective fusion |
| `SVS_FSMN_TAP` / `SVS_FSMN_FLAT` | use the checkpoint's FSMN weight layout |
| `SVS_BIAS_FOLD` | keep the separate bias pass |
| `SVS_RB_FUSED` | read the transcript back through the one-shot staging path |
| `SVS_SEG_CONCAT` | process VAD segments one at a time |

Diagnostics: `SVS_TRACE`, `SVS_TRACE_FINE`, `SVS_CPU_PROF`, `SVS_GPU_PROF`, `SVS_GPU_DEBUG`,
`SVS_GPU_DUMP_OPS`, `SVS_GPU_STOP`, `SVS_GPU_CHUNK`.

### Why wgpu?

Instead of relying on CUDA, ROCm, or other vendor-specific runtimes, this project uses **wgpu** as
a unified GPU abstraction, so one Rust binary runs on different platforms and GPU vendors.

### Project Status

✅ **Working and verified** — CLI and library, CPU and GPU, with the fidelity evidence above.
Development focused on GPU throughput and memory; the segmented path, the API surface and the
sibling project's conventions are still being polished.

### Related

* [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — the official model project
* [FunASR](https://github.com/modelscope/FunASR) — the official inference framework this port aligns to
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) — the sibling Rust + wgpu ASR port

### License

MIT, matching `Cargo.toml`.

This repository is an **independent Rust inference implementation** for loading and running the
officially released SenseVoiceSmall weights — not an official FunAudioLLM / Alibaba release, and
not affiliated with the original authors. Model weights remain under the terms of their respective
owners.
