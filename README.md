# SenseVoiceSmall wgpu

**SenseVoiceSmall speech recognition in Rust with wgpu.**

**English** · [简体中文](README.zh-CN.md)

A lightweight, cross-platform Rust implementation of [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice), using [wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration.

The goal is simple: run SenseVoiceSmall **locally and natively** without Python or vendor-specific GPU runtimes.

### Features

* 🦀 Pure Rust
* 🎮 GPU acceleration with wgpu
* 🌍 Cross-platform GPU support (Vulkan / DX12 / Metal)
* 🖥️ Windows / macOS / Linux
* ⚡ CPU fallback
* 📦 Offline local inference
* 🎙️ SenseVoiceSmall — `zh` / `en` / `yue` / `ja` / `ko`
* 🔤 Language selection, ITN on/off, emotion / event tags
* ✂️ VAD-agnostic — you supply the cut points
* 🧩 CLI + Rust library
* 🔢 f32 throughout — no f16 / bf16, no quantisation
* 🎯 Token-for-token aligned with the official graph at `dither = 0`

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

**No VAD model is bundled** — like FunASR, the VAD is a separate model and the ASR only wants its cut points. Get segments from any VAD (including FunASR's `fsmn-vad`) and pass them as `[start_ms, end_ms]` pairs.

### Quick Start

```bash
sensevoice --model ./SenseVoiceSmall --wav ./audio.wav
```

The run prints the device in use and timings on stderr, and the transcript on stdout.

| Option | Description |
|--------|-------------|
| `--model <dir>` | Model directory |
| `--wav <file>` | wav / mp3 / flac; non-16 kHz is resampled internally |
| `--lang <name>` | `auto` (default) / `zh` / `en` / `yue` / `ja` / `ko` / `nospeech` |
| `--itn` / `--no-itn` | Inverse text normalisation, on by default |
| `--vad-segments <json>` | Cut points: `[[start_ms,end_ms],…]` or `[{"start_ms":…,"end_ms":…}]`; `@file` reads them from a file |
| `--merge-ms <n>` | `0` (default) keeps the cuts as given; `15000` reproduces FunASR `merge_vad` |
| `--ban-emo-unk` | Mask the CTC `emo_unk` token (id 25009) |
| `--raw` | Print the untagged token text instead of the cleaned transcript |
| `--check-gold <file>` | Compare token ids / text / raw against a reference JSON |
| `--device <spec>` | `auto` (default), `cpu`, `vulkan[:i]`, `dx12`, `metal`, `#n`, or a name substring |
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
let mut model = Model::load(Path::new("SenseVoiceSmall"), "auto")?;
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
```

A model is loaded once and reused across calls. `Transcript` also carries `language`, `emotion`, `event`, `itn`, per-segment `token_ids` and timings.

Segment semantics match FunASR `inference_with_vad`: `start = int(ms * 16)`, `end = min(int(ms * 16), len)`, the audio after the last cut is not transcribed, and segment texts are joined with single spaces.

Also available:

* Device introspection — `list_devices()`, `Model::device()`
* Pre-allocation — `Model::reserve_samples(n)` keeps allocation outside the timing clock
* Rollback switches — every optimisation has a same-binary `SVS_*` environment switch (see `src/tuning.rs`), plus diagnostics such as `SVS_TRACE`, `SVS_GPU_PROF`, `SVS_CPU_PROF`

### Notes

The alignment target is the official graph at **`dither = 0`**. FunASR's `WavFrontend` defaults to `dither = 1.0`, which adds noise per frame and makes the reference non-reproducible — comparisons must disable it. See `tools/mini_eval/` for the harnesses and `docs/evidence/` for the recorded results.

### Why wgpu?

Instead of relying on CUDA, ROCm, or other vendor-specific runtimes, this project uses **wgpu** as a unified GPU abstraction, so one Rust binary runs on different platforms and GPU vendors.

### Project Status

🚧 **Active development** — CLI and library, CPU and GPU, with the fidelity harnesses in `tools/mini_eval/`.

### Related

* [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — the official model project
* [FunASR](https://github.com/modelscope/FunASR) — the official inference framework
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) — a sibling Rust + wgpu ASR port

### License

MIT, matching `Cargo.toml`.

This repository is an **independent Rust inference implementation** for loading and running the officially released SenseVoiceSmall weights — not an official FunAudioLLM / Alibaba release, and not affiliated with the original authors. Model weights remain under the terms of their respective owners.
