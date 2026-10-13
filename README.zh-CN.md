# SenseVoiceSmall wgpu

**用 Rust + wgpu 跑 SenseVoiceSmall 语音识别。**

[English](README.md) · **简体中文**

基于 [wgpu](https://github.com/gfx-rs/wgpu) 的轻量、跨平台 [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) Rust 实现。

目标很简单：**本地原生**运行 SenseVoiceSmall，不依赖 Python 和厂商专属 GPU 运行时。

### 特性

* 🦀 纯 Rust
* 🎮 wgpu GPU 加速
* 🌍 跨平台 GPU（Vulkan / DX12 / Metal）
* 🖥️ Windows / macOS / Linux
* ⚡ CPU 回退
* 📦 离线本地推理
* 🎙️ SenseVoiceSmall —— `zh` / `en` / `yue` / `ja` / `ko`
* 🔤 指定语言、ITN 开关、情感/事件标签
* ✂️ 与 VAD 无关 —— 调用方给切点
* 🧩 CLI + Rust 库
* 🔢 全程 f32 —— 不用 f16/bf16、不做量化
* 🎯 与官方图在 `dither = 0` 下逐 token 对齐

### 安装

作为 Cargo 依赖：

```toml
[dependencies]
sensevoice-small-wgpu = { git = "https://github.com/eclipse005/sensevoice-small-wgpu.git" }
```

或从源码构建 CLI：

```bash
git clone https://github.com/eclipse005/sensevoice-small-wgpu.git
cd sensevoice-small-wgpu
cargo build --release        # target/release/sensevoice
```

### 模型下载

仓库**不含权重**。下载官方 checkpoint（版权归原作者），把目录原样传给 `--model`：

- [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)（Hugging Face）
- [iic/SenseVoiceSmall](https://modelscope.cn/models/iic/SenseVoiceSmall)（ModelScope）

目录里需要有 `model.pt`、`am.mvn`、`chn_jpn_yue_eng_ko_spectok.bpe.model`。

**不内置 VAD 模型** —— 与 FunASR 结构一致：VAD 是独立模型，ASR 只要它的切点。用任意 VAD
（包括官方 `fsmn-vad`）拿到 `[start_ms, end_ms]` 后传进来即可。

### 快速开始

```bash
sensevoice --model ./SenseVoiceSmall --wav ./audio.wav
```

stderr 打印设备与耗时，stdout 是转写文本。

| 选项 | 说明 |
|--------|-------------|
| `--model <dir>` | 模型目录 |
| `--wav <file>` | wav / mp3 / flac；非 16 kHz 内部重采样 |
| `--lang <name>` | `auto`（默认）/ `zh` / `en` / `yue` / `ja` / `ko` / `nospeech` |
| `--itn` / `--no-itn` | 逆文本规整，默认开启 |
| `--vad-segments <json>` | 切点：`[[start_ms,end_ms],…]` 或 `[{"start_ms":…,"end_ms":…}]`；`@file` 从文件读 |
| `--merge-ms <n>` | `0`（默认）保持切点；`15000` 复现官方 `merge_vad` |
| `--ban-emo-unk` | 屏蔽 CTC 的 `emo_unk`（id 25009） |
| `--raw` | 输出带标签的原始 token 文本 |
| `--check-gold <file>` | 与参考 JSON 比对 token id / text / raw |
| `--device <spec>` | `auto`（默认）、`cpu`、`vulkan[:i]`、`dx12`、`metal`、`#n`，或名称子串 |
| `--list-devices` | 列出本机可用设备 |

带切点转写：

```bash
sensevoice --model ./SenseVoiceSmall --wav ./long.wav --vad-segments '[[1690,2280],[4750,6060],[7840,12100]]'
sensevoice --model ./SenseVoiceSmall --wav ./long.wav --vad-segments @cuts.json --merge-ms 15000
```

### 库用法

```rust
use sensevoice_small_wgpu::{load_audio, Model, Options, Segment};
use std::path::Path;

let samples = load_audio(Path::new("audio.wav"))?;              // 16 kHz 单声道 f32
let mut model = Model::load(Path::new("SenseVoiceSmall"), "auto")?;
model.reserve_samples(samples.len())?;                          // 预分配（在计时区外）

let opt = Options {
    language: "auto".into(),
    use_itn: true,
    ban_emo_unk: false,
    // 就是 VAD 产出的东西；留空 = 整段转写
    segments: cuts.iter()
        .map(|(s, e)| Segment { start_ms: *s, end_ms: *e })
        .collect(),
    // 0.0 保持切点；15000.0 = 官方 merge_vad（merge_length_s = 15）
    merge_ms: 0.0,
};

let out = model.transcribe(&samples, &opt)?;
println!("{}", out.text);            // out.raw 保留 <|en|><|NEUTRAL|>… 标签
```

模型加载一次、多次复用。`Transcript` 还带 `language`、`emotion`、`event`、`itn`、每段
`token_ids` 与耗时。

段语义与 FunASR `inference_with_vad` 一致：`start = int(ms * 16)`、`end = min(int(ms * 16), len)`、
最后一个切点之后的音频不转写、各段文本用单个空格拼接。

其它可用项：

* 设备信息 —— `list_devices()`、`Model::device()`
* 预分配 —— `Model::reserve_samples(n)`
* 回退开关与诊断 —— 每个优化都有同二进制 `SVS_*` 环境开关（见 `src/tuning.rs`），
  以及 `SVS_TRACE`、`SVS_GPU_PROF`、`SVS_CPU_PROF` 等诊断

### 说明

对齐目标是官方图在 **`dither = 0`** 时的结果。FunASR `WavFrontend` 默认 `dither = 1.0`（每帧加噪声），
原版自己就不可复现——做比较时必须关掉。评测脚本在 `tools/mini_eval/`，实测记录在 `docs/evidence/`。

### 为什么用 wgpu？

不依赖 CUDA、ROCm 或其它厂商专属运行时，用 **wgpu** 作为统一的 GPU 抽象，
让同一个 Rust 二进制跑在不同平台与不同厂商的 GPU 上。

### 项目状态

🚧 **开发中** —— CLI 与库、CPU 与 GPU 均可用，评测工具在 `tools/mini_eval/`。

### 相关

* [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) —— 官方模型项目
* [FunASR](https://github.com/modelscope/FunASR) —— 官方推理框架
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) —— 同源的 Rust + wgpu ASR 移植

### 许可

MIT（与 `Cargo.toml` 一致）。

本仓库是**独立的 Rust 推理实现**，用于加载和运行官方发布的 SenseVoiceSmall 权重；
不是 FunAudioLLM / 阿里官方发布，与原作者无隶属关系。模型权重遵循其各自所有者的条款。
