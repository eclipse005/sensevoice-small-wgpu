# SenseVoiceSmall wgpu

**用 Rust + wgpu 跑 SenseVoiceSmall 语音识别。**

[English](README.md) · **简体中文**

基于 [wgpu](https://github.com/gfx-rs/wgpu) 的轻量、跨平台 [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) Rust 实现。

目标很简单：**本地原生**运行 SenseVoiceSmall，不依赖 Python 和厂商专属 GPU 运行时，并与官方推理图保持**逐 token 对齐**。

### 特性

* 🦀 纯 Rust
* 🎮 wgpu GPU 加速
* 🌍 跨平台 GPU（Vulkan / DX12 / Metal）
* 🖥️ Windows / macOS / Linux
* ⚡ CPU 回退（同时是对齐基准）
* 📦 离线本地推理
* 🎙️ SenseVoiceSmall —— `zh` / `en` / `yue` / `ja` / `ko`
* 🔤 指定语言、ITN 开关、情感/事件标签
* ✂️ **与 VAD 无关** —— 调用方给切点，引擎负责其余
* 🧩 CLI + Rust 库
* 🔢 全程 f32 —— 不用 f16/bf16、不做量化
* 🎯 与官方 FunASR 图在 `dither = 0` 下逐字对齐

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

**不内置 VAD 模型。** 与 FunASR 的结构一致——VAD 是独立模型，ASR 只要它的切点。用任意 VAD
（包括官方的 `fsmn-vad`，例如 `max_single_segment_time = 30000`）拿到 `[start_ms, end_ms]` 后传进来即可。

### 快速开始

```bash
sensevoice --model ./SenseVoiceSmall --wav ./audio.wav
```

stderr 打印设备、耗时与 RTFx，stdout 是转写文本。

| 选项 | 说明 |
|--------|-------------|
| `--model <dir>` | 模型目录（默认 `D:\SenseVoiceSmall\model\SenseVoiceSmall`） |
| `--wav <file>` | wav / mp3 / flac；非 16 kHz 内部重采样 |
| `--lang <name>` | `auto`（默认）/ `zh` / `en` / `yue` / `ja` / `ko` / `nospeech` |
| `--itn` / `--no-itn` | 逆文本规整，**默认开启**（标点 + 数字格式） |
| `--vad-segments <json>` | 切点：`[[start_ms,end_ms],…]` 或 `[{"start_ms":…,"end_ms":…}]`；`@file` 从文件读 |
| `--merge-ms <n>` | `0`（默认）保持切点原样；`15000` 复现官方 `merge_vad`（`merge_length_s=15`，游标从 0 起，会并入开头静音） |
| `--ban-emo-unk` | 屏蔽 CTC 的 `emo_unk`（id 25009） |
| `--raw` | 输出带标签的原始 token 文本（`<\|en\|><\|NEUTRAL\|>…`） |
| `--check-gold <file>` | 与参考 JSON 比对 token id / text / raw，不一致退出码 1 |
| `--device <spec>` | `auto`（默认）、`cpu`、`vulkan[:i]`、`dx12`、`metal`、`#n`，或名称子串如 `P104` |
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
let mut model = Model::load(Path::new("SenseVoiceSmall"), "P104")?;
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
for seg in &out.segments {
    println!("{:>8.0}–{:<8.0} {}", seg.start_ms, seg.end_ms, seg.raw);
}
```

模型加载一次、多次复用。`Transcript` 还带 `language`、`emotion`、`event`、`itn`、每段
`token_ids`，以及 `Timing { frontend_ms, encode_ms, decode_ms, audio_s, rtfx }`。

段语义与 FunASR `inference_with_vad` 完全一致：`start = int(ms * 16)`（截断）、
`end = min(int(ms * 16), len)`、**最后一个切点之后的音频不转写**、各段文本用单个空格拼接。

### 性能

RTFx（音频秒 / 墙钟秒），不含模型加载，5 次取中位，P104-100 / Vulkan。整段转写：

| 素材 | FunASR CPU | **本仓 CPU** | FunASR CUDA（热） | **本仓 GPU** |
|---|---|---|---|---|
| 15s_en | 30.30 | **70.3** | 193.1 | **201.0** |
| 30s_zh | 31.67 | **95.4** | 239.8 | **246.5** |
| 90s_en | 25.39 | **88.5** | 192.8 | **219.6** |
| 90s_ja | 24.77 | **90.6** | 213.4 | **217.9** |
| 180s_en | 17.57 | **65.7** | 146.1 | **176.0** |
| 180s_zh | 15.66 | **65.0** | 145.6 | **169.3** |

用真实 VAD 切点（官方 `fsmn-vad`，`max_single_segment_time = 30000`）：

| 素材 | 官方全管线 | **本仓 GPU** |
|---|---|---|
| 180s_en（50 段） | 85 | **190.3** |
| 180s_zh（58 段） | 89 | **143.3** |
| 90s_ja（16 段） | 57 | **228.2** |
| 90s_en（16 段） | 48 | **204.3** |
| 180s_en + `merge_length_s=15` | 118 | **193.9** |
| 180s_zh + `merge_length_s=15` | 101 | **202.9** |

内存：峰值进程 RAM 969–979 MiB（CUDA 为 1.33–1.58 GB，六项全部更低）；峰值显存六项中五项更低，15s 高 4.7%。

### 对齐

* **全程 f32**：GPU 权重与激活都是 f32，累加保持 f32；CPU 路径是基准，GPU 逐 token 跟随。
* **对齐目标是 `dither = 0`**：FunASR `WavFrontend` 默认 `dither = 1.0`，每帧加噪声，**原版自己就不可复现**——
  同一段音频连续两次 `generate()` 文本都可能不同。比较时必须强制 `dither = 0`
  （见 `tools/mini_eval/dither0.py`），否则"参考"每次都不一样。
* **证据**（可复跑，见 `docs/evidence/README.md`）：6 个验收素材在 CPU 与 GPU 上都与 FunASR 的 token id 逐个一致；
  80 条 FLEURS 子集（en/zh/ja/yue）在条件一致的 `dither = 0` 下 **80/80 逐字相同**；
  VAD 路径对 `inference_with_vad` **8/8**；CPU 阶段全迹 78 段逐字节一致。
* 多段输入会把各段拼成一条序列、用块对角 attention/FSMN mask 一次前向完成；逐行 argmax
  与"逐段单独跑"位级一致。

### 环境开关

每个优化都有同二进制回退（见 `src/tuning.rs`）：

| 变量 | 置 `0` 的效果 |
|---|---|
| `SVS_TILE_SMALL` | 全部改用 128×128 GEMM tile |
| `SVS_TILE_WIDE` | N=512 投影留在 64×64 tile |
| `SVS_FUSE_RELU` / `SVS_FUSE_ADD` / `SVS_FUSE_MERGE` / `SVS_FUSE_SPLIT` | 关闭对应融合 |
| `SVS_FSMN_TAP` / `SVS_FSMN_FLAT` | 用 checkpoint 的 FSMN 权重布局 |
| `SVS_BIAS_FOLD` | 保留独立 bias pass |
| `SVS_RB_FUSED` | 回读走一次性 staging 路径 |
| `SVS_SEG_CONCAT` | VAD 段逐段处理 |

诊断：`SVS_TRACE`、`SVS_TRACE_FINE`、`SVS_CPU_PROF`、`SVS_GPU_PROF`、`SVS_GPU_DEBUG`、
`SVS_GPU_DUMP_OPS`、`SVS_GPU_STOP`、`SVS_GPU_CHUNK`。

### 为什么用 wgpu？

不依赖 CUDA、ROCm 或其它厂商专属运行时，用 **wgpu** 作为统一的 GPU 抽象，
让同一个 Rust 二进制跑在不同平台与不同厂商的 GPU 上。

### 项目状态

✅ **可用且已验证** —— CLI 与库、CPU 与 GPU，对齐证据见上。
仍在打磨 GPU 吞吐与内存、分段路径与 API 表面。

### 相关

* [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) —— 官方模型项目
* [FunASR](https://github.com/modelscope/FunASR) —— 本移植对齐的官方推理框架
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) —— 同源的 Rust + wgpu ASR 移植

### 许可

MIT（与 `Cargo.toml` 一致）。

本仓库是**独立的 Rust 推理实现**，用于加载和运行官方发布的 SenseVoiceSmall 权重；
不是 FunAudioLLM / 阿里官方发布，与原作者无隶属关系。模型权重遵循其各自所有者的条款。
