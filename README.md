# SenseVoiceSmall（Rust CPU + wgpu）

对齐目标是官方推理图在 **dither = 0** 时的结果。FunASR `WavFrontend` 默认 `dither=1.0`，每帧加随机噪声，不能逐位对齐。不要拿一次带 dither 的 FunASR 输出当 gold。

VAD 模型不内置。调用方传入毫秒段，或整段转写。官方 `merge_vad` 的游标从 0 开始，会改写精确的段，所以默认不合并。要复现官方 demo 的 15 秒合并，再传 `--merge-ms 15000`。一段超过 GPU 缓冲时直接报错，不会按时间悄悄切开。

CPU f32 是对齐基准。GPU 同为 f32 存储与累加，6 个 fixtures 上 GPU 与 CPU 的 token id 逐个一致，但短音频偶发分歧时仍以 CPU 为准。

## 选 CPU 还是 GPU

实测（P104-100 / 8 GB / Vulkan，20 核 CPU，dither=0，整段转录）：

| 素材 | 原版 FunASR CPU | 本仓 CPU | 原版 FunASR CUDA | 本仓 GPU |
|---|---|---|---|---|
| 15s_en | 30.30 | **57.25** | 25.84 | **65.83** |
| 30s_zh | 31.67 | **64.73** | 228.75 | 88.68 |
| 90s_en | 25.39 | **61.39** | 184.71 | 89.52 |
| 90s_ja | 24.77 | **64.15** | 210.02 | 88.88 |
| 180s_en | 17.57 | **48.08** | 144.52 | 72.07 |
| 180s_zh | 15.66 | **46.94** | 144.73 | 70.53 |

RTFx。两条路径的 6 个 fixtures 都是逐 token 对齐 gold。

CPU 压过原版 CPU 2.2–3.0 倍，进程内存 0.65–1.14 GB（原版 CPU 3–4.4 GB）。GPU 在短音频上已经快过原版 CUDA，长音频仍是它的 2/3：手写 WGSL GEMM 约 1.1 TFLOP/s，而原版走的是 cuBLAS。

## 寄存器而不是数组

GEMM 的 `TM×TN` 微块累加器必须写成**具名标量**。`array<array<f32, 8>, 8>` 虽然是合法 WGSL，但 naga/DXIL 允许把它放进 private memory —— 在 NVIDIA 上那由显存支撑，于是每次 FMA 都要一次 load 加一次 store。把数组换成标量后单次前向快了约 20 倍，k 的归约顺序不变，结果逐位一致。

分块参数 TM=8 / TN=8 / BK=8 是实测最优；BK=16 慢约 5%，BK=32 无差别。

## Windows TDR

GPU 一次提交跑超过约 2 秒会被 Windows 判定为挂起并复位设备。复位后所有 dispatch 变成空操作，`ids` 保持全零，**解码结果是空的但退出码为 0**。

整张图约 984 个 dispatch，所以按批次提交并逐批等待。批次大小随 `t` 反比缩放（`1400000 / t`，夹在 16..2048），目标每批约 1.1 秒；实测 T=1504 可单提交，T=3004 不行。批间等待约占 encode 的 1%。`SVS_GPU_CHUNK` 可覆盖，`SVS_GPU_STOP` 可截断指令流用于定位。

## 构建

```
cargo build --release
```

直接读官方目录里的 `model.pt`、`am.mvn` 和 `chn_jpn_yue_eng_ko_spectok.bpe.model`。数值就是这份检查点，不用再导出 safetensors。

## 命令

```
sensevoice --wav D:\SenseVoiceSmall\fixtures\15s_en.wav --device cpu --check-gold gold\15s_en.json
sensevoice --wav clip.wav --device auto
sensevoice --wav clip.wav --vad-segments "[[0,15000],[15000,30000]]"
sensevoice --wav clip.wav --vad-segments @segments.json --merge-ms 15000
sensevoice --list-devices
```

`--device`：`auto`、`cpu`、`vulkan`、`dx12`、`#n`，或适配器名称的一部分。立体声先转单声道。采样率不是 16 kHz 时，用 torchaudio 默认的 `sinc_interp_hann`（width 6，rolloff 0.99），和参考脚本一致。

语言 `--lang auto|zh|en|yue|ja|ko|nospeech`。ITN 默认开，`--no-itn` 关掉。

stdout 是后处理文本（`--raw` 则是带 `<|en|>` 这类标签的原文）。stderr 是设备、frontend / encode / decode 毫秒、音频秒数和 RTFx。RTFx 从权重进显存、pipeline 编译完之后才计时。
