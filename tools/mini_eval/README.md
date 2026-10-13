# mini_asr_data 对齐/RTFx 评测（SenseVoiceSmall）

对 `D:/mini_asr_data` 里 SenseVoice 支持的 4 语种（en/zh/ja/yue 各 20 条）跑四后端对比：
Python(FunASR) CPU/CUDA 与 Rust CPU/wgpu，比较**转录逐 token 对齐**与 **RTFx**。

## ⚠️ 关键：必须强制 dither=0（否则条件不一致）

- 官方 FunASR 前端默认 **`dither=1.0`**（每个窗口加随机噪声）→ 原版本身**每次运行结果都可能不同**。
- 我们的 Rust 移植以 **dither=0 的官方推理图**为对齐目标（确定性）。
- `AutoModel(...)` 返回后设 `m.kwargs["frontend"].dither = 0.0` **无效**——`generate()` 走 `extract_fbank`，
  用的是另一个 `WavFrontend` 实例。必须从类层面强制（见 `dither0.py`）：

```python
from funasr.frontends.wav_frontend import WavFrontend
_orig = WavFrontend.__call__
WavFrontend.__call__ = lambda self, *a, **k: (setattr(self, "dither", 0.0), _orig(self, *a, **k))[1]
```

（本次首轮 63/80 就是踩了这个坑：Python 侧在跑 dither=1.0。）

## 结果（dither=0，条件一致）

| 配对 | 一致率 |
|---|---|
| py_cuda vs rust_wgpu（auto/auto） | **80/80** |
| py_cuda vs rust_wgpu（强制语言/强制语言） | **80/80** |
| py_cpu vs rust_cpu（auto/auto） | **80/80** |

RTFx 中位（不含模型加载）：

| 语种 | py_cuda | py_cpu | rust_wgpu | rust_cpu |
|---|---|---|---|---|
| en | 135.9 | 22.1 | 142.6 | 57.6 |
| ja | 195.5 | 25.4 | 178.6 | 70.5 |
| yue | 179.7 | 23.3 | 158.4 | 66.5 |
| zh | 183.9 | 24.7 | 165.0 | 68.9 |
| 全部 | 181.6 | 24.1 | 156.4 | 65.8 |

（短素材固定开销占比高；长素材 15–180s fixtures 上 rust_wgpu 为 CUDA 的 1.02–1.20×。）

## 用法

```bash
PY=/c/Users/ADMIN/miniconda3/envs/asr/python.exe
$PY eval_py_d0.py cuda:0 out_py_cuda_d0.json          # Python（dither=0）
$PY eval_rust.py P104 out_rust_wgpu.json               # Rust wgpu
$PY eval_compare.py                                    # 一致率 + RTFx + ref CER/WER
```

## VAD/API 对齐（切点接口）

**设计：与 VAD 无关**——引擎只吃"切点"。Python(FunASR) 也是同一逻辑：把 VAD 换成桩（只返回切点），
`inference_with_vad` 就是"切片 → 逐段识别 → 文本空格拼接"。

### 参数映射

| FunASR (Python) | 本仓 Rust |
|---|---|
| `language="auto|zh|en|yue|ja|ko|nospeech"` | `--lang` / `Options.language` |
| `use_itn=True/False` | `--itn`/`--no-itn` / `Options.use_itn` |
| `ban_emo_unk=True` | `--ban-emo-unk` / `Options.ban_emo_unk` |
| `vad_segments=[[start_ms,end_ms],…]`（VAD 输出） | `--vad-segments JSON` / `Options.segments` |
| `merge_vad=True, merge_length_s=15` | `--merge-ms 15000` / `Options.merge_ms` |
| VAD 模型（fsmn-vad） | **不内置**：只要切点（VAD 无关）✓ |

段语义（`src/vad.rs` 与 FunASR `funasr/utils/vad_utils.py` 逐行对齐）：
- 切片 `start=int(ms*16)`、`end=min(int(ms*16), len)`（16 kHz，截断）；
- `merge_vad`：时间点取所有 start/end 去重排序，游标从 **0** 起（含前导静音），
  当"下一点 − 游标 ≥ max_length"时切一段，末尾补 `[游标, 最后时间点]`；
- **尾部不转写**（最后一个切点之后的内容被忽略）；
- 多段结果用 **单个空格** 拼接，顺序按原始切点顺序。

### 端到端验证（`api_align.py` + `api_align_rust.py`）

桩 VAD 把固定切点喂给 FunASR `inference_with_vad`，同一批切点喂给本仓 `--vad-segments`：

| 用例 | merge | 结果 |
|---|---|---|
| 180s_en / 90s_ja，10s 定长切点（18/9 段） | 关 / 开(15s) | **文本逐字节一致** |
| 180s_en / 90s_ja，带静音间隔且不覆盖尾部（8/4 段） | 关 / 开(15s) | **文本逐字节一致** |

合计 **8/8 完全一致**（含尾部忽略、merge、空格拼接语义）。

## 分段输入的 RTFx 优化（pad 自适应）

短段走 64x64 tile 族，pad 只需按 **64 行**对齐（长段 >512 行才需要 128）。
原来固定按 128 行对齐：一个 0.6 s 的 VAD 短段也要付整块 128 行 tile 的计算。
改为 `step = t <= 512 ? 64 : 128` 后（`src/encoder_gpu.rs::ensure`），真实 VAD 切点下：

| 用例（官方 fsmn-vad，max_single_segment_time=30000） | 优化前 | 优化后 | 官方全管线 | Python 仅ASR |
|---|---|---|---|---|
| 180s_en 50 段（merge=0） | 78.1 | **88.8** | 85 | 163 |
| 180s_zh 58 段 | 67.9 | **77.2** | 89 | 142 |
| 90s_ja 16 段 | 105.6 | **120.0** | 57 | 74 |
| 90s_en 16 段 | 106.1 | **118.5** | 48 | 62 |
| 180s_en merge=15s | 185.2 | **194.8** | 118 | 247 |
| 180s_zh merge=15s | 197.3 | **204.2** | 101 | 163 |
| 90s_ja merge=15s | 190.2 | **201.7** | 96 | 155 |
| 90s_en merge=15s | 191.4 | **196.9** | 100 | 146 |

文本一致：8/8（含 merge 与尾部语义）。6 个验收素材的 pad 本来就是最紧，gold 不受影响。

剩余瓶颈（可选下一步）：小段时每个 GEMM 只有 1-2 个 m-block、k 循环长 → 延迟受限；
根治办法是把多段**拼成一条序列 + 块对角 mask**（或 batch），预计再 1.5-2x。
