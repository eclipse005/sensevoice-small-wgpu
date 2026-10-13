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
