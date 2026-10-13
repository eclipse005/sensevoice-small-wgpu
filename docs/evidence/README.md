# 验收证据索引（objective 成功标准 ①–④）

本目录的脚本可直接复现所有数字；`-N 5` = 每素材跑 5 次取中位。
（`bench_median.ps1` / `bench_mem.ps1` / `bench_cuda_warm.py` 由 `_scratch/` 原样拷贝入仓。）

## ① GPU RTFx 6/6 ≥ CUDA warm 中位
- 脚本 `bench_median.ps1 -N 5`；CUDA 线来自 `bench_cuda_warm.py`（warmup 丢弃 + 3 次中位）。
- 证据 `median_n5.txt`（同一提交 c9ade40 上的 3 轮独立 N=5 复测）：

  | 素材 | 轮1 | 轮2 | 轮3 | 三轮最小 | 目标 | 最小余量 |
  |---|---|---|---|---|---|---|
  | 15s_en | 202.8 | 205.1 | 203.6 | 202.8 | 193.1 | +5.0% |
  | 30s_zh | 244.3 | 246.7 | 246.1 | 244.3 | 239.8 | +1.9% |
  | 90s_en | 218.3 | 219.7 | 218.1 | 218.1 | 192.8 | +13.1% |
  | 90s_ja | 216.6 | 218.1 | 216.5 | 216.5 | 213.4 | +1.5% |
  | 180s_en | 175.8 | 175.9 | 175.6 | 175.6 | 146.1 | +20.2% |
  | 180s_zh | 169.1 | 169.0 | 169.1 | 169.0 | 145.6 | +16.1% |

  CUDA 原始基线随仓提交：`bench_cuda_warm.json`（repeats=3，逐素材 median_s 与 rtfx）。
- 历轮（同协议）：5.8.7 报告 196.4/237.0/216.6/214.9/174.3/169.6（当时 30s 差 1.2%），
  5.8.9 起 6/6。表中数字为脚本原样输出的中位；HANDOFF 里「最优」指跨轮最好，不是单次最好。

## ② 峰值内存 / 显存
- **RAM**（`peak_mem.ps1`，10 ms 采样 WorkingSet64，Refresh 修正）→ `peak_rss_6.txt`：
  969 / 970 / 973 / 974 / 979 / 979 MiB vs CUDA 1.33–1.58 GB → **6/6 更低（低 29–40%）**。
- **devVRAM**（`bench_mem.ps1`，nvidia-smi 逐进程采样）→ `devvram_6.txt`：
  1093 / 1091 / 1349 / 1349 / 1631 / 1643 MiB vs CUDA 1044 / 1112 / 1438 / 1438 / 2144 / 2172
  → **5/6 更低；15s_en 1093 vs 1044 = +4.7%（该项目前是唯一未逐字满足的条目）**。
- 15s 缺口构成（实测核对）：f32 权重 893 MiB（CUDA 同为 f32：`torch.float32` 234.0M 参数 = 892.6 MiB）
  + 激活 ~39 MiB + Vulkan 驱动/管线/描述符 ~160 MiB；CUDA 的 1044 = 权重 893 + 激活/工作区 ~80 + 驱动 ~70。
  权重已打平，差异在驱动侧，无法在不降精度的前提下消除。

## ③ 计时区 CPU ≤5 ms（无 frontend/编码数学）
- 提交版插桩 `SVS_CPU_PROF=1`（`src/encoder_gpu.rs`；默认路径额外开销 < 1 µs）。
- 证据 `cpu_prof_30s_zh.txt`（6 次）、`cpu_prof_90s_en.txt`（3 次）。
- 结果：30s_zh total 2.13–2.69 ms（**中位 2.30**）、90s_en 2.38–2.60 ms；
  相位 = stage 0.00 + record 0.2 + upload 0.3 + sched 1.8 + readback 0.05 + decode 0.01 ms。
  全部为样本上传 / token 回读 / 命令调度；fbank·FFT·mel·LFR·CMVN·位置编码与全部编码器算子在 GPU。

## ④ 门禁全绿
- CPU/GPU 各 6/6 gold：`sensevoice.exe --device {cpu,P104} --check-gold gold/<w>.json`（每次改动均跑）。
- 12/12 测试：`cargo test --release`。
- CPU 全迹 78/78：`SVS_TRACE=1 ... --device cpu` → 与 `trace_base_90s_en.txt` 逐行比对
  （注意 base 由 git checkout 成 CRLF，比对前先 `tr -d '\r'`）。
- 零编译警告：`cargo build --release 2>&1 | grep -cE '^(warning|error)'` = 0。

## 度量口径与已知陷阱（供复核者）
- **RTFx**：本仓一律"每素材 5 次取中位"（`bench_median.ps1` 原样输出）；历轮数字见 HANDOFF §5.8.x，
  早期轮次曾出现 5/6（30s 差 1.2%），5.8.9 起稳定 6/6。
- **CUDA 线**：来自 `bench_cuda_warm.json`（warmup 丢弃 + repeats=3 取中位）。
  更早的 `bench_original_cuda.json` 里 `rtfx` 一栏有冷启动污染（15s=25.84），已废弃；该文件的
  内存/devVRAM 栏仍是 objective ② 的对照来源。
- **口径差异**：本仓 devVRAM 是 nvidia-smi 逐进程实际占用；CUDA 侧是 `torch.cuda.max_memory_reserved()`
  （含分配器缓存，一般 ≥ 实际）。即本仓在更严的口径下比较。
- **CPU 全迹比对**：`trace_base_90s_en.txt` 经 git checkout 后是 CRLF，比对前先 `tr -d '
'`。
- **权重精度核对**：CUDA 侧为 `torch.float32` 234.0M 参数 = 892.6 MiB（实测），与本仓 f32 slab ~893 MiB 打平。

## 回退开关可用性（约束"保留同二进制回退开关"）
- `fallback_switches.txt`：10 个开关（`SVS_TILE_SMALL/WIDE`、`SVS_FUSE_RELU/ADD/MERGE/SPLIT`、
  `SVS_FSMN_TAP/FLAT`、`SVS_BIAS_FOLD`、`SVS_RB_FUSED`）取回退值后，6 素材 GPU gold **全部逐 token 一致**。
- 该验证暴露并修复了一个真实缺陷：`SVS_FUSE_ADD=0` 时 fsmn 的 Out 绑定仍指向 `dst_buf`，
  而独立的 `add` op 读 `acts.fsmn`（从未写入）→ 回退路径输出错误 token；已改为按开关选择绑定。
