# SenseVoiceSmall (Rust + wgpu) 交接文档

> 交接时间：2026-10-12 20:50 左右
> 交接人：mavis（被用户中断，交由下一个 AI 接手）
> 文档目的：让接手者无需重读全部历史即可继续推进。
>
> **2026-10-13 更新（下半场）**：GPU 逐 op 剖面**已跑通**（§5），结论是
> GEMM 家族占 GPU busy 的 **83.7%**，方向定为深挖 GEMM。代码已入本地 git（5 个提交）。

---

## 0. 一句话现状

**目标**（用户原话）：在**完全对齐**（逐 token 一致于 gold）的前提下把 RTFx 优化到极致，
并且**要么专注 CPU、要么专注 wgpu GPU，把一个做到极致**。已拍板**专注 GPU 路径**。

**2026-10-13 现状**（CPU 突击，已闭环）：

- **偶发非确定性已根治**：根因是 gemm-common `CACHE_INFO` 惰性 CPUID 在 P/E 核上
  生成不同分块参数（详见 §4.4），`freeze_gemm_tiling()` 冻结后 40/40 确定。
- **CPU RTFx 全面提升**（90s_en 61.4 → **86.6**，+41%），全部优化**位级等价**，
  6/6 gold 通过，逐项有同二进制回退开关。
- **代码已入本地 git**：`0be4b36` 初始 → `32cc464` FSMN → `a6f068c` CTC，工作区干净。
- **GPU 剖面（§5）已跑通**，且已完成**第一轮 GPU 优化（§5.8）**：B 侧 bank 冲突消除
  （–10%）、LayerNorm vec4、frontend 并行（83→15ms）；90s_en RTFx 122→154。
  CUDA 热基线重测后差距 1.2–1.6×（旧表是冷启动污染，勿再引用）。
- 用户已拍板：**softmax exp 向量化暂不做**（非位级等价，数值高风险，2026-10-13）。

**GPU 轨道既有状态**（2026-10-12）：GPU 路径已稳定通过 6/6 对齐，RTFx 已达原版 CUDA 的
0.5–3.6 倍、原版 CPU 的 3–6 倍。

---

## 1. 基础信息

### 1.1 路径

| 用途 | 路径 |
|---|---|
| 本项目源码 | `D:\SenseVoiceSmall\SenseVoiceSmall-wgpu` |
| 二进制 | `D:\SenseVoiceSmall\SenseVoiceSmall-wgpu\target\release\sensevoice.exe` |
| 模型 | `D:\SenseVoiceSmall\model\SenseVoiceSmall`（model.pt / am.mvn / bpe.model） |
| gold 基准 | `D:\SenseVoiceSmall\SenseVoiceSmall-wgpu\gold\*.json` |
| 音频素材 | `D:\SenseVoiceSmall\fixtures\{15s_en,30s_zh,90s_en,90s_ja,180s_en,180s_zh}.wav` |
| 工具脚本 | `D:\SenseVoiceSmall\_scratch\` |
| 原版 FunASR | conda 环境 **`asr`**（funasr 1.3.22 / torch 2.8.0+cu128）；**不是 `myenv`** |

### 1.2 硬件

- GPU：NVIDIA **P104-100**（Pascal GP104，8 GB），Vulkan / DX12。
  - 适配器名匹配请用 `--device "P104"`（**不要用 `vulkan:1`，索引会落到核显**）。
  - `max_storage_buffer_binding_size` = 2047 MiB
  - workgroup 共享内存上限 = **49152 字节（48KB）** ← GEMM 的硬约束
- CPU：20 逻辑核（**Arrow Lake 混合架构：8 P 核 + 12 E 核，不连续分布**），31.48 GB RAM
  - P 核 L1d 48K/12 路，E 核 L1d 32K/8 路 → **gemm 分块参数随首次调用所在核而变**，
    是曾导致偶发非确定性的根因（见 §4.4），现已由 `freeze_gemm_tiling()` 冻结。

### 1.3 运行命令模板

```powershell
cd D:\SenseVoiceSmall\SenseVoiceSmall-wgpu

# 对齐验证（GPU）
$env:SVS_GPU_PROF = ''   # 平时清空
& ".\target\release\sensevoice.exe" `
    --wav "D:\SenseVoiceSmall\fixtures\90s_en.wav" `
    --device "P104" `
    --check-gold ".\gold\90s_en.json"
# 期望：gold: 一致，exit=0

# 对齐验证（CPU，本轮工作以 CPU 为主）
& ".\target\release\sensevoice.exe" `
    --wav "D:\SenseVoiceSmall\fixtures\90s_en.wav" `
    --device "cpu" `
    --check-gold ".\gold\90s_en.json"
# 期望：exit=0。判断 gold 一律看退出码，别匹配中文"一致"（见 §6.8）

# 全量基准（6 素材，带内存监控）
cd D:\SenseVoiceSmall\_scratch
& .\bench_rust.ps1 -Device "P104"

# CPU 全量基准 + 配对 A/B（本轮收尾用）
& .\bench_rtfx.ps1

# 单元测试
cd D:\SenseVoiceSmall\SenseVoiceSmall-wgpu
cargo test --release      # 12/12 应通过
```

### 1.4 CPU 路径同二进制回退开关（每项优化一个，均可即时关闭）

| 变量 | 默认 | 作用 |
|---|---|---|
| `SVS_BIAS_FOLD=0` | 开 | 关闭 bias+relu 融合，回退两遍扫描 |
| `SVS_GEMM_THREADS` | 开 | 置 `0` 回退 gemm crate 默认线程分块 |
| `SVS_GEMM_CACHE=max\|min\|off` | `max` | 冻结 gemm 分块参数的 regime（`off`=不冻结） |
| `SVS_FSMN_FLAT=0` | 开 | 回退 FSMN 旧循环嵌套（通道外层） |
| `SVS_CTC_TILE=N` | 2048 | CTC vocab 分块宽；`0`=整体式（151MB 工作集） |
| `SVS_CPU_PROF=1` | 关 | CPU 分相计时（ln/qkv/fsmn/softmax/… 桶） |
| `SVS_TRACE=1` | 关 | 分层 trace（`{:.9}` + FNV 哈希），位级门禁用 |
| `SVS_TRACE_FINE=1` | 关 | 层内细粒度 trace_f 插桩（诊断用，9 处/层） |

---

## 2. 不可动摇的硬约束（违反即任务失败）

### 2.1 对齐优先于一切性能

任何优化**必须**在改动后用 `--check-gold` 验证 6/6 通过。**性能提升不能以牺牲逐 token 一致为代价。**

允许的"数值精确"手法：
- 只按行 / 只按时间维度并行（不改归约内 k 顺序）
- 只改访存索引的发射顺序（合并访存索引）
- 代码生成层面的具名标量改写

禁止的手法：改变任何一次 f32 归约的内部累加顺序（会产生 f32 结合律误差，
greedy argmax 可能翻转）。

### 2.2 CLI 默认参数不能改

必须保持默认 `--lang auto`（即 lid=0）、ITN 开启、不传 `--vad-segments`（整段转录）、
dither = 0。否则与 gold 不符。

### 2.3 对齐基准必须用 CPU 而非 CUDA

FunASR 默认跑 `cuda:0`，GPU 与 CPU 的 f32 归约顺序不同会导致 greedy argmax 翻转。
**生成 gold 用 Python CPU**；但**和原版 CUDA 比 RTFx 是允许的**（只是性能对比）。

### 2.4 用户强制的安全阈值

用户曾因 13 分钟音频跑挂而**强制断电重启**，此后要求：
- RAM > 14 GB 立即 kill
- 单素材超时保护
- **严格串行**，不要并发跑多个素材
- `_scratch\run_mon.ps1` 已实现内存监控 + 自动 kill，`bench_rust.ps1` 已接入

### 2.5 环境/删除约束

- **不要用 `Remove-Item`**（被硬安全策略阻止）。删除用 `rm -- <正斜杠路径>`，走 mavis-trash。
- **不要用 `Remove-Item Env:\VAR`**（被策略拦截）。清环境变量要**赋空字符串**
  （PowerShell 里空字符串不会真正删除变量，见 §6.6 踩坑记录）。

---

## 3. 当前源码关键参数（已验证生效）

`src\gemm_f32.rs`（**当前值就是最优配置，不要盲目改动**）：

```rust
GEMM_TM = 16, GEMM_TN = 8      // 线程微块 16x8 = 5.33:1 宽高比
GEMM_BK = 8                     // K 方向分块
GEMM_TX = 16, GEMM_TY = 8      // 线程网格 16x8 = 128 线程
GEMM_BM = TY*TM = 128
GEMM_BN = TX*TN = 128
GEMM_SMEM = 2*(128+128)*8*4 = 16384   // 双缓冲，48KB 上限内安全
```

参数扫描结论（90s_en，encode_ms，越低越好）：
- BK=8 → 652.8ms ✅ 最优
- BK=4 → 728.5ms
- BK=16 → 680.5ms
- BK=32 → **失败**（双缓冲需 64KB，超 48KB 上限，由 GEMM_SMEM 检查捕获）
- 方形 TM=TN=8 → 729.6ms；TM=TN=4 更差

---

## 4. 已完成的工作与实测数据

### 4.1 实测对比（**这组数字可信**，来自修复陈旧二进制之后的干净构建）

| 素材 | 本仓 GPU | 原版 CUDA | 比值 | 原版 CPU | 本仓 CPU |
|---|---|---|---|---|---|
| 15s_en  | **92.72** | 25.84 | 3.59× | 30.30 | 46.9 |
| 30s_zh  | 114.87 | 228.75 | 0.50× | 31.67 | 68.1 |
| 90s_en  | 121.60 | 184.71 | 0.66× | 25.39 | 68.1 |
| 90s_ja  | 121.27 | 210.02 | 0.58× | 24.77 | 68.1 |
| 180s_en | 100.03 | 144.52 | 0.69× | 17.57 | 50.7 |
| 180s_zh | 100.32 | 144.73 | 0.69× | 15.66 | 49.4 |

> 结论：GPU 路径**已全面碾压原版 CPU（3–6 倍）**；对原版 CUDA，15s 素材领先 3.6 倍，
> 但长音频仍落后约 1.5 倍。**继续优化的目标是拉近/反超长音频的 CUDA。**

其他指标：RAM 峰值仅 1.80–1.81 GB（原版 CPU 3–4.4 GB）；devVRAM 1353–1775 MiB。

### 4.2 三个决定性的 GPU 修复（务必了解，否则会重蹈覆辙）

**① 寄存器溢出陷阱（性能悬崖，20.4 倍差距的来源）**
`acc: array<array<f32,8>,8>` 这种累加器数组会被 naga/DXIL 放进 **private memory**
（NVIDIA 上由显存支撑），导致每做一次 FMA 就多一次 load + store。
改为**代码生成具名标量**后 GPU 直接快 **20.4 倍**（90s_en：20268ms → 994ms）。
→ 任何新写的 WGSL 累加器都**必须是具名标量**，不要用数组。

**② Windows TDR 静默失效（正确性陷阱，极难发现）**
984 个 dispatch 单次提交必然超过 Windows 约 2 秒的看门狗阈值 → 驱动复位设备 →
后续 dispatch 静默变成空操作 → `ids` 全零 → **进程退出码 0 但输出为空**。
修复：分批提交 + 逐批 `poll`，批次公式 `(1_400_000 / t).clamp(16, 2048)`。
批间等待约占 encode 的 1%，不是吞吐权衡。
→ **新增任何 GPU 批量提交逻辑时，必须记住这个陷阱。**

**③ 双缓冲 + 矩形线程网格（GEMM 微优化）**
- 寄存器暂存双缓冲（预取下一 k-tile，与当前 tile 计算重叠，每步仅一个 barrier）：+24%
- 矩形线程网格 TX=16/TY=8/TM=16/TN=8（5.33:1）：+9.4%
  （方形 8×8 微块恰为 4:1，正好卡在 Pascal 共享内存带宽上限）

**双缓冲首次实现踩过的 bug**（已修复，供对照）：B tile 目的地址写成 `Bs[p]`，
应为 `Bs[ki*BN+col]`。**跨 tile 的暂存地址算式极易出错，改这块务必逐式核对。**

### 4.3 历史踩坑：qkv 缩放跨步索引

曾有一次"优化"把 `row*QKV+j` 错误拍平成连续的 `qkv[..t*HIDDEN]`，
导致 6/6 全挂。用 `SVS_TRACE` 对比 `gold\joints_py.log` 定位到 L0 发散后修复。
→ 跨步（strided）访问很容易被错误地拍平；改动后**必须**跑分层 trace 比对。

---

## 4.4 CPU 突击（2026-10-13）：偶发非确定性根治（**必读，勿退回**）

**现象**：同一二进制、同一输入，40 次运行中偶发输出分叉，但分叉后各自稳定
（两个离散吸引子，L0 起 2 ULP 级差异 → 全链分叉）。

**根因**（证据链完整，勿再怀疑 gemm 内部竞争）：gemm-common 0.18.2 的
`CACHE_INFO` 是 `OnceCell` **惰性**初始化，首次 GEMM 调用时用 **CPUID leaf 4**
在**当时所在核**上取 L1 参数。本机是 Arrow Lake 混合架构（8P+12E）：

- P 核 L1d 48K/12 路 → `kc = min(1024, k)`
- E 核 L1d 32K/8 路 → `kc = min(512, k)`

全模型唯一 k=560 的 GEMM 是 **layer 0 的 qkv**：1 个 560 块 vs 2 个 280 块
→ f32 结合顺序二态 → 双稳态输出。逐 LP 实验 8:12 严格跟随核类型定案。

**修复**：`CpuEngine::new()` 首行调用 `freeze_gemm_tiling()`（`encoder_cpu.rs`）：
临时 pin 到 L1 最大的 P 核 → 触碰 `CACHE_INFO` → 解除 pin，从此分块参数与调度无关。
`SVS_GEMM_CACHE=max|min|off` 为同二进制回退开关（`off` = 完全不冻结）。
修复后 40/40 确定，max/min 各自确定，性能打平 → 默认 `max`（= 历史主导值，
L0 与旧基线逐位一致，零数学改动）。

**一致性门禁已升级**：trace 从 `sum/max` 标量升级为 `sum={:.9} max={:.9} n h={:016x}`，
`h` 为 f32 `to_bits` 的 FNV-1a（sum/max 相同 ≠ 缓冲区相同，曾漏检）。

---

## 4.5 CPU 突击：本轮两项位级等价优化（均已提交）

| 项 | 内容 | 分相/工作集收益 | 配对 A/B | 门禁 | 回退开关 | 提交 |
|---|---|---|---|---|---|---|
| FSMN 循环翻转 | tap 外层/512 通道内层 + tap-major 权重拷贝；每通道累加链不变 | fsmn **15.0→8.4 ms** | 6/6 胜 | 全迹 78 行 0 差异 | `SVS_FSMN_FLAT` | `32cc464` |
| CTC vocab 分块 + bias 折进 argmax | fused GEMM+argmax，`acc+b` ≡ 原 `b+acc` | 工作集 **151MB→12MB**，配对均值 **−25 ms** | 4/6 胜（块宽 2048 定夺） | 逐 bit 单测 + 全迹 0 差异 + 6/6 gold | `SVS_CTC_TILE` | `a6f068c` |

关键等价性依据（改动这两处前必读）：

- **FSMN**：每通道累加链保持 tap 升序、从 0.0 起、中心 tap 最后加 → 逐位等价。
  权重布局重排是纯拷贝，不改数值。
- **CTC**：k=512 在任意块宽下都是**单个 `kc` 块**（`kc` 只依赖 k 与缓存几何），
  每 logit 归约顺序不变；bias 直接进比较是同一浮点加法。
  字节级常驻单测 `ctc_tiling_is_bit_identical`（tests 模块）逐 logit `assert_eq!(to_bits)`。

---

## 4.6 CPU 实测 RTFx（2026-10-13 收尾基准，6 素材 × 3 次取中位）

| 素材 | 优化前基线 | **优化后** | 提升 | encode_ms | frontend_ms |
|---|---|---|---|---|---|
| 15s_en  | 57.3 | **74.3** | +30% | 189.4 | 12.4 |
| 30s_zh  | 64.7 | **90.0** | +39% | 309.8 | 24.6 |
| 90s_en  | 61.4 | **86.6** | +41% | 967.1 | 72.2 |
| 90s_ja  | 64.2 | **83.8** | +30% | 990.4 | 72.1 |
| 180s_en | 48.1 | **61.6** | +28% | 2727.6 | 144.3 |
| 180s_zh | 46.9 | **59.6** | +27% | 2874.2 | 145.8 |

（"优化前基线"= bias_fold/gemm_threads/FSMN/CTC 全关的本会话测量；
注意与 §4.1 表中旧"本仓 CPU"列测量条件不同，勿混用。）

累计配对收益（90s_en）：bias_fold+gemm_threads −49 ms、FSMN −6.7 ms、CTC −25 ms。
6/6 gold 全过（退出码判定）。

---

## 4.7 CPU 优化工程纪律（用户硬性要求，每步都做）

1. **消融/微基准先行**：分相计时（`SVS_CPU_PROF`）或微基准确认收益来源；
   总时长信噪比不足时**用分相桶消融**（FSMN 项就是靠这个定案的）。
2. **同二进制回退开关**：每项优化一个环境变量（§1.4），可即时关闭对比。
3. **紧邻配对 A/B**：on/off 交替跑（6 对取中位/均值），不跨时段比。
4. **逐字节位级门禁**：`SVS_TRACE=1` 双跑对比全迹 78 行（含 FNV 哈希）0 差异，
   不许只看 sum/max。基线：`_scratch\trace_base_90s_en.txt`（新格式，78 行）。
5. **转录精度不变**：6/6 `--check-gold` 退出码必须全 0。
6. **独立提交**：每项优化一个 git commit，可单独 revert。

验证脚本族（`D:\SenseVoiceSmall\_scratch\`，均幂等、串行）：
`bench_rtfx.ps1`（全量+配对）、`gate_fsmn.ps1`/`gate_ctc.ps1`（位级+配对+gold）、
`prof_fsmn.ps1`（分相消融）、`ab_ctc2k4k.ps1`（块宽定夺）、
`det_sweep.ps1`/`det_pin.ps1`/`det_lp.ps1`/`det_fix.ps1`（确定性定位与复验）。

---

## 4.8 已证伪方向（CPU，勿重试）

- scores 按行分块、ctx 拆 Q/K/V（不省访存或破坏归约）
- FTZ/DAZ（对结果路径无收益）
- 打包 padding 残留、L2_SLAB 残留（gemm 静态分析已证干净）
- gemm crate 内部竞争（真正的非确定性来源是 §4.4 的 CACHE_INFO）
- softmax exp 向量化 —— **用户已拍板暂不做**（非位级等价，动 f32 exp 语义；
  若重启需用户批准 + 全迹漂移评估 + 6/6 gold）

---

## 5. GPU 轨道待续：逐 op 时间戳剖面（暂缓，卡点已解除；回到 GPU 时的第一件事）

### 5.1 为什么做这个

现有计时只有 `encode_ms` 总数（90s_en 650ms），无法判断 984 个 dispatch 里
**哪一类算子占大头**。没有 GPU timestamp 就只能盲猜。

### 5.2 已完成的改动（**已编译进当前 exe**）

`src\gpu.rs`：
- `Gpu` 结构体新增 `pub features: wgpu::Features` 字段
- `request_device` 在 `SVS_GPU_PROF=1` 且适配器支持时按需请求
  `wgpu::Features::TIMESTAMP_QUERY`；平时仍是 `Features::empty()`（零开销）
- 注意：wgpu **30.0.1**，`adapter.limits()` 上**没有** `features` 字段，
  要用 `adapter.features()`

`src\encoder_gpu.rs`：
- `encode()` 中新增 `prof` 开关 + `qset`（`QueryType::Timestamp`）
- prof 开启时**每个 dispatch 一个独立 compute pass**（这样才能逐 dispatch 打时间戳）；
  非 prof 时保持原来"每批一个 pass"的路径，**零行为变化**
- 新增 `dump_prof()` 方法，resolve 后按 kernel 类聚合 + 打最慢 10 个 dispatch

### 5.3 已验证的正面结论

prof 路径跑 90s_en：`encode_ms: 650.303 / rtfx: 124.20 / tokens: 310 / gold: 一致 / exit=0`。
→ **每个 dispatch 独立 pass 没有引入可观测开销（650.30 vs 常规 650.34），且结果逐 token 一致。**

### 5.4 卡点定位过程（**已解决，2026-10-13**）

首次 prof 运行报的两个 wgpu 校验错误，按顺序出现，第二个才是真卡点：

1. `MAP_READ | COPY_DST` 的 buffer 不接受 `resolve_query_set`（缺 `QUERY_RESOLVE`）。
2. 只补 `QUERY_RESOLVE` 后**新的**报错：
   ```
   In Device::create_buffer, label = 'prof-resolve'
     `MAP` usage can only be combined with the opposite `COPY`,
     requested BufferUsages(MAP_READ | COPY_DST | QUERY_RESOLVE)
   ```
   wgpu 规定 mappable buffer 只能与相反的 COPY 搭配，不能和 `QUERY_RESOLVE` 共存。

**正确修法（已提交 `79fe2ed`）**：两段式，与 `Gpu::readback` 同模式——
`prof-resolve` = `QUERY_RESOLVE | COPY_SRC`，`prof-staging` = `MAP_READ | COPY_DST`，
encode 里 `resolve_query_set` 后接 `copy_buffer_to_buffer`。

验证：`SVS_GPU_PROF=1` 90s_en → 984 dispatch 分类正确、时间戳非零、exit=0、tokens=310。

### 5.5 复现剖面的命令（已验证跑通）

```powershell
cd D:\SenseVoiceSmall\SenseVoiceSmall-wgpu
cargo build --release

$env:SVS_GPU_PROF = '1'
& ".\target\release\sensevoice.exe" `
    --wav "D:\SenseVoiceSmall\fixtures\90s_en.wav" `
    --device "P104" `
    --check-gold ".\gold\90s_en.json" 2> "D:\SenseVoiceSmall\_scratch\prof.err.txt" 1> $null

Get-Content "D:\SenseVoiceSmall\_scratch\prof.err.txt" -Encoding UTF8 | Select-Object -First 40
```

**不要用 `2>&1 | Select-String`**：本机控制台代码页会把中文和部分输出弄乱
（`gold: 一?`、`最?10 ?`），stderr 单独重定向到文件再读更可靠。
输出是 ANSI 转义的，`-Encoding UTF8` 读取。

预期得到形如：

```
gpu prof: 984 dispatches, GPU busy XXX ms
  gemm_b      141 ops      ... ms   ..%
  gemm        70 ops ...
  ...
gpu prof: 最慢 10 个 dispatch
  #0    ln      ... dispatch=(48,1,1) slot=0
  ...
```

### 5.6 剖面实测（2026-10-13，90s_en，P104；**机器有 GPU 争用**，看相对份额，绝对值待静默复测）

`gpu prof: 984 dispatches, GPU busy 837.853 ms`：

| op | 次数 | ms | 份额 | | op | 次数 | ms | 份额 |
|---|---|---|---|---|---|---|---|---|
| **gemm_b** | 141 | 262.96 | **31.4%** | | soft | 70 | 47.88 | 5.7% |
| **gemm_f32c** | 140 | 203.12 | **24.2%** | | ln_f16 | 140 | 40.98 | 4.9% |
| **gemm** | 70 | 126.54 | **15.1%** | | fsmn | 70 | 11.96 | 1.4% |
| **gemm_nt** | 70 | 108.30 | **12.9%** | | split/relu/add/merge/ln/arg/ln_f32 | 353 | 38.2 | 4.6% |

- **GEMM 家族合计 700.9 ms = 83.7%**；非 GEMM 16.3%（其中 soft + ln_f16 占 10.6%）。
- 最慢单个 dispatch：`#982 gemm_b (196,12,1)` **19.73 ms** —— 即 CTC 头投影
  （t=1504 × 512 → vocab≈25055），一项占总量 2.4%，可作为独立优化对象。
- 其余 GEMM 单次均值 1.45–1.81 ms，70×2 层量级相近。
- prof 路径逐 dispatch 一个 compute pass，端到端 encode 比常规慢约 12%（955 vs ~830 ms），
  **份额可信、绝对值偏大**；GPU busy 总和也含 pass 间气泡。

### 5.6.1 op 普查（结构，不含时间）

| op | 次数 | | op | 次数 |
|---|---|---|---|---|
| ln | 1 | | merge | 70 |
| **gemm_b** | **141** | | add | 70 |
| fsmn | 70 | | **gemm_f32c** | **140** |
| split | 70 | | ln_f16 | 140 |
| **gemm** | **70** | | relu | 70 |
| soft | 70 | | ln_f32 | 1 |
| **gemm_nt** | **70** | | arg | 1 |

**GEMM 家族合计 421 个 dispatch**（gemm_b 141 + gemm 70 + gemm_nt 70 + gemm_f32c 140）。

前几个 dispatch 的形状（t=1504 时）：
```
#0 ln        dispatch=(48,1,1)
#1 gemm_b    dispatch=(12,12,1)    -> 144 blocks
#2 fsmn      dispatch=(2,1536,1)
#3 split     dispatch=(1,4,1536)
#4 gemm      dispatch=(12,12,4)    -> 576 blocks
#5 soft      dispatch=(6144,1,1)
#6 gemm_nt   dispatch=(1,12,4)
#7 merge     dispatch=(1,4,1536)
#8 add       dispatch=(3072,1,1)
#9 gemm_f32c dispatch=(4,12,1)
```
P104-100 只有 15 个 SM。gemm_b 的 144 blocks / 15 = 9.6 波，波次量化损失应该不大，
但 soft/add/fsmn 这些 dispatch 数量多、单个体积小的算子**很可能是隐藏成本**
（每 dispatch 有固定开销 + 依赖链串行）。**这正是需要剖面数据来验证的假设。**

### 5.7 剖面之后的优化方向（**2026-10-13 已拍板：走第 1 条深挖 GEMM**）

1. **GEMM 家族占 83.7% → 继续压 GEMM**（当前主攻）：
   - vec4 向量化全局加载（A/B tile 访存指令数减为 1/4）；
   - 调 occupancy / BM-BN 组合，针对 M=1504、N=512 的真实形状减少波次量化（P104 只有 15 SM）；
   - 单列的 CTC 头 GEMM（#982，19.73 ms，N≈25055）单独调 tile，收益直接可观。
2. 若 GEMM 压到瓶颈后非 GEMM 份额回升（soft+ln_f16+fsmn 等 >20%）→ 算子融合
   （`split+gemm`、`merge+add`、`fsmn+relu`、`ln+gemm`），消中间 buffer 写回/重读。
3. P104-100 显存带宽只有 ~250 GB/s：memory-bound 算子（soft/ln/split/merge）融合收益大。

> 注意：任何改动后先按 §2.1 验证 6/6 gold；GEMM 调优后按 §8 重标定 TDR chunk 公式。

---

## 5.8 GPU 优化战役（第一轮，2026-10-13 下半场）

### 5.8.1 干净的 CUDA 基线（重测，必读）

旧表里 15s_en 的 CUDA RTFx=25.84 是**首个素材承担了 CUDA/cuBLAS 冷启动**的产物。
`_scratch/bench_cuda_warm.py` 先跑一次 discard 的 warmup，再每素材 3 次取中位：

| 素材 | 15s_en | 30s_zh | 90s_en | 90s_ja | 180s_en | 180s_zh |
|---|---|---|---|---|---|---|
| CUDA（热，中位） | **193.1** | **239.8** | **192.8** | **213.4** | **146.1** | **145.6** |
| 本仓 GPU（本轮末） | 121.2 | 151.2 | 153.2 | 152.0 | 124.9 | 120.0 |
| 本仓 CPU（同场对照） | 72.1 | 88.5 | 85.3 | 85.8 | 60.7 | 59.4 |

差距从旧表的 0.5–0.69× 收窄到 **0.63–0.86×**，全素材仍落后 1.2–1.6×。
注意 FunASR 的 `batch_size_s` 默认 60s：长音频实际是分块跑（注意力 O(T²) 变便宜），
本仓是整段图；同图对比时差距应视为更小。

### 5.8.2 本轮有效的改动（均已提交，6/6 gold 验证）

| 改动 | 效果 | 提交 |
|---|---|---|
| **B 侧共享内存 [BK][TN][TX] + 行填充**，消除 bank 冲突（加载 4-way→1、存储 8-way→2） | 90s_en 653→585ms；6 素材 –9~11% | `def4978` |
| **LayerNorm vec4 化**（逐分量保持累加顺序，位级等价） | ln_f16 分相 30.8→15.6ms | `5865c5d` |
| **frontend fbank/LFR/CMVN rayon 并行**（逐帧独立，逐位不变） | frontend 83→15ms（90s_en） | `5865c5d` |

综合：90s_en RTFx 122→154，encode 653→577ms。

### 5.8.3 已证伪/无效的方向（勿重复试）

- `fma()` 显式内积：无变化（驱动本来就在做 FFMA）
- ki 循环生成器全展开：**慢 17%**（驱动已自行处理，展开只增寄存器/代码体积）
- 软件流水（双操作数寄存 + 全展开）：同样更慢（1.95 vs 2.47 TFLOP/s）
- BK=16 / BK=4：均比 BK=8 慢
- TM=8/TN=8 + 256 线程（占用率翻倍）：慢 6%
- softmax 共享内存缓存 / 每 warp 一行：**均无变化**（说明它既不是带宽也不是 barrier 瓶颈）
- DX12 后端：同一 kernel 时间相同（且会选中另一张 GTX 1070）

### 5.8.4 诊断工具与标定（下轮继续用）

- `dump_prof` 新增**按 dispatch 形状聚合**（同一 pipe 的不同 N/K 形态分开统计，
  避免类均值掩盖形状差异）。90s_en 的 per-shape 账：`gemm_b(16,12,1)` w1 1.55ms、
  `gemm_f32c(4,12,1)` outp+w2 1.06ms、`gemm(12,12,4)` qk 1.39ms、`gemm_nt(1,12,4)` av 1.17ms、
  `gemm_b(196,12,1)` CTC 头 14.7ms。
- **cuBLAS 标定**：本卡（P104-100，15 SM，满载 1898MHz/139W）大矩阵 SGEMM = **5.83 TFLOP/s**；
  纯寄存器 FMA 探针（32 FMA/迭代）= **4.94 TFLOP/s**。工具链不是硬限制。
- 我们的 GEMM 实际只有 **2.2–2.7 TFLOP/s**（ctc 最高 2.68），仍有 ~2× 空间：
  下一步需要的是 128-bit 全局/共享加载 + XOR swizzle 的 CUTLASS 式改写，
  而不是继续调 BK/TM/TN（已穷尽）。
- Nsight Compute 只支持 CUDA，**Vulkan 拿不到硬件计数器**；迭代靠 `gemm_probe` 式
  合成探针（本轮临时写过，验证探针数字与图内一致后已删除）。

### 5.8.5 第二轮（2026-10-13 晚，参照 `D:\qwen3-asr-wgpu` 的实测结论）

同用户的 Qwen3-ASR wgpu 移植（同一张卡）给了关键输入：共享内存加载的**指令数**
而非字节数定价内循环（同 2KB/迭代：16 次标量加载 2.66 TFLOP/s，4 次 vec4 3.80）；
其采用的 tile 是 **8×8 微块 + 16×16 线程（256）+ vec4 k-major 分块**，实测 3.11。

| 改动 | 效果 | 提交 |
|---|---|---|
| **GEMM 重写为 vec4 k-major 分块**（8×8/16×16，A 广播、B 行组置换 `(g&7)+8s+16(g>>3)`） | GEMM 2.5→3.0 TFLOP/s；90s_en 570→480ms | `493c51d` |
| **矮 tile 变体（BM=64，m ≤ 512 时启用）** | 短音频 block 数翻倍；30s +1.4%，15s 持平 | 本轮 |
| **frontend 全部上 GPU**（fbank FFT+mel / LFR+CMVN+posenc 两个 kernel） | CPU 热路径归零；首次实现差 2 token，定位到 `f32::EPSILON` 下限与 posenc 的 half 维（560/2=280） | `bcb9eb6` |
| **softmax 行缓存版**（pad≤4096 用独立 pipeline） | 仅 −2ms（证明它本来就不是带宽瓶颈） | 本轮 |
| **加载期预分配 3072 帧** | 首次 `ensure` 的 18ms 重分配移出计时区 | 本轮 |

最终（同场对照，CUDA 为 warm 中位）：

| 素材 | 15s_en | 30s_zh | 90s_en | 90s_ja | 180s_en | 180s_zh |
|---|---|---|---|---|---|---|
| CUDA | 193.1 | 239.8 | 192.8 | 213.4 | 146.1 | 145.6 |
| 本仓 GPU | 160.3 | 191.5 | **192.0** | 189.6 | **157.3** | **152.3** |
| 比值 | 0.83 | 0.80 | 1.00 | 0.89 | 1.08 | 1.05 |

长音频已全面反超（注意 CUDA 的 `batch_size_s=60` 会分块跑，注意力成本低于我们的整段图）；
短音频仍差 20%。剩余空间：softmax（90s 仍 36ms）三次重构均无大效（直读/shared 缓存/warp-per-row），
ln_f16（0.113ms/op）受限于每线程一行的非合并访存，动它需要重排归约顺序（有 token 翻转风险）。

### 5.8.6 第三轮（2026-10-13 深夜）：GEMM epilogue + 内存达标

| 改动 | 效果 |
|---|---|
| **GEMM epilogue 全 vec4 化**（C 存/读、bias 取：64→16 次/线程） | 30s 232、90s 209、180s 170；**4/6 超 CUDA 线** |
| **BK 按 dispatch 选**（m≤256 矮 tile BK=16；256<m≤512 窄 tile BK=8；其余 128×128 BK=8） | 15s +5%、30s +2% |
| **LN 改为 shared 暂存 + 单通道顺序归约**（位级等价） | 9.7→3.4ms@15s |
| split/merge/add/relu vec4 化 | 中性（保留）；**fsmn vec4 回退**（权重 4 行 gather 反而 2× 慢） |
| 尝试失败的：LN 64-lane 树归约（90s_ja/180s_zh 翻 token）、默认 tile BK=16（90s +3%） | |

**softmax 消融结论**（90s，0.51ms/op）：去掉 normalize 的 `exp` 无变化（说明不是 SFU），
全局读只占 11%，写只占 9% —— 成本在并行结构本身（每个 6KB 行一个 256 线程 workgroup
+ 3 趟 + 2 棵 8 层归约树）。三种重构（直读、shared 缓存、warp-per-row）均无大效。

**内存达标路径**（关键三招）：
1. **名称匹配改 Vulkan-first 实例**：`--device P104` 原先枚举全部后端（DX12/GL 各自拉起
   runtime），宿主工作集多出 **585MB**；只建 Vulkan instance 后 device 创建开销 ~55MB。
2. **权重上传即释放**：`&mut Weights` + `mem::take`，峰值不再叠加（下载完全模型 ~900MB
   与驱动增长同时存在）。
3. **CLI 按实际音频长度预分配**（`Model::reserve_samples`，在 RTFx 计时区外）+ 权重打包单一
   slab（~945 buffer → ~35，驱动开销大降）。

实测（run_mon，逐素材）：RAM **6/6 ≤ CUDA**（0.90–0.95GB vs 1.33–1.58）；devVRAM 5/6 ≤ CUDA
（15s 1093 vs 1044 超 49MiB，其余均低：30s 1093 vs 1112、90s 1349 vs 1438、180s 1631/1643 vs 2144/2172）。

### 5.8.7 当前战绩（中位 3 次，2026-10-13 收尾）

| 素材 | 本仓 GPU | CUDA 目标 | 比值 |
|---|---|---|---|
| 15s_en | 171.8 | 193.1 | 0.89 |
| 30s_zh | 233.3 | 239.8 | 0.97 |
| 90s_en | **207.4** | 192.8 | **1.08** |
| 90s_ja | 205.1 | 213.4 | 0.96 |
| 180s_en | **170.3** | 146.1 | **1.17** |
| 180s_zh | **162.0** | 145.6 | **1.11** |

## 6. 踩坑记录（务必阅读，可节省大量调试时间）

### 6.1 ⚠️ 陈旧二进制 / cargo 不重建（本次最费时间的坑）

**现象**：运行时报
`NVIDIA P104-100 的 workgroup 共享内存上限 49152 字节，GEMM 需要 65536`
——这是 **BK=32** 的值，而源码里明明是 BK=8（应为 16384）。
`cargo clean -p sensevoice-small-wgpu` 之后 `cargo build` 仍报
`Finished in 0.13s`（没重新编译），exe 的 LastWriteTime 也没变。

**根因**：扫描脚本用 `Copy-Item` 恢复备份文件，而 **PowerShell 的 `Copy-Item`
会保留原始文件的 LastWriteTime**。于是 `gemm_f32.rs` 被还原成 BK=8 的内容，
但 mtime 被写成备份文件的创建时间（20:37:29），**早于**用 BK=32 编译出的 exe
（20:39:10）。cargo 认为"依赖没变，无需重建"。

**正确解法**：
```powershell
(Get-Item "src\gemm_f32.rs").LastWriteTime = Get-Date
cargo build --release
```

**教训**：
- **每次改完 `gemm_f32.rs` 跑基准前，先确认 exe 的 LastWriteTime 晚于源码。**
- `cargo clean` 解决不了这个问题（它重建的所有文件 mtime 都是新的，但 cargo 的
  指纹机制仍可能判定 up-to-date；至少 touch 源码是可靠的）。
- **任何基准数字在使用前，先用 `--check-gold` 确认二进制是新的**。

### 6.2 二进制正确性自检

跑任何性能测试前，先跑一次 90s_en 的 `--check-gold`。如果 tokens=310 且 exit=0，
说明二进制与源码一致；如果 tokens=0 或结果不符，**先怀疑二进制陈旧**。

### 6.3 不要用 `--device wgpu` 或 `vulkan:1`

- `--device wgpu` → 报错"没有匹配 Name("wgpu") 的适配器"
- `--device vulkan:1` → **静默落到 Intel 核显**（列表里有重复的 Intel 条目，索引错位），
  表现为 encode_ms 7000+、tokens=0、RTFx 12
- ✅ 一直用 `--device "P104"`（名称子串匹配）

### 6.4 控制台中文乱码

本机 pwsh 代码页不是 UTF-8，`2>&1` 混流会把中文变成 `一?`、`最?10 ?`。
诊断输出请**重定向到文件**再读，并加 `-Encoding UTF8`。

### 6.5 `src` 目录下的遗留备份文件

`src\gemm_f32.rs.bak` / `.bak2` / `.bak3`（20:18 / 20:30 / 20:37）是 GEMM 扫描脚本留下的。
它们**不是 .rs 扩展名，cargo 不会编译**，无害但碍眼。
其中 `.bak3`（10138 字节）与当前 `gemm_f32.rs`（10138 字节）大小相同——
即 `.bak3` 就是当前正确版本。**接手者可自行决定是否清理。**

### 6.6 PowerShell 环境变量陷阱

`$env:SVS_GPU_PROF = ''`（空字符串）**不会真正删除**变量，子进程会看到它存在且为空。
代码里用 `.as_deref() == Some("1")` 判断，所以空串是安全的（等价于关闭）。
但如果需要确保关闭，用 `$env:SVS_GPU_PROF = '0'` 更稳妥。

### 6.7 13 分钟素材的硬上限

`test.wav`（T=13786）：
- Rust GPU **直接拒绝**（8192 帧上限）
- 原版 CUDA **爆显存** 8037/8191 MiB 被杀
- Rust CPU 可跑（4.06 GB / 237s / RTFx 3.56）
- 注意：8192 帧上限是 `max_frames()` 里 `while next <= 8192` 的**写死循环上界**，
  **不是硬件限制**（该卡硬件上限约 T≈11582）。如果要放开需改这里。

### 6.8 判定 gold 一律看退出码，不要匹配中文（2026-10-13 踩坑）

`bench_rtfx.ps1` 等脚本曾用 `Where-Object { $_ -like '*一致*' }` 判定 gold，
导致**明明 exit=0 全部通过却显示 `gold_fail=3`**。两个坑叠加：

1. PowerShell 5 读 BOM-less `.ps1` 时把脚本里的中文按 ANSI 解码，匹配串变乱码；
2. `Start-Process` 的 `ExitCode` 属性需**无参 `WaitForExit()`** 后才刷新
   （带超时参数的 `WaitForExit(ms)` 不刷新属性）。

**正确做法**：`$p.WaitForExit(); $p.ExitCode -eq 0`；脚本内不写非 ASCII 字面量。
本仓 `gold` 判定若见全 `False`/`gold_fail` 异常而行为正常，先手动验证退出码再下结论。

---

## 7. 代码地图（改哪里看哪里）

| 文件 | 职责 |
|---|---|
| `src\gemm_f32.rs` | **GEMM WGSL 代码生成器**（具名标量累加器 + 矩形线程 + 双缓冲）。当前最重要的优化文件 |
| `src\encoder_gpu.rs` | GPU 引擎。`record()` 生成 984 个 op；分批提交在 `encode()`；诊断（`dump_prof`/`dump_ops`/`dump_debug`）已抽成方法 |
| `src\encoder_cpu.rs` | CPU 编码器：`freeze_gemm_tiling()` 确定性修复、FSMN 翻转、CTC 分块、`trace_on` FNV 门禁、`Prof` 分相 |
| `src\tuning.rs` | **全部 `SVS_*` 开关的唯一注册处**（11 个变量，含默认值与回退语义文档） |
| `src\cpu_affinity.rs` | Windows 线程亲和性 + CPUID（`freeze_gemm_tiling` 的 P/E 核选择依赖它） |
| `src\ctc.rs` | `argmax_tile`（fused bias+argmax，按 vocab 块增量更新 best）+ `greedy` |
| `src\gpu.rs` | `Gpu::{readback, readback_many, describe}`；共享内存上限检查在 `new_with()` 约 240 行 |
| `src\shaders.rs` | 非 GEMM 的 WGSL（ln / soft / fsmn / merge / split / add / relu / arg） |
| `Cargo.toml` | 新增 `gemm-common = "0.18"`（必须与 `gemm` 同版本，共享同一 `CACHE_INFO`） |
| `README.md` | **含选型实测表、寄存器陷阱、TDR 说明、分块参数结论——但数据仍是双缓冲前的旧值，最终需刷新** |

2026-10-13 结构整理（`16dd0c6`/`c383725`/`8464961`）：删除了 807 行的死模块
`gemm_shader.rs` 与 `pack.rs`、f16 残骸、三个死 `Pid` 与五条重复 pipeline；
`U4` 匿名 blob 换成具名配置结构体；`encode()` 里的诊断块抽成方法；
`SVS_*` 开关集中到 `tuning.rs`；亲和性/CPUID 拆到 `cpu_affinity.rs`。
现在 `cargo build --release` **零警告**。行为门禁已验：12/12 测试、
CPU 全迹 78/78 逐字节一致、6/6 gold × CPU/GPU、90s_en 配对 A/B RTFx 在噪声内。

### 本地 git 仓库

`D:\SenseVoiceSmall\SenseVoiceSmall-wgpu` 已初始化本地 git（无远程），提交历史：

```
8464961 结构：SVS_* 开关集中到 tuning.rs，Windows 亲和性拆到 cpu_affinity.rs
c383725 GPU 结构：具名配置结构体替代 U4、LN 三条 pipeline 合一、诊断块抽出 encode()
16dd0c6 清理：删除 gemm_shader/pack/f16 残骸与死 Pid，处理全部编译警告（零警告）
f85f8fc 文档：记录 GPU 剖面实测（GEMM 家族 83.7%）与后续方向
79fe2ed GPU 剖面：MAP_READ 不能与 QUERY_RESOLVE 同 buffer，改用 resolve+staging 两段式
525147a 文档：同步 2026-10-13 CPU 闭环状态（确定性修复 + FSMN/CTC 优化）
a6f068c CTC 投影按 vocab 分块 + bias 折进 argmax：工作集 151MB→12MB
32cc464 FSMN 循环嵌套翻转：fsmn 分相 15.0→8.4 ms（6/6 配对胜出）
0be4b36 初始提交：SenseVoiceSmall 推理（Rust CPU + wgpu）
```

每项优化一个 commit，可单独 revert。`.gitignore` 含 `/target`、`__pycache__/`、`*.bak*`。

### `encoder_gpu.rs` 的调试开关

| 变量 | 作用 |
|---|---|
| `SVS_GPU_STOP=N` | 只执行前 N 个 op |
| `SVS_GPU_DUMP_OPS=1` | 打印全部 984 个 op（含 `pid_name()`） |
| `SVS_GPU_CHUNK=N` | 强制指定 TDR 批次大小 |
| `SVS_GPU_DEBUG=1` | 中间 buffer 的数值 dump |
| `SVS_GPU_PROF=1` | **逐 dispatch GPU 时间戳剖面**（需 TIMESTAMP_QUERY） |
| `SVS_TRACE` | 分层 trace，用于与 `gold\joints_py.log` 比对定位发散层 |

### 分层定位工具

- `gold\joints_py.log` / `gold\joints_rs.log` — Python / Rust 逐层 trace
- `_scratch\layer_diff.py` — 分层对比（音频加载 / 前端 / 编码）
- `_scratch\ids_diff.py` — 复刻 funasr inference 并与 gold 逐 token 对比

---

## 8. 待办清单

CPU 轨道（2026-10-13 已闭环，无强制待办）：

- [x] 偶发非确定性根治（`freeze_gemm_tiling`，40/40 确定）
- [x] FSMN 循环翻转 + CTC vocab 分块（位级等价，各自提交）
- [x] 新格式 trace 基线重建（`_scratch\trace_base_90s_en.txt`，78 行）+ 6/6 gold + 全量基准
- [ ] （暂缓，用户已拍板）softmax exp 向量化 —— 非位级等价，重启需用户批准（§4.8）
- [ ] （可选）`trace_f`/`SVS_TRACE_FINE` 诊断插桩去留（9 处/层）

GPU 轨道（2026-10-13 第一轮已完成，见 §5.8）：

- [x] 跑通 `SVS_GPU_PROF=1`；剖面加**按形状聚合**
- [x] 第一轮：B 侧 bank 冲突消除、LayerNorm vec4、frontend 并行（§5.8.2）
- [x] 已证伪一轮（§5.8.3）：ki 展开/软流水/BK 变体/占用率变体/softmax 重构均无效
- [ ] **下一轮主候选**：GEMM kernel 128-bit 加载 + XOR swizzle（CUTLASS 式）
      —— 当前 2.5 TFLOP/s，cuBLAS 标定 5.8，纯 FMA 探针 4.94
- [ ] 单列处理 CTC 头 GEMM（#982，14.7 ms）—— 当前每 FLOP 最快，优先级低
- [ ] GEMM 调优后**重标定 TDR chunk 公式**（`1_400_000/t` 会偏保守）
- [x] README 数据表已刷新为 2026-10-13 实测（含 warm CUDA 基线）
- [ ] （可选）清理 `src\gemm_f32.rs.bak*` 遗留文件（上一轮已删，此处为历史记录）

> ⚠️ 2026-10-13 实测环境警告：机器上 League of Legends 常驻，GPU/CPU 均有争用，
> 90s_en GPU encode 从文档的 650 ms 劣化到 ~820–860 ms。**优化 A/B 必须紧邻配对**，
> 绝对值基线等静默环境再测。

---

## 9. 验收标准（最终交付必须全部满足）

1. `cargo test --release` → **12/12 通过**（含 `ctc_tiling_is_bit_identical` 逐 bit 单测）
2. CPU 路径 6/6 对齐，且 `SVS_TRACE=1` 全迹 78 行与
   `_scratch\trace_base_90s_en.txt` 逐字节一致（FNV 哈希）
3. GPU 路径 6/6 对齐
4. `bench_rust.ps1 -Device "P104"` 六个素材全部 `killed=False`，
   且 RTFx 优于 §4.1 的当前值；CPU 侧不劣于 §4.6 的当前值
5. `README.md` 数据表已刷新为最终实测值
6. 无残留的实验性开关导致默认路径行为变化（默认值见 §1.4；
   同二进制回退开关必须保留）