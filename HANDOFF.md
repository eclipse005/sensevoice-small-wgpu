# SenseVoiceSmall (Rust + wgpu) 交接文档

> 交接时间：2026-10-12 20:50 左右
> 交接人：mavis（被用户中断，交由下一个 AI 接手）
> 文档目的：让接手者无需重读全部历史即可继续推进。

---

## 0. 一句话现状

**目标**（用户原话）：在**完全对齐**（逐 token 一致于 gold）的前提下把 RTFx 优化到极致，
并且**要么专注 CPU、要么专注 wgpu GPU，把一个做到极致**。已拍板**专注 GPU 路径**。

**当前状态**：GPU 路径已稳定通过 6/6 对齐，RTFx 已达原版 CUDA 的 0.5–3.6 倍、
原版 CPU 的 3–6 倍。**正在做"逐 op GPU 时间戳剖面"以定位剩余瓶颈，该工作卡在一个
已修但未编译的 bug 上**（见 §5）。

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
- CPU：20 逻辑核，31.48 GB RAM

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

# 全量基准（6 素材，带内存监控）
cd D:\SenseVoiceSmall\_scratch
& .\bench_rust.ps1 -Device "P104"

# 单元测试
cd D:\SenseVoiceSmall\SenseVoiceSmall-wgpu
cargo test --release      # 10/10 应通过
```

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

## 5. 进行中的工作：逐 op GPU 时间戳剖面（**接手者的第一件事**）

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

### 5.4 当前卡点（**已修但未编译**）

首次 prof 运行：984 dispatch 分类完全正确，但**时间戳全为 0**，且末尾报
`GPU 校验失败: Validation Error`。根因：

```
In a CommandEncoder
  Error encountered while trying to resolve a query
    Usage flags BufferUsages(MAP_READ | COPY_DST) of Buffer with 'prof-resolve'
    label do not contain required usage flags BufferUsages(QUERY_RESOLVE)
```

**修复已写入源码**（`src\encoder_gpu.rs:571`，给 prof staging buffer 加
`wgpu::BufferUsages::QUERY_RESOLVE`），但**用户在我执行 `cargo build` 前中断了**。

当前 exe 时间戳 = **20:46:38**，**不含**该修复。

### 5.5 接手者的第一步（照做即可）

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

### 5.6 已知的 op 普查（结构，不含时间）

共 984 个 dispatch = 70 层 × 14 op + after_norm/tp_norm/ctc/arg：

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

### 5.7 剖面之后的优化方向（按预期收益排序，供接手者参考）

1. **如果 GEMM 家族占大头**（<80%）→ 继续压 GEMM：
   vec4 向量化全局加载（A/B tile 访存指令数减少）、调 occupancy、
   针对 M=1504 调 BM/BN 减少波次量化。
2. **如果 soft/add/fsmn/merge/split 占比显著**（>20%）→
   **融合算子**是最大杠杆（参考原版 CUDA 显然做了融合，否则不会快 1.5 倍）：
   例如 `split + gemm`、`merge + add`、`fsmn + relu`、`ln + gemm` 合并成单个 kernel，
   消掉中间 buffer 的写回/重读。
3. **中间 buffer 的读写带宽**：P104-100 显存带宽只有 ~250 GB/s，若算子是 memory-bound
   而非 compute-bound，融合的收益会更大。

---

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

---

## 7. 代码地图（改哪里看哪里）

| 文件 | 职责 |
|---|---|
| `src\gemm_f32.rs` | **GEMM WGSL 代码生成器**（具名标量累加器 + 矩形线程 + 双缓冲）。当前最重要的优化文件 |
| `src\encoder_gpu.rs` | GPU 引擎。`record()` 生成 984 个 op；分批提交在 `encode()`；新增 `dump_prof()` 剖面 |
| `src\encoder_cpu.rs` | CPU 编码器（rayon 并行 softmax/fsmn/layernorm、Scratch 复用、ping-pong buffer） |
| `src\ctc.rs` | `argmax_rows`（已 rayon 并行） |
| `src\gpu.rs` | `Gpu::{readback, readback_many, describe}`；共享内存上限检查在 `new_with()` 约 240 行 |
| `src\shaders.rs` | 非 GEMM 的 WGSL（ln / soft / fsmn / merge / split / add / relu / arg） |
| `README.md` | **含选型实测表、寄存器陷阱、TDR 说明、分块参数结论——但数据仍是双缓冲前的旧值，最终需刷新** |

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

- [ ] **【最高优先】编译并跑通 `SVS_GPU_PROF=1`**（§5.5，修复已在源码，只差 build）
- [ ] 用剖面数据决定优化方向（GEMM 深挖 vs 算子融合），见 §5.7
- [ ] vec4 向量化全局加载（A/B tile 访存指令数减少 1/4）—— 若剖面显示 GEMM 仍占大头
- [ ] GEMM 调优后**重标定 TDR chunk 公式**（单 dispatch 变快后 `(1_400_000/t)` 偏保守，
      批次变小 → 批间 poll 开销占比上升）
- [ ] **最终 6/6 对齐 + 完整基准 + 更新 `README.md`**（README 的数据表仍是双缓冲前的旧值，
      必须刷新为 §4.1 的新数字）
- [ ] （可选）清理 `src\gemm_f32.rs.bak*` 遗留文件

---

## 9. 验收标准（最终交付必须全部满足）

1. `cargo test --release` → **10/10 通过**
2. CPU 路径 6/6 对齐
3. GPU 路径 6/6 对齐
4. `bench_rust.ps1 -Device "P104"` 六个素材全部 `killed=False`，
   且 RTFx 优于 §4.1 的当前值
5. `README.md` 数据表已刷新为最终实测值
6. 无残留的实验性开关导致默认路径行为变化