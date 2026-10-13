//! f32 encoder. Weights, activations and GEMM stores match the official f32 graph.
//!
//! Activations are padded to a multiple of 128. Attention keys at and past the
//! real length are masked, and FSMN reads past that length as zeros, so the pad
//! does not change the valid frames.

use std::num::NonZeroU64;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::ctc::{greedy, VOCAB};
use crate::frontend::Frontend;
use crate::gemm_f32::{
    gemm_f32, gemm_f32_bias, gemm_f32_bias_tall, gemm_f32_narrow, gemm_f32_tall, GEMM_BM,
    GEMM_BM_TALL, GEMM_BN,
};
use crate::gpu::Gpu;
use crate::shaders;
use crate::tuning;
use crate::weights::{Weights, CTC_PAD, FFN, HIDDEN, QKV};

// The GEMM tile edge drives every GEMM dispatch, so it must track the shader's
// own BM/BN rather than being pinned to a literal.
const TILE: u32 = GEMM_BM;
const _: () = assert!(GEMM_BM == GEMM_BN);
const SLOTS: usize = 2048;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GDims {
    n: u32,
    k: u32,
    ldc: u32,
    bsa: u32,
    bsb: u32,
    bsc: u32,
    lda: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LnCfg {
    d: u32,
    eps: f32,
    a: u32,
    b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SplitCfg {
    t_pad: u32,
    t_valid: u32,
    scale: f32,
    p: u32,
}

// Per-op uniforms. Each one mirrors the WGSL struct of the same name in
// `shaders.rs`; the field names carry the meaning so a swap cannot go
// unnoticed. (The old shared `U4 { a, b, c, d }` blob put `t_valid`/`t_pad`
// in opposite slots for FSMN and merge.)

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FsmnCfg {
    t_valid: u32,
    t_pad: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SoftCfg {
    t_valid: u32,
    t_pad: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MergeCfg {
    t_pad: u32,
    t_valid: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddCfg {
    n: u32,
    mode: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReluCfg {
    n: u32,
    _a: u32,
    _b: u32,
    _c: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ArgCfg {
    t_valid: u32,
    n_vocab: u32,
    stride: u32,
    ban: u32,
}

#[derive(Clone, Copy)]
enum Hid {
    In,
    H0,
    H1,
}

#[derive(Clone, Copy)]
enum Pid {
    /// Layer 0's input norm (`d = 560`).
    Ln,
    /// Layer-input norm at `d = 512`.
    ///
    /// The name is historical: every activation is f32, and all three `Ln*`
    /// ids resolve to one pipeline and differ only in which buffer they bind.
    LnF32F16,
    /// Post-encoder norms (`after_norm`, `tp_norm`) at `d = 512`.
    LnF32,
    Gemm,
    GemmNt,
    GemmB,
    GemmF32c,
    /// The small-M tile (half the rows per workgroup): same kernels, doubled
    /// block count for short audio where a 128-row tile leaves the card idle.
    GemmTall,
    GemmNtTall,
    GemmBTall,
    GemmF32cTall,
    /// The narrow tile (BM=128, BN=64, BK=8) for the middle range of M.
    GemmNarrow,
    GemmNtNarrow,
    GemmBNarrow,
    GemmF32cNarrow,
    Fsmn,
    Split,
    Soft,
    /// [`Soft`] with the row staged in shared memory (pad <= 4096).
    SoftCached,
    Merge,
    Add,
    Relu,
    Arg,
}

struct Op {
    pipe: Pid,
    bg: usize,
    slot: u32,
    x: u32,
    y: u32,
    z: u32,
}

fn pid_name(p: Pid) -> &'static str {
    match p {
        Pid::Ln => "ln",
        Pid::LnF32F16 => "ln_f16",
        Pid::LnF32 => "ln_f32",
        Pid::Gemm => "gemm",
        Pid::GemmNt => "gemm_nt",
        Pid::GemmB => "gemm_b",
        Pid::GemmF32c => "gemm_f32c",
        Pid::GemmTall => "gemm_tall",
        Pid::GemmNtTall => "gemm_nt_tall",
        Pid::GemmBTall => "gemm_b_tall",
        Pid::GemmF32cTall => "gemm_f32c_tall",
        Pid::GemmNarrow => "gemm_narrow",
        Pid::GemmNtNarrow => "gemm_nt_narrow",
        Pid::GemmBNarrow => "gemm_b_narrow",
        Pid::GemmF32cNarrow => "gemm_f32c_narrow",
        Pid::Fsmn => "fsmn",
        Pid::Split => "split",
        Pid::Soft => "soft",
        Pid::SoftCached => "soft_cache",
        Pid::Merge => "merge",
        Pid::Add => "add",
        Pid::Relu => "relu",
        Pid::Arg => "arg",
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FbankCfg {
    n_samples: u32,
    n_frames: u32,
    _a: u32,
    _b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LfrCfg {
    n_fbank: u32,
    n_lfr: u32,
    t: u32,
    lid: u32,
    itn: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

struct LayerGpu {
    n1w: wgpu::Buffer,
    n1b: wgpu::Buffer,
    qkv: wgpu::Buffer,
    qkv_b: wgpu::Buffer,
    fsmn: wgpu::Buffer,
    outw: wgpu::Buffer,
    outb: wgpu::Buffer,
    n2w: wgpu::Buffer,
    n2b: wgpu::Buffer,
    w1: wgpu::Buffer,
    w1b: wgpu::Buffer,
    w2: wgpu::Buffer,
    w2b: wgpu::Buffer,
    in_dim: u32,
}

struct Acts {
    alloc: usize,
    in560: wgpu::Buffer,
    h0: wgpu::Buffer,
    h1: wgpu::Buffer,
    norm: wgpu::Buffer,
    qkv: wgpu::Buffer,
    ffn: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    scores: wgpu::Buffer,
    av: wgpu::Buffer,
    ctx: wgpu::Buffer,
    fsmn: wgpu::Buffer,
    logits: wgpu::Buffer,
    ids: wgpu::Buffer,
    // GPU frontend scratch: raw samples, one log-mel row per fbank frame, and
    // the per-LFR-row CMVN sums (the `feat_sum` diagnostic).
    wave: wgpu::Buffer,
    fbank: wgpu::Buffer,
    row_sums: wgpu::Buffer,
}

struct LayerItem {
    ln1: usize,
    qkv: usize,
    fsmn: usize,
    add: usize,
    outp: usize,
    ln2: usize,
    w1: usize,
    w2: usize,
    in_dim: u32,
    add_mode: u32,
}

enum Item {
    Layer(LayerItem),
    Norm { bg: usize, pipe: Pid },
}

struct Plan {
    items: Vec<Item>,
    split: usize,
    qk: usize,
    soft: usize,
    av: usize,
    merge: usize,
    relu: usize,
    ctc: usize,
    arg: usize,
    bgs: Vec<wgpu::BindGroup>,
}

struct Pipes {
    ln: wgpu::ComputePipeline,
    gemm: wgpu::ComputePipeline,
    gemm_nt: wgpu::ComputePipeline,
    gemm_b: wgpu::ComputePipeline,
    gemm_f32c: wgpu::ComputePipeline,
    gemm_tall: wgpu::ComputePipeline,
    gemm_nt_tall: wgpu::ComputePipeline,
    gemm_b_tall: wgpu::ComputePipeline,
    gemm_f32c_tall: wgpu::ComputePipeline,
    gemm_narrow: wgpu::ComputePipeline,
    gemm_nt_narrow: wgpu::ComputePipeline,
    gemm_b_narrow: wgpu::ComputePipeline,
    gemm_f32c_narrow: wgpu::ComputePipeline,
    fsmn: wgpu::ComputePipeline,
    split: wgpu::ComputePipeline,
    soft: wgpu::ComputePipeline,
    soft_cached: wgpu::ComputePipeline,
    merge: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    relu: wgpu::ComputePipeline,
    arg: wgpu::ComputePipeline,
    fbank: wgpu::ComputePipeline,
    lfr: wgpu::ComputePipeline,
    ln_l: wgpu::BindGroupLayout,
    gemm_l: wgpu::BindGroupLayout,
    gemm_b_l: wgpu::BindGroupLayout,
    fsmn_l: wgpu::BindGroupLayout,
    split_l: wgpu::BindGroupLayout,
    soft_l: wgpu::BindGroupLayout,
    merge_l: wgpu::BindGroupLayout,
    add_l: wgpu::BindGroupLayout,
    relu_l: wgpu::BindGroupLayout,
    arg_l: wgpu::BindGroupLayout,
    fbank_l: wgpu::BindGroupLayout,
    lfr_l: wgpu::BindGroupLayout,
}

pub struct GpuEngine {
    gpu: Gpu,
    pipes: Pipes,
    layers: Vec<LayerGpu>,
    after_w: wgpu::Buffer,
    after_b: wgpu::Buffer,
    tp_w: wgpu::Buffer,
    tp_b: wgpu::Buffer,
    ctc_w: wgpu::Buffer,
    ctc_b: wgpu::Buffer,
    // Frontend tables (constant across utterances) and their small uniforms.
    win_tab: wgpu::Buffer,
    mel_tab: wgpu::Buffer,
    embed_tab: wgpu::Buffer,
    cmvn_shift: wgpu::Buffer,
    cmvn_scale: wgpu::Buffer,
    pos_inv: wgpu::Buffer,
    fbank_cfg: wgpu::Buffer,
    lfr_cfg: wgpu::Buffer,
    /// Bind groups for the frontend kernels; rebuilt whenever `Acts` grows.
    front_bg: Option<(wgpu::BindGroup, wgpu::BindGroup)>,
    uni: wgpu::Buffer,
    slot: u64,
    acts: Option<Acts>,
    plan: Option<Plan>,
    max_t: usize,
}

impl GpuEngine {
    pub fn new(gpu: Gpu, weights: &Weights, front: &Frontend) -> Result<Self> {
        let slot = (gpu.limits.min_uniform_buffer_offset_alignment as u64).max(256);
        let max_t = max_frames(gpu.limits.max_storage_buffer_binding_size as u64);
        if max_t < TILE as usize {
            bail!("这张 GPU 的 storage 上限放不下一帧对齐后的注意力矩阵");
        }
        eprintln!("sensevoice: 编译 compute pipeline");
        let pipes = Pipes::new(&gpu)?;
        eprintln!("sensevoice: 上传权重");
        let mut layers = Vec::with_capacity(weights.layers.len() + weights.tp.len());
        for layer in weights.layers.iter().chain(weights.tp.iter()) {
            layers.push(upload_layer(&gpu, layer)?);
        }
        let after_w = up_f32(&gpu, &weights.after_w)?;
        let after_b = up_f32(&gpu, &weights.after_b)?;
        let tp_w = up_f32(&gpu, &weights.tp_w)?;
        let tp_b = up_f32(&gpu, &weights.tp_b)?;
        let ctc_w = up_mat(&gpu, &weights.ctc.w, weights.ctc.rows, weights.ctc.cols, CTC_PAD)?;
        let ctc_b = up_bias(&gpu, &weights.ctc.b, CTC_PAD)?;
        // Frontend tables: the same window/mel/CMVN the CPU path uses, plus
        // the position-encoding inverse table (`add_posenc`'s `inv`, computed
        // with the same formula so the kernels stay table-driven).
        let win_tab = up_f32(&gpu, front.window())?;
        let mel_tab = up_f32(&gpu, front.mel())?;
        let embed_tab = up_f32(&gpu, &weights.embed)?;
        let cmvn_shift = up_f32(&gpu, front.cmvn_shift())?;
        let cmvn_scale = up_f32(&gpu, front.cmvn_scale())?;
        let pos_inv = {
            // `add_posenc` runs over the 560-wide embedding, so its half is
            // EMBED_DIM/2 = 280 (not HIDDEN/2).
            let half = crate::weights::EMBED_DIM / 2;
            let log_inc = 10000f32.ln() / (half as f32 - 1.0);
            let v: Vec<f32> = (0..half).map(|i| (-log_inc * i as f32).exp()).collect();
            up_f32(&gpu, &v)?
        };
        let fbank_cfg = gpu.uniform("fbank-cfg", 256);
        let lfr_cfg = gpu.uniform("lfr-cfg", 256);
        let uni = gpu.uniform("uni", SLOTS as u64 * slot);
        gpu.flush()?;
        let mut engine = Self {
            gpu,
            pipes,
            layers,
            after_w,
            after_b,
            tp_w,
            tp_b,
            ctc_w,
            ctc_b,
            win_tab,
            mel_tab,
            embed_tab,
            cmvn_shift,
            cmvn_scale,
            pos_inv,
            fbank_cfg,
            lfr_cfg,
            front_bg: None,
            uni,
            slot,
            acts: None,
            plan: None,
            max_t,
        };
        // Pre-allocate for a typical clip instead of the 512-frame minimum: the
        // first utterance's `ensure` would otherwise reallocate (frame buffers,
        // bind groups, clears) inside the timed region — ~18 ms at 90 s. Model
        // load is outside the RTFx clock, so pay it there. 3072 frames covers
        // ~184 s of audio; longer clips still grow on demand.
        engine.alloc_for(3072.min(max_t))?;
        Ok(engine)
    }

    /// Run the fbank + LFR frontend kernels for one segment and return
    /// `feat_sum`. The encoder input lands in the shared 560-wide buffer that
    /// `encode` reads.
    pub fn run_frontend(
        &mut self,
        samples: &[f32],
        n_fbank: usize,
        t: usize,
        lid: u32,
        itn: u32,
    ) -> Result<()> {
        self.ensure(t)?;
        let (fbank_bg, lfr_bg) = self.front_bg.as_ref().context("frontend bind groups")?;
        let acts = self.acts.as_ref().unwrap();
        if (samples.len() * 4) as u64 > acts.wave.size() {
            bail!(
                "frontend 波形缓冲 {} 字节放不下 {} 个采样",
                acts.wave.size(),
                samples.len()
            );
        }
        self.gpu.write_at(&acts.wave, 0, bytemuck::cast_slice(samples));
        self.gpu.write_at(
            &self.fbank_cfg,
            0,
            bytemuck::bytes_of(&FbankCfg {
                n_samples: samples.len() as u32,
                n_frames: n_fbank as u32,
                _a: 0,
                _b: 0,
            }),
        );
        self.gpu.write_at(
            &self.lfr_cfg,
            0,
            bytemuck::bytes_of(&LfrCfg {
                n_fbank: n_fbank as u32,
                n_lfr: 0, // unused by the kernel; the row count is the grid
                t: t as u32,
                lid,
                itn,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            }),
        );
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frontend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipes.fbank);
            pass.set_bind_group(0, fbank_bg, &[]);
            pass.dispatch_workgroups(n_fbank as u32, 1, 1);
            pass.set_pipeline(&self.pipes.lfr);
            pass.set_bind_group(0, lfr_bg, &[]);
            pass.dispatch_workgroups(t as u32, 1, 1);
        }
        self.gpu.queue.submit([enc.finish()]);
        self.gpu.flush()?;
        if tuning::trace() {
            // Diagnostic: the same sum/max/FNV fingerprint `frontend::joint`
            // prints on the CPU side, so the two stages can be compared.
            for (label, buf, n) in [
                ("gpu_fbank", &acts.fbank, n_fbank * 80),
                ("gpu_x", &acts.in560, t * 560),
            ] {
                let raw = self.gpu.readback(buf, (n * 4) as u64)?;
                let mut sum = 0.0f64;
                let mut max = 0.0f32;
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for c in raw.chunks_exact(4) {
                    let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    sum += f64::from(v);
                    max = max.max(v.abs());
                    h ^= u64::from(v.to_bits());
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
                eprintln!("trace {label} sum={sum:.9} max={max:.9} n={n} h={h:016x}");
            }
        }
        Ok(())
    }

    /// Run the encoder graph on whatever `run_frontend` left in the shared
    /// input buffer.
    pub fn encode(&mut self, t: usize, n_lfr: usize, ban: Option<usize>) -> Result<(Vec<i32>, f32)> {
        let pad = self.ensure(t)?;
        let ids = self.acts.as_ref().unwrap().ids.clone();
        let (ops, blob) = self.record(pad, t, ban)?;
        if tuning::gpu_dump_ops() {
            Self::dump_ops(&ops, pad, t);
        }
        self.gpu.write_at(&self.uni, 0, &blob);
        let stop = tuning::gpu_stop();
        let run = ops.len().min(stop);
        let guard = self.gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        // Windows TDR resets the device when one submission keeps the GPU busy
        // past ~2 s. A lost device silently turns every later dispatch into a
        // no-op, leaving `ids` all zeros and the decode empty, so the graph is
        // split into batches that are each waited on.
        //
        // Per-dispatch cost grows with `t` (measured ~0.95 ms at t=1504, and
        // t=3004 in one submission already trips the watchdog). This targets
        // roughly 1.1 s of GPU work per submission; the wait between batches
        // costs about 1% of the encode, so this is not a throughput trade.
        let chunk = tuning::gpu_chunk().unwrap_or_else(|| (1_400_000 / t.max(1)).clamp(16, 2048));
        // Per-dispatch timing is a diagnostic build path: it needs one compute
        // pass per dispatch, so it only runs under `SVS_GPU_PROF=1`.
        let prof = tuning::gpu_prof()
            && self.gpu.features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && run * 2 <= 4096;
        if tuning::gpu_prof() && !prof {
            eprintln!("gpu prof: 不可用（缺 TIMESTAMP_QUERY 或 op 数超限）");
        }
        let qset = prof.then(|| {
            self.gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("svs-prof"),
                ty: wgpu::QueryType::Timestamp,
                count: (run * 2) as u32,
            })
        });
        let mut i = 0usize;
        while i < run {
            let end = (i + chunk).min(run);
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            if prof {
                for (k, op) in ops[i..end].iter().enumerate() {
                    let idx = (i + k) * 2;
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("svs-prof"),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: qset.as_ref().unwrap(),
                            beginning_of_pass_write_index: Some(idx as u32),
                            end_of_pass_write_index: Some(idx as u32 + 1),
                        }),
                    });
                    pass.set_pipeline(self.pipes.get(op.pipe));
                    pass.set_bind_group(
                        0,
                        &self.plan.as_ref().unwrap().bgs[op.bg],
                        &[op.slot as u32 * self.slot as u32],
                    );
                    pass.dispatch_workgroups(op.x, op.y, op.z);
                }
            } else {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("svs"),
                    timestamp_writes: None,
                });
                for op in &ops[i..end] {
                    pass.set_pipeline(self.pipes.get(op.pipe));
                    pass.set_bind_group(
                        0,
                        &self.plan.as_ref().unwrap().bgs[op.bg],
                        &[op.slot as u32 * self.slot as u32],
                    );
                    pass.dispatch_workgroups(op.x, op.y, op.z);
                }
            }
            self.gpu.queue.submit([enc.finish()]);
            self.gpu
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("poll batch")?;            i = end;
        }
        if prof {
            self.dump_prof(&qset.unwrap(), &ops)?;
        }
        // One map serves both readbacks: the token ids and the frontend's
        // per-row CMVN sums (the `feat_sum` diagnostic).
        let row_sums = self.acts.as_ref().unwrap().row_sums.clone();
        let raw = self
            .gpu
            .readback_many(&[(&ids, (t * 4) as u64), (&row_sums, (n_lfr * 4) as u64)])?;
        let (id_bytes, sum_bytes) = raw.split_at(t * 4);
        let mut feat_sum = 0.0f32;
        for c in sum_bytes.chunks_exact(4) {
            feat_sum += f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        if let Some(err) = pollster::block_on(guard.pop()) {
            bail!("GPU 校验失败: {err}");
        }
        let mut frames = Vec::with_capacity(t);
        for c in id_bytes.chunks_exact(4) {
            frames.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        if tuning::gpu_debug() {
            self.dump_debug(pad, t, ops.len(), &frames)?;
        }
        Ok((greedy(&frames), feat_sum))
    }
}

fn scan_f32(name: &str, bytes: &[u8]) {
    let mut nnan = 0u32;
    let mut ninf = 0u32;
    let mut max = 0f32;
    let mut n = 0u32;
    for c in bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        n += 1;
        if v.is_nan() {
            nnan += 1;
        } else if v.is_infinite() {
            ninf += 1;
        } else {
            max = max.max(v.abs());
        }
    }
    eprintln!("gpu debug {name} n={n} nnan={nnan} ninf={ninf} max={max}");
}

impl GpuEngine {
    fn ensure(&mut self, t: usize) -> Result<usize> {
        let pad = (t + TILE as usize - 1) / TILE as usize * TILE as usize;
        if pad > self.max_t {
            bail!(
                "这段音频有 {t} 帧（对齐到 {pad}），超过这张 GPU 单次能放下的 {} 帧。请把更短的 VAD 段通过 --vad-segments 传进来",
                self.max_t
            );
        }
        let have = self.acts.as_ref().map(|a| a.alloc).unwrap_or(0);
        if have < pad {
            self.alloc_for(pad)?;
        }
        Ok(pad)
    }

    fn alloc_for(&mut self, alloc: usize) -> Result<()> {
        let acts = Acts::new(&self.gpu, alloc)?;
        let plan = build_plan(self, &acts)?;
        self.clear(&acts)?;
        let front_bg = self.build_front_bgs(&acts);
        self.acts = Some(acts);
        self.plan = Some(plan);
        self.front_bg = Some(front_bg);
        Ok(())
    }

    /// The frontend kernels bind `Acts` buffers, so their bind groups follow
    /// every reallocation.
    fn build_front_bgs(&self, acts: &Acts) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let dev = &self.gpu.device;
        let fbank = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fbank-bg"),
            layout: &self.pipes.fbank_l,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: acts.wave.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.win_tab.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.mel_tab.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: acts.fbank.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.fbank_cfg.as_entire_binding() },
            ],
        });
        let lfr = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lfr-bg"),
            layout: &self.pipes.lfr_l,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: acts.fbank.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.embed_tab.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.cmvn_shift.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.cmvn_scale.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.pos_inv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: acts.row_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: acts.in560.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.lfr_cfg.as_entire_binding() },
            ],
        });
        (fbank, lfr)
    }

    fn clear(&self, acts: &Acts) -> Result<()> {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        for buf in acts.all() {
            enc.clear_buffer(buf, 0, None);
        }
        self.gpu.queue.submit([enc.finish()]);
        self.gpu.flush()
    }
}

// The uniform blob is rebuilt every encode. Storing it on the engine avoids
// threading a borrow through `record` and the command encoder together.
impl GpuEngine {
    /// Resolve the per-dispatch timestamps and print a kernel-class breakdown.
    ///
    /// Diagnostic only (`SVS_GPU_PROF=1`): it never changes dispatch order or
    /// count, so the transcript stays identical to a normal run.
    fn dump_prof(&self, qset: &wgpu::QuerySet, ops: &[Op]) -> Result<()> {
        let n = ops.len();
        let bytes = (n * 2 * 8) as u64;
        // wgpu forbids MAP_READ together with QUERY_RESOLVE (a mappable buffer
        // may only pair with the opposite COPY). Resolve into an intermediate
        // and copy that into the staging buffer, same as `Gpu::readback`.
        let resolve = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prof-resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prof-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.resolve_query_set(qset, 0..(n * 2) as u32, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
        self.gpu.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll prof")?;
        rx.recv().context("prof map cb")?.context("prof map buffer")?;
        let view = slice.get_mapped_range()?;
        let mut ts: Vec<u64> = Vec::with_capacity(n * 2);
        for c in view[..bytes as usize].chunks_exact(8) {
            ts.push(u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]));
        }
        drop(view);
        staging.unmap();

        let mut by_class: Vec<(&'static str, usize, u64)> = Vec::new();
        let mut per_op: Vec<(u64, usize)> = Vec::with_capacity(n);
        let mut total = 0u64;
        for (k, op) in ops.iter().enumerate() {
            let d = ts[2 * k + 1].saturating_sub(ts[2 * k]);
            total += d;
            per_op.push((d, k));
            let name = pid_name(op.pipe);
            match by_class.iter_mut().find(|(nm, ..)| *nm == name) {
                Some(e) => {
                    e.1 += 1;
                    e.2 += d;
                }
                None => by_class.push((name, 1, d)),
            }
        }
        let tot = total.max(1) as f64;
        eprintln!(
            "gpu prof: {} dispatches, GPU busy {:.3} ms",
            n,
            total as f64 / 1e6
        );
        by_class.sort_by_key(|e| std::cmp::Reverse(e.2));
        for (name, cnt, ns) in &by_class {
            eprintln!(
                "  {:<10} {:>4} ops {:>9.3} ms {:>5.1}%",
                name,
                cnt,
                *ns as f64 / 1e6,
                100.0 * *ns as f64 / tot
            );
        }
        per_op.sort_by_key(|e| std::cmp::Reverse(e.0));
        eprintln!("gpu prof: 最慢 10 个 dispatch");
        for (ns, k) in per_op.iter().take(10) {
            let op = &ops[*k];
            eprintln!(
                "  #{:<4} {:<10} {:>8.3} ms dispatch=({},{},{}) slot={}",
                k,
                pid_name(op.pipe),
                *ns as f64 / 1e6,
                op.x,
                op.y,
                op.z,
                op.slot
            );
        }

        // Same-class ops can have very different shapes (gemm_b is qkv at
        // N=1536, w1 at N=2048 and the CTC head at N=25088), so a class total
        // hides which shape is actually slow. Aggregate by dispatch shape.
        let mut by_shape: Vec<((&'static str, u32, u32, u32), usize, u64)> = Vec::new();
        for (k, op) in ops.iter().enumerate() {
            let d = ts[2 * k + 1].saturating_sub(ts[2 * k]);
            let key = (pid_name(op.pipe), op.x, op.y, op.z);
            match by_shape.iter_mut().find(|(ks, ..)| *ks == key) {
                Some(e) => {
                    e.1 += 1;
                    e.2 += d;
                }
                None => by_shape.push((key, 1, d)),
            }
        }
        by_shape.sort_by_key(|e| std::cmp::Reverse(e.2));
        eprintln!("gpu prof: 按形状聚合（Top 20）");
        for ((name, x, y, z), cnt, ns) in by_shape.iter().take(20) {
            let mean = *ns as f64 / 1e6 / (*cnt as f64);
            eprintln!(
                "  {:<10} ({:>3},{:>3},{:>1}) x{:<3} {:>9.3} ms  avg {:>7.3} ms  {:>5.1}%",
                name,
                x,
                y,
                z,
                cnt,
                *ns as f64 / 1e6,
                mean,
                100.0 * *ns as f64 / tot
            );
        }
        Ok(())
    }

    /// `SVS_GPU_DUMP_OPS=1`: print every recorded op with its bind group and
    /// dispatch shape. Diagnostic only.
    fn dump_ops(ops: &[Op], pad: usize, t: usize) {
        eprintln!("gpu ops total={} pad={pad} t={t}", ops.len());
        for (i, op) in ops.iter().enumerate() {
            eprintln!(
                "op {i:4} {:<10} bg={:<4} slot={:<4} dispatch=({},{},{})",
                pid_name(op.pipe),
                op.bg,
                op.slot,
                op.x,
                op.y,
                op.z
            );
        }
    }

    /// `SVS_GPU_DEBUG=1`: scan the intermediate activations of the last run.
    ///
    /// Diagnostic only. It reads (never writes) the buffers `encode` just
    /// filled, so the execution path stays readable and unchanged.
    fn dump_debug(&self, pad: usize, t: usize, n_ops: usize, frames: &[i32]) -> Result<()> {
        eprintln!("gpu debug slot={} ops={} pad={pad} t={t}", self.slot, n_ops);
        let a = self.acts.as_ref().unwrap();
        let hid4 = (pad * HIDDEN * 4) as u64;
        let norm4 = (pad * 560 * 4) as u64;
        let logits4 = ((t * VOCAB * 4) as u64).min((pad * CTC_PAD * 4) as u64);
        let ids4 = (t * 4) as u64;
        let parts: Vec<(&wgpu::Buffer, u64)> = vec![
            (&a.in560, 64),
            (&a.h0, hid4),
            (&a.h1, hid4),
            (&a.norm, norm4),
            (&a.ctx, hid4),
            (&a.qkv, (pad * QKV * 4) as u64),
            (&a.logits, logits4),
            (&a.ids, ids4),
        ];
        let blob = self.gpu.readback_many(&parts)?;
        // Same 4-byte alignment readback_many uses to lay the parts out.
        let mut offs: Vec<usize> = Vec::new();
        let mut at = 0usize;
        for (_, n) in &parts {
            offs.push(at);
            at += ((((*n).max(4)) + 3) & !3) as usize;
        }
        let seg = |i: usize| -> &[u8] { &blob[offs[i]..offs[i] + parts[i].1 as usize] };

        scan_f32("in", seg(0));
        scan_f32("h0", seg(1));
        scan_f32("h1", seg(2));
        scan_f32("norm", seg(3));
        scan_f32("ctx", seg(4));
        scan_f32("qkv", seg(5));
        scan_f32("logits", seg(6));
        let row = seg(6);
        let mut max_v = f32::NEG_INFINITY;
        let mut max_i = 0usize;
        let mut nnan = 0u32;
        let mut v0 = 0f32;
        for c in 0..VOCAB.min(row.len() / 4) {
            let v = f32::from_le_bytes([row[c * 4], row[c * 4 + 1], row[c * 4 + 2], row[c * 4 + 3]]);
            if c == 0 {
                v0 = v;
            }
            if v.is_nan() {
                nnan += 1;
            } else if v > max_v {
                max_v = v;
                max_i = c;
            }
        }
        let idbytes = seg(7);
        let ids_head: Vec<i32> = idbytes
            .chunks_exact(4)
            .take(12)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        eprintln!(
            "gpu debug logit0={v0} max={max_v} at={max_i} nnan={nnan} ids_head={ids_head:?} frames_len={}",
            frames.len()
        );
        Ok(())
    }

    fn record(&self, pad: usize, valid: usize, ban: Option<usize>) -> Result<(Vec<Op>, Vec<u8>)> {
        let slot = self.slot as usize;
        let mut blob = vec![0u8; SLOTS * slot];
        let mut n = 0u32;
        let mut ops = Vec::new();
        let plan = self.plan.as_ref().unwrap();
        let mut push = |bytes: &[u8]| -> Result<u32> {
            if n as usize >= SLOTS {
                bail!("uniform slab 用尽");
            }
            let at = n as usize * slot;
            blob[at..at + bytes.len()].copy_from_slice(bytes);
            let s = n;
            n += 1;
            Ok(s)
        };

        let pad_u = pad as u32;
        let valid_u = valid as u32;
        let scale = 128f32.sqrt().recip();

        for item in &plan.items {
            match item {
                Item::Layer(layer) => {
                    let ln1_pipe = if layer.in_dim == 560 { Pid::Ln } else { Pid::LnF32F16 };
                    push_ln(&mut push, &mut ops, ln1_pipe, layer.ln1, layer.in_dim, pad_u)?;
                    push_gemm(
                        &mut push, &mut ops, Pid::GemmB, layer.qkv, pad_u, QKV as u32,
                        layer.in_dim, QKV as u32, 0, 0, 0, layer.in_dim, 1,
                    )?;
                    let fsmn_slot = push(bytemuck::bytes_of(&FsmnCfg {
                        t_valid: valid_u,
                        t_pad: pad_u,
                        _p0: 0,
                        _p1: 0,
                    }))?;
                    ops.push(Op { pipe: Pid::Fsmn, bg: layer.fsmn, slot: fsmn_slot, x: 2, y: pad_u, z: 1 });
                    let split_slot = push(bytemuck::bytes_of(&SplitCfg {
                        t_pad: pad_u, t_valid: valid_u, scale, p: 0,
                    }))?;
                    ops.push(Op { pipe: Pid::Split, bg: plan.split, slot: split_slot, x: 1, y: 4, z: pad_u });
                    let bsa = pad_u * 128;
                    let bsc_scores = pad_u * pad_u;
                    push_gemm(
                        &mut push, &mut ops, Pid::Gemm, plan.qk, pad_u, pad_u, 128, pad_u,
                        bsa, bsa, bsc_scores, 128, 4,
                    )?;
                    let soft_slot = push(bytemuck::bytes_of(&SoftCfg {
                        t_valid: valid_u,
                        t_pad: pad_u,
                        _p0: 0,
                        _p1: 0,
                    }))?;
                    // The direct kernel re-reads the row three times; up to
                    // 4096 wide it fits in shared memory and costs one read.
                    let soft_pipe = if pad_u <= 4096 { Pid::SoftCached } else { Pid::Soft };
                    ops.push(Op { pipe: soft_pipe, bg: plan.soft, slot: soft_slot, x: 4 * pad_u, y: 1, z: 1 });
                    push_gemm(
                        &mut push, &mut ops, Pid::GemmNt, plan.av, pad_u, 128, pad_u, 128,
                        bsc_scores, bsa, pad_u * 128, pad_u, 4,
                    )?;
                    let merge_slot = push(bytemuck::bytes_of(&MergeCfg {
                        t_pad: pad_u,
                        t_valid: valid_u,
                        _p0: 0,
                        _p1: 0,
                    }))?;
                    ops.push(Op { pipe: Pid::Merge, bg: plan.merge, slot: merge_slot, x: 1, y: 4, z: pad_u });
                    let n_elem = pad_u * HIDDEN as u32;
                    let add_slot = push(bytemuck::bytes_of(&AddCfg {
                        n: n_elem,
                        mode: layer.add_mode,
                        _p0: 0,
                        _p1: 0,
                    }))?;
                    ops.push(Op {
                        pipe: Pid::Add,
                        bg: layer.add,
                        slot: add_slot,
                        x: n_elem.div_ceil(256),
                        y: 1,
                        z: 1,
                    });
                    push_gemm(
                        &mut push, &mut ops, Pid::GemmF32c, layer.outp, pad_u, HIDDEN as u32,
                        HIDDEN as u32, HIDDEN as u32, 0, 0, 0, HIDDEN as u32, 1,
                    )?;
                    push_ln(&mut push, &mut ops, Pid::LnF32F16, layer.ln2, HIDDEN as u32, pad_u)?;
                    push_gemm(
                        &mut push, &mut ops, Pid::GemmB, layer.w1, pad_u, FFN as u32,
                        HIDDEN as u32, FFN as u32, 0, 0, 0, HIDDEN as u32, 1,
                    )?;
                    let relu_n = pad_u * FFN as u32;
                    let relu_slot = push(bytemuck::bytes_of(&ReluCfg {
                        n: relu_n,
                        _a: 0,
                        _b: 0,
                        _c: 0,
                    }))?;
                    ops.push(Op {
                        pipe: Pid::Relu,
                        bg: plan.relu,
                        slot: relu_slot,
                        x: (relu_n + 255) / 256,
                        y: 1,
                        z: 1,
                    });
                    push_gemm(
                        &mut push, &mut ops, Pid::GemmF32c, layer.w2, pad_u, HIDDEN as u32,
                        FFN as u32, HIDDEN as u32, 0, 0, 0, FFN as u32, 1,
                    )?;
                }
                Item::Norm { bg, pipe } => {
                    push_ln(&mut push, &mut ops, *pipe, *bg, HIDDEN as u32, pad_u)?;
                }
            }
        }
        push_gemm(
            &mut push, &mut ops, Pid::GemmB, plan.ctc, pad_u, CTC_PAD as u32,
            HIDDEN as u32, CTC_PAD as u32, 0, 0, 0, HIDDEN as u32, 1,
        )?;
        let ban_id = match ban {
            Some(id) => id as u32,
            None => u32::MAX,
        };
        let arg_slot = push(bytemuck::bytes_of(&ArgCfg {
            t_valid: valid_u,
            n_vocab: VOCAB as u32,
            stride: CTC_PAD as u32,
            ban: ban_id,
        }))?;
        ops.push(Op { pipe: Pid::Arg, bg: plan.arg, slot: arg_slot, x: valid_u, y: 1, z: 1 });

        drop(push);
        let nbytes = n as usize * slot;
        Ok((ops, blob[..nbytes].to_vec()))
    }
}

fn push_ln(
    push: &mut impl FnMut(&[u8]) -> Result<u32>,
    ops: &mut Vec<Op>,
    pipe: Pid,
    bg: usize,
    d: u32,
    pad: u32,
) -> Result<()> {
    let slot = push(bytemuck::bytes_of(&LnCfg { d, eps: 1e-5, a: 0, b: 0 }))?;
    // Four rows per 128-thread workgroup (32 lanes each).
    ops.push(Op { pipe, bg, slot, x: pad / 4, y: 1, z: 1 });
    Ok(())
}

fn push_gemm(
    push: &mut impl FnMut(&[u8]) -> Result<u32>,
    ops: &mut Vec<Op>,
    pipe: Pid,
    bg: usize,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    bsa: u32,
    bsb: u32,
    bsc: u32,
    lda: u32,
    z: u32,
) -> Result<()> {
    let g = GDims { n, k, ldc, bsa, bsb, bsc, lda };
    let slot = push(bytemuck::bytes_of(&g))?;
    // Short audio leaves a 128-row tile with a grid too small to fill 15 SMs
    // (at pad = 512 an outp dispatch is 16 blocks).  The tall tile halves the
    // rows per workgroup, doubling the m axis; it is the same arithmetic.
    // Tile choice by M, all measured with paired A/B on this card:
    //   M <= 256  the tall tile (64x128, BK=16) wins by ~5% on 15s_en;
    //   256 < M <= 512  the narrow tile (128x64, BK=8) wins by ~1% on 30s_zh;
    //   larger M  the 128x128/BK=8 default, where the deeper slab and the
    //   extra blocks stop paying.
    let (pipe, bm, bn) = if m <= 256 {
        (
            match pipe {
                Pid::Gemm => Pid::GemmTall,
                Pid::GemmNt => Pid::GemmNtTall,
                Pid::GemmB => Pid::GemmBTall,
                Pid::GemmF32c => Pid::GemmF32cTall,
                other => other,
            },
            GEMM_BM_TALL,
            GEMM_BN,
        )
    } else if m <= 512 {
        (
            match pipe {
                Pid::Gemm => Pid::GemmNarrow,
                Pid::GemmNt => Pid::GemmNtNarrow,
                Pid::GemmB => Pid::GemmBNarrow,
                Pid::GemmF32c => Pid::GemmF32cNarrow,
                other => other,
            },
            128,
            64,
        )
    } else {
        (pipe, GEMM_BM, GEMM_BN)
    };
    ops.push(Op { pipe, bg, slot, x: n / bn, y: m / bm, z });
    Ok(())
}

fn max_frames(binding: u64) -> usize {
    let mut tp = 0usize;
    let mut next = TILE as usize;
    while next <= 8192 {
        let scores = 4u64 * next as u64 * next as u64 * 4;
        let ctc = next as u64 * CTC_PAD as u64 * 4;
        if scores > binding || ctc > binding {
            break;
        }
        tp = next;
        next += TILE as usize;
    }
    tp
}

impl Acts {
    fn new(gpu: &Gpu, alloc: usize) -> Result<Self> {
        let f32b = |rows: usize, cols: usize, label: &str| gpu.storage(label, (rows * cols * 4) as u64);
        Ok(Self {
            alloc,
            in560: f32b(alloc, 560, "in"),
            h0: f32b(alloc, HIDDEN, "h0"),
            h1: f32b(alloc, HIDDEN, "h1"),
            norm: f32b(alloc, 560, "norm"),
            qkv: f32b(alloc, QKV, "qkv"),
            ffn: f32b(alloc, FFN, "ffn"),
            q: f32b(4 * alloc, 128, "q"),
            k: f32b(4 * alloc, 128, "k"),
            v: f32b(4 * alloc, 128, "v"),
            scores: f32b(4 * alloc, alloc, "scores"),
            av: f32b(4 * alloc, 128, "av"),
            ctx: f32b(alloc, HIDDEN, "ctx"),
            fsmn: f32b(alloc, HIDDEN, "fsmn"),
            logits: f32b(alloc, CTC_PAD, "logits"),
            ids: gpu.storage("ids", (alloc * 4) as u64),
            // `wave` holds the raw segment; n <= n_fbank*160 and n_fbank <=
            // 6*alloc (the LFR rows plus the four query rows fit in `alloc`).
            wave: gpu.storage("wave", ((alloc * 960 + 960) * 4) as u64),
            fbank: gpu.storage("fbank", (alloc * 480 * 4) as u64),
            row_sums: gpu.storage("row_sums", (alloc * 4) as u64),
        })
    }

    fn all(&self) -> [&wgpu::Buffer; 15] {
        [
            &self.in560, &self.h0, &self.h1, &self.norm, &self.qkv, &self.ffn, &self.q, &self.k,
            &self.v, &self.scores, &self.av, &self.ctx, &self.fsmn, &self.logits, &self.ids,
        ]
    }

    fn hid(&self, id: Hid) -> &wgpu::Buffer {
        match id {
            Hid::In => &self.in560,
            Hid::H0 => &self.h0,
            Hid::H1 => &self.h1,
        }
    }
}

fn build_plan(engine: &GpuEngine, acts: &Acts) -> Result<Plan> {
    let dev = &engine.gpu.device;
    let pipes = &engine.pipes;
    let mut bgs = Vec::new();
    let mut push = |bg: wgpu::BindGroup| -> usize {
        let i = bgs.len();
        bgs.push(bg);
        i
    };
    let split = push(bind(
        dev, &pipes.split_l, "split",
        &[
            (0, &acts.qkv),
            (1, &acts.q),
            (2, &acts.k),
            (3, &acts.v),
        ],
        4,
        &engine.uni,
    ));
    let qk = push(bind(dev, &pipes.gemm_l, "qk", &[(0, &acts.q), (1, &acts.k), (2, &acts.scores)], 3, &engine.uni));
    let soft = push(bind(dev, &pipes.soft_l, "soft", &[(0, &acts.scores)], 1, &engine.uni));
    let av = push(bind(
        dev, &pipes.gemm_l, "av",
        &[(0, &acts.scores), (1, &acts.v), (2, &acts.av)],
        3,
        &engine.uni,
    ));
    let merge = push(bind(dev, &pipes.merge_l, "merge", &[(0, &acts.av), (1, &acts.ctx)], 2, &engine.uni));
    let relu = push(bind(dev, &pipes.relu_l, "relu", &[(0, &acts.ffn)], 1, &engine.uni));

    let mut items = Vec::new();
    let mut src = Hid::In;
    for i in 0..engine.layers.len() {
        let dst = match src {
            Hid::In | Hid::H1 => Hid::H0,
            Hid::H0 => Hid::H1,
        };
        let layer = &engine.layers[i];
        let src_buf = acts.hid(src);
        let dst_buf = acts.hid(dst);
        let ln1 = push(bind(
            dev, &pipes.ln_l, "ln1",
            &[(0, src_buf), (1, &layer.n1w), (2, &layer.n1b), (4, &acts.norm)],
            3,
            &engine.uni,
        ));
        let qkv = push(bind(
            dev, &pipes.gemm_b_l, "qkv",
            &[
                (0, &acts.norm),
                (1, &layer.qkv),
                (2, &acts.qkv),
                (4, &layer.qkv_b),
            ],
            3,
            &engine.uni,
        ));
        let fsmn = push(bind(
            dev, &pipes.fsmn_l, "fsmn",
            &[(0, &acts.qkv), (1, &layer.fsmn), (3, &acts.fsmn)],
            2,
            &engine.uni,
        ));
        let add = push(bind(
            dev, &pipes.add_l, "add",
            &[(0, src_buf), (1, &acts.fsmn), (2, dst_buf)],
            3,
            &engine.uni,
        ));
        let outp = push(bind(
            dev, &pipes.gemm_b_l, "out",
            &[(0, &acts.ctx), (1, &layer.outw), (2, dst_buf), (4, &layer.outb)],
            3,
            &engine.uni,
        ));
        let ln2 = push(bind(
            dev, &pipes.ln_l, "ln2",
            &[(0, dst_buf), (1, &layer.n2w), (2, &layer.n2b), (4, &acts.norm)],
            3,
            &engine.uni,
        ));
        let w1 = push(bind(
            dev, &pipes.gemm_b_l, "w1",
            &[(0, &acts.norm), (1, &layer.w1), (2, &acts.ffn), (4, &layer.w1b)],
            3,
            &engine.uni,
        ));
        let w2 = push(bind(
            dev, &pipes.gemm_b_l, "w2",
            &[(0, &acts.ffn), (1, &layer.w2), (2, dst_buf), (4, &layer.w2b)],
            3,
            &engine.uni,
        ));
        items.push(Item::Layer(LayerItem {
            ln1, qkv, fsmn, add, outp, ln2, w1, w2,
            in_dim: layer.in_dim,
            add_mode: if i == 0 { 0 } else { 1 },
        }));
        src = dst;
        if i + 1 == 50 {
            let dst_h = match src {
                Hid::H0 => Hid::H1,
                Hid::H1 => Hid::H0,
                Hid::In => unreachable!(),
            };
            let bg = push(bind(
                dev, &pipes.ln_l, "after",
                &[
                    (0, acts.hid(src)),
                    (1, &engine.after_w),
                    (2, &engine.after_b),
                    (4, acts.hid(dst_h)),
                ],
                3,
                &engine.uni,
            ));
            items.push(Item::Norm { bg, pipe: Pid::LnF32 });
            src = dst_h;
        }
    }
    let bg = push(bind(
        dev, &pipes.ln_l, "tp",
        &[
            (0, acts.hid(src)),
            (1, &engine.tp_w),
            (2, &engine.tp_b),
            (4, &acts.norm),
        ],
        3,
        &engine.uni,
    ));
    items.push(Item::Norm { bg, pipe: Pid::LnF32F16 });
    let ctc = push(bind(
        dev, &pipes.gemm_b_l, "ctc",
        &[
            (0, &acts.norm),
            (1, &engine.ctc_w),
            (2, &acts.logits),
            (4, &engine.ctc_b),
        ],
        3,
        &engine.uni,
    ));
    let arg = push(bind(
        dev, &pipes.arg_l, "arg",
        &[(0, &acts.logits), (2, &acts.ids)],
        1,
        &engine.uni,
    ));
    Ok(Plan { items, split, qk, soft, av, merge, relu, ctc, arg, bgs })
}

fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    bufs: &[(u32, &wgpu::Buffer)],
    uni_binding: u32,
    uni: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let mut entries: Vec<wgpu::BindGroupEntry> = bufs
        .iter()
        .map(|(b, buf)| wgpu::BindGroupEntry {
            binding: *b,
            resource: buf.as_entire_binding(),
        })
        .collect();
    entries.push(wgpu::BindGroupEntry {
        binding: uni_binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: uni,
            offset: 0,
            size: NonZeroU64::new(256),
        }),
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

impl Pipes {
    fn new(gpu: &Gpu) -> Result<Self> {
        let dev = &gpu.device;
        let ln_l = bgl(dev, "ln", &[ro(0), ro(1), ro(2), uni(3), rw(4)]);
        let gemm_l = bgl(dev, "gemm", &[ro(0), ro(1), rw(2), uni(3)]);
        let gemm_b_l = bgl(dev, "gemm-b", &[ro(0), ro(1), rw(2), uni(3), ro(4)]);
        let fsmn_l = bgl(dev, "fsmn", &[ro(0), ro(1), uni(2), rw(3)]);
        let split_l = bgl(dev, "split", &[ro(0), rw(1), rw(2), rw(3), uni(4)]);
        let soft_l = bgl(dev, "soft", &[rw(0), uni(1)]);
        let merge_l = bgl(dev, "merge", &[ro(0), rw(1), uni(2)]);
        let add_l = bgl(dev, "add", &[ro(0), ro(1), rw(2), uni(3)]);
        let relu_l = bgl(dev, "relu", &[rw(0), uni(1)]);
        let arg_l = bgl(dev, "arg", &[ro(0), uni(1), rw(2)]);
        let fbank_l = bgl(dev, "fbank", &[ro(0), ro(1), ro(2), rw(3), uni_s(4)]);
        let lfr_l = bgl(
            dev,
            "lfr",
            &[ro(0), ro(1), ro(2), ro(3), ro(4), rw(5), rw(6), uni_s(7)],
        );
        let pl = |l: &wgpu::BindGroupLayout, name: &str| {
            dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[Some(l)],
                immediate_size: 0,
            })
        };
        let ln_pl = pl(&ln_l, "ln");
        let gemm_pl = pl(&gemm_l, "gemm");
        let gemm_b_pl = pl(&gemm_b_l, "gemm-b");
        let fsmn_pl = pl(&fsmn_l, "fsmn");
        let split_pl = pl(&split_l, "split");
        let soft_pl = pl(&soft_l, "soft");
        let merge_pl = pl(&merge_l, "merge");
        let add_pl = pl(&add_l, "add");
        let relu_pl = pl(&relu_l, "relu");
        let arg_pl = pl(&arg_l, "arg");
        let fbank_pl = pl(&fbank_l, "fbank");
        let lfr_pl = pl(&lfr_l, "lfr");
        Ok(Self {
            ln: gpu.pipeline("ln", &shaders::layernorm(), "layernorm", &ln_pl)?,
            gemm: gpu.pipeline("gemm", &gemm_f32(false, false, false), "gemm", &gemm_pl)?,
            gemm_nt: gpu.pipeline("gemm-nt", &gemm_f32(true, false, false), "gemm", &gemm_pl)?,
            gemm_b: gpu.pipeline("gemm-b", &gemm_f32_bias(false), "gemm", &gemm_b_pl)?,
            gemm_f32c: gpu.pipeline("gemm-f32c", &gemm_f32_bias(true), "gemm", &gemm_b_pl)?,
            gemm_tall: gpu.pipeline("gemm-tall", &gemm_f32_tall(false, false, false), "gemm", &gemm_pl)?,
            gemm_nt_tall: gpu.pipeline("gemm-nt-tall", &gemm_f32_tall(true, false, false), "gemm", &gemm_pl)?,
            gemm_b_tall: gpu.pipeline("gemm-b-tall", &gemm_f32_bias_tall(false), "gemm", &gemm_b_pl)?,
            gemm_f32c_tall: gpu.pipeline("gemm-f32c-tall", &gemm_f32_bias_tall(true), "gemm", &gemm_b_pl)?,
            gemm_narrow: gpu.pipeline("gemm-narrow", &gemm_f32_narrow(false, false, false), "gemm", &gemm_pl)?,
            gemm_nt_narrow: gpu.pipeline("gemm-nt-narrow", &gemm_f32_narrow(true, false, false), "gemm", &gemm_pl)?,
            gemm_b_narrow: gpu.pipeline("gemm-b-narrow", &gemm_f32_narrow(false, false, true), "gemm", &gemm_b_pl)?,
            gemm_f32c_narrow: gpu.pipeline("gemm-f32c-narrow", &gemm_f32_narrow(false, true, true), "gemm", &gemm_b_pl)?,
            fsmn: gpu.pipeline("fsmn", &shaders::fsmn(), "fsmn", &fsmn_pl)?,
            split: gpu.pipeline("split", &shaders::split_qkv(), "split", &split_pl)?,
            soft: gpu.pipeline("soft", &shaders::softmax(), "softmax", &soft_pl)?,
            soft_cached: gpu.pipeline("soft-cache", &shaders::softmax_cached(), "softmax_cached", &soft_pl)?,
            merge: gpu.pipeline("merge", &shaders::merge_heads(), "merge", &merge_pl)?,
            add: gpu.pipeline("add", &shaders::add(), "add", &add_pl)?,
            relu: gpu.pipeline("relu", &shaders::relu(), "relu", &relu_pl)?,
            arg: gpu.pipeline("arg", &shaders::argmax(), "argmax", &arg_pl)?,
            fbank: gpu.pipeline("fbank", &shaders::fbank(), "fbank", &fbank_pl)?,
            lfr: gpu.pipeline("lfr", &shaders::lfr_features(), "lfr_features", &lfr_pl)?,
            ln_l,
            gemm_l,
            gemm_b_l,
            fsmn_l,
            split_l,
            soft_l,
            merge_l,
            add_l,
            relu_l,
            arg_l,
            fbank_l,
            lfr_l,
        })
    }

    fn get(&self, id: Pid) -> &wgpu::ComputePipeline {
        match id {
            // One WGSL source, one layout: the three Ln ids differ only in
            // which buffer they bind, never in behaviour.  The parallel kernel
            // is the default; `SVS_LN_PAR=0` rolls back.
            Pid::Ln | Pid::LnF32F16 | Pid::LnF32 => &self.ln,
            Pid::Gemm => &self.gemm,
            Pid::GemmNt => &self.gemm_nt,
            Pid::GemmB => &self.gemm_b,
            Pid::GemmF32c => &self.gemm_f32c,
            Pid::GemmTall => &self.gemm_tall,
            Pid::GemmNtTall => &self.gemm_nt_tall,
            Pid::GemmBTall => &self.gemm_b_tall,
            Pid::GemmF32cTall => &self.gemm_f32c_tall,
            Pid::GemmNarrow => &self.gemm_narrow,
            Pid::GemmNtNarrow => &self.gemm_nt_narrow,
            Pid::GemmBNarrow => &self.gemm_b_narrow,
            Pid::GemmF32cNarrow => &self.gemm_f32c_narrow,
            Pid::Fsmn => &self.fsmn,
            Pid::Split => &self.split,
            Pid::Soft => &self.soft,
            Pid::SoftCached => &self.soft_cached,
            Pid::Merge => &self.merge,
            Pid::Add => &self.add,
            Pid::Relu => &self.relu,
            Pid::Arg => &self.arg,
        }
    }
}

fn bgl(device: &wgpu::Device, label: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

fn ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    buf_entry(binding, true, false)
}
fn rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    buf_entry(binding, false, false)
}
fn uni(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: NonZeroU64::new(256),
        },
        count: None,
    }
}

/// A uniform bound whole (no dynamic offset): the frontend kernels have one
/// config each and no slot table.
fn uni_s(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn buf_entry(binding: u32, read_only: bool, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

fn upload_layer(gpu: &Gpu, layer: &crate::weights::Layer) -> Result<LayerGpu> {
    Ok(LayerGpu {
        n1w: up_f32(gpu, &layer.n1_w)?,
        n1b: up_f32(gpu, &layer.n1_b)?,
        qkv: up_mat(gpu, &layer.qkv.w, layer.qkv.rows, layer.qkv.cols, layer.qkv.rows)?,
        qkv_b: up_bias(gpu, &layer.qkv.b, layer.qkv.rows)?,
        fsmn: up_f32(gpu, &layer.fsmn)?,
        outw: up_mat(gpu, &layer.out.w, layer.out.rows, layer.out.cols, layer.out.rows)?,
        outb: up_bias(gpu, &layer.out.b, layer.out.rows)?,
        n2w: up_f32(gpu, &layer.n2_w)?,
        n2b: up_f32(gpu, &layer.n2_b)?,
        w1: up_mat(gpu, &layer.w1.w, layer.w1.rows, layer.w1.cols, layer.w1.rows)?,
        w1b: up_bias(gpu, &layer.w1.b, layer.w1.rows)?,
        w2: up_mat(gpu, &layer.w2.w, layer.w2.rows, layer.w2.cols, layer.w2.rows)?,
        w2b: up_bias(gpu, &layer.w2.b, layer.w2.rows)?,
        in_dim: layer.in_dim as u32,
    })
}

fn up_bias(gpu: &Gpu, data: &[f32], cols_pad: usize) -> Result<wgpu::Buffer> {
    let mut v = vec![0f32; cols_pad];
    v[..data.len()].copy_from_slice(data);
    up_f32(gpu, &v)
}

fn up_mat(gpu: &Gpu, data: &[f32], rows: usize, cols: usize, rows_pad: usize) -> Result<wgpu::Buffer> {
    let mut v = vec![0f32; rows_pad * cols];
    for r in 0..rows {
        let at = r * cols;
        v[at..at + cols].copy_from_slice(&data[at..at + cols]);
    }
    up_f32(gpu, &v)
}

fn up_f32(gpu: &Gpu, data: &[f32]) -> Result<wgpu::Buffer> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let buf = gpu.storage("f32", bytes.len() as u64);
    gpu.upload_chunked(&buf, &bytes)?;
    Ok(buf)
}
