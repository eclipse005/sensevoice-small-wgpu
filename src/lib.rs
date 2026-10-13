//! SenseVoiceSmall on Rust CPU and wgpu.
//!
//! The alignment target is the official inference graph with **dither = 0**.
//! FunASR's WavFrontend defaults dither to 1.0, which is random and cannot be
//! bit-matched. VAD is not embedded: pass segment times, or transcribe the
//! whole clip.

mod audio;
mod cpu_affinity;
mod ctc;
mod encoder_cpu;
mod encoder_gpu;
mod frontend;
mod gemm_f32;
mod gpu;
mod pt;
mod postprocess;
mod shaders;
mod tokenizer;
mod tuning;
mod vad;
mod weights;

use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Result};

use crate::ctc::{greedy, EMO_UNK};
use crate::encoder_cpu::CpuEngine;
use crate::encoder_gpu::GpuEngine;
use crate::frontend::{Frontend, LFR_N, SHIFT, WINDOW};
use crate::gpu::{DeviceSelector, Gpu};
use crate::postprocess::{read_tags, rich_post};
use crate::tokenizer::Tokenizer;
use crate::vad::{merge_vad, slice_samples, Segment};
use crate::weights::Weights;

pub use crate::gpu::list_devices as list_gpu_devices;
pub use crate::vad::Segment as VadSegment;

pub fn language_id(name: &str) -> Result<usize> {
    match name.to_ascii_lowercase().as_str() {
        "auto" => Ok(0),
        "zh" => Ok(3),
        "en" => Ok(4),
        "yue" => Ok(7),
        "ja" => Ok(11),
        "ko" => Ok(12),
        "nospeech" => Ok(13),
        other => bail!("未知语言 {other}，可选 auto|zh|en|yue|ja|ko|nospeech"),
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub language: String,
    pub use_itn: bool,
    pub ban_emo_unk: bool,
    /// Empty means the whole clip. Times are milliseconds.
    pub segments: Vec<Segment>,
    /// `<= 0` keeps the segments as given. A positive value runs FunASR `merge_vad`
    /// (the cursor starts at 0, so leading silence is included).
    pub merge_ms: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            language: "auto".into(),
            use_itn: true,
            ban_emo_unk: false,
            segments: Vec::new(),
            merge_ms: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SegmentOut {
    pub start_ms: f64,
    pub end_ms: f64,
    pub raw: String,
    pub token_ids: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct Timing {
    pub frontend_ms: f64,
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub audio_s: f64,
    pub rtfx: f64,
}

#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub raw: String,
    pub language: String,
    pub emotion: String,
    pub event: String,
    pub itn: String,
    pub segments: Vec<SegmentOut>,
    pub timing: Timing,
    pub feat_sum: f32,
    pub n_lfr: usize,
    pub n_input: usize,
}

enum Engine {
    Cpu(CpuEngine),
    Gpu(GpuEngine),
}

pub struct Model {
    front: Frontend,
    embed: Vec<f32>,
    tok: Tokenizer,
    engine: Engine,
    device: String,
}


/// Temporary memory probe (`SVS_MEM_TRACE=1`): peak working set at load phases.
fn mem_probe(tag: &str) {
    if std::env::var("SVS_MEM_TRACE").ok().as_deref() != Some("1") {
        return;
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct Pmc {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
            fn K32GetProcessMemoryInfo(
                process: *mut core::ffi::c_void,
                counters: *mut Pmc,
                cb: u32,
            ) -> i32;
        }
        let mut pmc = Pmc { cb: std::mem::size_of::<Pmc>() as u32, ..Default::default() };
        unsafe {
            if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
                eprintln!(
                    "mem_probe {tag}: peak_ws {:.1} MiB  cur {:.1} MiB",
                    pmc.peak_working_set_size as f64 / 1048576.0,
                    pmc.working_set_size as f64 / 1048576.0
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = tag;
}

impl Model {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self> {
        let sel = DeviceSelector::parse(device)?;
        let mut weights = Weights::load(model_dir)?;
        mem_probe("after Weights::load");
        let embed = weights.embed.clone();
        let front = Frontend::load(model_dir)?;
        let tok = Tokenizer::load(model_dir)?;
        mem_probe("after Frontend+Tokenizer");
        let (engine, device) = if sel == DeviceSelector::Cpu {
            (Engine::Cpu(CpuEngine::new(weights)), "cpu".to_string())
        } else {
            let gpu = pollster::block_on(Gpu::new_with(sel))?;
            let device = gpu.describe();
            eprintln!("sensevoice: {device}");
            let engine = GpuEngine::new(gpu, &mut weights, &front)?;
            mem_probe("after GpuEngine::new");
            (Engine::Gpu(engine), device)
        };
        Ok(Self {
            front,
            embed,
            tok,
            engine,
            device,
        })
    }

    /// Size the GPU frame buffers for a clip of `n_samples` samples. The CLI
    /// loads the audio before the model, so it can hand the length in and keep
    /// both the reallocation and the oversized preallocation out of the run.
    pub fn reserve_samples(&mut self, n_samples: usize) -> Result<()> {
        if let Engine::Gpu(gpu) = &mut self.engine {
            if n_samples >= WINDOW {
                let n_fbank = 1 + (n_samples - WINDOW) / SHIFT;
                let lfr_rows = (n_fbank + LFR_N - 1) / LFR_N;
                gpu.reserve_frames(lfr_rows + 4)?;
            }
        }
        Ok(())
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// `samples` are mono, 16 kHz, roughly in [-1, 1]. The clock starts here,
    /// after weights are resident and pipelines are compiled.
    pub fn transcribe(&mut self, samples: &[f32], opt: &Options) -> Result<Transcript> {
        let wall = Instant::now();
        let audio_s = samples.len() as f64 / f64::from(frontend::SAMPLE_RATE);
        let lid = language_id(&opt.language)?;
        let itn = if opt.use_itn { 14 } else { 15 };
        let ban = opt.ban_emo_unk.then_some(EMO_UNK);
        let merged;
        let jobs: Vec<(f64, f64, &[f32])> = if opt.segments.is_empty() {
            vec![(0.0, audio_s * 1000.0, samples)]
        } else if opt.merge_ms > 0.0 {
            merged = merge_vad(&opt.segments, opt.merge_ms, 0.0);
            merged
                .iter()
                .map(|s| (s.start_ms, s.end_ms, slice_samples(samples, *s)))
                .collect()
        } else {
            opt.segments
                .iter()
                .map(|s| (s.start_ms, s.end_ms, slice_samples(samples, *s)))
                .collect()
        };

        let mut frontend_ms = 0.0;
        let mut encode_ms = 0.0;
        let mut feat_sum = 0.0f32;
        let mut n_lfr = 0usize;
        let mut n_input = 0usize;
        let mut segments = Vec::new();
        let mut raws = Vec::new();
        // Multi-segment GPU runs concatenate the segments into one encoder pass.
        // A short segment's every GEMM grid is one or two m blocks, so 50 short
        // VAD cuts leave the card idle; one long sequence fills it. The softmax
        // and FSMN masks read the per-row segment ids so the utterances never
        // see each other. `SVS_SEG_CONCAT=0` restores the per-segment loop.
        let mut concat_done = false;
        if jobs.len() > 1 && tuning::seg_concat() {
            if let Engine::Gpu(gpu) = &mut self.engine {
                // (job index, first row, frame count, first row-sums slot)
                let mut layout: Vec<(usize, usize, usize, usize)> = Vec::new();
                let mut total_rows = 0usize;
                let mut total_lfr = 0usize;
                for (i, (_, _, audio)) in jobs.iter().enumerate() {
                    if audio.len() < WINDOW {
                        continue;
                    }
                    let n_fbank = 1 + (audio.len() - WINDOW) / SHIFT;
                    let lfr_rows = (n_fbank + LFR_N - 1) / LFR_N;
                    layout.push((i, total_rows, lfr_rows, total_lfr));
                    total_rows += lfr_rows + 4;
                    total_lfr += lfr_rows;
                }
                // Concatenation trades a fixed per-segment overhead (a short
                // segment's GEMM grids are one or two m blocks, so each cut
                // pays ~20 ms of underutilised launch) for an O(T^2) sweep
                // over the masked-out attention pairs (~50 ns each, measured).
                // Many short VAD cuts win big; a handful of 15 s segments lose.
                let waste: u64 = (total_rows * total_rows) as u64
                    - layout.iter().map(|l| ((l.2 + 4) * (l.2 + 4)) as u64).sum::<u64>();
                let worth_it = (layout.len() as u64) * 20_000_000 > 50 * waste;
                if layout.len() < 2 || !worth_it {
                    // Prefer the per-segment loop: one usable segment, or the
                    // attention's quadratic term costs more than the launches.
                } else {
                    gpu.reserve_frames(total_rows)?;
                    let mut ids: Vec<u32> = Vec::with_capacity(total_rows);
                    for si in 0..layout.len() {
                        for _ in 0..layout[si].2 + 4 {
                            ids.push(si as u32);
                        }
                    }
                    for &(i, row_off, lfr_rows, sum_off) in &layout {
                        let (_, _, audio) = jobs[i];
                        let n_fbank = 1 + (audio.len() - WINDOW) / SHIFT;
                        let t = lfr_rows + 4;
                        let t0 = Instant::now();
                        gpu.run_frontend_at(
                            audio, n_fbank, t, lid as u32, itn as u32, row_off, sum_off,
                        )?;
                        frontend_ms += elapsed_ms(t0);
                        n_lfr += lfr_rows;
                        n_input += t;
                    }
                    let t1 = Instant::now();
                    let (frames, fs) = gpu.encode_frames(total_rows, total_lfr, ban, Some(&ids))?;
                    encode_ms += elapsed_ms(t1);
                    feat_sum += fs;
                    for &(i, row_off, lfr_rows, _) in &layout {
                        let toks = greedy(&frames[row_off..row_off + lfr_rows + 4]);
                        let raw = self.tok.decode(&toks);
                        raws.push(raw.clone());
                        let (start, end, _) = jobs[i];
                        segments.push(SegmentOut {
                            start_ms: start,
                            end_ms: end,
                            raw,
                            token_ids: toks,
                        });
                    }
                    concat_done = true;
                }
            }
        }
        if !concat_done {
        for (start, end, audio) in jobs {
            if audio.len() < WINDOW {
                continue;
            }
            let ids = match &mut self.engine {
                Engine::Cpu(cpu) => {
                    // The CPU frontend stays the alignment oracle for this path.
                    let t0 = Instant::now();
                    let prep = self.front.prepare(audio, &self.embed, lid, itn)?;
                    frontend_ms += elapsed_ms(t0);
                    feat_sum += prep.feat_sum;
                    n_lfr += prep.n_lfr;
                    n_input += prep.t;
                    let t1 = Instant::now();
                    let ids = cpu.encode(&prep.x, prep.t, ban);
                    encode_ms += elapsed_ms(t1);
                    ids
                }
                Engine::Gpu(gpu) => {
                    // Same features, computed on the GPU: the CPU only uploads
                    // the samples and reads back the token ids.
                    let n_fbank = 1 + (audio.len() - WINDOW) / SHIFT;
                    let lfr_rows = (n_fbank + LFR_N - 1) / LFR_N;
                    let t = lfr_rows + 4;
                    let t0 = Instant::now();
                    gpu.run_frontend(audio, n_fbank, t, lid as u32, itn as u32)?;
                    frontend_ms += elapsed_ms(t0);
                    n_lfr += lfr_rows;
                    n_input += t;
                    let t1 = Instant::now();
                    let (ids, fs) = gpu.encode(t, lfr_rows, ban)?;
                    encode_ms += elapsed_ms(t1);
                    feat_sum += fs;
                    ids
                }
            };
            let raw = self.tok.decode(&ids);
            raws.push(raw.clone());
            segments.push(SegmentOut {
                start_ms: start,
                end_ms: end,
                raw,
                token_ids: ids,
            });
        }
        }
        let t2 = Instant::now();
        let raw = raws.join(" ");
        let text = rich_post(&raw);
        let tags = read_tags(&raw);
        let decode_ms = elapsed_ms(t2);
        let total = wall.elapsed().as_secs_f64().max(1e-9);
        Ok(Transcript {
            text,
            raw,
            language: tags.language,
            emotion: tags.emotion,
            event: tags.event,
            itn: tags.itn,
            segments,
            timing: Timing {
                frontend_ms,
                encode_ms,
                decode_ms,
                audio_s,
                rtfx: audio_s / total,
            },
            feat_sum,
            n_lfr,
            n_input,
        })
    }
}

fn elapsed_ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

pub fn load_audio(path: &Path) -> Result<Vec<f32>> {
    audio::load_audio(path)
}

pub fn list_devices() -> Result<Vec<String>> {
    crate::gpu::list_devices()
}
