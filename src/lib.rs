//! SenseVoiceSmall on Rust CPU and wgpu.
//!
//! The alignment target is the official inference graph with **dither = 0**.
//! FunASR's WavFrontend defaults dither to 1.0, which is random and cannot be
//! bit-matched. VAD is not embedded: pass segment times, or transcribe the
//! whole clip.

mod audio;
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
mod vad;
mod weights;

use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Result};

use crate::ctc::EMO_UNK;
use crate::encoder_cpu::CpuEngine;
use crate::encoder_gpu::GpuEngine;
use crate::frontend::{Frontend, WINDOW};
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

impl Model {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self> {
        let sel = DeviceSelector::parse(device)?;
        let weights = Weights::load(model_dir)?;
        let embed = weights.embed.clone();
        let front = Frontend::load(model_dir)?;
        let tok = Tokenizer::load(model_dir)?;
        let (engine, device) = if sel == DeviceSelector::Cpu {
            (Engine::Cpu(CpuEngine::new(weights)), "cpu".to_string())
        } else {
            let gpu = pollster::block_on(Gpu::new_with(sel))?;
            let device = gpu.describe();
            eprintln!("sensevoice: {device}");
            let engine = GpuEngine::new(gpu, &weights)?;
            drop(weights);
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
        for (start, end, audio) in jobs {
            if audio.len() < WINDOW {
                continue;
            }
            let t0 = Instant::now();
            let prep = self.front.prepare(audio, &self.embed, lid, itn)?;
            frontend_ms += elapsed_ms(t0);
            feat_sum += prep.feat_sum;
            n_lfr += prep.n_lfr;
            n_input += prep.t;
            let t1 = Instant::now();
            let ids = match &mut self.engine {
                Engine::Cpu(cpu) => cpu.encode(&prep.x, prep.t, ban),
                Engine::Gpu(gpu) => gpu.encode(&prep.x, prep.t, ban)?,
            };
            encode_ms += elapsed_ms(t1);
            let raw = self.tok.decode(&ids);
            raws.push(raw.clone());
            segments.push(SegmentOut {
                start_ms: start,
                end_ms: end,
                raw,
                token_ids: ids,
            });
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
