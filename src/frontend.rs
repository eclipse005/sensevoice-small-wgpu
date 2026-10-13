//! Kaldi fbank (torchaudio.compliance.kaldi, hamming, dither 0) + LFR + CMVN.
//!
//! The waveform is scaled by `1 << 15` before the filterbank, matching
//! `WavFrontend`. CMVN is `x += shift; x *= scale` from `am.mvn` (`<AddShift>`
//! then `<Rescale>`), not `(x - mean) / std`.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, Fft, FftPlanner};

use crate::weights::EMBED_DIM;

pub const SAMPLE_RATE: u32 = 16_000;
pub const MEL_BINS: usize = 80;
pub const LFR_M: usize = 7;
pub const LFR_N: usize = 6;
pub const WINDOW: usize = 400;
pub const SHIFT: usize = 160;
pub const FFT: usize = 512;
pub const EPS: f32 = f32::EPSILON;

pub struct Cmvn {
    pub shift: Vec<f32>,
    pub scale: Vec<f32>,
}

pub struct Frontend {
    cmvn: Cmvn,
    window: Vec<f32>,
    /// `[MEL_BINS, FFT/2 + 1]`, last column is the padded zero.
    mel: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

#[derive(Debug, Clone)]
pub struct Prepared {
    /// `[t, 560]`, queries prepended, scaled by `sqrt(512)`, sinusoidal PE added.
    pub x: Vec<f32>,
    pub t: usize,
    pub n_lfr: usize,
    /// Sum of LFR-CMVN features, before the four query rows.
    pub feat_sum: f32,
}

impl Frontend {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let cmvn = load_cmvn(&model_dir.join("am.mvn"))?;
        let mut planner = FftPlanner::<f32>::new();
        Ok(Self {
            cmvn,
            window: hamming(WINDOW),
            mel: mel_banks(),
            fft: planner.plan_fft_forward(FFT),
        })
    }

    pub fn prepare(&self, samples_unit: &[f32], embed: &[f32], lid: usize, itn: usize) -> Result<Prepared> {
        if samples_unit.len() < WINDOW {
            bail!(
                "音频只有 {} 个采样，短于 25 ms 窗（{} 点）",
                samples_unit.len(),
                WINDOW
            );
        }
        let (fbank, n_fbank) = self.fbank(samples_unit);
        joint("fbank", &fbank);
        let lfr = apply_lfr(&fbank, n_fbank);
        joint("lfr", &lfr);
        let n_lfr = lfr.len() / EMBED_DIM;
        let mut feats = lfr;
        apply_cmvn(&mut feats, &self.cmvn);
        joint("cmvn", &feats);
        let feat_sum = feats.iter().sum();
        let t = n_lfr + 4;
        let mut x = vec![0f32; t * EMBED_DIM];
        for (i, id) in [lid, 1usize, 2, itn].into_iter().enumerate() {
            let row = embed
                .get(id * EMBED_DIM..(id + 1) * EMBED_DIM)
                .context("embed 行越界")?;
            x[i * EMBED_DIM..(i + 1) * EMBED_DIM].copy_from_slice(row);
        }
        x[4 * EMBED_DIM..].copy_from_slice(&feats);
        let scale = 512f32.sqrt();
        for v in &mut x {
            *v *= scale;
        }
        add_posenc(&mut x, t, EMBED_DIM);
        Ok(Prepared {
            x,
            t,
            n_lfr,
            feat_sum,
        })
    }

    fn fbank(&self, samples_unit: &[f32]) -> (Vec<f32>, usize) {
        let n = samples_unit.len();
        let frames = 1 + (n - WINDOW) / SHIFT;
        let mut wave = vec![0f32; n];
        for (i, s) in samples_unit.iter().enumerate() {
            wave[i] = s * 32768.0;
        }
        let mut out = vec![0f32; frames * MEL_BINS];
        // Frames are independent; each rayon task keeps its own FFT scratch
        // and writes a contiguous run of output rows, so the per-frame
        // arithmetic (and therefore every bit) is the sequential one.
        const FRAMES_PER_TASK: usize = 16;
        out.par_chunks_mut(MEL_BINS * FRAMES_PER_TASK)
            .enumerate()
            .for_each(|(task, chunk)| {
                let mut spec = vec![Complex::new(0.0, 0.0); FFT];
                let mut power = vec![0f32; FFT / 2 + 1];
                for (i, orow) in chunk.chunks_mut(MEL_BINS).enumerate() {
                    let f = task * FRAMES_PER_TASK + i;
                    spec.fill(Complex::new(0.0, 0.0));
                    let start = f * SHIFT;
                    let mut mean = 0.0f32;
                    for i in 0..WINDOW {
                        mean += wave[start + i];
                    }
                    mean /= WINDOW as f32;
                    let mut prev = wave[start] - mean;
                    spec[0].re = prev - 0.97 * prev;
                    for i in 1..WINDOW {
                        let cur = wave[start + i] - mean;
                        spec[i].re = (cur - 0.97 * prev) * self.window[i];
                        prev = cur;
                    }
                    spec[0].re *= self.window[0];
                    self.fft.process(&mut spec);
                    for (k, p) in power.iter_mut().enumerate() {
                        let c = spec[k];
                        *p = c.re * c.re + c.im * c.im;
                    }
                    for (b, dst) in orow.iter_mut().enumerate() {
                        let row = &self.mel[b * (FFT / 2 + 1)..(b + 1) * (FFT / 2 + 1)];
                        let mut acc = 0.0f32;
                        for k in 0..power.len() {
                            acc += power[k] * row[k];
                        }
                        *dst = acc.max(EPS).ln();
                    }
                }
            });
        (out, frames)
    }
}

fn hamming(n: usize) -> Vec<f32> {
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.54f32 - 0.46f32 * (2.0 * std::f32::consts::PI * i as f32 / denom).cos())
        .collect()
}

fn mel_scale(freq: f64) -> f64 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

fn mel_banks() -> Vec<f32> {
    let n_bins = FFT / 2;
    let cols = n_bins + 1;
    let low = 20.0f64;
    let high = 8000.0f64;
    let mel_low = mel_scale(low);
    let mel_high = mel_scale(high);
    let delta = ((mel_high - mel_low) / (MEL_BINS as f64 + 1.0)) as f32;
    let mel_low_f = mel_low as f32;
    let bin_width = SAMPLE_RATE as f32 / FFT as f32;
    let mut mel = vec![0f32; MEL_BINS * cols];
    for b in 0..MEL_BINS {
        let left = mel_low_f + b as f32 * delta;
        let center = mel_low_f + (b as f32 + 1.0) * delta;
        let right = mel_low_f + (b as f32 + 2.0) * delta;
        for k in 0..n_bins {
            let freq = bin_width * k as f32;
            let m = 1127.0f32 * (1.0f32 + freq / 700.0f32).ln();
            let up = (m - left) / (center - left);
            let down = (right - m) / (right - center);
            mel[b * cols + k] = 0.0f32.max(up.min(down));
        }
    }
    mel
}

pub fn apply_lfr(frames: &[f32], t0: usize) -> Vec<f32> {
    assert_eq!(frames.len(), t0 * MEL_BINS);
    let t_lfr = (t0 + LFR_N - 1) / LFR_N;
    let left = (LFR_M - 1) / 2;
    let mut padded = Vec::with_capacity((t0 + left + 8) * MEL_BINS);
    for _ in 0..left {
        padded.extend_from_slice(&frames[..MEL_BINS]);
    }
    padded.extend_from_slice(frames);
    let t = t0 + left;
    let last_idx = (t as isize - LFR_M as isize).div_euclid(LFR_N as isize) + 1;
    let num_padding = LFR_M as isize - (t as isize - last_idx * LFR_N as isize);
    if num_padding > 0 {
        let num = (2.0 * LFR_M as f64 - 2.0 * t as f64
            + (t_lfr as f64 - 1.0 + last_idx as f64) * LFR_N as f64)
            / 2.0
            * (t_lfr as f64 - last_idx as f64);
        let n = num.trunc() as isize;
        if n > 0 {
            let last = padded[padded.len() - MEL_BINS..].to_vec();
            for _ in 0..n {
                padded.extend_from_slice(&last);
            }
        }
    }
    let mut out = vec![0f32; t_lfr * LFR_M * MEL_BINS];
    // LFR rows are independent copies out of the padded frame matrix.
    let need = LFR_M * MEL_BINS;
    out.par_chunks_mut(need).enumerate().for_each(|(ti, dst)| {
        let start = ti * LFR_N * MEL_BINS;
        dst.copy_from_slice(&padded[start..start + need]);
    });
    out
}

fn apply_cmvn(x: &mut [f32], cmvn: &Cmvn) {
    let d = EMBED_DIM;
    x.par_chunks_mut(d).for_each(|row| {
        for (v, (sh, sc)) in row.iter_mut().zip(cmvn.shift.iter().zip(cmvn.scale.iter())) {
            *v = (*v + sh) * sc;
        }
    });
}

pub fn add_posenc(x: &mut [f32], t: usize, depth: usize) {
    let half = depth / 2;
    let log_inc = 10000f32.ln() / (half as f32 - 1.0);
    let mut inv = vec![0f32; half];
    for i in 0..half {
        inv[i] = (-log_inc * i as f32).exp();
    }
    for pos in 1..=t {
        let row = &mut x[(pos - 1) * depth..pos * depth];
        for i in 0..half {
            let ang = pos as f32 * inv[i];
            row[i] += ang.sin();
            row[half + i] += ang.cos();
        }
    }
}

fn joint(name: &str, x: &[f32]) {
    if !crate::tuning::trace() {
        return;
    }
    // Same gate format as `encoder_cpu::trace_on`: full-precision scalars plus
    // an FNV-1a hash of the raw bits. `{:.6}` hides one-ULP drift, and a
    // sum/max pair alone can agree on two buffers that differ — the hash is
    // the actual byte-exact check.
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in x {
        sum += f64::from(*v);
        max = max.max(v.abs());
        h ^= u64::from(v.to_bits());
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    eprintln!("trace {name} sum={sum:.9} max={max:.9} n={} h={h:016x}", x.len());
}

pub fn load_cmvn(path: &Path) -> Result<Cmvn> {
    let text = std::fs::read_to_string(path).with_context(|| format!("读取 {}", path.display()))?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.split_whitespace().next().is_none()).collect();
    let mut shift = None;
    let mut scale = None;
    for (i, line) in lines.iter().enumerate() {
        let head = line.split_whitespace().next().unwrap_or("");
        if (head == "<AddShift>" || head == "<Rescale>") && i + 1 < lines.len() {
            let nxt: Vec<&str> = lines[i + 1].split_whitespace().collect();
            if nxt.first().copied() == Some("<LearnRateCoef>") && nxt.len() > 4 {
                let vals: Vec<f32> = nxt[3..nxt.len() - 1]
                    .iter()
                    .map(|s| {
                        s.parse::<f64>()
                            .map(|v| v as f32)
                            .with_context(|| format!("CMVN 数字 {s}"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if head == "<AddShift>" {
                    shift = Some(vals);
                } else {
                    scale = Some(vals);
                }
            }
        }
    }
    let shift = shift.context("am.mvn 缺少 <AddShift>")?;
    let scale = scale.context("am.mvn 缺少 <Rescale>")?;
    if shift.len() != EMBED_DIM || scale.len() != EMBED_DIM {
        bail!("CMVN 维度 {} / {}，期望 {EMBED_DIM}", shift.len(), scale.len());
    }
    Ok(Cmvn { shift, scale })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_15s() {
        let n = 240_000;
        assert_eq!(1 + (n - WINDOW) / SHIFT, 1498);
    }

    #[test]
    fn hamming_edges() {
        let w = hamming(400);
        assert!((w[0] - 0.08).abs() < 1e-6);
        assert!((w[399] - 0.08).abs() < 1e-6);
    }

    #[test]
    fn feat_sum_matches_15s_gold() {
        let dir = Path::new(r"D:\SenseVoiceSmall\model\SenseVoiceSmall");
        let wav = Path::new(r"D:\SenseVoiceSmall\fixtures\15s_en.wav");
        let front = Frontend::load(dir).unwrap();
        let samples = crate::audio::load_audio(wav).unwrap();
        assert_eq!(samples.len(), 240_000);
        let embed = vec![0f32; 16 * EMBED_DIM];
        let prep = front.prepare(&samples, &embed, 0, 14).unwrap();
        assert_eq!(prep.n_lfr, 250);
        assert_eq!(prep.t, 254);
        let gold = 3235.64599609375f32;
        let d = (prep.feat_sum - gold).abs();
        // Sinc resample accumulates in f32, so the sum is a few ulps off torch.
        assert!(d < 5e-2, "feat_sum {} gold {gold} delta {d}", prep.feat_sum);
    }
}
