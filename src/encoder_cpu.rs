//! f32 encoder. This is the alignment oracle against the Python reference.
//!
//! Every optimisation here is numerically exact: work is split across rows or
//! time steps only, so no reduction ever reassociates. A row's softmax, a
//! time step's FSMN tap sum, a row's LayerNorm mean/variance and a row's argmax
//! are all computed in the original order by the original code, just on
//! different rayon workers. Token ids therefore stay bit-identical to the
//! unoptimised version.

use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use crate::ctc::{argmax_rows, greedy, VOCAB};
use crate::weights::{Layer, Weights, HIDDEN, QKV};

const EPS: f32 = 1e-5;

fn q_scale() -> f32 {
    (128f32).sqrt().recip()
}

/// Per-phase wall-clock accumulators, enabled by `SVS_CPU_PROF=1`.
/// Off by default: `Prof::add` is one predictable branch on a `bool`.
#[derive(Default, Clone, Copy)]
pub struct Prof {
    pub on: bool,
    pub ln: f64,
    pub qkv: f64,
    pub fsmn: f64,
    pub scores: f64,
    pub softmax: f64,
    pub ctx: f64,
    pub out: f64,
    pub ln2: f64,
    pub w1: f64,
    pub relu: f64,
    pub w2: f64,
    pub bias: f64,
    pub tail: f64,
}

impl Prof {
    pub fn dump(&self, label: &str) {
        if !self.on {
            return;
        }
        let total = self.ln + self.qkv + self.fsmn + self.scores + self.softmax + self.ctx
            + self.out + self.ln2 + self.w1 + self.relu + self.w2 + self.bias + self.tail;
        eprintln!("cpu prof [{label}] total {total:.1} ms");
        let mut rows = [
            ("ln1", self.ln), ("qkv", self.qkv), ("fsmn", self.fsmn),
            ("scores_gemm", self.scores), ("softmax", self.softmax), ("ctx_gemm", self.ctx),
            ("out_gemm", self.out), ("ln2", self.ln2), ("w1_gemm", self.w1),
            ("relu", self.relu), ("w2_gemm", self.w2), ("bias_init", self.bias), ("other", self.tail),
        ];
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (name, ms) in rows {
            let pct = if total > 0.0 { ms / total * 100.0 } else { 0.0 };
            eprintln!("  {name:<12} {ms:9.2} ms  {pct:5.1}%");
        }
    }
}

/// Time `$body` into `$prof.$field` when profiling is on; otherwise just run it.
macro_rules! tprof {
    ($prof:expr, $field:ident, $body:expr) => {{
        if $prof.on {
            let t0 = std::time::Instant::now();
            let r = $body;
            $prof.$field += t0.elapsed().as_secs_f64() * 1e3;
            r
        } else {
            $body
        }
    }};
}

/// Per-utterance scratch, sized once and reused by every layer.
///
/// The attention score tile is `4 * t * t` floats — 144 MB at t = 3000 — so
/// re-allocating and re-zeroing it 70 times per utterance was pure overhead.
/// Every writer below fully overwrites what it reads, so reuse never carries
/// stale values into a result.
struct Scratch {
    norm: Vec<f32>,
    norm2: Vec<f32>,
    qkv: Vec<f32>,
    scores: Vec<f32>,
    ctx: Vec<f32>,
    mem: Vec<f32>,
    y: Vec<f32>,
    mid: Vec<f32>,
    ff: Vec<f32>,
}

impl Scratch {
    fn new(t: usize, d_in: usize, ffn: usize) -> Self {
        Self {
            norm: vec![0f32; t * d_in],
            norm2: vec![0f32; t * HIDDEN],
            qkv: vec![0f32; t * QKV],
            scores: vec![0f32; 4 * t * t],
            ctx: vec![0f32; t * HIDDEN],
            mem: vec![0f32; t * HIDDEN],
            y: vec![0f32; t * HIDDEN],
            mid: vec![0f32; t * ffn],
            ff: vec![0f32; t * HIDDEN],
        }
    }
}

pub struct CpuEngine {
    weights: Weights,
}

impl CpuEngine {
    pub fn new(weights: Weights) -> Self {
        freeze_gemm_tiling();
        Self { weights }
    }

    pub fn embed(&self) -> &[f32] {
        &self.weights.embed
    }

    /// Greedy CTC token ids for one utterance. `x` is `[t, 560]` already scaled and position-encoded.
    pub fn encode(&self, x: &[f32], t: usize, ban: Option<usize>) -> Vec<i32> {
        let mut prof = Prof {
            on: std::env::var("SVS_CPU_PROF").ok().as_deref() == Some("1"),
            ..Prof::default()
        };
        let ids = self.encode_inner(x, t, ban, &mut prof);
        if prof.on {
            prof.dump(&format!("t={t}"));
        }
        ids
    }

    fn encode_inner(&self, x: &[f32], t: usize, ban: Option<usize>, prof: &mut Prof) -> Vec<i32> {
        trace("input", x);
        let mut scratch = Scratch::new(
            t,
            self.weights.layers[0].in_dim,
            self.weights.layers[0].w1.rows,
        );
        // Ping-pong activation buffers, reused instead of re-allocated per layer.
        // Layer 0 reads the 560-d input and writes 512-d, so it lands in buf_a.
        let mut buf_a = vec![0f32; t * HIDDEN];
        let mut buf_b = vec![0f32; t * HIDDEN];
        self.layer(&self.weights.layers[0], x, &mut buf_a, t, false, &mut scratch, prof);
        trace("L0", &buf_a);
        let mut src_is_a = true;
        for (i, layer) in self.weights.layers.iter().enumerate().skip(1) {
            if src_is_a {
                self.layer(layer, &buf_a, &mut buf_b, t, true, &mut scratch, prof);
            } else {
                self.layer(layer, &buf_b, &mut buf_a, t, true, &mut scratch, prof);
            }
            src_is_a = !src_is_a;
            if std::env::var("SVS_TRACE").ok().as_deref() == Some("1") {
                let cur = if src_is_a { &buf_a } else { &buf_b };
                trace(&format!("L{i}"), cur);
            }
        }
        // Land the live activations in buf_a so there is one path for the rest.
        if !src_is_a {
            std::mem::swap(&mut buf_a, &mut buf_b);
        }
        tprof!(prof, ln, layer_norm_inplace(&mut buf_a, &self.weights.after_w, &self.weights.after_b, HIDDEN));
        trace("after_norm", &buf_a);
        let mut cur = std::mem::take(&mut buf_a);
        let mut nxt = vec![0f32; t * HIDDEN];
        for (i, layer) in self.weights.tp.iter().enumerate() {
            self.layer(layer, &cur, &mut nxt, t, true, &mut scratch, prof);
            std::mem::swap(&mut cur, &mut nxt);
            if std::env::var("SVS_TRACE").ok().as_deref() == Some("1") {
                trace(&format!("TP{i}"), &cur);
            }
        }
        tprof!(prof, ln2, layer_norm_inplace(&mut cur, &self.weights.tp_w, &self.weights.tp_b, HIDDEN));
        trace("tp_norm", &cur);
        let mut logits = vec![0f32; t * VOCAB];
        tprof!(prof, w2, linear_into(&mut logits, &cur, &self.weights.ctc.w, &self.weights.ctc.b, t, HIDDEN, VOCAB));
        trace("logits", &logits);
        let frames = tprof!(prof, tail, argmax_rows(&logits, t, VOCAB, ban));
        greedy(&frames)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        layer: &Layer,
        x: &[f32],
        dst: &mut [f32],
        t: usize,
        residual: bool,
        mut s: &mut Scratch,
        prof: &mut Prof,
    ) {
        let d_in = layer.in_dim;
        tprof!(prof, ln, layer_norm(&mut s.norm, x, &layer.n1_w, &layer.n1_b, t, d_in));
        trace_f("norm", &s.norm);
        sanm(&mut s, layer, t, prof);
        tprof!(prof, tail, if residual {
            dst.par_iter_mut()
                .zip(x.par_iter())
                .zip(s.y.par_iter())
                .for_each(|((d, a), b)| *d = *a + *b);
        } else {
            dst.copy_from_slice(&s.y);
        });
        tprof!(prof, ln2, layer_norm(&mut s.norm2, dst, &layer.n2_w, &layer.n2_b, t, HIDDEN));
        let rows = layer.w1.rows;
        if bias_fold() {
            tprof!(prof, w1, linear_gemm(&mut s.mid, &s.norm2, &layer.w1.w, t, HIDDEN, rows));
            // bias + relu fused into one pass over `mid` instead of a pre-fill
            // plus a relu sweep. `max(0, acc + b)` is the same float the
            // two-step sequence produced — `gbench bits` confirms k = 512 keeps
            // the whole reduction in one `kc` block.
            tprof!(prof, relu, {
                let b = &layer.w1.b;
                s.mid.par_chunks_mut(rows).for_each(|row| {
                    for (v, bv) in row.iter_mut().zip(b.iter()) {
                        let x = *v + *bv;
                        *v = if x < 0.0 { 0.0 } else { x };
                    }
                });
            });
        } else {
            tprof!(prof, bias, linear_bias(&mut s.mid, &layer.w1.b));
            tprof!(prof, w1, linear_acc(&mut s.mid, &s.norm2, &layer.w1.w, t, HIDDEN, rows));
            tprof!(prof, relu, {
                s.mid.par_chunks_mut(rows).for_each(|row| {
                    for v in row.iter_mut() {
                        *v = if *v < 0.0 { 0.0 } else { *v };
                    }
                });
            });
        }
        trace_f("mid", &s.mid);
        // `w2` is the one GEMM whose bias must stay *inside* the accumulation:
        // its k = 2048 gets sliced into several `kc` blocks, each doing
        // `dst = alpha*dst + beta*acc_block`. Pre-filling the bias and
        // accumulating yields `((b + acc1) + acc2 …)`, while an overwrite GEMM
        // plus a later `+ b` yields `(acc1 + acc2 …) + b` — a reassociation.
        // Verified with `_scratch/gbench bits`: every other encoder shape is
        // bit-identical either way, this one is not (268724/770048 bits differ).
        tprof!(prof, w2, linear_into(&mut s.ff, &s.mid, &layer.w2.w, &layer.w2.b, t, rows, HIDDEN));
        trace_f("ff", &s.ff);
        // `ff` already holds `bias + acc`, so this is a plain residual add.
        tprof!(prof, tail, {
            dst.par_chunks_mut(HIDDEN)
                .zip(s.ff.par_chunks(HIDDEN))
                .for_each(|(d, f)| {
                    for i in 0..HIDDEN {
                        d[i] += f[i];
                    }
                });
        });
    }
}

fn sanm(s: &mut Scratch, layer: &Layer, t: usize, prof: &mut Prof) {
    let d_in = layer.in_dim;
    let b = &layer.qkv.b;
    let scale = q_scale();
    if bias_fold() {
        tprof!(prof, qkv, linear_gemm(&mut s.qkv, &s.norm, &layer.qkv.w, t, d_in, QKV));
        // bias + Q scale fused into one pass. The old order was `qkv = acc + b`
        // over all 1536 cells, then `qkv[..512] *= scale`; fsmn only ever reads
        // the unscaled V block, so folding the scale in here is safe — it is
        // still applied to exactly `acc + b`, and only to the Q block. Rows are
        // walked by QKV: the first t*HIDDEN cells are not the Q block.
        tprof!(prof, bias, {
            s.qkv.par_chunks_mut(QKV).for_each(|row| {
                for c in 0..HIDDEN {
                    row[c] = (row[c] + b[c]) * scale;
                }
                for c in HIDDEN..QKV {
                    row[c] = row[c] + b[c];
                }
            });
        });
    } else {
        tprof!(prof, bias, linear_bias(&mut s.qkv, b));
        tprof!(prof, qkv, linear_acc(&mut s.qkv, &s.norm, &layer.qkv.w, t, d_in, QKV));
        tprof!(prof, bias, {
            s.qkv.par_chunks_mut(QKV).for_each(|row| {
                for c in 0..HIDDEN {
                    row[c] *= scale;
                }
            });
        });
    }
    tprof!(prof, fsmn, fsmn_into(&mut s.mem, &s.qkv, &layer.fsmn, t));
    trace_f("qkv", &s.qkv);
    trace_f("mem", &s.mem);
    let qkv = &s.qkv;
    let scores = &mut s.scores;
    tprof!(prof, scores, for h in 0..4 {
        unsafe {
            gemm_f32(
                t,
                t,
                128,
                scores.as_mut_ptr().add(h * t * t),
                1,
                t as isize,
                false,
                qkv.as_ptr().add(h * 128),
                1,
                QKV as isize,
                qkv.as_ptr().add(HIDDEN + h * 128),
                QKV as isize,
                1,
                0.0,
                1.0,
            );
        }
    });
    trace_f("scores", scores);
    // One softmax per head-row; rows are independent, so the parallel split
    // leaves every row's max/sum accumulation order untouched.
    tprof!(prof, softmax, scores.par_chunks_mut(t).for_each(softmax));
    trace_f("soft", scores);
    let ctx = &mut s.ctx;
    tprof!(prof, ctx, for h in 0..4 {
        unsafe {
            gemm_f32(
                t,
                128,
                t,
                ctx.as_mut_ptr().add(h * 128),
                1,
                HIDDEN as isize,
                false,
                scores.as_ptr().add(h * t * t),
                1,
                t as isize,
                qkv.as_ptr().add(2 * HIDDEN + h * 128),
                1,
                QKV as isize,
                0.0,
                1.0,
            );
        }
    });
    trace_f("ctx", ctx);
    if bias_fold() {
        // `out` bias folded into the `y += mem` sweep that had to run anyway:
        // old order was `y = acc + b` then `y += m`, i.e. `(acc + b) + m`.
        tprof!(prof, out, linear_gemm(&mut s.y, ctx, &layer.out.w, t, HIDDEN, HIDDEN));
        let ob = &layer.out.b;
        tprof!(prof, tail, {
            let mem = &s.mem;
            s.y.par_chunks_mut(HIDDEN).zip(mem.par_chunks(HIDDEN)).for_each(|(o, m)| {
                for i in 0..HIDDEN {
                    o[i] = (o[i] + ob[i]) + m[i];
                }
            });
        });
    } else {
        tprof!(prof, bias, linear_bias(&mut s.y, &layer.out.b));
        tprof!(prof, out, linear_acc(&mut s.y, ctx, &layer.out.w, t, HIDDEN, HIDDEN));
        let y = &mut s.y;
        let mem = &s.mem;
        tprof!(prof, tail, y.par_iter_mut().zip(mem.par_iter()).for_each(|(o, m)| *o += *m));
    }
    trace_f("y", &s.y);
}

/// Depthwise FSMN memory. Each time step is independent, and each output
/// element still sums its 11 taps in the original k order; the loop nest is
/// only swapped so the 11 source rows stream linearly instead of being
/// revisited once per output channel.
fn fsmn_into(out: &mut [f32], qkv: &[f32], weight: &[f32], t: usize) {
    if !fsmn_flat() {
        out.par_chunks_mut(HIDDEN)
            .enumerate()
            .for_each(|(time, dst)| {
                for c in 0..HIDDEN {
                    let mut acc = 0.0f32;
                    for k in 0..11 {
                        let src = time as isize + k as isize - 5;
                        if src >= 0 && (src as usize) < t {
                            let v = qkv[src as usize * QKV + 2 * HIDDEN + c];
                            acc += v * weight[c * 11 + k];
                        }
                    }
                    let v = qkv[time * QKV + 2 * HIDDEN + c];
                    dst[c] = acc + v;
                }
            });
        return;
    }
    // Tap-major copy of the depthwise kernel: `w_t[k * HIDDEN + c]` is the
    // weight `weight[c * 11 + k]`. 5632 floats once per call; without it the
    // inner loop would gather the weights at a 44-byte stride.
    let mut w_t = [0f32; 11 * HIDDEN];
    for (c, wc) in weight.chunks_exact(11).enumerate() {
        for (k, w) in wc.iter().enumerate() {
            w_t[k * HIDDEN + c] = *w;
        }
    }
    // Same numbers, different nest: every channel still adds its taps in
    // ascending k starting from 0.0 and the centre tap last, so each element is
    // the same sum in the same order — only now the 512 channels of a tap are
    // contiguous on both sides and the loop vectorises.
    out.par_chunks_mut(HIDDEN)
        .enumerate()
        .for_each(|(time, dst)| {
            dst.fill(0.0);
            for k in 0..11 {
                let src = time as isize + k as isize - 5;
                if src >= 0 && (src as usize) < t {
                    let row = &qkv[src as usize * QKV + 2 * HIDDEN..src as usize * QKV + 3 * HIDDEN];
                    let wrow = &w_t[k * HIDDEN..(k + 1) * HIDDEN];
                    for c in 0..HIDDEN {
                        dst[c] += row[c] * wrow[c];
                    }
                }
            }
            let centre = time * QKV + 2 * HIDDEN;
            for c in 0..HIDDEN {
                dst[c] += qkv[centre + c];
            }
        });
}

/// Same-binary rollback switch for the FSMN loop-nest swap. Both settings are
/// bit-identical — `_scratch` runs diff the full `SVS_TRACE` output — so this
/// exists only for paired A/B timing.
fn fsmn_flat() -> bool {
    static FLAT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAT.get_or_init(|| std::env::var("SVS_FSMN_FLAT").map_or(true, |v| v != "0"))
}

fn layer_norm(y: &mut [f32], x: &[f32], w: &[f32], b: &[f32], t: usize, d: usize) {
    // Rows are independent; each row's mean and variance keep their order.
    y.par_chunks_mut(d)
        .zip(x.par_chunks(d))
        .for_each(|(dst, src)| {
            let mut mean = 0.0f32;
            for v in src {
                mean += *v;
            }
            mean /= d as f32;
            let mut var = 0.0f32;
            for v in src {
                let z = *v - mean;
                var += z * z;
            }
            var /= d as f32;
            let inv = (var + EPS).sqrt().recip();
            for j in 0..d {
                dst[j] = (src[j] - mean) * inv * w[j] + b[j];
            }
        });
}

/// LayerNorm written back into the same buffer. Rows are independent, so a row
/// can read its own mean/variance and then overwrite itself cell by cell —
/// that drops the full-buffer `to_vec()` copy the two-buffer version needed.
fn layer_norm_inplace(x: &mut [f32], w: &[f32], b: &[f32], d: usize) {
    x.par_chunks_mut(d).for_each(|row| {
        let mut mean = 0.0f32;
        for v in row.iter() {
            mean += *v;
        }
        mean /= d as f32;
        let mut var = 0.0f32;
        for v in row.iter() {
            let z = *v - mean;
            var += z * z;
        }
        var /= d as f32;
        let inv = (var + EPS).sqrt().recip();
        for j in 0..d {
            row[j] = (row[j] - mean) * inv * w[j] + b[j];
        }
    });
}

fn softmax(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for v in row.iter() {
        m = m.max(*v);
    }
    let mut s = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    let inv = 1.0 / s;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// How many work chunks the `gemm` crate splits each GEMM into.
///
/// The default is two chunks per rayon thread rather than one. On a
/// heterogeneous CPU (here 8 P-cores + 12 E-cores) one chunk per thread leaves
/// every core holding exactly one chunk, so the slow E-cores dictate the finish
/// and the P-cores idle. Two each lets the P-cores steal the extra work.
/// Paired A/B on 90s_en, 6 interleaved rounds: 1145 ms vs 1223 ms, and the
/// faster setting won every one of the 6 pairs.
///
/// Only `m`/`n` are ever split, never `k`, so the f32 reduction order — and
/// therefore every output bit — is unchanged (verified against the 77-line
/// `SVS_TRACE` gate at 0 / 40 / 48). `SVS_GEMM_THREADS` overrides the default;
/// `0` asks for the crate's own "one per thread" behaviour. `_scratch/gbench
/// thr` sweeps the value.
fn gemm_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        // An empty value must count as "unset": shells make it easy to export a
        // variable as `""`, and `"".parse()` failing into `unwrap_or(0)` would
        // silently pick the crate default instead of the tuned one.
        std::env::var("SVS_GEMM_THREADS")
            .ok()
            .filter(|v| !v.is_empty())
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| 2 * rayon::current_num_threads())
    })
}

/// Same-binary rollback switch for the bias-folding experiment.
///
/// `SVS_BIAS_FOLD=1` (default) folds each bias into an elementwise pass that
/// had to run anyway; `SVS_BIAS_FOLD=0` falls back to the reference path of
/// pre-filling `dst` and letting the GEMM accumulate onto it. Both are
/// bit-identical (`_scratch/gbench bits`), so flipping this is a pure
/// performance A/B — and it lets a paired comparison run against one binary
/// instead of two builds on a machine whose timings drift 10% between runs.
fn bias_fold() -> bool {
    static FOLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FOLD.get_or_init(|| std::env::var("SVS_BIAS_FOLD").map_or(true, |v| v != "0"))
}

/// Freeze the `gemm` crate's tiling parameters before the first real GEMM.
///
/// `gemm` derives `kc` — the k-blocking that decides the f32 accumulation
/// order — from CPUID leaf 4, read lazily on whatever core the first call
/// happens to run on. On a hybrid CPU that is a coin flip: a P-core reports a
/// 48 KiB / 12-way L1d and gets `kc = min(1024, k)`, an E-core reports 32 KiB /
/// 8-way and gets `kc = min(512, k)`. Layer 0's `qkv` GEMM — the only k = 560
/// GEMM in the model — is then either one 560-step reduction or two 280-step
/// ones: two f32 associations, two results about 2 ULP apart, and every buffer
/// after L0 inherits the split. Measured on this machine: exactly two outcomes
/// over 100+ runs of one binary, each single logical processor deterministic,
/// and the outcome follows the core class exactly (8 P-cores always produce
/// one, 12 E-cores the other). Instrumentation and thread settings only change
/// which core the init lands on, which is why the difference looked random.
///
/// So choose the core class explicitly: pin, let the lazy table initialise
/// there, unpin. The default is the largest L1d — P-core parameters, i.e. the
/// result the earlier runs produced, so no reduction order changes.
/// `SVS_GEMM_CACHE=min` selects the E-core parameters, `off` leaves the coin
/// flip in place for A/B comparisons.
fn freeze_gemm_tiling() {
    let mode = std::env::var("SVS_GEMM_CACHE").ok().filter(|v| !v.is_empty());
    let saved = if mode.as_deref() == Some("off") {
        0
    } else {
        pin_to_l1_extreme(mode.as_deref() == Some("min"))
    };
    // Touching the lazy table *is* the one-shot init, so it must happen while
    // the thread sits on the chosen core.
    let info = *gemm_common::cache::CACHE_INFO;
    unpin(saved);
    if std::env::var("SVS_TRACE").ok().as_deref() == Some("1") {
        // Part of the bit gate: the same buffers can only be expected once the
        // tiling regime is the same, so record it in the trace.
        eprintln!(
            "trace tiling l1d={} l1_assoc={} l1_line={} l2={} l3={}",
            info[0].cache_bytes, info[0].associativity, info[0].cache_line_bytes,
            info[1].cache_bytes, info[2].cache_bytes,
        );
    }
}

#[cfg(all(windows, target_arch = "x86_64"))]
mod win_affinity {
    #[link(name = "kernel32")]
    extern "system" {
        pub fn GetCurrentThread() -> *mut core::ffi::c_void;
        pub fn SetThreadAffinityMask(thread: *mut core::ffi::c_void, mask: usize) -> usize;
    }
}

/// Pin to the logical processor with the extreme L1d size (`prefer_small` picks
/// the minimum, otherwise the maximum) and return the previous affinity mask so
/// the caller can restore it. 0 = nothing to restore.
#[cfg(all(windows, target_arch = "x86_64"))]
fn pin_to_l1_extreme(prefer_small: bool) -> usize {
    use win_affinity as w;
    // SAFETY: GetCurrentThread returns a pseudo-handle; SetThreadAffinityMask
    // either rejects the mask (0) or applies it and returns the previous one.
    unsafe {
        let me = w::GetCurrentThread();
        let mut saved = 0usize;
        let mut best: Option<(usize, u64)> = None;
        for lp in 0..64usize {
            let prev = w::SetThreadAffinityMask(me, 1usize << lp);
            if prev == 0 {
                continue;
            }
            if saved == 0 {
                saved = prev;
            }
            let Some(size) = l1d_bytes() else { continue };
            if best.map_or(true, |(_, b)| {
                if prefer_small {
                    size < b
                } else {
                    size > b
                }
            }) {
                best = Some((lp, size));
            }
        }
        if let Some((lp, _)) = best {
            w::SetThreadAffinityMask(me, 1usize << lp);
        }
        saved
    }
}

#[cfg(not(all(windows, target_arch = "x86_64")))]
fn pin_to_l1_extreme(_prefer_small: bool) -> usize {
    0
}

fn unpin(saved: usize) {
    #[cfg(all(windows, target_arch = "x86_64"))]
    if saved != 0 {
        // SAFETY: `saved` is a mask this thread held before (see above).
        unsafe {
            win_affinity::SetThreadAffinityMask(win_affinity::GetCurrentThread(), saved);
        }
    }
    #[cfg(not(all(windows, target_arch = "x86_64")))]
    let _ = saved;
}

/// L1 data cache size of the core this thread is currently running on, from
/// CPUID leaf 4 — the same leaf `gemm` reads. `None` if leaf 4 says nothing.
#[cfg(all(windows, target_arch = "x86_64"))]
fn l1d_bytes() -> Option<u64> {
    for sub in 0..16u32 {
        // SAFETY: CPUID is always available on x86_64.
        let r = unsafe { std::arch::x86_64::__cpuid_count(4, sub) };
        let cache_type = r.eax & 0x1f;
        if cache_type == 0 {
            break;
        }
        let level = (r.eax >> 5) & 0x7;
        // 1 = data, 2 = unified. Skip the instruction cache.
        if level == 1 && (cache_type == 1 || cache_type == 2) {
            let line = u64::from(r.ebx & 0xfff) + 1;
            let parts = u64::from((r.ebx >> 12) & 0x3ff) + 1;
            let ways = u64::from((r.ebx >> 22) & 0x3ff) + 1;
            let sets = u64::from(r.ecx) + 1;
            return Some(line * parts * ways * sets);
        }
    }
    None
}

/// Broadcast `bias` across every output row. Isolated so the profiler can see
/// it separately from the GEMM it feeds. Pure store traffic (~2 GB per
/// utterance), so it goes out across the rayon pool instead of one core.
fn linear_bias(out: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(out.len() % bias.len(), 0);
    out.par_chunks_mut(bias.len())
        .for_each(|row| row.copy_from_slice(bias));
}

/// `out = x @ w`, overwriting `out`.
///
/// Watch the argument order: this crate computes `dst = alpha * dst +
/// beta * acc`, and `read_dst = false` forces `alpha = 0`. So the overwrite
/// path is `alpha = 0, beta = 1` — `beta = 0` would zero the output.
///
/// Callers may fold the bias into an elementwise pass that had to run anyway
/// (relu, q-scale, residual add) instead of pre-filling `dst`. That is fewer
/// bytes moved and is bit-identical to pre-fill + accumulate **only when the
/// GEMM's whole k reduction lands in one `kc` block** — see `linear_into` for
/// the counter-example and `_scratch/gbench bits` for the measurement.
fn linear_gemm(out: &mut [f32], x: &[f32], w: &[f32], m: usize, k: usize, n: usize) {
    unsafe {
        gemm_f32(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            false,
            x.as_ptr(),
            1,
            k as isize,
            w.as_ptr(),
            k as isize,
            1,
            0.0,
            1.0,
        );
    }
}

/// `out += x @ w`, accumulating onto whatever is already in `out`.
fn linear_acc(out: &mut [f32], x: &[f32], w: &[f32], m: usize, k: usize, n: usize) {
    unsafe {
        gemm_f32(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            true,
            x.as_ptr(),
            1,
            k as isize,
            w.as_ptr(),
            k as isize,
            1,
            1.0,
            1.0,
        );
    }
}

/// `out = bias + x @ w`, pre-filling `dst` and letting the GEMM accumulate.
/// Required whenever the k reduction is split across several `kc` blocks —
/// see `w2` in `CpuEngine::layer`.
fn linear_into(out: &mut [f32], x: &[f32], w: &[f32], bias: &[f32], m: usize, k: usize, n: usize) {
    linear_bias(out, bias);
    linear_acc(out, x, w, m, k, n);
}

#[allow(clippy::too_many_arguments)]
unsafe fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut f32,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const f32,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const f32,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: f32,
    beta: f32,
) {
    unsafe {
        gemm(
            m,
            n,
            k,
            dst,
            dst_cs,
            dst_rs,
            read_dst,
            lhs,
            lhs_cs,
            lhs_rs,
            rhs,
            rhs_cs,
            rhs_rs,
            alpha,
            beta,
            false,
            false,
            false,
            Parallelism::Rayon(gemm_threads()),
        );
    }
}

#[allow(dead_code)]
fn trace(name: &str, x: &[f32]) {
    if std::env::var("SVS_TRACE").ok().as_deref() == Some("1") {
        trace_on(name, x);
    }
}

/// Per-phase buffers. `SVS_TRACE_FINE=1` — only for bisecting a run-to-run
/// difference down to the stage that first produces it.
#[allow(dead_code)]
fn trace_f(name: &str, x: &[f32]) {
    if std::env::var("SVS_TRACE_FINE").ok().as_deref() == Some("1") {
        trace_on(name, x);
    }
}

fn trace_on(name: &str, x: &[f32]) {
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    // FNV-1a over the raw f32 bits, in buffer order. `sum`/`max` alone are not
    // a byte-exact gate: two buffers that differ can share both scalars (a
    // difference that cancels in the sum and never reaches the max). The hash
    // is the actual "every byte" check; sum/max stay for human diffing.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in x {
        sum += f64::from(*v);
        max = max.max(v.abs());
        h ^= u64::from(v.to_bits());
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Full precision on purpose. Six decimals looks like plenty next to a sum
    // of ~1e6, but it hides a one-ULP change in a single element — and this
    // gate is only meaningful if it can see exactly that. `{:.9}` on a 1e6
    // magnitude is the whole f64 mantissa; on `max` it is the whole f32 one.
    eprintln!("trace {name} sum={sum:.9} max={max:.9} n={} h={h:016x}", x.len());
}
