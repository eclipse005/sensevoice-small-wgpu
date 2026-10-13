//! Same-binary A/B switches and diagnostic flags.
//!
//! Every knob is an environment variable, read once per process into a
//! `OnceLock` so hot loops never touch the environment. The value is fixed at
//! first use: an A/B comparison must run two processes that each saw the same
//! configuration for their whole lifetime.
//!
//! The default is always the validated path, and every switch has a rollback
//! value that restores the older behaviour bit-for-bit. That is what lets the
//! timing scripts compare one binary instead of two builds on a machine whose
//! timings drift between runs.

use std::sync::OnceLock;

use crate::ctc::VOCAB;

// ---- CPU encoder ----

/// Same-binary rollback switch for the FSMN loop-nest swap. Both settings are
/// bit-identical — `_scratch` runs diff the full `SVS_TRACE` output — so this
/// exists only for paired A/B timing.
pub fn fsmn_flat() -> bool {
    static FLAT: OnceLock<bool> = OnceLock::new();
    *FLAT.get_or_init(|| std::env::var("SVS_FSMN_FLAT").map_or(true, |v| v != "0"))
}

/// Vocab tile width for the fused CTC projection + argmax.
///
/// `SVS_CTC_TILE=<n>` picks the width; `0` restores the monolithic t × VOCAB
/// logits buffer as the same-binary rollback. Both are bit-identical — k = 512
/// is one `kc` block at any width, so the per-logit reduction is unchanged and
/// the bias fold only reorders an addition — and the `SVS_TRACE` gate diffs the
/// two settings against each other. The default is 2048 (12 MB of tile at
/// t = 1504): a width sweep on 90s_en was monotone from 2048 to 16384 and
/// 2048 won 4/6 paired runs against 4096.
pub fn ctc_tile() -> usize {
    static W: OnceLock<usize> = OnceLock::new();
    let w = *W.get_or_init(|| {
        std::env::var("SVS_CTC_TILE")
            .ok()
            .filter(|v| !v.is_empty())
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048)
    });
    if w == 0 {
        VOCAB
    } else {
        w.min(VOCAB)
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
pub fn gemm_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
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
pub fn bias_fold() -> bool {
    static FOLD: OnceLock<bool> = OnceLock::new();
    *FOLD.get_or_init(|| std::env::var("SVS_BIAS_FOLD").map_or(true, |v| v != "0"))
}

/// `SVS_GEMM_CACHE=max|min|off`: which core class's L1d geometry the `gemm`
/// tiling parameters are frozen to (`off` skips the freeze). See
/// `encoder_cpu::freeze_gemm_tiling` for why the value must be fixed before
/// the first GEMM.
pub fn gemm_cache_mode() -> Option<String> {
    std::env::var("SVS_GEMM_CACHE").ok().filter(|v| !v.is_empty())
}

/// `SVS_CPU_PROF=1`: per-phase wall-clock accounting in the CPU encoder.
pub fn cpu_prof() -> bool {
    env_eq1("SVS_CPU_PROF")
}

/// `SVS_TRACE=1`: layered `{:.9}` + FNV hash trace, the bit-exactness gate.
pub fn trace() -> bool {
    env_eq1("SVS_TRACE")
}

/// `SVS_TRACE_FINE=1`: per-stage trace inside a layer, for bisecting drift.
pub fn trace_fine() -> bool {
    env_eq1("SVS_TRACE_FINE")
}

// ---- GPU encoder ----

/// `SVS_GPU_PROF=1`: per-dispatch timestamps (needs TIMESTAMP_QUERY).
pub fn gpu_prof() -> bool {
    env_eq1("SVS_GPU_PROF")
}

/// `SVS_GPU_CHUNK=N`: override the TDR batch size.
pub fn gpu_chunk() -> Option<usize> {
    std::env::var("SVS_GPU_CHUNK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|c| *c > 0)
}

/// `SVS_GPU_STOP=N`: run only the first N recorded ops.
pub fn gpu_stop() -> usize {
    std::env::var("SVS_GPU_STOP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

/// `SVS_GPU_DUMP_OPS=1`: print every recorded op.
pub fn gpu_dump_ops() -> bool {
    env_eq1("SVS_GPU_DUMP_OPS")
}

/// `SVS_GPU_DEBUG=1`: scan the intermediate activations after the run.
pub fn gpu_debug() -> bool {
    env_eq1("SVS_GPU_DEBUG")
}

fn env_eq1(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}
