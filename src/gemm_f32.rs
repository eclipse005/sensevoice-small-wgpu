//! f32 GEMM matching the official PyTorch linear: C = A @ Wᵀ, f32 accumulate.
//!
//! 128×128 workgroup tile, 16×16 threads, each thread owning an 8×8 micro-tile
//! (64 named-scalar accumulators).  Tiles are staged through shared memory in
//! **k-major 16-byte chunks** — a chunk is four rows at one k, stored as
//! `vec4<f32>` — so a thread's eight A values and eight B values are two `vec4`
//! loads each instead of sixteen scalar ones.  On this card (P104-100) the
//! shared-load *count* is what prices the inner loop: the same bytes through 16
//! scalar loads run at 2.66 TFLOP/s and through 4 `vec4` loads at 3.80
//! (measured with a probe in a sibling port; the shipped Qwen3-ASR GEMM here is
//! the same tile geometry and reaches 3.1).
//!
//! Two details are load-bearing:
//!
//! - **A needs no permutation, B does.**  `A4[q*AQ + 2*ty]` depends only on
//!   `ty`, so the sixteen `tx` lanes of a phase all want the same sixteen
//!   bytes — a broadcast.  B's sixteen lanes would stride two chunks each and
//!   land on 2 bank groups per phase, so the row *groups* are permuted:
//!   `pos(g, s) = (g & 7) + 8*s + 16*(g >> 3)` spreads them over eight groups,
//!   the two-phase minimum for 16-byte loads.
//! - A `f32` array cannot be read as `vec4` in WGSL, so the slabs are declared
//!   `array<vec4<f32>, ..>` and every store writes a whole chunk.  That forces
//!   the global load stage to gather four rows at one k, which costs some
//!   coalescing; these GEMMs are compute-bound by ~150 FLOP/byte, so the trade
//!   is free here and would not be for a memory-bound shape.
//!
//! The micro-tile accumulators are emitted as **named scalars, not an array**.
//! A `array<array<f32, TM>, TN>` indexed by loop variables is legal WGSL but
//! naga/DXIL are free to place it in private (scratch) memory, which on NVIDIA
//! is backed by device memory — every FMA then costs a load and a store.
//!
//! Values, accumulation order and FMA count are unchanged from the scalar form:
//! each accumulator still takes its k contributions in ascending order, so the
//! result is bit-identical, and the k loop is unrolled four steps at a time only
//! so the compiler can hoist the next step's shared loads above the current
//! step's FMAs.

pub const GEMM_TM: usize = 8;
pub const GEMM_TN: usize = 8;
pub const GEMM_BK: usize = 8;
/// Thread grid inside a workgroup. 16×16 = 256 threads for the 128×128 tile:
/// the wider block is worth ~7% over 128 threads here (more warps to hide the
/// shared/global latency), and 8×8 keeps the 64-accumulator micro-tile that
/// fits the register file.
pub const GEMM_TX: usize = 16;
pub const GEMM_TY: usize = 16;

/// Rows of A / cols of C handled by one workgroup. `tid = ty*TX + tx`, so `ty`
/// walks rows and `tx` walks columns — swapping these overwrites neighbouring
/// tiles and silently corrupts the result.
pub const GEMM_BM: u32 = (GEMM_TY * GEMM_TM) as u32;
pub const GEMM_BN: u32 = (GEMM_TX * GEMM_TN) as u32;

/// Chunks per k row: one chunk is four rows at one k.
const AQ: usize = GEMM_BM as usize / 4;
const BQ: usize = GEMM_BN as usize / 4;
/// Chunks in one slab (per buffer, not per double-buffered pair).
const ASM: usize = GEMM_BM as usize * GEMM_BK / 4;
const BSM: usize = GEMM_BN as usize * GEMM_BK / 4;
const NTHR: usize = GEMM_TX * GEMM_TY;
/// Four `vec4` steps per unrolled iteration; the unrolled x86-64 comment about
/// the compiler seeing past one step applies here too.
const UNROLL_Q: usize = 4;

/// Shared-memory bytes the double-buffered tile needs: two alternating
/// `BM×BK/4` / `BN×BK/4` chunk pairs at 16 bytes each.
pub const GEMM_SMEM: u32 = (2 * (ASM + BSM) * 16) as u32;

pub fn gemm_f32(transb: bool, beta: bool, bias: bool) -> String {
    // One chunk per thread per slab is what makes the store addressing a single
    // expression; anything else needs a loop over chunks.
    assert_eq!(ASM, NTHR, "A slab must be one chunk per thread");
    assert_eq!(BSM, NTHR, "B slab must be one chunk per thread");
    assert_eq!(GEMM_BK % UNROLL_Q, 0, "BK must be a multiple of the unroll");
    let trans = if transb { "1" } else { "0" };
    let beta_s = if beta { "1" } else { "0" };
    let bias_s = if bias {
        "@group(0) @binding(4) var<storage, read> Bias: array<f32>;\n"
    } else {
        ""
    };

    // Named-scalar accumulators.
    let mut decl = String::new();
    let mut init = String::new();
    for i in 0..GEMM_TM {
        for j in 0..GEMM_TN {
            decl.push_str(&format!("    var c{i}{j}: f32;\n"));
            init.push_str(&format!("    c{i}{j} = 0.0;\n"));
        }
    }

    // Per-thread chunk coordinates. `ck` is the k this thread's chunk lives at
    // (constant across the loop), `crg` its row group; `bc`/`bslot` are the B
    // chunk's logical index and its permuted physical slot.
    let chunk_decl = "    let ck = tid >> 5u;\n    let crg = tid & 31u;\n    let bg = (tid >> 1u) & 15u;\n    let bs = tid & 1u;\n    let bslot = (bg & 7u) + 8u * bs + 16u * (bg >> 3u);\n".to_string();

    // Global load stage: each thread gathers one chunk (four rows at one k).
    let a_load = |kx: &str| -> String {
        (0..4)
            .map(|r| {
                format!(
                    "        pfa{r} = A[abase + (m0 + 4u * crg + {r}u) * gd.lda + {kx} + ck];\n"
                )
            })
            .collect()
    };
    let b_load = |kx: &str| -> String {
        if transb {
            (0..4)
                .map(|r| {
                    format!(
                        "        pfb{r} = W[bbase + ({kx} + ck) * gd.n + n0 + 4u * bc + {r}u];\n"
                    )
                })
                .collect()
        } else {
            (0..4)
                .map(|r| {
                    format!(
                        "        pfb{r} = W[bbase + (n0 + 4u * bc + {r}u) * gd.k + {kx} + ck];\n"
                    )
                })
                .collect()
        }
    };
    // `bc` is the logical chunk index (cols 4*bc .. +3); reuse the same
    // expression the store needs.
    let b_load_all = |kx: &str| -> String { format!("        let bc = tid & 31u;\n{}", b_load(kx)) };
    let store = |abuf: &str, bbuf: &str| -> String {
        format!(
            "        A4[{abuf} + tid] = vec4<f32>(pfa0, pfa1, pfa2, pfa3);\n\
             \x20       B4[{bbuf} + ck * BQ + bslot] = vec4<f32>(pfb0, pfb1, pfb2, pfb3);\n"
        )
    };

    // Compute: 4 k-steps per loop iteration, each thread reading two A chunks
    // (rows 8*ty..+7) and two B chunks (cols 8*tx..+7).  The accumulation order
    // per output is still ascending k, so the result is bit-identical.
    let mut compute = String::new();
    for u in 0..UNROLL_Q {
        compute.push_str(&format!(
            "        let av{u} = A4[abase4 + (q0 + {u}u) * AQ + ty * 2u];\n\
             \x20       let aw{u} = A4[abase4 + (q0 + {u}u) * AQ + ty * 2u + 1u];\n\
             \x20       let bv{u} = B4[bbase4 + (q0 + {u}u) * BQ + (tx & 7u) + 16u * (tx >> 3u)];\n\
             \x20       let bw{u} = B4[bbase4 + (q0 + {u}u) * BQ + (tx & 7u) + 8u + 16u * (tx >> 3u)];\n"
        ));
        for i in 0..GEMM_TM {
            let a = if i < 4 {
                format!("av{u}[{i}]")
            } else {
                format!("aw{u}[{}]", i - 4)
            };
            for j in 0..GEMM_TN {
                let b = if j < 4 {
                    format!("bv{u}[{j}]")
                } else {
                    format!("bw{u}[{}]", j - 4)
                };
                compute.push_str(&format!("        c{i}{j} = c{i}{j} + {a} * {b};\n"));
            }
        }
    }

    // Epilogue: one store per micro-tile element, accumulate, add bias, add C
    // when beta, store.
    let mut epilogue = String::new();
    for i in 0..GEMM_TM {
        epilogue.push_str(&format!("    let row{i} = m0 + ty * TM + {i}u;\n"));
    }
    for i in 0..GEMM_TM {
        for j in 0..GEMM_TN {
            let bias_line = if bias {
                format!("    o{i}_{j} = o{i}_{j} + Bias[col{i}_{j}];\n")
            } else {
                String::new()
            };
            epilogue.push_str(&format!(
                "    let col{i}_{j} = n0 + tx * TN + {j}u;\n\
                 var o{i}_{j} = c{i}{j};\n\
                 {bias_line}\
                 let dst{i}_{j} = cbase + row{i} * gd.ldc + col{i}_{j};\n\
                 if (BETA == 1u) {{ o{i}_{j} = o{i}_{j} + C[dst{i}_{j}]; }}\n\
                 C[dst{i}_{j}] = o{i}_{j};\n"
            ));
        }
    }

    format!(
        r#"struct GDims {{
    n: u32, k: u32, ldc: u32,
    bsa: u32, bsb: u32, bsc: u32, lda: u32,
}};
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> gd: GDims;
{bias_s}
const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {bk}u;
const TM: u32 = {tm}u;
const TN: u32 = {tn}u;
const AQ: u32 = {aq}u;
const BQ: u32 = {bq}u;
const TRANSB: u32 = {trans}u;
const BETA: u32 = {beta_s}u;
var<workgroup> A4: array<vec4<f32>, {asm2}>;
var<workgroup> B4: array<vec4<f32>, {bsm2}>;

@compute @workgroup_size({txn}, {tyn}, 1)
fn gemm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * {txn}u + tx;
    let m0 = wid.y * BM;
    let n0 = wid.x * BN;
    let abase = wid.z * gd.bsa;
    let bbase = wid.z * gd.bsb;
    let cbase = wid.z * gd.bsc;
{chunk_decl}{decl}{init}
    var pfa0: f32;
    var pfa1: f32;
    var pfa2: f32;
    var pfa3: f32;
    var pfb0: f32;
    var pfb1: f32;
    var pfb2: f32;
    var pfb3: f32;

    // Register-staged double buffering: the global loads for step i+1 are
    // issued before the FMAs on step i, so DRAM latency overlaps compute
    // instead of being exposed at a barrier.
{a_load0}{b_load0}{store0}
    var buf: u32 = 0u;
    for (var k0: u32 = 0u; k0 < gd.k; k0 = k0 + BK) {{
        let more = (k0 + BK) < gd.k;
        let knext = k0 + BK;
        workgroupBarrier();
        if (more) {{
{a_load1}{b_load1}        }}
        let abase4 = buf * {asm}u;
        let bbase4 = buf * {bsm}u;
        var q0: u32 = 0u;
        loop {{
            if (q0 >= BK) {{ break; }}
{compute}            q0 = q0 + {unroll}u;
        }}
        if (more) {{
            let nbuf = 1u - buf;
            {store1}
            buf = nbuf;
        }}
    }}
{epilogue}}}
"#,
        bm = GEMM_BM,
        bn = GEMM_BN,
        bk = GEMM_BK,
        tm = GEMM_TM,
        tn = GEMM_TN,
        aq = AQ,
        bq = BQ,
        txn = GEMM_TX,
        tyn = GEMM_TY,
        asm = ASM,
        bsm = BSM,
        asm2 = 2 * ASM,
        bsm2 = 2 * BSM,
        unroll = UNROLL_Q,
        chunk_decl = chunk_decl,
        a_load0 = a_load("0u"),
        b_load0 = b_load_all("0u"),
        a_load1 = a_load("knext"),
        b_load1 = b_load_all("knext"),
        store0 = store("0u", "0u"),
        store1 = store(&format!("nbuf * {ASM}u"), &format!("nbuf * {BSM}u")),
    )
}

pub fn gemm_f32_bias(beta: bool) -> String {
    gemm_f32(false, beta, true)
}
