//! f32 GEMM matching the official PyTorch linear: C = A @ Wᵀ, f32 accumulate.
//!
//! Two workgroup tiles share one body:
//!
//! - **[`gemm_f32`]`: 128×128**, 16×16 threads, 8×8 micro-tile (64 named-scalar
//!   accumulators).  The workhorse for the long-sequence shapes.
//! - **[`gemm_f32_tall`]: 64×128**, 16×8 threads, same micro-tile.  When M is
//!   small (short audio), a 128-row tile yields only a handful of blocks —
//!   `outp` at `M=512` is 16 blocks for 15 SMs — and the card idles in wave
//!   quantization.  Halving the row tile doubles the block count.
//!
//! Tiles are staged through shared memory in **k-major 16-byte chunks** — a
//! chunk is four rows at one k, stored as `vec4<f32>` — so a thread's eight A
//! values and eight B values are two `vec4` loads each instead of sixteen
//! scalar ones.  On this card (P104-100) the shared-load *count* is what
//! prices the inner loop: the same bytes through 16 scalar loads run at 2.66
//! TFLOP/s and through 4 `vec4` loads at 3.80.  Two details are load-bearing:
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
//! result is bit-identical.  The k loop is unrolled four steps at a time only so
//! the compiler can hoist the next step's shared loads above the current step's
//! FMAs.

pub const GEMM_TM: usize = 8;
pub const GEMM_TN: usize = 8;
pub const GEMM_BK: usize = 8;
/// Thread grid inside a workgroup for the 128×128 tile.
pub const GEMM_TX: usize = 16;
pub const GEMM_TY: usize = 16;

/// Rows of A / cols of C handled by the default workgroup tile. `tid = ty*TX +
/// tx`, so `ty` walks rows and `tx` walks columns — swapping these overwrites
/// neighbouring tiles and silently corrupts the result.
pub const GEMM_BM: u32 = (GEMM_TY * GEMM_TM) as u32;
pub const GEMM_BN: u32 = (GEMM_TX * GEMM_TN) as u32;
/// Rows of the tall (small-M) tile: `gemm_f32_tall` halves `TY`.
pub const GEMM_BM_TALL: u32 = (8 * GEMM_TM) as u32;

/// Four `vec4` chunk loads per unrolled k step.
const UNROLL_Q: usize = 4;

/// Shared-memory bytes the default tile's double buffer needs: two alternating
/// `BM×BK/4` / `BN×BK/4` chunk pairs at 16 bytes each.
/// The largest slab pair any variant allocates (the tall/BK=16 tile), kept as
/// the conservative number for the device-limit check in `gpu.rs`.
pub const GEMM_SMEM: u32 = {
    let normal = 2 * (GEMM_BM as usize + GEMM_BN as usize) * GEMM_BK / 4 * 16;
    let tall = 2 * (GEMM_BM_TALL as usize + GEMM_BN as usize) * 16 / 4 * 16;
    if normal > tall { normal as u32 } else { tall as u32 }
};

pub fn gemm_f32(transb: bool, beta: bool, bias: bool) -> String {
    gemm_f32_tiled(transb, beta, bias, GEMM_TY, GEMM_BK)
}

/// The small-M tile: same micro-tile, half the rows per workgroup and a deeper
/// k slab (BK=16), which halves the barrier count.  Measured on 30s_zh: the
/// 128-row/BK=8 tile gives 156 ms, the tall BK=16 one 145 ms; on 90s_en the
/// roles flip (454 vs 484 ms) because the deeper slab costs a block of
/// occupancy, so the encoder picks per dispatch.
pub fn gemm_f32_tall(transb: bool, beta: bool, bias: bool) -> String {
    gemm_f32_tiled(transb, beta, bias, 8, 16)
}

fn gemm_f32_tiled(transb: bool, beta: bool, bias: bool, tyn: usize, bk: usize) -> String {
    let bm = tyn * GEMM_TM;
    let bn = GEMM_TX * GEMM_TN;
    let nthr = GEMM_TX * tyn;
    let aq = bm / 4; // chunks per k row of A
    let bq = bn / 4;
    let asm = bm * bk / 4; // chunks per slab
    let bsm = bn * bk / 4;
    assert_eq!(asm % nthr, 0, "A slab must divide evenly over the threads");
    assert_eq!(bsm % nthr, 0, "B slab must divide evenly over the threads");
    assert_eq!(bk % UNROLL_Q, 0, "BK must be a multiple of the unroll");
    let n_a = asm / nthr;
    let n_b = bsm / nthr;
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

    // Chunk coordinates live at function scope: both the load blocks and the
    // store blocks (which are in different `if (more)` scopes) need them.
    // Chunk `tid + c*NTHR` is `(k, group) = (id / AQ, id % AQ)` for A and
    // `(k, chunk) = (id / BQ, id % BQ)` for B; the B chunk's physical slot is
    // the permutation `(g&7) + 8*s + 16*(g>>3)` with `g = chunk >> 1`.
    let mut coords = String::new();
    for c in 0..n_a {
        let n = nthr;
        coords.push_str(&format!(
            "    let aid{c} = tid + {c}u * {n}u;
                 let ak{c} = aid{c} / AQ;
                 let ag{c} = aid{c} % AQ;
"
        ));
    }
    for c in 0..n_b {
        let n = nthr;
        coords.push_str(&format!(
            "    let bid{c} = tid + {c}u * {n}u;
                 let bk{c} = bid{c} / BQ;
                 let bc{c} = bid{c} % BQ;
                 let bg{c} = bc{c} >> 1u;
                 let bs{c} = bc{c} & 1u;
                 let bp{c} = (bg{c} & 7u) + 8u * bs{c} + 16u * (bg{c} >> 3u);
"
        ));
    }
    // Global load stage: each thread gathers whole chunks (four rows at one k).
    let a_load = |kx: &str| -> String {
        let mut t = String::new();
        for c in 0..n_a {
            for r in 0..4 {
                t.push_str(&format!(
                    "        pfa{c}_{r} = A[abase + (m0 + 4u * ag{c} + {r}u) * gd.lda + {kx} + ak{c}];
"
                ));
            }
        }
        t
    };
    let b_load = |kx: &str| -> String {
        let mut t = String::new();
        for c in 0..n_b {
            for r in 0..4 {
                let line = if transb {
                    format!(
                        "        pfb{c}_{r} = W[bbase + ({kx} + bk{c}) * gd.n + n0 + 4u * bc{c} + {r}u];
"
                    )
                } else {
                    format!(
                        "        pfb{c}_{r} = W[bbase + (n0 + 4u * bc{c} + {r}u) * gd.k + {kx} + bk{c}];
"
                    )
                };
                t.push_str(&line);
            }
        }
        t
    };
    let store = |abuf: &str, bbuf: &str| -> String {
        let mut t = String::new();
        for c in 0..n_a {
            t.push_str(&format!(
                "        A4[{abuf} + aid{c}] = vec4<f32>(pfa{c}_0, pfa{c}_1, pfa{c}_2, pfa{c}_3);
"
            ));
        }
        for c in 0..n_b {
            t.push_str(&format!(
                "        B4[{bbuf} + bk{c} * BQ + bp{c}] = vec4<f32>(pfb{c}_0, pfb{c}_1, pfb{c}_2, pfb{c}_3);
"
            ));
        }
        t
    };

    // Register declarations for the staging values.
    let mut staged = String::new();
    for c in 0..n_a {
        for r in 0..4 {
            staged.push_str(&format!("    var pfa{c}_{r}: f32;\n"));
        }
    }
    for c in 0..n_b {
        for r in 0..4 {
            staged.push_str(&format!("    var pfb{c}_{r}: f32;\n"));
        }
    }

    // Compute: 4 k-steps per loop iteration, each thread reading two A chunks
    // (rows 8*ty..+7) and two B chunks (cols 8*tx..+7).
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

    // Epilogue: accumulate, add bias, add C when beta, store.
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

    let store0 = store("0u", "0u");
    let store1 = store("nbuf * ASM", "nbuf * BSM");

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
const ASM: u32 = {asm}u;
const BSM: u32 = {bsm}u;
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
{decl}{init}{staged}{coords}
{a_load0}{b_load0}{store0}
    var buf: u32 = 0u;
    for (var k0: u32 = 0u; k0 < gd.k; k0 = k0 + BK) {{
        let more = (k0 + BK) < gd.k;
        let knext = k0 + BK;
        workgroupBarrier();
        if (more) {{
{a_load1}{b_load1}        }}
        let abase4 = buf * ASM;
        let bbase4 = buf * BSM;
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
        bm = bm,
        bn = bn,
        bk = bk,
        tm = GEMM_TM,
        tn = GEMM_TN,
        aq = aq,
        bq = bq,
        asm = asm,
        bsm = bsm,
        txn = GEMM_TX,
        tyn = tyn,
        asm2 = 2 * asm,
        bsm2 = 2 * bsm,
        unroll = UNROLL_Q,
        staged = staged,
        coords = coords,
        a_load0 = a_load("0u"),
        b_load0 = b_load("0u"),
        a_load1 = a_load("knext"),
        b_load1 = b_load("knext"),
        store0 = store0,
        store1 = store1,
    )
}

pub fn gemm_f32_bias(beta: bool) -> String {
    gemm_f32(false, beta, true)
}

pub fn gemm_f32_bias_tall(beta: bool) -> String {
    gemm_f32_tall(false, beta, true)
}
