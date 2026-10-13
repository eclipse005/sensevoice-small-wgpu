//! f32 GEMM matching the official PyTorch linear: C = A @ Wᵀ, f32 accumulate.
//!
//! Tiled 16×16 threads, each owning a TM×TN micro-tile of C, staged through
//! shared-memory As/Bs in BK-deep k steps. M, N and K must be multiples of the
//! tile, which the encoder already pads to. `transb` reads W as [k, n]
//! (attention AV). `beta` adds the value already in C. `bias` adds a
//! per-column f32 vector.
//!
//! The micro-tile accumulators are emitted as **named scalars, not an array**.
//! A `array<array<f32, TM>, TN>` indexed by loop variables is legal WGSL but
//! naga/DXIL are free to place it in private (scratch) memory, which on NVIDIA
//! is backed by device memory — every FMA then costs a load and a store. The
//! generated form keeps all TM*TN accumulators in registers. It is mechanically
//! verbose and that is the point.
//!
//! The k reduction order is unchanged (k0 ascending, then ki ascending), so the
//! result is bit-identical to the array form and to the CPU oracle.

pub const GEMM_TM: usize = 16;
pub const GEMM_TN: usize = 8;
pub const GEMM_BK: usize = 8;
/// Thread grid inside a workgroup. Rectangular on purpose: the shared-memory
/// traffic a thread generates per k step is TM + TN floats for TM*TN FMAs, and
/// Pascal pairs 128 FMAs/cycle with 32 shared floats/cycle. A square 8×8
/// micro-tile is exactly 4:1, i.e. pinned to the shared-memory roof; widening
/// one side to 16×8 is 5.33:1 and buys headroom without growing the
/// 128×128 workgroup tile.
pub const GEMM_TX: usize = 16;
pub const GEMM_TY: usize = 8;

/// Rows of A / cols of C handled by one workgroup. `tid = ty*TX + tx`, so `ty`
/// walks rows and `tx` walks columns — swapping these overwrites neighbouring
/// tiles and silently corrupts the result.
pub const GEMM_BM: u32 = (GEMM_TY * GEMM_TM) as u32;
pub const GEMM_BN: u32 = (GEMM_TX * GEMM_TN) as u32;

/// Shared-memory bytes the double-buffered tile needs: two alternating
/// `BM×BK` / `BN×BK` slab pairs.
pub const GEMM_SMEM: u32 =
    (2 * (GEMM_BM as usize + GEMM_BN as usize) * GEMM_BK * 4) as u32;

fn reps(total: usize, threads: usize) -> usize {
    total.div_ceil(threads)
}

pub fn gemm_f32(transb: bool, beta: bool, bias: bool) -> String {
    let trans = if transb { "1" } else { "0" };
    let beta_s = if beta { "1" } else { "0" };
    let bias_s = if bias {
        "@group(0) @binding(4) var<storage, read> Bias: array<f32>;\n"
    } else {
        ""
    };
    // Which lane index runs fastest in the prefetch depends on how W is laid
    // out, and getting this wrong makes the whole load uncoalesced:
    //
    //   TRANSB == 0  W is [n, k], so consecutive `ki` are adjacent in memory.
    //                Loading with ki running fastest makes a warp read BK
    //                consecutive k of one n-row instead of BN lanes each
    //                striding a whole k.
    //   TRANSB == 1  W is [k, n], so consecutive `col` are adjacent and the
    //                col-fastest mapping is already the coalesced one.
    let nthr = GEMM_TX * GEMM_TY;
    let nthr2 = nthr;

    // Named-scalar accumulators.
    let mut decl = String::new();
    let mut init = String::new();
    for i in 0..GEMM_TM {
        for j in 0..GEMM_TN {
            decl.push_str(&format!("    var acc_{i}_{j}: f32;\n"));
            init.push_str(&format!("    acc_{i}_{j} = 0.0;\n"));
        }
    }
    let mut acc_load2 = String::new();
    for i in 0..GEMM_TM {
        acc_load2.push_str(&format!(
            "        av_{i} = As[abase_off + (ty * TM + {i}u) * BK + ki];\n"
        ));
    }
    let mut bv_load2 = String::new();
    for j in 0..GEMM_TN {
        bv_load2.push_str(&format!(
            "        bv_{j} = Bs[bbase_off + ki * BN + tx * TN + {j}u];\n"
        ));
    }
    let mut fma = String::new();
    for i in 0..GEMM_TM {
        for j in 0..GEMM_TN {
            fma.push_str(&format!("        acc_{i}_{j} = acc_{i}_{j} + av_{i} * bv_{j};\n"));
        }
    }
    let mut av_decl = String::new();
    let mut bv_decl = String::new();
    for i in 0..GEMM_TM {
        av_decl.push_str(&format!("    var av_{i}: f32;\n"));
    }
    for j in 0..GEMM_TN {
        bv_decl.push_str(&format!("    var bv_{j}: f32;\n"));
    }

    // Per-thread staging registers for the prefetched tile. Each thread owns
    // (BM*BK)/threads A cells and (BN*BK)/threads B cells.
    let ra = reps(GEMM_BM as usize * GEMM_BK, nthr2);
    let rb = reps(GEMM_BN as usize * GEMM_BK, nthr2);
    let mut stage_decl = String::new();
    for e in 0..ra {
        stage_decl.push_str(&format!("    var ra{e}: f32;\n"));
    }
    for e in 0..rb {
        stage_decl.push_str(&format!("    var rb{e}: f32;\n"));
    }
    let pidx = |e: usize| format!("(tid + {e}u * {nthr2}u)");
    let a_addr = |k: &str| -> String {
        (0..ra)
            .map(|e| {
                let p = pidx(e);
                format!(
                    "        ra{e} = A[abase + (m0 + {p} / BK) * gd.lda + ({k} + {p} % BK)];\n"
                )
            })
            .collect()
    };
    let b_addr = |k: &str| -> String {
        if transb {
            (0..rb)
                .map(|e| {
                    let p = pidx(e);
                    format!(
                        "        rb{e} = W[bbase + ({k} + {p} / BN) * gd.n + (n0 + {p} % BN)];\n"
                    )
                })
                .collect()
        } else {
            (0..rb)
                .map(|e| {
                    let p = pidx(e);
                    format!(
                        "        rb{e} = W[bbase + (n0 + {p} / BK) * gd.k + ({k} + {p} % BK)];\n"
                    )
                })
                .collect()
        }
    };
    let a_store = |off: &str| -> String {
        (0..ra)
            .map(|e| format!("        As[{off} + tid + {e}u * {nthr2}u] = ra{e};\n"))
            .collect()
    };
    // The B slab is [BK][BN], i.e. row stride BN. Under the non-transposed
    // (ki-fastest) load mapping the destination is ki*BN + col, which is NOT p;
    // under the transposed mapping it happens to equal p.
    let b_slot = |p: &str| -> String {
        if transb {
            format!("{p}")
        } else {
            format!("({p} % BK) * BN + ({p} / BK)")
        }
    };
    let b_store = |off: &str| -> String {
        (0..rb)
            .map(|e| {
                let p = pidx(e);
                format!("        Bs[{off} + {}] = rb{e};\n", b_slot(&p))
            })
            .collect()
    };
    let stage_load0 = format!("{}{}", a_addr("0u"), b_addr("0u"));
    let stage_store0 = format!("{}{}", a_store("0u"), b_store("0u"));
    let stage_load1 = format!("{}{}", a_addr("knext"), b_addr("knext"));
    let stage_store1 = format!("{}{}", a_store("nbuf * ASM"), b_store("nbuf * BSM"));

    // Epilogue: one store per micro-tile element, same order as the original
    // array form (accumulate, add bias, add C when beta, store).
    let mut store = String::new();
    for i in 0..GEMM_TM {
        store.push_str(&format!("    let row{i} = m0 + ty * TM + {i}u;\n"));
    }
    for i in 0..GEMM_TM {
        for j in 0..GEMM_TN {
            let bias_line = if bias {
                format!("    o{i}_{j} = o{i}_{j} + Bias[col{i}_{j}];\n")
            } else {
                String::new()
            };
            store.push_str(&format!(
                "    let col{i}_{j} = n0 + tx * TN + {j}u;\n\
                 var o{i}_{j} = acc_{i}_{j};\n\
                 {bias_line}\
                 let dst{i}_{j} = cbase + row{i} * gd.ldc + col{i}_{j};\n\
                 if (BETA == 1u) {{ o{i}_{j} = o{i}_{j} + C[dst{i}_{j}]; }}\n\
                 C[dst{i}_{j}] = o{i}_{j};\n"
            ));
        }
    }

    format!(
        r#"struct GDims {{
    m: u32, n: u32, k: u32, ldc: u32,
    bsa: u32, bsb: u32, bsc: u32, beta: u32, row0: u32, lda: u32,
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
const TRANSB: u32 = {trans}u;
const BETA: u32 = {beta_s}u;
const ASM: u32 = {asn}u;
const BSM: u32 = {bsn}u;
var<workgroup> As: array<f32, {as2n}>;
var<workgroup> Bs: array<f32, {bs2n}>;

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
{decl}{av_decl}{bv_decl}{stage_decl}{init}

    // Register-staged double buffering: the global loads for step i+1 are issued
    // before the FMAs on step i, so DRAM latency overlaps with compute instead
    // of being exposed at a barrier. Two shared slabs alternate, and the single
    // barrier per step is enough because step i writes the slab step i-1 read.
    {stage_load0}
    {stage_store0}
    var buf: u32 = 0u;
    for (var k0: u32 = 0u; k0 < gd.k; k0 = k0 + BK) {{
        let more = (k0 + BK) < gd.k;
        let knext = k0 + BK;
        workgroupBarrier();
        if (more) {{
            {stage_load1}
        }}
        let abase_off = buf * ASM;
        let bbase_off = buf * BSM;
        for (var ki: u32 = 0u; ki < BK; ki = ki + 1u) {{
{acc_load2}{bv_load2}{fma}        }}
        if (more) {{
            let nbuf = 1u - buf;
            {stage_store1}
            buf = nbuf;
        }}
    }}
{store}}}
"#,
        bm = GEMM_BM,
        bn = GEMM_BN,
        bk = GEMM_BK,
        tm = GEMM_TM,
        tn = GEMM_TN,
        asn = GEMM_BM as usize * GEMM_BK,
        bsn = GEMM_BN as usize * GEMM_BK,
        as2n = 2 * GEMM_BM as usize * GEMM_BK,
        bs2n = 2 * GEMM_BN as usize * GEMM_BK,
        txn = GEMM_TX,
        tyn = GEMM_TY,
    )
}

pub fn gemm_f32_bias(beta: bool) -> String {
    gemm_f32(false, beta, true)
}
