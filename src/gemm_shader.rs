/// Prefill GEMM:  C[m,n] f16 = A[m,k] f16 × W[n,k]ᵀ (or × W[n,k] when
/// `transb`), f32 accumulate, one f16 rounding.  Tile 128×128 per workgroup
/// (16×16 threads, 8×8 micro-tile), BK=16, software-pipelined (next tile
/// prefetched into registers during compute).  `beta=1` folds the residual add
/// into the epilogue: acc + f16(C_in), one rounding.
/// Tile of [`prefill_gemm`], as a single source of truth.
///
/// The kernel's own `BM`/`BN`/`BK` constants and **every caller's** padding and
/// dispatch grids must agree.  They are exported here so a caller cannot drift:
/// passing 64 where the kernel uses 128 leaves half of each axis uncomputed and
/// the result is silently wrong rather than an error.
pub const PREFILL_GEMM_TM: usize = 8;
pub const PREFILL_GEMM_TN: usize = 8;
pub const PREFILL_GEMM_BK: usize = 16;
/// M tile width — `m` must be padded to a multiple of this.
pub const PREFILL_GEMM_BM: usize = 16 * PREFILL_GEMM_TM;
/// N tile width — `n` (the weight's row count) must be padded to a multiple.
pub const PREFILL_GEMM_BN: usize = 16 * PREFILL_GEMM_TN;

/// `bias = 1` adds the per-column `Bias` vector (binding 4) to every output
/// before the `beta` residual — the audio tower's linears ship with biases and
/// a plain GEMM silently drops them.
///
/// `ldc` = C row stride in elements; `bsa`/`bsb` = per-batch operand strides in
/// **words** (the shader indexes `array<u32>` directly) and `bsc` = the per-batch
/// C stride in **elements** (the epilogue divides the flat C index by 2).  Batch
/// index is `wgid.z`; it is 1 for the plain GEMMs, where the units cannot show,
/// and 28 for the audio tower's attention — where a `bsc` given as words shifted
/// every block but the first by half a block.  `transb=1` reads W as a [k, n] f16
/// matrix instead of [n, k] (attention AV: V is [cur, d]).
///
/// `lda` is the A row stride in elements — `k` everywhere except the slabbed AV,
/// whose A operand (a score slab) is `T` wide while the tile it sweeps is
/// narrower.  `row0` shifts the B operand's rows (K or V rows = key positions)
/// and, on the two causal variants, the diagonal they test against.
///
/// The tile geometry comes from the `PREFILL_GEMM_*` constants above; callers
/// must pad `m`, `n` and the operand row strides with those same values.
pub fn prefill_gemm(transb: bool, beta: bool) -> String {
    prefill_gemm_impl(transb, beta, false, false, false, false, false, false)
}

/// [`prefill_gemm`] with an **int8-resident** W operand (the M3 decoder
/// weights).  Only the B side changes: a thread's four weight rows are read as
/// u32 words (4 int8 per word, `gd.k/4` words per row — the same four-word
/// loads the f16 form issues), unpacked, scaled by `S8[row]`, rounded to f16 by
/// `pack2x16float` (the GEMV's own rounding, and the host dequant's), and
/// stored into the same f32 chunk tile the compute loop consumes.  The A tile,
/// the compute loop and the epilogue are untouched, so the accumulation order
/// is the f16 GEMM's.
///
/// The byte position within a word is `ck % 4` — a per-thread constant, since
/// `kx` steps by `BK` (a multiple of 4) and the chunk's `ck` is fixed — and the
/// four scales are hoisted out of the k loop entirely.
///
/// Restricted to the plain weight form (`transb = 0`, no bias, no causal
/// skip): the attention GEMMs stay f16 and the bias form is unused by the
/// decoder.  Requires the v4 chunk tile (the default; `QASR_GEMM_V4=0` is the
/// scalar debug tile and does not have an int8 form).
pub fn prefill_gemm_w8(beta: bool) -> String {
    prefill_gemm_impl(false, beta, false, false, false, true, false, false)
}

/// [`prefill_gemm`] with the causal-attention tile skip: a tile wholly above the
/// diagonal (`n0 + row0 > m0 + BM - 1`, `row0` = the key offset of the score
/// slab) is entirely masked by the causal softmax, which never reads columns
/// past `row + 1`, so skipping it is bit-identical — it just stops writing ~half
/// of the `s × cur` score matrix.
pub fn prefill_gemm_causal() -> String {
    prefill_gemm_impl(false, false, false, true, false, false, false, false)
}

/// [`prefill_gemm_bias`]-style AV form (`transb = 1`) with the causal *k* bound:
/// the A operand is the softmax output, whose columns past `row + 1` are exactly
/// zero, so a row block only has to sweep `k < m0 + BM - row0` (`row0` = the
/// slab's key offset; zero on the flat path).  Skipping exact zeros from a sum is
/// bit-identical.
pub fn prefill_gemm_causal_av() -> String {
    prefill_gemm_impl(true, false, false, false, true, false, false, false)
}

/// As [`prefill_gemm`], plus the per-column bias add (binding 4).  A separate
/// entry point rather than a flag on the shared one: the binding must be
/// *declared* only for the variants that read it, and a declared-but-unbound
/// binding fails pipeline validation even when the read is dead code.
pub fn prefill_gemm_bias(transb: bool, beta: bool) -> String {
    prefill_gemm_impl(transb, beta, true, false, false, false, false, false)
}

/// Bias GEMM whose C tile is f32. Hidden states in this model exceed f16 at
/// the last SANM layer, so the residual epilogue keeps that tensor in f32.
/// A and W stay packed f16. `ldc` and `bsc` are still element strides.
pub fn prefill_gemm_bias_f32c(beta: bool) -> String {
    prefill_gemm_impl(false, beta, true, false, false, false, true, false)
}

/// No-bias GEMM that stores C as f32. Attention scores need the f32
/// accumulator; packing them to f16 before softmax can flip a close token.
pub fn prefill_gemm_f32c(transb: bool) -> String {
    prefill_gemm_impl(transb, false, false, false, false, false, true, false)
}

/// Bias GEMM that reads A as f32 and writes C as f16. Layer-norm output stays
/// f32 so a long clip does not quantize every linear's input.
pub fn prefill_gemm_bias_f32a() -> String {
    prefill_gemm_impl(false, false, true, false, false, false, false, true)
}

/// `QASR_GEMM_V4` -- the prefill/encoder GEMM's shared-memory tile: the default
/// (`1`) is the k-major 16-byte chunk tile, `0` restores the scalar per-(row, k)
/// tile with its row permutation.  Generator-side, read once per pipeline, so it
/// exists to re-run the A/B rather than to be set in production.  See the note in
/// [`prefill_gemm_impl`] for the measurements.
fn gemm_v4() -> bool {
    std::env::var("QASR_GEMM_V4").map(|s| s != "0").unwrap_or(true)
}

/// `QASR_GEMM_UNROLL` -- the k-unroll factor for the prefill/encoder GEMM.
/// `1` is the runtime q-loop (everything before `8x8DU*`), `4` is shipped;
/// anything not dividing `BK` falls back to `1`.  A generator-side switch, read
/// once per pipeline creation, so it exists to re-run the A/B, not to be set in
/// production.  See the unrolling note in [`prefill_gemm_impl`].
fn gemm_unroll() -> usize {
    std::env::var("QASR_GEMM_UNROLL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

fn prefill_gemm_impl(
    transb: bool,
    beta: bool,
    bias: bool,
    causal_skip: bool,
    causal_k: bool,
    w8: bool,
    c_f32: bool,
    a_f32: bool,
) -> String {
    let tm = PREFILL_GEMM_TM;
    let tn = PREFILL_GEMM_TN;
    let bk = PREFILL_GEMM_BK;
    let bm = 16 * tm;
    let bn = 16 * tn;
    let pad = bk + 1;
    let n_as = bm * bk / 256;
    let n_bs = bn * bk / 256;

    // ---- the vec4 (chunk) tile -------------------------------------------
    //
    // `gemm_bench`'s `V` column: the tile is stored **k-major in 16-byte
    // chunks** instead of one f32 per (row, k), so a thread's eight A values
    // and eight B values each become two `vec4` shared loads instead of eight
    // scalar ones.  `fma_probe` prices the difference with everything else held
    // fixed -- 16 scalar loads per 64 FMAs and 4 `vec4` loads per 80, moving the
    // same two kilobytes per warp per iteration -- at 2.66 against 3.80 TFLOP/s,
    // and the real GEMM sat at the 2.66 arm before this.  Across `gemm_bench`'s
    // ten shapes the change is **+26%** (mean 2.31 -> 2.91 TFLOP/s).
    //
    // A chunk is four rows at one k.  For A, chunk `q*AQ + r/4` holds rows
    // `4*(r/4) .. +3` at k = `q`, and a thread wants rows `8*ty .. +7` -- chunks
    // `q*AQ + 2*ty` and `+1`.  Its address depends only on `ty`, so the eight
    // lanes of a phase all want the same sixteen bytes: a broadcast.
    //
    // For B the same layout would have the sixteen `tx` lanes stride eight
    // words and land on two bank groups per phase, so the row *groups* are
    // permuted: chunk `q*BQ + pos(g, s)` holds `8*g + 4*s .. +3` with
    // `pos(g, s) = (g & 7) + 8*s + 16*(g >> 3)` -- the same job the scalar
    // form's `16*j + tx` permutation does, for the same reason.
    //
    // A `f32` array cannot be read as `vec4`, so the tile is *declared*
    // `array<vec4<f32>, ..>` and every store writes a whole chunk.  That forces
    // the load stage to gather four rows at one k per thread, which reads each
    // global word twice (both k of a word now have their own thread).  These
    // GEMMs are compute-bound by two orders of magnitude on every shape this
    // pipeline runs -- `m2304 k1024 n4096` is 1073 FLOP/byte -- so the trade is
    // free here and would not be for a memory-bound shape.
    //
    // Values, accumulation order and FMA count are untouched: bit-identical,
    // and `gemm_bench` checks that against a CPU matmul on every variant.
    //
    // The other operand of a `transb = 1` GEMM (the attention `A@V`, and the
    // encoder conv stem's im2col operand) is stored `[k][n]`, so four
    // consecutive `n` at one k is *two* words whose half-word parity alternates
    // per row instead of being one flag per thread.  That makes its chunk build a
    // different expression -- two registers and four `halve`s -- rather than a
    // different constant, and it reads half the words the row-major path does.
    //
    // Derived for `TM = TN = 8` (512 chunks over 256 threads is two per thread).
    let v4 = gemm_v4();
    let (aq, bq) = (bm / 4, bn / 4);
    let cstep = bm * bk / 4 / 2;
    if w8 {
        assert!(!transb && !bias && !causal_skip && !causal_k,
            "prefill_gemm_w8 supports only the plain weight operand (the attention GEMMs stay f16)");
        assert!(v4, "prefill_gemm_w8 requires the v4 chunk tile (QASR_GEMM_V4 != 0)");
    }
    assert_eq!(tm, 8, "the vec4 chunk store is derived for TM = 8");
    assert_eq!(tn, 8, "the vec4 chunk store is derived for TN = 8");
    assert_eq!(bk % 16, 0, "the vec4 chunk store assumes 8 k per chunk half");
    assert_eq!(n_as, 8, "the v4 A store assumes two chunks per thread");
    // Both B layouts move eight rows per thread; the k-major one needs four
    // words to do it (two per chunk) and the row-major one eight.
    assert_eq!(n_bs, 8, "the v4 B store assumes two chunks per thread");
    let mut s = String::new();
    let cty = if c_f32 { "f32" } else { "u32" };
    let aty = if a_f32 { "f32" } else { "u32" };
    s.push_str(&format!(
        "struct GDims {{ m: u32, n: u32, k: u32, ldc: u32, bsa: u32, bsb: u32, bsc: u32, beta: u32, row0: u32, lda: u32 }};\n\
         @group(0) @binding(0) var<storage, read>       A: array<{aty}>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<{cty}>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n"
    ));
    if bias {
        s.push_str("@group(0) @binding(4) var<storage, read> Bias: array<u32>;\n");
    }
    if w8 {
        s.push_str("@group(0) @binding(4) var<storage, read> S8: array<f32>;\n");
    }
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\n\
         const PAD: u32 = {pad}u;\nconst TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\n\
         const AQ: u32 = {aq}u;\nconst BQ: u32 = {bq}u;\nconst CSTEP: u32 = {cstep}u;\n\
         const TRANSB: u32 = {}u;\nconst BETA: u32 = {}u;\nconst BIAS: u32 = {}u;\n\
         const CAUSAL: u32 = {}u;\nconst CAUSAL_K: u32 = {}u;\n",
        u32::from(transb),
        u32::from(beta),
        u32::from(bias),
        u32::from(causal_skip),
        u32::from(causal_k),
    ));
    if v4 {
        s.push_str(&format!("var<workgroup> A4: array<vec4<f32>, {}>;\n", bm * bk / 4));
        s.push_str(&format!("var<workgroup> B4: array<vec4<f32>, {}>;\n", bn * bk / 4));
    } else {
        s.push_str(&format!("var<workgroup> As: array<f32, {}>;\n", bm * pad));
        s.push_str(&format!("var<workgroup> Bs: array<f32, {}>;\n", bn * pad));
    }
    s.push_str(
        "fn halve(w: u32, odd: bool) -> f32 {\n\
         \x20 let p = unpack2x16float(w);\n\
         \x20 return select(p.x, p.y, odd);\n\
         }\n",
    );
    if w8 {
        // The int8 unpack and the f16-rounded dequant — `q8` takes the
        // left-shift that brings a byte to the top of the word, `wq` the byte
        // *index* (`24 - 8*by` is its shift) and produces the same
        // `f16(q * s)` value the host dequant produced for M2 (pack2x16float
        // rounds to f16 nearest-even).
        s.push_str(
            "fn q8(w: u32, sh: u32) -> f32 {\n\
             \x20 return f32((bitcast<i32>(w << sh)) >> 24u);\n\
             }\n\
             fn wq(w: u32, by: u32, sc: f32) -> f32 {\n\
             \x20 return unpack2x16float(pack2x16float(vec2<f32>(q8(w, 24u - 8u * by) * sc, 0.0))).x;\n\
             }\n",
        );
    }

    s.push_str(&format!(
        "@compute @workgroup_size(16, 16)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {{\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / {kkd}u;\n\
         let alda = gd.lda{alda_div};\n\
         let abase = wid.z * gd.bsa;\n let wb = wid.z * gd.bsb;\n\
         let cbase = wid.z * gd.bsc;\n\
         if (CAUSAL == 1u && n0 + gd.row0 > m0 + BM - 1u) {{ return; }}\n\
         let klim = select(gd.k, min(gd.k, max(m0 + BM, gd.row0) - gd.row0), CAUSAL_K == 1u);\n",
        kkd = if w8 { 4 } else { 2 },
        alda_div = if a_f32 { "" } else { " / 2u" },
    ));
    if v4 {
        // The chunk index and half-word parity every store path is written in
        // terms of: thread (tx, ty) owns chunks `c0` and `c0 + CSTEP`, i.e.
        // `(k, rg) = (c0/AQ, c0%AQ)` and `(k+8, rg)`.  Both are the same four
        // rows at two `k`, so one parity flag covers all eight global words.
        // `bop`/`bog`/`bos` invert the B tile's `pos(g, s)` for chunk `c0`.
        s.push_str(
            " let c0 = ty * 16u + tx;\n\
             \x20 let crg = c0 % AQ;\n let ck = c0 / AQ;\n\
             \x20 let cvodd = ((ck & 1u) == 1u);\n\
             \x20 let bop = c0 % BQ;\n\
             \x20 let bog = ((bop >> 4u) << 3u) | (bop & 7u);\n\
             \x20 let bos = (bop >> 3u) & 1u;\n",
        );
        if w8 {
            // The int8 B tile: the thread's four weight rows (the same rows for
            // `c0` and `c0 + CSTEP`), their scales, and the byte position —
            // `kx` steps by `BK` (a multiple of 4), so `ck % 4` is the whole
            // k loop's byte-in-word index.
            s.push_str(
                " let brow = gd.row0 + n0 + 8u * bog + 4u * bos;\n\
                 \x20 let s0 = S8[brow]; let s1 = S8[brow + 1u]; let s2 = S8[brow + 2u]; let s3 = S8[brow + 3u];\n\
                 \x20 let by = ck % 4u;\n",
            );
        }
    }

    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }

    let load_as = |kx: &str| -> String {
        let mut t = String::new();
        if v4 {
            for half in 0..2 {
                let kw = format!("({kx} + ck + {}u)", half * 8);
                let mut comps = String::new();
                for e in 0..4 {
                    if a_f32 {
                        comps.push_str(&format!(
                            "\x20 A[abase + (m0 + 4u * crg + {e}u) * alda + {kw}],"
                        ));
                    } else {
                        comps.push_str(&format!(
                            "\x20 halve(A[abase + (m0 + 4u * crg + {e}u) * alda + {kw} / 2u], cvodd),"
                        ));
                    }
                }
                t.push_str(&format!(
                    "  A4[c0{}] = vec4<f32>({comps});\n",
                    if half == 0 { String::new() } else { " + CSTEP".into() }
                ));
            }
            return t;
        }
        for e in 0..n_as {
            if a_f32 {
                t.push_str(&format!(
                    "  As[(ty + {}u) * PAD + tx] = A[abase + (m0 + ty + {}u) * alda + ({kx} + tx)];\n",
                    e * 16, e * 16
                ));
            } else {
                t.push_str(&format!(
                    "  As[(ty + {}u) * PAD + tx] = halve(A[abase + (m0 + ty + {}u) * alda + ({kx} + tx) / 2u], (tx & 1u) == 1u);\n",
                    e * 16, e * 16
                ));
            }
        }
        t
    };
    // ---- B tile row permutation -------------------------------------------
    //
    // The compute loop reads `Bs[(tx * TN + j) * PAD + q]`, so for a fixed `j`
    // the sixteen `tx` lanes of a warp stride `TN * PAD = 8 * 17 = 136` words.
    // `136 mod 32 = 8`, so those lanes land on banks `{0, 8, 16, 24}` -- four
    // banks for sixteen lanes, a **4-way shared-memory conflict** costing four
    // LDS cycles per load.  A k-step issues 8 B loads and 64 FMAs, i.e. 32
    // banks-cycles of shared traffic against 16 issue cycles for the FMAs
    // (4 warps/SM/clock): shared memory, not FLOPs, set the rate.
    //
    // No padding fixes it: `8 * PAD mod 32` is always a multiple of 8, so the
    // sixteen lanes can never reach more than four banks.  Permuting the tile
    // does.  Value `B[n0 + 8*tx + j]` is stored in slot `16*j + tx` instead of
    // slot `8*tx + j`, and the compute read becomes
    // `Bs[(16*j + tx) * PAD + q]`.  `16 * 17 = 272 mod 32 = 16` moves `j` in
    // whole sixteen-word blocks while `tx` walks `17*tx mod 32` -- sixteen
    // distinct banks.  The store's row `v = ty + 16e` therefore goes to slot
    // `16*(v % tn) + v / tn` -- powers of two, so it folds to a mask and a shift.
    //
    // Each thread still reads the same eight values, still accumulates them in
    // the same order, and issues the same FMAs: bit-identical by construction.
    // `gemm_bench`'s `P` column prices it at +10% to +29% per shape.
    let bslot = |e: usize| -> String {
        format!(
            "(16u * ((ty + {off}u) % {tn}u) + (ty + {off}u) / {tn}u)",
            off = e * 16
        )
    };
    let load_bs = |kx: &str| -> String {
        let mut t = String::new();
        if v4 {
            if transb {
                // The k-major operand: four consecutive `n` at one k is *two*
                // words, and their half-word parity alternates per row rather
                // than being one flag per thread.  So this chunk is built from
                // two registers with the parity spelled out, and it reads half
                // the words the row-major path does.
                for half in 0..2 {
                    let base = format!(
                        "wb + (gd.row0 + {kx} + ck{}) * (gd.n / 2u)",
                        if half == 0 { String::new() } else { " + 8u".into() }
                    );
                    t.push_str(&format!(
                        "  B4[c0{cs}] = vec4<f32>(halve(W[{base} + {wo}], false), halve(W[{base} + {wo}], true), halve(W[{base} + {wo} + 1u], false), halve(W[{base} + {wo} + 1u], true));\n",
                        cs = if half == 0 { String::new() } else { " + CSTEP".into() },
                        wo = "(n0 + 8u * bog + 4u * bos) / 2u"
                    ));
                }
                return t;
            }
            if w8 {
                // The same four rows at one k, but each value is now a byte of
                // its row's word `(kx + ck + 8*half) / 4` — the byte position
                // itself is the per-thread constant `by`.
                for half in 0..2 {
                    let wo = format!("({kx} + ck + {}u) / 4u", half * 8);
                    let mut comps = String::new();
                    for e in 0..4 {
                        comps.push_str(&format!(
                            "\x20 wq(W[wb + (gd.row0 + n0 + 8u * bog + 4u * bos + {e}u) * kk + {wo}], by, s{e}),"
                        ));
                    }
                    t.push_str(&format!(
                        "  B4[c0{}] = vec4<f32>({comps});\n",
                        if half == 0 { String::new() } else { " + CSTEP".into() }
                    ));
                }
                return t;
            }
            for half in 0..2 {
                let kw = format!("({kx} + ck + {}u)", half * 8);
                let mut comps = String::new();
                for e in 0..4 {
                    comps.push_str(&format!(
                        "\x20 halve(W[wb + (gd.row0 + n0 + 8u * bog + 4u * bos + {e}u) * kk + {kw} / 2u], cvodd),"
                    ));
                }
                t.push_str(&format!(
                    "  B4[c0{}] = vec4<f32>({comps});\n",
                    if half == 0 { String::new() } else { " + CSTEP".into() }
                ));
            }
            return t;
        }
        if !transb {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "  Bs[{slot} * PAD + tx] = halve(W[wb + (gd.row0 + n0 + ty + {row}u) * kk + ({kx} + tx) / 2u], (tx & 1u) == 1u);\n",
                    slot = bslot(e),
                    row = e * 16
                ));
            }
        } else {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "  Bs[{slot} * PAD + tx] = halve(W[wb + (gd.row0 + {kx} + tx) * (gd.n / 2u) + (n0 + ty + {row}u) / 2u], ((n0 + ty + {row}u) & 1u) == 1u);\n",
                    slot = bslot(e),
                    row = e * 16
                ));
            }
        }
        t
    };
    let pf_as = |kx: &str| -> String {
        let mut t = String::new();
        if v4 {
            for half in 0..2 {
                let kw = format!("({kx} + ck + {}u)", half * 8);
                for e in 0..4 {
                    if a_f32 {
                        t.push_str(&format!(
                            "   pfa{} = A[abase + (m0 + 4u * crg + {e}u) * alda + {kw}];\n",
                            half * 4 + e
                        ));
                    } else {
                        t.push_str(&format!(
                            "   pfa{} = A[abase + (m0 + 4u * crg + {e}u) * alda + {kw} / 2u];\n",
                            half * 4 + e
                        ));
                    }
                }
            }
            return t;
        }
        for e in 0..n_as {
            if a_f32 {
                t.push_str(&format!(
                    "   pfa{e} = A[abase + (m0 + ty + {}u) * alda + ({kx} + tx)];\n", e * 16
                ));
            } else {
                t.push_str(&format!(
                    "   pfa{e} = A[abase + (m0 + ty + {}u) * alda + ({kx} + tx) / 2u];\n", e * 16
                ));
            }
        }
        t
    };
    let pf_bs = |kx: &str| -> String {
        let mut t = String::new();
        if v4 {
            if transb {
                // Two words per chunk, four registers total.
                for half in 0..2 {
                    let base = format!(
                        "wb + (gd.row0 + {kx} + ck{}) * (gd.n / 2u)",
                        if half == 0 { String::new() } else { " + 8u".into() }
                    );
                    t.push_str(&format!(
                        "   pfb{} = W[{base} + {wo}];\n   pfb{} = W[{base} + {wo} + 1u];\n",
                        half * 2,
                        half * 2 + 1,
                        wo = "(n0 + 8u * bog + 4u * bos) / 2u"
                    ));
                }
                return t;
            }
            if w8 {
                for half in 0..2 {
                    let wo = format!("({kx} + ck + {}u) / 4u", half * 8);
                    for e in 0..4 {
                        t.push_str(&format!(
                            "   pfb{} = W[wb + (gd.row0 + n0 + 8u * bog + 4u * bos + {e}u) * kk + {wo}];\n",
                            half * 4 + e
                        ));
                    }
                }
                return t;
            }
            for half in 0..2 {
                let kw = format!("({kx} + ck + {}u)", half * 8);
                for e in 0..4 {
                    t.push_str(&format!(
                        "   pfb{} = W[wb + (gd.row0 + n0 + 8u * bog + 4u * bos + {e}u) * kk + {kw} / 2u];\n",
                        half * 4 + e
                    ));
                }
            }
            return t;
        }
        if !transb {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "   pfb{e} = W[wb + (gd.row0 + n0 + ty + {}u) * kk + ({kx} + tx) / 2u];\n", e * 16
                ));
            }
        } else {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "   pfb{e} = W[wb + (gd.row0 + {kx} + tx) * (gd.n / 2u) + (n0 + ty + {}u) / 2u];\n", e * 16
                ));
            }
        }
        t
    };
    let pf_decl = || {
        let mut t = String::new();
        let pty = if a_f32 { "f32" } else { "u32" };
        let zero = if a_f32 { "0.0" } else { "0u" };
        for e in 0..n_as {
            t.push_str(&format!("  var pfa{e}: {pty} = {zero};\n"));
        }
        for e in 0..n_bs {
            t.push_str(&format!("  var pfb{e}: u32 = 0u;\n"));
        }
        t
    };
    let store_as = || {
        let mut t = String::new();
        if v4 {
            for half in 0..2 {
                let mut comps = String::new();
                for e in 0..4 {
                    if a_f32 {
                        comps.push_str(&format!("\x20 pfa{},", half * 4 + e));
                    } else {
                        comps.push_str(&format!("\x20 halve(pfa{}, cvodd),", half * 4 + e));
                    }
                }
                t.push_str(&format!(
                    "   A4[c0{}] = vec4<f32>({comps});\n",
                    if half == 0 { String::new() } else { " + CSTEP".into() }
                ));
            }
            return t;
        }
        for e in 0..n_as {
            if a_f32 {
                t.push_str(&format!(
                    "   As[(ty + {}u) * PAD + tx] = pfa{e};\n", e * 16
                ));
            } else {
                t.push_str(&format!(
                    "   As[(ty + {}u) * PAD + tx] = halve(pfa{e}, (tx & 1u) == 1u);\n", e * 16
                ));
            }
        }
        t
    };
    let store_bs = || {
        let mut t = String::new();
        if v4 {
            if transb {
                for half in 0..2 {
                    t.push_str(&format!(
                        "   B4[c0{cs}] = vec4<f32>(halve(pfb{r}, false), halve(pfb{r}, true), halve(pfb{r2}, false), halve(pfb{r2}, true));\n",
                        cs = if half == 0 { String::new() } else { " + CSTEP".into() },
                        r = half * 2,
                        r2 = half * 2 + 1
                    ));
                }
                return t;
            }
            if w8 {
                for half in 0..2 {
                    let mut comps = String::new();
                    for e in 0..4 {
                        comps.push_str(&format!("\x20 wq(pfb{}, by, s{e}),", half * 4 + e));
                    }
                    t.push_str(&format!(
                        "   B4[c0{}] = vec4<f32>({comps});\n",
                        if half == 0 { String::new() } else { " + CSTEP".into() }
                    ));
                }
                return t;
            }
            for half in 0..2 {
                let mut comps = String::new();
                for e in 0..4 {
                    comps.push_str(&format!("\x20 halve(pfb{}, cvodd),", half * 4 + e));
                }
                t.push_str(&format!(
                    "   B4[c0{}] = vec4<f32>({comps});\n",
                    if half == 0 { String::new() } else { " + CSTEP".into() }
                ));
            }
            return t;
        }
        for e in 0..n_bs {
            let sel = if transb {
                format!("((n0 + ty + {}u) & 1u) == 1u", e * 16)
            } else {
                "(tx & 1u) == 1u".to_string()
            };
            t.push_str(&format!(
                "   Bs[{slot} * PAD + tx] = halve(pfb{e}, {sel});\n",
                slot = bslot(e)
            ));
        }
        t
    };
    // ---- k-step unrolling -------------------------------------------------
    //
    // The q-loop was a runtime `loop { if (q >= BK) break; ... q += 1 }`.  Four
    // steps at a time is worth **5-10% per shape** over the runtime form -- and,
    // measured separately, so is any other factor: `gemm_bench`'s U2/U4/U8/U16
    // columns land at 2.29/2.33/2.30/2.36 mean TFLOP/s, i.e. inside each other's
    // noise.  So the win is not "more unrolling", it is letting the compiler see
    // past one k-step boundary at all: it can hoist the next step's shared loads
    // above the current step's FMAs, which a loop that may exit at any iteration
    // forbids.  Four is the smallest factor that buys it, so four it is.
    //
    // It is *only* worth anything together with the B permutation above: the
    // same unroll on the unpermuted tile measured 1.89 against 2.02
    // (`8x8DU16` vs `8x8D`), because with four lanes per bank there is nothing
    // to overlap.  The accumulation order per output is untouched -- each
    // `c{i}{j}` still takes its k contributions in ascending order -- so this is
    // bit-identical.
    let mut unroll_q = gemm_unroll();
    if unroll_q < 2 {
        unroll_q = 1;
    } else if bk % unroll_q != 0 {
        eprintln!("QASR_GEMM_UNROLL={unroll_q} does not divide BK={bk}; using 1");
        unroll_q = 1;
    }
    let compute = || {
        let mut t = String::new();
        // The v4 read: two chunks per operand -- rows `8*ty .. +7` of A and
        // `8*tx .. +7` of B -- with the same values, the same FMA count and the
        // same ascending-k accumulation order as the scalar form.
        let v4_read = |t: &mut String, idx: &str, tag: &str| {
            t.push_str(&format!("   let av{tag} = A4[{idx} * AQ + ty * 2u];\n"));
            t.push_str(&format!("   let aw{tag} = A4[{idx} * AQ + ty * 2u + 1u];\n"));
            t.push_str(&format!(
                "   let bv{tag} = B4[{idx} * BQ + (tx & 7u) + 16u * (tx >> 3u)];\n"
            ));
            t.push_str(&format!(
                "   let bw{tag} = B4[{idx} * BQ + (tx & 7u) + 8u + 16u * (tx >> 3u)];\n"
            ));
        };
        let v4_fma = |t: &mut String, tag: &str| {
            for i in 0..tm {
                let a = if i < 4 {
                    format!("av{tag}[{i}]")
                } else {
                    format!("aw{tag}[{}]", i - 4)
                };
                for j in 0..tn {
                    let b = if j < 4 {
                        format!("bv{tag}[{j}]")
                    } else {
                        format!("bw{tag}[{}]", j - 4)
                    };
                    t.push_str(&format!("   c{i}{j} = c{i}{j} + {a} * {b};\n"));
                }
            }
        };
        if unroll_q == 1 {
            t.push_str("  var q: u32 = 0u;\n  loop {\n   if (q >= BK) { break; }\n");
            if v4 {
                v4_read(&mut t, "q", "");
                v4_fma(&mut t, "");
            } else {
                for i in 0..tm {
                    t.push_str(&format!(
                        "   let a{i} = As[(ty * {tm}u + {i}u) * PAD + q];\n"
                    ));
                }
                for j in 0..tn {
                    t.push_str(&format!(
                        "   let b{j} = Bs[(16u * {j}u + tx) * PAD + q];\n"
                    ));
                }
                for i in 0..tm {
                    for j in 0..tn {
                        t.push_str(&format!("   c{i}{j} = c{i}{j} + a{i} * b{j};\n"));
                    }
                }
            }
            t.push_str("   q = q + 1u;\n  }\n");
            return t;
        }
        t.push_str("  var q0: u32 = 0u;\n  loop {\n   if (q0 >= BK) { break; }\n");
        for u in 0..unroll_q {
            if v4 {
                v4_read(&mut t, &format!("(q0 + {u}u)"), &format!("_{u}"));
                v4_fma(&mut t, &format!("_{u}"));
                continue;
            }
            for i in 0..tm {
                t.push_str(&format!(
                    "   let a{i}_{u} = As[(ty * {tm}u + {i}u) * PAD + (q0 + {u}u)];\n"
                ));
            }
            for j in 0..tn {
                t.push_str(&format!(
                    "   let b{j}_{u} = Bs[(16u * {j}u + tx) * PAD + (q0 + {u}u)];\n"
                ));
            }
            for i in 0..tm {
                for j in 0..tn {
                    t.push_str(&format!(
                        "   c{i}{j} = c{i}{j} + a{i}_{u} * b{j}_{u};\n"
                    ));
                }
            }
        }
        t.push_str(&format!("   q0 = q0 + {unroll_q}u;\n  }}\n"));
        t
    };

    s.push_str(" var k0: u32 = 0u;\n");
    s.push_str(&load_as("k0"));
    s.push_str(&load_bs("k0"));
    s.push_str(" workgroupBarrier();\n");
    s.push_str(" loop {\n  if (k0 >= klim) { break; }\n");
    s.push_str("  let kn = k0 + BK;\n");
    s.push_str(&pf_decl());
    s.push_str("  if (kn < klim) {\n");
    s.push_str(&pf_as("kn"));
    s.push_str(&pf_bs("kn"));
    s.push_str("  }\n");
    s.push_str(&compute());
    s.push_str("  workgroupBarrier();\n");
    s.push_str("  if (kn < klim) {\n");
    s.push_str(&store_as());
    s.push_str(&store_bs());
    s.push_str("  }\n  workgroupBarrier();\n");
    s.push_str("  k0 = kn;\n }\n");

    for i in 0..tm {
        s.push_str(&format!("let row{i} = m0 + ty * {tm}u + {i}u;\n"));
    }
    for i in 0..tm {
        for e in 0..tn / 2 {
            let je = 2 * e;
            let jo = je + 1;
            s.push_str(&format!(
                "var v{i}{e} = vec2<f32>(c{i}{je}, c{i}{jo});\n"
            ));
            if bias {
                s.push_str(&format!(
                    "v{i}{e} = v{i}{e} + unpack2x16float(Bias[(n0 + tx * {tn}u + {je}u) / 2u]);\n"
                ));
            }
            if c_f32 {
                s.push_str(&format!(
                    "let e{i}_{e} = cbase + row{i} * gd.ldc + n0 + tx * {tn}u + {je}u;\n"
                ));
                s.push_str(&format!(
                    "if (BETA == 1u) {{ v{i}{e} = v{i}{e} + vec2<f32>(C[e{i}_{e}], C[e{i}_{e} + 1u]); }}\n"
                ));
                s.push_str(&format!(
                    "C[e{i}_{e}] = v{i}{e}.x;\nC[e{i}_{e} + 1u] = v{i}{e}.y;\n"
                ));
            } else {
                s.push_str(&format!(
                    "let we{i}_{e} = (cbase + row{i} * gd.ldc + n0 + tx * {tn}u + {je}u) / 2u;\n"
                ));
                s.push_str(&format!(
                    "if (BETA == 1u) {{\n  let old = unpack2x16float(C[we{i}_{e}]);\n  v{i}{e} = v{i}{e} + old;\n}}\n"
                ));
                s.push_str(&format!(
                    "C[we{i}_{e}] = pack2x16float(v{i}{e});\n"
                ));
            }
        }
    }
    s.push_str("}\n");
    s
}
