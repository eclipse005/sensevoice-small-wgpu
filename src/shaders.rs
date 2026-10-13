//! Small compute shaders. The GEMM itself lives in `gemm_f32`.

pub fn layernorm() -> String {
    // One row is 128–140 vec4s. Four rows share a 128-thread workgroup whose
    // 32-lane row groups stage the row into shared memory with coalesced
    // loads, then **one lane** per row accumulates it in element order — the
    // exact sequential association the one-thread-per-row form used, so the
    // result is bit-identical — while the other 31 lanes go idle for those two
    // loops and rejoin for the elementwise write. Staging is what buys the
    // parallelism: without it every row is a chain of scalar-latency loads
    // with one warp per SM to hide them.
    "struct LnCfg { d: u32, eps: f32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read>       Src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       Wgt: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       Bia: array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             cfg: LnCfg;
@group(0) @binding(4) var<storage, read_write> Dst: array<vec4<f32>>;

const ROWQ: u32 = 140u;
var<workgroup> tile: array<vec4<f32>, 560>;
var<workgroup> stats: array<f32, 8>;

@compute @workgroup_size(128)
fn layernorm(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 4u + g;
    let d = cfg.d;
    let dv = d / 4u;
    let base = row * dv;
    for (var c: u32 = lane; c < dv; c = c + 32u) {
        tile[g * ROWQ + c] = Src[base + c];
    }
    workgroupBarrier();
    if (lane == 0u) {
        var mean = 0.0;
        let full = dv / 4u;
        for (var i: u32 = 0u; i < full; i = i + 1u) {
            let c = i * 4u;
            let v0 = tile[g * ROWQ + c];
            let v1 = tile[g * ROWQ + c + 1u];
            let v2 = tile[g * ROWQ + c + 2u];
            let v3 = tile[g * ROWQ + c + 3u];
            mean = mean + v0.x;
            mean = mean + v0.y;
            mean = mean + v0.z;
            mean = mean + v0.w;
            mean = mean + v1.x;
            mean = mean + v1.y;
            mean = mean + v1.z;
            mean = mean + v1.w;
            mean = mean + v2.x;
            mean = mean + v2.y;
            mean = mean + v2.z;
            mean = mean + v2.w;
            mean = mean + v3.x;
            mean = mean + v3.y;
            mean = mean + v3.z;
            mean = mean + v3.w;
        }
        for (var c: u32 = full * 4u; c < dv; c = c + 1u) {
            let v = tile[g * ROWQ + c];
            mean = mean + v.x;
            mean = mean + v.y;
            mean = mean + v.z;
            mean = mean + v.w;
        }
        stats[g * 2u] = mean / f32(d);
    }
    workgroupBarrier();
    let mean = stats[g * 2u];
    if (lane == 0u) {
        var var_ = 0.0;
        let full = dv / 4u;
        for (var i: u32 = 0u; i < full; i = i + 1u) {
            let c = i * 4u;
            let z0 = tile[g * ROWQ + c] - vec4<f32>(mean);
            let z1 = tile[g * ROWQ + c + 1u] - vec4<f32>(mean);
            let z2 = tile[g * ROWQ + c + 2u] - vec4<f32>(mean);
            let z3 = tile[g * ROWQ + c + 3u] - vec4<f32>(mean);
            var_ = var_ + z0.x * z0.x;
            var_ = var_ + z0.y * z0.y;
            var_ = var_ + z0.z * z0.z;
            var_ = var_ + z0.w * z0.w;
            var_ = var_ + z1.x * z1.x;
            var_ = var_ + z1.y * z1.y;
            var_ = var_ + z1.z * z1.z;
            var_ = var_ + z1.w * z1.w;
            var_ = var_ + z2.x * z2.x;
            var_ = var_ + z2.y * z2.y;
            var_ = var_ + z2.z * z2.z;
            var_ = var_ + z2.w * z2.w;
            var_ = var_ + z3.x * z3.x;
            var_ = var_ + z3.y * z3.y;
            var_ = var_ + z3.z * z3.z;
            var_ = var_ + z3.w * z3.w;
        }
        for (var c: u32 = full * 4u; c < dv; c = c + 1u) {
            let z = tile[g * ROWQ + c] - vec4<f32>(mean);
            var_ = var_ + z.x * z.x;
            var_ = var_ + z.y * z.y;
            var_ = var_ + z.z * z.z;
            var_ = var_ + z.w * z.w;
        }
        stats[g * 2u + 1u] = 1.0 / sqrt(var_ / f32(d) + cfg.eps);
    }
    workgroupBarrier();
    let inv = stats[g * 2u + 1u];
    for (var c: u32 = lane; c < dv; c = c + 32u) {
        Dst[base + c] = (tile[g * ROWQ + c] - vec4<f32>(mean)) * vec4<f32>(inv) * Wgt[c] + Bia[c];
    }
}
"
    .to_string()
}
pub fn split_qkv() -> String {
    "struct SplitCfg { t_pad: u32, t_valid: u32, scale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read> Qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> V: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> cfg: SplitCfg;

@compute @workgroup_size(128, 1, 1)
fn split(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d4 = gid.x;
    let h = gid.y;
    let t = gid.z;
    if (d4 >= 32u || h >= 4u || t >= cfg.t_pad) { return; }
    let dst = (h * cfg.t_pad + t) * 32u + d4;
    if (t >= cfg.t_valid) {
        Q[dst] = vec4<f32>(0.0);
        K[dst] = vec4<f32>(0.0);
        V[dst] = vec4<f32>(0.0);
        return;
    }
    let base = t * 1536u + h * 128u + d4 * 4u;
    Q[dst] = vec4<f32>(
        Qkv[base] * cfg.scale,
        Qkv[base + 1u] * cfg.scale,
        Qkv[base + 2u] * cfg.scale,
        Qkv[base + 3u] * cfg.scale,
    );
    K[dst] = vec4<f32>(Qkv[base + 512u], Qkv[base + 513u], Qkv[base + 514u], Qkv[base + 515u]);
    V[dst] = vec4<f32>(Qkv[base + 1024u], Qkv[base + 1025u], Qkv[base + 1026u], Qkv[base + 1027u]);
}
"
    .to_string()
}

pub fn merge_heads() -> String {
    "struct MergeCfg { t_pad: u32, t_valid: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read> Av: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> Ctx: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> cfg: MergeCfg;

@compute @workgroup_size(128, 1, 1)
fn merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d4 = gid.x;
    let h = gid.y;
    let t = gid.z;
    if (d4 >= 32u || h >= 4u || t >= cfg.t_pad) { return; }
    var val = vec4<f32>(0.0);
    if (t < cfg.t_valid) {
        val = Av[(h * cfg.t_pad + t) * 32u + d4];
    }
    Ctx[t * 128u + h * 32u + d4] = val;
}
"
    .to_string()
}

pub fn fsmn() -> String {
    // `cfg._p0` is the residual mode: 1 adds the layer input (`Src`) to the
    // memory output and stores the sum directly, which is what the separate
    // `add` pass used to do — the same `src + (acc + v)` association, one pass
    // fewer. 0 stores the memory output alone (layer 0).
    "struct FsmnCfg { t_valid: u32, t_pad: u32, mode: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read> Qkv: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<uniform> cfg: FsmnCfg;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> Src: array<f32>;

@compute @workgroup_size(256)
fn fsmn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let t = gid.y;
    let at = t * 512u + c;
    if (c >= 512u || t >= cfg.t_pad) { return; }
    if (t >= cfg.t_valid) {
        Out[at] = select(0.0, Src[at], cfg.mode == 1u);
        return;
    }
    var acc = 0.0;
    for (var k: u32 = 0u; k < 11u; k = k + 1u) {
        let src = i32(t + k) - 5;
        if (src >= 0 && u32(src) < cfg.t_valid) {
            let v = Qkv[u32(src) * 1536u + 1024u + c];
            acc = acc + v * W[c * 11u + k];
        }
    }
    let v = Qkv[t * 1536u + 1024u + c];
    let mem = acc + v;
    Out[at] = select(mem, Src[at] + mem, cfg.mode == 1u);
}
"
    .to_string()
}

pub fn softmax() -> String {
    "struct SoftCfg { t_valid: u32, t_pad: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<uniform> cfg: SoftCfg;
var<workgroup> red: array<f32, 256>;

fn score_at(row: u32, j: u32) -> f32 {
    return S[row * cfg.t_pad + j];
}

@compute @workgroup_size(256)
fn softmax(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let lane = lid.x;
    let n = cfg.t_pad;
    let valid = cfg.t_valid;
    var m = -1.0e30;
    for (var j: u32 = lane; j < valid; j = j + 256u) {
        m = max(m, score_at(row, j));
    }
    red[lane] = m;
    workgroupBarrier();
    for (var off: u32 = 128u; off > 0u; off = off >> 1u) {
        if (lane < off) { red[lane] = max(red[lane], red[lane + off]); }
        workgroupBarrier();
    }
    let maxv = red[0];
    var sum = 0.0;
    for (var j: u32 = lane; j < valid; j = j + 256u) {
        sum = sum + exp(score_at(row, j) - maxv);
    }
    red[lane] = sum;
    workgroupBarrier();
    for (var off: u32 = 128u; off > 0u; off = off >> 1u) {
        if (lane < off) { red[lane] = red[lane] + red[lane + off]; }
        workgroupBarrier();
    }
    let inv = 1.0 / red[0];
    for (var j: u32 = lane; j < n; j = j + 256u) {
        var v = 0.0;
        if (j < valid) { v = exp(score_at(row, j) - maxv) * inv; }
        S[row * n + j] = v;
    }
}
"
    .to_string()
}

/// [`softmax`] with the row staged in shared memory: one coalesced global read
/// and one write instead of three reads and a write. Used for rows up to
/// `SOFT_CACHE` wide; longer rows take the direct kernel. The lane-strided
/// scan and the tree are the same, so the values are unchanged.
pub fn softmax_cached() -> String {
    "struct SoftCfg { t_valid: u32, t_pad: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<uniform> cfg: SoftCfg;
var<workgroup> red: array<f32, 256>;
var<workgroup> row: array<f32, 4096>;

@compute @workgroup_size(256)
fn softmax_cached(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.x;
    let lane = lid.x;
    let n = cfg.t_pad;
    let valid = cfg.t_valid;
    for (var j: u32 = lane; j < n; j = j + 256u) {
        row[j] = S[r * n + j];
    }
    var m = -1.0e30;
    for (var j: u32 = lane; j < valid; j = j + 256u) {
        m = max(m, row[j]);
    }
    red[lane] = m;
    workgroupBarrier();
    for (var off: u32 = 128u; off > 0u; off = off >> 1u) {
        if (lane < off) { red[lane] = max(red[lane], red[lane + off]); }
        workgroupBarrier();
    }
    let maxv = red[0];
    var sum = 0.0;
    for (var j: u32 = lane; j < valid; j = j + 256u) {
        sum = sum + exp(row[j] - maxv);
    }
    red[lane] = sum;
    workgroupBarrier();
    for (var off: u32 = 128u; off > 0u; off = off >> 1u) {
        if (lane < off) { red[lane] = red[lane] + red[lane + off]; }
        workgroupBarrier();
    }
    let inv = 1.0 / red[0];
    for (var j: u32 = lane; j < n; j = j + 256u) {
        var v = 0.0;
        if (j < valid) { v = exp(row[j] - maxv) * inv; }
        S[r * n + j] = v;
    }
}
"
    .to_string()
}

pub fn add() -> String {
    "struct AddCfg { n: u32, mode: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> D: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> cfg: AddCfg;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 4u >= cfg.n) { return; }
    var o = B[i];
    if (cfg.mode == 1u) {
        o = A[i] + B[i];
    }
    D[i] = o;
}
"
    .to_string()
}

pub fn relu() -> String {
    "struct NCfg { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> cfg: NCfg;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i * 4u >= cfg.n) { return; }
    X[i] = max(X[i], vec4<f32>(0.0));
}
"
    .to_string()
}

pub fn argmax() -> String {
    "struct ArgCfg { t_valid: u32, n_vocab: u32, stride: u32, ban: u32 };
@group(0) @binding(0) var<storage, read> Logits: array<f32>;
@group(0) @binding(1) var<uniform> cfg: ArgCfg;
@group(0) @binding(2) var<storage, read_write> Ids: array<u32>;
var<workgroup> bestv: array<f32, 256>;
var<workgroup> besti: array<u32, 256>;
var<workgroup> seen: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let lane = lid.x;
    let n = cfg.n_vocab;
    var bv = -1.0e30;
    var bi = 0u;
    var sn = 0u;
    for (var c: u32 = lane; c < n; c = c + 256u) {
        if (c != cfg.ban) {
        let v = Logits[row * cfg.stride + c];
        if (sn == 0u || v > bv) {
            bv = v;
            bi = c;
            sn = 1u;
        }
        }
    }
    bestv[lane] = bv;
    besti[lane] = bi;
    seen[lane] = sn;
    workgroupBarrier();
    for (var off: u32 = 128u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            let os = seen[lane + off];
            let ov = bestv[lane + off];
            let oi = besti[lane + off];
            if (os == 1u && (seen[lane] == 0u || ov > bestv[lane] || (ov == bestv[lane] && oi < besti[lane]))) {
                bestv[lane] = ov;
                besti[lane] = oi;
                seen[lane] = 1u;
            }
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        Ids[row] = besti[0];
    }
}
"
    .to_string()
}

/// GPU frontend stage 1: one workgroup per frame. Applies the same window and
/// pre-emphasis as `frontend.rs`, runs a 512-point radix-2 FFT in shared
/// memory (bit-reversed input, natural output, the same negative-exponent
/// convention as rustfft), and reduces the power spectrum through the mel
/// filterbank to one log-mel row.
pub fn fbank() -> String {
    "struct FbankCfg { n_samples: u32, n_frames: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> Wave: array<f32>;
@group(0) @binding(1) var<storage, read> Window: array<f32>;
@group(0) @binding(2) var<storage, read> Mel: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<uniform> cfg: FbankCfg;

var<workgroup> spec: array<vec2<f32>, 512>;
var<workgroup> pw: array<f32, 257>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn fbank(@builtin(workgroup_id) wid: vec3<u32>,
         @builtin(local_invocation_id) lid: vec3<u32>) {
    let f = wid.x;
    let start = f * 160u;
    var part = 0.0;
    for (var i: u32 = lid.x; i < 400u; i = i + 128u) {
        part = part + Wave[start + i] * 32768.0;
    }
    red[lid.x] = part;
    workgroupBarrier();
    for (var off: u32 = 64u; off > 0u; off = off >> 1u) {
        if (lid.x < off) { red[lid.x] = red[lid.x] + red[lid.x + off]; }
        workgroupBarrier();
    }
    let mean = red[0] / 400.0;
    for (var i: u32 = lid.x; i < 512u; i = i + 128u) {
        var re = 0.0;
        if (i < 400u) {
            let cur = Wave[start + i] * 32768.0 - mean;
            if (i == 0u) {
                re = (cur - 0.97 * cur) * Window[0];
            } else {
                re = (cur - 0.97 * (Wave[start + i - 1u] * 32768.0 - mean)) * Window[i];
            }
        }
        spec[i] = vec2<f32>(re, 0.0);
    }
    workgroupBarrier();
    for (var e: u32 = 0u; e < 4u; e = e + 1u) {
        let i = lid.x + e * 128u;
        var r = 0u;
        var v = i;
        for (var b: u32 = 0u; b < 9u; b = b + 1u) {
            r = (r << 1u) | (v & 1u);
            v = v >> 1u;
        }
        if (i < r) {
            let t = spec[i];
            spec[i] = spec[r];
            spec[r] = t;
        }
    }
    workgroupBarrier();
    for (var s: u32 = 1u; s <= 9u; s = s + 1u) {
        let m = 1u << s;
        let half = m >> 1u;
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let b = lid.x + e * 128u;
            let grp = b / half;
            let j = b % half;
            let i0 = grp * m + j;
            let i1 = i0 + half;
            let ang = -6.283185307179586 * f32(j) / f32(m);
            let w = vec2<f32>(cos(ang), sin(ang));
            let u = spec[i0];
            let t = vec2<f32>(w.x * spec[i1].x - w.y * spec[i1].y,
                              w.x * spec[i1].y + w.y * spec[i1].x);
            spec[i0] = u + t;
            spec[i1] = u - t;
        }
        workgroupBarrier();
    }
    // The power spectrum is computed once into shared memory: every mel bin
    // needs all 257 bins, so deriving it inside the mel loop would repeat the
    // same 514 multiplies eighty times.
    for (var k: u32 = lid.x; k < 257u; k = k + 128u) {
        let c = spec[k];
        pw[k] = c.x * c.x + c.y * c.y;
    }
    workgroupBarrier();
    if (lid.x < 80u) {
        var acc = 0.0;
        for (var k: u32 = 0u; k < 257u; k = k + 1u) {
            acc = acc + pw[k] * Mel[lid.x * 257u + k];
        }
        Out[f * 80u + lid.x] = log(max(acc, 1.1920929e-7));
    }
}
"
    .to_string()
}

/// GPU frontend stage 2: one workgroup per output row. Rows `0..4` are the
/// query embeddings (language / event / itn), the rest gather their LFR window
/// out of the fbank rows (`clamp(lr*6 + m - 3)`, which reproduces the CPU's
/// prefix/suffix padding because both are copies of the boundary frame), apply
/// CMVN, then add the sinusoidal positions encoding and the `sqrt(512)` scale
/// in the same order as `frontend.rs`. LFR rows also write their CMVN sum, the
/// `feat_sum` diagnostic.
pub fn lfr_features() -> String {
    "struct LfrCfg { n_fbank: u32, n_lfr: u32, t: u32, lid: u32, itn: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> Fbank: array<f32>;
@group(0) @binding(1) var<storage, read> Embed: array<f32>;
@group(0) @binding(2) var<storage, read> Shift: array<f32>;
@group(0) @binding(3) var<storage, read> Scale: array<f32>;
@group(0) @binding(4) var<storage, read> PosInv: array<f32>;
@group(0) @binding(5) var<storage, read_write> RowSum: array<f32>;
@group(0) @binding(6) var<storage, read_write> X: array<f32>;
@group(0) @binding(7) var<uniform> cfg: LfrCfg;

var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn lfr_features(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let scale = sqrt(512.0);
    if (row < 4u) {
        let ids = array<u32, 4>(cfg.lid, 1u, 2u, cfg.itn);
        let base = ids[row] * 560u;
        for (var j: u32 = lid.x; j < 560u; j = j + 128u) {
            var v = Embed[base + j] * scale;
            let pos = f32(row + 1u);
            if (j < 280u) {
                v = v + sin(pos * PosInv[j]);
            } else {
                v = v + cos(pos * PosInv[j - 280u]);
            }
            X[row * 560u + j] = v;
        }
        return;
    }
    let lr = row - 4u;
    var part = 0.0;
    for (var j: u32 = lid.x; j < 560u; j = j + 128u) {
        let m = j / 80u;
        let c = j % 80u;
        var idx = i32(lr * 6u + m) - 3;
        if (idx < 0) { idx = 0; }
        let last = i32(cfg.n_fbank) - 1;
        if (idx > last) { idx = last; }
        var v = (Fbank[u32(idx) * 80u + c] + Shift[j]) * Scale[j];
        part = part + v;
        v = v * scale;
        let pos = f32(row + 1u);
        if (j < 280u) {
            v = v + sin(pos * PosInv[j]);
        } else {
            v = v + cos(pos * PosInv[j - 280u]);
        }
        X[row * 560u + j] = v;
    }
    red[lid.x] = part;
    workgroupBarrier();
    for (var off: u32 = 64u; off > 0u; off = off >> 1u) {
        if (lid.x < off) { red[lid.x] = red[lid.x] + red[lid.x + off]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { RowSum[lr] = red[0]; }
}
"
    .to_string()
}
