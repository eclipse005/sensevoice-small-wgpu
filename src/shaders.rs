//! Small compute shaders. The GEMM itself lives in `gemm_f32`.

pub fn layernorm() -> String {
    // One thread per row. The row is read as vec4 words, but the four
    // components are still accumulated x, y, z, w in order, so the f32 sum is
    // the scalar one bit for bit. `d` is always a multiple of 4 (512 or 560),
    // so the vec4 view is exact — no tail elements.
    "struct LnCfg { d: u32, eps: f32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read>       Src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       Wgt: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       Bia: array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             cfg: LnCfg;
@group(0) @binding(4) var<storage, read_write> Dst: array<vec4<f32>>;

@compute @workgroup_size(32)
fn layernorm(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let dv = d / 4u;
    let row = wid.x * 32u + lid.x;
    let base = row * dv;
    var mean = 0.0;
    for (var j: u32 = 0u; j < dv; j = j + 1u) {
        let v = Src[base + j];
        mean = mean + v.x;
        mean = mean + v.y;
        mean = mean + v.z;
        mean = mean + v.w;
    }
    mean = mean / f32(d);
    var var_ = 0.0;
    for (var j: u32 = 0u; j < dv; j = j + 1u) {
        let v = Src[base + j];
        let z = v - vec4<f32>(mean);
        var_ = var_ + z.x * z.x;
        var_ = var_ + z.y * z.y;
        var_ = var_ + z.z * z.z;
        var_ = var_ + z.w * z.w;
    }
    var_ = var_ / f32(d);
    let inv = 1.0 / sqrt(var_ + cfg.eps);
    for (var j: u32 = 0u; j < dv; j = j + 1u) {
        let v = Src[base + j];
        let g = Wgt[j];
        let b = Bia[j];
        Dst[base + j] = (v - vec4<f32>(mean)) * vec4<f32>(inv) * g + b;
    }
}
"
    .to_string()
}

pub fn split_qkv() -> String {
    "struct SplitCfg { t_pad: u32, t_valid: u32, scale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read> Qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> Q: array<f32>;
@group(0) @binding(2) var<storage, read_write> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<uniform> cfg: SplitCfg;

@compute @workgroup_size(128, 1, 1)
fn split(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let h = gid.y;
    let t = gid.z;
    if (d >= 128u || h >= 4u || t >= cfg.t_pad) { return; }
    let dst = (h * cfg.t_pad + t) * 128u + d;
    if (t >= cfg.t_valid) {
        Q[dst] = 0.0;
        K[dst] = 0.0;
        V[dst] = 0.0;
        return;
    }
    let src = t * 1536u + h * 128u + d;
    Q[dst] = Qkv[src] * cfg.scale;
    K[dst] = Qkv[src + 512u];
    V[dst] = Qkv[src + 1024u];
}
"
    .to_string()
}

pub fn merge_heads() -> String {
    "struct MergeCfg { t_pad: u32, t_valid: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read> Av: array<f32>;
@group(0) @binding(1) var<storage, read_write> Ctx: array<f32>;
@group(0) @binding(2) var<uniform> cfg: MergeCfg;

@compute @workgroup_size(128, 1, 1)
fn merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let h = gid.y;
    let t = gid.z;
    if (d >= 128u || h >= 4u || t >= cfg.t_pad) { return; }
    var val = 0.0;
    if (t < cfg.t_valid) {
        val = Av[(h * cfg.t_pad + t) * 128u + d];
    }
    Ctx[t * 512u + h * 128u + d] = val;
}
"
    .to_string()
}

pub fn fsmn() -> String {
    "struct FsmnCfg { t_valid: u32, t_pad: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read> Qkv: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<uniform> cfg: FsmnCfg;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(256)
fn fsmn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let t = gid.y;
    if (c >= 512u || t >= cfg.t_pad) { return; }
    if (t >= cfg.t_valid) {
        Out[t * 512u + c] = 0.0;
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
    Out[t * 512u + c] = acc + v;
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

pub fn add() -> String {
    "struct AddCfg { n: u32, mode: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> D: array<f32>;
@group(0) @binding(3) var<uniform> cfg: AddCfg;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
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
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<uniform> cfg: NCfg;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    X[i] = max(X[i], 0.0);
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
