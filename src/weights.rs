use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::pt::{self, Tensor};

pub const EMBED_DIM: usize = 560;
pub const HIDDEN: usize = 512;
pub const FFN: usize = 2048;
pub const QKV: usize = 1536;
pub const HEADS: usize = 4;
pub const DK: usize = 128;
pub const CTC_ROWS: usize = 25055;
pub const CTC_PAD: usize = 25088;
pub const N_EMBED: usize = 16;

#[derive(Clone)]
pub struct Linear {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone)]
pub struct Layer {
    pub n1_w: Vec<f32>,
    pub n1_b: Vec<f32>,
    pub qkv: Linear,
    pub fsmn: Vec<f32>,
    pub out: Linear,
    pub n2_w: Vec<f32>,
    pub n2_b: Vec<f32>,
    pub w1: Linear,
    pub w2: Linear,
    pub in_dim: usize,
}

pub struct Weights {
    pub embed: Vec<f32>,
    pub layers: Vec<Layer>,
    pub tp: Vec<Layer>,
    pub after_w: Vec<f32>,
    pub after_b: Vec<f32>,
    pub tp_w: Vec<f32>,
    pub tp_b: Vec<f32>,
    pub ctc: Linear,
}

impl Weights {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("model.pt");
        if !path.exists() {
            bail!("{} 不存在。把官方 model.pt 放在模型目录里", path.display());
        }
        let mut st = pt::load_state_dict(&path)?;
        let embed = matrix(&mut st, "embed.weight", &[N_EMBED, EMBED_DIM])?;
        let mut layers = vec![load_layer(&mut st, "encoder.encoders0.0")?];
        let mut i = 0usize;
        loop {
            let key = format!("encoder.encoders.{i}.norm1.weight");
            if !st.contains_key(&key) {
                break;
            }
            layers.push(load_layer(&mut st, &format!("encoder.encoders.{i}"))?);
            i += 1;
        }
        let mut tp = Vec::new();
        i = 0;
        loop {
            let key = format!("encoder.tp_encoders.{i}.norm1.weight");
            if !st.contains_key(&key) {
                break;
            }
            tp.push(load_layer(&mut st, &format!("encoder.tp_encoders.{i}"))?);
            i += 1;
        }
        if layers.len() != 50 || tp.len() != 20 {
            bail!("编码器层数 {} + tp {}，期望 50 + 20", layers.len(), tp.len());
        }
        if layers[0].in_dim != EMBED_DIM {
            bail!("第 0 层输入维 {}，期望 {EMBED_DIM}", layers[0].in_dim);
        }
        for (i, layer) in layers.iter().enumerate().skip(1) {
            if layer.in_dim != HIDDEN {
                bail!("第 {i} 层输入维 {}，期望 {HIDDEN}", layer.in_dim);
            }
        }
        Ok(Self {
            embed,
            layers,
            tp,
            after_w: vector(&mut st, "encoder.after_norm.weight", HIDDEN)?,
            after_b: vector(&mut st, "encoder.after_norm.bias", HIDDEN)?,
            tp_w: vector(&mut st, "encoder.tp_norm.weight", HIDDEN)?,
            tp_b: vector(&mut st, "encoder.tp_norm.bias", HIDDEN)?,
            ctc: linear(&mut st, "ctc.ctc_lo.weight", "ctc.ctc_lo.bias", CTC_ROWS, HIDDEN)?,
        })
    }
}

fn load_layer(st: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Layer> {
    let qkv = linear(
        st,
        &format!("{prefix}.self_attn.linear_q_k_v.weight"),
        &format!("{prefix}.self_attn.linear_q_k_v.bias"),
        QKV,
        0,
    )?;
    let in_dim = qkv.cols;
    let n1_w = vector(st, &format!("{prefix}.norm1.weight"), in_dim)?;
    let n1_b = vector(st, &format!("{prefix}.norm1.bias"), in_dim)?;
    let fsmn_t = tensor(st, &format!("{prefix}.self_attn.fsmn_block.weight"))?;
    if fsmn_t.data.len() != HIDDEN * 11 {
        bail!("{prefix} fsmn 长度 {}，期望 {}", fsmn_t.data.len(), HIDDEN * 11);
    }
    let out = linear(
        st,
        &format!("{prefix}.self_attn.linear_out.weight"),
        &format!("{prefix}.self_attn.linear_out.bias"),
        HIDDEN,
        HIDDEN,
    )?;
    let n2_w = vector(st, &format!("{prefix}.norm2.weight"), HIDDEN)?;
    let n2_b = vector(st, &format!("{prefix}.norm2.bias"), HIDDEN)?;
    let w1 = linear(
        st,
        &format!("{prefix}.feed_forward.w_1.weight"),
        &format!("{prefix}.feed_forward.w_1.bias"),
        FFN,
        HIDDEN,
    )?;
    let w2 = linear(
        st,
        &format!("{prefix}.feed_forward.w_2.weight"),
        &format!("{prefix}.feed_forward.w_2.bias"),
        HIDDEN,
        FFN,
    )?;
    Ok(Layer {
        n1_w,
        n1_b,
        qkv,
        fsmn: fsmn_t.data,
        out,
        n2_w,
        n2_b,
        w1,
        w2,
        in_dim,
    })
}

struct Owned {
    data: Vec<f32>,
    shape: Vec<usize>,
}

fn tensor(st: &mut HashMap<String, Tensor>, name: &str) -> Result<Owned> {
    let t = st.remove(name).with_context(|| format!("缺少权重 {name}"))?;
    Ok(Owned {
        data: t.data,
        shape: t.shape,
    })
}

fn vector(st: &mut HashMap<String, Tensor>, name: &str, n: usize) -> Result<Vec<f32>> {
    let t = tensor(st, name)?;
    if t.data.len() != n {
        bail!("{name} 长度 {}，期望 {n}", t.data.len());
    }
    Ok(t.data)
}

fn matrix(st: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = tensor(st, name)?;
    if t.shape != shape {
        bail!("{name} shape {:?}，期望 {shape:?}", t.shape);
    }
    Ok(t.data)
}

fn linear(st: &mut HashMap<String, Tensor>, wname: &str, bname: &str, rows: usize, cols: usize) -> Result<Linear> {
    let w = tensor(st, wname)?;
    if w.shape.len() != 2 || (rows != 0 && w.shape[0] != rows) || (cols != 0 && w.shape[1] != cols) {
        bail!("{wname} shape {:?}，期望 [{rows}, {cols}]", w.shape);
    }
    let b = vector(st, bname, w.shape[0])?;
    Ok(Linear {
        rows: w.shape[0],
        cols: w.shape[1],
        w: w.data,
        b,
    })
}
