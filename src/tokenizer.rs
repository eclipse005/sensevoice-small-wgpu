use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pieces: Vec<String>,
    control: Vec<bool>,
}

#[derive(Deserialize)]
struct File {
    pieces: Vec<String>,
    control_ids: Vec<usize>,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let bpe = model_dir.join("chn_jpn_yue_eng_ko_spectok.bpe.model");
        if bpe.exists() {
            return load_bpe(&bpe);
        }
        let path = model_dir.join("tokens.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("读取 {}。官方目录里应有 chn_jpn_yue_eng_ko_spectok.bpe.model", path.display()))?;
        let file: File = serde_json::from_str(&text).context("解析 tokens.json")?;
        let mut control = vec![false; file.pieces.len()];
        for id in file.control_ids {
            if let Some(slot) = control.get_mut(id) {
                *slot = true;
            }
        }
        Ok(Self {
            pieces: file.pieces,
            control,
        })
    }

    /// SentencePiece `decode`: skip control pieces, turn U+2581 into a space,
    /// and splice `<0xHH>` byte-fallback as raw bytes before the final UTF-8 decode.
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut raw = Vec::new();
        for &id in ids {
            if id < 0 {
                continue;
            }
            let i = id as usize;
            let Some(piece) = self.pieces.get(i) else {
                continue;
            };
            if self.control.get(i).copied().unwrap_or(false) {
                continue;
            }
            if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&piece[3..5], 16) {
                    raw.push(b);
                    continue;
                }
            }
            raw.extend(piece.replace('\u{2581}', " ").into_bytes());
        }
        String::from_utf8(raw).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }
}

/// SentencePiece `ModelProto`: repeated `pieces` (field 1). Type 3 is CONTROL.
fn load_bpe(path: &Path) -> Result<Tokenizer> {
    let bytes = std::fs::read(path).with_context(|| format!("读取 {}", path.display()))?;
    let mut pieces = Vec::new();
    let mut control = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let (tag, n) = varint(&bytes, i)?;
        i = n;
        let field = tag >> 3;
        let wire = tag & 7;
        if field == 1 && wire == 2 {
            let (len, n) = varint(&bytes, i)?;
            i = n;
            let end = i + len as usize;
            if end > bytes.len() {
                bail!("bpe 模型截断");
            }
            let (piece, ty) = parse_piece(&bytes[i..end])?;
            control.push(ty == 3);
            pieces.push(piece);
            i = end;
        } else {
            i = skip(&bytes, i, wire)?;
        }
    }
    if pieces.is_empty() {
        bail!("{} 里没有词表", path.display());
    }
    Ok(Tokenizer { pieces, control })
}

fn parse_piece(msg: &[u8]) -> Result<(String, i32)> {
    let mut piece = String::new();
    let mut ty = 1i32;
    let mut i = 0usize;
    while i < msg.len() {
        let (tag, n) = varint(msg, i)?;
        i = n;
        let field = tag >> 3;
        let wire = tag & 7;
        if field == 1 && wire == 2 {
            let (len, n) = varint(msg, i)?;
            i = n;
            let end = i + len as usize;
            piece = String::from_utf8(msg[i..end].to_vec()).context("词表项不是 utf-8")?;
            i = end;
        } else if field == 3 && wire == 0 {
            let (v, n) = varint(msg, i)?;
            ty = v as i32;
            i = n;
        } else {
            i = skip(msg, i, wire)?;
        }
    }
    Ok((piece, ty))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_bpe_matches_sentencepiece_control() {
        let path = Path::new(r"D:\SenseVoiceSmall\model\SenseVoiceSmall\chn_jpn_yue_eng_ko_spectok.bpe.model");
        let tok = load_bpe(path).unwrap();
        assert_eq!(tok.pieces.len(), 25055);
        assert!(!tok.control[0]);
        assert!(tok.control[1] && tok.control[2]);
        assert_eq!(tok.pieces[3], "\u{2581}the");
    }
}

fn skip(data: &[u8], mut i: usize, wire: u64) -> Result<usize> {
    match wire {
        0 => {
            let (_, n) = varint(data, i)?;
            Ok(n)
        }
        1 => Ok(i + 8),
        2 => {
            let (len, n) = varint(data, i)?;
            i = n + len as usize;
            if i > data.len() {
                bail!("protobuf 截断");
            }
            Ok(i)
        }
        5 => Ok(i + 4),
        _ => bail!("protobuf wire {wire}"),
    }
}

fn varint(data: &[u8], mut i: usize) -> Result<(u64, usize)> {
    let mut out = 0u64;
    let mut shift = 0;
    while i < data.len() && shift < 64 {
        let b = data[i];
        i += 1;
        out |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok((out, i));
        }
        shift += 7;
    }
    bail!("varint 截断")
}
