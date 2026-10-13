//! Host f16 packing. Low 16 bits are the first value, matching WGSL `pack2x16float`.

use half::f16;

pub fn pack_f16(src: &[f32]) -> Vec<u8> {
    assert!(src.len() % 2 == 0, "f16 pack needs an even length");
    let mut out = vec![0u8; src.len() * 2];
    for (i, pair) in src.chunks_exact(2).enumerate() {
        let lo = f16::from_f32(pair[0]).to_bits() as u32;
        let hi = f16::from_f32(pair[1]).to_bits() as u32;
        out[i * 4..i * 4 + 4].copy_from_slice(&(lo | (hi << 16)).to_le_bytes());
    }
    out
}

/// Row-major `[rows, cols]` -> `[rows_pad, cols]` f16, tail rows zero.
pub fn pack_f16_rows(src: &[f32], rows: usize, cols: usize, rows_pad: usize) -> Vec<u8> {
    assert_eq!(src.len(), rows * cols);
    assert!(cols % 2 == 0 && rows_pad >= rows);
    let mut out = vec![0u8; rows_pad * cols * 2];
    for r in 0..rows {
        let packed = pack_f16(&src[r * cols..(r + 1) * cols]);
        let at = r * cols * 2;
        out[at..at + packed.len()].copy_from_slice(&packed);
    }
    out
}

pub fn pack_f16_bias(src: &[f32], cols_pad: usize) -> Vec<u8> {
    assert!(cols_pad >= src.len() && cols_pad % 2 == 0);
    let mut tmp = vec![0f32; cols_pad];
    tmp[..src.len()].copy_from_slice(src);
    pack_f16(&tmp)
}
