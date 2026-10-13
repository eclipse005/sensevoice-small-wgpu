pub const BLANK: i32 = 0;
pub const EMO_UNK: usize = 25009;
pub const VOCAB: usize = 25055;

/// Argmax per row. Ties keep the smaller index. `ban` is skipped (logit treated as missing).
///
/// Rows are independent and each row's scan keeps its order, so running the
/// rows on rayon does not move a single comparison.
pub fn argmax_rows(logits: &[f32], t: usize, n: usize, ban: Option<usize>) -> Vec<i32> {
    use rayon::prelude::*;
    logits[..t * n]
        .par_chunks(n)
        .map(|s| {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, v) in s.iter().enumerate() {
                if ban == Some(i) {
                    continue;
                }
                if *v > bv {
                    bv = *v;
                    best = i;
                }
            }
            best as i32
        })
        .collect()
}

/// `unique_consecutive` then drop blank 0, fused the way the reference script does it.
pub fn greedy(frame_ids: &[i32]) -> Vec<i32> {
    let mut out = Vec::new();
    let mut prev: Option<i32> = None;
    for &id in frame_ids {
        if prev == Some(id) {
            continue;
        }
        if id != BLANK {
            out.push(id);
        }
        prev = Some(id);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_collapses_and_keeps_repeat_across_blank() {
        assert_eq!(greedy(&[5, 5, 0, 5, 0, 0, 7]), vec![5, 5, 7]);
    }

    #[test]
    fn argmax_tie_keeps_first() {
        let logits = [1.0, 1.0, 0.5];
        assert_eq!(argmax_rows(&logits, 1, 3, None), vec![0]);
        assert_eq!(argmax_rows(&logits, 1, 3, Some(0)), vec![1]);
    }
}
