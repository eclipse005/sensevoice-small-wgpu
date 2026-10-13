pub const BLANK: i32 = 0;
pub const EMO_UNK: usize = 25009;
pub const VOCAB: usize = 25055;

/// Fold one vocab tile of CTC logits into the running per-row argmax.
///
/// `tile` is the [rows][width] row-major block for vocab columns
/// `j0..j0 + width`, `bias` is the full per-column bias. Callers initialise
/// `frames` to 0 and `best_v` to NEG_INFINITY before the first tile and feed
/// tiles in ascending `j0`. Ties keep the smaller index and `ban` is skipped,
/// exactly like the whole-row scan this replaces: `acc + b` is the same float
/// as the old `b + acc`, so the winners cannot move.
///
/// Rows are independent and each row's scan keeps its order, so the rayon
/// split does not move a single comparison.
pub fn argmax_tile(
    tile: &[f32],
    width: usize,
    j0: usize,
    bias: &[f32],
    ban: Option<usize>,
    best_v: &mut [f32],
    frames: &mut [i32],
) {
    use rayon::prelude::*;
    best_v
        .par_iter_mut()
        .zip(frames.par_iter_mut())
        .zip(tile.par_chunks(width))
        .for_each(|((bv, best), row)| {
            for (j, x) in row.iter().enumerate() {
                let idx = j0 + j;
                if Some(idx) == ban {
                    continue;
                }
                let v = *x + bias[idx];
                if v > *bv {
                    *bv = v;
                    *best = idx as i32;
                }
            }
        });
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

    fn run_argmax(logits: &[f32], bias: &[f32], ban: Option<usize>) -> (Vec<f32>, Vec<i32>) {
        let mut best_v = vec![f32::NEG_INFINITY; 1];
        let mut frames = vec![0i32; 1];
        argmax_tile(logits, logits.len(), 0, bias, ban, &mut best_v, &mut frames);
        (best_v, frames)
    }

    #[test]
    fn greedy_collapses_and_keeps_repeat_across_blank() {
        assert_eq!(greedy(&[5, 5, 0, 5, 0, 0, 7]), vec![5, 5, 7]);
    }

    #[test]
    fn argmax_tie_keeps_first() {
        let logits = [1.0, 1.0, 0.5];
        let bias = [0.0; 3];
        assert_eq!(run_argmax(&logits, &bias, None).1, vec![0]);
        assert_eq!(run_argmax(&logits, &bias, Some(0)).1, vec![1]);
    }

    #[test]
    fn argmax_tile_matches_whole_row_across_tiles() {
        // Two tiles whose boundary sits between the two tied maxima: the winner
        // must still be the leftmost, and the bias must enter both tiles alike.
        let mut best_v = vec![f32::NEG_INFINITY; 1];
        let mut frames = vec![0i32; 1];
        let bias = [0.0, 1.0, 1.0, 0.0];
        argmax_tile(&[2.0, 1.5], 2, 0, &bias, None, &mut best_v, &mut frames);
        argmax_tile(&[1.5, 2.0], 2, 2, &bias, None, &mut best_v, &mut frames);
        // col 0: 2.0, col 1: 2.5, col 2: 2.5, col 3: 2.0 -> tie at 1 and 2.
        assert_eq!(frames, vec![1]);
        assert_eq!(best_v, vec![2.5]);
    }
}
