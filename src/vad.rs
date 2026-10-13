//! External VAD segments. This crate does not ship a VAD model.
//!
//! `merge_vad` matches FunASR `funasr/utils/vad_utils.py`: time points are the
//! unique starts and ends, the cursor begins at 0 (leading silence is included),
//! and a cut is emitted when the *next* point would reach `max_length`.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub start_ms: f64,
    pub end_ms: f64,
}

pub fn merge_vad(segments: &[Segment], max_length_ms: f64, min_length_ms: f64) -> Vec<Segment> {
    if segments.len() <= 1 {
        return segments.to_vec();
    }
    let mut times = Vec::with_capacity(segments.len() * 2);
    for s in segments {
        times.push(s.start_ms);
        times.push(s.end_ms);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| *a == *b);
    if times.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut bg = 0.0;
    for i in 0..times.len() - 1 {
        let time = times[i];
        if times[i + 1] - bg < max_length_ms {
            continue;
        }
        if time - bg > min_length_ms {
            out.push(Segment {
                start_ms: bg,
                end_ms: time,
            });
        }
        bg = time;
    }
    out.push(Segment {
        start_ms: bg,
        end_ms: *times.last().unwrap(),
    });
    out
}

/// `int(ms * 16)` at 16 kHz, clamped to the buffer. Matches FunASR `slice_padding_audio_samples`.
pub fn slice_samples(samples: &[f32], seg: Segment) -> &[f32] {
    let n = samples.len() as isize;
    let start = (seg.start_ms * 16.0).trunc() as isize;
    let end = (seg.end_ms * 16.0).trunc() as isize;
    let start = start.clamp(0, n) as usize;
    let end = end.clamp(0, n) as usize;
    if end <= start {
        &[]
    } else {
        &samples[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_starts_at_zero_and_respects_max() {
        let segs = [
            Segment { start_ms: 100.0, end_ms: 200.0 },
            Segment { start_ms: 300.0, end_ms: 500.0 },
        ];
        let m = merge_vad(&segs, 15_000.0, 0.0);
        assert_eq!(m, vec![Segment { start_ms: 0.0, end_ms: 500.0 }]);

        let segs = [
            Segment { start_ms: 0.0, end_ms: 10_000.0 },
            Segment { start_ms: 10_000.0, end_ms: 20_000.0 },
        ];
        let m = merge_vad(&segs, 15_000.0, 0.0);
        assert_eq!(
            m,
            vec![
                Segment { start_ms: 0.0, end_ms: 10_000.0 },
                Segment { start_ms: 10_000.0, end_ms: 20_000.0 },
            ]
        );
    }

    #[test]
    fn single_segment_is_unchanged() {
        let segs = [Segment { start_ms: 5.0, end_ms: 9.0 }];
        assert_eq!(merge_vad(&segs, 15_000.0, 0.0), segs);
    }
}
