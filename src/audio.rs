use std::path::Path;

use anyhow::{Context, Result};

/// Mono f32 waveform at 16 kHz, roughly in [-1, 1].
///
/// 16-bit wav is divided by 32768, so the frontend's later `* 32768` restores the
/// integer sample. Other rates use torchaudio's default sinc resample
/// (`sinc_interp_hann`, width 6, rolloff 0.99), which is what the reference
/// script uses. Stereo is averaged after the integer-to-float conversion.
pub fn load_audio(path: &Path) -> Result<Vec<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let (interleaved, sr, channels) = if ext == "wav" {
        match load_wav(path) {
            Ok(v) => v,
            Err(e) => load_symphonia(path).with_context(|| format!("wav 解码失败（{e:#}）"))?,
        }
    } else {
        load_symphonia(path)?
    };
    let mono = downmix(&interleaved, channels);
    Ok(resample_sinc(&mono, sr, 16_000))
}

fn downmix(samples: &[f32], channels: usize) -> Vec<f32> {
    let channels = channels.max(1);
    if channels == 1 {
        return samples.to_vec();
    }
    let frames = samples.len() / channels;
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut s = 0.0f32;
        for c in 0..channels {
            s += samples[i * channels + c];
        }
        out.push(s / channels as f32);
    }
    out
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// torchaudio.functional.resample defaults: sinc_interp_hann, width 6, rolloff 0.99.
fn resample_sinc(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let gcd = gcd_u32(from, to);
    let orig = from / gcd;
    let new = to / gcd;
    let base64 = f64::from(orig.min(new)) * 0.99;
    let width = (6.0 * f64::from(orig) / base64).ceil() as i32;
    let base = base64 as f32;
    let idx_len = (2 * width + orig as i32) as usize;
    let mut idx = vec![0f32; idx_len];
    for i in 0..idx_len {
        idx[i] = (-width + i as i32) as f32 / orig as f32;
    }
    let mut kernels = vec![0f32; new as usize * idx_len];
    let pi = std::f32::consts::PI;
    let scale = base / orig as f32;
    for j in 0..new as usize {
        let t0 = -(j as f32) / new as f32;
        for k in 0..idx_len {
            let mut t = (t0 + idx[k]) * base;
            t = t.clamp(-6.0, 6.0);
            let window = (t * pi / 6.0 / 2.0).cos().powi(2);
            t *= pi;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernels[j * idx_len + k] = sinc * window * scale;
        }
    }
    let left = width as usize;
    let right = width as usize + orig as usize;
    let padded_len = left + x.len() + right;
    let mut padded = vec![0f32; padded_len];
    padded[left..left + x.len()].copy_from_slice(x);
    let stride = orig as usize;
    let out_full = (padded_len - idx_len) / stride + 1;
    let total = out_full * new as usize;
    let ratio = ((new as f64) * (x.len() as f64) / (orig as f64)) as f32;
    let target = ratio.ceil() as usize;
    let mut y = vec![0f32; target];
    let new_u = new as usize;
    for n in 0..target.min(total) {
        let time = n / new_u;
        let ch = n % new_u;
        let start = time * stride;
        let kern = &kernels[ch * idx_len..];
        let mut acc = 0.0f32;
        for k in 0..idx_len {
            acc += padded[start + k] * kern[k];
        }
        y[n] = acc;
    }
    y
}

fn load_wav(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("打开 {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let denom = (1i64 << (spec.bits_per_sample.saturating_sub(1).min(31))) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / denom))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    Ok((samples, spec.sample_rate, channels))
}

fn load_symphonia(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).with_context(|| format!("打开 {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let hint = Hint::new();
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("探测音频容器")?;
    let mut format = probed.format;
    let track = format.default_track().context("没有音轨")?;
    let track_id = track.id;
    let sr = track.codec_params.sample_rate.context("缺少采样率")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("创建解码器")?;
    let mut pcm = Vec::new();
    let mut channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(e).context("读取音频包"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SError::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("解码音频包"),
        };
        channels = decoded.spec().channels.count();
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buf.copy_interleaved_ref(decoded);
        pcm.extend_from_slice(buf.samples());
    }
    Ok((pcm, sr, channels))
}
