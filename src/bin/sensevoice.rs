//! SenseVoiceSmall CLI.
//!
//! Text goes to stdout. Timing and the device line go to stderr, so the
//! transcript can be piped. The clock inside `Model::transcribe` starts after
//! weights and pipelines are ready.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use sensevoice_small_wgpu::{language_id, load_audio, list_devices, Model, Options, VadSegment};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sensevoice: {e:#}");
            ExitCode::from(1)
        }
    }
}


#[cfg(windows)]
fn rss_mb() -> f64 {
    #[repr(C)]
    #[derive(Default)]
    struct Pmc {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            counters: *mut Pmc,
            cb: u32,
        ) -> i32;
    }
    let mut pmc = Pmc { cb: std::mem::size_of::<Pmc>() as u32, ..Default::default() };
    unsafe {
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
            return pmc.peak_working_set_size as f64 / (1024.0 * 1024.0);
        }
    }
    -1.0
}

#[cfg(not(windows))]
fn rss_mb() -> f64 {
    -1.0
}

fn mem_trace(tag: &str) {
    if std::env::var("SVS_MEM_TRACE").ok().as_deref() == Some("1") {
        eprintln!("mem_trace {tag}: peak_ws {:.1} MiB", rss_mb());
    }
}


fn run() -> Result<()> {
    let args = Args::parse()?;
    if args.help {
        print_help();
        return Ok(());
    }
    if args.list_devices {
        for line in list_devices()? {
            println!("{line}");
        }
        return Ok(());
    }
    let wav = args.wav.as_ref().context("缺少 --wav")?;
    language_id(&args.lang)?;
    let samples = load_audio(wav)?;
    mem_trace("after load_audio");
    let opt = Options {
        language: args.lang.clone(),
        use_itn: args.use_itn,
        ban_emo_unk: args.ban_emo_unk,
        segments: args.segments.clone(),
        merge_ms: args.merge_ms,
    };
    mem_trace("before Model::load");
    let mut model = Model::load(&args.model, &args.device)?;
    model.reserve_samples(samples.len())?;
    mem_trace("after Model::load");
    let transcript = model.transcribe(&samples, &opt)?;
    mem_trace("after transcribe");
    if transcript.segments.is_empty() {
        bail!("没有够长的音频段（短于 25 ms 的段会被跳过）");
    }
    let shown = if args.raw {
        transcript.raw.as_str()
    } else {
        transcript.text.as_str()
    };
    println!("{shown}");
    let t = &transcript.timing;
    let n_tok: usize = transcript.segments.iter().map(|s| s.token_ids.len()).sum();
    eprintln!(
        "device: {}\nfrontend_ms: {:.3}\nencode_ms: {:.3}\ndecode_ms: {:.3}\naudio_s: {:.3}\nrtfx: {:.2}\nfeat_sum: {}\nn_lfr: {}\nn_input: {}\ntokens: {}",
        model.device(),
        t.frontend_ms,
        t.encode_ms,
        t.decode_ms,
        t.audio_s,
        t.rtfx,
        transcript.feat_sum,
        transcript.n_lfr,
        transcript.n_input,
        n_tok,
    );
    if let Some(path) = &args.check_gold {
        check_gold(path, &transcript)?;
    }
    Ok(())
}

fn check_gold(path: &std::path::Path, got: &sensevoice_small_wgpu::Transcript) -> Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("读取 {}", path.display()))?;
    let gold: Gold = serde_json::from_str(&text).with_context(|| format!("解析 {}", path.display()))?;
    let mut ids = Vec::new();
    for seg in &got.segments {
        ids.extend_from_slice(&seg.token_ids);
    }
    let mut bad = Vec::new();
    if ids != gold.token_ids {
        let n = ids.len().min(gold.token_ids.len());
        let mut at = None;
        for i in 0..n {
            if ids[i] != gold.token_ids[i] {
                at = Some(i);
                break;
            }
        }
        bad.push(format!(
            "token_ids 长度 {} vs {}，第一处不同 {:?}",
            ids.len(),
            gold.token_ids.len(),
            at.map(|i| (i, ids[i], gold.token_ids[i]))
        ));
    }
    if got.text != gold.text {
        bad.push("text 不一致".into());
    }
    if let Some(raw) = &gold.raw {
        if &got.raw != raw {
            bad.push("raw 不一致".into());
        }
    }
    if let Some(n) = gold.n_lfr {
        if got.n_lfr != n {
            bad.push(format!("n_lfr {} vs {n}", got.n_lfr));
        }
    }
    if let Some(n) = gold.n_input {
        if got.n_input != n {
            bad.push(format!("n_input {} vs {n}", got.n_input));
        }
    }
    if let Some(sum) = gold.feat_sum {
        let d = (got.feat_sum - sum).abs();
        eprintln!("feat_sum_delta: {d}");
        if d > 1e-2 {
            eprintln!("feat_sum 偏差 {d}（gold {sum}，got {}）。token 仍以 id 为准", got.feat_sum);
        }
    }
    if bad.is_empty() {
        eprintln!("gold: 一致 {}", path.display());
        Ok(())
    } else {
        for line in &bad {
            eprintln!("gold: {line}");
        }
        bail!("与 {} 不一致", path.display());
    }
}

#[derive(Deserialize)]
struct Gold {
    text: String,
    raw: Option<String>,
    token_ids: Vec<i32>,
    feat_sum: Option<f32>,
    n_lfr: Option<usize>,
    n_input: Option<usize>,
}

struct Args {
    help: bool,
    list_devices: bool,
    model: PathBuf,
    wav: Option<PathBuf>,
    lang: String,
    use_itn: bool,
    device: String,
    segments: Vec<VadSegment>,
    merge_ms: f64,
    ban_emo_unk: bool,
    raw: bool,
    check_gold: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            help: false,
            list_devices: false,
            model: PathBuf::from(r"D:\SenseVoiceSmall\model\SenseVoiceSmall"),
            wav: None,
            lang: "auto".into(),
            use_itn: true,
            device: "auto".into(),
            segments: Vec::new(),
            merge_ms: 0.0,
            ban_emo_unk: false,
            raw: false,
            check_gold: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "-h" | "--help" => args.help = true,
                "--list-devices" => args.list_devices = true,
                "--no-itn" => args.use_itn = false,
                "--itn" => args.use_itn = true,
                "--ban-emo-unk" => args.ban_emo_unk = true,
                "--raw" => args.raw = true,
                "--model" => args.model = PathBuf::from(need(&mut it, "--model")?),
                "--wav" => args.wav = Some(PathBuf::from(need(&mut it, "--wav")?)),
                "--lang" | "--language" => args.lang = need(&mut it, "--lang")?,
                "--device" => args.device = need(&mut it, "--device")?,
                "--vad-segments" => args.segments = parse_segments(&need(&mut it, "--vad-segments")?)?,
                "--merge-ms" => {
                    args.merge_ms = need(&mut it, "--merge-ms")?.parse().context("--merge-ms")?;
                }
                "--check-gold" => args.check_gold = Some(PathBuf::from(need(&mut it, "--check-gold")?)),
                other => bail!("未知参数 {other}，用 --help"),
            }
        }
        Ok(args)
    }
}

fn need(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    it.next().with_context(|| format!("{flag} 缺少值"))
}

fn parse_segments(spec: &str) -> Result<Vec<VadSegment>> {
    let text = if let Some(path) = spec.strip_prefix('@') {
        std::fs::read_to_string(path).with_context(|| format!("读取 {path}"))?
    } else {
        spec.to_string()
    };
    let v: serde_json::Value = serde_json::from_str(&text).context("VAD JSON")?;
    let arr = v.as_array().context("VAD JSON 必须是数组")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        if let Some(pair) = item.as_array() {
            if pair.len() != 2 {
                bail!("VAD 段必须是 [start_ms, end_ms]");
            }
            out.push(VadSegment {
                start_ms: pair[0].as_f64().context("start_ms")?,
                end_ms: pair[1].as_f64().context("end_ms")?,
            });
        } else if let Some(obj) = item.as_object() {
            let start = obj.get("start_ms").and_then(|x| x.as_f64()).context("start_ms")?;
            let end = obj.get("end_ms").and_then(|x| x.as_f64()).context("end_ms")?;
            out.push(VadSegment { start_ms: start, end_ms: end });
        } else {
            bail!("VAD 段必须是 [start_ms, end_ms] 或 {{\"start_ms\":..,\"end_ms\":..}}");
        }
    }
    Ok(out)
}

const HELP: &str = r#"sensevoice — SenseVoiceSmall（Rust CPU + wgpu）

对齐目标是 dither=0 的官方推理图。FunASR 默认 dither=1.0 带随机，不能逐位对齐。
VAD 模型不内置。不传段就转写整段；传段就按段切。超过 GPU 缓冲上限会报错，
不会悄悄按时间切开。

用法:
  sensevoice --wav FILE [选项]
  sensevoice --list-devices

选项:
  --model DIR          官方模型目录，含 model.pt、am.mvn、bpe.model
                       默认 D:\SenseVoiceSmall\model\SenseVoiceSmall
  --wav FILE           wav / mp3 / flac。非 16 kHz 用 torchaudio 默认 sinc 重采样
  --lang NAME          auto|zh|en|yue|ja|ko|nospeech，默认 auto
  --itn / --no-itn     默认开启 ITN（withitn）
  --device SPEC        auto|cpu|vulkan[:i]|dx12|metal|#n|名称子串，默认 auto
  --vad-segments JSON  [[start_ms,end_ms],...] 或 [{"start_ms":..,"end_ms":..}]
                       以 @ 开头则从文件读取
  --merge-ms N         默认 0，不改写传入的段。15000 才走官方 merge_vad
                       （起点从 0 开始，会把开头静音并进去）
  --ban-emo-unk        屏蔽 CTC 的 emo unk（id 25009）
  --raw                stdout 打未后处理的 token 文本
  --check-gold FILE    对比 token id、text、raw；不一致则退出码 1
  --list-devices       列出适配器，不加载模型
  --help

stdout 是文本。stderr 是设备、frontend/encode/decode、RTFx。
RTFx 不计模型加载和 pipeline 编译。
"#;

fn print_help() {
    println!("{HELP}");
}
