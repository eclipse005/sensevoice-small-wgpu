//! `rich_transcription_postprocess` from FunASR `postprocess_utils.py`.
//!
//! Emoji space-stripping walks a sorted set. CPython set order is hash-randomized;
//! the reference script sorts, and this matches that script.

const EMO: &[(&str, &str)] = &[
    ("<|HAPPY|>", "😊"),
    ("<|SAD|>", "😔"),
    ("<|ANGRY|>", "😡"),
    ("<|NEUTRAL|>", ""),
    ("<|FEARFUL|>", "😰"),
    ("<|DISGUSTED|>", "🤢"),
    ("<|SURPRISED|>", "😮"),
];

const EVENT: &[(&str, &str)] = &[
    ("<|BGM|>", "🎼"),
    ("<|Speech|>", ""),
    ("<|Applause|>", "👏"),
    ("<|Laughter|>", "😀"),
    ("<|Cry|>", "😭"),
    ("<|Sneeze|>", "🤧"),
    ("<|Breath|>", ""),
    ("<|Cough|>", "😷"),
];

const LANG: &[&str] = &["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>", "<|nospeech|>"];

const EMOJI_DICT: &[(&str, &str)] = &[
    ("<|nospeech|><|Event_UNK|>", "❓"),
    ("<|zh|>", ""),
    ("<|en|>", ""),
    ("<|yue|>", ""),
    ("<|ja|>", ""),
    ("<|ko|>", ""),
    ("<|nospeech|>", ""),
    ("<|HAPPY|>", "😊"),
    ("<|SAD|>", "😔"),
    ("<|ANGRY|>", "😡"),
    ("<|NEUTRAL|>", ""),
    ("<|BGM|>", "🎼"),
    ("<|Speech|>", ""),
    ("<|Applause|>", "👏"),
    ("<|Laughter|>", "😀"),
    ("<|FEARFUL|>", "😰"),
    ("<|DISGUSTED|>", "🤢"),
    ("<|SURPRISED|>", "😮"),
    ("<|Cry|>", "😭"),
    ("<|EMO_UNKNOWN|>", ""),
    ("<|Sneeze|>", "🤧"),
    ("<|Breath|>", ""),
    ("<|Cough|>", "😷"),
    ("<|Sing|>", ""),
    ("<|Speech_Noise|>", ""),
    ("<|withitn|>", ""),
    ("<|woitn|>", ""),
    ("<|GBG|>", ""),
    ("<|Event_UNK|>", ""),
];

fn emoji_set() -> Vec<&'static str> {
    let mut v = vec!["😊", "😔", "😡", "😰", "🤢", "😮", "🎼", "👏", "😀", "😭", "🤧", "😷"];
    v.sort_unstable();
    v
}

fn emo_chars() -> &'static [&'static str] {
    &["😊", "😔", "😡", "😰", "🤢", "😮"]
}

fn event_chars() -> &'static [&'static str] {
    &["🎼", "👏", "😀", "😭", "🤧", "😷"]
}

fn format_str_v2(input: &str) -> String {
    let mut counts = Vec::with_capacity(EMOJI_DICT.len());
    let mut s = input.to_string();
    for (tag, _) in EMOJI_DICT {
        let n = s.matches(tag).count();
        counts.push(n);
        if n > 0 && !tag.is_empty() {
            s = s.replace(tag, "");
        }
    }
    let count_of = |tag: &str| -> usize {
        EMOJI_DICT
            .iter()
            .position(|(t, _)| *t == tag)
            .map(|i| counts[i])
            .unwrap_or(0)
    };
    let mut emo = "<|NEUTRAL|>";
    for (tag, _) in EMO {
        if count_of(tag) > count_of(emo) {
            emo = tag;
        }
    }
    for (tag, glyph) in EVENT {
        if count_of(tag) > 0 {
            s = format!("{glyph}{s}");
        }
    }
    let emo_glyph = EMO.iter().find(|(t, _)| *t == emo).map(|(_, g)| *g).unwrap_or("");
    s.push_str(emo_glyph);
    for glyph in emoji_set() {
        s = s.replace(&format!(" {glyph}"), glyph);
        s = s.replace(&format!("{glyph} "), glyph);
    }
    s.trim_matches(' ').to_string()
}

fn glyph_of(ch: char) -> String {
    ch.to_string()
}

fn first_glyph(s: &str, set: &[&str]) -> Option<String> {
    let ch = s.chars().next()?;
    let g = glyph_of(ch);
    set.iter().any(|e| *e == g).then_some(g)
}

fn last_glyph(s: &str, set: &[&str]) -> Option<String> {
    let ch = s.chars().next_back()?;
    let g = glyph_of(ch);
    set.iter().any(|e| *e == g).then_some(g)
}

pub fn rich_post(input: &str) -> String {
    let mut s = input.replace("<|nospeech|><|Event_UNK|>", "❓");
    for lang in LANG {
        s = s.replace(lang, "<|lang|>");
    }
    let parts: Vec<String> = s
        .split("<|lang|>")
        .map(|p| format_str_v2(p).trim_matches(' ').to_string())
        .collect();
    let mut new_s = format!(" {}", parts.first().map(|p| p.as_str()).unwrap_or(""));
    // `new_s` always starts with a space, so its first character is not an event emoji.
    let mut cur_event: Option<String> = None;
    for part in parts.iter().skip(1) {
        if part.is_empty() {
            continue;
        }
        let mut part = part.clone();
        let ev = first_glyph(&part, event_chars());
        if ev.is_some() && cur_event == ev {
            part = part.chars().skip(1).collect();
        }
        if part.is_empty() {
            continue;
        }
        cur_event = first_glyph(&part, event_chars());
        let emo = last_glyph(&part, emo_chars());
        let prev_emo = last_glyph(&new_s, emo_chars());
        if emo.is_some() && emo == prev_emo {
            new_s.pop();
        }
        new_s.push_str(part.trim());
    }
    new_s = new_s.replace("The.", " ");
    new_s.trim().to_string()
}

/// Tags the model emits in front of the transcript. Missing tags stay empty.
#[derive(Debug, Clone, Default)]
pub struct Tags {
    pub language: String,
    pub emotion: String,
    pub event: String,
    pub itn: String,
}

pub fn read_tags(raw: &str) -> Tags {
    let mut tags = Tags::default();
    let bytes = raw.as_bytes();
    let mut i = 0;
    let mut found = Vec::new();
    while i + 1 < bytes.len() && bytes[i] == b'<' && bytes[i + 1] == b'|' {
        if let Some(end) = raw[i..].find("|>") {
            let tag = &raw[i + 2..i + end];
            found.push(tag.to_string());
            i += end + 2;
            if found.len() == 4 {
                break;
            }
        } else {
            break;
        }
    }
    if let Some(v) = found.get(0) {
        tags.language = v.clone();
    }
    if let Some(v) = found.get(1) {
        tags.emotion = v.clone();
    }
    if let Some(v) = found.get(2) {
        tags.event = v.clone();
    }
    if let Some(v) = found.get(3) {
        tags.itn = v.clone();
    }
    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zh_bgm_angry() {
        let raw = "<|zh|><|ANGRY|><|BGM|><|withitn|>你好";
        assert_eq!(rich_post(raw), "🎼你好😡");
    }

    #[test]
    fn neutral_speech_has_no_emoji() {
        let raw = "<|en|><|NEUTRAL|><|Speech|><|withitn|>Hello";
        assert_eq!(rich_post(raw), "Hello");
    }
}
