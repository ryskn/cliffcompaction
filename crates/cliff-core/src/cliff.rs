//! The compactor: `[head verbatim] + [one summary message] + [last
//! keep_recent turns verbatim]`.
//!
//! Operates on a slice of borrowed messages and returns a plan (head_len,
//! summary, cut) rather than a rebuilt list: the kept tail is always a
//! suffix of the input, so the compacted sequence is exactly
//! `messages[..head_len] + [summary] + messages[cut..]` and callers splice
//! it without cloning a single kept message.

use std::ops::Range;

use serde_json::Value;

use crate::config::Config;
use crate::dialects::{Dialect, SUMMARY_HEADER};
use crate::py::PyResult;

#[derive(Debug, Clone, PartialEq)]
pub struct Compaction {
    /// Number of verbatim head messages.
    pub head_len: usize,
    /// The synthetic summary message.
    pub summary: Value,
    /// Index into the INPUT list: `messages[cut..]` were kept verbatim.
    pub cut: usize,
}

impl Compaction {
    /// Length of the compacted sequence for an input of `n` messages.
    pub fn out_len(&self, n: usize) -> usize {
        self.head_len + 1 + (n - self.cut)
    }
}

/// Group messages into assistant-step turns (ranges into `body`).
///
/// A turn starts at each assistant message and includes the following
/// non-assistant messages (its observations). Leading non-assistant
/// messages (e.g. a prior summary) form their own group so they can be
/// dropped cleanly on re-compaction.
pub fn group_turns(body: &[&Value], dialect: &Dialect) -> PyResult<Vec<Range<usize>>> {
    if let Some(group) = dialect.group_turns {
        return group(body);
    }
    let mut turns = vec![];
    let mut start: Option<usize> = None;
    for (i, msg) in body.iter().enumerate() {
        if (dialect.is_assistant)(msg)? {
            if let Some(s) = start {
                turns.push(s..i);
            }
            start = Some(i);
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        turns.push(s..body.len());
    }
    Ok(turns)
}

/// Compact a message list. `Ok(None)` if there is nothing to gain.
pub fn compact(messages: &[&Value], dialect: &Dialect, cfg: &Config) -> PyResult<Option<Compaction>> {
    // Head: everything before the first assistant message...
    let mut first_assistant = None;
    for (i, m) in messages.iter().enumerate() {
        if (dialect.is_assistant)(m)? {
            first_assistant = Some(i);
            break;
        }
    }
    let Some(mut head_len) = first_assistant else {
        return Ok(None);
    };
    // ...minus trailing summary messages (re-compaction must not absorb them)
    // and messages that may not precede the injected user summary.
    while head_len > 0 {
        let last = messages[head_len - 1];
        let trim = (dialect.is_summary_message)(last)? || dialect.trim_from_head.is_some_and(|f| f(last));
        if !trim {
            break;
        }
        head_len -= 1;
    }

    let body = &messages[head_len..];
    let turns = group_turns(body, dialect)?;

    let keep_recent = cfg.keep_recent.max(0) as usize;
    if turns.len() <= keep_recent {
        return Ok(None);
    }
    let split = turns.len() - keep_recent;
    let to_compact = &turns[..split];
    let kept_start = turns.get(split).map_or(body.len(), |r| r.start);

    let mut parts: Vec<String> = vec![];
    for turn in to_compact {
        for msg in &body[turn.clone()] {
            parts.extend((dialect.summarize_message)(msg, cfg)?);
        }
    }

    let summary_text = if parts.is_empty() {
        SUMMARY_HEADER.to_string()
    } else {
        format!("{SUMMARY_HEADER}\n\n{}", parts.join("\n\n---\n\n"))
    };
    let summary = (dialect.user_message)(summary_text);

    let cut = head_len + kept_start;
    let result = Compaction { head_len, summary, cut };
    if result.out_len(messages.len()) >= messages.len() {
        return Ok(None); // no reduction — treat as nothing to compact
    }
    Ok(Some(result))
}
