//! API dialects: Anthropic Messages, OpenAI Chat Completions, OpenAI Responses.

use std::ops::Range;

use serde_json::Value;

use crate::config::Config;
use crate::py::PyResult;

pub mod anthropic;
pub mod openai_chat;
pub mod openai_responses;

/// The marker by which a previously injected summary is recognized (and
/// dropped on re-compaction). Keep it stable: changing a byte breaks
/// recognition of summaries produced by earlier versions — including the
/// Python implementation's, which share stores' worth of live sessions.
pub const SUMMARY_HEADER: &str =
    "The following is a summary of your previous actions (long observations omitted):";

/// Operations over raw request-body message dicts. Kept messages are never
/// rebuilt, only selected.
///
/// A table of plain function pointers, like the Python dataclass of
/// callables it ports: static dispatch-free, `Copy`, and a test can derive a
/// variant with one operation swapped (`Dialect { session_key: f, ..X }`).
/// Operations that can hit a Python exception on malformed input return
/// [`PyResult`].
#[derive(Clone, Copy)]
pub struct Dialect {
    pub name: &'static str,
    /// Canonical content digest of one message (volatile fields excluded).
    pub digest_message: fn(&Value) -> PyResult<String>,
    /// True if this message is a model turn (starts an assistant-step turn).
    pub is_assistant: fn(&Value) -> PyResult<bool>,
    /// Summarize one message into zero or more summary parts.
    pub summarize_message: fn(&Value, &Config) -> PyResult<Vec<String>>,
    /// Build the synthetic summary message.
    pub user_message: fn(String) -> Value,
    /// True if this message IS a previously injected cliff summary.
    pub is_summary_message: fn(&Value) -> PyResult<bool>,
    /// A stable per-conversation key the CLIENT supplied, or None. Display
    /// only (`cliff watch`); nothing in the request path reads it.
    pub session_key: fn(&Value) -> PyResult<Option<String>>,
    /// Request-body key holding the message list.
    pub messages_key: &'static str,
    /// Optional dialect-specific turn grouping (ranges into the input, a
    /// partition in order). None = every assistant message starts a turn.
    pub group_turns: Option<fn(&[&Value]) -> PyResult<Vec<Range<usize>>>>,
    /// Messages that must not sit at the end of the verbatim head.
    pub trim_from_head: Option<fn(&Value) -> bool>,
    /// How many user messages a session's opening turn may carry. Display only.
    pub opening_user_messages: usize,
}

/// Pick the dialect for a request path, or None for raw passthrough.
///
/// /v1/messages/count_tokens and /responses/compact are deliberately NOT
/// matched: they pass through verbatim.
pub fn detect(path: &str) -> Option<&'static Dialect> {
    let p = path.trim_end_matches('/');
    if p.ends_with("/v1/messages") || p.ends_with("/messages") {
        Some(&anthropic::DIALECT)
    } else if p.ends_with("/chat/completions") {
        Some(&openai_chat::DIALECT)
    } else if p.ends_with("/responses") {
        Some(&openai_responses::DIALECT)
    } else {
        None
    }
}

/// Look a dialect up by its `name` (conformance harness, debug dumps).
pub fn by_name(name: &str) -> Option<&'static Dialect> {
    [&anthropic::DIALECT, &openai_chat::DIALECT, &openai_responses::DIALECT]
        .into_iter()
        .find(|d| d.name == name)
}
