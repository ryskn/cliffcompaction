//! OpenAI Chat Completions dialect.
//!
//! Message shape notes:
//! - Tool results have their own role ("tool"), so classification is trivial.
//! - Assistant tool calls live in `tool_calls` (function name + JSON-string
//!   arguments); assistant `content` is the visible thought.
//! - System/developer messages found mid-conversation are treated like human
//!   text (kept verbatim, capped) — they are instructions, not observations.

use std::borrow::Cow;

use serde_json::{json, Value};

use super::{Dialect, SUMMARY_HEADER};
use crate::config::Config;
use crate::hashing::{canonical_json, digest_obj, Obj};
use crate::py::{self, PyResult};

// --- canonicalization -------------------------------------------------------

fn type_is(v: &Value, t: &str) -> bool {
    py::get(v, "type").and_then(Value::as_str) == Some(t)
}

fn canon_part(part: &Value) -> PyResult<Obj<'_>> {
    Ok(match py::get(part, "type").and_then(Value::as_str) {
        Some("text") => Obj::List(vec![Obj::Str("text"), Obj::get_or(part, "text", "")]),
        Some("image_url") => {
            let url = py::get_dict_or_empty(part, "image_url")?.map_or(&py::EMPTY_STR, |d| py::get_or_empty(d, "url"));
            Obj::List(vec![Obj::Str("image"), Obj::Owned(py::sha256_hex(py::str_of(url).as_bytes()))])
        }
        _ => Obj::List(vec![Obj::Str("other"), Obj::Owned(canonical_json(part))]),
    })
}

pub(crate) fn content_text(content: Option<&Value>) -> PyResult<Cow<'_, str>> {
    match content {
        Some(Value::String(s)) => Ok(Cow::Borrowed(s)),
        Some(Value::Array(items)) => Ok(Cow::Owned(py::join(
            "\n",
            items.iter().filter(|p| p.is_object() && type_is(p, "text")).map(|p| py::get_or_empty(p, "text")),
        )?)),
        None | Some(Value::Null) => Ok(Cow::Borrowed("")),
        Some(other) => Ok(Cow::Owned(py::repr(other))),
    }
}

pub fn digest_message(msg: &Value) -> PyResult<String> {
    let blocks = match py::get(msg, "content") {
        Some(c @ Value::String(_)) => vec![Obj::List(vec![Obj::Str("text"), Obj::Val(c)])],
        Some(Value::Array(items)) => {
            items.iter().filter(|p| p.is_object()).map(canon_part).collect::<PyResult<Vec<_>>>()?
        }
        _ => vec![],
    };
    let mut tool_calls = vec![];
    for tc in py::iter_or_empty(py::get(msg, "tool_calls"))?.iter().filter(|t| t.is_object()) {
        let function = py::get_dict_or_empty(tc, "function")?;
        let field = |key| function.map_or(Obj::Str(""), |f| Obj::get_or(f, key, ""));
        tool_calls.push(Obj::List(vec![Obj::get_or(tc, "id", ""), field("name"), field("arguments")]));
    }
    Ok(digest_obj(&Obj::List(vec![
        Obj::get_or(msg, "role", ""),
        Obj::List(blocks),
        Obj::List(tool_calls),
        Obj::get_or(msg, "tool_call_id", ""),
        Obj::get_or(msg, "name", ""),
    ])))
}

// --- classification ----------------------------------------------------------

fn role_is(msg: &Value, role: &str) -> bool {
    py::get(msg, "role").and_then(Value::as_str) == Some(role)
}

pub fn is_assistant(msg: &Value) -> PyResult<bool> {
    Ok(role_is(msg, "assistant"))
}

pub fn is_summary_message(msg: &Value) -> PyResult<bool> {
    Ok(role_is(msg, "user") && content_text(py::get(msg, "content"))?.starts_with(SUMMARY_HEADER))
}

/// No conversation id in this dialect. `user` is a per-user id, not a
/// per-session one, so keying on it would merge a user's sessions.
pub fn session_key(_body: &Value) -> PyResult<Option<String>> {
    Ok(None)
}

// --- summarization ------------------------------------------------------------

fn summarize_assistant(msg: &Value, cfg: &Config) -> PyResult<Vec<String>> {
    // Some OpenAI-compatible providers (DeepSeek, Kimi, ...) attach visible
    // reasoning as `reasoning_content`/`reasoning`; keep it as text.
    let mut thinking = String::new();
    if cfg.keep_thinking {
        let raw = py::get_truthy(msg, "reasoning_content").or_else(|| py::get_truthy(msg, "reasoning"));
        if let Some(Value::String(raw)) = raw {
            thinking = py::truncate(py::strip(raw), cfg.thinking_max_chars).into_owned();
        }
    }
    let content = content_text(py::get(msg, "content"))?;
    let thought = py::truncate(py::strip(&content), cfg.thought_max_chars);
    let mut sigs = vec![];
    for tc in py::iter_or_empty(py::get(msg, "tool_calls"))?.iter().filter(|t| t.is_object()) {
        let function = py::get_dict_or_empty(tc, "function")?;
        let args = function.map_or(&py::EMPTY_STR, |f| py::get_or_empty(f, "arguments"));
        let args: Cow<'_, str> = match args {
            Value::String(s) => Cow::Borrowed(s),
            other => Cow::Owned(canonical_json(other)),
        };
        let name = function.and_then(|f| py::get(f, "name")).map_or(Cow::Borrowed("?"), py::str_of);
        sigs.push(format!("[{name}] {}", py::truncate(&args, cfg.cmd_max_chars)));
    }
    let mut lines = vec![];
    if !thinking.is_empty() {
        lines.push(format!("thinking: {thinking}"));
    }
    if !thought.is_empty() {
        lines.push(format!("assistant: {thought}"));
    }
    if !sigs.is_empty() {
        lines.push(sigs.join("\n"));
    }
    Ok(if lines.is_empty() { vec![] } else { vec![lines.join("\n")] })
}

pub fn summarize_message(msg: &Value, cfg: &Config) -> PyResult<Vec<String>> {
    let role = py::get(msg, "role").and_then(Value::as_str);
    match role {
        Some("assistant") => summarize_assistant(msg, cfg),
        Some("tool") => {
            let text = content_text(py::get(msg, "content"))?;
            let text = py::strip(&text);
            if !text.is_empty() && (py::len(text) as i64) <= cfg.result_max_chars {
                Ok(vec![format!("result: {text}")])
            } else {
                Ok(vec![])
            }
        }
        Some("user") => {
            let text = content_text(py::get(msg, "content"))?;
            let text = py::strip(&text);
            if text.is_empty() || text.starts_with(SUMMARY_HEADER) {
                return Ok(vec![]);
            }
            let text = py::strip_task_notifications(text);
            let text = py::strip(&text);
            if text.is_empty() {
                return Ok(vec![]);
            }
            Ok(vec![format!("user: {}", py::truncate(text, cfg.human_max_chars))])
        }
        Some(role @ ("system" | "developer")) => {
            let text = content_text(py::get(msg, "content"))?;
            let text = py::strip(&text);
            if text.is_empty() {
                return Ok(vec![]);
            }
            Ok(vec![format!("{role}: {}", py::truncate(text, cfg.human_max_chars))])
        }
        _ => Ok(vec![]),
    }
}

pub fn user_message(text: String) -> Value {
    json!({"role": "user", "content": text})
}

pub static DIALECT: Dialect = Dialect {
    name: "openai",
    digest_message,
    is_assistant,
    summarize_message,
    user_message,
    is_summary_message,
    session_key,
    messages_key: "messages",
    group_turns: None,
    trim_from_head: None,
    opening_user_messages: 1,
};
