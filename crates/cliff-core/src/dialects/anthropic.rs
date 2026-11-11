//! Anthropic Messages API dialect.
//!
//! Message shape notes:
//! - `system` is a top-level request field, not a message — it passes
//!   through untouched and is not part of the hash chain.
//! - Tool results are user-role messages whose content blocks are typed
//!   `tool_result`; human prompts are user-role messages with `text` blocks.
//!   A single user message may mix both.
//! - `cache_control` markers move between requests (clients pin them to the
//!   newest message), so they are excluded from the canonical digest.

use std::borrow::Cow;

use serde_json::{json, Value};

use super::{Dialect, SUMMARY_HEADER};
use crate::config::Config;
use crate::hashing::{canonical_json, digest_obj, Obj};
use crate::py::{self, PyResult, EMPTY_STR};

// --- canonicalization -------------------------------------------------------

fn type_is(v: &Value, t: &str) -> bool {
    py::get(v, "type").and_then(Value::as_str) == Some(t)
}

/// Extract the text of a tool_result's content (str or block list).
fn result_text(content: Option<&Value>) -> PyResult<Cow<'_, str>> {
    match content {
        None | Some(Value::Null) => Ok(Cow::Borrowed("")),
        Some(Value::String(s)) => Ok(Cow::Borrowed(s)),
        Some(Value::Array(items)) => {
            let parts = items.iter().filter_map(|p| match p {
                Value::Object(_) if type_is(p, "text") => Some(py::get_or_empty(p, "text")),
                Value::String(_) => Some(p),
                _ => None,
            });
            Ok(Cow::Owned(py::join("\n", parts)?))
        }
        Some(other) => Ok(Cow::Owned(py::repr(other))),
    }
}

fn input_json(block: &Value) -> String {
    match py::get(block, "input") {
        Some(v) => canonical_json(v),
        None => "{}".into(),
    }
}

fn canon_block(block: &Value) -> PyResult<Obj<'_>> {
    let t = py::get(block, "type").and_then(Value::as_str).unwrap_or("");
    Ok(match t {
        "text" => Obj::List(vec![Obj::Str("text"), Obj::get_or(block, "text", "")]),
        "tool_use" => Obj::List(vec![
            Obj::Str("tool_use"),
            Obj::get_or(block, "id", ""),
            Obj::get_or(block, "name", ""),
            Obj::Owned(input_json(block)),
        ]),
        "tool_result" => Obj::List(vec![
            Obj::Str("tool_result"),
            Obj::get_or(block, "tool_use_id", ""),
            Obj::Owned(result_text(py::get(block, "content"))?.into_owned()),
            Obj::Bool(py::get(block, "is_error").is_some_and(py::truthy)),
        ]),
        // Content only — the signature is provider metadata.
        "thinking" => Obj::List(vec![Obj::Str("thinking"), Obj::get_or(block, "thinking", "")]),
        "redacted_thinking" => Obj::List(vec![Obj::Str("redacted_thinking"), Obj::get_or(block, "data", "")]),
        "image" | "document" => {
            let src = py::get_dict_or_empty(block, "source")?;
            let payload = src
                .and_then(|s| py::get_truthy(s, "data").or_else(|| py::get_truthy(s, "url")))
                .unwrap_or(&EMPTY_STR);
            let h = py::sha256_hex(py::str_of(payload).as_bytes());
            let src_type = match src {
                Some(s) => Obj::get_or(s, "type", ""),
                None => Obj::Str(""),
            };
            Obj::List(vec![Obj::Str(t), src_type, Obj::Owned(h)])
        }
        // Unknown block type: canonical dump minus volatile fields.
        _ => Obj::List(vec![Obj::Str("other"), Obj::Owned(py::canonical_json_without(block, &["cache_control"]))]),
    })
}

pub fn digest_message(msg: &Value) -> PyResult<String> {
    let blocks = match py::get(msg, "content") {
        Some(c @ Value::String(_)) => vec![Obj::List(vec![Obj::Str("text"), Obj::Val(c)])],
        Some(Value::Array(items)) => items
            .iter()
            .filter(|b| b.is_object())
            .map(canon_block)
            .collect::<PyResult<Vec<_>>>()?,
        _ => vec![],
    };
    Ok(digest_obj(&Obj::List(vec![Obj::get_or(msg, "role", ""), Obj::List(blocks)])))
}

// --- classification ----------------------------------------------------------

fn role_is(msg: &Value, role: &str) -> bool {
    py::get(msg, "role").and_then(Value::as_str) == Some(role)
}

pub fn is_assistant(msg: &Value) -> PyResult<bool> {
    Ok(role_is(msg, "assistant"))
}

pub fn is_summary_message(msg: &Value) -> PyResult<bool> {
    if !role_is(msg, "user") {
        return Ok(false);
    }
    match py::get(msg, "content") {
        Some(Value::String(s)) => Ok(s.starts_with(SUMMARY_HEADER)),
        Some(Value::Array(items)) => {
            for b in items {
                if b.is_object() && type_is(b, "text") {
                    return match py::get_truthy(b, "text") {
                        None => Ok(false),
                        Some(text) => py::startswith_value(text, SUMMARY_HEADER),
                    };
                }
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

/// The conversation id the client volunteered, if any.
///
/// `metadata.user_id` is a documented Anthropic field: an opaque
/// client-supplied string. Claude Code puts a JSON object there carrying a
/// `session_id`. Anything that is not that exact shape yields None. A bare
/// string is NOT used — it is a per-USER id, and keying on it would merge
/// every one of a user's sessions into a single row.
pub fn session_key(body: &Value) -> PyResult<Option<String>> {
    let Some(meta @ Value::Object(_)) = py::get(body, "metadata") else {
        return Ok(None);
    };
    let Some(raw) = py::get(meta, "user_id").and_then(Value::as_str) else {
        return Ok(None);
    };
    if !raw.starts_with('{') {
        return Ok(None);
    }
    let Ok(parsed) = serde_json::from_str::<Value>(raw) else {
        return Ok(None);
    };
    Ok(match py::get(&parsed, "session_id") {
        Some(Value::String(sid)) if !sid.is_empty() => Some(sid.clone()),
        _ => None,
    })
}

// --- summarization ------------------------------------------------------------

/// `"\n".join(x for x in items if x.strip()).strip()`
fn join_nonblank(items: &[&Value]) -> PyResult<String> {
    let mut kept = Vec::with_capacity(items.len());
    for x in items {
        if !py::strip_value(x)?.is_empty() {
            kept.push(*x);
        }
    }
    Ok(py::strip(&py::join("\n", kept)?).to_string())
}

fn summarize_assistant(msg: &Value, cfg: &Config) -> PyResult<Vec<String>> {
    let mut texts: Vec<&Value> = vec![];
    let mut thinkings: Vec<&Value> = vec![];
    let mut sigs: Vec<String> = vec![];
    match py::get(msg, "content") {
        Some(c @ Value::String(_)) => texts.push(c),
        Some(Value::Array(items)) => {
            for b in items.iter().filter(|b| b.is_object()) {
                match py::get(b, "type").and_then(Value::as_str) {
                    Some("text") => texts.push(py::get_or_empty(b, "text")),
                    // Kept as TEXT: the signed block itself is never re-sent
                    // from the compacted region, only its content.
                    Some("thinking") if cfg.keep_thinking => thinkings.push(py::get_or_empty(b, "thinking")),
                    Some("tool_use") => {
                        let name = py::get(b, "name").map_or(Cow::Borrowed("?"), py::str_of);
                        sigs.push(format!("[{name}] {}", py::truncate(&input_json(b), cfg.cmd_max_chars)));
                    }
                    // redacted_thinking (encrypted) / images: dropped from summaries
                    _ => {}
                }
            }
        }
        _ => {}
    }
    let mut lines: Vec<String> = vec![];
    let thinking = join_nonblank(&thinkings)?;
    let thinking = py::truncate(&thinking, cfg.thinking_max_chars);
    if !thinking.is_empty() {
        lines.push(format!("thinking: {thinking}"));
    }
    let thought = join_nonblank(&texts)?;
    let thought = py::truncate(&thought, cfg.thought_max_chars);
    if !thought.is_empty() {
        lines.push(format!("assistant: {thought}"));
    }
    if !sigs.is_empty() {
        lines.push(sigs.join("\n"));
    }
    Ok(if lines.is_empty() { vec![] } else { vec![lines.join("\n")] })
}

fn human_part(text: &str, cfg: &Config) -> Option<String> {
    let text = py::strip_task_notifications(text);
    let text = py::strip(&text);
    (!text.is_empty()).then(|| format!("user: {}", py::truncate(text, cfg.human_max_chars)))
}

/// Human text: verbatim (sanity-capped). Tool results: keep iff short.
/// A prior cliff summary: dropped entirely (never merged forward).
fn summarize_user(msg: &Value, cfg: &Config) -> PyResult<Vec<String>> {
    let mut parts = vec![];
    match py::get(msg, "content") {
        Some(Value::String(content)) => {
            if !content.starts_with(SUMMARY_HEADER) {
                parts.extend(human_part(content, cfg));
            }
        }
        Some(Value::Array(items)) => {
            for b in items.iter().filter(|b| b.is_object()) {
                match py::get(b, "type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = py::get_or_empty(b, "text");
                        if py::startswith_value(text, SUMMARY_HEADER)? {
                            continue;
                        }
                        // startswith succeeded, so it is a str.
                        parts.extend(human_part(text.as_str().unwrap_or(""), cfg));
                    }
                    Some("tool_result") => {
                        let text = result_text(py::get(b, "content"))?;
                        let text = py::strip(&text);
                        // long or empty observations dropped entirely
                        if !text.is_empty() && (py::len(text) as i64) <= cfg.result_max_chars {
                            parts.push(format!("result: {text}"));
                        }
                    }
                    // images/documents in the compacted region: dropped
                    _ => {}
                }
            }
        }
        _ => {}
    }
    Ok(parts)
}

pub fn summarize_message(msg: &Value, cfg: &Config) -> PyResult<Vec<String>> {
    if role_is(msg, "assistant") {
        return summarize_assistant(msg, cfg);
    }
    if role_is(msg, "user") {
        return summarize_user(msg, cfg);
    }
    if role_is(msg, "system") {
        // In-array system directives. Content-ful ones fold like
        // instructions; directive-only forms (content: []) yield nothing.
        let content = py::get(msg, "content");
        let text: Cow<'_, str> = match content {
            Some(Value::String(s)) => Cow::Borrowed(s),
            _ => {
                let blocks = py::iter_or_empty(content)?;
                Cow::Owned(py::join(
                    "\n",
                    blocks.iter().filter(|b| b.is_object() && type_is(b, "text")).map(|b| py::get_or_empty(b, "text")),
                )?)
            }
        };
        let text = py::strip(&text);
        if !text.is_empty() {
            return Ok(vec![format!("system: {}", py::truncate(text, cfg.human_max_chars))]);
        }
    }
    Ok(vec![])
}

fn is_system(msg: &Value) -> bool {
    role_is(msg, "system")
}

pub fn user_message(text: String) -> Value {
    json!({"role": "user", "content": text})
}

pub static DIALECT: Dialect = Dialect {
    name: "anthropic",
    digest_message,
    is_assistant,
    summarize_message,
    user_message,
    is_summary_message,
    session_key,
    messages_key: "messages",
    group_turns: None,
    // A content-ful system message must precede an assistant message or end
    // the array; it may not precede the injected user-role summary.
    trim_from_head: Some(is_system),
    opening_user_messages: 1,
};
