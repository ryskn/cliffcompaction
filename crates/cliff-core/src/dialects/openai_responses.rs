//! OpenAI Responses API dialect (stateless clients: full `input` resent).
//!
//! Item shape notes:
//! - History rides in `input` (not `messages`); the system prompt rides in
//!   the separate `instructions` field, which is never touched here.
//! - A model step spans MULTIPLE items: [reasoning?] then
//!   message/function_call, followed by function_call_output items.
//!   Providers reject a function_call whose paired reasoning item is
//!   missing, so turn grouping keeps contiguous model-output runs together.
//! - Reasoning items carry `encrypted_content` (opaque; dropped with their
//!   turn) and optionally readable `summary` blocks, which fold into
//!   summaries as thinking text.
//! - Item `id` and `status` are excluded from digests.

use std::borrow::Cow;
use std::ops::Range;

use serde_json::{json, Value};

use super::{Dialect, SUMMARY_HEADER};
use crate::config::Config;
use crate::hashing::{canonical_json, digest_obj, Obj};
use crate::py::{self, PyErr, PyResult};

/// Item types produced by the model. A contiguous run of these starts a
/// turn; an assistant-role message counts via `is_model_output`.
const MODEL_ITEM_TYPES: &[&str] = &[
    "reasoning",
    "function_call",
    "custom_tool_call",
    "local_shell_call",
    "web_search_call",
    "tool_search_call",
    "image_generation_call",
];

const TEXT_TYPES: &[&str] = &["input_text", "output_text", "text"];

/// `_item_type(item)`: the `type` field when truthy, else derived from the
/// presence of `role`. A truthy non-string `type` is carried as `Weird` so
/// the operations Python would raise on (`.endswith`, set membership of an
/// unhashable value) can raise here too.
enum ItemType<'a> {
    Name(&'a str),
    Weird(&'a Value),
}

fn item_type(item: &Value) -> ItemType<'_> {
    match py::get_truthy(item, "type") {
        Some(Value::String(s)) => ItemType::Name(s),
        Some(v) => ItemType::Weird(v),
        None if py::get(item, "role").is_some() => ItemType::Name("message"),
        None => ItemType::Name("other"),
    }
}

impl ItemType<'_> {
    fn is(&self, name: &str) -> bool {
        matches!(self, ItemType::Name(n) if *n == name)
    }

    /// `t.endswith(suffix)`: AttributeError for a non-string type.
    fn endswith(&self, suffix: &str) -> PyResult<bool> {
        match self {
            ItemType::Name(n) => Ok(n.ends_with(suffix)),
            ItemType::Weird(v) => Err(PyErr::attribute_error(format!(
                "'{}' object has no attribute 'endswith'",
                py::type_name(v)
            ))),
        }
    }
}

fn is_model_output(item: &Value) -> PyResult<bool> {
    match item_type(item) {
        ItemType::Name("message") => Ok(py::get(item, "role").and_then(Value::as_str) == Some("assistant")),
        ItemType::Name(n) => Ok(MODEL_ITEM_TYPES.contains(&n)),
        // `t in {...}` hashes t: lists and dicts are unhashable.
        ItemType::Weird(v @ (Value::Array(_) | Value::Object(_))) => {
            Err(PyErr::type_error(format!("unhashable type: '{}'", py::type_name(v))))
        }
        ItemType::Weird(_) => Ok(false),
    }
}

fn is_text_part(p: &Value) -> bool {
    p.is_object() && py::get(p, "type").and_then(Value::as_str).is_some_and(|t| TEXT_TYPES.contains(&t))
}

fn content_text(content: Option<&Value>) -> PyResult<Cow<'_, str>> {
    match content {
        Some(Value::String(s)) => Ok(Cow::Borrowed(s)),
        Some(Value::Array(items)) => Ok(Cow::Owned(py::join(
            "\n",
            items.iter().filter(|p| is_text_part(p)).map(|p| py::get_or_empty(p, "text")),
        )?)),
        None | Some(Value::Null) => Ok(Cow::Borrowed("")),
        Some(other) => Ok(Cow::Owned(py::repr(other))),
    }
}

/// `x if isinstance(x, str) else canonical_json(x)`
fn str_or_canonical(v: &Value) -> Cow<'_, str> {
    match v {
        Value::String(s) => Cow::Borrowed(s),
        other => Cow::Owned(canonical_json(other)),
    }
}

// --- canonicalization -------------------------------------------------------

fn canon_content(content: Option<&Value>) -> Vec<Obj<'_>> {
    match content {
        Some(c @ Value::String(_)) => vec![Obj::List(vec![Obj::Str("text"), Obj::Val(c)])],
        Some(Value::Array(items)) => items
            .iter()
            .filter(|p| p.is_object())
            .map(|p| {
                if is_text_part(p) {
                    Obj::List(vec![Obj::Str("text"), Obj::get_or(p, "text", "")])
                } else {
                    Obj::List(vec![Obj::Str("other"), Obj::Owned(canonical_json(p))])
                }
            })
            .collect(),
        _ => vec![],
    }
}

pub fn digest_message(item: &Value) -> PyResult<String> {
    let t = item_type(item);
    let obj = if t.is("message") {
        Obj::List(vec![
            Obj::Str("message"),
            Obj::get_or(item, "role", ""),
            Obj::List(canon_content(py::get(item, "content"))),
        ])
    } else if t.is("reasoning") {
        Obj::List(vec![
            Obj::Str("reasoning"),
            Obj::get_truthy_or(item, "encrypted_content", ""),
            Obj::Owned(py::get_truthy(item, "summary").map_or_else(|| "[]".into(), canonical_json)),
        ])
    } else if t.is("function_call_output") {
        let out = py::get(item, "output").map_or_else(|| Cow::Borrowed("null"), str_or_canonical);
        Obj::List(vec![Obj::Str("function_call_output"), Obj::get_or(item, "call_id", ""), Obj::Owned(out.into_owned())])
    } else if t.endswith("_call")? || t.is("function_call") {
        let ItemType::Name(name) = t else { unreachable!("endswith succeeded") };
        let args = py::get(item, "arguments").map_or(Cow::Borrowed(""), str_or_canonical);
        Obj::List(vec![
            Obj::Str(name),
            Obj::get_or(item, "call_id", ""),
            Obj::get_or(item, "name", ""),
            Obj::Owned(args.into_owned()),
        ])
    } else {
        // Unknown item types: canonical dump minus volatile fields.
        Obj::List(vec![Obj::Str("other"), Obj::Owned(py::canonical_json_without(item, &["id", "status"]))])
    };
    Ok(digest_obj(&obj))
}

// --- classification ----------------------------------------------------------

pub fn is_assistant(item: &Value) -> PyResult<bool> {
    is_model_output(item)
}

pub fn is_summary_message(item: &Value) -> PyResult<bool> {
    Ok(item_type(item).is("message")
        && py::get(item, "role").and_then(Value::as_str) == Some("user")
        && content_text(py::get(item, "content"))?.starts_with(SUMMARY_HEADER))
}

/// A turn starts at a model-output item whose predecessor is not one, so a
/// step's [reasoning, message/function_call] run stays in one turn with its
/// function_call_output observations. Leading non-model items form their
/// own group (same contract as the base grouping).
pub fn group_turns(items: &[&Value]) -> PyResult<Vec<Range<usize>>> {
    let mut turns = vec![];
    let mut start: Option<usize> = None;
    let mut prev_model = false;
    for (i, item) in items.iter().enumerate() {
        let model = is_model_output(item)?;
        if model && !prev_model {
            if let Some(s) = start {
                turns.push(s..i);
            }
            start = Some(i);
        } else if start.is_none() {
            start = Some(i);
        }
        prev_model = model;
    }
    if let Some(s) = start {
        turns.push(s..items.len());
    }
    Ok(turns)
}

/// `prompt_cache_key` (Codex sends its thread id there), or
/// `client_metadata.thread_id` — two spellings of one fact. Not `user`:
/// that is a per-user id.
pub fn session_key(body: &Value) -> PyResult<Option<String>> {
    if let Some(Value::String(key)) = py::get(body, "prompt_cache_key") {
        if !key.is_empty() {
            return Ok(Some(key.clone()));
        }
    }
    if let Some(meta @ Value::Object(_)) = py::get(body, "client_metadata") {
        if let Some(Value::String(tid)) = py::get(meta, "thread_id") {
            if !tid.is_empty() {
                return Ok(Some(tid.clone()));
            }
        }
    }
    Ok(None)
}

// --- summarization ------------------------------------------------------------

pub fn summarize_message(item: &Value, cfg: &Config) -> PyResult<Vec<String>> {
    let t = item_type(item);
    if t.is("message") {
        let role = py::get(item, "role").map_or(Cow::Borrowed(""), py::str_of);
        let text = content_text(py::get(item, "content"))?;
        let text = py::strip(&text);
        if text.is_empty() {
            return Ok(vec![]);
        }
        if role == "assistant" {
            return Ok(vec![format!("assistant: {}", py::truncate(text, cfg.thought_max_chars))]);
        }
        if text.starts_with(SUMMARY_HEADER) {
            return Ok(vec![]);
        }
        if role == "user" && py::get(item, "role").is_some_and(Value::is_string) {
            let text = py::strip_task_notifications(text);
            let text = py::strip(&text);
            if text.is_empty() {
                return Ok(vec![]);
            }
            return Ok(vec![format!("user: {}", py::truncate(text, cfg.human_max_chars))]);
        }
        return Ok(vec![format!("{role}: {}", py::truncate(text, cfg.human_max_chars))]);
    }
    if t.is("reasoning") {
        if !cfg.keep_thinking {
            return Ok(vec![]);
        }
        let parts = py::iter_or_empty(py::get(item, "summary"))?;
        let text = py::join("\n", parts.iter().filter(|p| p.is_object()).map(|p| py::get_or_empty(p, "text")))?;
        let text = py::strip(&text);
        if text.is_empty() {
            return Ok(vec![]); // encrypted-only reasoning: nothing foldable
        }
        return Ok(vec![format!("thinking: {}", py::truncate(text, cfg.thinking_max_chars))]);
    }
    if t.is("function_call_output") {
        let out: Cow<'_, str> = match py::get(item, "output") {
            Some(Value::String(s)) => Cow::Borrowed(s),
            other => {
                let text = content_text(other)?;
                if text.is_empty() {
                    Cow::Owned(other.map_or_else(|| "null".into(), canonical_json))
                } else {
                    text
                }
            }
        };
        let out = py::strip(&out);
        if !out.is_empty() && (py::len(out) as i64) <= cfg.result_max_chars {
            return Ok(vec![format!("result: {out}")]);
        }
        return Ok(vec![]);
    }
    if t.endswith("_call")? {
        let ItemType::Name(type_name) = t else { unreachable!("endswith succeeded") };
        let args = py::get(item, "arguments").map_or(Cow::Borrowed(""), str_or_canonical);
        let name = py::get_truthy(item, "name").map_or(Cow::Borrowed(type_name), py::str_of);
        return Ok(vec![format!("[{name}] {}", py::truncate(&args, cfg.cmd_max_chars))]);
    }
    Ok(vec![])
}

pub fn user_message(text: String) -> Value {
    json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    })
}

pub static DIALECT: Dialect = Dialect {
    name: "openai-responses",
    digest_message,
    is_assistant,
    summarize_message,
    user_message,
    is_summary_message,
    session_key,
    messages_key: "input",
    group_turns: Some(group_turns),
    trim_from_head: None,
    // Codex prepends an <environment_context> user message to the person's
    // own first message, so a genuine opening turn carries two.
    opening_user_messages: 2,
};
