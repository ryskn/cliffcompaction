//! Display identity for `cliff watch`: which conversation a request belongs
//! to, and whether it is one of its turns. Nothing in the request path
//! branches on these. (In Python they live in proxy.py; they are pure, so
//! the port keeps them with the core where the conformance harness can
//! check them.)

use serde_json::Value;

use crate::dialects::Dialect;
use crate::py;

/// No model turn: a scaffold's own call, not a turn of the conversation.
///
/// A session's opening turn has no model turn either and must not be swept
/// up: it carries at most `dialect.opening_user_messages` user messages, and
/// a session that has already spoken (`seen`) cannot be opening again.
pub fn is_side_call(msgs: &[Value], dialect: &Dialect, seen: bool) -> bool {
    for m in msgs {
        match (dialect.is_assistant)(m) {
            Ok(false) => {}
            // A model turn, or an unknown shape: treat it as a session.
            Ok(true) | Err(_) => return false,
        }
    }
    if seen {
        return true;
    }
    let n_user = msgs.iter().filter(|m| py::get(m, "role").and_then(Value::as_str) == Some("user")).count();
    n_user > dialect.opening_user_messages
}

/// Display id for one request's conversation: the client's session key
/// when it volunteers one (hashed — the blob sits next to account
/// identifiers), else the history's root hash. 12 hex chars.
pub fn session_id(chain: &[String], body: &Value, dialect: &Dialect) -> String {
    let root = chain.first().map_or("", |h| &h[..12]).to_string();
    match (dialect.session_key)(body) {
        Ok(Some(key)) if !key.is_empty() => py::sha256_hex(format!("session:{key}").as_bytes())[..12].to_string(),
        _ => root,
    }
}

/// Output caps a scaffold uses for probes rather than turns.
const CAP_KEYS: [&str; 3] = ["max_tokens", "max_output_tokens", "max_completion_tokens"];

/// A request asking for at most one output token is a liveness or quota
/// probe, not a turn.
pub fn is_probe(body: &Value) -> bool {
    CAP_KEYS.iter().any(|k| py::get(body, k).is_some_and(py::int_le_one))
}
