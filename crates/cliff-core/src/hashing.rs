//! Canonical digests and the prefix hash chain.
//!
//! h_0 = H(digest_0); h_i = H(h_{i-1} || digest_i). chain[i] identifies
//! messages[0..i] independent of volatile serialization details, so prefix
//! identity is a single map lookup.
//!
//! The Python implementation builds a nested list per message and hashes its
//! canonical JSON. [`Obj`] is that list, borrowed: it streams the same bytes
//! into SHA-256 without copying message text.

use serde_json::Value;

use crate::py::{self, HashSink, Sink, CANONICAL};

pub use crate::py::canonical_json;

/// One node of the canonical structure a dialect hashes per message.
pub enum Obj<'a> {
    Str(&'a str),
    Owned(String),
    Val(&'a Value),
    Bool(bool),
    List(Vec<Obj<'a>>),
}

impl<'a> Obj<'a> {
    /// `d.get(key, default)` where `d` is a dict.
    pub fn get_or(d: &'a Value, key: &str, default: &'static str) -> Obj<'a> {
        match py::get(d, key) {
            Some(v) => Obj::Val(v),
            None => Obj::Str(default),
        }
    }

    /// `d.get(key) or default`.
    pub fn get_truthy_or(d: &'a Value, key: &str, default: &'static str) -> Obj<'a> {
        match py::get_truthy(d, key) {
            Some(v) => Obj::Val(v),
            None => Obj::Str(default),
        }
    }

    fn write<S: Sink>(&self, sink: &mut S) {
        match self {
            Obj::Str(s) => py::write_str(s, false, sink),
            Obj::Owned(s) => py::write_str(s, false, sink),
            Obj::Val(v) => py::write_value(v, CANONICAL, sink, 0),
            Obj::Bool(true) => sink.put("true"),
            Obj::Bool(false) => sink.put("false"),
            Obj::List(items) => {
                sink.put("[");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        sink.put(",");
                    }
                    item.write(sink);
                }
                sink.put("]");
            }
        }
    }
}

/// `sha256(canonical_json(obj)).hexdigest()`.
pub fn digest_obj(obj: &Obj<'_>) -> String {
    let mut sink = HashSink::default();
    obj.write(&mut sink);
    sink.hexdigest()
}

/// `sha256(canonical_json(value)).hexdigest()`.
pub fn digest_value(v: &Value) -> String {
    digest_obj(&Obj::Val(v))
}

/// chain[i] = hash identifying messages[0..i] as a sequence.
///
/// Inherently sequential (each link depends on the previous one), but each
/// step hashes 128 bytes; the per-message digests feeding it are where the
/// work is, and those are computed in parallel by the engine.
pub fn chain_hashes(message_digests: &[String]) -> Vec<String> {
    let mut chain = Vec::with_capacity(message_digests.len());
    let mut prev = String::new();
    let mut buf = String::with_capacity(128);
    for d in message_digests {
        buf.clear();
        buf.push_str(&prev);
        buf.push_str(d);
        prev = py::sha256_hex(buf.as_bytes());
        chain.push(prev.clone());
    }
    chain
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn obj_matches_canonical_json_of_the_equivalent_value() {
        let v = json!({"z": 1, "a": "é\n"});
        let obj = Obj::List(vec![Obj::Str("user"), Obj::Val(&v), Obj::Bool(true), Obj::Owned("x".into())]);
        let as_value = json!(["user", {"a": "é\n", "z": 1}, true, "x"]);
        assert_eq!(digest_obj(&obj), py::sha256_hex(canonical_json(&as_value).as_bytes()));
    }

    #[test]
    fn chain_prefix_property() {
        let d = vec!["a".to_string(), "b".to_string()];
        let c2 = chain_hashes(&d);
        let c1 = chain_hashes(&d[..1]);
        assert_eq!(c2[0], c1[0]);
        assert_ne!(c2[1], c2[0]);
        assert_eq!(c1[0], py::sha256_hex(b"a"));
    }
}
