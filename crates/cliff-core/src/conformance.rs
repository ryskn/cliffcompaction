//! Conformance driver: runs one case of the cross-implementation suite and
//! reports what the Rust port computed, in the schema the Python oracle
//! (`scripts/conformance.py`) reports for the same case.
//!
//! The suite is the migration's safety net: the Python implementation is
//! the specification, cases are generated against it (golden files under
//! `tests/golden/` plus on-demand differential fuzzing), and any field that
//! differs is a porting bug — or a deliberate, documented divergence.
//!
//! Results never carry raw outputs, only digests of them (`out` is the
//! SHA-256 of the exact outgoing body bytes), which keeps golden files small
//! while still requiring byte-identical output.

use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::cliff::compact;
use crate::config::Config;
use crate::dialects::{self, Dialect};
use crate::engine::{billable_chars, Engine, RequestCtx};
use crate::identity;
use crate::images::tokens_for_payload;
use crate::py::{self, PyResult, DUMPS_ASCII, DUMPS_UNICODE};
use crate::store::PrefixStore;

const ERROR: &str = "<error>";

fn or_error<T: Into<Value>>(r: PyResult<T>) -> Value {
    r.map_or_else(|_| Value::from(ERROR), Into::into)
}

/// Build a Config from the case's overrides (unknown keys ignored).
pub fn config_from(v: Option<&Value>) -> Config {
    let mut cfg = Config::default();
    let Some(Value::Object(m)) = v else { return cfg };
    let int = |k: &str, d: i64| m.get(k).and_then(Value::as_i64).unwrap_or(d);
    let flag = |k: &str, d: bool| m.get(k).and_then(Value::as_bool).unwrap_or(d);
    cfg.threshold_tokens = int("threshold_tokens", cfg.threshold_tokens);
    cfg.keep_recent = int("keep_recent", cfg.keep_recent);
    cfg.thought_max_chars = int("thought_max_chars", cfg.thought_max_chars);
    cfg.cmd_max_chars = int("cmd_max_chars", cfg.cmd_max_chars);
    cfg.result_max_chars = int("result_max_chars", cfg.result_max_chars);
    cfg.human_max_chars = int("human_max_chars", cfg.human_max_chars);
    cfg.thinking_max_chars = int("thinking_max_chars", cfg.thinking_max_chars);
    cfg.keep_thinking = flag("keep_thinking", cfg.keep_thinking);
    cfg.strict = flag("strict", cfg.strict);
    cfg.store_max_entries = int("store_max_entries", cfg.store_max_entries as i64) as usize;
    cfg.store_max_bytes = int("store_max_bytes", cfg.store_max_bytes as i64) as usize;
    cfg
}

fn dialect_of(case: &Value) -> &'static Dialect {
    case.get("dialect").and_then(Value::as_str).and_then(dialects::by_name).unwrap_or(&dialects::anthropic::DIALECT)
}

/// Run one case; returns `{"id": ..., ...results}`.
pub fn run_case(case: &Value) -> Value {
    let mut out = match case.get("kind").and_then(Value::as_str) {
        Some("value") => match case.get("value_json").and_then(Value::as_str) {
            Some(raw) => match serde_json::from_str::<Value>(raw) {
                Ok(v) => run_value(&v),
                Err(_) => json!({"parse_error": true}),
            },
            None => run_value(&case["value"]),
        },
        Some("payload") => run_payload(case["payload"].as_str().unwrap_or("")),
        Some("message") => run_message(case),
        Some("compact") => run_compact(case),
        Some("session") => run_session(case),
        other => json!({"unknown_kind": other}),
    };
    out.as_object_mut().unwrap().insert("id".into(), case.get("id").cloned().unwrap_or(Value::Null));
    out
}

fn run_value(v: &Value) -> Value {
    json!({
        "canonical": py::canonical_json(v),
        "dumps": py::dumps(v, DUMPS_UNICODE),
        "dumps_ascii": py::dumps(v, DUMPS_ASCII),
        "dumps_len": py::dumps_len(v, DUMPS_UNICODE),
        "repr": py::repr(v),
        "truthy": py::truthy(v),
        "probe_cap": py::int_le_one(v),
    })
}

fn run_payload(payload: &str) -> Value {
    json!({"tokens": or_error(tokens_for_payload(payload))})
}

fn run_message(case: &Value) -> Value {
    let d = dialect_of(case);
    let cfg = config_from(case.get("config"));
    let m = &case["message"];
    json!({
        "digest": or_error((d.digest_message)(m)),
        "is_assistant": or_error((d.is_assistant)(m)),
        "is_summary": or_error((d.is_summary_message)(m)),
        "summary": or_error((d.summarize_message)(m, &cfg)),
        "billable": or_error(billable_chars(m)),
    })
}

fn run_compact(case: &Value) -> Value {
    let d = dialect_of(case);
    let cfg = config_from(case.get("config"));
    let msgs: Vec<&Value> = case["messages"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
    let result = match compact(&msgs, d, &cfg) {
        Err(_) => Value::from(ERROR),
        Ok(None) => Value::Null,
        Ok(Some(c)) => json!({
            "head_len": c.head_len,
            "cut": c.cut,
            "summary": py::dumps(&c.summary, DUMPS_UNICODE),
        }),
    };
    json!({"result": result})
}

fn sha(text: &str) -> String {
    py::sha256_hex(text.as_bytes())
}

fn snapshot(ctx: &RequestCtx) -> Map<String, Value> {
    let out = ctx.outgoing().to_json();
    let mut m = Map::new();
    m.insert("est_out".into(), ctx.est_tokens_out.into());
    m.insert("modified".into(), ctx.modified.into());
    m.insert("compacted".into(), ctx.compacted.into());
    m.insert("rung".into(), ctx.rung.into());
    m.insert("base_cut".into(), ctx.base_cut.into());
    m.insert("base_head".into(), ctx.base_head.into());
    m.insert("steps".into(), ctx.chain_steps.into());
    m.insert("fp".into(), ctx.summary_fp.clone().into());
    m.insert("out_msgs".into(), ctx.out_msgs.into());
    m.insert("out".into(), sha(&out).into());
    m.insert("out_len".into(), py::len(&out).into());
    m
}

/// Build the request body for `take` history messages plus `extra`,
/// placing the message list where the base body has its key.
pub fn session_body(case: &Value, d: &Dialect, take: usize, extra: &[Value]) -> Value {
    let mut body = case.get("body").cloned().unwrap_or_else(|| json!({}));
    let history = case["history"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let mut msgs: Vec<Value> = history[..take.min(history.len())].to_vec();
    msgs.extend_from_slice(extra);
    body.as_object_mut().unwrap().insert(d.messages_key.into(), Value::Array(msgs));
    body
}

fn run_session(case: &Value) -> Value {
    let d = dialect_of(case);
    let cfg = config_from(case.get("config"));
    let store = Arc::new(PrefixStore::new(cfg.store_max_entries, cfg.store_max_bytes));
    let engine = Engine::with_store(cfg, store.clone());
    let reactive = case.get("reactive").and_then(Value::as_bool).unwrap_or(false);
    let mut results = vec![];
    for req in case["requests"].as_array().into_iter().flatten() {
        let take = req.get("take").and_then(Value::as_u64).unwrap_or(0) as usize;
        let extra = req.get("extra").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
        let body = session_body(case, d, take, extra);
        let mut ctx = match engine.prepare(body, d) {
            Ok(ctx) => ctx,
            Err(_) => {
                results.push(json!({"error": true}));
                continue;
            }
        };
        let mut r = snapshot(&ctx);
        r.insert("chain".into(), sha(&ctx.chain.join("\n")).into());
        r.insert("est_in".into(), ctx.est_tokens_in.into());
        r.insert("over_budget".into(), ctx.over_budget.into());
        r.insert("sid".into(), identity::session_id(&ctx.chain, &ctx.body, d).into());
        r.insert("side_call".into(), identity::is_side_call(ctx.msgs(), d, false).into());
        r.insert("probe".into(), identity::is_probe(&ctx.body).into());
        if reactive {
            let mut steps = vec![];
            for _ in 0..5 {
                match engine.reactive(&mut ctx) {
                    Err(_) => {
                        steps.push(json!({"ret": ERROR}));
                        break;
                    }
                    Ok(ret) => {
                        let mut s = snapshot(&ctx);
                        s.insert("ret".into(), ret.into());
                        steps.push(Value::Object(s));
                        if !ret {
                            break;
                        }
                    }
                }
            }
            r.insert("reactive".into(), steps.into());
        }
        results.push(Value::Object(r));
    }
    json!({"requests": results, "store_len": store.len(), "store_bytes": store.nbytes()})
}
