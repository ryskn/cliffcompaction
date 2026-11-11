//! The watcher's event stream and session identity (port of
//! tests/test_events.py).

mod common;

use cliff::watch::Watcher;
use cliff_core::dialects::{anthropic, detect, openai_chat, openai_responses, Dialect};
use cliff_core::hashing::chain_hashes;
use cliff_core::identity::is_side_call;
use cliff_core::{Config, Engine, PyErr, PyResult};
use common::fixtures::fat_session;
use common::*;
use futures_util::StreamExt;
use serde_json::{json, Value};

async fn app() -> Harness {
    Harness::new(Config { threshold_tokens: 2000, ..Config::from_env() }).await
}

fn turn(msgs: &mut Vec<Value>, i: impl std::fmt::Display) -> Vec<Value> {
    msgs.push(json!({"role": "assistant", "content": [{"type": "text", "text": "x".repeat(4000)}]}));
    msgs.push(json!({"role": "user", "content": format!("turn {i}")}));
    msgs.clone()
}

fn meta(session_id: &str) -> Value {
    json!({"user_id": json!({"device_id": "dddddddd", "account_uuid": "aaaaaaaa", "session_id": session_id}).to_string()})
}

fn classifier(n: usize) -> Vec<Vec<Value>> {
    (0..n)
        .map(|i| {
            vec![
                json!({"role": "user", "content": "You are reviewing a tool call."}),
                json!({"role": "user", "content": format!("<transcript>{}", "y".repeat(500 * (i + 1)))}),
            ]
        })
        .collect()
}

fn watch(h: &Harness) -> Watcher {
    let mut w = Watcher::new(0);
    for ev in h.events() {
        w.on_event(&ev);
    }
    w
}

fn sids(events: &[Value]) -> std::collections::HashSet<String> {
    events.iter().map(|e| e["sid"].as_str().unwrap().to_string()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn proxy_emits_one_event_per_handled_request() {
    let h = app().await;
    let mut msgs = vec![json!({"role": "user", "content": "hello"})];
    for i in 0..5 {
        h.post("/v1/messages", &json!({"model": "test-model", "messages": turn(&mut msgs, i)})).await;
    }
    let events = h.events();
    assert_eq!(events.len(), 5);
    assert_eq!(sids(&events).len(), 1);
    assert_eq!(events[0]["kind"], "pass");
    let compact = events.iter().filter(|e| e["kind"] == "compact").next_back().expect("a compaction");
    assert!(compact["est_out"].as_i64() < compact["est_in"].as_i64());
    assert!(!compact["fp"].as_str().unwrap().is_empty() && compact["steps"].as_i64().unwrap() >= 1);
    assert_eq!(compact["model"], "test-model");
    assert_eq!(compact["dialect"], "anthropic");
}

#[tokio::test(flavor = "multi_thread")]
async fn events_from_two_sessions_are_separate() {
    let h = app().await;
    for opening in ["first session", "second session"] {
        h.post("/v1/messages", &json!({"model": "m", "messages": [{"role": "user", "content": opening}]})).await;
    }
    assert_eq!(sids(&h.events()).len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn untouched_paths_and_probes_emit_nothing() {
    let h = app().await;
    h.get("/v1/models").await;
    h.post("/v1/messages", &json!({"model": "m"})).await;
    let probe = json!({"role": "user", "content": "quota"});
    h.post("/v1/messages", &json!({"model": "m", "max_tokens": 1, "messages": [probe]})).await;
    h.post("/v1/chat/completions", &json!({"model": "m", "max_completion_tokens": 1, "messages": [probe]})).await;
    assert!(h.events().is_empty());
    h.post("/v1/messages", &json!({"model": "m", "max_tokens": 4096, "messages": [probe]})).await;
    assert_eq!(h.events().len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn watcher_builds_rows_from_events() {
    let h = app().await;
    let mut msgs = vec![json!({"role": "user", "content": "hello"})];
    for i in 0..5 {
        h.post("/v1/messages", &json!({"model": "claude-sonnet-4-5-20250929", "messages": turn(&mut msgs, i)})).await;
    }
    let w = watch(&h);
    assert_eq!(w.requests, 5);
    assert!(w.compactions >= 1);
    let sessions = w.live_sessions();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].model, "sonnet-4-5");
    assert_eq!(sessions[0].compactions, w.compactions);
    assert!(w.render(&cliff::ui::Term::plain(100, 30)).contains("⟨cliff⟩"));
}

#[tokio::test(flavor = "multi_thread")]
async fn sse_endpoint_streams_backlog_then_live_events() {
    let h = app().await;
    h.post("/v1/messages", &json!({"model": "m", "messages": [{"role": "user", "content": "one"}]})).await;
    let resp = h.get("/__cliff__/events").await;
    assert_eq!(resp.headers()["content-type"], "text/event-stream; charset=utf-8");
    assert_eq!(resp.headers()["cache-control"], "no-store");
    let mut body = resp.bytes_stream();
    let first = String::from_utf8(body.next().await.unwrap().unwrap().to_vec()).unwrap();
    assert!(first.starts_with("data: {\"seq\": 1, "), "{first}"); // backlog, Python JSON layout
    assert_eq!(h.state.hub.subscribers(), 1);
    h.post("/v1/messages", &json!({"model": "m", "messages": [{"role": "user", "content": "two"}]})).await;
    let live = String::from_utf8(body.next().await.unwrap().unwrap().to_vec()).unwrap();
    assert!(live.contains("\"seq\": 2"));
    drop(body); // a gone watcher unsubscribes
    for _ in 0..50 {
        if h.state.hub.subscribers() == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(h.state.hub.subscribers(), 0);
}

// --- session identity -----------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn background_calls_share_the_session_they_ride_on() {
    let h = app().await;
    let m = meta("sess-A");
    let mut msgs = vec![json!({"role": "user", "content": "one"})];
    for word in ["two", "three", "four"] {
        h.post("/v1/messages", &json!({"model": "m", "messages": turn(&mut msgs, word), "metadata": m})).await;
        let mut suggestion = msgs.clone();
        suggestion.push(json!({"role": "user", "content": "[SUGGESTION MODE: ...]"}));
        h.post("/v1/messages", &json!({"model": "m", "messages": suggestion, "metadata": m})).await;
    }
    assert_eq!(sids(&h.events()).len(), 1);
    assert_eq!(watch(&h).live_sessions().len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn client_session_ids_separate_identical_openings_and_are_hashed() {
    let h = app().await;
    for label in ["A", "B"] {
        let mut msgs = vec![json!({"role": "user", "content": "hello!"})];
        for i in 0..2 {
            h.post("/v1/messages", &json!({"model": "m", "messages": turn(&mut msgs, i), "metadata": meta(&format!("sess-{label}"))})).await;
        }
    }
    assert_eq!(sids(&h.events()).len(), 2);
    assert_eq!(watch(&h).live_sessions().len(), 2);
    let blob = serde_json::to_string(&h.events()).unwrap();
    assert!(!blob.contains("sess-A") && !blob.contains("aaaaaaaa") && !blob.contains("dddddddd"));
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_client_session_id_the_root_hash_keys_the_row() {
    let h = app().await;
    let mut msgs = vec![json!({"role": "user", "content": "hello"})];
    for i in 0..3 {
        h.post("/v1/messages", &json!({"model": "m", "messages": turn(&mut msgs, i)})).await;
    }
    let mut branch = msgs[..msgs.len() - 4].to_vec();
    h.post("/v1/messages", &json!({"model": "m", "messages": turn(&mut branch, "b")})).await;
    assert_eq!(sids(&h.events()).len(), 1);
    assert_eq!(watch(&h).live_sessions().len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn side_calls_keep_the_session_id_but_stay_out_of_its_row() {
    let h = app().await;
    let m = meta("sess-A");
    let mut msgs = vec![json!({"role": "user", "content": "one"})];
    h.post("/v1/messages", &json!({"model": "m", "messages": turn(&mut msgs, 0), "metadata": m})).await;
    let first = h.events()[0].clone();
    for call in classifier(3) {
        h.post("/v1/messages", &json!({"model": "m", "messages": call, "metadata": m})).await;
    }
    let events = h.events();
    assert_eq!(events.iter().map(|e| e["aux"].as_bool().unwrap()).collect::<Vec<_>>(), [false, true, true, true]);
    assert_eq!(sids(&events).len(), 1);
    let w = watch(&h);
    let rows = w.live_sessions();
    assert_eq!(rows.len(), 1);
    assert_eq!(w.requests, 4);
    assert_eq!(rows[0].total, first["total"].as_i64().unwrap());
    assert_eq!(rows[0].est, first["est_in"].as_i64().unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn classifier_calls_make_no_row_even_without_a_client_id() {
    let h = app().await;
    for msgs in classifier(6) {
        h.post("/v1/messages", &json!({"model": "m", "messages": msgs})).await;
    }
    let events = h.events();
    assert_eq!(events.len(), 6);
    assert!(events.iter().all(|e| e["aux"] == true));
    let w = watch(&h);
    assert!(w.live_sessions().is_empty());
    assert_eq!(w.requests, 6);
    assert!(w.feed.len() == 6 && w.render(&cliff::ui::Term::plain(100, 40)).contains("aux"));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_background_call_after_the_session_speaks_is_a_side_call() {
    let h = app().await;
    let m = meta("sess-A");
    let mut msgs = vec![json!({"role": "user", "content": "one"})];
    h.post("/v1/messages", &json!({"model": "m", "messages": turn(&mut msgs, 0), "metadata": m})).await;
    h.post("/v1/messages", &json!({"model": "m", "messages": [{"role": "user", "content": "x".repeat(4000)}], "metadata": m})).await;
    let events = h.events();
    assert!(events[0]["aux"] == false && events[1]["aux"] == true);
    assert_eq!(watch(&h).live_sessions()[0].est, events[0]["est_in"].as_i64().unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn side_calls_never_pollute_lineage() {
    let h = app().await;
    let mut msgs = vec![json!({"role": "user", "content": "hello"})];
    let calls = classifier(4);
    for (i, call) in calls.into_iter().enumerate() {
        h.post("/v1/messages", &json!({"model": "m", "messages": turn(&mut msgs, i)})).await;
        h.post("/v1/messages", &json!({"model": "m", "messages": call})).await;
    }
    let events = h.events();
    let session: Vec<Value> = events.iter().filter(|e| e["aux"] == false).cloned().collect();
    assert_eq!(sids(&session).len(), 1);
    assert_eq!(watch(&h).sessions.len(), 1);
    assert_eq!(events.iter().filter(|e| e["aux"] == true).count(), 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn reactive_replay_stays_one_session() {
    let h = Harness::with(
        Config { threshold_tokens: 8000, ..Config::from_env() },
        UpInner { picky: Some(20000), ..UpInner::default() },
        None,
    )
    .await;
    assert_eq!(h.post("/v1/messages", &json!({"model": "m", "messages": fat_session(8, 6000)})).await.status(), 200);
    let events = h.events();
    assert!(events.len() >= 2); // at least one replay emitted
    assert_eq!(sids(&events).len(), 1);
    assert_eq!(watch(&h).live_sessions().len(), 1);
}

#[test]
fn side_call_rule_across_dialects() {
    let cases: [(&Dialect, Vec<Value>, Vec<Value>); 3] = [
        (
            &anthropic::DIALECT,
            vec![json!({"role": "user", "content": "a"}), json!({"role": "user", "content": "b"})],
            vec![json!({"role": "user", "content": "a"}), json!({"role": "assistant", "content": "b"})],
        ),
        (
            &openai_chat::DIALECT,
            vec![json!({"role": "user", "content": "a"}), json!({"role": "user", "content": "b"})],
            vec![json!({"role": "user", "content": "a"}), json!({"role": "assistant", "content": "b"})],
        ),
        (
            &openai_responses::DIALECT,
            vec![
                json!({"type": "message", "role": "user", "content": "a"}),
                json!({"type": "message", "role": "user", "content": "b"}),
                json!({"type": "message", "role": "user", "content": "c"}),
            ],
            vec![
                json!({"type": "message", "role": "user", "content": "a"}),
                json!({"type": "function_call", "call_id": "c", "name": "bash", "arguments": "{}"}),
            ],
        ),
    ];
    for (d, side, session) in &cases {
        assert!(is_side_call(side, d, false), "{}", d.name);
        assert!(!is_side_call(session, d, false) && !is_side_call(session, d, true), "{}", d.name);
        assert!(!is_side_call(&side[..1], d, false) && is_side_call(&side[..1], d, true), "{}", d.name);
    }
    // Codex's opening turn vs. the title-generation call beside it.
    let opening = [
        json!({"type": "message", "role": "developer", "content": "You are Codex"}),
        json!({"type": "message", "role": "user", "content": "<environment_context>…"}),
        json!({"type": "message", "role": "user", "content": "Hello!"}),
    ];
    assert!(!is_side_call(&opening, &openai_responses::DIALECT, false));
    assert!(is_side_call(&opening, &openai_responses::DIALECT, true));
    // Claude Code's first request: [user, in-array system directive].
    let cc = [json!({"role": "user", "content": "hello"}), json!({"role": "system", "content": "Available agent types: ..."})];
    assert!(!is_side_call(&cc, detect("/v1/messages").unwrap(), false));
}

#[test]
fn session_key_parses_only_the_shape_it_knows() {
    assert_eq!(anthropic::session_key(&json!({"metadata": meta("sess-1")})).unwrap().as_deref(), Some("sess-1"));
    for body in [json!({}), json!({"metadata": "x"}), json!({"metadata": {"user_id": "{bad json"}}), json!({"metadata": {"user_id": "{}"}}), json!({"metadata": {"user_id": "user-42"}})] {
        assert_eq!(anthropic::session_key(&body).unwrap(), None, "{body}");
    }
    let r = openai_responses::session_key;
    assert_eq!(r(&json!({"prompt_cache_key": "thread-1"})).unwrap().as_deref(), Some("thread-1"));
    assert_eq!(r(&json!({"client_metadata": {"thread_id": "thread-1"}})).unwrap().as_deref(), Some("thread-1"));
    for body in [json!({}), json!({"prompt_cache_key": ""}), json!({"client_metadata": "x"}), json!({"client_metadata": {}}), json!({"user": "user-42"})] {
        assert_eq!(r(&body).unwrap(), None, "{body}");
    }
}

fn boom(_: &Value) -> PyResult<Option<String>> {
    Err(PyErr("RuntimeError: boom".into()))
}

/// Session ids are display state; if reading the client's field blows up,
/// the row degrades to the root-keyed id and the request path carries on.
#[tokio::test(flavor = "multi_thread")]
async fn identity_work_can_never_break_a_request() {
    static BOOM: Dialect = Dialect { session_key: boom, ..anthropic::DIALECT };
    let cfg = Config { threshold_tokens: 2000, ..Config::from_env() };
    let engine = Engine::new(cfg.clone());
    let store = engine.store.clone();
    let h = Harness::with(cfg, UpInner::default(), Some(engine)).await;
    // Drive the boom dialect through the same emit path the proxy uses.
    let mut msgs = vec![json!({"role": "user", "content": "hello"})];
    for i in 0..5 {
        let body = json!({"model": "m", "messages": turn(&mut msgs, i), "metadata": meta("sess-A")});
        let ctx = h.state.engine.prepare(body, &BOOM).unwrap();
        let sid = cliff_core::identity::session_id(&ctx.chain, &ctx.body, &BOOM);
        let root = chain_hashes(&[anthropic::digest_message(&msgs[0]).unwrap()])[0][..12].to_string();
        assert_eq!(sid, root);
    }
    assert!(!store.is_empty()); // still compacting
}
