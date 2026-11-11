//! End-to-end proxy tests against a mock upstream (ports of
//! tests/test_proxy.py, the proxy half of test_escalation.py and
//! test_responses_dialect.py).

mod common;

use cliff::proxy::{is_context_error, on_cpu, pick_upstream};
use cliff_core::{Config, SUMMARY_HEADER};
use common::fixtures::*;
use common::*;
use serde_json::{json, Value};

fn cfg(threshold_tokens: i64, keep_recent: i64) -> Config {
    Config { threshold_tokens, keep_recent, ..Config::default() }
}

fn has_summary(msgs: &[Value]) -> bool {
    msgs.iter().any(|m| m["content"].as_str().is_some_and(|c| c.starts_with(SUMMARY_HEADER)))
}

#[tokio::test(flavor = "multi_thread")]
async fn small_request_passthrough_bytes() {
    let h = Harness::new(cfg(1_000_000, 3)).await;
    let raw = serde_json::to_vec(&a_body(a_session(3, 3000))).unwrap();
    let r = h.client.post(format!("{}/v1/messages", h.base)).header("x-api-key", "k").body(raw.clone()).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let last = h.up.last();
    assert_eq!(last.body.as_ref(), raw.as_slice()); // verbatim: the exact bytes we sent
    assert_eq!(last.headers["x-api-key"], "k");
    assert_eq!(last.headers["accept-encoding"], "identity");
}

#[tokio::test(flavor = "multi_thread")]
async fn over_threshold_compacted() {
    let h = Harness::new(cfg(2_000, 1)).await;
    let body = a_body(a_session(10, 3000));
    assert_eq!(h.post("/v1/messages", &body).await.status(), 200);
    let sent = h.up.last().messages();
    assert!(sent.len() < body["messages"].as_array().unwrap().len());
    assert!(sent[1]["content"].as_str().unwrap().starts_with(SUMMARY_HEADER));
}

#[tokio::test(flavor = "multi_thread")]
async fn followup_substituted() {
    let h = Harness::new(cfg(2_000, 1)).await;
    let mut msgs = a_session(10, 3000);
    h.post("/v1/messages", &a_body(msgs.clone())).await;
    msgs.push(json!({"role": "assistant", "content": [{"type": "text", "text": "next step"}]}));
    h.post("/v1/messages", &a_body(msgs)).await;
    let sent = h.up.last().messages();
    assert!(sent[1]["content"].as_str().unwrap().starts_with(SUMMARY_HEADER));
    assert_eq!(sent.last().unwrap()["content"][0]["text"], "next step");
}

#[tokio::test(flavor = "multi_thread")]
async fn shadow_mode_never_modifies() {
    let h = Harness::new(Config { shadow: true, ..cfg(2_000, 1) }).await;
    let raw = serde_json::to_vec(&a_body(a_session(10, 3000))).unwrap();
    assert_eq!(h.post_raw("/v1/messages", raw.clone()).await.status(), 200);
    assert_eq!(h.up.last().body.as_ref(), raw.as_slice());
}

#[tokio::test(flavor = "multi_thread")]
async fn reactive_retry_on_context_400() {
    let h = Harness::new(cfg(1_000_000, 1)).await;
    h.up.push(400, json!({"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long: 210000 tokens > 200000 maximum"}}));
    let r = h.post("/v1/messages", &a_body(a_session(10, 3000))).await;
    assert_eq!(r.status(), 200); // replayed after compaction
    assert_eq!(h.up.count(), 2);
    assert!(h.up.last().messages()[1]["content"].as_str().unwrap().starts_with(SUMMARY_HEADER));
}

#[tokio::test(flavor = "multi_thread")]
async fn non_context_400_passed_through() {
    let h = Harness::new(cfg(1_000_000, 3)).await;
    h.up.push(400, json!({"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens must be positive"}}));
    let r = h.post("/v1/messages", &a_body(a_session(4, 3000))).await;
    assert_eq!(r.status(), 400);
    assert_eq!(h.up.count(), 1);
    assert!(r.text().await.unwrap().contains("max_tokens"));
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_json_fail_open() {
    let h = Harness::new(cfg(1, 3)).await;
    let r = h.post_raw("/v1/messages", b"{not json".to_vec()).await;
    assert_eq!(r.status(), 200);
    assert_eq!(h.up.last().body.as_ref(), b"{not json");
}

#[tokio::test(flavor = "multi_thread")]
async fn count_tokens_passthrough() {
    let h = Harness::new(cfg(1, 1)).await;
    let raw = serde_json::to_vec(&a_body(a_session(10, 3000))).unwrap();
    assert_eq!(h.post_raw("/v1/messages/count_tokens", raw.clone()).await.status(), 200);
    assert_eq!(h.up.last().body.as_ref(), raw.as_slice());
}

#[tokio::test(flavor = "multi_thread")]
async fn upstream_routing() {
    let (anthropic, a_up) = start_upstream(UpInner::default()).await;
    let (openai, o_up) = start_upstream(UpInner::default()).await;
    let h = Harness::with(
        Config { anthropic_upstream: anthropic, openai_upstream: openai, ..cfg(1_000_000, 3) },
        UpInner::default(),
        None,
    )
    .await;
    h.post("/v1/messages", &a_body(a_session(2, 3000))).await;
    assert_eq!(a_up.count(), 1);
    h.post("/v1/chat/completions", &json!({"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(o_up.count(), 1);
    h.get("/v1/models?x=1").await;
    assert_eq!(a_up.count(), 2);
    assert_eq!(a_up.last().query.as_deref(), Some("x=1"));
}

#[test]
fn leftover_paths_follow_single_configured_upstream() {
    let only_openai = Config { openai_upstream: "https://openai.example".into(), ..Config::default() };
    assert_eq!(pick_upstream(&only_openai, "/v1/models"), "https://openai.example");
    assert_eq!(pick_upstream(&only_openai, "/api/hello"), "https://openai.example");
    assert_eq!(pick_upstream(&only_openai, "/v1/messages"), "https://api.anthropic.com");
    let only_anthropic = Config { anthropic_upstream: "https://kimi.example".into(), ..Config::default() };
    assert_eq!(pick_upstream(&only_anthropic, "/v1/models"), "https://kimi.example");
    assert_eq!(pick_upstream(&only_anthropic, "/v1/messages/count_tokens"), "https://kimi.example");
    assert_eq!(pick_upstream(&Config::default(), "/v1/responses/compact"), "https://api.openai.com");
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_bytes_relayed() {
    let h = Harness::new(cfg(1_000_000, 3)).await;
    let sse = b"event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: done\ndata: {}\n\n".to_vec();
    h.up.0.lock().unwrap().script.push_back(Scripted { status: 200, headers: vec![("content-type", "text/event-stream")], body: sse.clone() });
    let r = h.post("/v1/messages", &a_body(a_session(2, 3000))).await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "text/event-stream");
    assert_eq!(r.bytes().await.unwrap().as_ref(), sse.as_slice());
}

#[tokio::test(flavor = "multi_thread")]
async fn status_endpoint() {
    let h = Harness::new(Config::default()).await;
    let data: Value = serde_json::from_slice(&h.get("/__cliff__/status").await.bytes().await.unwrap()).unwrap();
    assert_eq!(data["name"], "cliffcompaction");
    assert_eq!(data["implementation"], "rust");
    assert_eq!(data["shadow"], false);
    assert_eq!(data["watchers"], 0);
    assert!(data.get("uptime_s").is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn debug_dir_dumps_requests() {
    let dir = tempfile::tempdir().unwrap();
    let h = Harness::new(Config { debug_dir: dir.path().display().to_string(), ..cfg(2_000, 1) }).await;
    assert_eq!(h.post("/v1/messages", &a_body(a_session(10, 3000))).await.status(), 200);
    let files: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
    assert_eq!(files.len(), 1);
    let rec: Value = serde_json::from_str(&std::fs::read_to_string(files[0].as_ref().unwrap().path()).unwrap()).unwrap();
    assert_eq!(rec["dialect"], "anthropic");
    assert_eq!(rec["modified"], true);
    assert_eq!(rec["incoming_messages"].as_array().unwrap().len(), a_session(10, 3000).len());
    assert!(rec["outgoing_messages"][1]["content"].as_str().unwrap().starts_with(SUMMARY_HEADER));
    assert_eq!(h.up.last().messages().len(), rec["outgoing_messages"].as_array().unwrap().len());
}

// --- reactive ladder & strict mode (test_escalation.py) ---------------------------------

async fn picky(cfg: Config, max_chars: usize) -> Harness {
    Harness::with(cfg, UpInner { picky: Some(max_chars), ..UpInner::default() }, None).await
}

#[tokio::test(flavor = "multi_thread")]
async fn reactive_ladder_succeeds_at_lower_rung() {
    let h = picky(cfg(8000, 3), 20000).await;
    assert_eq!(h.post("/v1/messages", &a_body(fat_session(8, 6000))).await.status(), 200);
    assert!(h.up.count() >= 2); // original + at least one replay
    assert!(has_summary(&h.up.last().messages()));
}

#[tokio::test(flavor = "multi_thread")]
async fn reactive_rung3_truncates_summary() {
    let h = picky(cfg(1000, 1), 6000).await;
    assert_eq!(h.post("/v1/messages", &a_body(fat_session(10, 300))).await.status(), 200);
    let sent = h.up.last().messages();
    let text = sent.iter().find_map(|m| m["content"].as_str().filter(|c| c.starts_with(SUMMARY_HEADER))).unwrap();
    if text.contains("make") {
        assert!(!text.contains("make 0") || text.contains("make 8"));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn reactive_gives_up_and_returns_400() {
    let h = picky(cfg(2000, 1), 10).await;
    assert_eq!(h.post("/v1/messages", &a_body(fat_session(6, 6000))).await.status(), 400);
    assert!(h.up.count() <= 5); // original + at most one per remaining rung
}

#[tokio::test(flavor = "multi_thread")]
async fn strict_refuses_over_budget_request_without_looking_like_a_context_error() {
    let h = Harness::with(Config { strict: true, ..cfg(1000, 3) }, UpInner { unreachable: true, ..UpInner::default() }, None).await;
    let r = h.post("/v1/messages", &a_body(fat_session(1, 40000))).await;
    assert_eq!(r.status(), 400);
    let body = r.bytes().await.unwrap();
    assert!(!is_context_error(&body)); // a chained proxy must not read it as overflow
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["error"]["type"], "cliff_over_budget");
}

#[tokio::test(flavor = "multi_thread")]
async fn soft_send_remains_the_default() {
    let h = picky(cfg(1000, 3), usize::MAX).await;
    assert_eq!(h.post("/v1/messages", &a_body(fat_session(1, 40000))).await.status(), 200);
    assert_eq!(h.up.count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn strict_forwards_what_fits() {
    let h = picky(Config { strict: true, ..cfg(4000, 3) }, usize::MAX).await;
    assert_eq!(h.post("/v1/messages", &a_body(fat_session(8, 6000))).await.status(), 200);
    assert_eq!(h.up.count(), 1);
    assert!(h.up.last().messages().len() < 17);
}

#[tokio::test(flavor = "multi_thread")]
async fn strict_is_inert_in_shadow_mode() {
    let h = picky(Config { strict: true, shadow: true, ..cfg(1000, 3) }, usize::MAX).await;
    assert_eq!(h.post("/v1/messages", &a_body(fat_session(1, 40000))).await.status(), 200);
    assert_eq!(h.up.count(), 1);
}

// --- OpenAI Responses -----------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn proxy_responses_route_and_compaction() {
    let (anthropic, a_up) = start_upstream(UpInner::default()).await;
    let h = Harness::with(Config { anthropic_upstream: anthropic, ..cfg(2_000, 1) }, UpInner::default(), None).await;
    assert_eq!(h.post("/v1/responses", &r_body(codex_session(8))).await.status(), 200);
    assert_eq!(a_up.count(), 0);
    let sent = h.up.last().json();
    assert!(sent["input"][3]["content"][0]["text"].as_str().unwrap().starts_with(SUMMARY_HEADER));
    assert_eq!(sent["store"], false); // rest of the body untouched
}

#[tokio::test(flavor = "multi_thread")]
async fn proxy_responses_compact_endpoint_passthrough() {
    let h = Harness::new(cfg(1, 3)).await;
    let raw = serde_json::to_vec(&r_body(codex_session(4))).unwrap();
    assert_eq!(h.post_raw("/v1/responses/compact", raw.clone()).await.status(), 200);
    assert_eq!(h.up.last().body.as_ref(), raw.as_slice()); // no dialect claims it
}

// --- concurrency & fail-open (Rust-only) --------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn a_panicking_cpu_job_fails_open_instead_of_aborting() {
    let r: Result<(), String> = on_cpu(|| panic!("boom")).await;
    assert_eq!(r.unwrap_err(), "boom");
    assert_eq!(on_cpu(|| 41 + 1).await.unwrap(), 42); // the pool survived
}

/// Many sessions at once: every request is answered, and each session's
/// upstream body is byte-identical to what a lone request produces.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_sessions_get_their_own_compactions() {
    let h = std::sync::Arc::new(Harness::new(cfg(2_000, 1)).await);
    let mut tasks = vec![];
    for s in 0..24 {
        let h = h.clone();
        tasks.push(tokio::spawn(async move {
            let mut msgs = a_session(10, 3000);
            msgs[0] = a_user(&format!("session {s}: fix the failing test"));
            let r = h.post("/v1/messages", &a_body(msgs)).await;
            r.status()
        }));
    }
    for t in tasks {
        assert_eq!(t.await.unwrap(), 200);
    }
    let reqs = h.up.requests();
    assert_eq!(reqs.len(), 24);
    assert!(reqs.iter().all(|r| has_summary(&r.messages())));
    assert_eq!(h.state.engine.store.len(), 24);
}
