//! Engine pipeline and escalation ladder (ports of tests/test_engine.py and
//! the engine half of tests/test_escalation.py).

mod common;

use std::sync::Arc;

use cliff_core::dialects::anthropic::DIALECT;
use cliff_core::{Config, Engine, RequestCtx};
use common::*;
use serde_json::{json, Value};

fn engine(threshold_tokens: i64, keep_recent: i64) -> Engine {
    Engine::new(Config { threshold_tokens, keep_recent, ..Config::default() })
}

fn grow(msgs: &[Value], start: usize, n: usize, result_chars: usize) -> Vec<Value> {
    let mut out = msgs.to_vec();
    for i in start..start + n {
        let tid = format!("tu_{i}");
        out.push(a_assistant(&format!("Step {i}"), Some((&tid, "bash", json!({"command": format!("cmd {i}")})))));
        out.push(a_result(&tid, &"R".repeat(result_chars)));
    }
    out
}

fn n_summaries(ctx: &RequestCtx) -> usize {
    ctx.substituted_values().iter().filter(|m| is_summary(m)).count()
}

fn summary_text(ctx: &RequestCtx) -> String {
    ctx.substituted_values().iter().find(|m| is_summary(m)).map(|m| content_str(m).to_string()).unwrap_or_default()
}

fn prepare(e: &Engine, msgs: Vec<Value>) -> RequestCtx {
    e.prepare(a_body(msgs), &DIALECT).unwrap()
}

#[test]
fn under_threshold_passthrough() {
    let ctx = prepare(&engine(1_000_000, 3), a_session(4, 3000));
    assert!(!ctx.modified);
    assert_eq!(ctx.outgoing_json(), cliff_core::py::dumps(&a_body(a_session(4, 3000)), cliff_core::py::DUMPS_UNICODE).into_bytes());
}

#[test]
fn compact_store_and_substitute() {
    let e = engine(2_000, 1);
    let msgs = a_session(10, 3000);
    let ctx1 = prepare(&e, msgs.clone());
    assert!(ctx1.compacted && ctx1.modified);
    assert_eq!(n_summaries(&ctx1), 1);
    assert_eq!(e.store.len(), 1);
    assert!(ctx1.est_tokens_out < ctx1.est_tokens_in);

    // Scaffold keeps growing the ORIGINAL history (it never saw C).
    let msgs2 = grow(&msgs, 10, 1, 10);
    let ctx2 = prepare(&e, msgs2.clone());
    assert!(ctx2.modified);
    let out = ctx2.substituted_values();
    assert_eq!(*out[0], msgs2[0]);
    assert!(is_summary(out[1]));
    assert_eq!(out[2..].iter().map(|m| (*m).clone()).collect::<Vec<_>>(), msgs2[ctx2.base_cut..].to_vec());
}

#[test]
fn recompaction_mapping_and_flatness() {
    let e = engine(2_000, 1);
    let msgs = a_session(10, 3000);
    prepare(&e, msgs.clone());
    let msgs2 = grow(&msgs, 10, 8, 2000);
    let ctx = prepare(&e, msgs2.clone());
    assert!(ctx.compacted);
    assert_eq!(n_summaries(&ctx), 1);
    assert_eq!(*ctx.substituted_values()[0], msgs2[0]);
    assert_eq!(*ctx.substituted_values().last().unwrap(), msgs2.last().unwrap());
    assert_eq!(e.store.len(), 2); // entries at both cut depths

    let msgs3 = grow(&msgs2, 18, 1, 10);
    let ctx3 = prepare(&e, msgs3);
    assert!(ctx3.base_cut > msgs.len()); // matched the second, deeper entry
    assert_eq!(n_summaries(&ctx3), 1);
}

#[test]
fn diverging_branches_share_prefix() {
    let e = engine(2_000, 1);
    let msgs = a_session(10, 3000);
    prepare(&e, msgs.clone());
    let branch_a = grow(&msgs, 10, 1, 10);
    let mut branch_b = msgs.clone();
    branch_b.push(a_assistant("different continuation", None));
    let ctx_a = prepare(&e, branch_a.clone());
    let ctx_b = prepare(&e, branch_b.clone());
    assert!(ctx_a.modified && ctx_b.modified);
    assert_eq!(*ctx_a.substituted_values().last().unwrap(), branch_a.last().unwrap());
    assert_eq!(*ctx_b.substituted_values().last().unwrap(), branch_b.last().unwrap());
}

#[test]
fn history_mutation_breaks_match_fail_open() {
    let e = engine(2_000, 1);
    let mut msgs = a_session(10, 3000);
    prepare(&e, msgs.clone());
    msgs[3] = json!({"role": "user", "content": "history rewritten by scaffold"});
    let small = Engine::with_store(Config { threshold_tokens: 1_000_000, ..Config::default() }, e.store.clone());
    assert!(!prepare(&small, msgs).modified); // no match anywhere -> verbatim
}

#[test]
fn reactive_compacts_regardless_of_threshold() {
    let e = engine(1_000_000, 1);
    let mut ctx = prepare(&e, a_session(10, 3000));
    assert!(!ctx.modified);
    assert!(e.reactive(&mut ctx).unwrap());
    assert!(ctx.compacted);
    assert_eq!(n_summaries(&ctx), 1);
}

#[test]
fn store_loss_replays_chain_not_accumulate() {
    // A whole day's history arriving at once (store lost overnight) must
    // yield the same bounded, recent-only summary the live chain would have.
    let cfg = Config { threshold_tokens: 2_000, keep_recent: 1, ..Config::default() };
    let live = Engine::new(cfg.clone());
    let mut msgs = a_session(2, 3000);
    let mut live_summary = String::new();
    for i in 2..60 {
        live_summary = summary_text(&prepare(&live, msgs.clone()));
        msgs = grow(&msgs, i, 1, 2000);
    }
    let fresh = Engine::new(cfg.clone());
    let ctx = prepare(&fresh, msgs);
    assert!(ctx.compacted);
    let fresh_summary = summary_text(&ctx);
    assert_eq!(n_summaries(&ctx), 1);
    assert!(!fresh_summary.contains("Step 2") && !fresh_summary.contains("Step 10"));
    assert!(fresh_summary.len() < 3 * live_summary.len().max(1_000));
    assert!(ctx.est_tokens_out <= cfg.threshold_tokens * 2);
}

#[test]
fn reactive_ladder_terminates() {
    let e = engine(2_000, 1);
    let mut ctx = prepare(&e, a_session(10, 3000));
    assert!(ctx.compacted);
    let mut attempts = 0;
    while e.reactive(&mut ctx).unwrap() && attempts < 10 {
        attempts += 1;
    }
    assert!(attempts <= 4);
    assert!(!e.reactive(&mut ctx).unwrap());
    assert!(!e.reactive(&mut ctx).unwrap()); // stays exhausted
}

// --- escalation ------------------------------------------------------------------

#[test]
fn proactive_escalates_when_floor_over_threshold() {
    let ctx = prepare(&engine(4000, 3), fat_session(8, 6000));
    assert!(ctx.compacted && ctx.rung >= 1);
    assert!(ctx.est_tokens_out <= 4000);
    assert_eq!(ctx.substituted_values().iter().filter(|m| m["role"] == "assistant").count(), 1);
}

#[test]
fn proactive_case1_few_giant_turns() {
    let ctx = prepare(&engine(3000, 3), fat_session(3, 6000));
    assert!(ctx.compacted && ctx.rung >= 1);
    assert!(ctx.substituted.len() < ctx.msgs().len());
}

#[test]
fn proactive_soft_send_on_giant_live_turn() {
    let ctx = prepare(&engine(1000, 3), fat_session(1, 40000));
    assert!(!ctx.outgoing_json().is_empty()); // no exception; soft path
}

#[test]
fn no_escalation_without_assistant_turns() {
    let ctx = prepare(&engine(100, 3), vec![json!({"role": "user", "content": "x".repeat(30000)})]);
    assert!(!ctx.compacted && !ctx.modified && ctx.rung == 0);
}

#[test]
fn strict_walks_rung3_before_refusing() {
    let e = Engine::new(Config { threshold_tokens: 1000, keep_recent: 1, strict: true, ..Config::default() });
    let ctx = prepare(&e, fat_session(10, 300));
    assert_eq!(ctx.rung, 3);
    assert!(!ctx.over_budget);
    assert!(ctx.est_tokens_out <= 1000);
}

#[test]
fn default_ladder_stops_at_rung2() {
    assert!(prepare(&engine(1000, 1), fat_session(10, 300)).rung <= 2);
}

#[test]
fn over_budget_flag_clear_when_under_threshold() {
    let e = Engine::new(Config { threshold_tokens: 4000, keep_recent: 3, strict: true, ..Config::default() });
    assert!(!prepare(&e, fat_session(8, 6000)).over_budget);
}

// --- concurrency (Rust-only) ---------------------------------------------------------

/// The store is shared across threads; concurrent identical requests race
/// to fill the same key with the same (deterministic) entry, and every
/// thread must see the same outgoing bytes a sequential run produces.
#[test]
fn concurrent_prepares_are_deterministic() {
    let e = Arc::new(engine(2_000, 1));
    let msgs = a_session(30, 3000);
    let expected = Engine::new(e.cfg.clone()).prepare(a_body(msgs.clone()), &DIALECT).unwrap().outgoing_json();
    let handles: Vec<_> = (0..16)
        .map(|_| {
            let (e, msgs) = (e.clone(), msgs.clone());
            std::thread::spawn(move || e.prepare(a_body(msgs), &DIALECT).unwrap().outgoing_json())
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), expected);
    }
    assert_eq!(e.store.len(), 1);
}
