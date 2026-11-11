//! Core compaction semantics, both dialects (port of tests/test_cliff.py).

mod common;

use cliff_core::cliff::{compact, group_turns, Compaction};
use cliff_core::dialects::{anthropic::DIALECT as ANTHROPIC, openai_chat::DIALECT as OPENAI};
use cliff_core::{Config, SUMMARY_HEADER};
use common::*;
use serde_json::{json, Value};

fn cfg(keep_recent: i64) -> Config {
    Config { keep_recent, ..Config::default() }
}

fn refs(msgs: &[Value]) -> Vec<&Value> {
    msgs.iter().collect()
}

/// The compacted sequence: head + summary + kept tail.
fn out<'a>(msgs: &'a [Value], c: &'a Compaction) -> Vec<&'a Value> {
    msgs[..c.head_len].iter().chain([&c.summary]).chain(&msgs[c.cut..]).collect()
}

fn run(msgs: &[Value], cfg: &Config) -> Compaction {
    compact(&refs(msgs), &ANTHROPIC, cfg).unwrap().expect("compacted")
}

fn summary_of(msgs: &[Value], cfg: &Config) -> String {
    content_str(&run(msgs, cfg).summary).to_string()
}

// --- structure ----------------------------------------------------------------

#[test]
fn basic_structure_anthropic() {
    let msgs = a_session(10, 3000);
    let c = run(&msgs, &cfg(2));
    let o = out(&msgs, &c);
    assert!(std::ptr::eq(o[0], &msgs[0])); // head verbatim, same object
    assert_eq!(o[1]["role"], "user");
    assert!(content_str(o[1]).starts_with(SUMMARY_HEADER));
    assert_eq!(o.len(), 2 + 4); // last 2 turns (4 messages) verbatim
    assert!(std::ptr::eq(o[2], &msgs[msgs.len() - 4]));
    assert_eq!(c.cut, msgs.len() - 4);
    assert_eq!(o[2]["role"], "assistant"); // turn boundary, no orphan tool_result
}

#[test]
fn basic_structure_openai() {
    let msgs = o_session(10, 3000);
    let c = compact(&refs(&msgs), &OPENAI, &cfg(2)).unwrap().unwrap();
    let o = out(&msgs, &c);
    assert!(std::ptr::eq(o[0], &msgs[0]) && std::ptr::eq(o[1], &msgs[1]));
    assert!(content_str(o[2]).starts_with(SUMMARY_HEADER));
    assert_eq!(o[3..].len(), 4);
    assert_eq!(o[3]["role"], "assistant");
}

#[test]
fn nothing_to_compact() {
    assert!(compact(&refs(&a_session(2, 3000)), &ANTHROPIC, &cfg(3)).unwrap().is_none());
}

#[test]
fn no_assistant_yet() {
    assert!(compact(&refs(&[a_user("task")]), &ANTHROPIC, &cfg(1)).unwrap().is_none());
}

// --- content classes -----------------------------------------------------------

#[test]
fn long_results_dropped_short_kept() {
    let s = summary_of(&a_session(10, 3000), &cfg(1));
    assert!(!s.contains(&"X".repeat(600)));
    assert!(s.contains("result: test_1 passed (short output)"));
}

#[test]
fn tool_signatures_present_and_truncated() {
    let s = summary_of(&a_session(6, 3000), &Config { keep_recent: 1, cmd_max_chars: 20, ..Config::default() });
    assert!(s.contains("[bash]"));
    for line in s.lines().filter(|l| l.starts_with("[bash] ")) {
        assert!(line.chars().count() <= "[bash] ".len() + 23);
    }
}

fn thought_session(thought: &str) -> Vec<Value> {
    vec![
        a_user("task"),
        a_assistant(thought, Some(("t0", "bash", json!({"command": "ls"})))),
        a_result("t0", "ok"),
        a_assistant("done", None),
        a_result("t0", "bye"),
    ]
}

#[test]
fn assistant_text_full_by_default() {
    let long = "T".repeat(2000);
    assert!(summary_of(&thought_session(&long), &cfg(1)).contains(&long));
}

#[test]
fn assistant_text_capped_when_configured() {
    let long = "T".repeat(2000);
    let s = summary_of(&thought_session(&long), &Config { keep_recent: 1, thought_max_chars: 300, ..Config::default() });
    assert!(!s.contains(&long));
    assert!(s.contains(&format!("{}...", "T".repeat(300))));
}

#[test]
fn human_text_kept_verbatim() {
    let mut msgs = a_session(8, 3000);
    let human = "IMPORTANT: use the staging database, never prod. ".repeat(20);
    msgs.insert(5, a_user(&human));
    assert!(summary_of(&msgs, &cfg(1)).contains(human.trim()));
}

#[test]
fn mixed_message_human_text_and_tool_result() {
    let long_out = "Z".repeat(5000);
    let msgs = vec![
        a_user("task"),
        a_assistant("step 0", Some(("t0", "bash", json!({"command": "ls"})))),
        json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t0", "content": long_out},
            {"type": "text", "text": "actually, stop and use branch dev-2"},
        ]}),
        a_assistant("step 1", Some(("t1", "bash", json!({"command": "pwd"})))),
        a_result("t1", "ok"),
        a_assistant("step 2", Some(("t2", "bash", json!({"command": "id"})))),
        a_result("t2", "ok"),
    ];
    let s = summary_of(&msgs, &cfg(1));
    assert!(s.contains("actually, stop and use branch dev-2"));
    assert!(!s.contains(&long_out));
}

fn thinking_session() -> Vec<Value> {
    vec![
        a_user("task"),
        json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "The mock is not reset between cases, that is the real bug.", "signature": "EqQBCgIYAsignedblob"},
            {"type": "text", "text": "step 0 visible text"},
            {"type": "tool_use", "id": "t0", "name": "bash", "input": {"command": "ls"}},
        ]}),
        a_result("t0", "ok"),
        a_assistant("step 1", Some(("t1", "bash", json!({"command": "pwd"})))),
        a_result("t1", "ok"),
        a_assistant("step 2", None),
        a_result("t2", "ok"),
    ]
}

#[test]
fn thinking_kept_as_text_in_summary() {
    let s = summary_of(&thinking_session(), &cfg(1));
    assert!(s.contains("thinking: The mock is not reset between cases"));
    assert!(s.contains("step 0 visible text"));
    assert!(!s.contains("signedblob"));
}

#[test]
fn thinking_cap_independent_of_thought_cap() {
    let s = summary_of(&thinking_session(), &Config { keep_recent: 1, thought_max_chars: 10, ..Config::default() });
    assert!(s.contains("thinking: The mock is not reset between cases, that is the real bug."));
    assert!(s.contains("step 0 vis..."));
    let s2 = summary_of(&thinking_session(), &Config { keep_recent: 1, thinking_max_chars: 12, ..Config::default() });
    assert!(s2.contains("thinking: The mock is ..."));
    assert!(s2.contains("step 0 visible text"));
}

#[test]
fn thinking_droppable() {
    let s = summary_of(&thinking_session(), &Config { keep_recent: 1, keep_thinking: false, ..Config::default() });
    assert!(!s.contains("thinking:"));
    assert!(s.contains("step 0 visible text"));
}

#[test]
fn openai_reasoning_content_kept_as_text() {
    let mut msgs = o_session(6, 3000);
    msgs[2]["reasoning_content"] = json!("I should check the failing test first.");
    let c = compact(&refs(&msgs), &OPENAI, &cfg(1)).unwrap().unwrap();
    assert!(content_str(&c.summary).contains("thinking: I should check the failing test first."));
}

#[test]
fn speaker_tags_in_summary() {
    let mut msgs = a_session(8, 3000);
    msgs.insert(5, a_user("please target the dev branch"));
    let s = summary_of(&msgs, &cfg(1));
    assert!(s.contains("user: please target the dev branch"));
    assert!(s.contains("assistant: Step 0:"));
    assert!(s.contains("result: test_1 passed"));
}

#[test]
fn task_notifications_stripped_from_summary() {
    let notif = format!("<task-notification>\n<task-id>abc</task-id>\n<result>{}</result>\n</task-notification>", "R".repeat(3000));
    let mut msgs = vec![
        a_user("task"),
        a_assistant("step 0", Some(("t0", "bash", json!({"command": "ls"})))),
        a_user(&format!("{notif}\nalso: please use the dev branch")),
        a_assistant("step 1", Some(("t1", "bash", json!({"command": "pwd"})))),
        a_result("t1", "ok"),
        a_assistant("step 2", None),
        a_result("t2", "ok"),
    ];
    let s = summary_of(&msgs, &cfg(1));
    assert!(!s.contains("task-notification") && !s.contains("RRRR"));
    assert!(s.contains("also: please use the dev branch"));
    msgs[2] = a_user(&notif); // a notification-only user message contributes nothing
    assert!(!summary_of(&msgs, &cfg(1)).contains("task-notification"));
}

#[test]
fn images_dropped_from_summary() {
    let msgs = vec![
        a_user("task"),
        a_assistant("step 0", Some(("t0", "bash", json!({"command": "ls"})))),
        json!({"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            {"type": "text", "text": "see the screenshot"},
        ]}),
        a_assistant("step 1", Some(("t1", "bash", json!({"command": "pwd"})))),
        a_result("t1", "ok"),
        a_assistant("step 2", None),
        a_result("t2", "ok"),
    ];
    let s = summary_of(&msgs, &cfg(1));
    assert!(!s.contains("AAAA"));
    assert!(s.contains("see the screenshot"));
}

// --- re-compaction semantics -----------------------------------------------------

#[test]
fn recompaction_stays_flat() {
    let msgs = a_session(10, 3000);
    let c1 = run(&msgs, &cfg(1));
    // scaffold-side: history keeps growing; proxy-side: C + new tail
    let mut grown: Vec<Value> = out(&msgs, &c1).into_iter().cloned().collect();
    for i in 10..16 {
        let tid = format!("tu_{i}");
        grown.push(a_assistant(&format!("Step {i}"), Some((&tid, "bash", json!({"command": format!("cmd {i}")})))));
        grown.push(a_result(&tid, &"K".repeat(2000)));
    }
    let c2 = run(&grown, &cfg(1));
    let o = out(&grown, &c2);
    let summaries: Vec<&&Value> = o.iter().filter(|m| is_summary(m)).collect();
    assert_eq!(summaries.len(), 1); // exactly one summary — never nested
    assert!(!content_str(summaries[0]).contains("Step 0")); // not merged forward
    assert!(content_str(summaries[0]).contains("Step 14"));
    assert_eq!(*o[0], msgs[0]); // head survived both compactions
}

#[test]
fn turn_grouping_orphan_summary() {
    let msgs = a_session(4, 3000);
    let c = run(&msgs, &cfg(1));
    let o = out(&msgs, &c);
    let body = &o[1..];
    let turns = group_turns(body, &ANTHROPIC).unwrap();
    assert!(is_summary(body[turns[0].start]));
    assert!(turns[1..].iter().all(|t| body[t.start]["role"] == "assistant"));
}

#[test]
fn system_messages_never_precede_the_summary() {
    let mut msgs = vec![json!({"role": "user", "content": "the task"}), json!({"role": "system", "content": "directive: be careful"})];
    for i in 0..6 {
        msgs.push(json!({"role": "assistant", "content": [
            {"type": "text", "text": format!("step {i}")},
            {"type": "tool_use", "id": format!("t{i}"), "name": "bash", "input": {"command": format!("make {i}")}},
        ]}));
        msgs.push(json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": format!("t{i}"), "content": "R".repeat(2000)}]}));
    }
    let c = run(&msgs, &cfg(2));
    assert_eq!(c.head_len, 1); // system message trimmed out of the head...
    let o = out(&msgs, &c);
    assert_eq!(o[0]["role"], "user");
    assert!(content_str(o[1]).contains("system: directive: be careful")); // ...folded in
    for (i, m) in o.iter().enumerate() {
        if m["role"] == "system" && m.get("content").is_some_and(|c| !c.is_null()) {
            assert!(i == o.len() - 1 || o[i + 1]["role"] == "assistant");
        }
    }
}

#[test]
fn system_message_in_tail_and_directive_only_dropped() {
    let mut msgs = vec![json!({"role": "user", "content": "task"})];
    for i in 0..5 {
        msgs.push(json!({"role": "assistant", "content": format!("step {i}")}));
        msgs.push(json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": format!("t{i}"), "content": "R".repeat(1500)}]}));
        if i == 1 {
            msgs.push(json!({"role": "system", "content": []}));
        }
    }
    let c = run(&msgs, &cfg(2));
    assert!(!content_str(&c.summary).contains("system:"));
    assert_eq!(*out(&msgs, &c).last().unwrap(), msgs.last().unwrap());
}
