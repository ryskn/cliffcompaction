//! Ports of tests/test_hashing.py, test_store.py, test_images.py and the
//! non-proxy half of test_responses_dialect.py.

mod common;

use std::sync::Arc;

use base64::Engine as _;
use cliff_core::cliff::{compact, group_turns};
use cliff_core::dialects::{anthropic, openai_chat, openai_responses};
use cliff_core::engine::{billable_chars, estimate_tokens, msg_chars};
use cliff_core::hashing::chain_hashes;
use cliff_core::images::{dimensions, image_payloads, tokens_for_payload, DEFAULT_IMAGE_TOKENS, MAX_IMAGE_TOKENS};
use cliff_core::store::{entry_size, Entry, PrefixStore};
use cliff_core::{Config, Engine, SUMMARY_HEADER};
use common::*;
use serde_json::{json, Value};

// --- hashing ------------------------------------------------------------------------

fn a_digest(m: &Value) -> String {
    anthropic::digest_message(m).unwrap()
}

#[test]
fn cache_control_ignored() {
    let plain = json!({"role": "user", "content": [{"type": "text", "text": "hello"}]});
    let marked = json!({"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]});
    assert_eq!(a_digest(&plain), a_digest(&marked));
}

#[test]
fn string_vs_block_content_equal() {
    let s = json!({"role": "user", "content": "hello"});
    let b = json!({"role": "user", "content": [{"type": "text", "text": "hello"}]});
    assert_eq!(a_digest(&s), a_digest(&b));
    assert_eq!(openai_chat::digest_message(&s).unwrap(), openai_chat::digest_message(&b).unwrap());
}

#[test]
fn content_changes_digest() {
    assert_ne!(a_digest(&a_user("one")), a_digest(&a_user("two")));
}

#[test]
fn tool_result_string_vs_blocks_equal() {
    let s = a_result("tu_1", "output text");
    let b = json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type": "text", "text": "output text"}]}]});
    assert_eq!(a_digest(&s), a_digest(&b));
}

#[test]
fn chain_prefix_property() {
    let msgs = [a_user("task"), a_assistant("step", Some(("t1", "bash", json!({"command": "ls"}))))];
    let d: Vec<String> = msgs.iter().map(a_digest).collect();
    let chain2 = chain_hashes(&d);
    assert_eq!(chain2[0], chain_hashes(&d[..1])[0]);
    assert_ne!(chain2[1], chain2[0]);
}

#[test]
fn openai_tool_call_identity() {
    let d = |m: Value| openai_chat::digest_message(&m).unwrap();
    let m1 = o_assistant("thinking", Some(("c1", "bash", json!({"command": "ls"}))));
    assert_eq!(d(m1.clone()), d(o_assistant("thinking", Some(("c1", "bash", json!({"command": "ls"}))))));
    assert_ne!(d(m1), d(o_assistant("thinking", Some(("c1", "bash", json!({"command": "rm"}))))));
    assert_ne!(d(o_tool("c1", "ok")), d(json!({"role": "user", "content": "ok"})));
}

// --- store ----------------------------------------------------------------------------

fn an_entry(chars: usize) -> Entry {
    Entry::new(2, Arc::new(json!({"role": "user", "content": "x".repeat(chars)})), 10)
}

#[test]
fn put_tracks_total_size() {
    let store = PrefixStore::default();
    let e = an_entry(1000);
    let size = entry_size(&e.summary);
    store.put("h1".into(), e);
    assert_eq!(store.nbytes(), size);
    assert!(store.nbytes() >= 1000);
    assert_eq!(store.get("h1").unwrap().size, size);
}

#[test]
fn replacing_a_key_does_not_double_count() {
    let store = PrefixStore::default();
    store.put("h1".into(), an_entry(5000));
    store.put("h1".into(), an_entry(1000));
    assert_eq!(store.len(), 1);
    assert!(store.nbytes() < 2000);
}

#[test]
fn evicts_least_recently_used_until_under_byte_budget() {
    let store = PrefixStore::new(4096, 10_000);
    for i in 0..6 {
        store.put(format!("h{i}"), an_entry(3000));
    }
    assert!(store.nbytes() <= 10_000);
    assert_eq!(store.len(), 3);
    assert!(store.get("h0").is_none());
    assert!(store.get("h5").is_some());
}

#[test]
fn get_refreshes_recency_so_a_hot_entry_survives() {
    let store = PrefixStore::new(4096, 10_000);
    for i in 0..3 {
        store.put(format!("h{i}"), an_entry(3000));
    }
    store.get("h0"); // h1 is now the least recently used
    store.put("h9".into(), an_entry(3000));
    assert!(store.get("h0").is_some());
    assert!(store.get("h1").is_none());
}

#[test]
fn entry_count_still_bounds_when_entries_are_tiny() {
    let store = PrefixStore::new(4, 1_000_000_000);
    for i in 0..20 {
        store.put(format!("h{i}"), an_entry(10));
    }
    assert_eq!(store.len(), 4);
    assert!(store.get("h19").is_some());
}

#[test]
fn an_oversized_entry_is_kept_rather_than_evicting_to_empty() {
    let store = PrefixStore::new(4096, 100);
    store.put("big".into(), an_entry(50_000));
    assert_eq!(store.len(), 1);
    assert!(store.nbytes() > 100);
    store.put("big2".into(), an_entry(50_000));
    assert_eq!(store.len(), 1);
    assert!(store.get("big").is_none() && store.get("big2").is_some());
}

#[test]
fn engine_builds_its_store_from_config_or_keeps_the_one_it_was_handed() {
    let e = Engine::new(Config { store_max_entries: 9, store_max_bytes: 4321, ..Config::default() });
    assert_eq!((e.store.max_entries(), e.store.max_bytes()), (9, 4321));
    let mine = Arc::new(PrefixStore::new(7, 1234));
    let e = Engine::with_store(Config::default(), mine.clone());
    assert!(Arc::ptr_eq(&e.store, &mine));
}

#[test]
fn default_budget_holds_a_realistic_working_set() {
    let store = PrefixStore::default();
    for i in 0..200 {
        store.put(format!("h{i}"), an_entry(50_000));
    }
    assert_eq!(store.len(), 200);
}

// --- images ------------------------------------------------------------------------------

fn png_bytes(w: u32, h: u32, payload_bytes: usize) -> Vec<u8> {
    let mut b = b"\x89PNG\r\n\x1a\n".to_vec();
    b.extend(13u32.to_be_bytes());
    b.extend(b"IHDR");
    b.extend(w.to_be_bytes());
    b.extend(h.to_be_bytes());
    b.extend(b"\x08\x06\x00\x00\x00");
    b.extend(vec![0u8; payload_bytes]);
    b
}

fn jpeg_bytes(w: u16, h: u16, exif_bytes: u16) -> Vec<u8> {
    let mut b = b"\xff\xd8\xff\xe1".to_vec();
    b.extend((exif_bytes + 2).to_be_bytes());
    b.extend(vec![0u8; exif_bytes as usize]);
    b.extend(b"\xff\xc0");
    b.extend(17u16.to_be_bytes());
    b.push(8);
    b.extend(h.to_be_bytes());
    b.extend(w.to_be_bytes());
    b.extend([0u8; 12]);
    b
}

fn data_uri(raw: &[u8]) -> String {
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(raw))
}

#[test]
fn reads_dimensions_of_every_format() {
    assert_eq!(dimensions(&png_bytes(1536, 1024, 0)), Some((1536, 1024)));
    assert_eq!(dimensions(&jpeg_bytes(800, 600, 4000)), Some((800, 600)));
    let mut gif = b"GIF89a".to_vec();
    gif.extend(320u16.to_le_bytes());
    gif.extend(240u16.to_le_bytes());
    gif.extend([0u8; 8]);
    assert_eq!(dimensions(&gif), Some((320, 240)));
    let mut webp = b"RIFF\0\0\0\0WEBPVP8X".to_vec();
    webp.extend([0u8; 8]);
    webp.extend(&1023u32.to_le_bytes()[..3]);
    webp.extend(&767u32.to_le_bytes()[..3]);
    assert_eq!(dimensions(&webp), Some((1024, 768)));
    assert_eq!(dimensions(b"not an image at all"), None);
    assert_eq!(dimensions(b""), None);
}

#[test]
fn matches_the_published_visual_token_table() {
    for (w, h, expected) in [(200, 200, 64), (1000, 1000, 1296), (1092, 1092, 1521), (1920, 1080, 2691), (2000, 1500, 3888), (3840, 2160, MAX_IMAGE_TOKENS)] {
        assert_eq!(tokens_for_payload(&data_uri(&png_bytes(w, h, 0))).unwrap(), expected, "{w}x{h}");
    }
}

#[test]
fn payload_length_does_not_change_the_estimate() {
    let small = data_uri(&png_bytes(1536, 1024, 1_000));
    let large = data_uri(&png_bytes(1536, 1024, 2_000_000));
    assert!(large.len() > 100 * small.len());
    assert_eq!(tokens_for_payload(&small), tokens_for_payload(&large));
    assert_eq!(tokens_for_payload(&data_uri(&png_bytes(8000, 8000, 0))).unwrap(), MAX_IMAGE_TOKENS);
    assert_eq!(tokens_for_payload("https://example.com/cat.png").unwrap(), DEFAULT_IMAGE_TOKENS);
    assert_eq!(tokens_for_payload("data:image/png;base64,!!!!").unwrap(), DEFAULT_IMAGE_TOKENS);
}

#[test]
fn finds_payloads_in_all_three_dialect_shapes() {
    let body = json!({"messages": [
        {"content": [{"type": "image", "source": {"data": "AAAA"}}]},
        {"content": [{"type": "input_image", "image_url": "data:image/png;base64,BBBB"}]},
        {"content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]},
    ]});
    let mut found = vec![];
    image_payloads(&body, &mut found).unwrap();
    assert_eq!(found, ["AAAA", "data:image/png;base64,BBBB", "https://x/y.png"]);
    let doc = json!({"content": [{"type": "document", "source": {"data": "x".repeat(5000)}}]});
    let mut found = vec![];
    image_payloads(&doc, &mut found).unwrap();
    assert!(found.is_empty());
}

#[test]
fn estimate_is_no_longer_dominated_by_base64() {
    let uri = data_uri(&png_bytes(1536, 1024, 2_000_000));
    let body = json!({"model": "m", "input": [
        {"content": [{"type": "input_text", "text": "hello ".repeat(100)}]},
        {"content": [{"type": "input_image", "image_url": uri}]},
    ]});
    let est = estimate_tokens(&body).unwrap();
    assert!(uri.len() > 2_600_000 && est < 5_000);
    assert!(est as u64 > tokens_for_payload(&uri).unwrap());
    let text = json!({"model": "m", "input": [{"content": [{"type": "input_text", "text": "x".repeat(40_000)}]}]});
    assert_eq!(estimate_tokens(&text).unwrap(), cliff_core::py::dumps(&text, cliff_core::py::DUMPS_UNICODE).len() as i64 / 4);
}

#[test]
fn replay_accounting_agrees_with_the_trigger() {
    let uri = data_uri(&png_bytes(1536, 1024, 400_000));
    let msg = json!({"role": "user", "content": [{"type": "image", "source": {"data": uri}}]});
    assert_eq!(msg_chars(&msg).unwrap(), billable_chars(&msg).unwrap() + 2);
    let msgs: Vec<Value> = (0..8).map(|_| msg.clone()).collect();
    let per_msg: i64 = msgs.iter().map(|m| msg_chars(m).unwrap()).sum::<i64>() / 4;
    let body = json!({"model": "m", "messages": msgs});
    assert!(per_msg <= estimate_tokens(&body).unwrap() + 8);
    assert!(per_msg < 8 * MAX_IMAGE_TOKENS as i64 + 1_000);
}

// --- OpenAI Responses ---------------------------------------------------------------------

#[test]
fn model_step_stays_one_turn() {
    let items = codex_session(3);
    let refs: Vec<&Value> = items.iter().collect();
    let turns = group_turns(&refs, &openai_responses::DIALECT).unwrap();
    assert_eq!(turns.len(), 4); // leading head group + 3 step turns
    let kinds: Vec<&str> = items[turns[1].clone()].iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["reasoning", "function_call", "function_call_output"]);
    assert_eq!(turns.last().unwrap().end, items.len()); // a partition, in order
    let items = [r_user("task"), reasoning(0, None), r_assistant("thinking done"), fcall(0, "echo hi"), fout(0, "ok (exit 0)")];
    let refs: Vec<&Value> = items.iter().collect();
    let turns = group_turns(&refs, &openai_responses::DIALECT).unwrap();
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[1].len(), 4); // reasoning + message + call + output together
}

#[test]
fn responses_digest_ignores_id_and_status_but_tracks_content() {
    let d = |m: &Value| openai_responses::digest_message(m).unwrap();
    let mut b = fcall(1, "echo hi");
    b["id"] = json!("totally-different");
    b["status"] = json!("in_progress");
    assert_eq!(d(&fcall(1, "echo hi")), d(&b));
    assert_ne!(d(&fcall(1, "echo hi")), d(&fcall(2, "echo hi")));
    assert_ne!(d(&reasoning(1, None)), d(&reasoning(2, None)));
}

#[test]
fn compact_codex_history() {
    let items = codex_session(6);
    let refs: Vec<&Value> = items.iter().collect();
    let c = compact(&refs, &openai_responses::DIALECT, &Config { keep_recent: 2, ..Config::default() }).unwrap().unwrap();
    assert_eq!(c.head_len, 3);
    assert_eq!(c.summary["role"], "user");
    let text = c.summary["content"][0]["text"].as_str().unwrap();
    assert!(text.starts_with(SUMMARY_HEADER));
    assert!(text.contains("[exec_command]") && text.contains("step1 ok") && text.contains("thinking: planning step 0"));
    assert!(!text.contains("LOG ") && !text.contains("enc-"));
    let kept = &items[c.cut..];
    let kinds: Vec<&str> = kept[..3].iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["reasoning", "function_call", "function_call_output"]);
    assert!(kept[0]["encrypted_content"].as_str().unwrap().starts_with("enc-4"));
}

#[test]
fn engine_prefix_substitution_on_input_key() {
    let e = Engine::new(Config { threshold_tokens: 1, keep_recent: 1, ..Config::default() });
    let items = codex_session(5);
    let ctx = e.prepare(r_body(items.clone()), &openai_responses::DIALECT).unwrap();
    assert!(ctx.compacted);
    let out: Value = serde_json::from_slice(&ctx.outgoing_json()).unwrap();
    assert!(out["input"].as_array().unwrap().len() < items.len());
    assert_eq!(out["instructions"], r_body(vec![])["instructions"]);

    let mut items2 = items.clone();
    items2.extend([reasoning(7, Some("planning step 7")), fcall(7, "make step7"), fout(7, "ok")]);
    let ctx2 = e.prepare(r_body(items2.clone()), &openai_responses::DIALECT).unwrap();
    assert!(ctx2.modified);
    let out2: Value = serde_json::from_slice(&ctx2.outgoing_json()).unwrap();
    assert!(out2["input"][3]["content"][0]["text"].as_str().unwrap().starts_with(SUMMARY_HEADER));
    let tail = &out2["input"].as_array().unwrap()[out2["input"].as_array().unwrap().len() - 3..];
    assert_eq!(tail, &items2[items2.len() - 3..]);
}
