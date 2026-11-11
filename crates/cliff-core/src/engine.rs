//! Request pipeline: match stored prefixes, substitute, compact over
//! threshold.
//!
//! Clients resend the ORIGINAL history each request, so the store is keyed
//! by original-prefix chain hashes; compaction runs on the substituted
//! sequence and results are stored under the longer original prefix.
//!
//! Coordinate mapping when a stored entry matched:
//!
//! ```text
//! substituted = msgs[..base_head] + [summary] + msgs[base_cut..]
//! ```
//!
//! so index sub_cut in substituted coordinates maps back to the original
//! list as base_cut + (sub_cut - (base_head + 1)).
//!
//! ## Parallelism
//!
//! `prepare` is CPU-bound and synchronous; the proxy runs it on the rayon
//! pool, never on an async worker. Inside it, the per-message work — the
//! canonical digest of every message, and its billable size when a
//! compaction needs it — is an independent map over messages and runs as a
//! rayon `par_iter`, forked against the whole-body size estimate with
//! `rayon::join`. The hash chain and the threshold replay are inherently
//! sequential scans over those results and stay serial.

use std::sync::{Arc, OnceLock};

use log::{info, warn};
use rayon::prelude::*;
use serde_json::Value;

use crate::cliff::{compact, Compaction};
use crate::config::Config;
use crate::dialects::{Dialect, SUMMARY_HEADER};
use crate::hashing::chain_hashes;
use crate::images::{image_payload_here, image_payloads, tokens_for_payload};
use crate::py::{self, PyErr, PyResult, Sink, DUMPS_ASCII, DUMPS_UNICODE};
use crate::store::{Entry, PrefixStore};

/// Below this many messages, per-message work stays on the calling thread.
const PAR_MIN_LEN: usize = 8;

// --- sizing ------------------------------------------------------------------------

fn summary_fingerprint(summary: &Value) -> String {
    let text = match py::get(summary, "content") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => py::dumps(other, DUMPS_ASCII),
        None => String::new(),
    };
    py::sha256_hex(text.as_bytes())[..8].to_string()
}

fn adjust_for_images(mut chars: i64, payloads: &[&str]) -> PyResult<i64> {
    for payload in payloads {
        chars -= py::len(payload) as i64;
        chars += tokens_for_payload(payload)? as i64 * 4;
    }
    Ok(chars.max(0))
}

/// Serialized length of `obj`, with image payloads priced by dimensions.
///
/// The unit is characters so callers can compare against a chars/4 budget:
/// an image contributes its estimated tokens x4 instead of its base64
/// length. Every size decision must come through here: the trigger and the
/// replay loop that reproduces past trigger points have to agree on what a
/// message costs.
pub fn billable_chars(obj: &Value) -> PyResult<i64> {
    let chars = py::dumps_len(obj, DUMPS_UNICODE) as i64;
    let mut payloads = vec![];
    image_payloads(obj, &mut payloads)?;
    adjust_for_images(chars, &payloads)
}

/// chars/4 over the serialized request body, image blocks counted by size.
pub fn estimate_tokens(body: &Value) -> PyResult<i64> {
    Ok(billable_chars(body)? / 4)
}

/// `{**body, messages_key: items}` without copying anything: the request
/// body with its message list swapped for borrowed items. Serializes to the
/// exact bytes Python's `json.dumps` produces for the equivalent dict
/// (assigning to an existing key keeps its position).
pub struct BodyView<'a> {
    pub body: &'a Value,
    pub key: &'a str,
    pub items: Vec<&'a Value>,
}

impl BodyView<'_> {
    pub fn write<S: Sink>(&self, fmt: py::Fmt, sink: &mut S) {
        let map = self.body.as_object().expect("request body is a JSON object");
        py::write_object_with(map.keys().map(String::as_str), fmt, sink, 0, |key, sink, level| {
            if key == self.key {
                py::write_array(self.items.iter().copied(), fmt, sink, level);
            } else {
                py::write_value(&map[key], fmt, sink, level);
            }
        });
    }

    pub fn dumps_len(&self, fmt: py::Fmt) -> usize {
        let mut n = py::CharCount::default();
        self.write(fmt, &mut n);
        n.0
    }

    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write(DUMPS_UNICODE, &mut out);
        out
    }

    pub fn billable_chars(&self) -> PyResult<i64> {
        let chars = self.dumps_len(DUMPS_UNICODE) as i64;
        let mut payloads = vec![];
        // image_payloads() on the virtual dict: the top level first, then
        // its values with the message list substituted.
        if let Some(found) = image_payload_here(self.body)? {
            payloads.push(found);
        } else {
            for (key, value) in self.body.as_object().into_iter().flatten() {
                if key == self.key {
                    for item in &self.items {
                        image_payloads(item, &mut payloads)?;
                    }
                } else {
                    image_payloads(value, &mut payloads)?;
                }
            }
        }
        adjust_for_images(chars, &payloads)
    }

    pub fn estimate_tokens(&self) -> PyResult<i64> {
        Ok(self.billable_chars()? / 4)
    }
}

/// The proxy's precondition for handing a body to the engine: a dict whose
/// message list is a non-empty list of dicts.
pub fn is_engine_input(body: &Value, dialect: &Dialect) -> bool {
    match py::get(body, dialect.messages_key) {
        Some(Value::Array(msgs)) => !msgs.is_empty() && msgs.iter().all(Value::is_object),
        _ => false,
    }
}

// --- request context -------------------------------------------------------------

/// One message of the outgoing sequence: an original message (by index into
/// the request) or a synthetic summary.
#[derive(Clone, Debug)]
pub enum MsgRef {
    Orig(usize),
    Synth(Arc<Value>),
}

pub struct RequestCtx {
    pub body: Value,
    pub dialect: &'static Dialect,
    pub chain: Vec<String>,
    /// `Engine._msg_chars` per original message, computed in parallel the
    /// first time a compaction needs it.
    msg_chars: OnceLock<Vec<PyResult<i64>>>,
    /// Provenance of `substituted`: 0/0 means no substitution.
    pub base_cut: usize,
    pub base_head: usize,
    pub substituted: Vec<MsgRef>,
    /// Substitution and/or compaction happened.
    pub modified: bool,
    /// A compaction happened during prepare().
    pub compacted: bool,
    /// Highest escalation rung applied: 0 = base config only,
    /// 1 = keep_recent=1, 2 = +thought/thinking caps, 3 = summary truncated.
    pub rung: u8,
    /// Still over threshold once the ladder was exhausted.
    pub over_budget: bool,
    pub est_tokens_in: i64,
    pub est_tokens_out: i64,
    // Observability only.
    pub chain_steps: usize,
    pub summary_fp: String,
    pub out_msgs: usize,
}

impl RequestCtx {
    pub fn msgs(&self) -> &[Value] {
        self.body[self.dialect.messages_key].as_array().map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn resolve<'a>(&'a self, r: &'a MsgRef) -> &'a Value {
        match r {
            MsgRef::Orig(i) => &self.msgs()[*i],
            MsgRef::Synth(v) => v,
        }
    }

    fn resolve_all<'a>(&'a self, refs: &'a [MsgRef]) -> Vec<&'a Value> {
        refs.iter().map(|r| self.resolve(r)).collect()
    }

    /// The outgoing message list.
    pub fn substituted_values(&self) -> Vec<&Value> {
        self.resolve_all(&self.substituted)
    }

    fn view<'a>(&'a self, items: Vec<&'a Value>) -> BodyView<'a> {
        BodyView { body: &self.body, key: self.dialect.messages_key, items }
    }

    /// `outgoing_body()`: the body as it will be sent.
    pub fn outgoing(&self) -> BodyView<'_> {
        if self.modified {
            self.view(self.substituted_values())
        } else {
            self.view(self.msgs().iter().collect())
        }
    }

    /// `json.dumps(outgoing_body(), ensure_ascii=False).encode()`.
    pub fn outgoing_json(&self) -> Vec<u8> {
        self.outgoing().to_json().into_bytes()
    }

    fn msg_chars(&self) -> &[PyResult<i64>] {
        self.msg_chars.get_or_init(|| {
            self.msgs()
                .par_iter()
                .with_min_len(PAR_MIN_LEN)
                // +2 for the separator this message adds to the serialized array.
                .map(|m| billable_chars(m).map(|c| c + 2))
                .collect()
        })
    }

    fn chars_of(&self, r: &MsgRef) -> PyResult<i64> {
        match r {
            MsgRef::Orig(i) => self.msg_chars()[*i].clone(),
            MsgRef::Synth(v) => Ok(billable_chars(v)? + 2),
        }
    }
}

// --- the engine --------------------------------------------------------------------

pub struct Engine {
    pub cfg: Config,
    pub store: Arc<PrefixStore>,
}

impl Engine {
    pub fn new(cfg: Config) -> Engine {
        let store = Arc::new(PrefixStore::new(cfg.store_max_entries, cfg.store_max_bytes));
        Engine { cfg, store }
    }

    pub fn with_store(cfg: Config, store: Arc<PrefixStore>) -> Engine {
        Engine { cfg, store }
    }

    pub fn prepare(&self, body: Value, dialect: &'static Dialect) -> PyResult<RequestCtx> {
        if !is_engine_input(&body, dialect) {
            return Err(PyErr("ValueError: not an engine input".into()));
        }
        let msgs = body[dialect.messages_key].as_array().expect("checked above");
        let (digests, est_in) = rayon::join(
            || {
                msgs.par_iter()
                    .with_min_len(PAR_MIN_LEN)
                    .map(|m| (dialect.digest_message)(m))
                    .collect::<PyResult<Vec<String>>>()
            },
            || estimate_tokens(&body),
        );
        let chain = chain_hashes(&digests?);
        let est_tokens_in = est_in?;
        let n = msgs.len();

        let mut ctx = RequestCtx {
            body,
            dialect,
            chain,
            msg_chars: OnceLock::new(),
            base_cut: 0,
            base_head: 0,
            substituted: (0..n).map(MsgRef::Orig).collect(),
            modified: false,
            compacted: false,
            rung: 0,
            over_budget: false,
            est_tokens_in,
            est_tokens_out: 0,
            chain_steps: 0,
            summary_fp: String::new(),
            out_msgs: 0,
        };

        // Longest stored prefix wins.
        if let Some((i, entry)) = self.store.longest_prefix(&ctx.chain) {
            if entry.cut != i + 1 || entry.head_len > entry.cut {
                warn!("store entry inconsistent at depth {}; ignoring", i + 1);
            } else {
                ctx.base_cut = entry.cut;
                ctx.base_head = entry.head_len;
                ctx.substituted = (0..entry.head_len)
                    .map(MsgRef::Orig)
                    .chain([MsgRef::Synth(entry.summary.clone())])
                    .chain((entry.cut..n).map(MsgRef::Orig))
                    .collect();
                ctx.modified = true;
                info!(
                    "match: prefix depth {}/{} (head={}, summary#{})",
                    entry.cut,
                    n,
                    entry.head_len,
                    summary_fingerprint(&entry.summary)
                );
            }
        }

        // Proactive trigger on the outgoing (post-substitution) size.
        ctx.est_tokens_out = ctx.outgoing().estimate_tokens()?;
        let threshold = self.cfg.threshold_tokens;
        if ctx.est_tokens_out > threshold {
            self.compact_chain(&mut ctx, "proactive", false, None)?;
            // Escalate while still over budget: harsher knobs, cliff()
            // semantics untouched. Truncation (rung 3) is normally
            // reactive-only; strict mode walks it too and then refuses.
            let rungs: &[u8] = if self.cfg.strict { &[1, 2, 3] } else { &[1, 2] };
            for &rung in rungs {
                if ctx.est_tokens_out <= threshold {
                    break;
                }
                if rung == 3 {
                    if self.truncate_summary(&mut ctx, "strict")? {
                        ctx.rung = 3;
                    }
                } else if self.compact_chain(
                    &mut ctx,
                    &format!("escalated rung{rung}"),
                    true,
                    Some(&self.rung_cfg(rung)),
                )? {
                    ctx.rung = rung;
                }
            }
            if ctx.est_tokens_out > threshold {
                ctx.over_budget = true;
                if ctx.compacted {
                    info!(
                        "over budget after escalation (~{}k est > {}k); {}",
                        ctx.est_tokens_out / 1000,
                        threshold / 1000,
                        if self.cfg.strict { "refusing (strict)" } else { "sending anyway" }
                    );
                }
            }
        }
        Ok(ctx)
    }

    /// Called on an upstream context-length error. Walks the escalation
    /// ladder one rung per call; Ok(true) if the request should be replayed.
    pub fn reactive(&self, ctx: &mut RequestCtx) -> PyResult<bool> {
        if !ctx.compacted && ctx.rung == 0 && self.compact_chain(ctx, "reactive", true, None)? {
            return Ok(true);
        }
        while ctx.rung < 3 {
            ctx.rung += 1;
            if ctx.rung < 3 {
                let reason = format!("reactive rung{}", ctx.rung);
                if self.compact_chain(ctx, &reason, true, Some(&self.rung_cfg(ctx.rung)))? {
                    return Ok(true);
                }
            } else if self.truncate_summary(ctx, "reactive")? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Derived config for an escalation rung. Rung 1: minimal tail.
    /// Rung 2: additionally the lean summary knobs.
    fn rung_cfg(&self, rung: u8) -> Config {
        let mut cfg = Config { keep_recent: 1, ..self.cfg.clone() };
        if rung >= 2 {
            let cap = self.cfg.thought_max_chars;
            cfg.thought_max_chars = if cap <= 0 { 300 } else { cap.min(300) };
            cfg.keep_thinking = false;
        }
        cfg
    }

    fn summary_text(msg: &Value) -> &str {
        match py::get(msg, "content") {
            Some(Value::String(s)) => s,
            Some(Value::Array(blocks)) => blocks
                .iter()
                .filter(|b| b.is_object())
                .find_map(|b| py::get(b, "text").and_then(Value::as_str))
                .unwrap_or(""),
            _ => "",
        }
    }

    /// Last-resort rung: shrink the existing summary to fit the threshold
    /// budget, keeping the NEWEST parts. Never touches the head or the tail.
    fn truncate_summary(&self, ctx: &mut RequestCtx, reason: &str) -> PyResult<bool> {
        if !ctx.compacted || ctx.base_cut == 0 {
            return Ok(false);
        }
        let idx = ctx.base_head;
        let text = Self::summary_text(ctx.resolve(&ctx.substituted[idx])).to_string();
        if !text.starts_with(SUMMARY_HEADER) {
            return Ok(false);
        }
        let others: Vec<&Value> = ctx
            .substituted
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, r)| ctx.resolve(r))
            .collect();
        let fixed = ctx.view(others).dumps_len(DUMPS_UNICODE) as i64;
        let budget = self.cfg.threshold_tokens * 4 - fixed - SUMMARY_HEADER.len() as i64 - 64;
        let parts: Vec<&str> = py::strip(&text[SUMMARY_HEADER.len()..]).split("\n\n---\n\n").collect();
        let mut kept: Vec<&str> = vec![];
        let mut used: i64 = 0;
        for part in parts.iter().rev() {
            // newest first
            let n = py::len(part) as i64;
            if used + n > budget.max(0) {
                break;
            }
            kept.push(part);
            used += n + 9; // separator overhead
        }
        kept.reverse();
        let new_text = if kept.is_empty() {
            SUMMARY_HEADER.to_string()
        } else {
            format!("{SUMMARY_HEADER}\n\n{}", kept.join("\n\n---\n\n"))
        };
        if py::len(&new_text) >= py::len(&text) {
            return Ok(false); // nothing gained
        }
        let n_kept = kept.len();
        let new_summary = Arc::new((ctx.dialect.user_message)(new_text));
        ctx.substituted[idx] = MsgRef::Synth(new_summary.clone());
        ctx.modified = true;
        let before = ctx.est_tokens_out;
        ctx.est_tokens_out = ctx.outgoing().estimate_tokens()?;
        let fp = summary_fingerprint(&new_summary);
        self.store
            .put(ctx.chain[ctx.base_cut - 1].clone(), Entry::new(ctx.base_head, new_summary, ctx.base_cut));
        info!(
            "{} rung3: summary truncated ({} -> {} parts, ~{}k -> ~{}k est tokens) summary#{}",
            reason,
            parts.len(),
            n_kept,
            before / 1000,
            ctx.est_tokens_out / 1000,
            fp
        );
        Ok(true)
    }

    /// Compact by replaying threshold crossings over the sequence.
    ///
    /// Feeds messages in order, compacting whenever the running estimate
    /// crosses the threshold; each step drops the prior step's summary, so a
    /// long history arriving at once (empty store, late attach) yields the
    /// same last-cycle-only summary an incrementally built chain would.
    ///
    /// With `force`, a sequence that never crosses is compacted once at the
    /// end (reactive path: the provider rejected it regardless).
    fn compact_chain(&self, ctx: &mut RequestCtx, reason: &str, force: bool, knobs: Option<&Config>) -> PyResult<bool> {
        let knobs = knobs.unwrap_or(&self.cfg);
        let n = ctx.msgs().len();
        let threshold_chars = self.cfg.threshold_tokens * 4;

        let mut replay = Replay::start(ctx, reason)?;
        for i in replay.orig_cut..n {
            replay.feed(i)?;
            if replay.chars > threshold_chars {
                let Some(result) = compact(&ctx.resolve_all(&replay.working), ctx.dialect, knobs)? else {
                    continue; // not enough turns yet; keep feeding
                };
                if !replay.apply(result)? {
                    // MIRRORED QUIRK: Python returns here without adopting
                    // earlier generations into the context or storing them.
                    return Ok(replay.n_compactions > 0);
                }
            }
        }
        if replay.n_compactions == 0 && force {
            if let Some(result) = compact(&ctx.resolve_all(&replay.working), ctx.dialect, knobs)? {
                replay.apply(result)?;
            }
        }

        let Replay { working, head_len, orig_cut, last_summary, n_compactions, .. } = replay;
        let Some(last_summary) = last_summary else {
            info!("compact ({}): nothing to compact", reason);
            return Ok(false);
        };

        let before = ctx.est_tokens_out;
        let out_len = working.len();
        ctx.substituted = working;
        ctx.base_cut = orig_cut;
        ctx.base_head = head_len;
        ctx.modified = true;
        ctx.compacted = true;
        ctx.est_tokens_out = ctx.outgoing().estimate_tokens()?;
        ctx.chain_steps = n_compactions;
        ctx.summary_fp = summary_fingerprint(&last_summary);
        ctx.out_msgs = out_len;
        self.store.put(ctx.chain[orig_cut - 1].clone(), Entry::new(head_len, last_summary, orig_cut));
        info!(
            "compact ({}): ~{}k -> ~{}k est tokens, {} -> {} messages, {} chain step(s), stored@{} summary#{}",
            reason,
            before / 1000,
            ctx.est_tokens_out / 1000,
            n,
            out_len,
            n_compactions,
            orig_cut,
            ctx.summary_fp
        );
        Ok(true)
    }
}

/// The running state of `compact_chain`'s replay (the Python version keeps
/// it in `nonlocal` variables shared with a nested `apply()`).
///
/// Invariant: `working == replacement + msgs[orig_cut..fed]`, where
/// replacement = msgs[..head_len] + [summary] once a compaction exists, and
/// `chars` is the billable size of `{**body, key: working}`.
struct Replay<'a> {
    ctx: &'a RequestCtx,
    reason: &'a str,
    fixed_chars: i64,
    working: Vec<MsgRef>,
    chars: i64,
    head_len: usize,
    orig_cut: usize,
    have_summary: bool,
    last_summary: Option<Arc<Value>>,
    n_compactions: usize,
}

impl<'a> Replay<'a> {
    fn start(ctx: &'a RequestCtx, reason: &'a str) -> PyResult<Replay<'a>> {
        let fixed_chars = ctx.view(vec![]).billable_chars()?;
        let mut replay = if ctx.base_cut > 0 {
            Replay {
                working: ctx.substituted[..ctx.base_head + 1].to_vec(),
                head_len: ctx.base_head,
                orig_cut: ctx.base_cut,
                have_summary: true,
                ..Replay::empty(ctx, reason, fixed_chars)
            }
        } else {
            Replay::empty(ctx, reason, fixed_chars)
        };
        replay.chars = replay.recount()?;
        Ok(replay)
    }

    fn empty(ctx: &'a RequestCtx, reason: &'a str, fixed_chars: i64) -> Replay<'a> {
        Replay {
            ctx,
            reason,
            fixed_chars,
            working: vec![],
            chars: fixed_chars,
            head_len: 0,
            orig_cut: 0,
            have_summary: false,
            last_summary: None,
            n_compactions: 0,
        }
    }

    fn recount(&self) -> PyResult<i64> {
        let mut chars = self.fixed_chars;
        for r in &self.working {
            chars += self.ctx.chars_of(r)?;
        }
        Ok(chars)
    }

    fn feed(&mut self, i: usize) -> PyResult<()> {
        let r = MsgRef::Orig(i);
        self.chars += self.ctx.chars_of(&r)?;
        self.working.push(r);
        Ok(())
    }

    /// Map result.cut to original coordinates and adopt the result.
    /// Ok(false) is Python's `apply()` returning False (fail-open).
    fn apply(&mut self, result: Compaction) -> PyResult<bool> {
        let new_cut = if self.have_summary {
            if result.cut < self.head_len + 1 {
                warn!("compact ({}): cut {} inside replacement; fail-open", self.reason, result.cut);
                return Ok(false);
            }
            self.orig_cut + (result.cut - (self.head_len + 1))
        } else {
            result.cut
        };
        if !(1..=self.ctx.msgs().len()).contains(&new_cut) {
            warn!("compact ({}): cut maps out of range; fail-open", self.reason);
            return Ok(false);
        }
        let summary = Arc::new(result.summary);
        let mut next: Vec<MsgRef> = self.working[..result.head_len].to_vec();
        next.push(MsgRef::Synth(summary.clone()));
        next.extend_from_slice(&self.working[result.cut..]);
        self.working = next;
        self.chars = self.recount()?;
        self.head_len = result.head_len;
        self.orig_cut = new_cut;
        self.have_summary = true;
        self.last_summary = Some(summary);
        self.n_compactions += 1;
        Ok(true)
    }
}

/// `Engine._msg_chars`: what one message adds to the serialized array.
pub fn msg_chars(msg: &Value) -> PyResult<i64> {
    Ok(billable_chars(msg)? + 2)
}
