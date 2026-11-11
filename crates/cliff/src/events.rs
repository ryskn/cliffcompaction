//! Bounded in-memory event stream for `cliff watch`.
//!
//! Emission is fire-and-forget: `emit()` never blocks and never fails, so a
//! watcher that stalls or dies cannot affect the request path. Events live
//! only in memory, capped by `maxlen`; nothing is written to disk.
//!
//! Fan-out is one bounded `mpsc` channel per subscriber fed with
//! `try_send`: a subscriber that falls `QUEUE_DEPTH` events behind loses
//! the NEWEST events (counted in `dropped`), exactly like the Python
//! `asyncio.Queue.put_nowait` it replaces. (`tokio::sync::broadcast` would
//! drop the oldest instead; either keeps the request path non-blocking.)
//! The hub's own state sits behind a `std::sync::Mutex` held for a few
//! pointer pushes, never across an `.await`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use cliff_core::py::{self, DUMPS_UNICODE};
use serde_json::{Map, Value};
use tokio::sync::mpsc;

/// Per-subscriber queue depth.
pub const QUEUE_DEPTH: usize = 512;

pub type Event = Arc<Value>;

struct Inner {
    buf: VecDeque<Event>,
    maxlen: usize,
    subs: Vec<(u64, mpsc::Sender<Event>)>,
    next_sub: u64,
    seq: u64,
    dropped: u64,
}

pub struct EventHub {
    inner: Mutex<Inner>,
}

impl Default for EventHub {
    fn default() -> Self {
        EventHub::new(1024)
    }
}

/// A live subscription; unsubscribes when dropped (the Python `finally:
/// hub.unsubscribe(q)`).
pub struct Subscription {
    hub: Arc<EventHub>,
    id: u64,
    pub rx: mpsc::Receiver<Event>,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.hub.lock().subs.retain(|(id, _)| *id != self.id);
    }
}

pub fn now_secs() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

impl EventHub {
    pub fn new(maxlen: usize) -> EventHub {
        EventHub {
            inner: Mutex::new(Inner { buf: VecDeque::new(), maxlen, subs: vec![], next_sub: 0, seq: 0, dropped: 0 }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Record an event: `{"seq": n, "t": now, **fields}`. Never fails.
    pub fn emit(&self, fields: Vec<(&str, Value)>) {
        let mut inner = self.lock();
        inner.seq += 1;
        let mut ev = Map::new();
        ev.insert("seq".into(), inner.seq.into());
        ev.insert("t".into(), Value::from(now_secs()));
        for (k, v) in fields {
            ev.insert(k.into(), v);
        }
        let ev: Event = Arc::new(Value::Object(ev));
        if inner.buf.len() == inner.maxlen {
            inner.buf.pop_front();
        }
        if inner.maxlen > 0 {
            inner.buf.push_back(ev.clone());
        }
        let mut dropped = 0;
        inner.subs.retain(|(_, tx)| match tx.try_send(ev.clone()) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => {
                dropped += 1;
                true
            }
            Err(mpsc::error::TrySendError::Closed(_)) => false,
        });
        inner.dropped += dropped;
    }

    pub fn backlog(&self) -> Vec<Event> {
        self.lock().buf.iter().cloned().collect()
    }

    /// Subscribe and take the backlog atomically: no event is both in the
    /// backlog and queued, and none falls between them.
    pub fn subscribe(self: &Arc<Self>) -> (Subscription, Vec<Event>) {
        let (tx, rx) = mpsc::channel(QUEUE_DEPTH);
        let mut inner = self.lock();
        inner.next_sub += 1;
        let id = inner.next_sub;
        inner.subs.push((id, tx));
        let backlog = inner.buf.iter().cloned().collect();
        drop(inner);
        (Subscription { hub: self.clone(), id, rx }, backlog)
    }

    pub fn subscribers(&self) -> usize {
        self.lock().subs.len()
    }

    pub fn dropped(&self) -> u64 {
        self.lock().dropped
    }
}

/// One SSE frame, serialized like Python's `json.dumps(ev, ensure_ascii=False)`.
pub fn sse(ev: &Value) -> Vec<u8> {
    let mut out = String::from("data: ");
    py::write_value(ev, DUMPS_UNICODE, &mut out, 0);
    out.push_str("\n\n");
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hub_is_bounded_and_ordered() {
        let hub = EventHub::new(3);
        for i in 0..6 {
            hub.emit(vec![("kind", json!("pass")), ("i", json!(i))]);
        }
        let buf = hub.backlog();
        assert_eq!(buf.iter().map(|e| e["i"].as_i64().unwrap()).collect::<Vec<_>>(), [3, 4, 5]);
        assert_eq!(buf.iter().map(|e| e["seq"].as_i64().unwrap()).collect::<Vec<_>>(), [4, 5, 6]);
    }

    #[test]
    fn slow_subscriber_drops_instead_of_growing() {
        let hub = Arc::new(EventHub::default());
        let (mut sub, _) = hub.subscribe();
        for i in 0..1000 {
            hub.emit(vec![("kind", json!("pass")), ("i", json!(i))]);
        }
        let mut queued = 0;
        while sub.rx.try_recv().is_ok() {
            queued += 1;
        }
        assert_eq!(queued, 512);
        assert_eq!(hub.dropped(), 1000 - 512);
        drop(sub);
        assert_eq!(hub.subscribers(), 0);
    }

    #[test]
    fn sse_frame_is_python_json() {
        let frame = sse(&json!({"seq": 1, "t": 1.5, "sid": "é"}));
        assert_eq!(String::from_utf8(frame).unwrap(), "data: {\"seq\": 1, \"t\": 1.5, \"sid\": \"é\"}\n\n");
    }
}
