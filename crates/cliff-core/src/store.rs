//! Prefix store: chain_hash(original prefix S) -> compacted replacement.
//!
//! An entry holds only (head_len, summary, cut); the substitution for a
//! request extending prefix S (|S| = cut) is msgs[..head_len] + [summary] +
//! msgs[cut..]. Head bytes come from the CURRENT request, so volatile fields
//! (cache_control) are never replayed stale. The store is a cache: the
//! compactor is deterministic, so lost entries are recomputed identically on
//! demand.
//!
//! Concurrency: one `std::sync::Mutex` around an O(1) LRU. Critical sections
//! are a hash lookup plus a list splice, never held across an `.await` (the
//! store is only touched from CPU jobs), which is the case the Tokio docs
//! recommend a blocking mutex for. Because entries are a deterministic
//! function of their key, two requests racing to fill the same key write the
//! same value: no coordination (single-flight, actor) is needed for
//! correctness.

use std::sync::{Arc, Mutex, MutexGuard};

use hashlink::LinkedHashMap;
use serde_json::Value;

use crate::py::{self, DUMPS_UNICODE};

#[derive(Debug, Clone)]
pub struct Entry {
    pub head_len: usize,
    /// Shared, immutable: a hit hands out a pointer, not a copy.
    pub summary: Arc<Value>,
    /// Length of the original prefix S this entry replaces.
    pub cut: usize,
    /// Serialized chars; filled by `put()`.
    pub size: usize,
}

impl Entry {
    pub fn new(head_len: usize, summary: Arc<Value>, cut: usize) -> Entry {
        Entry { head_len, summary, cut, size: 0 }
    }
}

/// Serialized size of a summary, in characters (`len(json.dumps(summary,
/// ensure_ascii=False))`). A proxy for its memory, not a guarantee.
pub fn entry_size(summary: &Value) -> usize {
    py::dumps_len(summary, DUMPS_UNICODE)
}

struct Inner {
    data: LinkedHashMap<String, Entry>,
    bytes: usize,
}

/// LRU cache of compacted prefixes, bounded by total size and entry count.
pub struct PrefixStore {
    max_entries: usize,
    max_bytes: usize,
    inner: Mutex<Inner>,
}

impl Default for PrefixStore {
    fn default() -> Self {
        PrefixStore::new(4096, 64 * 1024 * 1024)
    }
}

impl PrefixStore {
    pub fn new(max_entries: usize, max_bytes: usize) -> PrefixStore {
        PrefixStore { max_entries, max_bytes, inner: Mutex::new(Inner { data: LinkedHashMap::new(), bytes: 0 }) }
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // A panic while holding the lock cannot leave the map itself torn
        // (LinkedHashMap operations are panic-safe); recover rather than
        // turning one bad request into a dead proxy.
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn get(&self, chain_hash: &str) -> Option<Entry> {
        let mut inner = self.lock();
        let entry = inner.data.to_back(chain_hash)?;
        Some(entry.clone())
    }

    /// The deepest stored prefix of `chain`: scans from the longest prefix
    /// down under a single lock acquisition, refreshing only the hit (the
    /// Python loop calls `get` per depth; misses never touch recency, so the
    /// result and the LRU order are identical).
    pub fn longest_prefix(&self, chain: &[String]) -> Option<(usize, Entry)> {
        let mut inner = self.lock();
        for i in (0..chain.len()).rev() {
            if let Some(entry) = inner.data.to_back(&chain[i]) {
                return Some((i, entry.clone()));
            }
        }
        None
    }

    pub fn put(&self, chain_hash: String, mut entry: Entry) {
        entry.size = entry_size(&entry.summary);
        let mut inner = self.lock();
        if let Some(old) = inner.data.remove(&chain_hash) {
            inner.bytes -= old.size;
        }
        inner.bytes += entry.size;
        inner.data.insert(chain_hash, entry);
        // Never evict down to empty: an entry over budget on its own is
        // still the one the next request will look for.
        while inner.data.len() > 1 && (inner.data.len() > self.max_entries || inner.bytes > self.max_bytes) {
            if let Some((_, evicted)) = inner.data.pop_front() {
                inner.bytes -= evicted.size;
            }
        }
    }

    pub fn nbytes(&self) -> usize {
        self.lock().bytes
    }

    pub fn len(&self) -> usize {
        self.lock().data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
