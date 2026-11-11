//! CliffCompaction core.
//!
//! Pure, synchronous and deterministic: everything a request needs between
//! "bytes arrived" and "bytes to send", with no IO and no async. The proxy
//! runs it on a CPU pool; tests and the conformance harness call it
//! directly.
//!
//! Module map (each mirrors the Python module of the same name):
//!
//! - [`py`]        Python semantics the port must reproduce (the oracle's
//!                 `len`, `strip`, `json.dumps`, truthiness, exceptions)
//! - [`hashing`]   canonical digests and the prefix hash chain
//! - [`images`]    image token estimation
//! - [`dialects`]  Anthropic Messages, OpenAI Chat, OpenAI Responses
//! - [`cliff`]     the compactor
//! - [`store`]     the LRU prefix store
//! - [`engine`]    the request pipeline (match, substitute, compact, escalate)

pub mod cliff;
pub mod config;
pub mod conformance;
pub mod dialects;
pub mod engine;
pub mod hashing;
pub mod identity;
pub mod images;
pub mod py;
pub mod store;

pub use config::Config;
pub use dialects::{Dialect, SUMMARY_HEADER};
pub use engine::{Engine, RequestCtx};
pub use py::{PyErr, PyResult};
pub use store::{Entry, PrefixStore};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
