//! CliffCompaction: the IO shell around `cliff-core` — the proxy, the event
//! hub, the CLI, daemon management and `cliff watch`.

pub mod cli;
pub mod daemon;
pub mod events;
pub mod logging;
pub mod proxy;
pub mod ui;
pub mod watch;
