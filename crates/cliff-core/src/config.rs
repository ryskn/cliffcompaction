//! Configuration. Settable via CLIFF_* environment variables; CLI flags
//! override.

use crate::py;

pub const DEFAULT_ANTHROPIC_UPSTREAM: &str = "https://api.anthropic.com";
pub const DEFAULT_OPENAI_UPSTREAM: &str = "https://api.openai.com";

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    // --- compaction knobs ---
    /// Proactive trigger: compact when the outgoing request exceeds this
    /// (estimated at chars/4).
    pub threshold_tokens: i64,
    /// Number of recent assistant-step turns kept verbatim.
    pub keep_recent: i64,
    /// Max chars of assistant text kept per summarized turn. 0 = unlimited.
    pub thought_max_chars: i64,
    /// Max chars of a tool-call signature (serialized arguments).
    pub cmd_max_chars: i64,
    /// Tool results longer than this are dropped from the summary entirely.
    pub result_max_chars: i64,
    /// Sanity cap for human text inside the summary.
    pub human_max_chars: i64,
    /// Keep thinking/reasoning content in summaries as text. Signed thinking
    /// BLOCKS are never re-sent from the compacted region either way.
    pub keep_thinking: bool,
    /// Max chars of thinking text kept per summarized turn. 0 = unlimited.
    pub thinking_max_chars: i64,

    // --- proxy behavior ---
    /// Observe, hash, log — but never modify a request.
    pub shadow: bool,
    /// Refuse rather than forward a request still over threshold once the
    /// escalation ladder is exhausted (measurement runs).
    pub strict: bool,
    /// Dump every dialect-handled request here as JSON. Observability only.
    pub debug_dir: String,
    pub anthropic_upstream: String,
    pub openai_upstream: String,
    pub host: String,
    pub port: u16,

    // --- state store ---
    pub store_max_entries: usize,
    pub store_max_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            threshold_tokens: 200_000,
            keep_recent: 3,
            thought_max_chars: 0,
            cmd_max_chars: 150,
            result_max_chars: 500,
            human_max_chars: 20_000,
            keep_thinking: true,
            thinking_max_chars: 0,
            shadow: false,
            strict: false,
            debug_dir: String::new(),
            anthropic_upstream: DEFAULT_ANTHROPIC_UPSTREAM.into(),
            openai_upstream: DEFAULT_OPENAI_UPSTREAM.into(),
            host: "127.0.0.1".into(),
            port: 8257,
            store_max_entries: 4096,
            store_max_bytes: 64 * 1024 * 1024,
        }
    }
}

fn env_int(name: &str, default: i64) -> i64 {
    std::env::var(name).ok().and_then(|v| py::parse_int(&v)).unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(py::strip(&v).to_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => default,
    }
}

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

impl Config {
    pub fn from_env() -> Config {
        let d = Config::default();
        Config {
            threshold_tokens: env_int("CLIFF_THRESHOLD_TOKENS", d.threshold_tokens),
            keep_recent: env_int("CLIFF_KEEP_RECENT", d.keep_recent),
            thought_max_chars: env_int("CLIFF_THOUGHT_MAX_CHARS", d.thought_max_chars),
            cmd_max_chars: env_int("CLIFF_CMD_MAX_CHARS", d.cmd_max_chars),
            result_max_chars: env_int("CLIFF_RESULT_MAX_CHARS", d.result_max_chars),
            human_max_chars: env_int("CLIFF_HUMAN_MAX_CHARS", d.human_max_chars),
            keep_thinking: env_bool("CLIFF_KEEP_THINKING", d.keep_thinking),
            thinking_max_chars: env_int("CLIFF_THINKING_MAX_CHARS", d.thinking_max_chars),
            shadow: env_bool("CLIFF_SHADOW", false),
            strict: env_bool("CLIFF_STRICT", false),
            debug_dir: env_str("CLIFF_DEBUG_DIR", ""),
            anthropic_upstream: env_str("CLIFF_ANTHROPIC_UPSTREAM", DEFAULT_ANTHROPIC_UPSTREAM),
            openai_upstream: env_str("CLIFF_OPENAI_UPSTREAM", DEFAULT_OPENAI_UPSTREAM),
            host: env_str("CLIFF_HOST", "127.0.0.1"),
            port: u16::try_from(env_int("CLIFF_PORT", 8257)).unwrap_or(8257),
            store_max_entries: usize::try_from(env_int("CLIFF_STORE_MAX_ENTRIES", 4096)).unwrap_or(4096),
            store_max_bytes: usize::try_from(env_int("CLIFF_STORE_MAX_BYTES", 64 * 1024 * 1024))
                .unwrap_or(64 * 1024 * 1024),
        }
    }
}
