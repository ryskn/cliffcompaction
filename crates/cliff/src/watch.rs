//! `cliff watch`: a live view of the sessions moving through the proxy.
//!
//! Reads the daemon's SSE event stream and renders one row per session.
//! Works against either implementation's daemon (the event schema is the
//! same). Counts cover what this watcher has seen — the backlog it was
//! handed on connect plus everything since.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{IsTerminal, Read, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cliff_core::py;
use futures_util::StreamExt;
use serde_json::Value;

use crate::events::now_secs;
use crate::ui::{Term, BRAND, DIM, FAINT, IDLE, TEXT, YELLOW};

/// Seconds without traffic before the row dims...
pub const IDLE_AFTER: f64 = 120.0;
/// ...and then leaves the list.
pub const DROP_AFTER: f64 = 1800.0;
/// Per-session samples kept for the sparkline.
pub const SERIES: usize = 240;
/// Events kept for the feed.
pub const FEED: usize = 200;
/// A row spans at least this fraction of the threshold.
pub const MIN_SPAN: f64 = 0.25;

const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// `claude-sonnet-4-5-20250929` -> `sonnet-4-5`: drop the vendor prefix and
/// the release date.
pub fn short_model(name: &str) -> String {
    let mut name = if name.is_empty() { "?" } else { name };
    if name.len() >= 9 && name.is_char_boundary(name.len() - 9) {
        let tail = &name[name.len() - 9..];
        if tail.starts_with('-') && tail[1..].bytes().all(|b| b.is_ascii_digit()) {
            name = &name[..name.len() - 9];
        }
    }
    for prefix in ["claude-", "anthropic/", "openai/"] {
        if let Some(rest) = name.strip_prefix(prefix) {
            name = rest;
        }
    }
    if name.is_empty() { "?".into() } else { name.into() }
}

/// Python's `f"{x:.Nf}"`: round half to even on the exact binary value.
/// (Rust's `{:.N}` does the same; this keeps the intent explicit.)
fn fixed(x: f64, digits: usize) -> String {
    format!("{x:.digits$}")
}

pub fn kfmt(n: i64) -> String {
    if n >= 1_000_000 {
        format!("{}M", fixed(n as f64 / 1_000_000.0, 1))
    } else if n >= 1000 {
        format!("{}k", fixed(n as f64 / 1000.0, 0))
    } else {
        n.to_string()
    }
}

pub fn agefmt(sec: f64) -> String {
    if sec < 60.0 {
        format!("{}s", sec as i64)
    } else if sec < 3600.0 {
        format!("{}m", (sec / 60.0).floor() as i64)
    } else {
        format!("{}h", (sec / 3600.0).floor() as i64)
    }
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `ev.get(key) or default` for the integer fields.
fn int(ev: &Value, key: &str) -> i64 {
    ev.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn text<'a>(ev: &'a Value, key: &str) -> &'a str {
    ev.get(key).and_then(Value::as_str).unwrap_or("")
}

fn truthy(ev: &Value, key: &str) -> bool {
    ev.get(key).is_some_and(py::truthy)
}

fn event_time(ev: &Value) -> f64 {
    ev.get("t").and_then(Value::as_f64).filter(|t| *t != 0.0).unwrap_or_else(now_secs)
}

pub struct Session {
    pub sid: String,
    series: VecDeque<i64>,
    marks: VecDeque<u64>,
    n: u64,
    pub compactions: u64,
    pub model: String,
    pub dialect: String,
    pub cut: i64,
    pub total: i64,
    pub est: i64,
    pub last: f64,
}

impl Session {
    fn new(sid: &str, ev: &Value) -> Session {
        let mut s = Session {
            sid: sid.into(),
            series: VecDeque::new(),
            marks: VecDeque::new(),
            n: 0,
            compactions: 0,
            model: String::new(),
            dialect: String::new(),
            cut: 0,
            total: 0,
            est: 0,
            last: 0.0,
        };
        s.update(ev);
        s
    }

    fn update(&mut self, ev: &Value) {
        self.model = short_model(match text(ev, "model") {
            "" => "?",
            m => m,
        });
        self.dialect = match text(ev, "dialect") {
            "" => "?".into(),
            d => d.into(),
        };
        self.cut = int(ev, "cut");
        self.total = int(ev, "total");
        self.est = match int(ev, "est_out") {
            0 => int(ev, "est_in"),
            e => e,
        };
        self.last = event_time(ev);
        if self.series.len() == SERIES {
            self.series.pop_front();
        }
        self.series.push_back(self.est);
        if text(ev, "kind") == "compact" {
            self.compactions += 1;
            if self.marks.len() == SERIES {
                self.marks.pop_front();
            }
            self.marks.push_back(self.n);
        }
        self.n += 1;
    }

    /// One cell per request, newest at the right edge. The ceiling is the
    /// threshold; the floor follows the session's own low-water mark.
    fn spark(&self, term: &Term, width: usize, threshold: i64, idle: bool) -> String {
        let skip = self.series.len().saturating_sub(width);
        let pts: Vec<i64> = self.series.iter().skip(skip).copied().collect();
        let first = self.n - pts.len() as u64;
        let marks: HashSet<u64> = self.marks.iter().copied().filter(|m| *m >= first).collect();
        let hi = pts.iter().copied().max().unwrap_or(0).max(threshold);
        // `int(threshold * MIN_SPAN) or 1`
        let floor = match (threshold as f64 * MIN_SPAN) as i64 {
            0 => 1,
            f => f,
        };
        let span = (hi - pts.iter().copied().min().unwrap_or(0)).max(floor);
        let lo = hi - span;
        let mut out = " ".repeat(width - pts.len());
        for (i, v) in pts.iter().enumerate() {
            let f = (v - lo) as f64 / span as f64;
            let idx = ((f * (BARS.len() - 1) as f64 + 0.5) as i64).clamp(0, BARS.len() as i64 - 1) as usize;
            let hot = marks.contains(&(first + i as u64));
            out.push_str(&term.c(if idle { IDLE } else if hot { YELLOW } else { BRAND }, &BARS[idx].to_string()));
        }
        out
    }
}

pub struct Watcher {
    pub port: u16,
    pub sessions: HashMap<String, Session>,
    pub feed: VecDeque<Value>,
    pub requests: u64,
    pub compactions: u64,
    pub window_start: f64,
    pub threshold: i64,
    pub keep_recent: i64,
    pub shadow: bool,
    pub uptime: f64,
    pub connected: bool,
    pub paused: bool,
    pub stop: bool,
}

impl Watcher {
    pub fn new(port: u16) -> Watcher {
        Watcher {
            port,
            sessions: HashMap::new(),
            feed: VecDeque::new(),
            requests: 0,
            compactions: 0,
            window_start: now_secs(),
            threshold: 200_000,
            keep_recent: 3,
            shadow: false,
            uptime: 0.0,
            connected: false,
            paused: false,
            stop: false,
        }
    }

    fn push_feed(&mut self, ev: &Value) {
        if self.feed.len() == FEED {
            self.feed.pop_front();
        }
        self.feed.push_back(ev.clone());
    }

    pub fn on_event(&mut self, ev: &Value) {
        let sid = text(ev, "sid").to_string();
        if sid.is_empty() {
            return;
        }
        self.window_start = self.window_start.min(event_time(ev));
        self.requests += 1;
        if truthy(ev, "aux") {
            // Real traffic and real money, but not a turn of the
            // conversation: the feed only, never the session's row.
            self.push_feed(ev);
            return;
        }
        if text(ev, "kind") == "compact" {
            self.compactions += 1;
        }
        match self.sessions.get_mut(&sid) {
            Some(s) => s.update(ev),
            None => {
                self.sessions.insert(sid.clone(), Session::new(&sid, ev));
            }
        }
        self.push_feed(ev);
    }

    pub fn live_sessions(&self) -> Vec<&Session> {
        let now = now_secs();
        let mut live: Vec<&Session> = self.sessions.values().filter(|s| now - s.last < DROP_AFTER).collect();
        live.sort_by(|a, b| b.last.total_cmp(&a.last));
        live
    }

    pub fn render(&self, t: &Term) -> String {
        let w = t.columns;
        let now = now_secs();
        let sessions = self.live_sessions();
        let idle_n = sessions.iter().filter(|s| now - s.last >= IDLE_AFTER).count();
        let rows = t.rows.max(20);
        let dot = t.c(FAINT, " · ");

        let mut head = format!(
            " {}{}   {}{}{dot}{}{}",
            t.bold(BRAND, "⟨cliff⟩"),
            t.c(DIM, " watch"),
            t.c(DIM, "threshold "),
            t.c(TEXT, &kfmt(self.threshold)),
            t.c(DIM, "keep "),
            t.c(TEXT, &self.keep_recent.to_string()),
        );
        if self.uptime != 0.0 {
            head += &format!("{dot}{}{}", t.c(DIM, "up "), t.c(TEXT, &agefmt(self.uptime)));
        }
        if self.shadow {
            head += &format!("{dot}{}", t.c(YELLOW, "shadow"));
        }
        if !self.connected {
            head += &format!("   {}", t.c(YELLOW, "reconnecting…"));
        } else if self.paused {
            head += &format!("   {}", t.c(YELLOW, "paused"));
        }
        let mut lines = vec![String::new(), head, String::new()];

        let sw = w.saturating_sub(64).max(10);
        if sessions.is_empty() {
            lines.push(format!("  {}", t.c(FAINT, "no sessions yet — start an agent and it will appear here")));
            lines.push(String::new());
        }
        for s in &sessions {
            let age = now - s.last;
            let idle = age >= IDLE_AFTER;
            let body = if idle { IDLE } else { TEXT };
            let depth = if s.cut != 0 { format!("{}/{}", s.cut, s.total) } else { format!("—/{}", s.total) };
            let count = if s.compactions != 0 { format!("{}×", s.compactions) } else { "—".into() };
            lines.push(format!(
                "  {} {}  {}{}{}{}{}{}{}",
                t.c(if idle { IDLE } else { BRAND }, if idle { "○" } else { "●" }),
                t.c(if idle { IDLE } else { DIM }, py::prefix_chars(&s.sid, 4)),
                t.c(body, &format!("{:<15}", py::prefix_chars(&s.model, 14))),
                t.c(FAINT, &format!("{:<10}", py::prefix_chars(&s.dialect, 9))),
                s.spark(t, sw, self.threshold, idle),
                t.c(if idle { IDLE } else { DIM }, &format!("{depth:>12}")),
                t.c(body, &format!("{:>7}", kfmt(s.est))),
                t.c(if s.compactions != 0 { if idle { IDLE } else { BRAND } } else { FAINT }, &format!("{count:>5}")),
                t.c(FAINT, &format!("{:>6}", agefmt(age))),
            ));
        }

        // The feed gets whatever vertical space is left.
        let room = rows.saturating_sub(lines.len() + 8).max(3);
        lines.extend([String::new(), t.rule(""), String::new()]);
        let skip = self.feed.len().saturating_sub(room);
        for ev in self.feed.iter().skip(skip) {
            lines.push(self.feed_line(t, ev));
        }
        lines.extend([String::new(), t.rule(""), String::new()]);
        lines.push(format!(
            "  {}{}{}{}{}{}{}{}{}{}{}{}",
            t.c(TEXT, &thousands(self.requests)),
            t.c(DIM, " requests"),
            t.c(FAINT, "   "),
            t.c(TEXT, &self.compactions.to_string()),
            t.c(DIM, " compactions"),
            t.c(FAINT, "   "),
            t.c(TEXT, &sessions.len().to_string()),
            t.c(DIM, " sessions"),
            if idle_n > 0 { t.c(DIM, &format!(" · {idle_n} idle")) } else { String::new() },
            t.c(FAINT, "      "),
            t.c(DIM, "watching "),
            t.c(FAINT, &agefmt(now - self.window_start)),
        ));
        lines.extend([String::new(), format!("  {}", t.c(FAINT, "p pause  ·  q quit"))]);
        lines.join("\n")
    }

    pub fn feed_line(&self, t: &Term, ev: &Value) -> String {
        let clock = chrono::DateTime::from_timestamp(event_time(ev) as i64, 0)
            .map(|d| d.with_timezone(&chrono::Local).format("%H:%M:%S").to_string())
            .unwrap_or_default();
        let sid = py::prefix_chars(text(ev, "sid"), 4);
        let est = |k| kfmt(int(ev, k));
        match text(ev, "kind") {
            "compact" => {
                let steps = match int(ev, "steps") {
                    0 => 1,
                    s => s,
                };
                let replayed = if steps > 1 { format!("replayed {steps}×   ") } else { String::new() };
                let fp = match text(ev, "fp") {
                    "" => "-",
                    f => f,
                };
                let body = format!(
                    "{:<20}{}→{} msgs   {replayed}#{fp}",
                    format!("{} → {} est", est("est_in"), est("est_out")),
                    int(ev, "total"),
                    int(ev, "out_msgs"),
                );
                format!("  {} {}  {}{}", t.c(DIM, &clock), t.c(BRAND, sid), t.bold(YELLOW, "⚡ compact "), t.c(TEXT, &body))
            }
            "match" => {
                // What the client sent, and what the substitution sends on.
                let body = format!(
                    "{:<20}{} → {} est",
                    format!("depth {}/{}", int(ev, "cut"), int(ev, "total")),
                    est("est_in"),
                    est("est_out")
                );
                format!("  {} {}  {}{}", t.c(DIM, &clock), t.c(FAINT, sid), t.c(BRAND, "· match   "), t.c(DIM, &body))
            }
            _ => {
                let body = format!("{:<20}{} est", format!("{} msgs", int(ev, "total")), est("est_in"));
                // Side calls keep their session's id — that is where the
                // cost lands — and say plainly that they are not a turn.
                let label = if truthy(ev, "aux") { "· aux     " } else { "· pass    " };
                format!("  {} {}  {}{}", t.c(DIM, &clock), t.c(FAINT, sid), t.c(FAINT, label), t.c(DIM, &body))
            }
        }
    }
}

// --- the live loop -------------------------------------------------------------------

/// Follow the daemon's event stream, reconnecting if it goes away.
async fn stream(w: Arc<Mutex<Watcher>>, port: u16) {
    let client = reqwest::Client::builder().connect_timeout(Duration::from_secs(10)).build().expect("HTTP client");
    let events_url = format!("http://127.0.0.1:{port}/__cliff__/events");
    let status_url = format!("http://127.0.0.1:{port}/__cliff__/status");
    loop {
        if w.lock().unwrap().stop {
            return;
        }
        if let Ok(resp) = client.get(&status_url).timeout(Duration::from_secs(10)).send().await {
            if let Some(st) = resp.bytes().await.ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
                let mut w = w.lock().unwrap();
                w.threshold = st.get("threshold_tokens").and_then(Value::as_i64).unwrap_or(w.threshold);
                w.keep_recent = st.get("keep_recent").and_then(Value::as_i64).unwrap_or(w.keep_recent);
                w.shadow = st.get("shadow").is_some_and(py::truthy);
                w.uptime = st.get("uptime_s").and_then(Value::as_f64).unwrap_or(0.0);
            }
        }
        let ok = match client.get(&events_url).send().await.and_then(|r| r.error_for_status()) {
            Ok(resp) => {
                w.lock().unwrap().connected = true;
                let mut body = resp.bytes_stream();
                let mut buf: Vec<u8> = vec![];
                loop {
                    match body.next().await {
                        Some(Ok(chunk)) => {
                            buf.extend_from_slice(&chunk);
                            while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                                let line: Vec<u8> = buf.drain(..=pos).collect();
                                let line = String::from_utf8_lossy(&line);
                                let line = line.trim_end_matches(['\n', '\r']);
                                if let Some(data) = line.strip_prefix("data: ") {
                                    if let Ok(ev) = serde_json::from_str::<Value>(data) {
                                        w.lock().unwrap().on_event(&ev);
                                    }
                                }
                            }
                            if w.lock().unwrap().stop {
                                return;
                            }
                        }
                        _ => break,
                    }
                }
                false
            }
            Err(_) => false,
        };
        if !ok {
            w.lock().unwrap().connected = false;
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
}

async fn draw(w: Arc<Mutex<Watcher>>, mut term: Term) {
    loop {
        {
            let w = w.lock().unwrap();
            if w.stop {
                return;
            }
            if !w.paused {
                term.refresh();
                let frame = w.render(&term);
                let mut out = std::io::stdout().lock();
                let _ = write!(out, "\x1b[H\x1b[J{frame}");
                let _ = out.flush();
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// q quits, p pauses. Only when stdin is a terminal (read on a plain
/// thread: a blocking read has no business on the runtime).
fn keys(w: Arc<Mutex<Watcher>>) {
    let mut byte = [0u8; 1];
    let mut stdin = std::io::stdin();
    while stdin.read(&mut byte).is_ok_and(|n| n == 1) {
        let mut w = w.lock().unwrap();
        match byte[0] {
            b'q' | b'Q' | 0x03 => {
                w.stop = true;
                return;
            }
            b'p' | b'P' | b' ' => w.paused = !w.paused,
            _ => {}
        }
    }
}

#[cfg(unix)]
struct Cbreak(libc::termios);

#[cfg(unix)]
impl Cbreak {
    fn enter() -> Option<Cbreak> {
        // SAFETY: plain termios calls on stdin with structs we own.
        unsafe {
            let mut old: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut old) != 0 {
                return None;
            }
            let mut new = old;
            new.c_lflag &= !(libc::ECHO | libc::ICANON);
            new.c_cc[libc::VMIN] = 1;
            new.c_cc[libc::VTIME] = 0;
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSAFLUSH, &new);
            Some(Cbreak(old))
        }
    }
}

#[cfg(unix)]
impl Drop for Cbreak {
    fn drop(&mut self) {
        // SAFETY: restores the attributes captured in enter().
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSADRAIN, &self.0);
        }
    }
}

pub fn run(port: u16) -> i32 {
    if crate::daemon::http_get_json(port, "/__cliff__/status", Duration::from_secs(3)).is_none() {
        eprintln!("error: no cliff proxy answering on port {port}. Run `cliff status` to check the daemon.");
        return 1;
    }
    let term = Term::new();
    let w = Arc::new(Mutex::new(Watcher::new(port)));
    let interactive = std::io::stdin().is_terminal();
    #[cfg(unix)]
    let _cbreak = if interactive { Cbreak::enter() } else { None };
    if interactive {
        let w = w.clone();
        std::thread::spawn(move || keys(w));
    }
    let tty = term.tty;
    if tty {
        print!("\x1b[?1049h\x1b[?25l"); // alternate screen, hide cursor
        let _ = std::io::stdout().flush();
    }
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime");
    rt.block_on(async {
        let tasks = [tokio::spawn(stream(w.clone(), port)), tokio::spawn(draw(w.clone(), term))];
        loop {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                _ = tokio::signal::ctrl_c() => { w.lock().unwrap().stop = true; }
            }
            if w.lock().unwrap().stop {
                break;
            }
        }
        for t in tasks {
            t.abort();
        }
    });
    if tty {
        print!("\x1b[?25h\x1b[?1049l");
        let _ = std::io::stdout().flush();
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_model_like_python() {
        assert_eq!(short_model("claude-sonnet-4-5-20250929"), "sonnet-4-5");
        assert_eq!(short_model("claude-haiku-4-5"), "haiku-4-5");
        assert_eq!(short_model("k3"), "k3");
        assert_eq!(short_model(""), "?");
    }

    #[test]
    fn formatters_like_python() {
        assert_eq!(kfmt(999), "999");
        assert_eq!(kfmt(12_400), "12k");
        assert_eq!(kfmt(12_500), "12k"); // half to even, as Python's format
        assert_eq!(kfmt(13_500), "14k");
        assert_eq!(kfmt(2_500_000), "2.5M");
        assert_eq!(agefmt(9.0), "9s");
        assert_eq!(agefmt(300.0), "5m");
        assert_eq!(agefmt(7300.0), "2h");
        assert_eq!(thousands(1234567), "1,234,567");
    }

    #[test]
    fn events_without_the_aux_field_still_make_rows() {
        let mut w = Watcher::new(0);
        w.on_event(&serde_json::json!({"sid": "abcd1234", "kind": "pass", "total": 2, "est_in": 100}));
        assert_eq!(w.live_sessions().len(), 1);
    }
}
