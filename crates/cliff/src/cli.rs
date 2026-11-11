//! cliff CLI.
//!
//! ```text
//! cliff enable [options]                     # daily driver: supervised daemon + shell env wiring
//! cliff enable --name X --port N [options]   # an additional daemon (own port/flags/log), no env wiring
//! cliff disable [--name X]                   # remove the daemon (and env wiring, for the default)
//! cliff restart [--name X]                   # pick up an upgrade; keeps the daemon's flags
//! cliff status [--name X]                    # daemon / env / store health
//! cliff watch [--name X]                     # live view of sessions through the proxy
//! cliff serve [--shadow] [--port N] ...      # run the proxy in the foreground
//! cliff run [--shadow] [options] -- CMD ...  # wrap one command: proxy up, env set, run, tear down
//! ```

use std::path::PathBuf;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use cliff_core::Config;
use log::{error, info};
use serde_json::Value;

use crate::ui::{banner, Term, BRAND, DIM, FAINT, TEXT, YELLOW};
use crate::{daemon, logging, proxy};

const ABOUT: &str = "CliffCompaction — autocompaction for coding agents, as a transparent API proxy";

fn py_int(s: &str) -> Result<i64, String> {
    cliff_core::py::parse_int(s).ok_or_else(|| format!("invalid int value: '{s}'"))
}

fn py_port(s: &str) -> Result<u16, String> {
    u16::try_from(py_int(s)?).map_err(|_| format!("invalid port: '{s}'"))
}

#[derive(Args, Clone, Debug, Default)]
pub struct Common {
    /// observe and log, never modify a request
    #[arg(long)]
    pub shadow: bool,
    /// fail a request that is still over --threshold after the escalation ladder instead of sending it anyway (for measurement runs)
    #[arg(long)]
    pub strict: bool,
    /// proactive compaction threshold in est. tokens (default 200000)
    #[arg(long, value_parser = py_int)]
    pub threshold: Option<i64>,
    /// recent turns kept verbatim (default 3)
    #[arg(long, value_parser = py_int)]
    pub keep_recent: Option<i64>,
    /// cap on assistant text per summarized turn; 0 = unlimited (default)
    #[arg(long, value_parser = py_int)]
    pub thought_max_chars: Option<i64>,
    /// tool results longer than this are dropped (default 500)
    #[arg(long, value_parser = py_int)]
    pub result_max_chars: Option<i64>,
    /// exclude thinking/reasoning text from summaries (leaner configuration)
    #[arg(long)]
    pub drop_thinking: bool,
    /// cap on thinking text per summarized turn; 0 = unlimited (default), independent of --thought-max-chars
    #[arg(long, value_parser = py_int)]
    pub thinking_max_chars: Option<i64>,
    /// Anthropic upstream base URL
    #[arg(long)]
    pub anthropic_upstream: Option<String>,
    /// OpenAI-compatible upstream base URL
    #[arg(long)]
    pub openai_upstream: Option<String>,
    /// dump each handled request's incoming/outgoing message arrays as JSON files here
    #[arg(long)]
    pub debug_dir: Option<String>,
    #[arg(short, long)]
    pub verbose: bool,
}

fn nonempty(v: &Option<String>) -> Option<&String> {
    v.as_ref().filter(|s| !s.is_empty())
}

impl Common {
    pub fn config(&self) -> Config {
        let mut cfg = Config::from_env();
        cfg.shadow |= self.shadow;
        cfg.strict |= self.strict;
        cfg.threshold_tokens = self.threshold.unwrap_or(cfg.threshold_tokens);
        cfg.keep_recent = self.keep_recent.unwrap_or(cfg.keep_recent);
        cfg.thought_max_chars = self.thought_max_chars.unwrap_or(cfg.thought_max_chars);
        cfg.result_max_chars = self.result_max_chars.unwrap_or(cfg.result_max_chars);
        if self.drop_thinking {
            cfg.keep_thinking = false;
        }
        cfg.thinking_max_chars = self.thinking_max_chars.unwrap_or(cfg.thinking_max_chars);
        if let Some(u) = nonempty(&self.anthropic_upstream) {
            cfg.anthropic_upstream = u.clone();
        }
        if let Some(u) = nonempty(&self.openai_upstream) {
            cfg.openai_upstream = u.clone();
        }
        if let Some(d) = nonempty(&self.debug_dir) {
            cfg.debug_dir = d.clone();
        }
        cfg
    }

    /// Forward explicitly-set flags to the daemon's serve command.
    pub fn serve_args(&self) -> Vec<String> {
        let mut out = vec![];
        let mut flag = |set: bool, f: &str| {
            if set {
                out.push(f.to_string());
            }
        };
        flag(self.shadow, "--shadow");
        flag(self.strict, "--strict");
        let mut opt = |v: Option<String>, f: &str| {
            if let Some(v) = v {
                out.push(f.to_string());
                out.push(v);
            }
        };
        opt(self.threshold.map(|v| v.to_string()), "--threshold");
        opt(self.keep_recent.map(|v| v.to_string()), "--keep-recent");
        opt(self.thought_max_chars.map(|v| v.to_string()), "--thought-max-chars");
        opt(self.result_max_chars.map(|v| v.to_string()), "--result-max-chars");
        if self.drop_thinking {
            out.push("--drop-thinking".into());
        }
        let mut opt = |v: Option<String>, f: &str| {
            if let Some(v) = v {
                out.push(f.to_string());
                out.push(v);
            }
        };
        opt(self.thinking_max_chars.map(|v| v.to_string()), "--thinking-max-chars");
        opt(nonempty(&self.anthropic_upstream).cloned(), "--anthropic-upstream");
        opt(nonempty(&self.openai_upstream).cloned(), "--openai-upstream");
        opt(nonempty(&self.debug_dir).cloned(), "--debug-dir");
        if self.verbose {
            out.push("-v".into());
        }
        out
    }
}

/// instance name (default: the unnamed daemon that owns the shell env)
#[derive(Args, Clone, Debug, Default)]
pub struct Instance {
    #[arg(long, value_parser = py_port)]
    pub port: Option<u16>,
    /// instance name (default: the unnamed daemon that owns the shell env)
    #[arg(long)]
    pub name: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// run the proxy in the foreground
    Serve {
        #[arg(long, value_parser = py_port)]
        port: Option<u16>,
        #[arg(long)]
        host: Option<String>,
        #[command(flatten)]
        common: Common,
    },
    /// wrap a command behind the proxy
    Run {
        #[command(flatten)]
        common: Common,
    },
    /// install the supervised daemon + shell env wiring
    Enable {
        #[arg(long, value_parser = py_port)]
        port: Option<u16>,
        /// install an additional named daemon alongside the default: own --port (required), flags and log; no shell env wiring — point your client at its port
        #[arg(long)]
        name: Option<String>,
        /// shell profile file to wire (default: auto-detect)
        #[arg(long)]
        profile: Option<String>,
        /// install the daemon but skip shell env wiring
        #[arg(long)]
        no_env: bool,
        #[command(flatten)]
        common: Common,
    },
    /// remove the daemon and env wiring
    Disable {
        /// instance name (default: the unnamed daemon that owns the shell env)
        #[arg(long)]
        name: Option<String>,
        /// shell profile file to unwire (default: auto-detect)
        #[arg(long)]
        profile: Option<String>,
    },
    /// daemon / env / store health
    Status {
        #[command(flatten)]
        instance: Instance,
    },
    /// restart the daemon in place (pick up an upgrade); flags unchanged
    Restart {
        #[command(flatten)]
        instance: Instance,
    },
    /// live view of sessions through the proxy
    Watch {
        #[command(flatten)]
        instance: Instance,
    },
}

#[derive(Parser, Debug)]
#[command(name = "cliffcompaction", bin_name = "cliff", version, about = ABOUT)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

fn mode(shadow: bool, strict: bool) -> &'static str {
    if shadow {
        "shadow"
    } else if strict {
        "strict"
    } else {
        "active"
    }
}

fn status_mode(status: &Value) -> &'static str {
    let flag = |k| status.get(k).is_some_and(cliff_core::py::truthy);
    mode(flag("shadow"), flag("strict"))
}

fn with_name(cmd: &str, name: Option<&str>) -> String {
    name.map_or(cmd.to_string(), |n| format!("{cmd} --name {n}"))
}

/// Validated --name, or the exit code after printing the error.
fn instance(name: &Option<String>) -> Result<Option<String>, i32> {
    daemon::validate_name(name.as_deref()).map_err(|e| {
        eprintln!("error: {e}");
        1
    })
}

/// --port if given, else the port baked into the installed service, else the default.
fn resolve_port(port: Option<u16>, name: Option<&str>) -> u16 {
    port.or_else(|| daemon::installed_port(name)).unwrap_or_else(|| Config::from_env().port)
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime")
}

pub fn cmd_serve(port: Option<u16>, host: Option<String>, common: &Common) -> i32 {
    let mut cfg = common.config();
    cfg.port = port.unwrap_or(cfg.port);
    if let Some(h) = host.filter(|h| !h.is_empty()) {
        cfg.host = h;
    }
    info!(
        "cliffcompaction {} serving on http://{}:{} ({} mode, threshold ~{}k tokens, keep_recent={})",
        cliff_core::VERSION,
        cfg.host,
        cfg.port,
        mode(cfg.shadow, cfg.strict),
        cfg.threshold_tokens / 1000,
        cfg.keep_recent
    );
    runtime().block_on(async move {
        let listener = match tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port)).await {
            Ok(l) => l,
            Err(e) => {
                // uvicorn's STARTUP_FAILURE: `cliff status` reads exit 3 as
                // "port already in use?".
                error!("could not bind {}:{}: {e}", cfg.host, cfg.port);
                return 3;
            }
        };
        let (router, _) = proxy::create_app(cfg, None);
        match proxy::serve(listener, router, proxy::shutdown_signal()).await {
            Ok(()) => 0,
            Err(e) => {
                error!("server error: {e}");
                1
            }
        }
    })
}

pub fn cmd_run(common: &Common, command: &[String]) -> i32 {
    if command.is_empty() {
        eprintln!("usage: cliff run [options] -- COMMAND [ARGS...]");
        return 2;
    }
    let mut cfg = common.config();
    let listener = match std::net::TcpListener::bind((cfg.host.as_str(), 0)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: could not bind a local port: {e}");
            return 1;
        }
    };
    cfg.port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    let base = format!("http://{}:{}", cfg.host, cfg.port);
    let proxy_mode = mode(cfg.shadow, cfg.strict);
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = std::thread::spawn(move || {
        runtime().block_on(async move {
            let _ = listener.set_nonblocking(true);
            let Ok(listener) = tokio::net::TcpListener::from_std(listener) else { return };
            let (router, _) = proxy::create_app(cfg, None);
            let _ = proxy::serve(listener, router, async {
                let _ = stop_rx.await;
            })
            .await;
        });
    });
    #[cfg(unix)]
    let _guard = IgnoreSigint::install();
    info!("proxy on {base} ({proxy_mode} mode); running: {}", command.join(" "));
    let status = std::process::Command::new(&command[0])
        .args(&command[1..])
        .env("ANTHROPIC_BASE_URL", &base)
        .env("OPENAI_BASE_URL", format!("{base}/v1"))
        .env("OPENAI_API_BASE", format!("{base}/v1")) // legacy SDKs
        .status();
    let _ = stop_tx.send(());
    let _ = server.join();
    match status {
        Ok(s) => exit_code(s),
        Err(e) => {
            eprintln!("error: could not run {}: {e}", command[0]);
            1
        }
    }
}

fn exit_code(s: std::process::ExitStatus) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = s.signal() {
            return 128 + sig;
        }
    }
    s.code().unwrap_or(1)
}

/// While the wrapped command runs, SIGINT goes to it (same process group)
/// and is ignored here. A no-op handler rather than SIG_IGN: handlers reset
/// to the default across exec, so the child still gets a normal Ctrl-C.
#[cfg(unix)]
struct IgnoreSigint(libc::sigaction);

#[cfg(unix)]
impl IgnoreSigint {
    fn install() -> IgnoreSigint {
        extern "C" fn noop(_: libc::c_int) {}
        // SAFETY: installs an async-signal-safe no-op handler; the previous
        // action is restored on drop.
        unsafe {
            let mut new: libc::sigaction = std::mem::zeroed();
            new.sa_sigaction = noop as *const () as libc::sighandler_t;
            libc::sigemptyset(&mut new.sa_mask);
            let mut old: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGINT, &new, &mut old);
            IgnoreSigint(old)
        }
    }
}

#[cfg(unix)]
impl Drop for IgnoreSigint {
    fn drop(&mut self) {
        // SAFETY: restores the action saved in install().
        unsafe {
            libc::sigaction(libc::SIGINT, &self.0, std::ptr::null_mut());
        }
    }
}

fn pad(text: &str, width: usize) -> String {
    // Left-align in a column, always leaving a gap when the text overflows.
    if text.chars().count() < width { format!("{text:<width$}") } else { format!("{text}  ") }
}

fn short_path(path: &std::path::Path) -> String {
    let text = path.display().to_string();
    let home = std::env::var("HOME").unwrap_or_default();
    match text.strip_prefix(&home) {
        Some(rest) if !home.is_empty() => format!("~{rest}"),
        _ => text,
    }
}

fn print_enable_screen(status: &Value, port: u16, profile: &std::path::Path, wired: bool, name: Option<&str>) {
    let term = Term::new();
    let row = |mark: &str, key: &str, val: &str, hint: &str, mark_rgb| {
        let body = if hint.is_empty() { term.c(TEXT, val) } else { term.c(TEXT, &pad(val, 34)) + &term.c(FAINT, hint) };
        format!("  {} {}{body}", term.c(mark_rgb, mark), term.c(DIM, &format!("{key:<9}")))
    };
    let ok = if term.unicode { "✓" } else { "+" };
    let dot = if term.unicode { "·" } else { "-" };
    let threshold = format!("{}k", status.get("threshold_tokens").and_then(Value::as_i64).unwrap_or(0) / 1000);
    let keep = status.get("keep_recent").map(Value::to_string).unwrap_or_default();
    let mut lines = vec![row(ok, "daemon", &format!("supervised on 127.0.0.1:{port}"), "auto-restarts, survives reboots", BRAND)];
    if let Some(n) = name {
        lines.push(row(ok, "instance", n, "alongside the default daemon", BRAND));
    }
    if wired {
        lines.push(row(ok, "env", &format!("wired in {}", short_path(profile)), "ANTHROPIC_BASE_URL, OPENAI_BASE_URL", BRAND));
    } else if name.is_some() {
        lines.push(row("!", "env", "not wired (named instance)", &format!("point your client at http://127.0.0.1:{port}"), YELLOW));
    } else {
        lines.push(row("!", "env", "not wired (--no-env)", "set ANTHROPIC_BASE_URL/OPENAI_BASE_URL yourself", YELLOW));
    }
    lines.push(row(ok, "config", &format!("threshold {threshold} {dot} keep {keep} {dot} {}", status_mode(status)), "", BRAND));
    lines.push(row(" ", "logs", &short_path(&daemon::log_path(name)), "", BRAND));

    let mut steps = vec![
        ("open a new terminal".to_string(), "agents pick up the env there".to_string()),
        ("run your agent as usual".to_string(), "no flags, no integration".to_string()),
        (with_name("cliff status", name), format!("check on it {dot} {} turns it off", with_name("cliff disable", name))),
    ];
    if name.is_some() {
        steps[0] = (format!("point your client at http://127.0.0.1:{port}"), "e.g. a Codex model provider base_url".into());
        steps[1] = ("run it as usual".into(), "the default daemon keeps serving the env".into());
    } else if !wired {
        steps[0] = (format!("export ANTHROPIC_BASE_URL=http://127.0.0.1:{port}"), "and OPENAI_BASE_URL".into());
    }
    lines.extend([String::new(), term.rule("next"), String::new()]);
    for (i, (text, hint)) in steps.iter().enumerate() {
        lines.push(format!("  {}  {}{}", term.c(BRAND, &(i + 1).to_string()), term.c(TEXT, &pad(text, 31)), term.c(FAINT, hint)));
    }
    let mut all = banner(&term, "  ");
    all.extend(lines);
    all.push(String::new());
    term.out(&all);
}

pub fn cmd_enable(port: Option<u16>, name: &Option<String>, profile: &Option<String>, no_env: bool, common: &Common) -> i32 {
    let name = match instance(name) {
        Ok(n) => n,
        Err(code) => return code,
    };
    let name = name.as_deref();
    if let (Some(n), None) = (name, port) {
        eprintln!("error: a named instance needs its own port: cliff enable --name {n} --port N ...");
        return 1;
    }
    let port = port.unwrap_or_else(|| Config::from_env().port);
    if !daemon::service_running(name).0 && daemon::port_in_use(port) {
        eprintln!(
            "error: port {port} is already in use (a manual `cliff serve`, or another cliff instance?). Stop it first, or pass --port for a different one."
        );
        return 1;
    }
    if let Err(e) = daemon::start_service(port, &common.serve_args(), name) {
        eprintln!("error: {e}");
        return 1;
    }
    let Some(status) = daemon::wait_healthy(port, Duration::from_secs(8)) else {
        eprintln!("error: daemon installed but not responding on port {port}; check {}", daemon::log_path(name).display());
        return 1;
    };
    let profile = profile.as_ref().map_or_else(daemon::default_profile, PathBuf::from);
    // Only one daemon can own ANTHROPIC_BASE_URL/OPENAI_BASE_URL: the default.
    let wired = name.is_none() && !no_env;
    if wired {
        if let Err(e) = daemon::wire_profile(&profile, port) {
            eprintln!("error: could not wire {}: {e}", profile.display());
            return 1;
        }
    }
    print_enable_screen(&status, port, &profile, wired, name);
    0
}

pub fn cmd_disable(name: &Option<String>, profile: &Option<String>) -> i32 {
    let name = match instance(name) {
        Ok(n) => n,
        Err(code) => return code,
    };
    daemon::stop_service(name.as_deref());
    if let Some(n) = &name {
        println!("cliffcompaction instance '{n}' disabled: daemon removed");
        return 0;
    }
    let profile = profile.as_ref().map_or_else(daemon::default_profile, PathBuf::from);
    let _ = daemon::unwire_profile(&profile);
    println!("cliffcompaction disabled: daemon removed, env wiring removed");
    println!("  open a new terminal for the env change to take effect");
    let others: Vec<String> = daemon::installed_names().into_iter().flatten().collect();
    if !others.is_empty() {
        println!("  named instances still installed: {} (cliff disable --name X)", others.join(", "));
    }
    0
}

pub fn cmd_status(instance_args: &Instance) -> i32 {
    let name = match instance(&instance_args.name) {
        Ok(n) => n,
        Err(code) => return code,
    };
    let name = name.as_deref();
    let port = resolve_port(instance_args.port, name);
    let installed = daemon::service_installed(name);
    let (running, last_exit) = daemon::service_running(name);
    let status = daemon::probe(port);
    if let Some(n) = name {
        println!("instance         : {n}");
    }
    println!("daemon installed : {}", if installed { "yes" } else { "no" });
    if installed {
        let mut detail = if running { "running".to_string() } else { "NOT running".to_string() };
        if let (false, Some(code)) = (running, last_exit.filter(|c| *c != 0)) {
            detail += &format!(" (last exit {code}{}", if code == 3 { "; port already in use?)" } else { ")" });
        }
        println!("daemon service   : {detail}");
    }
    match &status {
        Some(st) => {
            let int = |k| st.get(k).and_then(Value::as_i64).unwrap_or(0);
            let implementation = st.get("implementation").and_then(Value::as_str).unwrap_or("python");
            println!(
                "proxy responding : yes on port {port} ({} mode, threshold ~{}k tokens, keep_recent={}, store entries={} ({:.1}MB), {implementation})",
                status_mode(st),
                int("threshold_tokens") / 1000,
                st.get("keep_recent").map(Value::to_string).unwrap_or_default(),
                st.get("store_entries").map(Value::to_string).unwrap_or_default(),
                int("store_bytes") as f64 / 1e6,
            );
        }
        None => println!("proxy responding : no (port {port})"),
    }
    if name.is_none() {
        let profile = daemon::default_profile();
        println!(
            "env wired        : {} ({})",
            if daemon::profile_is_wired(&profile) { "yes" } else { "no" },
            profile.display()
        );
        let others: Vec<String> = daemon::installed_names().into_iter().flatten().collect();
        if !others.is_empty() {
            println!("named instances  : {} (cliff status --name X)", others.join(", "));
        }
    }
    if installed && status.is_none() {
        println!("hint: check the log at {}", daemon::log_path(name).display());
    }
    if installed && !running && status.is_some() {
        println!(
            "warning: port {port} is answered by another proxy, not the daemon (a manual `cliff serve` still running?); stop it and re-run `{}`",
            with_name("cliff enable", name)
        );
    }
    if let Some(st) = &status {
        // Missing field rather than a false one: the daemon predates the check.
        if st.get("code_stale").is_none_or(cliff_core::py::truthy) {
            println!(
                "warning: cliffcompaction was installed or changed after the daemon started, so the daemon is still serving the older code; run `{}`",
                with_name("cliff restart", name)
            );
        }
    }
    0
}

pub fn cmd_restart(instance_args: &Instance) -> i32 {
    let name = match instance(&instance_args.name) {
        Ok(n) => n,
        Err(code) => return code,
    };
    let name = name.as_deref();
    if !daemon::service_installed(name) {
        eprintln!("error: no daemon installed (run `{}` first)", with_name("cliff enable", name));
        return 1;
    }
    let port = resolve_port(instance_args.port, name);
    if let Err(e) = daemon::restart_service(name) {
        eprintln!("error: {e}");
        return 1;
    }
    let Some(status) = daemon::wait_healthy(port, Duration::from_secs(8)) else {
        eprintln!("error: daemon did not come back on port {port}; check the log at {}", daemon::log_path(name).display());
        return 1;
    };
    let label = name.map_or("cliff".to_string(), |n| format!("cliff[{n}]"));
    println!(
        "{label} restarted on port {port} ({} mode, threshold ~{}k tokens, keep_recent={})",
        status_mode(&status),
        status.get("threshold_tokens").and_then(Value::as_i64).unwrap_or(0) / 1000,
        status.get("keep_recent").map(Value::to_string).unwrap_or_default()
    );
    0
}

pub fn cmd_watch(instance_args: &Instance) -> i32 {
    match instance(&instance_args.name) {
        Ok(name) => crate::watch::run(resolve_port(instance_args.port, name.as_deref())),
        Err(code) => code,
    }
}

fn verbose(cmd: &Cmd) -> bool {
    match cmd {
        Cmd::Serve { common, .. } | Cmd::Run { common } | Cmd::Enable { common, .. } => common.verbose,
        _ => false,
    }
}

/// Entry point; returns the process exit code.
pub fn main(args: Vec<String>) -> i32 {
    // Split off the wrapped command for `run` before the parser sees it.
    let (argv, command) = match args.iter().position(|a| a == "--") {
        Some(i) => (args[..i].to_vec(), args[i + 1..].to_vec()),
        None => (args, vec![]),
    };
    let cli = match Cli::try_parse_from(std::iter::once("cliff".to_string()).chain(argv)) {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return e.exit_code();
        }
    };
    logging::setup(verbose(&cli.cmd));
    match &cli.cmd {
        Cmd::Serve { port, host, common } => cmd_serve(*port, host.clone(), common),
        Cmd::Run { common } => cmd_run(common, &command),
        Cmd::Enable { port, name, profile, no_env, common } => cmd_enable(*port, name, profile, *no_env, common),
        Cmd::Disable { name, profile } => cmd_disable(name, profile),
        Cmd::Status { instance } => cmd_status(instance),
        Cmd::Restart { instance } => cmd_restart(instance),
        Cmd::Watch { instance } => cmd_watch(instance),
    }
}
