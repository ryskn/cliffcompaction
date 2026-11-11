//! cliff enable / disable / status: user-service install (launchd /
//! systemd --user) plus ANTHROPIC_BASE_URL / OPENAI_BASE_URL wiring in the
//! shell profile. The service runs `<this binary> serve ...` by absolute
//! path, so PATH does not matter.
//!
//! Labels, unit names, log paths and profile markers are byte-identical to
//! the Python implementation's: switching between the two is `cliff enable`
//! from either, and the other's service definition is simply replaced.
//!
//! Instances: the default (unnamed) daemon owns the shell env wiring; named
//! instances get their own port, flags, service label and log.
//! `name = None` everywhere means the default.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::Value;

pub const LABEL: &str = "com.cliffcompaction.proxy";
pub const SYSTEMD_UNIT: &str = "cliffcompaction.service";
pub const MARK_BEGIN: &str = "# >>> cliffcompaction >>>";
pub const MARK_END: &str = "# <<< cliffcompaction <<<";

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

// --- instances ------------------------------------------------------------------

/// The name unchanged, or an error: it becomes part of a launchd label, a
/// systemd unit name and a log file name, so keep it to a safe charset.
pub fn validate_name(name: Option<&str>) -> Result<Option<String>, String> {
    let Some(name) = name.filter(|n| !n.is_empty()) else { return Ok(None) };
    let mut chars = name.chars();
    let first_ok = chars.next().is_some_and(|c| c.is_ascii_alphanumeric());
    let rest_ok = chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !first_ok || !rest_ok || name.len() > 32 {
        return Err(format!("invalid instance name '{name}': use letters, digits, '-' or '_' (max 32 chars)"));
    }
    Ok(Some(name.to_string()))
}

pub fn label(name: Option<&str>) -> String {
    name.map_or(LABEL.to_string(), |n| format!("{LABEL}.{n}"))
}

pub fn systemd_unit(name: Option<&str>) -> String {
    name.map_or(SYSTEMD_UNIT.to_string(), |n| format!("cliffcompaction-{n}.service"))
}

fn sorted_dir(folder: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(folder)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    names.sort();
    names
}

/// Every installed instance, default first (as None), then named ones sorted.
pub fn installed_names() -> Vec<Option<String>> {
    let mut names = vec![];
    if cfg!(target_os = "macos") {
        if plist_path(None).exists() {
            names.push(None);
        }
        let prefix = format!("{LABEL}.");
        for f in sorted_dir(&home().join("Library/LaunchAgents")) {
            if let Some(n) = f.strip_prefix(&prefix).and_then(|r| r.strip_suffix(".plist")) {
                names.push(Some(n.to_string()));
            }
        }
    } else if cfg!(target_os = "linux") {
        if systemd_unit_path(None).exists() {
            names.push(None);
        }
        for f in sorted_dir(&home().join(".config/systemd/user")) {
            if let Some(n) = f.strip_prefix("cliffcompaction-").and_then(|r| r.strip_suffix(".service")) {
                names.push(Some(n.to_string()));
            }
        }
    }
    names
}

/// The --port baked into the installed service definition, if any.
pub fn installed_port(name: Option<&str>) -> Option<u16> {
    let argv: Vec<String> = if cfg!(target_os = "macos") {
        let text = std::fs::read_to_string(plist_path(name)).ok()?;
        plist_program_arguments(&text)
    } else if cfg!(target_os = "linux") {
        let text = std::fs::read_to_string(systemd_unit_path(name)).ok()?;
        let line = text.lines().filter(|l| l.starts_with("ExecStart=")).last()?;
        line["ExecStart=".len()..].split_whitespace().map(str::to_string).collect()
    } else {
        return None;
    };
    let i = argv.iter().position(|t| t == "--port")?;
    cliff_core::py::parse_int(argv.get(i + 1)?).and_then(|p| u16::try_from(p).ok())
}

// --- shell profile wiring -----------------------------------------------------

pub fn default_profile() -> PathBuf {
    let shell = std::env::var("SHELL").unwrap_or_default();
    let shell = Path::new(&shell).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    match shell.as_str() {
        "fish" => home().join(".config/fish/conf.d/cliffcompaction.fish"),
        "bash" => home().join(".bashrc"),
        _ => home().join(".zshrc"),
    }
}

pub fn env_block(port: u16, fish: bool) -> String {
    let base = format!("http://127.0.0.1:{port}");
    let lines = if fish {
        [
            format!("set -gx ANTHROPIC_BASE_URL \"{base}\""),
            format!("set -gx OPENAI_BASE_URL \"{base}/v1\""),
            format!("set -gx OPENAI_API_BASE \"{base}/v1\""),
        ]
    } else {
        [
            format!("export ANTHROPIC_BASE_URL=\"{base}\""),
            format!("export OPENAI_BASE_URL=\"{base}/v1\""),
            format!("export OPENAI_API_BASE=\"{base}/v1\""),
        ]
    };
    format!("{MARK_BEGIN}\n{}\n{MARK_END}\n", lines.join("\n"))
}

fn strip_block(text: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for line in text.split_inclusive('\n') {
        match line.trim() {
            t if t == MARK_BEGIN => inside = true,
            t if t == MARK_END => inside = false,
            _ if !inside => out.push_str(line),
            _ => {}
        }
    }
    out
}

fn is_fish(profile: &Path) -> bool {
    profile.extension().is_some_and(|e| e == "fish")
}

/// Add (or refresh) the env block. Idempotent.
pub fn wire_profile(profile: &Path, port: u16) -> std::io::Result<()> {
    if let Some(parent) = profile.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if is_fish(profile) {
        // fish conf.d: the whole file is ours.
        return std::fs::write(profile, env_block(port, true));
    }
    let mut text = strip_block(&std::fs::read_to_string(profile).unwrap_or_default());
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    std::fs::write(profile, text + &env_block(port, false))
}

pub fn unwire_profile(profile: &Path) -> std::io::Result<()> {
    if !profile.exists() {
        return Ok(());
    }
    if is_fish(profile) {
        return std::fs::remove_file(profile);
    }
    let text = std::fs::read_to_string(profile)?;
    std::fs::write(profile, strip_block(&text))
}

pub fn profile_is_wired(profile: &Path) -> bool {
    std::fs::read_to_string(profile).is_ok_and(|t| t.contains(MARK_BEGIN))
}

// --- service definitions --------------------------------------------------------

pub fn exe() -> String {
    std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_else(|_| "cliff".into())
}

pub fn serve_argv(port: u16, serve_args: &[String]) -> Vec<String> {
    let mut argv = vec![exe(), "serve".into(), "--port".into(), port.to_string()];
    argv.extend_from_slice(serve_args);
    argv
}

pub fn log_path(name: Option<&str>) -> PathBuf {
    let suffix = name.map_or(String::new(), |n| format!("-{n}"));
    if cfg!(target_os = "macos") {
        home().join(format!("Library/Logs/cliffcompaction{suffix}.log"))
    } else {
        home().join(format!(".local/state/cliffcompaction/proxy{suffix}.log"))
    }
}

pub fn plist_path(name: Option<&str>) -> PathBuf {
    home().join("Library/LaunchAgents").join(format!("{}.plist", label(name)))
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'").replace("&amp;", "&")
}

/// The plist `plistlib.dumps` writes for this dict (keys sorted).
pub fn plist_content(port: u16, serve_args: &[String], name: Option<&str>) -> String {
    let log = xml_escape(&log_path(name).display().to_string());
    let args: String =
        serve_argv(port, serve_args).iter().map(|a| format!("\t\t<string>{}</string>\n", xml_escape(a))).collect();
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
<plist version=\"1.0\">\n<dict>\n\
\t<key>KeepAlive</key>\n\t<true/>\n\
\t<key>Label</key>\n\t<string>{}</string>\n\
\t<key>ProgramArguments</key>\n\t<array>\n{args}\t</array>\n\
\t<key>RunAtLoad</key>\n\t<true/>\n\
\t<key>StandardErrorPath</key>\n\t<string>{log}</string>\n\
\t<key>StandardOutPath</key>\n\t<string>{log}</string>\n\
</dict>\n</plist>\n",
        xml_escape(&label(name))
    )
}

/// `ProgramArguments` out of a plist written by either implementation.
pub fn plist_program_arguments(text: &str) -> Vec<String> {
    let Some(start) = text.find("<key>ProgramArguments</key>") else { return vec![] };
    let rest = &text[start..];
    let Some(end) = rest.find("</array>") else { return vec![] };
    rest[..end]
        .split("<string>")
        .skip(1)
        .filter_map(|s| s.split_once("</string>").map(|(v, _)| xml_unescape(v)))
        .collect()
}

pub fn systemd_unit_path(name: Option<&str>) -> PathBuf {
    home().join(".config/systemd/user").join(systemd_unit(name))
}

pub fn systemd_unit_content(port: u16, serve_args: &[String], name: Option<&str>) -> String {
    let exec_start = serve_argv(port, serve_args).join(" ");
    let desc = name.map_or("CliffCompaction proxy".to_string(), |n| format!("CliffCompaction proxy ({n})"));
    format!(
        "[Unit]\nDescription={desc}\n\n[Service]\nExecStart={exec_start}\nRestart=always\nRestartSec=1\n\n[Install]\nWantedBy=default.target\n"
    )
}

// --- service control --------------------------------------------------------------

/// (exit code, stdout, stderr) of a service-manager command.
pub fn run(cmd: &[&str]) -> (i32, String, String) {
    match Command::new(cmd[0]).args(&cmd[1..]).output() {
        Ok(o) => (
            o.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&o.stdout).into_owned(),
            String::from_utf8_lossy(&o.stderr).into_owned(),
        ),
        Err(e) => (-1, String::new(), e.to_string()),
    }
}

fn gui_domain() -> String {
    // SAFETY: getuid has no preconditions.
    format!("gui/{}", unsafe { libc::getuid() })
}

pub fn start_service(port: u16, serve_args: &[String], name: Option<&str>) -> Result<(), String> {
    let lbl = label(name);
    if cfg!(target_os = "macos") {
        let path = plist_path(name);
        let mkdirs = || -> std::io::Result<()> {
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::create_dir_all(log_path(name).parent().unwrap())?;
            Ok(())
        };
        mkdirs().map_err(|e| e.to_string())?;
        let target = format!("{}/{lbl}", gui_domain());
        let p = path.display().to_string();
        run(&["launchctl", "bootout", &target]);
        std::fs::write(&path, plist_content(port, serve_args, name)).map_err(|e| e.to_string())?;
        // bootout is asynchronous: launchd may still hold the label when the
        // bootstrap arrives ("Bootstrap failed: 5"). Retry briefly.
        let mut res = run(&["launchctl", "bootstrap", &gui_domain(), &p]);
        for _ in 0..10 {
            if res.0 == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(300));
            run(&["launchctl", "bootout", &target]);
            res = run(&["launchctl", "bootstrap", &gui_domain(), &p]);
        }
        if res.0 != 0 {
            return Err(format!("launchctl bootstrap failed: {}", res.2.trim()));
        }
        Ok(())
    } else if cfg!(target_os = "linux") {
        let path = systemd_unit_path(name);
        std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
        std::fs::write(&path, systemd_unit_content(port, serve_args, name)).map_err(|e| e.to_string())?;
        let unit = systemd_unit(name);
        for cmd in [
            vec!["systemctl", "--user", "daemon-reload"],
            vec!["systemctl", "--user", "enable", "--now", &unit],
            vec!["systemctl", "--user", "restart", &unit],
        ] {
            let res = run(&cmd);
            if res.0 != 0 {
                return Err(format!("{} failed: {}", cmd.join(" "), res.2.trim()));
            }
        }
        Ok(())
    } else {
        Err(format!(
            "unsupported platform for `cliff enable`: {} (use `cliff serve` or `cliff run` instead)",
            std::env::consts::OS
        ))
    }
}

/// Restart the installed service in place, leaving its definition (and the
/// flags baked into it) alone.
pub fn restart_service(name: Option<&str>) -> Result<(), String> {
    if cfg!(target_os = "macos") {
        let res = run(&["launchctl", "kickstart", "-k", &format!("{}/{}", gui_domain(), label(name))]);
        if res.0 != 0 {
            return Err(format!("launchctl kickstart failed: {}", res.2.trim()));
        }
        Ok(())
    } else if cfg!(target_os = "linux") {
        let res = run(&["systemctl", "--user", "restart", &systemd_unit(name)]);
        if res.0 != 0 {
            return Err(format!("systemctl restart failed: {}", res.2.trim()));
        }
        Ok(())
    } else {
        Err(format!("unsupported platform: {}", std::env::consts::OS))
    }
}

pub fn stop_service(name: Option<&str>) {
    if cfg!(target_os = "macos") {
        run(&["launchctl", "bootout", &format!("{}/{}", gui_domain(), label(name))]);
        let _ = std::fs::remove_file(plist_path(name));
    } else if cfg!(target_os = "linux") {
        run(&["systemctl", "--user", "disable", "--now", &systemd_unit(name)]);
        let _ = std::fs::remove_file(systemd_unit_path(name));
        run(&["systemctl", "--user", "daemon-reload"]);
    }
}

pub fn service_installed(name: Option<&str>) -> bool {
    if cfg!(target_os = "macos") {
        plist_path(name).exists()
    } else if cfg!(target_os = "linux") {
        systemd_unit_path(name).exists()
    } else {
        false
    }
}

/// (running, last_exit_status) for the installed service. A unit on disk
/// says nothing about the service running: a stale proxy squatting the
/// port makes the daemon die on bind while probe() still gets answers.
pub fn service_running(name: Option<&str>) -> (bool, Option<i64>) {
    if cfg!(target_os = "macos") {
        let lbl = label(name);
        let (_, out, _) = run(&["launchctl", "list"]);
        for line in out.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 && parts[2].trim() == lbl {
                let running = parts[0].trim() != "-";
                return (running, parts[1].trim().parse().ok());
            }
        }
        (false, None)
    } else if cfg!(target_os = "linux") {
        let (_, out, _) = run(&["systemctl", "--user", "is-active", &systemd_unit(name)]);
        (out.trim() == "active", None)
    } else {
        (false, None)
    }
}

/// True if something already listens on the loopback port.
pub fn port_in_use(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_err()
}

// --- health -------------------------------------------------------------------------

/// GET a JSON document from a local cliff proxy (either implementation).
/// A deliberately tiny HTTP/1.1 client: the CLI's status checks should not
/// need an async runtime.
pub fn http_get_json(port: u16, path: &str, timeout: Duration) -> Option<Value> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, timeout).ok()?;
    stream.set_read_timeout(Some(timeout)).ok()?;
    stream.set_write_timeout(Some(timeout)).ok()?;
    write!(stream, "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAccept: application/json\r\nConnection: close\r\n\r\n").ok()?;
    let mut raw = vec![];
    stream.read_to_end(&mut raw).ok()?;
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = String::from_utf8_lossy(&raw[..split]).to_lowercase();
    let status_ok = head.lines().next().is_some_and(|l| l.split_whitespace().nth(1) == Some("200"));
    if !status_ok {
        return None;
    }
    let mut body = raw[split + 4..].to_vec();
    if head.lines().any(|l| l.starts_with("transfer-encoding:") && l.contains("chunked")) {
        body = dechunk(&body)?;
    }
    serde_json::from_slice(&body).ok()
}

fn dechunk(mut data: &[u8]) -> Option<Vec<u8>> {
    let mut out = vec![];
    loop {
        let line_end = data.windows(2).position(|w| w == b"\r\n")?;
        let size_text = String::from_utf8_lossy(&data[..line_end]);
        let size = usize::from_str_radix(size_text.split(';').next()?.trim(), 16).ok()?;
        data = &data[line_end + 2..];
        if size == 0 {
            return Some(out);
        }
        out.extend_from_slice(data.get(..size)?);
        data = data.get(size + 2..)?;
    }
}

/// GET the proxy status endpoint; None if unreachable.
pub fn probe(port: u16) -> Option<Value> {
    http_get_json(port, "/__cliff__/status", Duration::from_secs(1))
}

pub fn wait_healthy(port: u16, deadline: Duration) -> Option<Value> {
    let end = Instant::now() + deadline;
    while Instant::now() < end {
        if let Some(status) = probe(port) {
            return Some(status);
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    None
}
