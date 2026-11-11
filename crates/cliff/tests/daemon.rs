//! Daemon management: profile wiring and service definitions, no
//! launchctl/systemctl (port of tests/test_daemon.py).
//!
//! These read $HOME, so they run as one test in their own binary.

use std::path::Path;

use cliff::cli::Common;
use cliff::daemon::*;

fn check_profiles(dir: &Path) {
    let block = env_block(8257, false);
    assert!(block.contains("export ANTHROPIC_BASE_URL=\"http://127.0.0.1:8257\""));
    assert!(block.contains("export OPENAI_BASE_URL=\"http://127.0.0.1:8257/v1\""));
    assert!(block.starts_with(MARK_BEGIN) && block.trim_end().ends_with(MARK_END));
    assert!(env_block(9000, true).contains("set -gx ANTHROPIC_BASE_URL \"http://127.0.0.1:9000\""));

    // Idempotent refresh; user content untouched.
    let rc = dir.join(".zshrc");
    std::fs::write(&rc, "# my rc\nexport FOO=1\n").unwrap();
    wire_profile(&rc, 8257).unwrap();
    wire_profile(&rc, 9001).unwrap();
    let text = std::fs::read_to_string(&rc).unwrap();
    assert_eq!(text.matches(MARK_BEGIN).count(), 1);
    assert!(text.contains("9001") && !text.contains("8257") && text.contains("export FOO=1"));
    assert!(profile_is_wired(&rc));

    // Clean unwire.
    let rc2 = dir.join(".bashrc");
    std::fs::write(&rc2, "export FOO=1\n").unwrap();
    wire_profile(&rc2, 8257).unwrap();
    unwire_profile(&rc2).unwrap();
    assert_eq!(std::fs::read_to_string(&rc2).unwrap(), "export FOO=1\n");
    assert!(!profile_is_wired(&rc2));

    // fish: the conf.d file is ours.
    let fish = dir.join("conf.d/cliffcompaction.fish");
    wire_profile(&fish, 8257).unwrap();
    assert!(std::fs::read_to_string(&fish).unwrap().contains("set -gx"));
    unwire_profile(&fish).unwrap();
    assert!(!fish.exists());
}

fn check_service_definitions() {
    let exe = std::env::current_exe().unwrap().display().to_string();
    let unit = systemd_unit_content(8257, &[], None);
    assert!(unit.contains(&format!("ExecStart={exe} serve --port 8257")));
    assert!(unit.contains("Restart=always"));

    let plist = plist_content(8257, &["--shadow".to_string()], None);
    let args = plist_program_arguments(&plist);
    assert_eq!(args, [exe.as_str(), "serve", "--port", "8257", "--shadow"]);
    assert!(plist.contains("<key>Label</key>\n\t<string>com.cliffcompaction.proxy</string>"));
    assert!(plist.contains("<key>KeepAlive</key>\n\t<true/>") && plist.contains("<key>RunAtLoad</key>\n\t<true/>"));

    // Byte-identical to what Python's plistlib writes for the same dict.
    if let Ok(out) = std::process::Command::new("python3")
        .args(["-I", "-c", "import plistlib,sys,json; sys.stdout.buffer.write(plistlib.dumps(json.loads(sys.argv[1])))"])
        .arg(serde_json::json!({
            "Label": "com.cliffcompaction.proxy", "ProgramArguments": args, "RunAtLoad": true, "KeepAlive": true,
            "StandardOutPath": log_path(None).display().to_string(), "StandardErrorPath": log_path(None).display().to_string(),
        }).to_string())
        .output()
    {
        if out.status.success() {
            assert_eq!(String::from_utf8(out.stdout).unwrap(), plist);
            eprintln!("plist: byte-identical to plistlib");
        }
    }

    // name=None must produce exactly what every existing install uses, or
    // its plist/unit/log would move on upgrade (and Python <-> Rust would
    // stop being a drop-in swap).
    assert_eq!(label(None), "com.cliffcompaction.proxy");
    assert_eq!(systemd_unit(None), "cliffcompaction.service");
    assert_eq!(plist_path(None).file_name().unwrap(), "com.cliffcompaction.proxy.plist");
    let log = log_path(None);
    assert!(["cliffcompaction.log", "proxy.log"].contains(&log.file_name().unwrap().to_str().unwrap()));

    // Named instances get their own label, unit and log.
    assert_eq!(label(Some("kimi")), "com.cliffcompaction.proxy.kimi");
    assert_eq!(systemd_unit(Some("kimi")), "cliffcompaction-kimi.service");
    assert_ne!(log_path(Some("kimi")), log_path(None));
    assert!(systemd_unit_content(8305, &[], Some("kimi")).contains("Description=CliffCompaction proxy (kimi)"));
}

fn check_names() {
    assert_eq!(validate_name(None), Ok(None));
    assert_eq!(validate_name(Some("")), Ok(None));
    assert_eq!(validate_name(Some("kimi-2_b")), Ok(Some("kimi-2_b".into())));
    for bad in ["-x", "a b", "a/b", "é", &"x".repeat(33)] {
        assert!(validate_name(Some(bad)).is_err(), "{bad}");
    }
}

#[cfg(target_os = "linux")]
fn check_installed(home: &Path) {
    let units = home.join(".config/systemd/user");
    std::fs::create_dir_all(&units).unwrap();
    assert_eq!(installed_port(None), None);
    // Written by the Python implementation: the port is still found.
    std::fs::write(
        units.join("cliffcompaction.service"),
        "[Service]\nExecStart=/usr/bin/python3 -m cliffcompaction.cli serve --port 8999 --shadow\n",
    )
    .unwrap();
    std::fs::write(units.join("cliffcompaction-kimi.service"), systemd_unit_content(8305, &[], Some("kimi"))).unwrap();
    std::fs::write(units.join("cliffcompaction-a.service"), systemd_unit_content(8306, &[], Some("a"))).unwrap();
    assert_eq!(installed_port(None), Some(8999));
    assert_eq!(installed_port(Some("kimi")), Some(8305));
    assert_eq!(installed_names(), [None, Some("a".into()), Some("kimi".into())]);
    assert!(service_installed(None) && !service_installed(Some("nope")));
}

fn check_serve_args() {
    let c = Common {
        shadow: true,
        strict: true,
        threshold: Some(120_000),
        keep_recent: Some(2),
        thought_max_chars: Some(300),
        result_max_chars: Some(400),
        drop_thinking: true,
        thinking_max_chars: Some(50),
        anthropic_upstream: Some("https://a.example".into()),
        openai_upstream: Some("https://o.example".into()),
        debug_dir: Some("/tmp/d".into()),
        verbose: true,
    };
    let args = c.serve_args();
    for flag in [
        "--shadow", "--strict", "--threshold", "--keep-recent", "--thought-max-chars", "--result-max-chars",
        "--drop-thinking", "--thinking-max-chars", "--anthropic-upstream", "--openai-upstream", "--debug-dir", "-v",
    ] {
        assert!(args.iter().any(|a| a == flag), "{flag} missing from {args:?}");
    }
    // ...and every one of them parses back into the same config.
    let mut argv = vec!["serve".to_string()];
    argv.extend(args);
    let parsed = <cliff::cli::Cli as clap::Parser>::try_parse_from(std::iter::once("cliff".to_string()).chain(argv)).unwrap();
    let cliff::cli::Cmd::Serve { common, .. } = parsed.cmd else { panic!("not serve") };
    assert_eq!(common.config(), c.config());
}

#[test]
fn daemon_management() {
    let home = tempfile::tempdir().unwrap();
    // SAFETY: single-threaded at this point; this binary has only this test.
    unsafe { std::env::set_var("HOME", home.path()) };
    check_profiles(home.path());
    check_service_definitions();
    check_names();
    #[cfg(target_os = "linux")]
    check_installed(home.path());
    check_serve_args();

    // CLI guards that need no service manager.
    assert_eq!(cliff::cli::main(vec!["enable".into(), "--name".into(), "kimi".into()]), 1);
    assert_eq!(cliff::cli::main(vec!["enable".into(), "--name".into(), "bad name".into(), "--port".into(), "9".into()]), 1);
    assert_eq!(cliff::cli::main(vec!["restart".into(), "--name".into(), "ghost".into()]), 1);
    assert_eq!(cliff::cli::main(vec!["run".into()]), 2); // no command after --
    assert_eq!(cliff::cli::main(vec!["bogus".into()]), 2);

    // probe/port_in_use against a real listener.
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    assert!(port_in_use(port));
    drop(l);
    assert!(probe(port).is_none());
}
