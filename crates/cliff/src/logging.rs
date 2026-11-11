//! `%(asctime)s cliff %(levelname)s %(message)s` on stderr, like the Python
//! CLI's handler. Only this project's targets are logged; HTTP-stack
//! internals stay quiet (uvicorn ran at log_level="warning").

use std::io::Write;

use log::{Level, LevelFilter, Log, Metadata, Record};

struct StderrLogger {
    level: LevelFilter,
}

fn ours(target: &str) -> bool {
    target.starts_with("cliff")
}

impl Log for StderrLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level && ours(metadata.target())
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let level = match record.level() {
            Level::Error => "ERROR",
            Level::Warn => "WARNING",
            Level::Info => "INFO",
            Level::Debug | Level::Trace => "DEBUG",
        };
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S,%3f");
        let _ = writeln!(std::io::stderr().lock(), "{now} cliff {level} {}", record.args());
    }

    fn flush(&self) {}
}

/// Install the logger (INFO, or DEBUG with -v). Idempotent.
pub fn setup(verbose: bool) {
    let level = if verbose { LevelFilter::Debug } else { LevelFilter::Info };
    let logger: &'static StderrLogger = Box::leak(Box::new(StderrLogger { level }));
    if log::set_logger(logger).is_ok() {
        log::set_max_level(level);
    }
}
