//! Per-request CPU cost of the Rust engine: parse + prepare + serialize.
//! The counterpart of scripts/bench_prepare.py (same body, same scenarios).
//!
//!     cargo run --release -p cliff-core --example bench_prepare -- bench.json
//!     RAYON_NUM_THREADS=1 cargo run ...   # without per-message parallelism

use std::time::Instant;

use cliff_core::dialects::anthropic::DIALECT;
use cliff_core::{Config, Engine};
use serde_json::{json, Value};

fn main() {
    let path = std::env::args().nth(1).expect("bench.json");
    let data: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let s = &data["sessions"][0];
    let raw = serde_json::to_vec(&json!({
        "model": "claude-sonnet-5", "max_tokens": 32000, "system": s["system"], "messages": s["history"],
    }))
    .unwrap();
    let one = |engine: &Engine| {
        let t = Instant::now();
        let ctx = engine.prepare(serde_json::from_slice(&raw).unwrap(), &DIALECT).unwrap();
        if ctx.modified {
            std::hint::black_box(ctx.outgoing_json());
        }
        (t.elapsed().as_secs_f64() * 1000.0, ctx.modified, ctx.compacted)
    };
    let run = |name: &str, make: &dyn Fn() -> Engine| {
        let mut times = vec![];
        let mut flags = (false, false);
        for _ in 0..15 {
            let e = make();
            let (ms, m, c) = one(&e);
            times.push(ms);
            flags = (m, c);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{name:8} median {:8.2} ms   ({:.2} MB body, {} msgs, modified={} compacted={}, threads={})",
            times[times.len() / 2],
            raw.len() as f64 / 1e6,
            s["history"].as_array().unwrap().len(),
            flags.0,
            flags.1,
            rayon::current_num_threads()
        );
    };
    run("pass", &|| Engine::new(Config { threshold_tokens: 1_000_000_000, ..Config::default() }));
    let warm = std::sync::Arc::new(Engine::new(Config { threshold_tokens: 50_000, ..Config::default() }));
    one(&warm);
    run("match", &|| Engine::with_store(warm.cfg.clone(), warm.store.clone()));
    run("compact", &|| Engine::new(Config { threshold_tokens: 50_000, ..Config::default() }));
}
