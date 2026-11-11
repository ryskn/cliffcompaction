//! Where does prepare() spend its time? (development aid)
use std::time::Instant;
use cliff_core::dialects::anthropic::DIALECT;
use cliff_core::{engine, hashing, py};
use serde_json::{json, Value};

fn time<T>(name: &str, n: u32, mut f: impl FnMut() -> T) {
    let t = Instant::now();
    for _ in 0..n { std::hint::black_box(f()); }
    println!("{name:28} {:8.3} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
}

fn main() {
    let data: Value = serde_json::from_slice(&std::fs::read(std::env::args().nth(1).unwrap()).unwrap()).unwrap();
    let s = &data["sessions"][0];
    let raw = serde_json::to_vec(&json!({"model": "m", "max_tokens": 1, "system": s["system"], "messages": s["history"]})).unwrap();
    let body: Value = serde_json::from_slice(&raw).unwrap();
    let msgs = body["messages"].as_array().unwrap();
    time("parse (serde Value)", 20, || serde_json::from_slice::<Value>(&raw).unwrap());
    time("digest all msgs (serial)", 20, || msgs.iter().map(|m| (DIALECT.digest_message)(m).unwrap()).collect::<Vec<_>>());
    time("estimate_tokens(body)", 20, || engine::estimate_tokens(&body).unwrap());
    time("dumps_len only", 20, || py::dumps_len(&body, py::DUMPS_UNICODE));
    time("dumps to String", 20, || py::dumps(&body, py::DUMPS_UNICODE));
    time("canonical_json(body)", 20, || py::canonical_json(&body));
    time("sha256 of raw bytes", 20, || py::sha256_hex(&raw));
    time("serde to_vec(body)", 20, || serde_json::to_vec(&body).unwrap());
    let d: Vec<String> = msgs.iter().map(|m| (DIALECT.digest_message)(m).unwrap()).collect();
    time("chain_hashes", 20, || hashing::chain_hashes(&d));
}
