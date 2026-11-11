//! Load generator for comparing proxy implementations.
//!
//!     loadgen upstream PORT                  # fast mock upstream
//!     loadgen drive PROXY_URL bench.json     # parallel growing sessions
//!     loadgen digest UPSTREAM_URL            # what the upstream received
//!
//! `drive` runs every session of scripts/bench_data.py's workload
//! concurrently; within a session, request k sends history[..1 + 2k] as an
//! agent would. The upstream keeps an order-independent digest (XOR of
//! SHA-256s) of every body it received, so two proxy implementations can be
//! shown to have forwarded byte-identical requests.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::routing::get;
use axum::Router;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Default)]
struct Received {
    count: u64,
    bytes: u64,
    xor: [u8; 32],
}

async fn upstream(port: u16) {
    let state = Arc::new(Mutex::new(Received::default()));
    let app = Router::new()
        .route(
            "/__digest",
            get(|State(s): State<Arc<Mutex<Received>>>| async move {
                let r = s.lock().unwrap();
                let hex: String = r.xor.iter().map(|b| format!("{b:02x}")).collect();
                json!({"count": r.count, "bytes": r.bytes, "xor": hex}).to_string()
            }),
        )
        .fallback(|State(s): State<Arc<Mutex<Received>>>, req: Request| async move {
            let body = to_bytes(req.into_body(), usize::MAX).await.unwrap();
            let h = Sha256::digest(&body);
            let mut r = s.lock().unwrap();
            r.count += 1;
            r.bytes += body.len() as u64;
            for (x, b) in r.xor.iter_mut().zip(h.iter()) {
                *x ^= b;
            }
            axum::response::Response::builder()
                .header("content-type", "application/json")
                .body(Body::from(r#"{"id":"msg","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#))
                .unwrap()
        })
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn drive(proxy: String, data_path: String) {
    let data: Value = serde_json::from_slice(&std::fs::read(&data_path).unwrap()).unwrap();
    let turns = data["turns"].as_u64().unwrap() as usize;
    let client = reqwest::Client::builder().no_proxy().pool_max_idle_per_host(64).build().unwrap();
    let start = Instant::now();
    let mut tasks = vec![];
    for s in data["sessions"].as_array().unwrap().clone() {
        let (client, proxy) = (client.clone(), proxy.clone());
        tasks.push(tokio::spawn(async move {
            // Pre-serialize each message once; a request is a concatenation.
            let head = format!(
                "{{\"model\":\"claude-sonnet-5\",\"max_tokens\":32000,\"system\":{},\"messages\":[",
                serde_json::to_string(&s["system"]).unwrap()
            );
            let msgs: Vec<String> =
                s["history"].as_array().unwrap().iter().map(|m| serde_json::to_string(m).unwrap()).collect();
            let mut latencies = vec![];
            let mut sent = 0usize;
            for k in 1..=turns {
                let body = format!("{head}{}]}}", msgs[..1 + 2 * k].join(","));
                sent += body.len();
                let t = Instant::now();
                let r = client.post(format!("{proxy}/v1/messages")).body(body).send().await.unwrap();
                assert_eq!(r.status(), 200);
                r.bytes().await.unwrap();
                latencies.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            (latencies, sent)
        }));
    }
    // Meanwhile, a cheap request every 20ms: does heavy traffic starve it?
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let pinger = {
        let (client, proxy, done) = (client.clone(), proxy.clone(), done.clone());
        tokio::spawn(async move {
            let mut lat = vec![];
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                let t = Instant::now();
                let r = client.get(format!("{proxy}/__cliff__/status")).send().await.unwrap();
                r.bytes().await.unwrap();
                lat.push(t.elapsed().as_secs_f64() * 1000.0);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            lat
        })
    };
    let mut all = vec![];
    let mut sent = 0;
    for t in tasks {
        let (l, s) = t.await.unwrap();
        all.extend(l);
        sent += s;
    }
    let wall = start.elapsed().as_secs_f64();
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut pings = pinger.await.unwrap();
    pings.sort_by(f64::total_cmp);
    let ppct = |p: f64| pings[((pings.len() as f64 * p) as usize).min(pings.len() - 1)];
    all.sort_by(f64::total_cmp);
    let pct = |p: f64| all[((all.len() as f64 * p) as usize).min(all.len() - 1)];
    println!(
        "{} requests in {wall:.2}s = {:.0} req/s, {:.0} MB/s in | latency ms p50 {:.1}  p90 {:.1}  p99 {:.1}  max {:.1}",
        all.len(),
        all.len() as f64 / wall,
        sent as f64 / 1e6 / wall,
        pct(0.5),
        pct(0.9),
        pct(0.99),
        all[all.len() - 1]
    );
    println!(
        "status pings during load: {}  latency ms p50 {:.1}  p99 {:.1}  max {:.1}",
        pings.len(),
        ppct(0.5),
        ppct(0.99),
        pings[pings.len() - 1]
    );
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("upstream") => upstream(args[1].parse().unwrap()).await,
        Some("drive") => drive(args[1].clone(), args[2].clone()).await,
        Some("digest") => {
            let r = reqwest::Client::builder().no_proxy().build().unwrap();
            let body = r.get(format!("{}/__digest", args[1])).send().await.unwrap().bytes().await.unwrap();
            println!("{}", String::from_utf8_lossy(&body));
        }
        _ => eprintln!("usage: loadgen upstream PORT | drive PROXY_URL DATA | digest UPSTREAM_URL"),
    }
}
