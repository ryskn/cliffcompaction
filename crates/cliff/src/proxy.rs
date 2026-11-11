//! The transparent proxy.
//!
//! Fail-open: any failure (unparseable body, unknown path, engine error,
//! even a panic) results in verbatim passthrough. Shadow mode runs the full
//! pipeline but forwards every request verbatim. The single exception is
//! strict mode: a request still over budget once the escalation ladder is
//! exhausted is refused rather than forwarded.
//!
//! ## Concurrency architecture
//!
//! - **IO on tokio.** Accepting connections, reading bodies, talking to the
//!   upstream and streaming its response back run on the multi-threaded
//!   work-stealing runtime. Nothing CPU-heavy runs between two `.await`s.
//! - **CPU on rayon.** Parsing a multi-megabyte body, hashing every message,
//!   compacting and re-serializing take milliseconds — far past the
//!   10–100µs an async task may hold a worker. That work is shipped to the
//!   rayon pool with [`on_cpu`] (rayon::spawn + a oneshot back to the
//!   task), where the engine further splits per-message work across cores.
//! - **Shared state behind short blocking locks.** The prefix store, the
//!   session tracker and the event hub are each a `std::sync::Mutex` held for
//!   a lookup or a push and never across an `.await`.
//! - **Watchers can never block requests.** Events go out through bounded
//!   per-subscriber channels with `try_send`; a slow watcher loses events.
//! - **Cancellation by drop.** A client that hangs up drops the response
//!   stream, which drops the upstream response and closes that connection.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime};

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use bytes::Bytes;
use cliff_core::config::{DEFAULT_ANTHROPIC_UPSTREAM, DEFAULT_OPENAI_UPSTREAM};
use cliff_core::dialects::{self, Dialect};
use cliff_core::engine::is_engine_input;
use cliff_core::identity;
use cliff_core::py::{self, DUMPS_UNICODE};
use cliff_core::{Config, Engine, RequestCtx};
use futures_util::{Stream, StreamExt};
use hashlink::LinkedHashMap;
use log::{debug, error, info, warn};
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::events::{self, EventHub};

const STRIP_REQUEST_HEADERS: &[&str] = &["host", "content-length", "connection", "transfer-encoding", "accept-encoding", "expect"];
const STRIP_RESPONSE_HEADERS: &[&str] = &["content-length", "transfer-encoding", "connection", "content-encoding"];
const ALLOWED_METHODS: &[Method] =
    &[Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::PATCH, Method::HEAD, Method::OPTIONS];

/// Distinct conversations kept for `is_side_call` / `cliff status`.
pub const MAX_SESSIONS: usize = 256;

// Providers phrase context-overflow errors inconsistently; match broadly.
static CONTEXT_ERROR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)context.?length|context.?window|prompt is too long|input.tokens|maximum.context|exceeds.limit|too.many.tokens|reduce.the.(?:length|input)|context_length_exceeded",
    )
    .unwrap()
});

pub fn is_context_error(body: &[u8]) -> bool {
    CONTEXT_ERROR_RE.is_match(&String::from_utf8_lossy(body))
}

/// Modification time of the running binary. A supervised daemon keeps
/// serving the code it started with; comparing this against the start time
/// is how `cliff status` notices an upgrade underneath it.
pub fn code_mtime() -> Option<SystemTime> {
    std::env::current_exe().ok()?.metadata().ok()?.modified().ok()
}

pub fn pick_upstream<'a>(cfg: &'a Config, path: &str) -> &'a str {
    let p = path.trim_end_matches('/');
    if p.ends_with("/chat/completions") || p.ends_with("/responses") || p.ends_with("/responses/compact") {
        return &cfg.openai_upstream;
    }
    if p.contains("/v1/messages") || p.ends_with("/messages") {
        return &cfg.anthropic_upstream;
    }
    // Neither dialect claims this path and the path alone cannot decide
    // (/v1/models exists on both APIs): follow the upstream the user
    // explicitly configured when that is unambiguous.
    if cfg.openai_upstream != DEFAULT_OPENAI_UPSTREAM && cfg.anthropic_upstream == DEFAULT_ANTHROPIC_UPSTREAM {
        return &cfg.openai_upstream;
    }
    &cfg.anthropic_upstream
}

// --- CPU offload ------------------------------------------------------------------

/// Run CPU-bound work on the rayon pool and await its result.
///
/// A panic inside the job is caught and returned as `Err` (an uncaught
/// panic in `rayon::spawn` would abort the process): the request then fails
/// open like any other engine error.
pub async fn on_cpu<T: Send + 'static>(job: impl FnOnce() -> T + Send + 'static) -> Result<T, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    rayon::spawn(move || {
        let result = catch_unwind(AssertUnwindSafe(job)).map_err(|panic| {
            panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".into())
        });
        let _ = tx.send(result);
    });
    rx.await.map_err(|_| "cpu job dropped".to_string())?
}

// --- state -------------------------------------------------------------------------

pub struct AppState {
    pub cfg: Config,
    pub engine: Arc<Engine>,
    pub hub: Arc<EventHub>,
    /// Distinct conversations that have taken a turn (LRU, display only).
    seen_sids: Mutex<LinkedHashMap<String, ()>>,
    started_at: SystemTime,
    debug_seq: AtomicU64,
    client: reqwest::Client,
}

impl AppState {
    pub fn sessions_tracked(&self) -> usize {
        self.seen_sids.lock().unwrap_or_else(|p| p.into_inner()).len()
    }
}

pub fn create_app(cfg: Config, engine: Option<Engine>) -> (Router, Arc<AppState>) {
    let engine = Arc::new(engine.unwrap_or_else(|| Engine::new(cfg.clone())));
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(600))
        .http1_only()
        .build()
        .expect("HTTP client");
    let state = Arc::new(AppState {
        cfg,
        engine,
        hub: Arc::new(EventHub::default()),
        seen_sids: Mutex::new(LinkedHashMap::new()),
        started_at: SystemTime::now(),
        debug_seq: AtomicU64::new(0),
        client,
    });
    let router = Router::new()
        // Non-GET methods on these paths fall through to the proxy, as the
        // Starlette catch-all route did.
        .route("/__cliff__/status", get(status).fallback(handle))
        .route("/__cliff__/events", get(events_stream).fallback(handle))
        .fallback(handle)
        .with_state(state.clone());
    (router, state)
}

// --- observability --------------------------------------------------------------------

/// One event per handled request. Observability only: it can never fail
/// the request.
fn emit_request(state: &AppState, ctx: Option<&RequestCtx>, dialect: &Dialect) {
    let Some(ctx) = ctx else { return };
    if catch_unwind(AssertUnwindSafe(|| emit_request_inner(state, ctx, dialect))).is_err() {
        error!("event emit failed (request unaffected)");
    }
}

fn emit_request_inner(state: &AppState, ctx: &RequestCtx, dialect: &Dialect) {
    if ctx.chain.is_empty() || identity::is_probe(&ctx.body) {
        return;
    }
    let model = ctx.body.get("model").and_then(Value::as_str).unwrap_or("?");
    // A side call is labelled with the session that fired it, but never
    // enters the tracker: only a turn makes a session "seen".
    let sid = identity::session_id(&ctx.chain, &ctx.body, dialect);
    let aux = {
        let mut seen = state.seen_sids.lock().unwrap_or_else(|p| p.into_inner());
        let aux = identity::is_side_call(ctx.msgs(), dialect, seen.contains_key(&sid));
        if !aux {
            seen.remove(&sid);
            seen.insert(sid.clone(), ());
            while seen.len() > MAX_SESSIONS {
                seen.pop_front();
            }
        }
        aux
    };
    let kind = if ctx.compacted { "compact" } else if ctx.modified { "match" } else { "pass" };
    let est_out = if ctx.est_tokens_out != 0 { ctx.est_tokens_out } else { ctx.est_tokens_in };
    state.hub.emit(vec![
        ("kind", json!(kind)),
        ("sid", json!(sid)),
        ("aux", json!(aux)),
        ("model", json!(model)),
        ("dialect", json!(dialect.name)),
        ("cut", json!(ctx.base_cut)),
        ("total", json!(ctx.msgs().len())),
        ("out_msgs", json!(ctx.out_msgs)),
        ("est_in", json!(ctx.est_tokens_in)),
        ("est_out", json!(est_out)),
        ("steps", json!(ctx.chain_steps)),
        ("rung", json!(ctx.rung)),
        ("fp", json!(ctx.summary_fp)),
    ]);
}

/// Write what cliff saw and what it sent. Failures never propagate.
fn debug_dump(state: &AppState, path: &str, ctx: &RequestCtx) {
    let run = || -> std::io::Result<String> {
        std::fs::create_dir_all(&state.cfg.debug_dir)?;
        let n = state.debug_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let fname = std::path::Path::new(&state.cfg.debug_dir)
            .join(format!("{}-{n:04}.json", chrono::Local::now().format("%Y%m%d-%H%M%S")));
        let key = ctx.dialect.messages_key;
        // Everything but the message array: what distinguishes a scaffold's
        // own background calls from a conversational turn lives here.
        let mut fields = Map::new();
        for (k, v) in ctx.body.as_object().into_iter().flatten() {
            if k == key || k == "system" {
                continue;
            }
            let v = match v {
                Value::Array(tools) if k == "tools" => json!(format!("<{} tools>", tools.len())),
                v => v.clone(),
            };
            fields.insert(k.clone(), v);
        }
        let outgoing = if ctx.modified {
            Value::Array(ctx.substituted_values().into_iter().cloned().collect())
        } else {
            Value::Null
        };
        let record = json!({
            "path": path,
            "dialect": ctx.dialect.name,
            "modified": ctx.modified,
            "compacted": ctx.compacted,
            "est_tokens_in": ctx.est_tokens_in,
            "est_tokens_out": ctx.est_tokens_out,
            "request_fields": fields,
            "incoming_messages": ctx.msgs(),
            "outgoing_messages": outgoing,
        });
        let text = py::dumps(&record, py::Fmt { indent: Some(1), ..DUMPS_UNICODE });
        std::fs::write(&fname, text)?;
        Ok(fname.display().to_string())
    };
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(fname)) => info!("debug: dumped request to {fname}"),
        Ok(Err(e)) => error!("debug dump failed (request unaffected): {e}"),
        Err(_) => error!("debug dump failed (request unaffected)"),
    }
}

// --- upstream ----------------------------------------------------------------------

fn filtered(headers: &HeaderMap, strip: &[&str]) -> HeaderMap {
    let mut out = HeaderMap::with_capacity(headers.len());
    for (k, v) in headers {
        if !strip.contains(&k.as_str()) {
            out.append(k.clone(), v.clone());
        }
    }
    out
}

struct Incoming {
    method: Method,
    path: String,
    query: Option<String>,
    headers: HeaderMap,
}

async fn send_upstream(state: &AppState, req: &Incoming, upstream: &str, body: Bytes) -> reqwest::Result<reqwest::Response> {
    let mut headers = filtered(&req.headers, STRIP_REQUEST_HEADERS);
    headers.insert(header::ACCEPT_ENCODING, HeaderValue::from_static("identity"));
    let mut url = format!("{}{}", upstream.trim_end_matches('/'), req.path);
    if let Some(q) = &req.query {
        url.push('?');
        url.push_str(q);
    }
    state.client.request(req.method.clone(), url).headers(headers).body(body).send().await
}

/// Relays the upstream body as it arrives, logging time to first byte and
/// leaving a trace when the upstream fails mid-stream or the client hangs
/// up (the stream is dropped before it finished).
struct RelayTrace {
    t_start: Option<Instant>,
    started: Instant,
    n_bytes: usize,
    done: bool,
}

impl RelayTrace {
    /// The stream ran to its end (or failed upstream): not a disconnect.
    fn finish(&mut self) {
        self.done = true;
    }
}

impl Drop for RelayTrace {
    fn drop(&mut self) {
        if !self.done {
            debug!(
                "relay: client disconnected after {} bytes / {:.1}s",
                self.n_bytes,
                self.started.elapsed().as_secs_f64()
            );
        }
    }
}

struct RelayState {
    upstream: std::pin::Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>,
    trace: RelayTrace,
    first: bool,
    finished: bool,
}

impl RelayState {
    fn first_byte(&mut self) {
        if self.first {
            if let Some(t) = self.trace.t_start {
                debug!("timing: first_byte {:.0}ms", t.elapsed().as_secs_f64() * 1000.0);
            }
            self.first = false;
        }
    }
}

fn relay_stream(resp: reqwest::Response, t_start: Option<Instant>) -> impl Stream<Item = reqwest::Result<Bytes>> {
    let state = RelayState {
        upstream: Box::pin(resp.bytes_stream()),
        trace: RelayTrace { t_start, started: Instant::now(), n_bytes: 0, done: false },
        first: t_start.is_some(),
        finished: false,
    };
    futures_util::stream::unfold(state, |mut s| async move {
        if s.finished {
            return None;
        }
        match s.upstream.next().await {
            Some(Ok(chunk)) => {
                s.first_byte();
                s.trace.n_bytes += chunk.len();
                Some((Ok(chunk), s))
            }
            Some(Err(e)) => {
                // Mid-stream upstream failure: the client sees a truncated
                // response; leave a trace so it is attributable.
                warn!(
                    "relay: upstream stream failed after {} bytes / {:.1}s: {e}",
                    s.trace.n_bytes,
                    s.trace.started.elapsed().as_secs_f64()
                );
                s.trace.finish();
                s.finished = true;
                Some((Err(e), s))
            }
            None => {
                s.first_byte();
                s.trace.finish();
                None
            }
        }
    })
}

fn relay(resp: reqwest::Response, t_start: Option<Instant>) -> Response {
    let status = resp.status();
    let headers = filtered(resp.headers(), STRIP_RESPONSE_HEADERS);
    let mut out = Response::new(Body::from_stream(relay_stream(resp, t_start)));
    *out.status_mut() = status;
    *out.headers_mut() = headers;
    out
}

fn buffered(status: StatusCode, headers: HeaderMap, data: Bytes) -> Response {
    let mut out = Response::new(Body::from(data));
    *out.status_mut() = status;
    *out.headers_mut() = headers;
    out
}

fn json_response(status: StatusCode, value: &Value) -> Response {
    // Starlette's JSONResponse: compact separators, ensure_ascii=False.
    let body = serde_json::to_vec(value).unwrap_or_default();
    let mut out = (status, body).into_response();
    out.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    out
}

// --- handlers --------------------------------------------------------------------

/// Parse + prepare + serialize, on the CPU pool. `Ok(None)`: not an engine
/// input (no messages array) — forwarded verbatim without an event.
fn prepare_job(state: &AppState, raw: &[u8], dialect: &'static Dialect) -> Result<Option<(RequestCtx, Option<Bytes>)>, String> {
    // json.loads(bytes) accepts a UTF-8 BOM.
    let raw = raw.strip_prefix(b"\xef\xbb\xbf").unwrap_or(raw);
    let body: Value = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
    if !body.is_object() || !is_engine_input(&body, dialect) {
        return Ok(None);
    }
    let ctx = state.engine.prepare(body, dialect).map_err(|e| e.to_string())?;
    let mut out = None;
    if ctx.modified {
        if state.cfg.shadow {
            info!(
                "shadow: would send ~{}k instead of ~{}k est tokens",
                ctx.est_tokens_out / 1000,
                ctx.est_tokens_in / 1000
            );
        } else {
            out = Some(Bytes::from(ctx.outgoing_json()));
        }
    }
    Ok(Some((ctx, out)))
}

async fn handle(State(state): State<Arc<AppState>>, request: Request) -> Response {
    let t0 = Instant::now();
    if !ALLOWED_METHODS.contains(request.method()) {
        return (StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed").into_response();
    }
    let (parts, body) = request.into_parts();
    let raw = match to_bytes(body, usize::MAX).await {
        Ok(raw) => raw,
        Err(e) => {
            warn!("could not read request body: {e}");
            return (StatusCode::BAD_REQUEST, "could not read request body").into_response();
        }
    };
    let t_read = Instant::now();
    let incoming = Incoming {
        method: parts.method,
        path: parts.uri.path().to_string(),
        query: parts.uri.query().map(str::to_string),
        headers: parts.headers,
    };
    let path = incoming.path.clone();
    let cfg = &state.cfg;
    let upstream = pick_upstream(cfg, &path).to_string();
    let dialect = dialects::detect(&path);

    let mut ctx: Option<RequestCtx> = None;
    let mut out_body = raw.clone();
    if let Some(dialect) = dialect.filter(|_| incoming.method == Method::POST && !raw.is_empty()) {
        let (job_state, job_raw) = (state.clone(), raw.clone());
        match on_cpu(move || prepare_job(&job_state, &job_raw, dialect)).await {
            Ok(Ok(Some((c, body)))) => {
                if let Some(body) = body {
                    out_body = body;
                }
                ctx = Some(c);
            }
            Ok(Ok(None)) => {}
            Ok(Err(e)) | Err(e) => error!("prepare failed; passing through verbatim: {e}"),
        }
        if !cfg.debug_dir.is_empty() {
            if let Some(c) = ctx.take() {
                let (dump_state, dump_path) = (state.clone(), path.clone());
                ctx = tokio::task::spawn_blocking(move || {
                    debug_dump(&dump_state, &dump_path, &c);
                    c
                })
                .await
                .ok();
            }
        }
        emit_request(&state, ctx.as_ref(), dialect);
    }

    // Strict mode: a request still over threshold with the ladder exhausted
    // never reaches the provider. Inert under --shadow.
    if let Some(c) = ctx.as_ref().filter(|c| cfg.strict && !cfg.shadow && c.over_budget) {
        warn!(
            "strict: refusing request at ~{}k est tokens after escalation (budget {}k, rung {})",
            c.est_tokens_out / 1000,
            cfg.threshold_tokens / 1000,
            c.rung
        );
        let message = format!(
            "cliff strict mode: request is ~{} est tokens with the escalation ladder exhausted (rung {}), over the configured budget of {}. Refusing to forward.",
            c.est_tokens_out, c.rung, cfg.threshold_tokens
        );
        return json_response(
            StatusCode::BAD_REQUEST,
            &json!({"type": "error", "error": {"type": "cliff_over_budget", "message": message}}),
        );
    }

    let t_prep = Instant::now();
    let resp = match send_upstream(&state, &incoming, &upstream, out_body).await {
        Ok(resp) => resp,
        Err(e) => {
            error!("upstream error: {e}");
            return json_response(
                StatusCode::BAD_GATEWAY,
                &json!({"error": {"type": "upstream_error", "message": e.to_string()}}),
            );
        }
    };
    let status = resp.status();
    debug!("access: {} {} -> {} ({:.0}ms)", incoming.method, path, status.as_u16(), t0.elapsed().as_secs_f64() * 1000.0);
    if status.as_u16() >= 400 {
        warn!("upstream returned {} for {} {}", status.as_u16(), incoming.method, path);
    }
    if ctx.is_some() {
        debug!(
            "timing: read {:.0}ms, prepare {:.0}ms, upstream_headers {:.0}ms, status={}",
            (t_read - t0).as_secs_f64() * 1000.0,
            (t_prep - t_read).as_secs_f64() * 1000.0,
            t_prep.elapsed().as_secs_f64() * 1000.0,
            status.as_u16()
        );
    }

    // Reactive fallback: on a context-length 400, walk the escalation ladder
    // (base force-compact -> keep_recent=1 -> +caps -> truncated summary),
    // replaying after each rung until the provider accepts.
    if status == StatusCode::BAD_REQUEST && !cfg.shadow {
        if let (Some(c), Some(dialect)) = (ctx.take(), dialect) {
            let headers = filtered(resp.headers(), STRIP_RESPONSE_HEADERS);
            let data = resp.bytes().await.unwrap_or_default();
            return reactive(&state, &incoming, &upstream, dialect, c, headers, data, t0).await;
        }
    }
    relay(resp, ctx.is_some().then_some(t0))
}

#[allow(clippy::too_many_arguments)]
async fn reactive(
    state: &Arc<AppState>,
    incoming: &Incoming,
    upstream: &str,
    dialect: &'static Dialect,
    mut ctx: RequestCtx,
    headers: HeaderMap,
    mut data: Bytes,
    t0: Instant,
) -> Response {
    if !is_context_error(&data) {
        return buffered(StatusCode::BAD_REQUEST, headers, data);
    }
    loop {
        let engine = state.engine.clone();
        let step = on_cpu(move || {
            let step = engine.reactive(&mut ctx);
            let body = matches!(step, Ok(true)).then(|| Bytes::from(ctx.outgoing_json()));
            (ctx, step, body)
        })
        .await;
        let (c, step, body) = match step {
            Ok(r) => r,
            Err(e) => {
                error!("reactive compaction failed; returning original 400: {e}");
                break;
            }
        };
        ctx = c;
        let body = match (step, body) {
            (Ok(true), Some(body)) => body,
            (Ok(_), _) => {
                warn!("reactive: escalation exhausted; returning 400");
                break;
            }
            (Err(e), _) => {
                error!("reactive compaction failed; returning original 400: {e}");
                break;
            }
        };
        info!("reactive: replaying compacted request (rung {})", ctx.rung);
        emit_request(state, Some(&ctx), dialect);
        let t_replay = Instant::now();
        let resp2 = match send_upstream(state, incoming, upstream, body).await {
            Ok(r) => r,
            Err(e) => {
                error!("upstream error on replay: {e}");
                break;
            }
        };
        let status2 = resp2.status();
        debug!(
            "access: {} {} -> {} ({:.0}ms, replay)",
            incoming.method,
            incoming.path,
            status2.as_u16(),
            t_replay.elapsed().as_secs_f64() * 1000.0
        );
        if status2 == StatusCode::BAD_REQUEST {
            let headers2 = filtered(resp2.headers(), STRIP_RESPONSE_HEADERS);
            let data2 = resp2.bytes().await.unwrap_or_default();
            if is_context_error(&data2) {
                data = data2; // still too big: next rung
                continue;
            }
            return buffered(StatusCode::BAD_REQUEST, headers2, data2);
        }
        if status2.as_u16() >= 400 {
            warn!("replay still failed: upstream returned {}", status2.as_u16());
        }
        return relay(resp2, Some(t0));
    }
    // MIRRORED QUIRK: the body may be the last replay's, the headers are the
    // original response's.
    buffered(StatusCode::BAD_REQUEST, headers, data)
}

async fn status(State(state): State<Arc<AppState>>) -> Response {
    let engine = &state.engine;
    let stale = code_mtime().is_some_and(|m| m > state.started_at);
    let uptime = state.started_at.elapsed().map(|d| d.as_secs()).unwrap_or(0);
    json_response(
        StatusCode::OK,
        &json!({
            "name": "cliffcompaction",
            "version": cliff_core::VERSION,
            "implementation": "rust",
            "shadow": state.cfg.shadow,
            "strict": state.cfg.strict,
            "threshold_tokens": state.cfg.threshold_tokens,
            "keep_recent": state.cfg.keep_recent,
            "store_entries": engine.store.len(),
            "store_bytes": engine.store.nbytes(),
            "sessions_tracked": state.sessions_tracked(),
            // Installed under a running daemon: it is still serving the code
            // it started with, and looks perfectly healthy doing it.
            "code_stale": stale,
            "watchers": state.hub.subscribers(),
            "uptime_s": uptime,
        }),
    )
}

/// SSE stream for `cliff watch`: recent backlog, then live events, with a
/// keepalive comment every 10s to detect a gone client.
async fn events_stream(State(state): State<Arc<AppState>>) -> Response {
    let (sub, backlog) = state.hub.subscribe();
    let backlog = futures_util::stream::iter(backlog.into_iter().map(|ev| Ok::<_, std::convert::Infallible>(Bytes::from(events::sse(&ev)))));
    let live = futures_util::stream::unfold(sub, |mut sub| async move {
        let frame = match tokio::time::timeout(Duration::from_secs(10), sub.rx.recv()).await {
            Ok(Some(ev)) => Bytes::from(events::sse(&ev)),
            Ok(None) => return None,
            Err(_) => Bytes::from_static(b": keepalive\n\n"),
        };
        Some((Ok(frame), sub))
    });
    let mut resp = Response::new(Body::from_stream(backlog.chain(live)));
    let h = resp.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    h.insert(HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"));
    resp
}

// --- serving -------------------------------------------------------------------------

/// Serve `router` on an already-bound listener until `shutdown` resolves.
pub async fn serve(
    listener: tokio::net::TcpListener,
    router: Router,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router).with_graceful_shutdown(shutdown).await
}

/// SIGINT or SIGTERM.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
}
