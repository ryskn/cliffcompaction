//! Test harness: a real mock upstream server and a real proxy, both on
//! ephemeral loopback ports (the Rust stand-in for httpx.MockTransport +
//! Starlette's TestClient).
#![allow(dead_code)]

pub mod fixtures;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::Router;
use bytes::Bytes;
use cliff::proxy::{create_app, AppState};
use cliff_core::{Config, Engine};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    pub query: Option<String>,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl Recorded {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).expect("upstream got JSON")
    }
    pub fn messages(&self) -> Vec<Value> {
        self.json()["messages"].as_array().cloned().unwrap_or_default()
    }
}

pub struct Scripted {
    pub status: u16,
    pub headers: Vec<(&'static str, &'static str)>,
    pub body: Vec<u8>,
}

#[derive(Default)]
pub struct UpInner {
    pub requests: Vec<Recorded>,
    pub script: VecDeque<Scripted>,
    /// Reject with a context error until the request fits.
    pub picky: Option<usize>,
    /// Fail the test if anything is forwarded.
    pub unreachable: bool,
}

#[derive(Clone, Default)]
pub struct Upstream(pub Arc<Mutex<UpInner>>);

impl Upstream {
    pub fn requests(&self) -> Vec<Recorded> {
        self.0.lock().unwrap().requests.clone()
    }
    pub fn last(&self) -> Recorded {
        self.requests().last().cloned().expect("upstream saw a request")
    }
    pub fn count(&self) -> usize {
        self.0.lock().unwrap().requests.len()
    }
    pub fn push(&self, status: u16, body: Value) {
        self.0.lock().unwrap().script.push_back(Scripted {
            status,
            headers: vec![("content-type", "application/json")],
            body: serde_json::to_vec(&body).unwrap(),
        });
    }
}

async fn upstream_handler(State(up): State<Upstream>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body = to_bytes(body, usize::MAX).await.unwrap();
    let mut inner = up.0.lock().unwrap();
    assert!(!inner.unreachable, "strict mode forwarded an over-budget request");
    inner.requests.push(Recorded {
        method: parts.method.to_string(),
        path: parts.uri.path().to_string(),
        query: parts.uri.query().map(str::to_string),
        headers: parts.headers,
        body: body.clone(),
    });
    let reply = if let Some(s) = inner.script.pop_front() {
        s
    } else if inner.picky.is_some_and(|max| body.len() > max) {
        Scripted {
            status: 400,
            headers: vec![("content-type", "application/json")],
            body: br#"{"error": {"type": "invalid_request_error", "message": "prompt is too long"}}"#.to_vec(),
        }
    } else {
        Scripted {
            status: 200,
            headers: vec![("content-type", "application/json")],
            body: br#"{"id": "msg_1", "role": "assistant", "content": []}"#.to_vec(),
        }
    };
    let mut resp = Response::new(Body::from(reply.body));
    *resp.status_mut() = StatusCode::from_u16(reply.status).unwrap();
    for (k, v) in reply.headers {
        resp.headers_mut().insert(k, v.parse().unwrap());
    }
    resp
}

async fn listen(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

pub async fn start_upstream(inner: UpInner) -> (String, Upstream) {
    let up = Upstream(Arc::new(Mutex::new(inner)));
    let router = Router::new().fallback(upstream_handler).with_state(up.clone());
    (listen(router).await, up)
}

pub struct Harness {
    pub base: String,
    pub state: Arc<AppState>,
    pub up: Upstream,
    pub client: reqwest::Client,
}

impl Harness {
    pub async fn new(cfg: Config) -> Harness {
        Harness::with(cfg, UpInner::default(), None).await
    }

    /// Both upstreams point at one mock unless the config names others.
    pub async fn with(mut cfg: Config, inner: UpInner, engine: Option<Engine>) -> Harness {
        let (up_base, up) = start_upstream(inner).await;
        if cfg.anthropic_upstream == cliff_core::config::DEFAULT_ANTHROPIC_UPSTREAM {
            cfg.anthropic_upstream = up_base.clone();
        }
        if cfg.openai_upstream == cliff_core::config::DEFAULT_OPENAI_UPSTREAM {
            cfg.openai_upstream = up_base;
        }
        let (router, state) = create_app(cfg, engine);
        let base = listen(router).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        Harness { base, state, up, client }
    }

    pub async fn post(&self, path: &str, body: &Value) -> reqwest::Response {
        self.post_raw(path, serde_json::to_vec(body).unwrap()).await
    }

    pub async fn post_raw(&self, path: &str, body: Vec<u8>) -> reqwest::Response {
        self.client.post(format!("{}{path}", self.base)).body(body).send().await.unwrap()
    }

    pub async fn get(&self, path: &str) -> reqwest::Response {
        self.client.get(format!("{}{path}", self.base)).send().await.unwrap()
    }

    pub fn events(&self) -> Vec<Value> {
        self.state.hub.backlog().iter().map(|e| (**e).clone()).collect()
    }
}
