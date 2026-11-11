# Migrating CliffCompaction to Rust

Status: **the Rust implementation is feature-complete and at verified parity
with the Python implementation**. The Python package remains the reference
implementation (the test oracle) and the fallback; packaging has not been
switched. This document records how the port was done, why, how parity is
verified, what is deliberately different, and what is left before the
Python code can be retired.

```
crates/cliff-core   pure, synchronous core (canonicalization, hashing, dialects,
                    compaction, prefix store, request engine) — no IO, no async
crates/cliff        the IO shell: proxy (tokio + axum + reqwest), event hub,
                    CLI, daemon management, `cliff watch` — binary `cliff`
scripts/conformance.py   case generator + Python oracle + differential fuzzer
```

## 1. Methodology

The approach follows what practitioners who completed comparable ports
report, rather than a big-bang rewrite:

| Practice | Source | How it was applied here |
| --- | --- | --- |
| Keep the system shippable at every step; replace components while it keeps running | fish shell's C++→Rust port ([Fish of Theseus](https://fishshell.com/blog/rustport/), [plan](https://github.com/fish-shell/fish-shell/blob/master/doc_internal/fish-riir-plan.md)) | Python stays installed and working. The Rust daemon is a drop-in: same port, HTTP surface, event schema, service label, systemd unit, log path and shell-profile markers. Switching is `cliff enable` from either implementation; so is rolling back. |
| Port faithfully first, refactor later; keep the old structure so before/after comparison finds mistranslations | fish, [Waleed Khan](https://blog.waleedkhan.name/port-python-to-rust/) | One Rust module per Python module, same function names, same comments. Python semantics that Rust primitives don't share live in one module, `cliff_core::py` (below), so the rest reads like the Python it came from. |
| Use the old implementation as the answer key; byte-diff outputs | [Behavioral parity, not compilation](https://chyshkala.com/blog/the-real-work-in-porting-code-behavioral-parity-not-compilation) | `scripts/conformance.py` runs generated inputs through the Python code; the Rust port must produce identical results, including the SHA-256 of every outgoing body. |
| Mirror known bugs during the port, mark them, fix them later at the source | same | Python's latent exceptions (which make a request fail open) are reproduced as `Err` values and marked `MIRRORED BUG` / `MIRRORED QUIRK` (§4). |
| Differential fuzzing finds what examples miss — but not what the generator never produces | [710/710](https://dev.to/sujalshah/my-port-passes-710-out-of-710-here-is-everything-that-number-does-not-tell-you-4pp1), [what fuzzing can't find](https://dev.to/sanjaysah/the-bug-that-differential-fuzzing-cannot-find-1gi4) | The harness itself was mutation-tested (§3); every missed mutant became a generator improvement (boundary values, threshold sweeps). |
| Panics/`unwrap` replacing real error handling were the main source of port bugs | fish | No `unwrap` on input-derived data. Exceptions become `PyResult` (`Err(PyErr)`); a panic in a CPU job is caught and the request fails open. |

### The Python-semantics layer (`cliff_core::py`)

Everything where Python and the obvious Rust primitive disagree, verified
against the interpreter rather than from memory:

| Python | naive Rust | consequence if missed |
| --- | --- | --- |
| `len(s)` counts code points | `s.len()` counts bytes | truncation caps, result-size cutoffs, budgets wrong on non-ASCII |
| `str.strip()` / `\s` include U+001C–U+001F | `trim()` / Unicode `White_Space` don't | summary text differs |
| `json.dumps` writes `", "`/`": "`, `float.__repr__`, escapes like CPython | serde_json writes compact, ryu floats | token estimates (chars/4) and outgoing bytes differ |
| dicts keep insertion order; ints are unbounded | `Value` sorts keys; numbers become f64 | key order and big ints change |
| truthiness, `isinstance(True, int)` | — | probe detection, `or` defaults |
| `str(obj)` / `repr(obj)` of non-string content | — | digests of odd tool results differ |
| `base64.b64decode(validate=False)` | strict decoders | image size estimates differ |
| exceptions caught by the fail-open path | panics | a malformed request would take the proxy down |

serde_json runs with `preserve_order` and `arbitrary_precision`; numbers are
re-emitted exactly as Python's `json` round-trips them (`1E5` → `100000.0`,
`-0` → `0`, `1e400` → `Infinity`).

## 2. Architecture: concurrency the Rust way

The Python proxy is one asyncio event loop: parsing, hashing and compacting a
multi-megabyte body runs on the loop and stalls every other request while it
does. The Rust port splits the work by kind, following the Tokio
maintainers' guidance ([Async: what is blocking?](https://ryhl.io/blog/async-what-is-blocking/):
no more than ~10–100 µs between `.await`s; CPU-bound work belongs on rayon,
bridged with a oneshot channel):

```
            tokio multi-thread runtime (IO)                 rayon pool (CPU)
 client ──► accept, read body ──► on_cpu(job) ──oneshot──► parse JSON
                                                            ├─ join ─┬─ par_iter: digest each message
                                                            │        └─ estimate whole body
                                                            ├─ chain hashes (sequential)
                                                            ├─ store lookup / compaction replay
                                                            └─ serialize outgoing body
        ◄──────────────────────────────────── result ◄──────┘
        send upstream (reqwest), stream response back (Body::from_stream)
        emit event ──try_send──► per-watcher bounded channels (drop on full)
```

* **Functional core, imperative shell.** `cliff-core` is synchronous and
  deterministic; it never sees a socket or a runtime. It runs on any thread
  pool, and tests call it directly.
* **CPU work on rayon** via `proxy::on_cpu` (`rayon::spawn` + oneshot). The
  job is wrapped in `catch_unwind`: an unwinding panic inside `rayon::spawn`
  would abort the process, and here it becomes a fail-open passthrough
  instead.
* **Data parallelism where the data is independent.** A message's digest
  depends on nothing else, so digests run as a `par_iter` forked
  (`rayon::join`) against the whole-body size estimate. The hash chain and
  the threshold replay are inherently sequential scans and stay serial.
  Per-message billable sizes are computed in parallel only when a compaction
  needs them.
* **Shared state behind short blocking locks.** The prefix store (an O(1)
  LRU), the session tracker and the event hub each sit behind a
  `std::sync::Mutex` held for a lookup or a push and never across an
  `.await`, which is the case the [Tokio docs](https://docs.rs/tokio/latest/tokio/sync/struct.Mutex.html)
  recommend a blocking mutex for. The store needs no further coordination
  (single-flight, actor): an entry is a deterministic function of its key,
  so concurrent fills write the same value.
* **Watchers can never block requests.** Each `cliff watch` subscriber has a
  bounded `mpsc` channel fed with `try_send`; a slow one loses its newest
  events, exactly like the Python `put_nowait` it replaces.
* **Cancellation by drop.** A client that hangs up drops the response
  stream, which drops the upstream response and closes that connection.

## 3. Verification

| Layer | What | Where |
| --- | --- | --- |
| Ported unit tests | every pytest that exercises behavior, as Rust tests | `crates/*/tests/`, `#[cfg(test)]` modules |
| Golden corpus | 300 generated cases (513 requests: compactions, escalation rungs 1–3, prefix matches, reactive ladders, fail-open errors, all three dialects) with the Python oracle's results; runs in `cargo test`, no Python needed | `crates/cliff-core/tests/golden/` |
| Differential fuzzing | any number of fresh random cases, Python vs Rust live | `uv run python scripts/conformance.py fuzz --n 3000 [--scale large]` |
| Mutation testing of the harness | typical porting mistakes injected one at a time; the fuzzer must notice | §3.1 |
| End-to-end byte parity under load | 16 concurrent agent sessions × 150 turns through each proxy: the upstream's order-independent digest of all 2,400 bodies (284,681,976 bytes) is identical | §5 |
| Cross-implementation interop | Python `cliff status` ↔ Rust daemon and vice versa; the Rust plist is byte-identical to `plistlib.dumps` | `crates/cliff/tests/daemon.rs` |

Totals at the time of writing: 123 Rust tests and the 161 Python tests pass;
differential fuzzing with the final generator over 10,385 cases (1,396 of them
at large scale), including 3,794 after the performance work in §5, found
0 mismatches.

### 3.1 Mutation testing the conformance harness

A harness that reports "0 mismatches" proves little unless it would have
reported the bugs a port actually gets. Each mutant below is a realistic
porting mistake, injected alone into the Rust code; 1,500 fuzz cases were
then run against it.

| Injected mistake | Mismatching cases |
| --- | --- |
| `strip()` via Rust `trim()` (misses U+001C–U+001F) | 441 |
| `len()` counts bytes | 1175 |
| `json.dumps` with compact separators | 1357 |
| float repr switches to exponent at 1e15 instead of 1e16 | 39 |
| `truthy(0.0) == true` | 48 |
| `cache_control` kept in unknown-block digests | 30 |
| `tool_result.is_error` ignored in digests | 90 |
| `result_max_chars` off by one (`<` for `<=`) | 5 |
| Responses dialect loses its turn grouping | 108 |
| system messages not trimmed from the head | 56 |
| strict ladder skips rung 3 | 29 |
| truncation separator overhead 8 instead of 9 | 1 |
| images priced by base64 length | 357 |
| base64 decoder doesn't stop at padding | 92 |
| mirrored `IndexError` "fixed" | 7 |

**15 of 15 caught.**

The first round missed four of these (float layout at 1e15–1e16, `cache_control`
on unknown blocks, results exactly at `result_max_chars`, the truncation
separator overhead). Each was a generator blind spot, fixed by adding
boundary values and *threshold sweeps* — one session replayed at consecutive
thresholds so the truncation budget walks across every summary-part
boundary instead of landing near one by chance.

## 4. Differences from the Python implementation

### Mirrored bugs (reproduced on purpose; fix in both, then regenerate goldens)

* A `data:` image URI without a comma raises `IndexError` in
  `_header_bytes`; the request is forwarded verbatim, uncompacted.
* `source` (Anthropic image/document) or `image_url`/`function` (Chat) that is
  truthy but not an object raises `AttributeError`; same consequence.
* `_compact_chain`: when `apply()` fails after earlier generations
  succeeded, it returns `True` without adopting or storing them.
* Reactive ladder: the final 400 can carry the last replay's body with the
  original response's headers.

### Deliberate divergences

* **Inputs serde_json rejects but Python accepts** — lone UTF-16 surrogate
  escapes (`"\ud800"`), `NaN`/`Infinity` literals, UTF-16/32-encoded bodies.
  These fail open: forwarded verbatim, not compacted. (A UTF-8 BOM is
  accepted, as in Python.)
* `repr()` of non-string tool-result content treats *unassigned* code points
  as printable (CPython consults the full Unicode database); assigned
  non-printable characters are handled.
* Duplicate request/response headers are all forwarded (the Python dict
  comprehension kept only the last, dropping e.g. repeated `set-cookie`).
* The request path is forwarded as received (Starlette forwarded the
  percent-decoded path).
* `cliff run`: Ctrl-C goes to the wrapped command only. Python's
  `subprocess.run` killed the agent 0.25 s after the first Ctrl-C, which
  agents use to cancel a step. A command killed by signal N exits 128+N.
* `/__cliff__/status` adds `"implementation": "rust"`, which the Rust
  `cliff status` prints; `code_stale` compares the binary's mtime instead of
  the package sources'.
* Upstream HTTP: HTTP/1.1 like httpx's default; connect timeout 30 s and
  per-read timeout 600 s (no separate write/pool timeouts).
* `test_timing_logged_at_debug` is not ported; the same debug lines are
  emitted.

## 5. Results

Measured on 4-core aarch64 with a 2-CPU cgroup quota; workload from
`scripts/bench_data.py` (Claude Code–shaped sessions: thinking, text,
tool calls, log-normal tool results).

**Per-request CPU cost** (`scripts/bench_prepare.py` vs
`examples/bench_prepare.rs`; parse + prepare + serialize, one 0.53 MB body
of 301 messages, median of 15):

| path | Python | Rust, 1 thread | Rust, 2 threads |
| --- | --- | --- | --- |
| under threshold (parse, digest, estimate) | 17.2 ms | 3.5 ms | 2.9 ms |
| stored-prefix match (substitute, re-serialize) | 14.1 ms | 3.3 ms | 2.5 ms |
| cold compaction (replay from empty store) | 27.6 ms | 5.1 ms | 4.1 ms |

Two profile-driven fixes got Rust there from 7–10 ms: SHA-256 had been the
portable software implementation (`sha2`'s `asm` feature enables the ARMv8 /
x86 SHA instructions, which Python gets from OpenSSL), and the JSON writer
scanned strings per `char` where a byte scan is exact (everything CPython
escapes is ASCII).

**Whole proxy under parallel agents** (`examples/loadgen.rs`: 16 sessions ×
150 growing requests = 2,400 requests, 285 MB, threshold 50k, against an
in-process mock upstream):

| | Python | Rust |
| --- | --- | --- |
| wall time | 31.1 s | 4.4 s |
| request latency p50 / p99 / max | 151 / 881 / 1528 ms | 24 / 73 / 94 ms |
| bytes forwarded upstream | identical digest | identical digest |

**Is the tokio/rayon split pulling its weight?** Same binary, CPU work run
inline on the tokio workers instead (ablation), with a cheap
`GET /__cliff__/status` every 20 ms during the load:

| | rayon offload (shipped) | inline on tokio workers |
| --- | --- | --- |
| wall time (2 runs) | 4.40 / 4.45 s | 4.61 / 4.69 s |
| request p50 | 24 ms | 30 ms |
| status ping p50 / p99 | **1.1** / 34 ms | 13–15 / 35 ms |

Keeping CPU work off the IO threads keeps cheap requests responsive while
heavy ones are being compacted (median 1.1 ms instead of ~14 ms). The p99 is
set by the 2-CPU quota's throttling either way.

## 6. Remaining steps (decisions for the maintainers)

1. **Soak with real traffic.** Run the Rust daemon for real sessions
   (`cargo install --path crates/cliff && cliff enable`; roll back with the
   Python `cliff enable`). For a no-risk check first, record real requests
   with `--debug-dir` and replay them through both implementations; the
   conformance driver already compares whole sessions.
2. **Ship the binary through PyPI** the way Ruff and uv do: switch
   `pyproject.toml` to `build-backend = "maturin"` with
   `[tool.maturin] bindings = "bin"` and `manifest-path =
   "crates/cliff/Cargo.toml"`, and build platform wheels with
   [maturin-action](https://github.com/PyO3/maturin-action). `uv tool install
   cliffcompaction` and `pip install` keep working and install the `cliff`
   binary. This replaces the Python package on PyPI, so it is a release
   decision, not part of this change.
3. **Retire the Python implementation** after a release cycle, keeping
   `scripts/conformance.py` and the golden corpus as the regression suite.
4. **Then fix the mirrored bugs and simplify** `cliff_core::py` on purpose
   (e.g. send compact JSON upstream), regenerating the goldens with each
   intended behavior change.

## Commands

```bash
cargo test                                           # Rust tests incl. golden parity
uv run pytest                                        # Python tests (the oracle)
uv run python scripts/conformance.py fuzz --n 3000   # differential fuzzing
uv run python scripts/conformance.py golden          # regenerate the golden corpus
cargo build --release && ./target/release/cliff serve --port 8257
python3 scripts/bench_data.py bench.json             # benchmark workload
uv run python scripts/bench_prepare.py bench.json
cargo run --release -p cliff-core --example bench_prepare -- bench.json
```
