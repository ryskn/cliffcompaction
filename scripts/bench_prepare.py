"""Per-request CPU cost of the Python engine: json.loads + prepare + dumps.

    uv run python scripts/bench_prepare.py bench.json
Scenarios (session 0's full history as one request body):
    pass     under threshold: parse, digest every message, estimate
    match    a stored prefix hit: substitute, re-estimate, re-serialize
    compact  cold store, over threshold: replay compaction from scratch
"""
import json, statistics, sys, time
sys.path.insert(0, "src")
from cliffcompaction.config import Config
from cliffcompaction.dialects.anthropic import DIALECT
from cliffcompaction.engine import Engine
import logging; logging.getLogger("cliffcompaction").disabled = True

data = json.load(open(sys.argv[1]))
s = data["sessions"][0]
raw = json.dumps({"model": "claude-sonnet-5", "max_tokens": 32000, "system": s["system"], "messages": s["history"]}).encode()

def one(engine):
    t = time.perf_counter()
    ctx = engine.prepare(json.loads(raw), DIALECT)
    if ctx.modified:
        json.dumps(ctx.outgoing_body(), ensure_ascii=False).encode()
    return (time.perf_counter() - t) * 1000, ctx

def run(name, make, n=15):
    times = []
    for _ in range(n):
        engine = make()
        ms, ctx = one(engine)
        times.append(ms)
    print(f"{name:8s} median {statistics.median(times):8.2f} ms   ({len(raw)/1e6:.2f} MB body, {len(s['history'])} msgs, modified={ctx.modified} compacted={ctx.compacted})")

run("pass", lambda: Engine(Config(threshold_tokens=10**9)))
warm = Engine(Config(threshold_tokens=50_000)); one(warm)
run("match", lambda: warm)
run("compact", lambda: Engine(Config(threshold_tokens=50_000)))
