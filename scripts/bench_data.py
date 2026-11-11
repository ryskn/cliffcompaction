"""Benchmark workload: parallel agent sessions growing turn by turn.

Writes a JSON file the Rust load generator (`cargo run --release -p
cliffcompaction --example loadgen`) and scripts/bench_prepare.py both read:

    {"sessions": [{"system": str, "history": [msg, ...]}, ...], "turns": N}

Request k of a session sends history[: 1 + 2k] — what a Claude Code-style
agent resends every step. Deterministic (seeded).
"""

from __future__ import annotations

import argparse
import json
import random

WORDS = ("the a module test fix error value return import def class self parse build "
         "cargo pytest assert none true false file line trace stack config path").split()


def text(r: random.Random, n: int) -> str:
    return " ".join(r.choice(WORDS) for _ in range(max(1, n // 6)))[:n]


def session(r: random.Random, sid: int, turns: int) -> dict:
    msgs = [{"role": "user", "content": f"[session {sid}] Fix the failing integration test in repo X. " + text(r, 400)}]
    for i in range(turns):
        tid = f"toolu_{sid}_{i}"
        msgs.append({"role": "assistant", "content": [
            {"type": "thinking", "thinking": text(r, r.randint(100, 900)), "signature": "sig" + "x" * 80},
            {"type": "text", "text": text(r, r.randint(50, 500))},
            {"type": "tool_use", "id": tid, "name": r.choice(["Bash", "Read", "Edit", "Grep"]),
             "input": {"command": text(r, r.randint(20, 120)), "timeout": 120000}},
        ]})
        size = int(min(40_000, r.lognormvariate(7.0, 1.3)))  # median ~1.1k chars, long tail
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": text(r, size), "is_error": r.random() < 0.05},
        ]})
    return {"system": "You are Claude Code. " + text(r, 12_000), "history": msgs}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=int, default=16)
    p.add_argument("--turns", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("out")
    a = p.parse_args()
    r = random.Random(a.seed)
    data = {"turns": a.turns, "sessions": [session(r, s, a.turns) for s in range(a.sessions)]}
    with open(a.out, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    full = sum(len(json.dumps(s["history"], ensure_ascii=False)) for s in data["sessions"]) / a.sessions
    print(f"wrote {a.out}: {a.sessions} sessions x {a.turns} turns, final history ~{full / 1e6:.2f} MB "
          f"(~{full / 4 / 1000:.0f}k est tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
