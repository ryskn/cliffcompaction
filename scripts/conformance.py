"""Cross-implementation conformance suite: the Python package is the oracle.

The Rust port (crates/) must reproduce the Python implementation byte for
byte. This script generates cases, runs them through the Python code, and
either writes golden files for `cargo test` or diffs live against the Rust
driver (`cargo run --release -p cliff-core --example conformance`).

    uv run python scripts/conformance.py golden            # refresh tests/golden/
    uv run python scripts/conformance.py fuzz --n 3000     # differential fuzz vs Rust
    uv run python scripts/conformance.py fuzz --n 500 --scale large --seed 7

Case kinds (one JSON object per line):

    value    a JSON value: canonical/dumps/repr/truthiness (the py layer)
    payload  an image payload string: token estimate
    message  one message through every dialect operation
    compact  one compact() call
    session  a sequence of requests sharing an Engine (and store), with an
             optional reactive ladder after each

Results carry digests of outputs, not outputs, so golden files stay small
while still demanding byte-identical bodies.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import logging
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cliffcompaction.cliff import compact  # noqa: E402
from cliffcompaction.config import Config  # noqa: E402
from cliffcompaction.dialects import anthropic, openai_chat, openai_responses  # noqa: E402
from cliffcompaction.dialects.base import SUMMARY_HEADER  # noqa: E402
from cliffcompaction.engine import Engine, billable_chars  # noqa: E402
from cliffcompaction.hashing import canonical_json  # noqa: E402
from cliffcompaction.images import tokens_for_payload  # noqa: E402
from cliffcompaction.proxy import is_side_call, session_id  # noqa: E402
from cliffcompaction.store import PrefixStore  # noqa: E402

logging.getLogger("cliffcompaction").disabled = True

DIALECTS = {d.name: d for d in (anthropic.DIALECT, openai_chat.DIALECT, openai_responses.DIALECT)}
ERROR = "<error>"
GOLDEN = ROOT / "crates" / "cliff-core" / "tests" / "golden"

# =============================================================================
# oracle
# =============================================================================


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def guard(fn):
    try:
        return fn()
    except Exception:
        return ERROR


def make_cfg(over: dict | None) -> Config:
    return Config(**(over or {}))


def is_probe(body: dict) -> bool:
    # Verbatim from proxy.create_app (a closure there, so not importable).
    for key in ("max_tokens", "max_output_tokens", "max_completion_tokens"):
        cap = body.get(key)
        if isinstance(cap, int) and cap <= 1:
            return True
    return False


def snapshot(ctx) -> dict:
    out = json.dumps(ctx.outgoing_body(), ensure_ascii=False)
    return {
        "est_out": ctx.est_tokens_out,
        "modified": ctx.modified,
        "compacted": ctx.compacted,
        "rung": ctx.rung,
        "base_cut": ctx.base_cut,
        "base_head": ctx.base_head,
        "steps": ctx.chain_steps,
        "fp": ctx.summary_fp,
        "out_msgs": ctx.out_msgs,
        "out": sha(out),
        "out_len": len(out),
    }


def oracle(case: dict) -> dict:
    kind = case["kind"]
    if kind == "value":
        if "value_json" in case:
            try:
                v = json.loads(case["value_json"])
            except ValueError:
                return {"id": case["id"], "parse_error": True}
        else:
            v = case["value"]
        out = {
            "canonical": canonical_json(v),
            "dumps": json.dumps(v, ensure_ascii=False),
            "dumps_ascii": json.dumps(v),
            "dumps_len": len(json.dumps(v, ensure_ascii=False)),
            "repr": repr(v),
            "truthy": bool(v),
            "probe_cap": isinstance(v, int) and v <= 1,
        }
    elif kind == "payload":
        out = {"tokens": guard(lambda: tokens_for_payload(case["payload"]))}
    elif kind == "message":
        d, cfg, m = DIALECTS[case["dialect"]], make_cfg(case.get("config")), case["message"]
        out = {
            "digest": guard(lambda: d.digest_message(m)),
            "is_assistant": guard(lambda: d.is_assistant(m)),
            "is_summary": guard(lambda: d.is_summary_message(m)),
            "summary": guard(lambda: d.summarize_message(m, cfg)),
            "billable": guard(lambda: billable_chars(m)),
        }
    elif kind == "compact":
        d, cfg = DIALECTS[case["dialect"]], make_cfg(case.get("config"))
        try:
            res = compact(case["messages"], d, cfg)
            result = None if res is None else {
                "head_len": res.head_len,
                "cut": res.cut,
                "summary": json.dumps(res.summary, ensure_ascii=False),
            }
        except Exception:
            result = ERROR
        out = {"result": result}
    elif kind == "session":
        out = oracle_session(case)
    else:
        raise ValueError(kind)
    out["id"] = case["id"]
    return out


def oracle_session(case: dict) -> dict:
    d = DIALECTS[case["dialect"]]
    cfg = make_cfg(case.get("config"))
    store = PrefixStore(cfg.store_max_entries, cfg.store_max_bytes)
    eng = Engine(cfg, store=store)
    results = []
    for req in case["requests"]:
        body = copy.deepcopy(case["body"])
        body[d.messages_key] = copy.deepcopy(case["history"][: req["take"]] + req.get("extra", []))
        try:
            ctx = eng.prepare(body, d)
        except Exception:
            results.append({"error": True})
            continue
        r = snapshot(ctx)
        r["chain"] = sha("\n".join(ctx.chain))
        r["est_in"] = ctx.est_tokens_in
        r["over_budget"] = ctx.over_budget
        r["sid"] = session_id(ctx, d)
        r["side_call"] = is_side_call(ctx.msgs, d, False)
        r["probe"] = is_probe(ctx.body)
        if case.get("reactive"):
            steps = []
            for _ in range(5):
                try:
                    ret = eng.reactive(ctx)
                except Exception:
                    steps.append({"ret": ERROR})
                    break
                s = snapshot(ctx)
                s["ret"] = ret
                steps.append(s)
                if not ret:
                    break
            r["reactive"] = steps
        results.append(r)
    return {"requests": results, "store_len": len(store), "store_bytes": store.nbytes}


# =============================================================================
# generator
# =============================================================================

WORDS = ("fix the bug in module parse step run test pytest cargo build error trace "
         "the a of to and assert value None True return import def class self").split()
UNICODE = ["é", "日本語", "😀", "Ünïcödé", " ", "\xa0", "　", "​",
           "\x1c", "\x1f", "\x7f", "\x00", "\x08", "\x0c", "ß", "Ω", "🧪🔥"]
SPECIAL = ['"', "'", "\\", "\n", "\t", "\r", "  ", "/", "<b>", "{", "}", "[", "]"]
FLOATS = [0.5, 0.1, 1e-7, 1.5e16, 1e22, -0.0, 123.456, 1e5, 2.0, 1e-5, 1e-4, 3.14159, -2.5e-300,
          # float repr layout boundaries (fixed below 1e16, exponent from 1e16 / below 1e-4)
          1e15, 1e16, 9999999999999998.0, 123456789012345.67, 0.00011, 9.5e-5]


class Gen:
    def __init__(self, seed: int, scale: str):
        self.r = random.Random(seed)
        self.big = scale == "large"
        # "golden": short histories, so the committed corpus stays small.
        self.turns = {"golden": 10, "small": 30, "large": 60}[scale]
        self.weird = 0.004  # per-message chance of a shape Python raises on

    # --- primitives ---------------------------------------------------------

    def chance(self, p: float) -> bool:
        return self.r.random() < p

    def text(self, lo: int = 0, hi: int = 60) -> str:
        r = self.r
        n = r.randint(lo, hi)
        out = []
        for _ in range(n):
            x = r.random()
            if x < 0.75:
                out.append(r.choice(WORDS) + " ")
            elif x < 0.88:
                out.append(r.choice(UNICODE))
            else:
                out.append(r.choice(SPECIAL))
        s = "".join(out)
        if self.chance(0.04):
            s = " \t\n\x1c " if self.chance(0.5) else ""
        if self.chance(0.03):
            s = SUMMARY_HEADER + "\n\n" + s
        if self.chance(0.04):
            note = "<task-notification>\n<task-id>x</task-id>" + "R" * r.randint(0, 300) + "</task-notification>"
            s = (s + note + r.choice(["\n", " \x1c\n", "", "　"]) + self.text(0, 5)) if self.chance(0.6) else note
        return s

    def blob(self, lo: int, hi: int) -> str:
        if self.chance(0.15):
            # Exactly at a result_max_chars boundary (default 500, presets
            # 50/200/2000), with no whitespace for strip() to shave off.
            edge = self.r.choice([50, 200, 500, 2000])
            return "x" * (edge + self.r.choice([-1, 0, 1]))
        if self.big:
            lo, hi = lo * 6, hi * 6
        elif self.turns <= 10:
            hi = min(hi, 600)
        n = self.r.randint(lo, hi)
        base = self.text(1, 8) or "x"
        return (base * (n // max(len(base), 1) + 1))[:n]

    def number(self):
        r = self.r
        x = r.random()
        if x < 0.4:
            return r.randint(-5, 5000)
        if x < 0.5:
            return r.choice([2**70, -(2**65), 0, 1, -1])
        return r.choice(FLOATS)

    def value(self, depth: int = 0):
        r = self.r
        x = r.random()
        if depth >= 3 or x < 0.35:
            return self.text(0, 6)
        if x < 0.55:
            return self.number()
        if x < 0.62:
            return r.choice([True, False, None])
        if x < 0.8:
            return [self.value(depth + 1) for _ in range(r.randint(0, 4))]
        return {self.key(): self.value(depth + 1) for _ in range(r.randint(0, 4))}

    def key(self) -> str:
        return self.r.choice(["command", "path", "a", "b", "z", "é", "Z", "_x", "10", "2", "cmd", "args"])

    def ident(self, prefix: str) -> str:
        return f"{prefix}_{self.r.randint(0, 99999)}"

    # --- images -------------------------------------------------------------

    def image_bytes(self) -> bytes:
        r = self.r
        w, h = r.choice([0, 1, 28, 29, 200, 1092, 1536, 3840, 8000, 65535]), r.choice([0, 1, 27, 200, 1024, 2160])
        kind = r.choice(["png", "jpeg", "gif", "webp8x", "webp8", "webp8l", "junk"])
        if kind == "png":
            return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h) + b"\x08\x06\x00\x00\x00"
        if kind == "jpeg":
            exif = b"\xff\xe1" + struct.pack(">H", 2 + 40) + b"\x00" * 40
            sof = b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", h, w)
            return b"\xff\xd8" + exif + sof + b"\x00" * 12
        if kind == "gif":
            return b"GIF89a" + struct.pack("<HH", w, h) + b"\x00" * 8
        if kind == "webp8x":
            return (b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8X" + b"\x00" * 8
                    + max(w - 1, 0).to_bytes(3, "little") + max(h - 1, 0).to_bytes(3, "little"))
        if kind == "webp8":
            return (b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8 " + b"\x00" * 10
                    + (w & 0x3FFF).to_bytes(2, "little") + (h & 0x3FFF).to_bytes(2, "little"))
        if kind == "webp8l":
            bits = (max(w - 1, 0) & 0x3FFF) | ((max(h - 1, 0) & 0x3FFF) << 14)
            return b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8L" + b"\x00" * 5 + bits.to_bytes(4, "little") + b"\x00" * 5
        return bytes(r.randint(0, 255) for _ in range(r.randint(0, 40)))

    def payload(self) -> str:
        r = self.r
        x = r.random()
        b64 = base64.b64encode(self.image_bytes()).decode()
        if x < 0.55:
            return "data:image/png;base64," + b64
        if x < 0.65:
            return "data:image/png;base64," + r.choice(["!!!!", "QU==QUJD", "AB!D", "QUJ", b64[:-1], b64 + "=", "é" + b64])
        if x < 0.75:
            return "https://example.com/" + self.ident("img") + ".png"
        if x < 0.85:
            return b64  # bare base64 (Anthropic `source.data`)
        if x < 0.855:
            return "data:image/png;base64"  # Python IndexError (fail-open)
        return "data:image/png," + b64  # not base64-flagged, still decoded

    # --- anthropic ------------------------------------------------------------

    def cache(self, block: dict) -> dict:
        if self.chance(0.1):
            block["cache_control"] = {"type": "ephemeral"}
        return block

    def a_text_block(self) -> dict:
        if self.chance(self.weird):
            return {"type": "text", "text": self.r.choice([5, None, ["x"], {"a": 1}])}
        return self.cache({"type": "text", "text": self.text(0, 40)})

    def a_image_block(self) -> dict:
        r = self.r
        kind = r.choice(["image", "document"])
        if self.chance(self.weird * 2):
            return {"type": kind, "source": r.choice(["str-source", ["x"], 7, None, ""])}
        src = {"type": r.choice(["base64", "url"])}
        if self.chance(0.8):
            src["data" if self.chance(0.7) else "url"] = self.payload()
        if self.chance(0.05):
            src["data"] = r.choice([5, None, ""])
        return {"type": kind, "source": src}

    def a_tool_result(self) -> dict:
        r = self.r
        x = r.random()
        if x < 0.6:
            content = self.blob(0, 900) if self.chance(0.5) else self.text(0, 30)
        elif x < 0.85:
            content = [self.r.choice([{"type": "text", "text": self.text(0, 20)}, self.text(0, 5), {"type": "image", "source": {"data": "AAAA"}}])
                       for _ in range(r.randint(0, 3))]
        elif x < 0.95:
            content = self.value(1)
        else:
            content = [{"type": "text", "text": 5}] if self.chance(self.weird * 10) else None
        block = {"type": "tool_result", "tool_use_id": self.ident("tu"), "content": content}
        if self.chance(0.2):
            block["is_error"] = r.choice([True, False, 0, 1, "", "yes", None])
        if self.chance(0.1):
            del block["content"]
        return self.cache(block)

    def a_user(self) -> dict:
        r = self.r
        if self.chance(0.3):
            return {"role": "user", "content": self.text(1, 40)}
        blocks = []
        for _ in range(r.randint(1, 3)):
            x = r.random()
            if x < 0.45:
                blocks.append(self.a_tool_result())
            elif x < 0.8:
                blocks.append(self.a_text_block())
            elif x < 0.95:
                blocks.append(self.a_image_block())
            else:
                blocks.append(r.choice(["bare string", 5, {"type": "mystery", "x": self.value(1)}]))
        return {"role": "user", "content": blocks}

    def a_assistant(self) -> dict:
        r = self.r
        if self.chance(0.15):
            return {"role": "assistant", "content": self.text(0, 50)}
        blocks = []
        for _ in range(r.randint(1, 4)):
            x = r.random()
            if x < 0.3:
                blocks.append(self.a_text_block())
            elif x < 0.45:
                b = {"type": "thinking", "thinking": self.text(0, 40), "signature": "sig" + self.ident("s")}
                if self.chance(self.weird):
                    b["thinking"] = None
                blocks.append(b)
            elif x < 0.5:
                blocks.append({"type": "redacted_thinking", "data": "enc" + self.ident("r")})
            elif x < 0.9:
                b = {"type": "tool_use", "id": self.ident("tu"), "name": r.choice(["bash", "edit", "read"]), "input": self.value(1) if self.chance(0.3) else {"command": self.text(1, 30)}}
                if self.chance(0.05):
                    del b["input"]
                if self.chance(0.03):
                    b["name"] = r.choice([None, 5, {"x": "y"}])
                blocks.append(self.cache(b))
            else:
                blocks.append(self.cache({"type": r.choice(["server_tool_use", "web_search_tool_result", "custom"]), "id": self.ident("x"), "payload": self.value(1)}))
        return {"role": "assistant", "content": blocks}

    def a_system(self) -> dict:
        r = self.r
        content = r.choice([self.text(1, 20), [], [{"type": "text", "text": self.text(0, 20)}], None])
        if self.chance(self.weird * 3):
            content = r.choice([5, True, {"a": 1}])
        return {"role": "system", "content": content}

    def anthropic_history(self) -> list:
        r = self.r
        msgs = [{"role": "user", "content": self.text(3, 30)}]
        if self.chance(0.3):
            msgs.append(self.a_system())
        for _ in range(r.randint(0, self.turns)):
            msgs.append(self.a_assistant())
            for _ in range(r.choice([1, 1, 1, 2])):
                msgs.append(self.a_user())
            if self.chance(0.06):
                msgs.append(self.a_system())
            if self.chance(0.03):
                msgs.append({"role": "user", "content": SUMMARY_HEADER + "\n\nold summary"})
        if self.chance(0.05):
            msgs.append({"role": r.choice(["tool", None, "assistant"]), "content": self.text(0, 10)})
        return msgs

    # --- openai chat ------------------------------------------------------------

    def o_content(self):
        r = self.r
        x = r.random()
        if x < 0.6:
            return self.text(0, 40)
        if x < 0.75:
            return None
        if x < 0.99:
            return [r.choice([{"type": "text", "text": self.text(0, 20)},
                              {"type": "image_url", "image_url": {"url": self.payload()}},
                              {"type": "input_audio", "x": 1}]) for _ in range(r.randint(0, 3))]
        return r.choice([5, {"a": "b"}, [{"type": "text", "text": 5}], [{"type": "image_url", "image_url": "str"}]])

    def o_assistant(self) -> dict:
        r = self.r
        m = {"role": "assistant", "content": self.o_content()}
        if self.chance(0.7):
            calls = []
            for _ in range(r.randint(1, 2)):
                args = json.dumps({"command": self.text(1, 20)}, ensure_ascii=r.random() < 0.5) if self.chance(0.8) else self.value(1)
                fn = {"name": r.choice(["bash", "edit"]), "arguments": args}
                if self.chance(0.02):
                    fn = r.choice([None, {}, "notadict", {"name": 5}])
                calls.append({"id": self.ident("call"), "type": "function", "function": fn})
            m["tool_calls"] = calls
        if self.chance(self.weird):
            m["tool_calls"] = r.choice(["str", 5, {"a": 1}, None])
        if self.chance(0.2):
            m[r.choice(["reasoning_content", "reasoning"])] = r.choice([self.text(0, 30), self.value(1)])
        return m

    def openai_history(self) -> list:
        r = self.r
        msgs = [{"role": "system", "content": self.text(2, 20)}, {"role": "user", "content": self.text(2, 20)}]
        for _ in range(r.randint(0, self.turns)):
            msgs.append(self.o_assistant())
            for _ in range(r.choice([1, 1, 2])):
                x = r.random()
                if x < 0.7:
                    msgs.append({"role": "tool", "tool_call_id": self.ident("call"), "content": self.blob(0, 900) if self.chance(0.5) else self.o_content()})
                elif x < 0.9:
                    msgs.append({"role": "user", "content": self.o_content()})
                else:
                    msgs.append({"role": r.choice(["developer", "system"]), "content": self.o_content(), "name": self.text(0, 2)})
        return msgs

    # --- openai responses ------------------------------------------------------------

    def r_content(self, role: str):
        r = self.r
        kind = "output_text" if role == "assistant" else "input_text"
        x = r.random()
        if x < 0.2:
            return self.text(0, 30)
        if x < 0.99:
            return [r.choice([{"type": kind, "text": self.text(0, 30)}, {"type": "input_image", "image_url": self.payload()}, {"type": "refusal", "refusal": "no"}])
                    for _ in range(r.randint(0, 2))]
        return r.choice([None, 5, [{"type": "input_text", "text": None}]])

    def r_message(self, role: str) -> dict:
        item = {"type": "message", "role": role, "content": self.r_content(role)}
        if self.chance(0.1):
            del item["type"]
        if self.chance(0.2):
            item["id"] = self.ident("msg")
            item["status"] = "completed"
        return item

    def r_step(self) -> list:
        r = self.r
        items = []
        if self.chance(0.7):
            rs = {"type": "reasoning", "id": self.ident("rs"), "encrypted_content": "enc" * r.randint(0, 5)}
            if self.chance(0.6):
                rs["summary"] = [r.choice([{"type": "summary_text", "text": self.text(0, 20)}, "x"]) for _ in range(r.randint(0, 2))]
            if self.chance(self.weird):
                rs["summary"] = r.choice([5, {"a": 1}, "str"])
            items.append(rs)
        x = r.random()
        if x < 0.6:
            t = r.choice(["function_call", "custom_tool_call", "local_shell_call", "web_search_call", "tool_search_call"])
            args = json.dumps({"cmd": self.text(1, 20)}) if self.chance(0.8) else self.value(1)
            call = {"type": t, "id": self.ident("fc"), "call_id": self.ident("call"), "name": r.choice(["exec_command", "apply_patch", None, ""]), "arguments": args, "status": "completed"}
            items.append(call)
            if self.chance(0.9):
                out = r.choice([self.blob(0, 900), self.text(0, 20), None, [{"type": "input_text", "text": self.text(0, 10)}], {"a": self.value(1)}, []])
                items.append({"type": "function_call_output", "call_id": call["call_id"], "output": out})
        elif x < 0.9:
            items.append(self.r_message("assistant"))
        else:
            items.append({"type": r.choice(["compaction", "item_reference", "mcp_call"]), "id": self.ident("x"), "data": self.value(1), "status": "x"})
        if self.chance(0.05):
            items.append(self.r_message("user"))
        if self.chance(self.weird):
            items.append({"type": r.choice([5, True, ["x"], {"a": 1}]), "role": "user"})
        if self.chance(0.02):
            items.append({"x": 1})  # no type, no role: "other"
        return items

    def responses_history(self) -> list:
        r = self.r
        items = [self.r_message("developer"), self.r_message("user"), self.r_message("user")]
        for _ in range(r.randint(0, self.turns)):
            items.extend(self.r_step())
        return items

    # --- cases ------------------------------------------------------------------------

    def config(self) -> dict:
        r = self.r
        if self.chance(0.15):
            # Tight budget: drives the ladder to rung 3 (summary truncation).
            return {"threshold_tokens": r.randint(60, 2500), "keep_recent": 1, "strict": self.chance(0.6),
                    "result_max_chars": r.choice([500, 2000])}
        cfg = {
            "threshold_tokens": r.choice([100, 300, 600, 1000, 2000, 4000, 10**6]) * (4 if self.big else 1),
            "keep_recent": r.choice([0, 1, 1, 2, 3, 3, 4]),
        }
        if self.chance(0.3):
            cfg["thought_max_chars"] = r.choice([0, 10, 50, 300, -1])
        if self.chance(0.2):
            cfg["cmd_max_chars"] = r.choice([0, 20, 150])
        if self.chance(0.3):
            cfg["result_max_chars"] = r.choice([0, 50, 200, 2000])
        if self.chance(0.2):
            cfg["human_max_chars"] = r.choice([10, 100])
        if self.chance(0.3):
            cfg["keep_thinking"] = False
        if self.chance(0.2):
            cfg["thinking_max_chars"] = r.choice([5, 30])
        if self.chance(0.25):
            cfg["strict"] = True
        if self.chance(0.15):
            cfg["store_max_bytes"] = r.choice([100, 2000])
        if self.chance(0.1):
            cfg["store_max_entries"] = r.choice([1, 2])
        return cfg

    def body(self, dialect: str) -> dict:
        r = self.r
        key = "input" if dialect == "openai-responses" else "messages"
        body = {"model": r.choice(["claude-sonnet-5", "gpt-5", "kimi"])}
        if dialect == "openai-responses":
            body["instructions"] = self.text(1, 20)
        elif dialect == "anthropic" and self.chance(0.7):
            body["system"] = self.text(1, 20) if self.chance(0.6) else [{"type": "text", "text": self.text(1, 10), "cache_control": {"type": "ephemeral"}}]
        body[key] = None
        cap_key = {"anthropic": "max_tokens", "openai": "max_completion_tokens", "openai-responses": "max_output_tokens"}[dialect]
        body[cap_key] = r.choice([4096, 4096, 1, 0, True, 1.0, -3])
        if self.chance(0.5):
            body["temperature"] = r.choice([0.7, 1, 0.0, 1e-3])
        if self.chance(0.3):
            body["tools"] = [{"name": "bash", "input_schema": {"type": "object"}}]
        if self.chance(0.3):
            if dialect == "anthropic":
                body["metadata"] = {"user_id": r.choice([json.dumps({"session_id": self.ident("sess"), "device_id": "d"}), "user-42", "{bad", "{}"])}
            elif dialect == "openai-responses":
                body[r.choice(["prompt_cache_key", "client_metadata"])] = r.choice([self.ident("thread"), {"thread_id": self.ident("t")}, ""])
        if self.chance(0.2):
            body["stream"] = True
        if self.chance(0.05):
            body["image_note"] = {"type": "image", "source": {"data": self.payload()}}
        return body

    def history(self, dialect: str) -> list:
        return {"anthropic": self.anthropic_history, "openai": self.openai_history, "openai-responses": self.responses_history}[dialect]()

    def session(self, cid: int) -> dict:
        r = self.r
        dialect = r.choice(list(DIALECTS))
        history = self.history(dialect)
        n = len(history)
        takes = sorted(r.randint(1, n) for _ in range(r.randint(1, 5)))
        requests = [{"take": t} for t in takes]
        if self.chance(0.25) and n > 2:  # a branch off an earlier prefix
            alt = self.history(dialect)[-r.randint(1, 3):]
            requests.append({"take": r.randint(1, n), "extra": alt})
        if self.chance(0.1):  # a rewritten history
            requests.append({"take": 0, "extra": self.history(dialect)})
        return {
            "id": cid, "kind": "session", "dialect": dialect, "config": self.config(),
            "body": self.body(dialect), "history": history, "requests": requests,
            "reactive": self.chance(0.4),
        }

    def message_case(self, cid: int) -> dict:
        dialect = self.r.choice(list(DIALECTS))
        msgs = self.history(dialect)
        return {"id": cid, "kind": "message", "dialect": dialect, "config": self.config(), "message": self.r.choice(msgs)}

    def compact_case(self, cid: int) -> dict:
        dialect = self.r.choice(list(DIALECTS))
        return {"id": cid, "kind": "compact", "dialect": dialect, "config": self.config(), "messages": self.history(dialect)}

    def value_case(self, cid: int) -> dict:
        if self.chance(0.15):
            raw = self.r.choice(["1e400", "-1e400", "1E5", "1.0e2", "-0", "1e-7", "[1.5e300, 2e-320]", "0.30000000000000004",
                                 "123456789012345678901234567890", '{"a":1,"a":2}', '"\\ud800"', "NaN", '"\\u00e9"'])
            return {"id": cid, "kind": "value", "value_json": raw}
        return {"id": cid, "kind": "value", "value": self.value()}

    def sweep(self, cid: int, width: int) -> list[dict]:
        """One short-turn session at consecutive thresholds, so the
        truncation budget (threshold*4 - fixed) walks across every summary
        part boundary instead of landing near one by chance."""
        r = self.r
        dialect = r.choice(["anthropic", "openai"])
        msgs = [{"role": "user", "content": "task: " + self.text(2, 6)}]
        for i in range(r.randint(12, 12 + self.turns)):
            if dialect == "anthropic":
                msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"step {i} " + self.text(1, 6)},
                             {"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {"cmd": f"make {i}"}}]})
                msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": self.text(1, 8)}]})
            else:
                msgs.append({"role": "assistant", "content": f"step {i} " + self.text(1, 6),
                             "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]})
                msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": self.text(1, 8)})
        body = self.body(dialect)
        size = len(json.dumps(msgs, ensure_ascii=False)) // 4
        start = r.randint(size // 6, size // 2)
        strict = self.chance(0.5)
        return [{"id": cid + k, "kind": "session", "dialect": dialect,
                 "config": {"threshold_tokens": start + k, "keep_recent": 1, "strict": strict},
                 "body": body, "history": msgs, "requests": [{"take": len(msgs)}], "reactive": not strict}
                for k in range(width)]

    def cases(self, n: int):
        cid = 0
        while cid < n:
            if self.chance(0.02):
                block = self.sweep(cid, min(12 if self.turns <= 10 else 40, n - cid))
                yield from block
                cid += len(block)
                continue
            yield from self.mixed(cid)
            cid += 1

    def mixed(self, cid: int):
        mix = [(self.session, 0.55), (self.message_case, 0.2), (self.compact_case, 0.1), (self.value_case, 0.1),
               (lambda cid: {"id": cid, "kind": "payload", "payload": self.payload()}, 0.05)]
        x, acc = self.r.random(), 0.0
        for make, p in mix:
            acc += p
            if x < acc:
                yield make(cid)
                return
        yield self.session(cid)


# =============================================================================
# commands
# =============================================================================

# Inputs Python accepts but serde_json rejects at parse time (the proxy then
# forwards them verbatim). Documented in docs/rust-migration.md.
KNOWN_PARSE_DIVERGENCES = {'"\\ud800"', "NaN"}


def dump_lines(objs) -> str:
    return "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs)


def generate(n: int, seed: int, scale: str) -> list[dict]:
    cases = list(Gen(seed, scale).cases(n))
    return [c for c in cases if c.get("value_json") not in KNOWN_PARSE_DIVERGENCES]


def run_rust(cases_text: str) -> list[dict]:
    cmd = ["cargo", "run", "-q", "--release", "-p", "cliff-core", "--example", "conformance"]
    proc = subprocess.run(cmd, input=cases_text.encode(), capture_output=True, cwd=ROOT)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode())
        raise SystemExit("rust driver failed")
    return [json.loads(line) for line in proc.stdout.decode().split("\n") if line]


def diff(case: dict, want, got, path="") -> list[str]:
    if isinstance(want, dict) and isinstance(got, dict):
        out = []
        for k in sorted(set(want) | set(got)):
            out += diff(case, want.get(k), got.get(k), f"{path}.{k}")
        return out
    if isinstance(want, list) and isinstance(got, list) and len(want) == len(got):
        out = []
        for i, (w, g) in enumerate(zip(want, got)):
            out += diff(case, w, g, f"{path}[{i}]")
        return out
    if want != got or type(want) is not type(got):
        return [f"{path}: python={want!r} rust={got!r}"]
    return []


def cmd_golden(args) -> int:
    cases = generate(args.n, args.seed, "golden")
    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "cases.jsonl").write_text(dump_lines(cases))
    (GOLDEN / "expected.jsonl").write_text(dump_lines(oracle(c) for c in cases))
    size = sum(p.stat().st_size for p in GOLDEN.iterdir())
    print(f"wrote {len(cases)} cases to {GOLDEN} ({size / 1e6:.2f} MB)")
    return 0


def cmd_fuzz(args) -> int:
    cases = generate(args.n, args.seed, args.scale)
    text = dump_lines(cases)
    want = [oracle(c) for c in cases]
    got = run_rust(text)
    bad = 0
    for case, w, g in zip(cases, want, got):
        problems = diff(case, w, g)
        if problems:
            bad += 1
            if bad <= args.show:
                print(f"--- case {case['id']} ({case['kind']}, {case.get('dialect', '')})")
                for p in problems[:8]:
                    print("   ", p)
                if args.dump:
                    Path(args.dump).mkdir(parents=True, exist_ok=True)
                    (Path(args.dump) / f"case-{case['id']}.json").write_text(json.dumps(case, ensure_ascii=False, indent=1))
    kinds = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    errors = sum(1 for w in want for r in w.get("requests", []) if r.get("error"))
    compacted = sum(1 for w in want for r in w.get("requests", []) if r.get("compacted"))
    print(f"{len(cases)} cases {kinds}: {bad} mismatching "
          f"(sessions: {compacted} compacted requests, {errors} fail-open errors)")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden")
    g.add_argument("--n", type=int, default=400)
    g.add_argument("--seed", type=int, default=20261109)
    f = sub.add_parser("fuzz")
    f.add_argument("--n", type=int, default=2000)
    f.add_argument("--seed", type=int, default=int.from_bytes(os.urandom(4), "little"))
    f.add_argument("--scale", choices=["golden", "small", "large"], default="small")
    f.add_argument("--show", type=int, default=10)
    f.add_argument("--dump", help="write mismatching cases here")
    args = p.parse_args()
    if args.cmd == "fuzz":
        print(f"seed {args.seed}")
    return {"golden": cmd_golden, "fuzz": cmd_fuzz}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
