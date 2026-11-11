//! Fixture builders: realistic Anthropic / OpenAI agent message lists
//! (ports of tests/util.py).
#![allow(dead_code)]

use serde_json::{json, Value};

// --- Anthropic Messages ------------------------------------------------------

pub fn a_user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

/// Assistant message with optional (tool_use_id, name, input).
pub fn a_assistant(text: &str, tool: Option<(&str, &str, Value)>) -> Value {
    let mut content = vec![];
    if !text.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }
    if let Some((id, name, input)) = tool {
        content.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
    }
    json!({"role": "assistant", "content": content})
}

pub fn a_result(tool_use_id: &str, text: &str) -> Value {
    json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": text}]})
}

/// task + n_turns of (assistant tool call, tool result). Even turns get a
/// long observation, odd turns a short one.
pub fn a_session(n_turns: usize, long_result_chars: usize) -> Vec<Value> {
    let mut msgs = vec![a_user("Fix the failing test in repo X.")];
    for i in 0..n_turns {
        let tid = format!("tu_{i}");
        msgs.push(a_assistant(
            &format!("Step {i}: I will inspect module {i} to find the bug."),
            Some((&tid, "bash", json!({"command": format!("pytest tests/test_{i}.py -x")}))),
        ));
        if i % 2 == 0 {
            msgs.push(a_result(&tid, &"X".repeat(long_result_chars)));
        } else {
            msgs.push(a_result(&tid, &format!("test_{i} passed (short output)")));
        }
    }
    msgs
}

pub fn a_body(msgs: Vec<Value>) -> Value {
    json!({"model": "claude-sonnet-5", "max_tokens": 4096, "system": "You are a coding agent.", "messages": msgs})
}

// --- OpenAI Chat Completions -------------------------------------------------

pub fn o_assistant(text: &str, tool: Option<(&str, &str, Value)>) -> Value {
    let mut msg = json!({"role": "assistant", "content": text});
    if let Some((id, name, args)) = tool {
        msg["tool_calls"] = json!([{"id": id, "type": "function", "function": {"name": name, "arguments": args.to_string()}}]);
    }
    msg
}

pub fn o_tool(tool_call_id: &str, text: &str) -> Value {
    json!({"role": "tool", "tool_call_id": tool_call_id, "content": text})
}

pub fn o_session(n_turns: usize, long_result_chars: usize) -> Vec<Value> {
    let mut msgs = vec![
        json!({"role": "system", "content": "You are a coding agent."}),
        json!({"role": "user", "content": "Fix the failing test."}),
    ];
    for i in 0..n_turns {
        let tid = format!("call_{i}");
        msgs.push(o_assistant(
            &format!("Step {i}: inspecting module {i}."),
            Some((&tid, "bash", json!({"command": format!("pytest tests/test_{i}.py -x")}))),
        ));
        if i % 2 == 0 {
            msgs.push(o_tool(&tid, &"Y".repeat(long_result_chars)));
        } else {
            msgs.push(o_tool(&tid, &format!("test_{i} passed")));
        }
    }
    msgs
}

// --- fat sessions (test_escalation.py) -----------------------------------------

pub fn fat_session(n_turns: usize, result_chars: usize) -> Vec<Value> {
    let mut msgs = vec![json!({"role": "user", "content": "the task: fix the bug"})];
    for i in 0..n_turns {
        msgs.push(json!({"role": "assistant", "content": [
            {"type": "text", "text": format!("thought {i}: {}", "y".repeat(2000))},
            {"type": "tool_use", "id": format!("t{i}"), "name": "bash", "input": {"command": format!("make {i}")}},
        ]}));
        msgs.push(json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": format!("t{i}"), "content": "R".repeat(result_chars)},
        ]}));
    }
    msgs
}

// --- OpenAI Responses (test_responses_dialect.py) --------------------------------

pub fn r_dev(text: &str) -> Value {
    json!({"type": "message", "role": "developer", "content": [{"type": "input_text", "text": text}]})
}

pub fn r_user(text: &str) -> Value {
    json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
}

pub fn r_assistant(text: &str) -> Value {
    json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
}

pub fn reasoning(i: usize, summary_text: Option<&str>) -> Value {
    let mut item = json!({"type": "reasoning", "id": format!("rs_{i}"), "encrypted_content": format!("enc-{i}").repeat(10)});
    if let Some(text) = summary_text {
        item["summary"] = json!([{"type": "summary_text", "text": text}]);
    }
    item
}

pub fn fcall(i: usize, cmd: &str) -> Value {
    json!({
        "type": "function_call", "id": format!("fc_{i}"), "call_id": format!("call_{i}"),
        "name": "exec_command", "arguments": json!({"cmd": cmd}).to_string(), "status": "completed",
    })
}

pub fn fout(i: usize, output: &str) -> Value {
    json!({"type": "function_call_output", "call_id": format!("call_{i}"), "output": output})
}

/// Codex-shaped history: head (developer + env + task), then n_steps of
/// [reasoning, function_call, function_call_output].
pub fn codex_session(n_steps: usize) -> Vec<Value> {
    let mut items = vec![
        r_dev("<skills_instructions>be a good agent</skills_instructions>"),
        r_user("<environment_context><cwd>/tmp</cwd></environment_context>"),
        r_user("Task: run the build. Codeword: PELICAN."),
    ];
    for i in 0..n_steps {
        items.push(reasoning(i, Some(&format!("planning step {i}"))));
        items.push(fcall(i, &format!("make step{i}")));
        let out = if i % 2 == 0 { format!("LOG {}", "x".repeat(2000)) } else { format!("step{i} ok") };
        items.push(fout(i, &out));
    }
    items
}

pub fn r_body(items: Vec<Value>) -> Value {
    json!({
        "model": "gpt-5",
        "instructions": format!("You are a coding agent.{}", "x".repeat(200)),
        "input": items, "store": false, "stream": true,
    })
}

pub fn content_str(m: &Value) -> &str {
    m["content"].as_str().unwrap_or("")
}

pub fn is_summary(m: &Value) -> bool {
    m["role"] == "user" && content_str(m).starts_with(cliff_core::SUMMARY_HEADER)
}
