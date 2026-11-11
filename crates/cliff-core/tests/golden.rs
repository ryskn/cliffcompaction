//! Parity with the Python implementation, without Python.
//!
//! `tests/golden/cases.jsonl` holds generated inputs and `expected.jsonl`
//! what the Python oracle computed for them (`scripts/conformance.py
//! golden`). Every field must match: any difference is a porting bug or an
//! undocumented divergence.

use std::fs;
use std::path::Path;

use serde_json::Value;

fn read_jsonl(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .split('\n') // not lines(): JSON text may hold U+2028 etc. literally
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON line"))
        .collect()
}

fn diff(want: &Value, got: &Value, path: &str, out: &mut Vec<String>) {
    match (want, got) {
        (Value::Object(w), Value::Object(g)) => {
            let mut keys: Vec<&String> = w.keys().chain(g.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                diff(w.get(k).unwrap_or(&Value::Null), g.get(k).unwrap_or(&Value::Null), &format!("{path}.{k}"), out);
            }
        }
        (Value::Array(w), Value::Array(g)) if w.len() == g.len() => {
            for (i, (a, b)) in w.iter().zip(g).enumerate() {
                diff(a, b, &format!("{path}[{i}]"), out);
            }
        }
        _ if want != got => out.push(format!("{path}: python={want} rust={got}")),
        _ => {}
    }
}

#[test]
fn matches_the_python_oracle() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let cases = read_jsonl(&dir.join("cases.jsonl"));
    let expected = read_jsonl(&dir.join("expected.jsonl"));
    assert_eq!(cases.len(), expected.len());
    assert!(cases.len() >= 100, "golden corpus unexpectedly small");

    let mut failures = vec![];
    for (case, want) in cases.iter().zip(&expected) {
        let got = cliff_core::conformance::run_case(case);
        let mut problems = vec![];
        diff(want, &got, "", &mut problems);
        if !problems.is_empty() {
            failures.push(format!("case {} ({}):\n    {}", case["id"], case["kind"], problems[..problems.len().min(5)].join("\n    ")));
        }
    }
    assert!(failures.is_empty(), "{} of {} cases diverge from Python:\n{}", failures.len(), cases.len(), failures[..failures.len().min(10)].join("\n"));
}
