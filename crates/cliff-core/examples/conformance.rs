//! Conformance driver for `scripts/conformance.py fuzz`: JSONL cases on
//! stdin, JSONL results on stdout. Cases run in parallel (each is
//! independent); output order matches input order.

use std::io::{self, BufRead, Write};

use rayon::prelude::*;
use serde_json::Value;

fn main() {
    let cases: Vec<Value> = io::stdin()
        .lock()
        .lines()
        .map(|l| serde_json::from_str(&l.expect("stdin")).expect("case is JSON"))
        .collect();
    let results: Vec<String> = cases
        .par_iter()
        .map(|c| serde_json::to_string(&cliff_core::conformance::run_case(c)).unwrap())
        .collect();
    let mut out = io::BufWriter::new(io::stdout().lock());
    for r in results {
        writeln!(out, "{r}").unwrap();
    }
}
