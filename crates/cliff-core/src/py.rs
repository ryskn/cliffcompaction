//! Python semantics, in one place.
//!
//! The port is behavior-for-behavior with the Python implementation, which
//! stays the test oracle until the migration is complete. Every spot where a
//! Python primitive and its obvious Rust counterpart disagree is handled
//! here, so the dialects and the engine can read like the Python they were
//! ported from:
//!
//! | Python                                 | naive Rust                 | here                 |
//! |----------------------------------------|----------------------------|----------------------|
//! | `len(s)` counts code points            | `s.len()` counts bytes     | [`len`]              |
//! | `s.strip()` uses `str.isspace`         | `trim()` misses `\x1c-\x1f` | [`strip`]           |
//! | `json.dumps` (`", "`, float `repr`)    | `serde_json::to_string`    | [`dumps`], [`Fmt`]   |
//! | truthiness, `isinstance(x, int)`       | —                          | [`truthy`], [`int_le_one`] |
//! | `str(obj)` / `repr(obj)`               | —                          | [`str_of`]           |
//! | exceptions caught by a fail-open path  | panics                     | [`PyErr`]            |
//!
//! Once the Python implementation is retired, this module is where the
//! deliberate simplifications go (e.g. sending compact JSON upstream).

use std::borrow::Cow;
use std::fmt;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Number, Value};
use sha2::{Digest, Sha256};

// --- exceptions --------------------------------------------------------------

/// A Python exception the reference implementation would have raised.
///
/// Every call site that Python wraps in `except Exception` (the proxy's
/// fail-open contract) receives one of these as an `Err` instead. Nothing in
/// the port panics on malformed input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyErr(pub String);

impl PyErr {
    pub fn type_error(msg: impl Into<String>) -> Self {
        PyErr(format!("TypeError: {}", msg.into()))
    }
    pub fn attribute_error(msg: impl Into<String>) -> Self {
        PyErr(format!("AttributeError: {}", msg.into()))
    }
    pub fn index_error(msg: impl Into<String>) -> Self {
        PyErr(format!("IndexError: {}", msg.into()))
    }
}

impl fmt::Display for PyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PyErr {}

pub type PyResult<T> = Result<T, PyErr>;

// --- strings -------------------------------------------------------------------

/// `len(s)`: Python counts code points, not UTF-8 bytes.
#[inline]
pub fn len(s: &str) -> usize {
    if s.is_ascii() { s.len() } else { s.chars().count() }
}

/// `str.isspace()` for one character. Rust's `char::is_whitespace` is the
/// Unicode `White_Space` property, which lacks the four information
/// separators U+001C..U+001F that CPython also treats as whitespace.
#[inline]
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// `s.strip()`.
#[inline]
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `s[:n]` by code points.
pub fn prefix_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `dialects.base.truncate`: ellipsis past `max_chars`; `<= 0` is unlimited.
pub fn truncate(text: &str, max_chars: i64) -> Cow<'_, str> {
    if max_chars > 0 {
        let n = max_chars as usize;
        if let Some((i, _)) = text.char_indices().nth(n) {
            let mut out = String::with_capacity(i + 3);
            out.push_str(&text[..i]);
            out.push_str("...");
            return Cow::Owned(out);
        }
    }
    Cow::Borrowed(text)
}

static TASK_NOTIFICATION_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Python's `\s` follows str.isspace(), hence the extra \x1c-\x1f.
    Regex::new(r"(?s)<task-notification>.*?</task-notification>[\s\x1c-\x1f]*").unwrap()
});

/// `dialects.base.strip_task_notifications`.
pub fn strip_task_notifications(text: &str) -> Cow<'_, str> {
    TASK_NOTIFICATION_RE.replace_all(text, "")
}

/// `sep.join(items)` over JSON values: a non-string item is the TypeError
/// Python's `str.join` raises.
pub fn join<'a>(sep: &str, items: impl IntoIterator<Item = &'a Value>) -> PyResult<String> {
    let mut out = String::new();
    for (i, v) in items.into_iter().enumerate() {
        let s = v
            .as_str()
            .ok_or_else(|| PyErr::type_error(format!("sequence item {i}: expected str instance")))?;
        if i > 0 {
            out.push_str(sep);
        }
        out.push_str(s);
    }
    Ok(out)
}

/// `x.strip()` where `x` came out of a request body: AttributeError unless
/// it is a string.
pub fn strip_value(v: &Value) -> PyResult<&str> {
    v.as_str()
        .map(strip)
        .ok_or_else(|| PyErr::attribute_error(format!("'{}' object has no attribute 'strip'", type_name(v))))
}

/// `x.startswith(prefix)` on a body value: AttributeError unless a string.
pub fn startswith_value(v: &Value, prefix: &str) -> PyResult<bool> {
    v.as_str()
        .map(|s| s.starts_with(prefix))
        .ok_or_else(|| {
            PyErr::attribute_error(format!("'{}' object has no attribute 'startswith'", type_name(v)))
        })
}

pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if is_int(n) => "int",
        Value::Number(_) => "float",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

// --- dict access -------------------------------------------------------------------

/// `d.get(key)` on a value known to be a dict: `None` when missing. A
/// present JSON `null` is `Some(Value::Null)`, which matters for
/// `d.get(key, default)`.
#[inline]
pub fn get<'a>(d: &'a Value, key: &str) -> Option<&'a Value> {
    d.as_object().and_then(|m| m.get(key))
}

/// `d.get(key) or fallback`.
#[inline]
pub fn get_truthy<'a>(d: &'a Value, key: &str) -> Option<&'a Value> {
    get(d, key).filter(|v| truthy(v))
}

/// `(d.get(key) or {})` followed by `.get(...)`: the AttributeError Python
/// raises when the value is truthy but not a dict.
pub fn get_dict_or_empty<'a>(d: &'a Value, key: &str) -> PyResult<Option<&'a Value>> {
    match get_truthy(d, key) {
        None => Ok(None),
        Some(v @ Value::Object(_)) => Ok(Some(v)),
        Some(v) => Err(PyErr::attribute_error(format!(
            "'{}' object has no attribute 'get'",
            type_name(v)
        ))),
    }
}

/// Items produced by `for x in (value or [])`: a list yields its elements,
/// a dict its keys, a string its characters. Only list elements can ever be
/// dicts, and every caller filters on `isinstance(x, dict)`, so keys and
/// characters are reported as "nothing". A truthy number or bool is the
/// TypeError Python raises for iterating it.
pub fn iter_or_empty<'a>(v: Option<&'a Value>) -> PyResult<&'a [Value]> {
    match v {
        None => Ok(&[]),
        Some(v) if !truthy(v) => Ok(&[]),
        Some(Value::Array(a)) => Ok(a),
        Some(Value::Object(_)) | Some(Value::String(_)) => Ok(&[]),
        Some(v) => Err(PyErr::type_error(format!("'{}' object is not iterable", type_name(v)))),
    }
}

// --- truthiness and ints ------------------------------------------------------------

/// Python truthiness of a JSON value.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.to_string().parse::<f64>().map_or(true, |f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Whether Python's json module would decode this number as an `int`.
pub fn is_int(n: &Number) -> bool {
    !n.to_string().contains(['.', 'e', 'E'])
}

/// `isinstance(v, int) and v <= 1` — bool is an int subclass in Python.
pub fn int_le_one(v: &Value) -> bool {
    match v {
        Value::Bool(_) => true,
        Value::Number(n) if is_int(n) => {
            let text = n.to_string();
            text.starts_with('-') || text == "0" || text == "1"
        }
        _ => false,
    }
}

/// `int(text)` for configuration values: surrounding whitespace, a sign and
/// single underscores between digits are accepted, as in Python.
pub fn parse_int(text: &str) -> Option<i64> {
    let t = strip(text);
    let (neg, digits) = match t.as_bytes().first()? {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return None;
    }
    let mut value: i64 = 0;
    for c in digits.chars() {
        if c == '_' {
            continue;
        }
        let d = c.to_digit(10)? as i64;
        value = value.checked_mul(10)?.checked_add(d)?;
    }
    Some(if neg { -value } else { value })
}

// --- hashing -----------------------------------------------------------------------

/// `hashlib.sha256(data).hexdigest()`.
pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0xf) as usize] as char);
    }
    out
}

// --- json.dumps ------------------------------------------------------------------------

/// Output of the serializer: a `String`, a code-point counter, or a hasher.
pub trait Sink {
    fn put(&mut self, s: &str);
}

impl Sink for String {
    #[inline]
    fn put(&mut self, s: &str) {
        self.push_str(s);
    }
}

/// Counts code points without materializing the text: `len(json.dumps(x))`.
#[derive(Default)]
pub struct CharCount(pub usize);

impl Sink for CharCount {
    #[inline]
    fn put(&mut self, s: &str) {
        self.0 += len(s);
    }
}

/// Feeds the UTF-8 bytes straight into SHA-256: `sha256(dumps(x).encode())`.
#[derive(Default)]
pub struct HashSink(pub Sha256);

impl Sink for HashSink {
    #[inline]
    fn put(&mut self, s: &str) {
        self.0.update(s.as_bytes());
    }
}

impl HashSink {
    pub fn hexdigest(self) -> String {
        hex(&self.0.finalize())
    }
}

/// `json.dumps` options that occur in the Python implementation.
#[derive(Clone, Copy, Debug)]
pub struct Fmt {
    pub sort_keys: bool,
    /// `separators=(",", ":")` instead of the default `(", ", ": ")`.
    pub compact: bool,
    pub ensure_ascii: bool,
    pub indent: Option<usize>,
}

/// `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
pub const CANONICAL: Fmt = Fmt { sort_keys: true, compact: true, ensure_ascii: false, indent: None };
/// `json.dumps(obj, ensure_ascii=False)`
pub const DUMPS_UNICODE: Fmt = Fmt { sort_keys: false, compact: false, ensure_ascii: false, indent: None };
/// `json.dumps(obj)`
pub const DUMPS_ASCII: Fmt = Fmt { sort_keys: false, compact: false, ensure_ascii: true, indent: None };

impl Fmt {
    fn item_sep(&self) -> &'static str {
        if self.compact || self.indent.is_some() { "," } else { ", " }
    }
    fn key_sep(&self) -> &'static str {
        if self.compact { ":" } else { ": " }
    }
}

/// Serialize like Python's `json.dumps` with the given options.
pub fn dumps(v: &Value, fmt: Fmt) -> String {
    let mut out = String::new();
    write_value(v, fmt, &mut out, 0);
    out
}

/// `len(json.dumps(v, ...))`, without building the string.
pub fn dumps_len(v: &Value, fmt: Fmt) -> usize {
    let mut n = CharCount::default();
    write_value(v, fmt, &mut n, 0);
    n.0
}

/// `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
pub fn canonical_json(v: &Value) -> String {
    dumps(v, CANONICAL)
}

/// `canonical_json({k: v for k, v in d.items() if k not in skip})`.
pub fn canonical_json_without(d: &Value, skip: &[&str]) -> String {
    let Some(map) = d.as_object() else {
        return canonical_json(d);
    };
    let mut entries: Vec<(&String, &Value)> = map.iter().filter(|(k, _)| !skip.contains(&k.as_str())).collect();
    if entries.is_empty() {
        return "{}".into();
    }
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::new();
    write_entries(entries.into_iter(), CANONICAL, &mut out, 0);
    out
}

/// A `""` to hand out where Python writes `d.get(key, "")`.
pub static EMPTY_STR: Value = Value::String(String::new());

/// `d.get(key, "")`.
#[inline]
pub fn get_or_empty<'a>(d: &'a Value, key: &str) -> &'a Value {
    get(d, key).unwrap_or(&EMPTY_STR)
}

fn newline<S: Sink + ?Sized>(fmt: Fmt, sink: &mut S, level: usize) {
    if let Some(indent) = fmt.indent {
        sink.put("\n");
        for _ in 0..indent * level {
            sink.put(" ");
        }
    }
}

pub fn write_value<S: Sink + ?Sized>(v: &Value, fmt: Fmt, sink: &mut S, level: usize) {
    match v {
        Value::Null => sink.put("null"),
        Value::Bool(true) => sink.put("true"),
        Value::Bool(false) => sink.put("false"),
        Value::Number(n) => sink.put(&json_number(n)),
        Value::String(s) => write_str(s, fmt.ensure_ascii, sink),
        Value::Array(items) => write_array(items.iter(), fmt, sink, level),
        Value::Object(map) => {
            if map.is_empty() {
                sink.put("{}");
            } else if fmt.sort_keys {
                let mut entries: Vec<(&String, &Value)> = map.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                write_entries(entries.into_iter(), fmt, sink, level);
            } else {
                write_entries(map.iter(), fmt, sink, level);
            }
        }
    }
}

fn write_entries<'a, S: Sink + ?Sized>(
    entries: impl Iterator<Item = (&'a String, &'a Value)>,
    fmt: Fmt,
    sink: &mut S,
    level: usize,
) {
    for (i, (key, value)) in entries.enumerate() {
        sink.put(if i == 0 { "{" } else { fmt.item_sep() });
        newline(fmt, sink, level + 1);
        write_str(key, fmt.ensure_ascii, sink);
        sink.put(fmt.key_sep());
        write_value(value, fmt, sink, level + 1);
    }
    newline(fmt, sink, level);
    sink.put("}");
}

/// A JSON array of borrowed items (the engine serializes substituted message
/// lists without cloning them).
pub fn write_array<'a, S: Sink + ?Sized>(
    items: impl ExactSizeIterator<Item = &'a Value>,
    fmt: Fmt,
    sink: &mut S,
    level: usize,
) {
    if items.len() == 0 {
        sink.put("[]");
        return;
    }
    sink.put("[");
    for (i, item) in items.enumerate() {
        if i > 0 {
            sink.put(fmt.item_sep());
        }
        newline(fmt, sink, level + 1);
        write_value(item, fmt, sink, level + 1);
    }
    newline(fmt, sink, level);
    sink.put("]");
}

/// A JSON object whose entries are produced by the caller. Each entry's
/// value is written by `write` so callers can splice in borrowed views.
pub fn write_object_with<'a, S: Sink + ?Sized, I, F>(entries: I, fmt: Fmt, sink: &mut S, level: usize, mut write: F)
where
    I: Iterator<Item = &'a str>,
    F: FnMut(&str, &mut S, usize),
{
    let mut any = false;
    for (i, key) in entries.enumerate() {
        sink.put(if i == 0 { "{" } else { fmt.item_sep() });
        any = true;
        newline(fmt, sink, level + 1);
        write_str(key, fmt.ensure_ascii, sink);
        sink.put(fmt.key_sep());
        write(key, sink, level + 1);
    }
    if !any {
        sink.put("{}");
        return;
    }
    newline(fmt, sink, level);
    sink.put("}");
}

/// Bytes CPython escapes with `ensure_ascii=False`: `"`, `\\` and C0
/// controls. All ASCII, so a byte scan is exact on UTF-8 (bytes of
/// multi-byte sequences are >= 0x80 and never match).
static NEEDS_ESCAPE: [bool; 256] = {
    let mut t = [false; 256];
    let mut b = 0;
    while b < 0x20 {
        t[b] = true;
        b += 1;
    }
    t[b'"' as usize] = true;
    t[b'\\' as usize] = true;
    t
};

fn write_escape<S: Sink + ?Sized>(c: char, sink: &mut S) {
    match c {
        '"' => sink.put("\\\""),
        '\\' => sink.put("\\\\"),
        '\n' => sink.put("\\n"),
        '\r' => sink.put("\\r"),
        '\t' => sink.put("\\t"),
        '\x08' => sink.put("\\b"),
        '\x0c' => sink.put("\\f"),
        c => {
            let mut buf = [0u16; 2];
            for unit in c.encode_utf16(&mut buf) {
                sink.put(&format!("\\u{:04x}", unit));
            }
        }
    }
}

/// A JSON string literal, escaped exactly like CPython's encoder.
pub fn write_str<S: Sink + ?Sized>(s: &str, ensure_ascii: bool, sink: &mut S) {
    sink.put("\"");
    let mut start = 0;
    if ensure_ascii {
        for (i, c) in s.char_indices() {
            let cp = c as u32;
            if cp >= 0x20 && cp <= 0x7e && c != '"' && c != '\\' {
                continue;
            }
            if start < i {
                sink.put(&s[start..i]);
            }
            start = i + c.len_utf8();
            write_escape(c, sink);
        }
    } else {
        for (i, &b) in s.as_bytes().iter().enumerate() {
            if !NEEDS_ESCAPE[b as usize] {
                continue;
            }
            if start < i {
                sink.put(&s[start..i]);
            }
            start = i + 1;
            write_escape(b as char, sink);
        }
    }
    if start < s.len() {
        sink.put(&s[start..]);
    }
    sink.put("\"");
}

/// A number as Python's json module writes it after `json.loads`: integer
/// literals stay exact (Python ints are unbounded), everything else goes
/// through `float` and comes back out as `float.__repr__`.
pub fn json_number(n: &Number) -> String {
    let text = n.to_string();
    if is_int(n) {
        return normalize_int(&text);
    }
    let f: f64 = text.parse().unwrap_or(f64::NAN);
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity".into() } else { "-Infinity".into() }
    } else {
        float_repr(f)
    }
}

fn normalize_int(text: &str) -> String {
    // JSON forbids leading zeros, so the only non-canonical int is -0.
    if text == "-0" { "0".into() } else { text.into() }
}

/// `float.__repr__` (shortest round-trip digits, CPython's layout rules).
pub fn float_repr(f: f64) -> String {
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    if !f.is_finite() {
        return if f.is_nan() { "nan".into() } else if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    // Rust's LowerExp prints the shortest digits that round-trip, e.g. "1.5e-7".
    let sci = format!("{:e}", f.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1;
    let mut out = String::new();
    if f < 0.0 {
        out.push('-');
    }
    if -4 < decpt && decpt <= 16 {
        if decpt <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-decpt) as usize));
            out.push_str(&digits);
        } else if decpt as usize >= digits.len() {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', decpt as usize - digits.len()));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..decpt as usize]);
            out.push('.');
            out.push_str(&digits[decpt as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push_str(&format!("e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs()));
    }
    out
}

// --- str() / repr() ------------------------------------------------------------------------

/// `str(v)` for a value out of a decoded body: a string is itself, anything
/// else is its `repr`.
pub fn str_of(v: &Value) -> Cow<'_, str> {
    match v {
        Value::String(s) => Cow::Borrowed(s),
        other => Cow::Owned(repr(other)),
    }
}

/// `repr(v)` of a decoded JSON value (None/True/False, int, float, str,
/// list, dict).
pub fn repr(v: &Value) -> String {
    let mut out = String::new();
    write_repr(v, &mut out);
    out
}

fn write_repr(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(n) => {
            let text = n.to_string();
            if is_int(n) {
                out.push_str(&normalize_int(&text));
            } else {
                out.push_str(&float_repr(text.parse().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => str_repr(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_repr(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                str_repr(k, out);
                out.push_str(": ");
                write_repr(item, out);
            }
            out.push('}');
        }
    }
}

/// `repr(s)` for a str: CPython's quote choice and escapes.
pub fn str_repr(s: &str, out: &mut String) {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    out.push(quote);
    for c in s.chars() {
        let cp = c as u32;
        match c {
            c if c == quote || c == '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if cp < 0x20 || cp == 0x7f => out.push_str(&format!("\\x{cp:02x}")),
            _ if cp < 0x7f => out.push(c),
            _ if is_printable(c) => out.push(c),
            _ if cp <= 0xff => out.push_str(&format!("\\x{cp:02x}")),
            _ if cp <= 0xffff => out.push_str(&format!("\\u{cp:04x}")),
            _ => out.push_str(&format!("\\U{cp:08x}")),
        }
    }
    out.push(quote);
}

/// `str.isprintable()` for one non-ASCII character.
///
/// CPython consults the full Unicode database (categories Cc, Cf, Cs, Co,
/// Cn, Zl, Zp and Zs other than space are unprintable). This covers every
/// assigned code point in those categories that a request body plausibly
/// carries; unassigned code points (Cn) are the documented gap — they only
/// matter inside `repr()` of a non-string tool result, which no client
/// sends.
fn is_printable(c: char) -> bool {
    let cp = c as u32;
    !matches!(cp,
        0x80..=0xa0 | 0xad
        | 0x600..=0x605 | 0x61c | 0x6dd | 0x70f | 0x890 | 0x891 | 0x8e2
        | 0x1680 | 0x180e
        | 0x2000..=0x200f | 0x2028..=0x202f | 0x205f..=0x2064 | 0x2066..=0x206f
        | 0x3000 | 0xd800..=0xf8ff | 0xfdd0..=0xfdef | 0xfeff | 0xfff9..=0xfffb
        | 0xfffe | 0xffff
        | 0x110bd | 0x110cd | 0x13430..=0x1343f | 0x1bca0..=0x1bca3 | 0x1d173..=0x1d17a
        | 0xe0001 | 0xe0020..=0xe007f | 0xf0000..=0x10ffff
    ) && (cp & 0xfffe) != 0xfffe
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strip_matches_python_isspace() {
        assert_eq!(strip("\x1c a \x1f"), "a");
        assert_eq!(strip("\u{3000}x\u{a0}"), "x");
        assert_eq!(strip("\u{200b}x"), "\u{200b}x"); // not whitespace in either
    }

    #[test]
    fn len_counts_code_points() {
        assert_eq!(len("é😀"), 2);
        assert_eq!(truncate("é😀abc", 2), "é😀...");
        assert_eq!(truncate("abc", 0), "abc");
        assert_eq!(truncate("abc", -1), "abc");
        assert_eq!(truncate("abc", 3), "abc");
    }

    #[test]
    fn dumps_default_separators_and_escapes() {
        let v: Value = serde_json::from_str(r#"{"a":[1,2.5,"x\u007f \u0001é😀"]}"#).unwrap();
        assert_eq!(dumps(&v, DUMPS_UNICODE), "{\"a\": [1, 2.5, \"x\u{7f}\u{2028}\\u0001é😀\"]}");
        let v = json!({"a": "é😀\u{7f}"});
        assert_eq!(dumps(&v, DUMPS_ASCII), r#"{"a": "\u00e9\ud83d\ude00\u007f"}"#);
        assert_eq!(dumps_len(&v, DUMPS_UNICODE), len(&dumps(&v, DUMPS_UNICODE)));
    }

    #[test]
    fn dumps_numbers_like_python() {
        let v: Value = serde_json::from_str("[1e16, 1.5e-7, 1e5, 100000.0, 0.1, 1e22, 12345678901234567890123, -0, 1e400, 0.0001, 0.00001, -0.0]").unwrap();
        assert_eq!(
            dumps(&v, DUMPS_UNICODE),
            "[1e+16, 1.5e-07, 100000.0, 100000.0, 0.1, 1e+22, 12345678901234567890123, 0, Infinity, 0.0001, 1e-05, -0.0]"
        );
    }

    #[test]
    fn canonical_sorts_keys() {
        let v = json!({"b": 1, "a": {"d": [], "c": {}}});
        assert_eq!(canonical_json(&v), r#"{"a":{"c":{},"d":[]},"b":1}"#);
    }

    #[test]
    fn indent_one_like_json_dump() {
        let v = json!({"a": [1, {"b": []}], "c": {}});
        assert_eq!(
            dumps(&v, Fmt { indent: Some(1), ..DUMPS_UNICODE }),
            "{\n \"a\": [\n  1,\n  {\n   \"b\": []\n  }\n ],\n \"c\": {}\n}"
        );
    }

    #[test]
    fn repr_like_python() {
        let v: Value = serde_json::from_str(r#"{"a": [1, 2.0, null, true, "it's", "q\"'"], "b": "\n é"}"#).unwrap();
        assert_eq!(repr(&v), r#"{'a': [1, 2.0, None, True, "it's", 'q"\''], 'b': '\n\xa0é'}"#);
    }

    #[test]
    fn truthiness_and_ints() {
        let v: Value = serde_json::from_str("[0, 0.0, -0.0, 1, \"\", [], {}, null, false]").unwrap();
        let t: Vec<bool> = v.as_array().unwrap().iter().map(truthy).collect();
        assert_eq!(t, [false, false, false, true, false, false, false, false, false]);
        let v: Value = serde_json::from_str("[1, 0, -5, 2, 1.0, true, false, -99999999999999999999999]").unwrap();
        let t: Vec<bool> = v.as_array().unwrap().iter().map(int_le_one).collect();
        assert_eq!(t, [true, true, true, false, false, true, true, true]);
    }

    #[test]
    fn parse_int_like_python() {
        assert_eq!(parse_int(" 1_000 "), Some(1000));
        assert_eq!(parse_int("-5"), Some(-5));
        assert_eq!(parse_int("1__0"), None);
        assert_eq!(parse_int("x"), None);
        assert_eq!(parse_int(""), None);
    }

    #[test]
    fn duplicate_keys_keep_first_position_last_value() {
        let v: Value = serde_json::from_str(r#"{"a":1,"b":2,"a":3}"#).unwrap();
        assert_eq!(dumps(&v, DUMPS_UNICODE), r#"{"a": 3, "b": 2}"#);
    }
}
