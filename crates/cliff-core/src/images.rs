//! Token estimation for image blocks.
//!
//! The rest of a request estimates at chars/4, which holds for text because
//! tokenizers average ~4 characters per token on prose and code. It does not
//! hold for an image: the block's characters are base64 of a compressed file,
//! and its token cost is set by the picture's dimensions, which the payload
//! length barely correlates with. A 1536x1024 screenshot is ~2.7M base64
//! characters -- chars/4 calls that 665k tokens, against a real cost near 2k.
//!
//! The formula below is Anthropic's: 28x28-pixel patches, capped at the
//! high-resolution tier's ceiling. Erring high is deliberate. Overcounting
//! compacts a little early; undercounting sends a request the provider
//! refuses for length, which fail-open cannot undo.

use serde_json::Value;

use crate::py::{self, PyErr, PyResult};

pub const PATCH_PX: u64 = 28;
/// Providers downscale before counting, so cost is bounded no matter the input.
pub const MAX_IMAGE_TOKENS: u64 = 4_784;
/// Dimensions unavailable: a hosted URL, an unknown format, a truncated payload.
pub const DEFAULT_IMAGE_TOKENS: u64 = 1_568;
/// Enough base64 to cover a JPEG's SOF marker past any EXIF block.
const HEADER_B64_CHARS: usize = 8192;

fn be16(b: &[u8]) -> u32 {
    u32::from(b[0]) << 8 | u32::from(b[1])
}
fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}
fn le16(b: &[u8]) -> u32 {
    u32::from(b[0]) | u32::from(b[1]) << 8
}
fn le24(b: &[u8]) -> u32 {
    u32::from(b[0]) | u32::from(b[1]) << 8 | u32::from(b[2]) << 16
}

fn png_dimensions(raw: &[u8]) -> Option<(u32, u32)> {
    if !raw.starts_with(b"\x89PNG\r\n\x1a\n") || raw.len() < 24 {
        return None;
    }
    Some((be32(&raw[16..20]), be32(&raw[20..24])))
}

fn gif_dimensions(raw: &[u8]) -> Option<(u32, u32)> {
    if !(raw.starts_with(b"GIF87a") || raw.starts_with(b"GIF89a")) || raw.len() < 10 {
        return None;
    }
    Some((le16(&raw[6..8]), le16(&raw[8..10])))
}

fn jpeg_dimensions(raw: &[u8]) -> Option<(u32, u32)> {
    if !raw.starts_with(b"\xff\xd8") {
        return None;
    }
    let (mut i, n) = (2usize, raw.len());
    while i + 9 < n {
        if raw[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = raw[i + 1];
        // Standalone markers carry no length field.
        if marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
            i += 2;
            continue;
        }
        let seg_len = be16(&raw[i + 2..i + 4]) as usize;
        // SOF0-SOF15, minus the ones that are not frame headers.
        if (0xC0..=0xCF).contains(&marker) && ![0xC4, 0xC8, 0xCC].contains(&marker) {
            let height = be16(&raw[i + 5..i + 7]);
            let width = be16(&raw[i + 7..i + 9]);
            return Some((width, height));
        }
        if seg_len < 2 {
            return None;
        }
        i += 2 + seg_len;
    }
    None
}

fn webp_dimensions(raw: &[u8]) -> Option<(u32, u32)> {
    if !raw.starts_with(b"RIFF") || raw.len() < 30 || &raw[8..12] != b"WEBP" {
        return None;
    }
    match &raw[12..16] {
        b"VP8X" => Some((le24(&raw[24..27]) + 1, le24(&raw[27..30]) + 1)),
        b"VP8 " => Some((le16(&raw[26..28]) & 0x3FFF, le16(&raw[28..30]) & 0x3FFF)),
        b"VP8L" => {
            let bits = u32::from_le_bytes([raw[21], raw[22], raw[23], raw[24]]);
            Some(((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1))
        }
        _ => None,
    }
}

/// Width and height from a decoded image header, or None if unreadable.
/// Readers are tried in the Python order; one that yields a zero dimension
/// falls through to the next.
pub fn dimensions(raw: &[u8]) -> Option<(u32, u32)> {
    let readers: [fn(&[u8]) -> Option<(u32, u32)>; 4] =
        [png_dimensions, jpeg_dimensions, gif_dimensions, webp_dimensions];
    readers.iter().filter_map(|r| r(raw)).find(|(w, h)| *w > 0 && *h > 0)
}

/// `base64.b64decode(s, validate=False)`: CPython's non-strict
/// `binascii.a2b_base64`, step for step. Characters outside the alphabet are
/// skipped, a completed pad sequence ends the input, and a dangling partial
/// quad is an error.
pub fn b64decode_lenient(s: &str) -> Option<Vec<u8>> {
    if !s.is_ascii() {
        return None; // b64decode(str) encodes to ASCII first: ValueError
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut quad_pos, mut leftchar, mut pads) = (0u8, 0u8, 0u8);
    for &ch in s.as_bytes() {
        if ch == b'=' {
            if quad_pos >= 2 {
                pads += 1;
                if quad_pos + pads >= 4 {
                    return Some(out);
                }
            }
            continue;
        }
        let v = match ch {
            b'A'..=b'Z' => ch - b'A',
            b'a'..=b'z' => ch - b'a' + 26,
            b'0'..=b'9' => ch - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => continue,
        };
        pads = 0;
        match quad_pos {
            0 => {
                quad_pos = 1;
                leftchar = v;
            }
            1 => {
                quad_pos = 2;
                out.push((leftchar << 2) | (v >> 4));
                leftchar = v & 0x0f;
            }
            2 => {
                quad_pos = 3;
                out.push((leftchar << 4) | (v >> 2));
                leftchar = v & 0x03;
            }
            _ => {
                quad_pos = 0;
                out.push((leftchar << 6) | v);
                leftchar = 0;
            }
        }
    }
    if quad_pos != 0 { None } else { Some(out) }
}

/// Decode just enough of a base64 payload to read an image header.
///
/// MIRRORED BUG: a `data:` URI without a comma raises IndexError in the
/// Python implementation (`payload.split(",", 1)[1]`), which aborts the
/// request's preparation and sends it verbatim. Reproduced as an `Err` so
/// both implementations fail open on the same inputs; fix in both at once.
fn header_bytes(payload: &str) -> PyResult<Option<Vec<u8>>> {
    let b64 = if payload.starts_with("data:") {
        match payload.split_once(',') {
            Some((_, rest)) => rest,
            None => return Err(PyErr::index_error("list index out of range")),
        }
    } else {
        payload
    };
    let b64 = py::prefix_chars(b64, HEADER_B64_CHARS);
    let n = py::len(b64);
    let b64 = py::prefix_chars(b64, n - n % 4);
    if b64.is_empty() {
        return Ok(None);
    }
    Ok(b64decode_lenient(b64))
}

/// Estimated token cost of one image, from its dimensions where readable.
pub fn tokens_for_payload(payload: &str) -> PyResult<u64> {
    if !payload.starts_with("data:") {
        // A hosted URL: nothing to measure, but it still costs a real image.
        return Ok(DEFAULT_IMAGE_TOKENS);
    }
    let raw = header_bytes(payload)?;
    let dims = raw.as_deref().filter(|r| !r.is_empty()).and_then(dimensions);
    let Some((w, h)) = dims else {
        return Ok(DEFAULT_IMAGE_TOKENS);
    };
    let patches = u64::from(w).div_ceil(PATCH_PX) * u64::from(h).div_ceil(PATCH_PX);
    Ok(patches.clamp(1, MAX_IMAGE_TOKENS))
}

/// Every image payload string in a request body, across all dialects.
///
/// ```text
/// Anthropic  {"type": "image", "source": {"data": ...}}
/// Responses  {"type": "input_image", "image_url": "data:..."}
/// Chat       {"type": "image_url", "image_url": {"url": "data:..."}}
/// ```
///
/// Documents are deliberately not included: a PDF's cost tracks its text,
/// not its pixels. They keep the chars/4 estimate, which errs high.
pub fn image_payloads<'a>(obj: &'a Value, out: &mut Vec<&'a str>) -> PyResult<()> {
    match obj {
        Value::Object(map) => {
            if let Some(found) = image_payload_here(obj)? {
                out.push(found);
                return Ok(());
            }
            for value in map.values() {
                image_payloads(value, out)?;
            }
            Ok(())
        }
        Value::Array(items) => {
            for value in items {
                image_payloads(value, out)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// The payload a dict carries itself, if it is an image block (without
/// descending into its values).
pub fn image_payload_here(obj: &Value) -> PyResult<Option<&str>> {
    let kind = py::get(obj, "type").and_then(Value::as_str);
    match kind {
        Some("image") => {
            // MIRRORED BUG: `(obj.get("source") or {}).get("data")` raises
            // AttributeError when `source` is truthy but not a dict.
            let data = py::get_dict_or_empty(obj, "source")?.and_then(|s| py::get(s, "data"));
            Ok(data.and_then(Value::as_str))
        }
        Some("input_image") | Some("image_url") => {
            let mut url = py::get(obj, "image_url");
            if let Some(v @ Value::Object(_)) = url {
                url = py::get(v, "url");
            }
            Ok(url.and_then(Value::as_str))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lenient_b64_matches_cpython() {
        assert_eq!(b64decode_lenient("!!!!"), Some(vec![]));
        assert_eq!(b64decode_lenient("AB!D"), None);
        assert_eq!(b64decode_lenient("QUJD"), Some(b"ABC".to_vec()));
        assert_eq!(b64decode_lenient("QUJ"), None);
        assert_eq!(b64decode_lenient("QU==QUJD"), Some(b"A".to_vec()));
        assert_eq!(b64decode_lenient("QUJDRA"), None);
        assert_eq!(b64decode_lenient("iVBORw0KGgo="), Some(b"\x89PNG\r\n\x1a\n".to_vec()));
    }

    #[test]
    fn data_uri_without_comma_is_the_python_index_error() {
        assert!(tokens_for_payload("data:image/png;base64").is_err());
        assert_eq!(tokens_for_payload("https://x/y.png"), Ok(DEFAULT_IMAGE_TOKENS));
        assert_eq!(tokens_for_payload("data:image/png;base64,!!!!"), Ok(DEFAULT_IMAGE_TOKENS));
    }
}
