//! Scope parsing and the canonical `scope_key` (CONTRACTS C-9, A-3).

use std::collections::BTreeMap;

use crate::domain::Slot;

/// Keys sorted, `k=v` joined by `&`, keys and values percent-encoded with RFC 3986 unreserved
/// characters left as-is. Must match `terminal_models.core.scope.scope_key` byte for byte.
pub fn scope_key(scope: &BTreeMap<String, String>) -> String {
    scope
        .iter()
        .map(|(k, v)| format!("{}={}", encode(k), encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Check query scope params against the slot's dimensions and allowlist; report every problem.
pub fn validate(
    slot: &Slot,
    params: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, String> {
    let mut problems = Vec::new();
    let unknown: Vec<&str> = params
        .keys()
        .filter(|k| !slot.dimensions.contains(k))
        .map(String::as_str)
        .collect();
    if !unknown.is_empty() {
        problems.push(format!("unknown parameters: {}", unknown.join(", ")));
    }
    let missing: Vec<&str> = slot
        .dimensions
        .iter()
        .filter(|d| params.get(*d).is_none_or(|v| v.is_empty()))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        problems.push(format!("missing scope parameters: {}", missing.join(", ")));
    }
    for (dim, allowed) in &slot.allowed {
        if let Some(v) = params.get(dim)
            && !v.is_empty()
            && !allowed.contains(v)
        {
            problems.push(format!("{dim} must be one of: {}", allowed.join(", ")));
        }
    }
    if problems.is_empty() {
        Ok(params.clone())
    } else {
        Err(problems.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Vector {
        scope: BTreeMap<String, String>,
        key: String,
    }

    #[test]
    fn matches_shared_contract_vectors() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../contracts/fixtures/scope_key_vectors.json"
        );
        let vectors: Vec<Vector> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(!vectors.is_empty());
        for v in vectors {
            assert_eq!(scope_key(&v.scope), v.key);
        }
    }
}
