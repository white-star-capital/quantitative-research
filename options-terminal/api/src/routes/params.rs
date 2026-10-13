//! Query-string handling shared by slot routes (A-3, A-20).

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};

use crate::error::AppError;

const IDENTIFIER_MAX: usize = 64;

/// `^[a-z][a-z0-9_]{2,63}$` (CONTRACTS §2), checked before any lookup (A-20).
pub fn valid_identifier(s: &str) -> bool {
    let b = s.as_bytes();
    (3..=IDENTIFIER_MAX).contains(&b.len())
        && b[0].is_ascii_lowercase()
        && b.iter()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'_')
}

pub type Params = BTreeMap<String, String>;

/// Split query pairs into reserved control params and scope params. Duplicate keys are a 400:
/// `?underlying=BTC&underlying=ETH` has no single meaning.
pub fn split(
    pairs: Vec<(String, String)>,
    reserved: &[&str],
) -> Result<(Params, Params), AppError> {
    let mut control = BTreeMap::new();
    let mut scope = BTreeMap::new();
    for (k, v) in pairs {
        let target = if reserved.contains(&k.as_str()) {
            &mut control
        } else {
            &mut scope
        };
        if target.insert(k.clone(), v).is_some() {
            return Err(AppError::InvalidQuery(format!(
                "parameter '{k}' given more than once"
            )));
        }
    }
    Ok((control, scope))
}

pub fn utc_param(name: &str, value: &str) -> Result<DateTime<Utc>, AppError> {
    if !value.ends_with('Z') {
        return Err(AppError::InvalidQuery(format!(
            "{name} must be a UTC timestamp ending in Z"
        )));
    }
    DateTime::parse_from_rfc3339(value)
        .map(|t| t.with_timezone(&Utc))
        .map_err(|_| AppError::InvalidQuery(format!("{name} must be an RFC 3339 UTC timestamp")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers() {
        assert!(valid_identifier("vol_surface"));
        assert!(!valid_identifier("vs"));
        assert!(!valid_identifier("Vol_surface"));
        assert!(!valid_identifier("vol-surface"));
        assert!(!valid_identifier(&"a".repeat(65)));
    }

    #[test]
    fn duplicates_rejected() {
        let pairs = vec![
            ("underlying".into(), "BTC".into()),
            ("underlying".into(), "ETH".into()),
        ];
        assert!(split(pairs, &["role"]).is_err());
    }
}
