//! P-3 / A-33: the Rust pricing port must reproduce the Python reference on every vector in
//! contracts/fixtures/pricing/. Blocking in CI. Python is the reference: a mismatch is a Rust bug
//! unless the Python tests prove otherwise.
//!
//! Tolerances (guide §6.3): price relative 1e-9 (absolute 1e-12 near zero), IV absolute 1e-8,
//! greeks relative 1e-8; flags, reasons and error codes exact. Flag messages are prose and are
//! not compared.

use std::path::{Path, PathBuf};

use options_terminal_api::pricing::{
    black76::{Black76, black76, implied_vol},
    evaluate::{Config, Request, evaluate},
    surface::{Attr, StoredRun, parse_utc},
};
use serde_json::Value;

fn pricing_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../contracts/fixtures/pricing")
}

fn load(name: &str) -> Value {
    let path = pricing_dir().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

const IV_KEYS: &[&str] = &["iv", "iv_mark", "iv_bid", "iv_ask", "iv_quote", "sigma"];
const GREEK_KEYS: &[&str] = &[
    "delta",
    "delta_spot",
    "gamma_1pct",
    "vega_1pt",
    "theta_1d",
    "p_itm",
];

fn close(key: &str, actual: f64, expected: f64) -> bool {
    let diff = (actual - expected).abs();
    if IV_KEYS.contains(&key) {
        diff <= 1e-8
    } else if key == "vol_edge_vp" {
        diff <= 1e-6 // vol points: IV tolerance × 100
    } else if GREEK_KEYS.contains(&key) {
        diff <= 1e-12 + 1e-8 * expected.abs()
    } else {
        diff <= 1e-12 + 1e-9 * expected.abs()
    }
}

/// Structural comparison with per-key numeric tolerance. Returns mismatches as "path: a vs e".
fn compare(actual: &Value, expected: &Value, path: &str, key: &str, out: &mut Vec<String>) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            if !close(key, a, e) {
                out.push(format!("{path}: {a} vs {e}"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            let mut keys: Vec<&String> = a.keys().chain(e.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                if k == "message" && path.contains("flags") {
                    continue;
                }
                compare(
                    &a.get(k).cloned().unwrap_or(Value::Null),
                    &e.get(k).cloned().unwrap_or(Value::Null),
                    &format!("{path}.{k}"),
                    k,
                    out,
                );
            }
        }
        (Value::Array(a), Value::Array(e)) if a.len() == e.len() => {
            for (i, (x, y)) in a.iter().zip(e).enumerate() {
                compare(x, y, &format!("{path}[{i}]"), key, out);
            }
        }
        (a, e) if a == e => {}
        (a, e) => out.push(format!("{path}: {a} vs {e}")),
    }
}

#[test]
fn black76_vectors() {
    let doc = load("black76.json");
    let cases = doc["cases"].as_array().unwrap();
    assert!(cases.len() >= 500);
    let mut bad = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let x = &c["inputs"];
        let f = |k: &str| x[k].as_f64().unwrap();
        let got = black76(
            f("F"),
            f("K"),
            f("T"),
            f("sigma"),
            f("r"),
            x["is_call"].as_bool().unwrap(),
            f("S"),
        )
        .unwrap();
        let actual = match got {
            Black76::Intrinsic { price } => serde_json::json!({ "price": price }),
            Black76::Full(g) => serde_json::json!({
                "price": g.price, "delta": g.delta, "delta_spot": g.delta_spot, "gamma_1pct": g.gamma_1pct,
                "vega_1pt": g.vega_1pt, "theta_1d": g.theta_1d, "p_itm": g.p_itm,
            }),
        };
        compare(&actual, &c["expected"], &format!("case {i}"), "", &mut bad);
    }
    assert!(
        bad.is_empty(),
        "{} mismatches:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[test]
fn implied_vol_vectors() {
    let doc = load("implied_vol.json");
    let mut bad = Vec::new();
    for (i, c) in doc["cases"].as_array().unwrap().iter().enumerate() {
        let x = &c["inputs"];
        let f = |k: &str| x[k].as_f64().unwrap();
        let got = implied_vol(
            f("F"),
            f("K"),
            f("T"),
            f("price"),
            f("r"),
            x["is_call"].as_bool().unwrap(),
            f("S"),
        );
        let actual = match got {
            Ok(iv) => serde_json::json!({ "iv": iv, "reason": null }),
            Err(reason) => serde_json::json!({ "iv": null, "reason": reason }),
        };
        compare(&actual, &c["expected"], &format!("case {i}"), "", &mut bad);
    }
    assert!(
        bad.is_empty(),
        "{} mismatches:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

fn vector_files(prefix: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(pricing_dir())
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix) && n.ends_with(".json"))
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no {prefix}*.json vectors");
    names
}

#[test]
fn interpolation_vectors() {
    for name in vector_files("interpolation") {
        let doc = load(&name);
        let run = StoredRun::from_envelope(&doc["run"]).unwrap();
        let mut bad = Vec::new();
        for (i, c) in doc["cases"].as_array().unwrap().iter().enumerate() {
            let (k, t) = (
                c["inputs"]["k"].as_f64().unwrap(),
                c["inputs"]["T"].as_f64().unwrap(),
            );
            let (fwd, f_flags) = run.surface.forward(t);
            let mut actual = serde_json::json!({ "forward": fwd, "forward_flags": f_flags });
            for (key, attr) in [
                ("iv", Attr::Iv),
                ("iv_bid", Attr::Bid),
                ("iv_ask", Attr::Ask),
            ] {
                actual[key] = match run.surface.sigma(k, t, attr) {
                    None => Value::Null,
                    Some(p) => serde_json::json!({
                        "sigma": p.sigma, "flags": p.flags, "interpolation": p.interpolation,
                        "expiries": p.expiries.iter().map(|e| options_terminal_api::pricing::evaluate::utc(e.expiry)).collect::<Vec<_>>(),
                    }),
                };
            }
            compare(
                &actual,
                &c["expected"],
                &format!("{name} case {i}"),
                "",
                &mut bad,
            );
        }
        assert!(
            bad.is_empty(),
            "{} mismatches:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }
}

#[test]
fn evaluate_vectors() {
    for name in vector_files("evaluate") {
        let doc = load(&name);
        let run = StoredRun::from_envelope(&doc["run"]).unwrap();
        let config = Config::from_value(&doc["config"]).unwrap();
        let mut bad = Vec::new();
        for (i, c) in doc["cases"].as_array().unwrap().iter().enumerate() {
            let p = &c["request"];
            let s = |k: &str| p[k].as_str().map(str::to_owned);
            let req = Request {
                option_type: s("option_type").unwrap(),
                expiry: parse_utc(p["expiry"].as_str().unwrap()).unwrap(),
                strike: p["strike"].as_f64().unwrap(),
                size: p["size"].as_f64().unwrap_or(1.0),
                premium: p["premium"].as_f64(),
                premium_unit: s("premium_unit"),
                side: s("side"),
                spot_ref: p["spot_ref"].as_f64(),
                quoted_at: p["quoted_at"].as_str().map(|t| parse_utc(t).unwrap()),
            };
            let actual = match evaluate(&run, &req, &config, c["run_stale"].as_bool().unwrap()) {
                Ok(data) => serde_json::json!({ "data": data }),
                Err(e) => serde_json::json!({ "error": e.code }),
            };
            compare(
                &actual,
                &c["expected"],
                &format!("{name} case {i}"),
                "",
                &mut bad,
            );
        }
        assert!(
            bad.is_empty(),
            "{} mismatches:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }
}

#[test]
fn comparator_catches_drift() {
    use serde_json::json;
    let check = |a: Value, e: Value| {
        let mut out = Vec::new();
        compare(&a, &e, "x", "", &mut out);
        out.len()
    };
    assert_eq!(
        check(json!({"price": 1.0 + 1e-10}), json!({"price": 1.0})),
        0
    );
    assert_eq!(
        check(json!({"price": 1.0 + 1e-7}), json!({"price": 1.0})),
        1
    );
    assert_eq!(
        check(json!({"iv_mark": 0.5 + 1e-7}), json!({"iv_mark": 0.5})),
        1
    );
    assert_eq!(
        check(
            json!({"flags": [{"code": "a", "message": "x"}]}),
            json!({"flags": [{"code": "b", "message": "x"}]})
        ),
        1
    );
    assert_eq!(
        check(
            json!({"flags": [{"code": "a", "message": "x"}]}),
            json!({"flags": [{"code": "a", "message": "y"}]})
        ),
        0
    );
    assert_eq!(check(json!({"a": 1.0}), json!({"a": 1.0, "b": null})), 0);
    assert_eq!(check(json!({"a": 1.0}), json!({"a": 1.0, "b": 2.0})), 1);
    assert_eq!(check(json!([1.0]), json!([1.0, 2.0])), 1);
}
