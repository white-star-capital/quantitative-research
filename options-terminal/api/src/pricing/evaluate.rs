//! Evaluation over a stored run (guide §6.2). Port of `core/quant/evaluate.py`: same check
//! order, same flags in the same order, same output shape.

use chrono::{DateTime, Utc};
use serde_json::{Map, Value, json};

use super::black76::{Black76, black76, implied_vol};
use super::surface::{Attr, StoredRun, Surface, year_fraction};

const DISTRIBUTION_POINTS: usize = 200;
const STRIKE_RANGE: (f64, f64) = (0.2, 5.0);
const PREMIUM_UNITS: [&str; 4] = ["usd_per_unit", "usd_total", "pct_spot", "coin_per_unit"];

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{code}: {message}")]
pub struct EvalError {
    pub code: &'static str,
    pub message: String,
}

impl EvalError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// The slot YAML's `evaluation` block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub quote_max_age_minutes: f64,
    pub spot_move_flag_pct: f64,
    pub max_tenor_multiple: f64,
}

impl Config {
    pub fn from_value(v: &Value) -> Option<Self> {
        Some(Self {
            quote_max_age_minutes: v["quote_max_age_minutes"].as_f64()?,
            spot_move_flag_pct: v["spot_move_flag_pct"].as_f64()?,
            max_tenor_multiple: v["max_tenor_multiple"].as_f64()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub option_type: String,
    pub expiry: DateTime<Utc>,
    pub strike: f64,
    pub size: f64,
    pub premium: Option<f64>,
    pub premium_unit: Option<String>,
    pub side: Option<String>,
    pub spot_ref: Option<f64>,
    pub quoted_at: Option<DateTime<Utc>>,
}

pub fn utc(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn flag(code: &str, message: impl Into<String>) -> Value {
    json!({ "code": code, "message": message.into() })
}

fn flag_text(code: &str) -> &'static str {
    match code {
        "strike_extrapolated" => {
            "Strike is outside the listed strikes; vol held flat at nearest node"
        }
        "tenor_extrapolated_short" => "Expiry is before the first listed expiry; vol held flat",
        "tenor_extrapolated_long" => {
            "Expiry is after the last listed expiry; vol and forward extended"
        }
        _ => "",
    }
}

fn opt(v: Option<f64>) -> Value {
    v.map_or(Value::Null, Value::from)
}

fn price(
    f: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    is_call: bool,
    s: f64,
) -> Result<Black76, EvalError> {
    black76(f, k, t, sigma, r, is_call, s)
        .map_err(|_| EvalError::new("run_incompatible", "non-positive pricing input"))
}

/// §6.2 steps 1–6.
pub fn evaluate(
    run: &StoredRun,
    req: &Request,
    config: &Config,
    run_stale: bool,
) -> Result<Value, EvalError> {
    if req.option_type != "call" && req.option_type != "put" {
        return Err(EvalError::new(
            "invalid_option_type",
            "option_type must be call or put",
        ));
    }
    let is_call = req.option_type == "call";
    if !(req.size > 0.0 && req.size.is_finite()) {
        return Err(EvalError::new("invalid_size", "size must be positive"));
    }
    if !(req.strike > 0.0 && req.strike.is_finite()) {
        return Err(EvalError::new(
            "strike_out_of_range",
            "strike must be positive",
        ));
    }

    let (surface, s, r) = (&run.surface, run.spot, run.usd_rate);
    let t = year_fraction(run.as_of, req.expiry);
    if t <= 0.0 {
        return Err(EvalError::new(
            "expiry_in_past",
            "expiry is at or before the run's valuation time",
        ));
    }
    if t > config.max_tenor_multiple * surface.t_last() {
        return Err(EvalError::new(
            "tenor_out_of_range",
            format!(
                "expiry is beyond {}x the last listed expiry",
                config.max_tenor_multiple
            ),
        ));
    }

    let (f, f_flags) = surface.forward(t);
    let (lo, hi) = STRIKE_RANGE;
    if !(lo * f <= req.strike && req.strike <= hi * f) {
        return Err(EvalError::new(
            "strike_out_of_range",
            format!(
                "strike must be within [{lo}F, {hi}F] = [{:.6}, {:.6}]",
                lo * f,
                hi * f
            ),
        ));
    }
    let k_abs = req.strike;
    let k = (k_abs / f).ln();
    let mark = surface
        .sigma(k, t, Attr::Iv)
        .ok_or_else(|| EvalError::new("run_incompatible", "a node has no mark IV"))?;
    let bid = surface.sigma(k, t, Attr::Bid);
    let ask = surface.sigma(k, t, Attr::Ask);
    let sigma = mark.sigma;

    let mut flags: Vec<Value> = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for code in mark.flags.iter().chain(f_flags.iter()) {
        if !seen.contains(code) {
            seen.push(code);
            flags.push(flag(code, flag_text(code)));
        }
    }
    if mark.nodes.iter().any(|n| n.quality != "good") {
        flags.push(flag(
            "low_quality_nodes",
            "A bracketing node is not two-sided and tight",
        ));
    }
    let same_day: Vec<_> = surface
        .expiries
        .iter()
        .filter(|e| e.expiry.date_naive() == req.expiry.date_naive())
        .collect();
    if !same_day.is_empty()
        && same_day
            .iter()
            .all(|e| (e.expiry - req.expiry).num_seconds().abs() >= 60)
    {
        flags.push(flag(
            "expiry_time_differs_from_venue",
            format!("Venue expiry that day is {}", utc(same_day[0].expiry)),
        ));
    }
    if run_stale {
        flags.push(flag(
            "run_stale",
            "The run is older than the slot's max age",
        ));
    }

    let g = match price(f, k_abs, t, sigma, r, is_call, s)? {
        Black76::Full(g) => g,
        Black76::Intrinsic { .. } => {
            return Err(EvalError::new("run_incompatible", "zero vol or tenor"));
        }
    };
    let fair = g.price;
    let dte = t * 365.0;
    let side_sign = if req.side.as_deref() == Some("sell") {
        -1.0
    } else {
        1.0
    };
    let screen_bid = bid
        .as_ref()
        .map(|b| price(f, k_abs, t, b.sigma, r, is_call, s).map(|p| p.price()))
        .transpose()?;
    let screen_ask = ask
        .as_ref()
        .map(|a| price(f, k_abs, t, a.sigma, r, is_call, s).map(|p| p.price()))
        .transpose()?;

    let greeks = [
        ("delta", g.delta),
        ("delta_spot", g.delta_spot),
        ("gamma_1pct", g.gamma_1pct),
        ("vega_1pt", g.vega_1pt),
        ("theta_1d", g.theta_1d),
    ];
    let mut per_unit: Map<String, Value> = greeks
        .iter()
        .map(|(k, v)| ((*k).to_owned(), json!(v)))
        .collect();
    per_unit.insert("p_itm".into(), json!(g.p_itm));
    let position: Map<String, Value> = greeks
        .iter()
        .map(|(k, v)| ((*k).to_owned(), json!(v * req.size * side_sign)))
        .collect();

    let quote = match req.premium {
        None => Value::Null,
        Some(premium) => quote(
            req, run, config, premium, f, k_abs, t, sigma, is_call, fair, screen_bid, screen_ask,
            &mut flags,
        )?,
    };

    Ok(json!({
        "run_id": run.run_id,
        "as_of": utc(run.as_of),
        "valuation_time": utc(run.as_of),
        "underlying": run.underlying,
        "spot": s,
        "contract": {
            "option_type": req.option_type,
            "expiry": utc(req.expiry),
            "strike": k_abs,
            "size": req.size,
            "T": t,
            "dte": dte,
        },
        "market": {
            "forward": f,
            "k": k,
            "iv_mark": sigma,
            "iv_bid": opt(bid.as_ref().map(|b| b.sigma)),
            "iv_ask": opt(ask.as_ref().map(|a| a.sigma)),
            "interpolation": mark.interpolation,
            "bracketing_expiries": mark.expiries.iter().map(|e| utc(e.expiry)).collect::<Vec<_>>(),
            "screen_bid_usd": opt(screen_bid),
            "screen_ask_usd": opt(screen_ask),
        },
        "fair": {
            "per_unit_usd": fair,
            "total_usd": fair * req.size,
            "pct_spot": fair / s,
            "greeks_per_unit": per_unit,
            "greeks_position": position,
            "breakeven": if is_call { k_abs + fair } else { k_abs - fair },
            "yield_ann": fair / s * 365.0 / dte,
        },
        "quote": quote,
        "distribution": distribution(f, t, surface)?,
        "flags": flags,
    }))
}

#[allow(clippy::too_many_arguments)]
fn quote(
    req: &Request,
    run: &StoredRun,
    config: &Config,
    premium: f64,
    f: f64,
    k: f64,
    t: f64,
    sigma: f64,
    is_call: bool,
    fair: f64,
    screen_bid: Option<f64>,
    screen_ask: Option<f64>,
    flags: &mut Vec<Value>,
) -> Result<Value, EvalError> {
    let (s, r) = (run.spot, run.usd_rate);
    let side = match req.side.as_deref() {
        Some(side @ ("sell" | "buy")) => side,
        _ => {
            return Err(EvalError::new(
                "premium_requires_side",
                "premium needs side=sell or side=buy",
            ));
        }
    };
    let unit = match req.premium_unit.as_deref() {
        Some(u) if PREMIUM_UNITS.contains(&u) => u,
        _ => {
            return Err(EvalError::new(
                "premium_requires_unit",
                format!(
                    "premium needs premium_unit, one of: {}",
                    PREMIUM_UNITS.join(", ")
                ),
            ));
        }
    };
    if !(premium > 0.0 && premium.is_finite()) {
        return Err(EvalError::new(
            "invalid_premium",
            "premium must be positive",
        ));
    }
    if req.spot_ref.is_some_and(|x| !(x > 0.0 && x.is_finite())) {
        return Err(EvalError::new(
            "invalid_spot_ref",
            "spot_ref must be positive",
        ));
    }
    let prem = match unit {
        "usd_per_unit" => premium,
        "usd_total" => premium / req.size,
        "pct_spot" => {
            let spot_ref = req.spot_ref.ok_or_else(|| {
                EvalError::new(
                    "pct_spot_requires_spot_ref",
                    "pct_spot premiums need spot_ref",
                )
            })?;
            premium / 100.0 * spot_ref
        }
        _ => premium * s, // coin_per_unit
    };

    let s_ref = req.spot_ref.unwrap_or(s);
    let f_ref = f * s_ref / s;
    let iv = implied_vol(f_ref, k, t, prem, r, is_call, s_ref);
    let sell = side == "sell";
    let edge = if sell { prem - fair } else { fair - prem };
    let vol_edge = iv
        .ok()
        .map(|v| (v - sigma) * 100.0 * if sell { 1.0 } else { -1.0 });

    let (position, frac) = match (screen_bid, screen_ask) {
        (Some(b), Some(a)) => {
            let pos = if prem < b {
                "below_bid"
            } else if prem > a {
                "above_ask"
            } else {
                "inside"
            };
            let width = a - b;
            (
                pos,
                if width > 0.0 {
                    Some((prem - b) / width)
                } else {
                    None
                },
            )
        }
        _ => ("n/a", None),
    };

    let max_age = config.quote_max_age_minutes;
    if let Some(quoted_at) = req.quoted_at {
        let age_min = (run.as_of - quoted_at).num_milliseconds() as f64 / 60_000.0;
        if age_min > max_age {
            flags.push(flag(
                "quote_stale",
                format!(
                    "Quote is {:.1} days old relative to the run (limit {max_age} minutes)",
                    age_min / 1440.0
                ),
            ));
        }
    }
    if let Some(spot_ref) = req.spot_ref {
        let moved = (s / spot_ref - 1.0).abs() * 100.0;
        if moved > config.spot_move_flag_pct {
            flags.push(flag(
                "spot_moved",
                format!("Spot moved {moved:.2}% from the quote's spot ref"),
            ));
        }
    }

    let dte = t * 365.0;
    Ok(json!({
        "side": side,
        "premium_per_unit_usd": prem,
        "premium_total_usd": prem * req.size,
        "iv_quote": opt(iv.ok()),
        "iv_quote_status": iv.err().unwrap_or("ok"),
        "pct_of_fair": opt((fair > 0.0).then(|| prem / fair)),
        "edge_per_unit_usd": edge,
        "edge_total_usd": edge * req.size,
        "vol_edge_vp": opt(vol_edge),
        "screen_position": position,
        "screen_position_frac": opt(frac),
        "fair_at_spot_ref": price(f_ref, k, t, sigma, r, is_call, s_ref)?.price(),
        "yield_ann": prem / s * 365.0 / dte,
    }))
}

/// Lognormal pdf at the ATM vol for T (ignores skew): μ = ln F − ½σ²T over F·e^(±3σ√T).
fn distribution(f: f64, t: f64, surface: &Surface) -> Result<Value, EvalError> {
    let atm = surface
        .sigma(0.0, t, Attr::Iv)
        .ok_or_else(|| EvalError::new("run_incompatible", "no ATM vol"))?;
    let sd = atm.sigma * t.sqrt();
    let mu = f.ln() - 0.5 * sd * sd;
    let (lo, hi) = (f * (-3.0 * sd).exp(), f * (3.0 * sd).exp());
    let step = (hi - lo) / (DISTRIBUTION_POINTS - 1) as f64;
    let xs: Vec<f64> = (0..DISTRIBUTION_POINTS)
        .map(|i| lo + i as f64 * step)
        .collect();
    let pdf: Vec<f64> = xs
        .iter()
        .map(|x| {
            (-((x.ln() - mu).powi(2)) / (2.0 * sd * sd)).exp()
                / (x * sd * (2.0 * std::f64::consts::PI).sqrt())
        })
        .collect();
    Ok(json!({ "basis": "lognormal_atm", "sigma": atm.sigma, "x": xs, "pdf": pdf }))
}
