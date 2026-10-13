//! Smile and surface interpolation (guide §3.2–3.3). Port of `core/quant/smile.py` plus
//! `surface_from_payload` from `core/quant/evaluate.py`.

use chrono::{DateTime, Utc};
use serde_json::Value;

use super::evaluate::EvalError;

pub const SECONDS_PER_YEAR: f64 = 365.0 * 86400.0;
pub const EXACT_TENOR_YEARS: f64 = 60.0 / SECONDS_PER_YEAR;
pub const STRIKE_EXTRAPOLATED: &str = "strike_extrapolated";
pub const TENOR_SHORT: &str = "tenor_extrapolated_short";
pub const TENOR_LONG: &str = "tenor_extrapolated_long";

pub fn year_fraction(start: DateTime<Utc>, end: DateTime<Utc>) -> f64 {
    let micros = (end - start).num_microseconds().unwrap_or(i64::MAX);
    micros as f64 / 1e6 / SECONDS_PER_YEAR
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attr {
    Iv,
    Bid,
    Ask,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub k: f64,
    pub iv: f64,
    pub iv_bid: Option<f64>,
    pub iv_ask: Option<f64>,
    pub quality: String,
}

impl Node {
    fn get(&self, attr: Attr) -> Option<f64> {
        match attr {
            Attr::Iv => Some(self.iv),
            Attr::Bid => self.iv_bid,
            Attr::Ask => self.iv_ask,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expiry {
    pub expiry: DateTime<Utc>,
    pub t: f64,
    pub forward: f64,
    pub nodes: Vec<Node>, // sorted by k
}

#[derive(Debug, Clone)]
pub struct SmilePoint<'a> {
    pub sigma: f64,
    pub flags: Vec<&'static str>,
    pub nodes: Vec<&'a Node>,
}

/// Index of the last element <= x (Python `bisect_right(xs, x) - 1`).
fn bisect_right_minus_one(xs: impl Iterator<Item = f64>, x: f64) -> usize {
    xs.take_while(|v| *v <= x).count().saturating_sub(1)
}

impl Expiry {
    pub fn new(expiry: DateTime<Utc>, t: f64, forward: f64, mut nodes: Vec<Node>) -> Self {
        nodes.sort_by(|a, b| a.k.total_cmp(&b.k));
        Self {
            expiry,
            t,
            forward,
            nodes,
        }
    }

    pub fn k_min(&self) -> f64 {
        self.nodes[0].k
    }

    pub fn k_max(&self) -> f64 {
        self.nodes[self.nodes.len() - 1].k
    }

    /// σ_i(k) from the node attribute; None if a bracketing node lacks it.
    pub fn sigma(&self, k: f64, attr: Attr) -> Option<SmilePoint<'_>> {
        let first = &self.nodes[0];
        let last = &self.nodes[self.nodes.len() - 1];
        fn single(node: &Node, off: bool, attr: Attr) -> Option<SmilePoint<'_>> {
            node.get(attr).map(|v| SmilePoint {
                sigma: v,
                flags: if off {
                    vec![STRIKE_EXTRAPOLATED]
                } else {
                    vec![]
                },
                nodes: vec![node],
            })
        }
        if k <= first.k {
            return single(first, k < first.k, attr);
        }
        if k >= last.k {
            return single(last, k > last.k, attr);
        }
        let j = bisect_right_minus_one(self.nodes.iter().map(|n| n.k), k);
        let (lo, hi) = (&self.nodes[j], &self.nodes[j + 1]);
        let weight = (k - lo.k) / (hi.k - lo.k);
        let (v_lo, v_hi) = (lo.get(attr)?, hi.get(attr)?);
        let (w_lo, w_hi) = (v_lo.powi(2) * self.t, v_hi.powi(2) * self.t);
        let w = w_lo + (w_hi - w_lo) * weight;
        Some(SmilePoint {
            sigma: (w / self.t).sqrt(),
            flags: vec![],
            nodes: vec![lo, hi],
        })
    }
}

#[derive(Debug, Clone)]
pub struct SurfacePoint<'a> {
    pub sigma: f64,
    pub flags: Vec<&'static str>,
    pub expiries: Vec<&'a Expiry>,
    pub nodes: Vec<&'a Node>,
    pub interpolation: &'static str,
}

#[derive(Debug, Clone)]
pub struct Surface {
    pub spot: f64,
    pub expiries: Vec<Expiry>, // sorted by t
}

fn status(flags: &[&'static str], default: &'static str) -> &'static str {
    if flags.is_empty() {
        default
    } else {
        "extrapolated"
    }
}

impl Surface {
    pub fn new(spot: f64, mut expiries: Vec<Expiry>) -> Self {
        expiries.sort_by(|a, b| a.t.total_cmp(&b.t));
        Self { spot, expiries }
    }

    pub fn t_last(&self) -> f64 {
        self.expiries[self.expiries.len() - 1].t
    }

    /// ln F linear in T through (0, ln S), (T_i, ln F_i); last slope extended beyond T_n.
    pub fn forward(&self, t: f64) -> (f64, Vec<&'static str>) {
        let mut pts = vec![(0.0, self.spot.ln())];
        pts.extend(self.expiries.iter().map(|e| (e.t, e.forward.ln())));
        let n = pts.len();
        let (((t0, f0), (t1, f1)), flags) = if t >= pts[n - 1].0 {
            let flags = if t - pts[n - 1].0 >= EXACT_TENOR_YEARS {
                vec![TENOR_LONG]
            } else {
                vec![]
            };
            ((pts[n - 2], pts[n - 1]), flags)
        } else {
            let j = bisect_right_minus_one(pts.iter().map(|p| p.0), t);
            ((pts[j], pts[j + 1]), vec![])
        };
        ((f0 + (f1 - f0) * (t - t0) / (t1 - t0)).exp(), flags)
    }

    fn exact(&self, t: f64) -> Option<&Expiry> {
        // Python bisect_left, then check j-1 and j.
        let j = self.expiries.iter().take_while(|e| e.t < t).count();
        [j.checked_sub(1), Some(j)]
            .into_iter()
            .flatten()
            .filter(|i| *i < self.expiries.len())
            .map(|i| &self.expiries[i])
            .find(|e| (e.t - t).abs() < EXACT_TENOR_YEARS)
    }

    /// σ(k, T) per §3.3. None only when a bid/ask lookup lacks a bracketing node quote.
    pub fn sigma(&self, k: f64, t: f64, attr: Attr) -> Option<SurfacePoint<'_>> {
        if let Some(e) = self.exact(t) {
            let p = e.sigma(k, attr)?;
            let interpolation = status(&p.flags, "exact");
            return Some(SurfacePoint {
                sigma: p.sigma,
                flags: p.flags,
                expiries: vec![e],
                nodes: p.nodes,
                interpolation,
            });
        }
        let (first, last) = (&self.expiries[0], &self.expiries[self.expiries.len() - 1]);
        if t < first.t || t > last.t {
            let (e, flag) = if t < first.t {
                (first, TENOR_SHORT)
            } else {
                (last, TENOR_LONG)
            };
            let p = e.sigma(k, attr)?;
            let mut flags = p.flags;
            flags.push(flag);
            return Some(SurfacePoint {
                sigma: p.sigma,
                flags,
                expiries: vec![e],
                nodes: p.nodes,
                interpolation: "extrapolated",
            });
        }
        let j = bisect_right_minus_one(self.expiries.iter().map(|e| e.t), t);
        let (a, b) = (&self.expiries[j], &self.expiries[j + 1]);
        let (pa, pb) = (a.sigma(k, attr)?, b.sigma(k, attr)?);
        let (w_a, w_b) = (pa.sigma.powi(2) * a.t, pb.sigma.powi(2) * b.t);
        let w = w_a + (w_b - w_a) * (t - a.t) / (b.t - a.t);
        let mut flags: Vec<&'static str> = Vec::new();
        for f in pa.flags.iter().chain(pb.flags.iter()) {
            if !flags.contains(f) {
                flags.push(f);
            }
        }
        let mut nodes = pa.nodes;
        nodes.extend(pb.nodes);
        let interpolation = status(&flags, "interpolated");
        Some(SurfacePoint {
            sigma: (w / t).sqrt(),
            flags,
            expiries: vec![a, b],
            nodes,
            interpolation,
        })
    }
}

/// A stored run, ready to evaluate (port of `surface_from_payload`).
#[derive(Debug, Clone)]
pub struct StoredRun {
    pub run_id: String,
    pub as_of: DateTime<Utc>,
    pub underlying: String,
    pub spot: f64,
    pub usd_rate: f64,
    pub surface: Surface,
}

fn incompatible(what: &str) -> EvalError {
    EvalError::new(
        "run_incompatible",
        format!("stored run is missing or malformed: {what}"),
    )
}

fn section<'a>(payload: &'a Value, key: &str) -> Result<&'a Value, EvalError> {
    payload["sections"]
        .as_array()
        .and_then(|s| s.iter().find(|s| s["key"] == key))
        .map(|s| &s["payload"])
        .ok_or_else(|| incompatible(key))
}

/// Table rows as (column key → cell) maps.
fn records(table: &Value) -> Result<Vec<serde_json::Map<String, Value>>, EvalError> {
    let keys: Vec<String> = table["columns"]
        .as_array()
        .ok_or_else(|| incompatible("columns"))?
        .iter()
        .filter_map(|c| c["key"].as_str().map(str::to_owned))
        .collect();
    let rows = table["rows"]
        .as_array()
        .ok_or_else(|| incompatible("rows"))?;
    Ok(rows
        .iter()
        .filter_map(Value::as_array)
        .map(|row| keys.iter().cloned().zip(row.iter().cloned()).collect())
        .collect())
}

pub fn parse_utc(value: &str) -> Option<DateTime<Utc>> {
    if !value.ends_with('Z') || !value.contains('T') {
        return None;
    }
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

impl StoredRun {
    pub fn from_envelope(envelope: &Value) -> Result<Self, EvalError> {
        let run_id = envelope["run_id"]
            .as_str()
            .ok_or_else(|| incompatible("run_id"))?;
        let as_of = envelope["as_of"]
            .as_str()
            .and_then(parse_utc)
            .ok_or_else(|| incompatible("as_of"))?;
        let underlying = envelope["scope"]["underlying"]
            .as_str()
            .ok_or_else(|| incompatible("scope"))?;
        let payload = &envelope["payload"];

        let mut spot = None;
        let mut usd_rate = None;
        for item in section(payload, "context")?["items"]
            .as_array()
            .ok_or_else(|| incompatible("context"))?
        {
            match item["key"].as_str() {
                Some("spot") => spot = item["value"].as_f64(),
                Some("usd_rate") => usd_rate = item["value"].as_f64(),
                _ => {}
            }
        }
        let spot = spot
            .filter(|s| *s > 0.0)
            .ok_or_else(|| incompatible("context.spot"))?;
        let usd_rate = usd_rate.ok_or_else(|| incompatible("context.usd_rate"))?;

        let mut nodes: std::collections::HashMap<String, Vec<Node>> = Default::default();
        for row in records(section(payload, "chain")?)? {
            if row.get("is_node").and_then(Value::as_bool) != Some(true) {
                continue;
            }
            let num = |k: &str| row.get(k).and_then(Value::as_f64);
            let node = Node {
                k: num("k").ok_or_else(|| incompatible("chain.k"))?,
                iv: num("iv_mark").ok_or_else(|| incompatible("chain.iv_mark"))?,
                iv_bid: num("iv_bid"),
                iv_ask: num("iv_ask"),
                quality: row
                    .get("node_quality")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            };
            let expiry = row
                .get("expiry_ts")
                .and_then(Value::as_str)
                .ok_or_else(|| incompatible("chain.expiry_ts"))?;
            nodes.entry(expiry.to_owned()).or_default().push(node);
        }

        let mut expiries = Vec::new();
        for row in records(section(payload, "expiries")?)? {
            let ts = row
                .get("expiry_ts")
                .and_then(Value::as_str)
                .ok_or_else(|| incompatible("expiries.expiry_ts"))?;
            let when = parse_utc(ts).ok_or_else(|| incompatible("expiries.expiry_ts"))?;
            let forward = row
                .get("forward")
                .and_then(Value::as_f64)
                .ok_or_else(|| incompatible("expiries.forward"))?;
            let expiry_nodes = nodes
                .remove(ts)
                .filter(|n| !n.is_empty())
                .ok_or_else(|| incompatible("expiry nodes"))?;
            expiries.push(Expiry::new(
                when,
                year_fraction(as_of, when),
                forward,
                expiry_nodes,
            ));
        }
        if expiries.is_empty() {
            return Err(incompatible("expiries"));
        }
        Ok(Self {
            run_id: run_id.to_owned(),
            as_of,
            underlying: underlying.to_owned(),
            spot,
            usd_rate,
            surface: Surface::new(spot, expiries),
        })
    }
}
