//! `GET /v1/slots/{slot_id}/evaluate` (API §3.2, guide §6.2). Generic over slots: dispatch is on
//! the slot definition's `evaluation.evaluator` (A-32), so no slot id is compiled in (A-27).
//! Pure over a stored run: no writes, no market data, no Python.
//!
//! Privacy (P-5..P-8): the request span logs the path only, responses are `no-store`, and the
//! route has its own rate-limit bucket (A-35). The query string is never logged.

use std::{collections::BTreeMap, sync::Arc};

use axum::{
    Json,
    extract::{Path, Query, State},
    http::{HeaderValue, header},
    response::{IntoResponse, Response},
};
use uuid::Uuid;

use super::{
    params::{split, valid_identifier},
    slots::{EnvelopeResponse, lookup, resolve_latest},
};
use crate::{
    cache::LatestCache,
    catalog::Catalog,
    domain::{RunMeta, SlotStatus, scope},
    error::AppError,
    pricing::{
        evaluate::{Config, EvalError, Request, evaluate as run_evaluation},
        surface::{StoredRun as PricedRun, parse_utc},
    },
    repo::RunRepository,
    state::{Clock, SurfaceCache},
};

const EVALUATORS: &[&str] = &["black76_surface_v1"];

const CONTROL: &[&str] = &[
    "run_id",
    "role",
    "implementation",
    "option_type",
    "expiry",
    "strike",
    "size",
    "premium",
    "premium_unit",
    "side",
    "spot_ref",
    "quoted_at",
];

fn bad(code: &'static str, message: impl Into<String>) -> AppError {
    AppError::Evaluate(EvalError::new(code, message))
}

fn number(
    params: &BTreeMap<String, String>,
    key: &str,
    code: &'static str,
) -> Result<Option<f64>, AppError> {
    params
        .get(key)
        .map(|v| {
            v.parse::<f64>()
                .map_err(|_| bad(code, format!("{key} must be a number")))
        })
        .transpose()
}

fn timestamp(
    params: &BTreeMap<String, String>,
    key: &str,
    code: &'static str,
) -> Result<Option<chrono::DateTime<chrono::Utc>>, AppError> {
    params
        .get(key)
        .map(|v| {
            parse_utc(v).ok_or_else(|| {
                bad(
                    code,
                    format!("{key} must be an ISO 8601 UTC time with a time, ending in Z"),
                )
            })
        })
        .transpose()
}

fn request(c: &BTreeMap<String, String>) -> Result<Request, AppError> {
    Ok(Request {
        option_type: c
            .get("option_type")
            .cloned()
            .ok_or_else(|| bad("invalid_option_type", "option_type is required"))?,
        expiry: timestamp(c, "expiry", "invalid_expiry")?
            .ok_or_else(|| bad("invalid_expiry", "expiry is required"))?,
        strike: number(c, "strike", "invalid_strike")?
            .ok_or_else(|| bad("invalid_strike", "strike is required"))?,
        size: number(c, "size", "invalid_size")?.unwrap_or(1.0),
        premium: number(c, "premium", "invalid_premium")?,
        premium_unit: c.get("premium_unit").cloned(),
        side: c.get("side").cloned(),
        spot_ref: number(c, "spot_ref", "invalid_spot_ref")?,
        quoted_at: timestamp(c, "quoted_at", "invalid_quoted_at")?,
    })
}

#[utoipa::path(get, path = "/v1/slots/{slot_id}/evaluate", responses(
    (status = 200, body = EnvelopeResponse, description = "Evaluation over a stored run (guide §6.2)"),
    (status = 400, description = "Invalid request; problem type names the field (e.g. expiry_in_past)"),
    (status = 404, description = "unknown-slot, no-evaluator, run_not_found or no-output-yet"),
))]
#[allow(clippy::too_many_arguments)]
pub async fn evaluate(
    State(catalog): State<Catalog>,
    State(cache): State<LatestCache>,
    State(runs): State<Arc<dyn RunRepository>>,
    State(surfaces): State<SurfaceCache>,
    State(clock): State<Clock>,
    Path(slot_id): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Result<Response, AppError> {
    let slot = lookup(&catalog, &slot_id)?;
    let evaluation = &slot.definition["evaluation"];
    match evaluation["evaluator"].as_str() {
        Some(e) if EVALUATORS.contains(&e) => {}
        _ => return Err(AppError::NoEvaluator(slot_id)),
    }
    let config = Config::from_value(evaluation)
        .ok_or_else(|| AppError::Internal(format!("{slot_id}: malformed evaluation block")))?;

    let (control, scope_params) = split(pairs, CONTROL)?;
    let scope = scope::validate(&slot, &scope_params).map_err(AppError::InvalidQuery)?;
    let req = request(&control)?;

    let (role, run) = match control.get("run_id") {
        Some(id) => {
            if control.contains_key("role") || control.contains_key("implementation") {
                return Err(AppError::InvalidQuery(
                    "run_id pins the run; drop role and implementation".into(),
                ));
            }
            let uuid = Uuid::try_parse(id).map_err(|_| AppError::RunNotFound(id.clone()))?;
            let run = runs
                .by_id(uuid)
                .await?
                .ok_or_else(|| AppError::RunNotFound(id.clone()))?;
            let same_scope =
                run.envelope["scope"] == serde_json::to_value(&scope).unwrap_or_default();
            if run.slot_id != slot.slot_id || !same_scope {
                return Err(AppError::RunNotFound(id.clone()));
            }
            if run.envelope["status"] != "success" {
                return Err(AppError::RunFailed(id.clone()));
            }
            (run.role, Arc::new(run))
        }
        None => {
            if slot.status == SlotStatus::Retired {
                return Err(AppError::SlotRetired(slot_id));
            }
            if control
                .get("implementation")
                .is_some_and(|i| !valid_identifier(i))
            {
                return Err(AppError::InvalidQuery(
                    "implementation is not a valid identifier".into(),
                ));
            }
            resolve_latest(&slot, &scope, &control, &cache, runs.as_ref()).await?
        }
    };

    let priced = match surfaces.get(&run.run_id) {
        Some(p) => p,
        None => {
            let p = Arc::new(PricedRun::from_envelope(&run.envelope).map_err(AppError::Evaluate)?);
            surfaces.insert(run.run_id, p.clone());
            p
        }
    };
    let meta = RunMeta::new(role, run.as_of, slot.max_age_seconds, clock());
    let data = run_evaluation(&priced, &req, &config, meta.stale).map_err(AppError::Evaluate)?;
    tracing::info!(run_id = %run.run_id, slot_id = %slot.slot_id, "served evaluation");

    let mut res = Json(EnvelopeResponse::new(data, meta)).into_response();
    res.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok(res)
}
