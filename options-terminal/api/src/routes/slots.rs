//! Slot endpoints. Nothing here knows any particular slot or implementation (A-27).

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::{Value, json};
use utoipa::ToSchema;
use uuid::Uuid;

use super::params::{split, utc_param, valid_identifier};
use crate::{
    cache::{CacheKey, LatestCache},
    catalog::Catalog,
    domain::{
        Role, RunMeta, Slot, SlotStatus, StoredRun, scope,
        slot::{SlotDetail, SlotSummary},
    },
    error::AppError,
    repo::{RunQuery, RunRepository},
    state::Clock,
};

const LIMIT_DEFAULT: i64 = 50;
const LIMIT_MAX: i64 = 500;
const LIMIT_MAX_WITH_PAYLOAD: i64 = 50;

#[derive(Serialize, ToSchema)]
pub struct SlotList {
    data: Vec<SlotSummary>,
    #[schema(value_type = Object)]
    meta: Value,
}

#[derive(Serialize, ToSchema)]
pub struct EnvelopeResponse {
    /// The stored envelope, unchanged (A-7).
    #[schema(value_type = Object)]
    data: Value,
    meta: RunMeta,
}

impl EnvelopeResponse {
    pub fn new(data: Value, meta: RunMeta) -> Self {
        Self { data, meta }
    }
}

#[derive(Serialize, ToSchema)]
pub struct RunList {
    #[schema(value_type = Vec<Object>)]
    data: Vec<Value>,
    #[schema(value_type = Object)]
    meta: Value,
}

pub fn lookup(catalog: &Catalog, slot_id: &str) -> Result<Arc<Slot>, AppError> {
    if catalog.snapshot().is_none() {
        return Err(AppError::Unavailable("slot catalog not loaded"));
    }
    if !valid_identifier(slot_id) {
        return Err(AppError::UnknownSlot(slot_id.to_owned()));
    }
    catalog
        .get(slot_id)
        .ok_or_else(|| AppError::UnknownSlot(slot_id.to_owned()))
}

/// Active slots, ordered by `display.order` then id.
#[utoipa::path(get, path = "/v1/slots", responses((status = 200, body = SlotList)))]
pub async fn list_slots(State(catalog): State<Catalog>) -> Result<Json<SlotList>, AppError> {
    let snapshot = catalog
        .snapshot()
        .ok_or(AppError::Unavailable("slot catalog not loaded"))?;
    let mut slots: Vec<&Arc<Slot>> = snapshot
        .by_id
        .values()
        .filter(|s| s.status == SlotStatus::Active)
        .collect();
    let order = |s: &Slot| {
        s.definition
            .pointer("/display/order")
            .and_then(Value::as_i64)
            .unwrap_or(i64::MAX)
    };
    slots.sort_by(|a, b| {
        order(a)
            .cmp(&order(b))
            .then_with(|| a.slot_id.cmp(&b.slot_id))
    });
    Ok(Json(SlotList {
        data: slots.into_iter().map(|s| s.summary()).collect(),
        meta: json!({ "api_version": "1", "next_cursor": null }),
    }))
}

/// One slot, active or retired, with its production and shadow implementations.
#[utoipa::path(get, path = "/v1/slots/{slot_id}", responses(
    (status = 200, body = SlotDetail), (status = 404, description = "unknown-slot"),
))]
pub async fn get_slot(
    State(catalog): State<Catalog>,
    Path(slot_id): Path<String>,
) -> Result<Json<SlotDetail>, AppError> {
    Ok(Json(lookup(&catalog, &slot_id)?.detail()))
}

/// Latest successful envelope for a scope. Production by default; shadows need
/// `implementation` (A-4).
#[utoipa::path(get, path = "/v1/slots/{slot_id}/latest", responses(
    (status = 200, body = EnvelopeResponse),
    (status = 304, description = "Unchanged since If-None-Match"),
    (status = 400, description = "invalid-query"),
    (status = 404, description = "unknown-slot or no-output-yet"),
    (status = 410, description = "slot-retired"),
))]
pub async fn latest(
    State(catalog): State<Catalog>,
    State(cache): State<LatestCache>,
    State(runs): State<Arc<dyn RunRepository>>,
    State(clock): State<Clock>,
    Path(slot_id): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    let slot = lookup(&catalog, &slot_id)?;
    if slot.status == SlotStatus::Retired {
        return Err(AppError::SlotRetired(slot_id));
    }
    let (control, scope_params) = split(pairs, &["role", "implementation"])?;
    let scope = scope::validate(&slot, &scope_params).map_err(AppError::InvalidQuery)?;
    let (role, run) = resolve_latest(&slot, &scope, &control, &cache, runs.as_ref()).await?;

    let meta = RunMeta::new(role, run.as_of, slot.max_age_seconds, clock());
    let etag = format!("\"{}\"", run.run_id);
    let cache_control = format!("public, max-age={}", meta.cache_max_age());
    tracing::info!(run_id = %run.run_id, slot_id = %slot.slot_id, implementation = %run.implementation_id, "served latest");

    let not_modified = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|t| t.trim() == etag || t.trim() == "*"));
    let mut res = if not_modified {
        StatusCode::NOT_MODIFIED.into_response()
    } else {
        Json(EnvelopeResponse {
            data: run.envelope.clone(),
            meta,
        })
        .into_response()
    };
    set_header(&mut res, header::ETAG, &etag);
    set_header(&mut res, header::CACHE_CONTROL, &cache_control);
    Ok(res)
}

/// Run history, newest first, keyset-paginated (A-5). Works for retired slots too.
#[utoipa::path(get, path = "/v1/slots/{slot_id}/runs", responses(
    (status = 200, body = RunList), (status = 400, description = "invalid-query"),
    (status = 404, description = "unknown-slot"),
))]
pub async fn runs(
    State(catalog): State<Catalog>,
    State(repo): State<Arc<dyn RunRepository>>,
    Path(slot_id): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Result<Json<RunList>, AppError> {
    let slot = lookup(&catalog, &slot_id)?;
    let (c, scope_params) = split(
        pairs,
        &[
            "role",
            "implementation",
            "from",
            "to",
            "status",
            "limit",
            "cursor",
            "include_payload",
        ],
    )?;
    let scope = scope::validate(&slot, &scope_params).map_err(AppError::InvalidQuery)?;

    let include_payload = match c.get("include_payload").map(String::as_str) {
        None | Some("false") => false,
        Some("true") => true,
        Some(_) => {
            return Err(AppError::InvalidQuery(
                "include_payload must be true or false".into(),
            ));
        }
    };
    let max = if include_payload {
        LIMIT_MAX_WITH_PAYLOAD
    } else {
        LIMIT_MAX
    };
    let limit = match c.get("limit") {
        None => LIMIT_DEFAULT.min(max),
        Some(l) => l
            .parse::<i64>()
            .ok()
            .filter(|n| (1..=max).contains(n))
            .ok_or_else(|| AppError::InvalidQuery(format!("limit must be between 1 and {max}")))?,
    };
    let status = match c.get("status").map(String::as_str) {
        None => None,
        Some(s @ ("success" | "failed")) => Some(s.to_owned()),
        Some(_) => {
            return Err(AppError::InvalidQuery(
                "status must be success or failed".into(),
            ));
        }
    };
    let implementation = c.get("implementation").cloned();
    if implementation
        .as_deref()
        .is_some_and(|i| !valid_identifier(i))
    {
        return Err(AppError::InvalidQuery(
            "implementation is not a valid identifier".into(),
        ));
    }

    let query = RunQuery {
        slot_id: slot.slot_id.clone(),
        scope_key: scope::scope_key(&scope),
        role: parse_role(c.get("role"))?,
        implementation,
        status,
        from: c.get("from").map(|v| utc_param("from", v)).transpose()?,
        to: c.get("to").map(|v| utc_param("to", v)).transpose()?,
        before: c.get("cursor").map(|v| decode_cursor(v)).transpose()?,
        include_payload,
        limit: limit + 1,
    };
    let mut rows = repo.list(&query).await?;
    let next_cursor = if rows.len() as i64 > limit {
        rows.truncate(usize::try_from(limit).unwrap_or(0));
        rows.last().map(encode_cursor)
    } else {
        None
    };
    Ok(Json(RunList {
        data: rows.into_iter().map(|r| r.envelope).collect(),
        meta: json!({ "api_version": "1", "next_cursor": next_cursor }),
    }))
}

/// Latest successful run for (slot, scope), production by default; shadows need
/// `implementation` (A-4). Shared by /latest and /evaluate so both see the same run.
pub async fn resolve_latest(
    slot: &Slot,
    scope: &std::collections::BTreeMap<String, String>,
    control: &std::collections::BTreeMap<String, String>,
    cache: &LatestCache,
    runs: &dyn RunRepository,
) -> Result<(Role, Arc<StoredRun>), AppError> {
    let role = parse_role(control.get("role"))?;
    let implementation = match (role, control.get("implementation")) {
        (Role::Production, None) => None,
        (Role::Production, Some(_)) => {
            return Err(AppError::InvalidQuery(
                "implementation is only accepted with role=shadow; use /runs to filter production history".into(),
            ));
        }
        (Role::Shadow, Some(i)) if slot.is_shadow(i) => Some(i.clone()),
        (Role::Shadow, _) => return Err(shadow_choice(slot)),
    };
    let key = CacheKey {
        slot_id: slot.slot_id.clone(),
        scope_key: scope::scope_key(scope),
        role,
        implementation,
    };
    if let Some(run) = cache.get(&key) {
        return Ok((role, run));
    }
    let run = runs
        .latest(
            &key.slot_id,
            &key.scope_key,
            role,
            key.implementation.as_deref(),
        )
        .await?
        .ok_or(AppError::NoOutputYet)?;
    let run = Arc::new(run);
    cache.insert(key, run.clone());
    Ok((role, run))
}

fn parse_role(value: Option<&String>) -> Result<Role, AppError> {
    match value {
        None => Ok(Role::Production),
        Some(v) => Role::parse(v)
            .ok_or_else(|| AppError::InvalidQuery("role must be production or shadow".into())),
    }
}

fn shadow_choice(slot: &Slot) -> AppError {
    let ids = slot.shadow_ids();
    AppError::InvalidQuery(if ids.is_empty() {
        format!("slot '{}' has no shadow implementations", slot.slot_id)
    } else {
        format!(
            "role=shadow needs implementation=<id>; current shadows: {}",
            ids.join(", ")
        )
    })
}

fn set_header(res: &mut Response, name: header::HeaderName, value: &str) {
    if let Ok(v) = HeaderValue::from_str(value) {
        res.headers_mut().insert(name, v);
    }
}

/// Opaque to clients: `<as_of µs>.<run_id>` (A-5).
pub fn encode_cursor(run: &StoredRun) -> String {
    format!("{}.{}", run.as_of.timestamp_micros(), run.run_id.simple())
}

pub fn decode_cursor(cursor: &str) -> Result<(DateTime<Utc>, Uuid), AppError> {
    let bad = || AppError::InvalidQuery("cursor is invalid".into());
    let (micros, id) = cursor.split_once('.').ok_or_else(bad)?;
    let as_of = micros
        .parse::<i64>()
        .ok()
        .and_then(DateTime::from_timestamp_micros)
        .ok_or_else(bad)?;
    let run_id = Uuid::try_parse(id).map_err(|_| bad())?;
    Ok((as_of, run_id))
}
