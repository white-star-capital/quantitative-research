use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderValue, header},
    response::{IntoResponse, Response},
};
use uuid::Uuid;

use super::slots::EnvelopeResponse;
use crate::{
    catalog::Catalog, domain::RunMeta, error::AppError, repo::RunRepository, state::Clock,
};

/// One envelope by run ID, any role and status. Immutable, so cached for a year (A-12).
#[utoipa::path(get, path = "/v1/runs/{run_id}", responses(
    (status = 200, body = EnvelopeResponse), (status = 404, description = "unknown-run"),
))]
pub async fn get_run(
    State(repo): State<Arc<dyn RunRepository>>,
    State(catalog): State<Catalog>,
    State(clock): State<Clock>,
    Path(run_id): Path<String>,
) -> Result<Response, AppError> {
    let id = Uuid::try_parse(&run_id).map_err(|_| AppError::UnknownRun(run_id.clone()))?;
    let run = repo.by_id(id).await?.ok_or(AppError::UnknownRun(run_id))?;
    // model_runs.slot_id references slots, so the slot exists unless the catalog is mid-reload.
    let max_age = catalog
        .get(&run.slot_id)
        .map(|s| s.max_age_seconds)
        .ok_or(AppError::Unavailable("slot catalog not loaded"))?;
    tracing::info!(run_id = %run.run_id, slot_id = %run.slot_id, "served run");
    let meta = RunMeta::new(run.role, run.as_of, max_age, clock());
    let mut res = Json(EnvelopeResponse::new(run.envelope, meta)).into_response();
    res.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=31536000, immutable"),
    );
    Ok(res)
}
