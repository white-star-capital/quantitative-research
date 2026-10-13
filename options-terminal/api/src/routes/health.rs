use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde_json::json;

use crate::{catalog::Catalog, error::AppError, repo::RunRepository};

/// Liveness only; no DB call.
#[utoipa::path(get, path = "/health", responses((status = 200, description = "Process is up")))]
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// 200 when the DB answers and the slot catalog has loaded (A-28); 503 otherwise.
#[utoipa::path(get, path = "/ready", responses(
    (status = 200, description = "Ready"),
    (status = 503, description = "Database unreachable or catalog not loaded"),
))]
pub async fn ready(
    State(runs): State<Arc<dyn RunRepository>>,
    State(catalog): State<Catalog>,
) -> Result<impl IntoResponse, AppError> {
    if catalog.snapshot().is_none() {
        return Err(AppError::Unavailable("slot catalog not loaded"));
    }
    runs.ping()
        .await
        .map_err(|_| AppError::Unavailable("database unreachable"))?;
    Ok((StatusCode::OK, Json(json!({ "status": "ready" }))))
}
