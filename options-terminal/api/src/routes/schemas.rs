use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
};
use serde_json::Value;

use crate::{error::AppError, state::Schemas};

/// JSON Schema for a payload kind and version, as published in `contracts/` (CONTRACTS C-1).
#[utoipa::path(get, path = "/v1/schemas/{kind}/{version}", responses(
    (status = 200, description = "JSON Schema (draft 2020-12)"),
    (status = 404, description = "unknown-schema"),
))]
pub async fn get_schema(
    State(schemas): State<Arc<Schemas>>,
    Path((kind, version)): Path<(String, String)>,
) -> Result<Json<Value>, AppError> {
    let version = version.strip_prefix('v').unwrap_or(&version).to_owned();
    schemas
        .0
        .get(&(kind.clone(), version.clone()))
        .map(|s| Json(s.as_ref().clone()))
        .ok_or(AppError::UnknownSchema(kind, version))
}
