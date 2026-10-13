//! One error type → RFC 7807 `application/problem+json` (A-10). No internals leak: 500s carry
//! a generic detail and the real cause goes to the log. `request_id` is added by
//! `middleware::problem`, which also converts framework errors (405, 408, 413, 429).

use axum::{
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Serialize;

pub const PROBLEM_JSON: &str = "application/problem+json";

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Slot '{0}' does not exist")]
    UnknownSlot(String),
    #[error("Slot '{0}' is retired; its run history is still available")]
    SlotRetired(String),
    #[error("No successful run yet for this slot and scope")]
    NoOutputYet,
    #[error("Run '{0}' does not exist")]
    UnknownRun(String),
    #[error("No schema for payload kind '{0}' version '{1}'")]
    UnknownSchema(String, String),
    #[error("{0}")]
    InvalidQuery(String),
    #[error("{0}")]
    Unavailable(&'static str),
    #[error("internal error")]
    Internal(String),
}

impl AppError {
    pub fn slug(&self) -> &'static str {
        match self {
            Self::UnknownSlot(_) => "unknown-slot",
            Self::SlotRetired(_) => "slot-retired",
            Self::NoOutputYet => "no-output-yet",
            Self::UnknownRun(_) => "unknown-run",
            Self::UnknownSchema(..) => "unknown-schema",
            Self::InvalidQuery(_) => "invalid-query",
            Self::Unavailable(_) => "unavailable",
            Self::Internal(_) => "internal",
        }
    }

    pub fn status(&self) -> StatusCode {
        match self {
            Self::UnknownSlot(_)
            | Self::NoOutputYet
            | Self::UnknownRun(_)
            | Self::UnknownSchema(..) => StatusCode::NOT_FOUND,
            Self::SlotRetired(_) => StatusCode::GONE,
            Self::InvalidQuery(_) => StatusCode::BAD_REQUEST,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl From<sqlx::Error> for AppError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::PoolTimedOut => Self::Unavailable("database unavailable"),
            other => Self::Internal(other.to_string()),
        }
    }
}

/// Body before `request_id` is attached. `type` is a slug here; the problem middleware expands
/// it to `<API_PROBLEM_BASE_URL>/<slug>`.
#[derive(Debug, Serialize)]
pub struct Problem {
    #[serde(rename = "type")]
    pub kind: String,
    pub title: String,
    pub status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

pub fn problem_response(status: StatusCode, slug: &str, detail: Option<String>) -> Response {
    let body = Problem {
        kind: slug.to_owned(),
        title: status.canonical_reason().unwrap_or("Error").to_owned(),
        status: status.as_u16(),
        detail,
    };
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
    let mut res = (status, bytes).into_response();
    res.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(PROBLEM_JSON));
    res
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let detail = match &self {
            Self::Internal(cause) => {
                tracing::error!(error = %cause, "internal error");
                "Internal error; quote the request_id when reporting it".to_owned()
            }
            other => other.to_string(),
        };
        problem_response(self.status(), self.slug(), Some(detail))
    }
}
