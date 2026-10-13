//! Every error response becomes `application/problem+json` with a full `type` URL and the
//! `request_id` (A-10), including errors produced by axum and tower layers (404, 405, 408,
//! 413, 429), which would otherwise be plain text.

use std::sync::Arc;

use axum::{
    body::{Body, to_bytes},
    extract::{Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::Next,
    response::Response,
};
use serde_json::{Value, json};

use crate::error::PROBLEM_JSON;

pub const REQUEST_ID: &str = "x-request-id";
const MAX_ERROR_BODY: usize = 64 * 1024;

#[derive(Clone)]
pub struct ProblemBase(pub Arc<str>);

pub async fn problem(State(base): State<ProblemBase>, req: Request, next: Next) -> Response {
    let request_id = req
        .headers()
        .get(REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let res = next.run(req).await;
    let status = res.status();
    if !(status.is_client_error() || status.is_server_error()) {
        return res;
    }

    let (mut parts, body) = res.into_parts();
    let is_problem = parts
        .headers
        .get(header::CONTENT_TYPE)
        .is_some_and(|v| v.as_bytes().starts_with(PROBLEM_JSON.as_bytes()));
    let bytes = to_bytes(body, MAX_ERROR_BODY).await.unwrap_or_default();

    let mut value: Value = if is_problem {
        serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({}))
    } else {
        let detail = if status.is_server_error() {
            "The server could not complete the request".to_owned()
        } else {
            String::from_utf8_lossy(&bytes).trim().to_owned()
        };
        json!({
            "type": slug_for(status),
            "title": status.canonical_reason().unwrap_or("Error"),
            "status": status.as_u16(),
            "detail": if detail.is_empty() { Value::Null } else { Value::String(detail) },
        })
    };

    if let Some(obj) = value.as_object_mut() {
        let slug = obj
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("error")
            .to_owned();
        if !slug.contains("://") {
            obj.insert("type".into(), Value::String(format!("{}/{slug}", base.0)));
        }
        obj.retain(|_, v| !v.is_null());
        obj.insert(
            "request_id".into(),
            request_id.map_or(Value::Null, Value::String),
        );
    }

    parts
        .headers
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(PROBLEM_JSON));
    parts.headers.remove(header::CONTENT_LENGTH);
    Response::from_parts(
        parts,
        Body::from(serde_json::to_vec(&value).unwrap_or_default()),
    )
}

fn slug_for(status: StatusCode) -> &'static str {
    match status {
        StatusCode::NOT_FOUND => "not-found",
        StatusCode::METHOD_NOT_ALLOWED => "method-not-allowed",
        StatusCode::BAD_REQUEST => "invalid-query",
        StatusCode::REQUEST_TIMEOUT | StatusCode::SERVICE_UNAVAILABLE => "unavailable",
        StatusCode::PAYLOAD_TOO_LARGE => "request-too-large",
        StatusCode::TOO_MANY_REQUESTS => "rate-limited",
        s if s.is_server_error() => "internal",
        _ => "error",
    }
}
