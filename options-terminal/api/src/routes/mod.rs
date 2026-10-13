//! Router assembly: generic routes under /v1, the auth seam, and the middleware stack.

pub mod health;
pub mod params;
pub mod runs;
pub mod schemas;
pub mod slots;

use std::sync::Arc;

use axum::{
    Json, Router,
    body::Body,
    http::{HeaderName, Method, Request, StatusCode, header},
    middleware::from_fn_with_state,
    routing::get,
};
use governor::middleware::NoOpMiddleware;
use tower_governor::{
    GovernorLayer,
    governor::{GovernorConfig, GovernorConfigBuilder},
    key_extractor::PeerIpKeyExtractor,
};
use tower_http::{
    compression::CompressionLayer,
    cors::{AllowOrigin, CorsLayer},
    limit::RequestBodyLimitLayer,
    map_request_body::MapRequestBodyLayer,
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    timeout::TimeoutLayer,
    trace::{DefaultOnResponse, TraceLayer},
};
use tracing::Level;
use utoipa::OpenApi;

use crate::{
    middleware::{
        auth::auth,
        headers::security_headers,
        problem::{ProblemBase, REQUEST_ID, problem},
    },
    state::AppState,
};

const BODY_LIMIT: usize = 16 * 1024;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Options Terminal API",
        version = "1",
        description = "Read-only access to stored model runs. Slots and implementations are data: they come and go without API changes."
    ),
    paths(
        health::health,
        health::ready,
        slots::list_slots,
        slots::get_slot,
        slots::latest,
        slots::runs,
        runs::get_run,
        schemas::get_schema,
    ),
    components(schemas(
        slots::SlotList,
        slots::EnvelopeResponse,
        slots::RunList,
        crate::domain::RunMeta,
        crate::domain::Role,
        crate::domain::SlotStatus,
        crate::domain::ImplementationRef,
        crate::domain::slot::SlotSummary,
        crate::domain::slot::SlotDetail,
        crate::domain::slot::ScopeSpec,
        crate::domain::slot::Implementations,
    ))
)]
pub struct ApiDoc;

async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}

/// Per-IP limiter keyed on the socket address (A-18: forwarded headers are not trusted until
/// proxy CIDRs are configured).
pub type RateLimit = Arc<GovernorConfig<PeerIpKeyExtractor, NoOpMiddleware>>;

pub fn rate_limit(per_minute: u32, burst: u32) -> Option<RateLimit> {
    GovernorConfigBuilder::default()
        .per_millisecond(60_000 / u64::from(per_minute))
        .burst_size(burst)
        .finish()
        .map(Arc::new)
}

/// GET-only routes (A-1); axum answers other methods on these paths with 405.
/// `rate_limit` is None in handler tests, which run without a socket address.
pub fn router(state: AppState, rate_limit: Option<RateLimit>) -> Router {
    let config = state.config.clone();

    let mut v1 = Router::new()
        .route("/slots", get(slots::list_slots))
        .route("/slots/{slot_id}", get(slots::get_slot))
        .route("/slots/{slot_id}/latest", get(slots::latest))
        .route("/slots/{slot_id}/runs", get(slots::runs))
        .route("/runs/{run_id}", get(runs::get_run))
        .route("/schemas/{kind}/{version}", get(schemas::get_schema))
        .layer(from_fn_with_state(state.clone(), auth));
    // Probes (/health, /ready) are not rate limited, so load balancers never see 429.
    if let Some(config) = rate_limit {
        v1 = v1.layer(GovernorLayer::<_, _, Body>::new(config));
    }

    let app: Router = Router::new()
        .route("/health", get(health::health))
        .route("/ready", get(health::ready))
        .route("/openapi.json", get(openapi))
        .nest("/v1", v1)
        .fallback(|| async { StatusCode::NOT_FOUND })
        .with_state(state);

    let request_id = HeaderName::from_static(REQUEST_ID);
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(config.cors_origins.clone()))
        .allow_methods([Method::GET, Method::HEAD, Method::OPTIONS])
        .allow_headers([header::IF_NONE_MATCH])
        .expose_headers([header::ETAG, request_id.clone()]);
    let [nosniff, referrer, csp] = security_headers();

    // Router::layer wraps outward: the last layer added sees the request first.
    app.layer(MapRequestBodyLayer::new(Body::new))
        .layer(RequestBodyLimitLayer::new(BODY_LIMIT))
        .layer(TimeoutLayer::with_status_code(StatusCode::SERVICE_UNAVAILABLE, config.request_timeout))
        .layer(cors)
        .layer(from_fn_with_state(ProblemBase(Arc::from(config.problem_base_url.as_str())), problem))
        .layer(csp)
        .layer(referrer)
        .layer(nosniff)
        .layer(CompressionLayer::new())
        .layer(PropagateRequestIdLayer::new(request_id.clone()))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|req: &Request<Body>| {
                    let id = req.headers().get(REQUEST_ID).and_then(|v| v.to_str().ok()).unwrap_or("");
                    tracing::info_span!("request", method = %req.method(), path = %req.uri().path(), request_id = %id)
                })
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .layer(SetRequestIdLayer::new(request_id, MakeRequestUuid))
}

#[cfg(test)]
mod tests;
