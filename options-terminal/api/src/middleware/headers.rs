//! A-16: security headers on every response. The API serves JSON only.

use axum::http::{HeaderName, HeaderValue, header};
use tower_http::set_header::SetResponseHeaderLayer;

pub fn security_headers() -> [SetResponseHeaderLayer<HeaderValue>; 3] {
    let set = |name: HeaderName, value: &'static str| {
        SetResponseHeaderLayer::overriding(name, HeaderValue::from_static(value))
    };
    [
        set(header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        set(header::REFERRER_POLICY, "no-referrer"),
        set(
            header::CONTENT_SECURITY_POLICY,
            "default-src 'none'; frame-ancestors 'none'",
        ),
    ]
}
