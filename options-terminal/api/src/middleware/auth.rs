//! Auth seam (A-17). A no-op in v1; API keys or OAuth later are a new `AuthMode` arm plus
//! config, with no route changes.

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};

use crate::config::AuthMode;

pub async fn auth(State(mode): State<AuthMode>, req: Request, next: Next) -> Response {
    match mode {
        AuthMode::None => next.run(req).await,
    }
}
