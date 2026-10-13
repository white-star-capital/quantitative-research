//! Composition root only: config → state → router → serve (§8).

use std::{process::ExitCode, sync::Arc, time::Duration};

use chrono::Utc;
use options_terminal_api::{
    cache::{LatestCache, spawn_background},
    catalog::Catalog,
    config::Config,
    repo::{RunRepository, SlotRepository, postgres::PgRepo},
    routes,
    state::{AppState, Schemas},
};
use sqlx::postgres::PgPoolOptions;
use tracing_subscriber::{EnvFilter, fmt};

const DB_CONNECT_ATTEMPTS: u32 = 5;
const SHUTDOWN_DRAIN: Duration = Duration::from_secs(10);

#[tokio::main]
async fn main() -> ExitCode {
    fmt()
        .json()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_current_span(true)
        .init();

    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            tracing::error!(%message, "startup failed");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), String> {
    let config = Arc::new(Config::from_env().map_err(|e| e.to_string())?);
    let schemas = Schemas::load(&config.contracts_dir).map_err(|e| {
        format!(
            "loading schemas from {}: {e}",
            config.contracts_dir.display()
        )
    })?;

    let pool = connect(&config).await?;
    let repo = Arc::new(PgRepo::new(pool.clone()));
    let runs: Arc<dyn RunRepository> = repo.clone();
    let slots: Arc<dyn SlotRepository> = repo;

    let catalog = Catalog::default();
    let cache = LatestCache::new(config.cache_ttl);
    catalog
        .reload(slots.as_ref())
        .await
        .map_err(|e| format!("loading slot catalog: {e}"))?;
    spawn_background(
        pool,
        catalog.clone(),
        cache.clone(),
        slots.clone(),
        config.catalog_refresh,
    );

    let state = AppState {
        runs,
        slots,
        catalog,
        cache,
        surfaces: options_terminal_api::state::surface_cache(),
        schemas: Arc::new(schemas),
        config: config.clone(),
        clock: Arc::new(Utc::now),
    };
    let limiter = routes::rate_limit(config.rate_per_minute, config.rate_burst)
        .ok_or("API_RATE_PER_MINUTE / API_RATE_BURST produce an invalid limiter")?;
    let eval_limiter = routes::rate_limit(config.eval_rate_per_minute, config.eval_rate_burst)
        .ok_or("API_EVAL_RATE_PER_MINUTE / API_EVAL_RATE_BURST produce an invalid limiter")?;
    let app = routes::router(state, Some(limiter), Some(eval_limiter));

    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .map_err(|e| format!("binding {}: {e}", config.bind))?;
    tracing::info!(bind = %config.bind, "listening");
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown())
    .await
    .map_err(|e| format!("server error: {e}"))
}

async fn connect(config: &Config) -> Result<sqlx::PgPool, String> {
    let options = PgPoolOptions::new()
        .max_connections(config.db_max_connections)
        .acquire_timeout(config.db_acquire_timeout);
    let mut wait = Duration::from_secs(1);
    for attempt in 1..=DB_CONNECT_ATTEMPTS {
        match options.clone().connect(&config.database_url).await {
            Ok(pool) => return Ok(pool),
            Err(e) if attempt < DB_CONNECT_ATTEMPTS => {
                tracing::warn!(attempt, error = %e, "database not reachable; retrying");
                tokio::time::sleep(wait).await;
                wait *= 2;
            }
            Err(e) => {
                return Err(format!(
                    "database unreachable after {attempt} attempts: {e}"
                ));
            }
        }
    }
    unreachable!("loop returns on the last attempt")
}

/// SIGTERM or Ctrl-C: stop accepting, then give in-flight requests up to 10 s (§6).
async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { () = ctrl_c => {}, () = term => {} }
    tracing::info!("shutting down; draining in-flight requests");
    tokio::spawn(async {
        tokio::time::sleep(SHUTDOWN_DRAIN).await;
        tracing::warn!("drain timeout reached; exiting");
        std::process::exit(0);
    });
}
