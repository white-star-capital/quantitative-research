//! Integration tests against real Postgres (A-24). Each test gets a fresh database with
//! contracts/db/migrations applied. Run with:
//!   DATABASE_URL=postgres://… cargo test --test postgres -- --ignored
//!
//! Envelopes here are structural (payload `{}`); fixture-seeded tests over
//! contracts/fixtures/valid (A-25) are added with M0.

use std::{sync::Arc, time::Duration};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use options_terminal_api::{
    cache::{LatestCache, spawn_background},
    catalog::Catalog,
    config::Config,
    repo::{RunRepository, SlotRepository, postgres::PgRepo},
    routes::router,
    state::{AppState, Schemas},
};
use serde_json::{Value, json};
use sqlx::PgPool;
use tower::ServiceExt;
use uuid::Uuid;

async fn seed_slot(pool: &PgPool, slot_id: &str, production: &str, shadows: Value) {
    let definition = json!({
        "slot_id": slot_id,
        "scope": { "dimensions": ["underlying"], "allowed": { "underlying": ["BTC", "ETH"] } },
        "display": { "order": 10 }
    });
    sqlx::query(
        "INSERT INTO slots (slot_id, title, description, payload_kind, payload_version,
            scope_dimensions, max_age_seconds, definition, production_impl, production_version,
            shadow_implementations, status)
         VALUES ($1, $1, '', 'surface', '1', ARRAY['underlying'], 900, $2, $3, '1.0.0', $4, 'active')
         ON CONFLICT (slot_id) DO UPDATE SET production_impl = EXCLUDED.production_impl,
            shadow_implementations = EXCLUDED.shadow_implementations, updated_at = now()",
    )
    .bind(slot_id)
    .bind(definition)
    .bind(production)
    .bind(shadows)
    .execute(pool)
    .await
    .unwrap();
}

async fn seed_run(
    pool: &PgPool,
    slot_id: &str,
    implementation: &str,
    role: &str,
    minute: u32,
) -> Uuid {
    let run_id = Uuid::now_v7();
    let as_of = format!("2026-10-13T17:{minute:02}:00Z");
    let envelope = json!({
        "envelope_version": "1", "run_id": run_id, "slot_id": slot_id,
        "implementation_id": implementation, "implementation_version": "1.0.0",
        "scope": { "underlying": "BTC" }, "status": "success",
        "as_of": as_of, "started_at": as_of, "finished_at": as_of,
        "inputs": { "sources": [], "hash": format!("sha256:{}", "0".repeat(64)) },
        "code": { "git_sha": "a1b2c3d" }, "warnings": [],
        "payload_kind": "surface", "payload_version": "1", "payload": {}
    });
    sqlx::query(
        "INSERT INTO model_runs (run_id, slot_id, implementation_id, implementation_version, role,
            scope_key, scope, status, as_of, started_at, finished_at, input_hash, git_sha, envelope)
         VALUES ($1, $2, $3, '1.0.0', $4, 'underlying=BTC', '{\"underlying\":\"BTC\"}', 'success',
            $5::timestamptz, $5::timestamptz, $5::timestamptz, 'sha256:0', 'a1b2c3d', $6)",
    )
    .bind(run_id)
    .bind(slot_id)
    .bind(implementation)
    .bind(role)
    .bind(&as_of)
    .bind(envelope)
    .execute(pool)
    .await
    .unwrap();
    run_id
}

async fn app(pool: PgPool, background: bool) -> (axum::Router, Catalog) {
    let repo = Arc::new(PgRepo::new(pool.clone()));
    let slots: Arc<dyn SlotRepository> = repo.clone();
    let catalog = Catalog::default();
    let cache = LatestCache::new(Duration::from_secs(3600));
    catalog.reload(slots.as_ref()).await.unwrap();
    if background {
        spawn_background(
            pool,
            catalog.clone(),
            cache.clone(),
            slots.clone(),
            Duration::from_secs(3600),
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    let config = Config::from_lookup(|k| (k == "DATABASE_URL").then(|| "unused".into())).unwrap();
    let state = AppState {
        runs: repo as Arc<dyn RunRepository>,
        slots,
        catalog: catalog.clone(),
        cache,
        schemas: Arc::new(Schemas::default()),
        config: Arc::new(config),
        clock: Arc::new(chrono::Utc::now),
    };
    (router(state, None), catalog)
}

async fn get(app: &axum::Router, uri: &str) -> (StatusCode, Value) {
    let res = app
        .clone()
        .oneshot(Request::get(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        },
    )
}

async fn eventually(app: &axum::Router, uri: &str, pointer: &str, expected: &str) {
    for _ in 0..50 {
        if get(app, uri)
            .await
            .1
            .pointer(pointer)
            .and_then(Value::as_str)
            == Some(expected)
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("{uri} never had {pointer} = {expected}");
}

#[sqlx::test(migrations = "../contracts/db/migrations")]
#[ignore = "needs Postgres: DATABASE_URL=… cargo test --test postgres -- --ignored"]
async fn endpoints_against_postgres(pool: PgPool) {
    seed_slot(
        &pool,
        "vol_surface",
        "vol_surface_a",
        json!([{ "implementation": "vol_surface_b", "version": "0.1.0" }]),
    )
    .await;
    let first = seed_run(&pool, "vol_surface", "vol_surface_a", "production", 0).await;
    let latest_id = seed_run(&pool, "vol_surface", "vol_surface_a", "production", 5).await;
    seed_run(&pool, "vol_surface", "vol_surface_b", "shadow", 7).await;
    let (app, _) = app(pool, false).await;

    let (s, b) = get(&app, "/v1/slots").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(b["data"][0]["slot_id"], "vol_surface");

    let (s, b) = get(&app, "/v1/slots/vol_surface/latest?underlying=BTC").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(b["data"]["run_id"], latest_id.to_string());

    let (_, b) = get(
        &app,
        "/v1/slots/vol_surface/latest?underlying=BTC&role=shadow&implementation=vol_surface_b",
    )
    .await;
    assert_eq!(b["data"]["implementation_id"], "vol_surface_b");

    let (_, b) = get(&app, "/v1/slots/vol_surface/runs?underlying=BTC&limit=1").await;
    assert_eq!(b["data"][0]["run_id"], latest_id.to_string());
    assert!(b["data"][0].get("payload").is_none());
    let cursor = b["meta"]["next_cursor"].as_str().unwrap().to_owned();
    let (_, b) = get(
        &app,
        &format!("/v1/slots/vol_surface/runs?underlying=BTC&limit=1&cursor={cursor}"),
    )
    .await;
    assert_eq!(b["data"][0]["run_id"], first.to_string());
    assert_eq!(b["meta"]["next_cursor"], Value::Null);

    let (s, b) = get(&app, &format!("/v1/runs/{first}")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(b["data"]["run_id"], first.to_string());

    assert_eq!(get(&app, "/ready").await.0, StatusCode::OK);
}

#[sqlx::test(migrations = "../contracts/db/migrations")]
#[ignore = "needs Postgres: DATABASE_URL=… cargo test --test postgres -- --ignored"]
async fn models_come_and_go_via_notify(pool: PgPool) {
    seed_slot(&pool, "vol_surface", "vol_surface_a", json!([])).await;
    seed_run(&pool, "vol_surface", "vol_surface_a", "production", 0).await;
    let (app, _) = app(pool.clone(), true).await;
    let latest = "/v1/slots/vol_surface/latest?underlying=BTC";
    eventually(&app, latest, "/data/implementation_id", "vol_surface_a").await;

    // Promote B, add a new slot: registry sync + NOTIFY slots (C-13a).
    seed_slot(&pool, "vol_surface", "vol_surface_b", json!([])).await;
    seed_slot(&pool, "term_structure", "term_structure_a", json!([])).await;
    sqlx::query("NOTIFY slots").execute(&pool).await.unwrap();
    eventually(
        &app,
        "/v1/slots/vol_surface",
        "/implementations/production/implementation",
        "vol_surface_b",
    )
    .await;
    assert_eq!(
        get(&app, "/v1/slots/term_structure").await.0,
        StatusCode::OK
    );

    // B's first run + NOTIFY model_runs (C-13) replaces the cached A output.
    seed_run(&pool, "vol_surface", "vol_surface_b", "production", 10).await;
    sqlx::query("NOTIFY model_runs, 'vol_surface'")
        .execute(&pool)
        .await
        .unwrap();
    eventually(&app, latest, "/data/implementation_id", "vol_surface_b").await;

    // Retire the slot: /latest is gone, history stays.
    sqlx::query("UPDATE slots SET status = 'retired' WHERE slot_id = 'vol_surface'")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("NOTIFY slots").execute(&pool).await.unwrap();
    for _ in 0..50 {
        if get(&app, latest).await.0 == StatusCode::GONE {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(get(&app, latest).await.0, StatusCode::GONE);
    assert_eq!(
        get(&app, "/v1/slots/vol_surface/runs?underlying=BTC")
            .await
            .0,
        StatusCode::OK
    );
}

#[sqlx::test(migrations = "../contracts/db/migrations")]
#[ignore = "needs Postgres: DATABASE_URL=… cargo test --test postgres -- --ignored"]
async fn api_ro_role_cannot_write(pool: PgPool) {
    let mut conn = pool.acquire().await.unwrap();
    sqlx::query("SET ROLE api_ro")
        .execute(&mut *conn)
        .await
        .unwrap();
    assert!(
        sqlx::query("SELECT count(*) FROM model_runs")
            .execute(&mut *conn)
            .await
            .is_ok()
    );
    let err = sqlx::query("DELETE FROM model_runs")
        .execute(&mut *conn)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("permission denied"), "{err}");
}
