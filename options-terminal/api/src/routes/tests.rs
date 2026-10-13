//! Handler tests with in-memory repositories (A-23). The lifecycle tests are the point: slots
//! and implementations change underneath a running server.

use std::sync::{Arc, Mutex};

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode, header},
};
use chrono::{DateTime, Duration, Utc};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

use super::router;
use crate::{
    cache::LatestCache,
    catalog::Catalog,
    config::Config,
    domain::{ImplementationRef, Role, SlotStatus},
    repo::{RunRepository, SlotRepository, memory::MemoryRepo},
    state::{AppState, Schemas},
    test_support::{run, run_with_role, slot, t0},
};

struct Harness {
    repo: Arc<MemoryRepo>,
    catalog: Catalog,
    cache: LatestCache,
    now: Arc<Mutex<DateTime<Utc>>>,
    app: Router,
}

impl Harness {
    fn new() -> Self {
        let repo = Arc::new(MemoryRepo::default());
        let catalog = Catalog::default();
        let now = Arc::new(Mutex::new(t0() + Duration::seconds(60)));
        let clock_now = now.clone();
        let config = Config::from_lookup(|k| match k {
            "DATABASE_URL" => Some("postgres://unused".into()),
            "API_CORS_ORIGINS" => Some("http://localhost:5173".into()),
            _ => None,
        })
        .unwrap();
        // TTL long enough that tests prove invalidation, not expiry, refreshes the cache.
        let cache = LatestCache::new(std::time::Duration::from_secs(3600));
        let state = AppState {
            runs: repo.clone() as Arc<dyn RunRepository>,
            slots: repo.clone() as Arc<dyn SlotRepository>,
            catalog: catalog.clone(),
            cache: cache.clone(),
            schemas: Arc::new(Schemas::default()),
            config: Arc::new(config),
            clock: Arc::new(move || *clock_now.lock().unwrap()),
        };
        Self {
            repo,
            catalog,
            cache,
            now,
            app: router(state, None),
        }
    }

    /// What the registry sync + `NOTIFY slots` + catalog reload do in production.
    fn with_slots(self, slots: Vec<crate::domain::Slot>) -> Self {
        *self.repo.slots.lock().unwrap() = slots.clone();
        for id in self.catalog.replace(slots) {
            self.cache.invalidate_slot(&id);
        }
        self
    }

    /// What the runner's insert + `NOTIFY model_runs '<slot_id>'` do in production.
    fn new_run(&self, scope: &[(&str, &str)], status: &str, r: crate::domain::StoredRun) {
        let slot_id = r.slot_id.clone();
        self.repo.add_run(scope, status, r);
        self.cache.invalidate_slot(&slot_id);
    }

    async fn get(&self, uri: &str) -> (StatusCode, axum::http::HeaderMap, Value) {
        self.send(Method::GET, uri, &[]).await
    }

    async fn send(
        &self,
        method: Method,
        uri: &str,
        headers: &[(&str, &str)],
    ) -> (StatusCode, axum::http::HeaderMap, Value) {
        let mut req = Request::builder().method(method).uri(uri);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let res = self
            .app
            .clone()
            .oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = res.status();
        let headers = res.headers().clone();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        (status, headers, body)
    }
}

const BTC: &[(&str, &str)] = &[("underlying", "BTC")];

fn assert_problem(
    status: StatusCode,
    headers: &axum::http::HeaderMap,
    body: &Value,
    code: StatusCode,
    slug: &str,
) {
    assert_eq!(status, code, "{body}");
    assert_eq!(headers[header::CONTENT_TYPE], "application/problem+json");
    assert_eq!(
        body["type"],
        format!("https://options-terminal/problems/{slug}"),
        "{body}"
    );
    assert!(
        body["request_id"].as_str().is_some_and(|s| !s.is_empty()),
        "{body}"
    );
}

#[tokio::test]
async fn health_needs_no_db_ready_needs_catalog_and_db() {
    let h = Harness::new();
    assert_eq!(h.get("/health").await.0, StatusCode::OK);
    let (s, hd, b) = h.get("/ready").await;
    assert_problem(s, &hd, &b, StatusCode::SERVICE_UNAVAILABLE, "unavailable");

    let h = h.with_slots(vec![slot("vol_surface")]);
    assert_eq!(h.get("/ready").await.0, StatusCode::OK);
    *h.repo.down.lock().unwrap() = true;
    assert_eq!(h.get("/health").await.0, StatusCode::OK);
    assert_eq!(h.get("/ready").await.0, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn slot_list_is_active_only_and_ordered() {
    let mut retired = slot("old_metric");
    retired.status = SlotStatus::Retired;
    let mut first = slot("smile");
    first.definition = serde_json::json!({ "display": { "order": 1 } });
    let h = Harness::new().with_slots(vec![slot("vol_surface"), retired, first]);

    let (s, _, b) = h.get("/v1/slots").await;
    assert_eq!(s, StatusCode::OK);
    let ids: Vec<&str> = b["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["slot_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["smile", "vol_surface"]);
    assert_eq!(b["data"][0]["production_impl"], "smile_a");
    assert_eq!(b["meta"]["next_cursor"], Value::Null);

    // Retired slots remain describable.
    let (s, _, b) = h.get("/v1/slots/old_metric").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(b["status"], "retired");
}

#[tokio::test]
async fn latest_serves_envelope_with_meta_etag_and_304() {
    let h = Harness::new().with_slots(vec![slot("vol_surface")]);
    let r = run("vol_surface", "vol_surface_a", 0);
    h.repo.add_run(BTC, "success", r.clone());

    let (s, hd, b) = h.get("/v1/slots/vol_surface/latest?underlying=BTC").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(
        b["data"], r.envelope,
        "envelope passes through unchanged (A-7)"
    );
    assert_eq!(b["meta"]["role"], "production");
    assert_eq!(b["meta"]["age_seconds"], 60);
    assert_eq!(b["meta"]["stale"], false);
    let etag = hd[header::ETAG].to_str().unwrap().to_owned();
    assert_eq!(etag, format!("\"{}\"", r.run_id));
    assert_eq!(hd[header::CACHE_CONTROL], "public, max-age=30");
    assert_eq!(hd["x-content-type-options"], "nosniff");

    let (s, hd, _) = h
        .send(
            Method::GET,
            "/v1/slots/vol_surface/latest?underlying=BTC",
            &[("if-none-match", &etag)],
        )
        .await;
    assert_eq!(s, StatusCode::NOT_MODIFIED);
    assert_eq!(hd[header::ETAG], etag.as_str());
}

#[tokio::test]
async fn staleness_flips_at_max_age() {
    let h = Harness::new().with_slots(vec![slot("vol_surface")]);
    h.repo
        .add_run(BTC, "success", run("vol_surface", "vol_surface_a", 0));
    *h.now.lock().unwrap() = t0() + Duration::seconds(901);
    let (_, hd, b) = h.get("/v1/slots/vol_surface/latest?underlying=BTC").await;
    assert_eq!(
        b["meta"]["stale"], true,
        "stale outputs are still 200 (A-8)"
    );
    assert_eq!(hd[header::CACHE_CONTROL], "public, max-age=0");
}

#[tokio::test]
async fn latest_errors() {
    let mut retired = slot("old_metric");
    retired.status = SlotStatus::Retired;
    let h = Harness::new().with_slots(vec![slot("vol_surface"), retired]);
    h.repo
        .add_run(BTC, "failed", run("vol_surface", "vol_surface_a", 0));

    let cases = [
        (
            "/v1/slots/vol_surfce/latest?underlying=BTC",
            StatusCode::NOT_FOUND,
            "unknown-slot",
        ),
        (
            "/v1/slots/Bad-Id/latest",
            StatusCode::NOT_FOUND,
            "unknown-slot",
        ),
        (
            "/v1/slots/vol_surface/latest",
            StatusCode::BAD_REQUEST,
            "invalid-query",
        ),
        (
            "/v1/slots/vol_surface/latest?underlying=DOGE",
            StatusCode::BAD_REQUEST,
            "invalid-query",
        ),
        (
            "/v1/slots/vol_surface/latest?underlying=BTC&strike=1",
            StatusCode::BAD_REQUEST,
            "invalid-query",
        ),
        (
            "/v1/slots/vol_surface/latest?underlying=BTC&underlying=ETH",
            StatusCode::BAD_REQUEST,
            "invalid-query",
        ),
        (
            "/v1/slots/vol_surface/latest?underlying=BTC&role=staging",
            StatusCode::BAD_REQUEST,
            "invalid-query",
        ),
        // Failed runs never become latest (C-11).
        (
            "/v1/slots/vol_surface/latest?underlying=BTC",
            StatusCode::NOT_FOUND,
            "no-output-yet",
        ),
        (
            "/v1/slots/old_metric/latest?underlying=BTC",
            StatusCode::GONE,
            "slot-retired",
        ),
    ];
    for (uri, code, slug) in cases {
        let (s, hd, b) = h.get(uri).await;
        assert_problem(s, &hd, &b, code, slug);
    }
    let (_, _, b) = h.get("/v1/slots/vol_surface/latest?underlying=DOGE").await;
    assert!(
        b["detail"].as_str().unwrap().contains("BTC, ETH"),
        "400 lists allowed values (A-3)"
    );
}

#[tokio::test]
async fn any_number_of_shadows_each_addressable() {
    let mut s = slot("vol_surface");
    s.shadows = vec![
        ImplementationRef {
            implementation: "vol_surface_b".into(),
            version: "0.3.0".into(),
        },
        ImplementationRef {
            implementation: "vol_surface_c".into(),
            version: "0.1.0".into(),
        },
    ];
    let h = Harness::new().with_slots(vec![s]);
    h.repo
        .add_run(BTC, "success", run("vol_surface", "vol_surface_a", 0));
    h.repo.add_run(
        BTC,
        "success",
        run_with_role("vol_surface", "vol_surface_b", 10, Role::Shadow),
    );
    h.repo.add_run(
        BTC,
        "success",
        run_with_role("vol_surface", "vol_surface_c", 20, Role::Shadow),
    );

    let (st, hd, b) = h
        .get("/v1/slots/vol_surface/latest?underlying=BTC&role=shadow")
        .await;
    assert_problem(st, &hd, &b, StatusCode::BAD_REQUEST, "invalid-query");
    assert!(
        b["detail"]
            .as_str()
            .unwrap()
            .contains("vol_surface_b, vol_surface_c")
    );

    for imp in ["vol_surface_b", "vol_surface_c"] {
        let (st, _, b) = h
            .get(&format!(
                "/v1/slots/vol_surface/latest?underlying=BTC&role=shadow&implementation={imp}"
            ))
            .await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(b["data"]["implementation_id"], imp);
        assert_eq!(b["meta"]["role"], "shadow");
    }

    // Production is unaffected by newer shadow runs.
    let (_, _, b) = h.get("/v1/slots/vol_surface/latest?underlying=BTC").await;
    assert_eq!(b["data"]["implementation_id"], "vol_surface_a");

    // A retired shadow is no longer addressable on /latest.
    let (st, _, _) = h
        .get("/v1/slots/vol_surface/latest?underlying=BTC&role=shadow&implementation=vol_surface_x")
        .await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn promotion_rollback_and_retirement_without_restart() {
    let h = Harness::new().with_slots(vec![slot("vol_surface")]);
    h.repo
        .add_run(BTC, "success", run("vol_surface", "vol_surface_a", 0));
    let latest = "/v1/slots/vol_surface/latest?underlying=BTC";
    assert_eq!(
        h.get(latest).await.2["data"]["implementation_id"],
        "vol_surface_a"
    );

    // Promote B (registry sync rewrites the slots table; catalog reloads).
    let mut promoted = slot("vol_surface");
    promoted.production = Some(ImplementationRef {
        implementation: "vol_surface_b".into(),
        version: "1.0.0".into(),
    });
    let h = h.with_slots(vec![promoted]);
    // A-29: A's last output is served until B produces one.
    assert_eq!(
        h.get(latest).await.2["data"]["implementation_id"],
        "vol_surface_a"
    );
    assert_eq!(
        h.get("/v1/slots/vol_surface").await.2["implementations"]["production"]["implementation"],
        "vol_surface_b"
    );

    // B's first successful run takes over.
    h.new_run(BTC, "success", run("vol_surface", "vol_surface_b", 30));
    assert_eq!(
        h.get(latest).await.2["data"]["implementation_id"],
        "vol_surface_b"
    );

    // Rollback to A: B's output stays latest until A runs again (A-29).
    let h = h.with_slots(vec![slot("vol_surface")]);
    assert_eq!(
        h.get(latest).await.2["data"]["implementation_id"],
        "vol_surface_b"
    );
    h.new_run(BTC, "success", run("vol_surface", "vol_surface_a", 60));
    assert_eq!(
        h.get(latest).await.2["data"]["implementation_id"],
        "vol_surface_a"
    );

    // New slot appears and the old one retires, with no restart.
    let mut retired = slot("vol_surface");
    retired.status = SlotStatus::Retired;
    let h = h.with_slots(vec![retired, slot("term_structure")]);
    assert_eq!(h.get(latest).await.0, StatusCode::GONE);
    assert_eq!(h.get("/v1/slots/term_structure").await.0, StatusCode::OK);
    // History of a retired slot stays readable.
    let (st, _, b) = h.get("/v1/slots/vol_surface/runs?underlying=BTC").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(b["data"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn runs_paginate_newest_first_without_payload_by_default() {
    let h = Harness::new().with_slots(vec![slot("vol_surface")]);
    for i in 0..3 {
        h.repo
            .add_run(BTC, "success", run("vol_surface", "vol_surface_a", i * 60));
    }
    h.repo
        .add_run(BTC, "failed", run("vol_surface", "vol_surface_a", 500));

    let (s, _, b) = h
        .get("/v1/slots/vol_surface/runs?underlying=BTC&status=success&limit=2")
        .await;
    assert_eq!(s, StatusCode::OK);
    let page = b["data"].as_array().unwrap();
    assert_eq!(page.len(), 2);
    assert!(page[0]["as_of"].as_str() > page[1]["as_of"].as_str());
    assert!(
        page[0].get("payload").is_none(),
        "payload omitted unless include_payload=true"
    );
    let cursor = b["meta"]["next_cursor"].as_str().unwrap().to_owned();

    let (_, _, b) = h
        .get(&format!(
            "/v1/slots/vol_surface/runs?underlying=BTC&status=success&limit=2&cursor={cursor}"
        ))
        .await;
    assert_eq!(b["data"].as_array().unwrap().len(), 1);
    assert_eq!(b["meta"]["next_cursor"], Value::Null);

    let (_, _, b) = h
        .get("/v1/slots/vol_surface/runs?underlying=BTC&include_payload=true&limit=1")
        .await;
    assert!(b["data"][0].get("payload").is_some());

    for bad in [
        "limit=0",
        "limit=501",
        "include_payload=true&limit=51",
        "status=ok",
        "cursor=xyz",
        "from=2026-10-13T17:00:00%2B02:00",
        "implementation=Not-An-Id",
    ] {
        let (s, _, _) = h
            .get(&format!("/v1/slots/vol_surface/runs?underlying=BTC&{bad}"))
            .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{bad}");
    }
}

#[tokio::test]
async fn run_by_id_is_immutable_and_any_role() {
    let h = Harness::new().with_slots(vec![slot("vol_surface")]);
    let r = run_with_role("vol_surface", "vol_surface_b", 0, Role::Shadow);
    h.repo.add_run(BTC, "success", r.clone());

    let (s, hd, b) = h.get(&format!("/v1/runs/{}", r.run_id)).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(b["data"], r.envelope);
    assert_eq!(b["meta"]["role"], "shadow");
    assert_eq!(
        hd[header::CACHE_CONTROL],
        "public, max-age=31536000, immutable"
    );

    let (s, hd, b) = h.get("/v1/runs/not-a-uuid").await;
    assert_problem(s, &hd, &b, StatusCode::NOT_FOUND, "unknown-run");
}

#[tokio::test]
async fn non_get_unknown_paths_and_schemas_are_problems() {
    let h = Harness::new().with_slots(vec![slot("vol_surface")]);
    let (s, hd, b) = h.send(Method::POST, "/v1/slots", &[]).await;
    assert_problem(
        s,
        &hd,
        &b,
        StatusCode::METHOD_NOT_ALLOWED,
        "method-not-allowed",
    );
    let (s, hd, b) = h.send(Method::DELETE, "/v1/runs/x", &[]).await;
    assert_problem(
        s,
        &hd,
        &b,
        StatusCode::METHOD_NOT_ALLOWED,
        "method-not-allowed",
    );
    let (s, hd, b) = h.get("/v2/anything").await;
    assert_problem(s, &hd, &b, StatusCode::NOT_FOUND, "not-found");
    let (s, hd, b) = h.get("/v1/schemas/surface/1").await;
    assert_problem(s, &hd, &b, StatusCode::NOT_FOUND, "unknown-schema");
}

#[tokio::test]
async fn openapi_documents_every_route() {
    let h = Harness::new();
    let (s, _, b) = h.get("/openapi.json").await;
    assert_eq!(s, StatusCode::OK);
    assert!(b["openapi"].as_str().unwrap().starts_with("3.1"));
    for path in [
        "/health",
        "/ready",
        "/v1/slots",
        "/v1/slots/{slot_id}",
        "/v1/slots/{slot_id}/latest",
        "/v1/slots/{slot_id}/runs",
        "/v1/runs/{run_id}",
        "/v1/schemas/{kind}/{version}",
    ] {
        assert!(b["paths"].get(path).is_some(), "missing {path}");
    }
}

#[tokio::test]
async fn cors_allows_configured_origin_only() {
    let h = Harness::new().with_slots(vec![slot("vol_surface")]);
    let (_, hd, _) = h
        .send(
            Method::GET,
            "/v1/slots",
            &[("origin", "http://localhost:5173")],
        )
        .await;
    assert_eq!(
        hd[header::ACCESS_CONTROL_ALLOW_ORIGIN],
        "http://localhost:5173"
    );
    let (_, hd, _) = h
        .send(
            Method::GET,
            "/v1/slots",
            &[("origin", "https://evil.example")],
        )
        .await;
    assert!(hd.get(header::ACCESS_CONTROL_ALLOW_ORIGIN).is_none());
}
