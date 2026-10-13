//! Postgres repositories. Connected as `api_ro` (A-2); every query is parameter-bound (A-20)
//! and checked at compile time against `.sqlx/` offline metadata (§9.3).

use std::collections::BTreeMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

use super::{RunQuery, RunRepository, SlotRepository};
use crate::domain::{ImplementationRef, Role, Slot, SlotStatus, StoredRun};

#[derive(Clone)]
pub struct PgRepo {
    pool: PgPool,
}

impl PgRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

fn decode_err(msg: String) -> sqlx::Error {
    sqlx::Error::Decode(msg.into())
}

fn run(
    run_id: Uuid,
    slot_id: String,
    implementation_id: String,
    role: String,
    as_of: DateTime<Utc>,
    envelope: Value,
) -> Result<StoredRun, sqlx::Error> {
    let role = Role::parse(&role).ok_or_else(|| decode_err(format!("bad role '{role}'")))?;
    Ok(StoredRun {
        run_id,
        slot_id,
        implementation_id,
        role,
        as_of,
        envelope,
    })
}

#[async_trait]
impl SlotRepository for PgRepo {
    async fn load_all(&self) -> Result<Vec<Slot>, sqlx::Error> {
        let rows = sqlx::query!(
            r#"SELECT slot_id, title, description, payload_kind, payload_version,
                      scope_dimensions, max_age_seconds, definition, production_impl,
                      production_version, shadow_implementations, status
               FROM slots ORDER BY slot_id"#
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|r| {
                let allowed: BTreeMap<String, Vec<String>> = r
                    .definition
                    .pointer("/scope/allowed")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|e| decode_err(format!("{}: scope.allowed: {e}", r.slot_id)))?
                    .unwrap_or_default();
                let shadows: Vec<ImplementationRef> =
                    serde_json::from_value(r.shadow_implementations).map_err(|e| {
                        decode_err(format!("{}: shadow_implementations: {e}", r.slot_id))
                    })?;
                let status = match r.status.as_str() {
                    "active" => SlotStatus::Active,
                    "retired" => SlotStatus::Retired,
                    other => return Err(decode_err(format!("{}: status '{other}'", r.slot_id))),
                };
                let production = match (r.production_impl, r.production_version) {
                    (Some(implementation), Some(version)) => Some(ImplementationRef {
                        implementation,
                        version,
                    }),
                    _ => None,
                };
                Ok(Slot {
                    slot_id: r.slot_id,
                    title: r.title,
                    description: r.description,
                    payload_kind: r.payload_kind,
                    payload_version: r.payload_version,
                    dimensions: r.scope_dimensions,
                    allowed,
                    max_age_seconds: i64::from(r.max_age_seconds),
                    production,
                    shadows,
                    status,
                    definition: r.definition,
                })
            })
            .collect()
    }
}

#[async_trait]
impl RunRepository for PgRepo {
    async fn latest(
        &self,
        slot_id: &str,
        scope_key: &str,
        role: Role,
        implementation: Option<&str>,
    ) -> Result<Option<StoredRun>, sqlx::Error> {
        let row = sqlx::query!(
            r#"SELECT run_id, slot_id, implementation_id, role, as_of, envelope
               FROM model_runs
               WHERE slot_id = $1 AND scope_key = $2 AND role = $3 AND status = 'success'
                 AND ($4::text IS NULL OR implementation_id = $4)
               ORDER BY as_of DESC, run_id DESC
               LIMIT 1"#,
            slot_id,
            scope_key,
            role.as_str(),
            implementation,
        )
        .fetch_optional(&self.pool)
        .await?;
        row.map(|r| {
            run(
                r.run_id,
                r.slot_id,
                r.implementation_id,
                r.role,
                r.as_of,
                r.envelope,
            )
        })
        .transpose()
    }

    async fn by_id(&self, run_id: Uuid) -> Result<Option<StoredRun>, sqlx::Error> {
        let row = sqlx::query!(
            r#"SELECT run_id, slot_id, implementation_id, role, as_of, envelope
               FROM model_runs WHERE run_id = $1"#,
            run_id,
        )
        .fetch_optional(&self.pool)
        .await?;
        row.map(|r| {
            run(
                r.run_id,
                r.slot_id,
                r.implementation_id,
                r.role,
                r.as_of,
                r.envelope,
            )
        })
        .transpose()
    }

    async fn list(&self, q: &RunQuery) -> Result<Vec<StoredRun>, sqlx::Error> {
        let (before_as_of, before_id) = q.before.unzip();
        let rows = sqlx::query!(
            r#"SELECT run_id, slot_id, implementation_id, role, as_of,
                      CASE WHEN $10 THEN envelope ELSE envelope - 'payload' END AS "envelope!"
               FROM model_runs
               WHERE slot_id = $1 AND scope_key = $2 AND role = $3
                 AND ($4::text IS NULL OR implementation_id = $4)
                 AND ($5::text IS NULL OR status = $5)
                 AND ($6::timestamptz IS NULL OR as_of >= $6)
                 AND ($7::timestamptz IS NULL OR as_of <= $7)
                 AND ($8::timestamptz IS NULL OR (as_of, run_id) < ($8, $9::uuid))
               ORDER BY as_of DESC, run_id DESC
               LIMIT $11"#,
            q.slot_id,
            q.scope_key,
            q.role.as_str(),
            q.implementation.as_deref(),
            q.status.as_deref(),
            q.from,
            q.to,
            before_as_of,
            before_id,
            q.include_payload,
            q.limit,
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|r| {
                run(
                    r.run_id,
                    r.slot_id,
                    r.implementation_id,
                    r.role,
                    r.as_of,
                    r.envelope,
                )
            })
            .collect()
    }

    async fn ping(&self) -> Result<(), sqlx::Error> {
        sqlx::query_scalar!("SELECT 1 AS \"one!\"")
            .fetch_one(&self.pool)
            .await
            .map(|_| ())
    }
}
