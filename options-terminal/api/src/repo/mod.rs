//! Data access behind traits so handlers can be tested without a database (A-23).

#[cfg(test)]
pub mod memory;
pub mod postgres;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::domain::{Role, Slot, StoredRun};

#[async_trait]
pub trait SlotRepository: Send + Sync {
    /// Every slot, active and retired. The catalog decides what to expose.
    async fn load_all(&self) -> Result<Vec<Slot>, sqlx::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunQuery {
    pub slot_id: String,
    pub scope_key: String,
    pub role: Role,
    pub implementation: Option<String>,
    pub status: Option<String>,
    pub from: Option<DateTime<Utc>>,
    pub to: Option<DateTime<Utc>>,
    /// Keyset cursor: rows strictly older than (as_of, run_id).
    pub before: Option<(DateTime<Utc>, Uuid)>,
    pub include_payload: bool,
    pub limit: i64,
}

#[async_trait]
pub trait RunRepository: Send + Sync {
    /// CONTRACTS C-11 / C-11a: newest successful run for slot, scope and role, optionally
    /// pinned to one implementation (required for shadows).
    async fn latest(
        &self,
        slot_id: &str,
        scope_key: &str,
        role: Role,
        implementation: Option<&str>,
    ) -> Result<Option<StoredRun>, sqlx::Error>;

    async fn by_id(&self, run_id: Uuid) -> Result<Option<StoredRun>, sqlx::Error>;

    /// Newest first. Envelopes have `payload` removed unless `include_payload`.
    async fn list(&self, query: &RunQuery) -> Result<Vec<StoredRun>, sqlx::Error>;

    async fn ping(&self) -> Result<(), sqlx::Error>;
}
