//! In-memory repositories for handler unit tests (A-23).

use std::sync::Mutex;

use async_trait::async_trait;
use uuid::Uuid;

use super::{RunQuery, RunRepository, SlotRepository};
use crate::domain::{Role, Slot, StoredRun, scope};

#[derive(Default)]
pub struct MemoryRepo {
    pub slots: Mutex<Vec<Slot>>,
    /// (scope_key, status, run)
    pub runs: Mutex<Vec<(String, String, StoredRun)>>,
    pub down: Mutex<bool>,
}

impl MemoryRepo {
    pub fn add_run(&self, scope: &[(&str, &str)], status: &str, run: StoredRun) {
        let scope = scope
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        self.runs
            .lock()
            .unwrap()
            .push((scope::scope_key(&scope), status.to_owned(), run));
    }

    fn check(&self) -> Result<(), sqlx::Error> {
        if *self.down.lock().unwrap() {
            Err(sqlx::Error::PoolTimedOut)
        } else {
            Ok(())
        }
    }
}

#[async_trait]
impl SlotRepository for MemoryRepo {
    async fn load_all(&self) -> Result<Vec<Slot>, sqlx::Error> {
        self.check()?;
        Ok(self.slots.lock().unwrap().clone())
    }
}

#[async_trait]
impl RunRepository for MemoryRepo {
    async fn latest(
        &self,
        slot_id: &str,
        scope_key: &str,
        role: Role,
        implementation: Option<&str>,
    ) -> Result<Option<StoredRun>, sqlx::Error> {
        self.check()?;
        Ok(self
            .runs
            .lock()
            .unwrap()
            .iter()
            .filter(|(k, status, r)| {
                r.slot_id == slot_id
                    && k == scope_key
                    && r.role == role
                    && status == "success"
                    && implementation.is_none_or(|i| r.implementation_id == i)
            })
            .max_by_key(|(_, _, r)| (r.as_of, r.run_id))
            .map(|(_, _, r)| r.clone()))
    }

    async fn by_id(&self, run_id: Uuid) -> Result<Option<StoredRun>, sqlx::Error> {
        self.check()?;
        Ok(self
            .runs
            .lock()
            .unwrap()
            .iter()
            .find(|(_, _, r)| r.run_id == run_id)
            .map(|(_, _, r)| r.clone()))
    }

    async fn list(&self, q: &RunQuery) -> Result<Vec<StoredRun>, sqlx::Error> {
        self.check()?;
        let mut rows: Vec<StoredRun> = self
            .runs
            .lock()
            .unwrap()
            .iter()
            .filter(|(k, status, r)| {
                r.slot_id == q.slot_id
                    && *k == q.scope_key
                    && r.role == q.role
                    && q.implementation
                        .as_ref()
                        .is_none_or(|i| &r.implementation_id == i)
                    && q.status.as_ref().is_none_or(|s| s == status)
                    && q.from.is_none_or(|f| r.as_of >= f)
                    && q.to.is_none_or(|t| r.as_of <= t)
                    && q.before.is_none_or(|b| (r.as_of, r.run_id) < b)
            })
            .map(|(_, _, r)| r.clone())
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse((r.as_of, r.run_id)));
        rows.truncate(usize::try_from(q.limit).unwrap_or(0));
        if !q.include_payload {
            for r in &mut rows {
                if let Some(obj) = r.envelope.as_object_mut() {
                    obj.remove("payload");
                }
            }
        }
        Ok(rows)
    }

    async fn ping(&self) -> Result<(), sqlx::Error> {
        self.check()
    }
}
