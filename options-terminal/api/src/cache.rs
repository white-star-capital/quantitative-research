//! Latest-envelope cache (A-13, A-14, A-30) and the Postgres LISTEN task that keeps it and the
//! slot catalog current.

use std::{sync::Arc, time::Duration};

use moka::sync::Cache;
use sqlx::{PgPool, postgres::PgListener};

use crate::{
    catalog::Catalog,
    domain::{Role, StoredRun},
    repo::SlotRepository,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub slot_id: String,
    pub scope_key: String,
    pub role: Role,
    /// None = any implementation (production lookups); Some for shadows (A-30).
    pub implementation: Option<String>,
}

#[derive(Clone)]
pub struct LatestCache {
    inner: Cache<CacheKey, Arc<StoredRun>>,
}

impl LatestCache {
    pub fn new(ttl: Duration) -> Self {
        Self {
            inner: Cache::builder()
                .max_capacity(10_000)
                .time_to_live(ttl)
                .support_invalidation_closures()
                .build(),
        }
    }

    pub fn get(&self, key: &CacheKey) -> Option<Arc<StoredRun>> {
        self.inner.get(key)
    }

    pub fn insert(&self, key: CacheKey, run: Arc<StoredRun>) {
        self.inner.insert(key, run);
    }

    pub fn invalidate_slot(&self, slot_id: &str) {
        let slot_id = slot_id.to_owned();
        // Only fails if closures weren't enabled at build time; fall back to a full clear.
        if self
            .inner
            .invalidate_entries_if(move |k, _| k.slot_id == slot_id)
            .is_err()
        {
            self.clear();
        }
    }

    pub fn clear(&self) {
        self.inner.invalidate_all();
    }
}

/// LISTEN on `model_runs` (new run → drop that slot's entries, C-13) and `slots` (registry
/// sync → reload catalog, C-13a). On any listener failure: clear the cache, back off,
/// reconnect (A-13). Also reloads the catalog every `refresh` in case a NOTIFY was missed.
pub fn spawn_background(
    pool: PgPool,
    catalog: Catalog,
    cache: LatestCache,
    slots: Arc<dyn SlotRepository>,
    refresh: Duration,
) {
    {
        let (catalog, cache, slots) = (catalog.clone(), cache.clone(), slots.clone());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(refresh);
            tick.tick().await;
            loop {
                tick.tick().await;
                refresh_catalog(&catalog, &cache, slots.as_ref()).await;
            }
        });
    }

    tokio::spawn(async move {
        let mut backoff = Duration::from_secs(1);
        loop {
            match listen(&pool, &catalog, &cache, slots.as_ref()).await {
                Ok(()) => backoff = Duration::from_secs(1),
                Err(err) => tracing::error!(error = %err, "LISTEN connection lost; cache cleared"),
            }
            cache.clear();
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    });
}

async fn listen(
    pool: &PgPool,
    catalog: &Catalog,
    cache: &LatestCache,
    slots: &dyn SlotRepository,
) -> Result<(), sqlx::Error> {
    let mut listener = PgListener::connect_with(pool).await?;
    listener.listen_all(["model_runs", "slots"]).await?;
    // Anything could have changed while we weren't listening.
    cache.clear();
    refresh_catalog(catalog, cache, slots).await;
    tracing::info!("listening for model_runs and slots notifications");

    loop {
        let note = listener.recv().await?;
        match note.channel() {
            "model_runs" => cache.invalidate_slot(note.payload()),
            "slots" => refresh_catalog(catalog, cache, slots).await,
            _ => {}
        }
    }
}

async fn refresh_catalog(catalog: &Catalog, cache: &LatestCache, slots: &dyn SlotRepository) {
    match catalog.reload(slots).await {
        Ok(changed) => changed.iter().for_each(|id| cache.invalidate_slot(id)),
        // A-28: keep serving the previous catalog.
        Err(err) => tracing::error!(error = %err, "slot catalog reload failed; keeping previous"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::run;

    fn key(slot: &str, implementation: Option<&str>) -> CacheKey {
        CacheKey {
            slot_id: slot.into(),
            scope_key: "underlying=BTC".into(),
            role: Role::Production,
            implementation: implementation.map(Into::into),
        }
    }

    #[test]
    fn invalidates_one_slot_only() {
        let cache = LatestCache::new(Duration::from_secs(60));
        cache.insert(
            key("vol_surface", None),
            Arc::new(run("vol_surface", "vol_surface_svi", 0)),
        );
        cache.insert(
            key("vol_surface", Some("vol_surface_sabr")),
            Arc::new(run("vol_surface", "vol_surface_sabr", 0)),
        );
        cache.insert(key("smile", None), Arc::new(run("smile", "smile_basic", 0)));
        cache.invalidate_slot("vol_surface");
        cache.inner.run_pending_tasks();
        assert!(cache.get(&key("vol_surface", None)).is_none());
        assert!(
            cache
                .get(&key("vol_surface", Some("vol_surface_sabr")))
                .is_none()
        );
        assert!(cache.get(&key("smile", None)).is_some());
    }
}
