//! In-memory slot catalog (A-27, A-28). Slots and their implementations come and go through
//! registry syncs in `models/`; the catalog follows the `slots` table without a restart.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::{domain::Slot, repo::SlotRepository};

pub struct Snapshot {
    pub by_id: HashMap<String, Arc<Slot>>,
}

#[derive(Clone, Default)]
pub struct Catalog {
    // std lock: held only to clone or swap an Arc, never across .await (A-14).
    inner: Arc<RwLock<Option<Arc<Snapshot>>>>,
}

impl Catalog {
    /// None until the first successful load; `/ready` is 503 until then.
    pub fn snapshot(&self) -> Option<Arc<Snapshot>> {
        self.inner.read().map(|g| g.clone()).unwrap_or(None)
    }

    pub fn get(&self, slot_id: &str) -> Option<Arc<Slot>> {
        self.snapshot().and_then(|s| s.by_id.get(slot_id).cloned())
    }

    /// Swap in a new set of slots. Returns the IDs of slots that were added, removed or
    /// changed in any way (status, production, shadows, definition), so their cache entries
    /// can be dropped (A-30).
    pub fn replace(&self, slots: Vec<Slot>) -> Vec<String> {
        let next: HashMap<String, Arc<Slot>> = slots
            .into_iter()
            .map(|s| (s.slot_id.clone(), Arc::new(s)))
            .collect();
        let prev = self.snapshot();
        let mut changed: Vec<String> = match &prev {
            None => next.keys().cloned().collect(),
            Some(prev) => {
                let mut ids: Vec<String> = next
                    .iter()
                    .filter(|(id, slot)| prev.by_id.get(*id).is_none_or(|p| p != *slot))
                    .map(|(id, _)| id.clone())
                    .collect();
                ids.extend(
                    prev.by_id
                        .keys()
                        .filter(|id| !next.contains_key(*id))
                        .cloned(),
                );
                ids
            }
        };
        changed.sort();
        if let Ok(mut guard) = self.inner.write() {
            *guard = Some(Arc::new(Snapshot { by_id: next }));
        }
        changed
    }

    pub async fn reload(&self, repo: &dyn SlotRepository) -> Result<Vec<String>, sqlx::Error> {
        let slots = repo.load_all().await?;
        let count = slots.len();
        let changed = self.replace(slots);
        if !changed.is_empty() {
            tracing::info!(slots = count, changed = ?changed, "slot catalog reloaded");
        }
        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::slot;

    #[test]
    fn reports_added_changed_and_removed_slots() {
        let catalog = Catalog::default();
        assert!(catalog.snapshot().is_none());
        assert_eq!(
            catalog.replace(vec![slot("vol_surface"), slot("smile")]),
            ["smile", "vol_surface"]
        );

        let mut promoted = slot("vol_surface");
        promoted.production.as_mut().unwrap().implementation = "vol_surface_sabr".into();
        assert_eq!(
            catalog.replace(vec![promoted, slot("term_structure")]),
            ["smile", "term_structure", "vol_surface"]
        );
        assert!(catalog.get("smile").is_none());
        assert_eq!(
            catalog
                .get("vol_surface")
                .unwrap()
                .production
                .as_ref()
                .unwrap()
                .implementation,
            "vol_surface_sabr"
        );
    }
}
