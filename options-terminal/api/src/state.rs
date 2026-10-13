//! AppState + FromRef so handlers extract only what they need (§8).

use std::{collections::HashMap, path::Path, sync::Arc};

use axum::extract::FromRef;
use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::{
    cache::LatestCache,
    catalog::Catalog,
    config::{AuthMode, Config},
    repo::{RunRepository, SlotRepository},
};

pub type Clock = Arc<dyn Fn() -> DateTime<Utc> + Send + Sync>;

/// Payload JSON Schemas from `contracts/schemas/payloads/<kind>/v<version>.json`, keyed by
/// (kind, version). Loaded at startup; schemas only change with a contracts release.
#[derive(Default)]
pub struct Schemas(pub HashMap<(String, String), Arc<Value>>);

impl Schemas {
    pub fn load(contracts_dir: &Path) -> std::io::Result<Self> {
        let mut map = HashMap::new();
        let root = contracts_dir.join("schemas").join("payloads");
        if !root.is_dir() {
            return Err(std::io::Error::other(format!(
                "{} is not a directory",
                root.display()
            )));
        }
        for kind in std::fs::read_dir(&root)? {
            let kind = kind?;
            if !kind.file_type()?.is_dir() {
                continue;
            }
            let kind_name = kind.file_name().to_string_lossy().into_owned();
            for file in std::fs::read_dir(kind.path())? {
                let path = file?.path();
                let Some(version) = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .and_then(|n| n.strip_prefix('v'))
                    .and_then(|n| n.strip_suffix(".json"))
                else {
                    continue;
                };
                let schema: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)
                    .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
                map.insert((kind_name.clone(), version.to_owned()), Arc::new(schema));
            }
        }
        Ok(Self(map))
    }
}

#[derive(Clone)]
pub struct AppState {
    pub runs: Arc<dyn RunRepository>,
    pub slots: Arc<dyn SlotRepository>,
    pub catalog: Catalog,
    pub cache: LatestCache,
    pub schemas: Arc<Schemas>,
    pub config: Arc<Config>,
    pub clock: Clock,
}

macro_rules! from_ref {
    ($($field:ident: $ty:ty),* $(,)?) => {$(
        impl FromRef<AppState> for $ty {
            fn from_ref(state: &AppState) -> Self {
                state.$field.clone()
            }
        }
    )*};
}

from_ref!(
    runs: Arc<dyn RunRepository>,
    slots: Arc<dyn SlotRepository>,
    catalog: Catalog,
    cache: LatestCache,
    schemas: Arc<Schemas>,
    config: Arc<Config>,
    clock: Clock,
);

impl FromRef<AppState> for AuthMode {
    fn from_ref(state: &AppState) -> Self {
        state.config.auth_mode
    }
}
