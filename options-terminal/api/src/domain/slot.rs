//! Slots are data, not code (A-27): everything here comes from the `slots` table.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum SlotStatus {
    Active,
    Retired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ImplementationRef {
    pub implementation: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
    pub slot_id: String,
    pub title: String,
    pub description: String,
    pub payload_kind: String,
    pub payload_version: String,
    pub dimensions: Vec<String>,
    /// Optional per-dimension allowlist, from `definition.scope.allowed`.
    pub allowed: BTreeMap<String, Vec<String>>,
    pub max_age_seconds: i64,
    pub production: Option<ImplementationRef>,
    pub shadows: Vec<ImplementationRef>,
    pub status: SlotStatus,
    /// Full slot YAML as JSON (CONTRACTS §6).
    pub definition: Value,
}

impl Slot {
    pub fn is_shadow(&self, implementation: &str) -> bool {
        self.shadows
            .iter()
            .any(|s| s.implementation == implementation)
    }

    pub fn shadow_ids(&self) -> Vec<&str> {
        self.shadows
            .iter()
            .map(|s| s.implementation.as_str())
            .collect()
    }

    pub fn summary(&self) -> SlotSummary {
        SlotSummary {
            slot_id: self.slot_id.clone(),
            title: self.title.clone(),
            description: self.description.clone(),
            payload_kind: self.payload_kind.clone(),
            payload_version: self.payload_version.clone(),
            scope: ScopeSpec {
                dimensions: self.dimensions.clone(),
                allowed: self.allowed.clone(),
            },
            max_age_seconds: self.max_age_seconds,
            production_impl: self.production.as_ref().map(|p| p.implementation.clone()),
            production_version: self.production.as_ref().map(|p| p.version.clone()),
            display: self
                .definition
                .get("display")
                .cloned()
                .unwrap_or(Value::Null),
        }
    }

    pub fn detail(&self) -> SlotDetail {
        SlotDetail {
            summary: self.summary(),
            status: self.status,
            conventions: self
                .definition
                .get("conventions")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            implementations: Implementations {
                production: self.production.clone(),
                shadow: self.shadows.clone(),
            },
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ScopeSpec {
    pub dimensions: Vec<String>,
    pub allowed: BTreeMap<String, Vec<String>>,
}

/// `GET /v1/slots` item.
#[derive(Debug, Serialize, ToSchema)]
pub struct SlotSummary {
    pub slot_id: String,
    pub title: String,
    pub description: String,
    pub payload_kind: String,
    pub payload_version: String,
    pub scope: ScopeSpec,
    pub max_age_seconds: i64,
    pub production_impl: Option<String>,
    pub production_version: Option<String>,
    #[schema(value_type = Object)]
    pub display: Value,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct Implementations {
    pub production: Option<ImplementationRef>,
    pub shadow: Vec<ImplementationRef>,
}

/// `GET /v1/slots/{slot_id}`.
#[derive(Debug, Serialize, ToSchema)]
pub struct SlotDetail {
    #[serde(flatten)]
    pub summary: SlotSummary,
    pub status: SlotStatus,
    pub conventions: String,
    pub implementations: Implementations,
}
