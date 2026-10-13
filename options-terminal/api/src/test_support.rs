//! Builders for unit tests. Structural values only, no market data: payloads are `{}`.

use std::collections::BTreeMap;

use chrono::{DateTime, Duration, TimeZone, Utc};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::domain::{ImplementationRef, Role, Slot, SlotStatus, StoredRun, run::utc};

pub fn t0() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 10, 13, 17, 0, 0).unwrap()
}

pub fn slot(id: &str) -> Slot {
    Slot {
        slot_id: id.into(),
        title: format!("{id} title"),
        description: String::new(),
        payload_kind: "surface".into(),
        payload_version: "1".into(),
        dimensions: vec!["underlying".into()],
        allowed: BTreeMap::from([("underlying".into(), vec!["BTC".into(), "ETH".into()])]),
        max_age_seconds: 900,
        production: Some(ImplementationRef {
            implementation: format!("{id}_a"),
            version: "1.0.0".into(),
        }),
        shadows: vec![],
        status: SlotStatus::Active,
        definition: json!({ "display": { "order": 10 } }),
    }
}

/// A run `offset` seconds after t0, with a time-ordered id so later runs sort later.
pub fn run(slot_id: &str, implementation: &str, offset: i64) -> StoredRun {
    run_with_role(slot_id, implementation, offset, Role::Production)
}

pub fn run_with_role(slot_id: &str, implementation: &str, offset: i64, role: Role) -> StoredRun {
    let as_of = t0() + Duration::seconds(offset);
    let millis = u64::try_from(as_of.timestamp_millis()).unwrap();
    let run_id = Uuid::new_v7(uuid::Timestamp::from_unix_time(
        millis / 1000,
        ((millis % 1000) * 1_000_000) as u32,
        0,
        0,
    ));
    StoredRun {
        run_id,
        slot_id: slot_id.into(),
        implementation_id: implementation.into(),
        role,
        as_of,
        envelope: envelope(run_id, slot_id, implementation, as_of),
    }
}

pub fn envelope(run_id: Uuid, slot_id: &str, implementation: &str, as_of: DateTime<Utc>) -> Value {
    json!({
        "envelope_version": "1",
        "run_id": run_id.to_string(),
        "slot_id": slot_id,
        "implementation_id": implementation,
        "implementation_version": "1.0.0",
        "scope": { "underlying": "BTC" },
        "status": "success",
        "as_of": utc(as_of),
        "started_at": utc(as_of),
        "finished_at": utc(as_of),
        "inputs": { "sources": [], "hash": format!("sha256:{}", "0".repeat(64)) },
        "code": { "git_sha": "a1b2c3d" },
        "warnings": [],
        "payload_kind": "surface",
        "payload_version": "1",
        "payload": {}
    })
}
