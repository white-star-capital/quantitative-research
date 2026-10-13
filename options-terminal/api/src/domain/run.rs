//! A stored run. The envelope is passed through as stored (A-7, A-21); only the columns the
//! API needs for routing, caching and staleness are typed.

use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Production,
    Shadow,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::Shadow => "shadow",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "production" => Some(Self::Production),
            "shadow" => Some(Self::Shadow),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredRun {
    pub run_id: Uuid,
    pub slot_id: String,
    pub implementation_id: String,
    pub role: Role,
    pub as_of: DateTime<Utc>,
    pub envelope: Value,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RunMeta {
    pub api_version: &'static str,
    pub role: Role,
    pub age_seconds: i64,
    pub max_age_seconds: i64,
    pub stale: bool,
    pub served_at: String,
}

impl RunMeta {
    /// A-8: age = served_at − as_of; stale = age > max_age.
    pub fn new(role: Role, as_of: DateTime<Utc>, max_age_seconds: i64, now: DateTime<Utc>) -> Self {
        let age_seconds = (now - as_of).num_seconds();
        Self {
            api_version: "1",
            role,
            age_seconds,
            max_age_seconds,
            stale: age_seconds > max_age_seconds,
            served_at: utc(now),
        }
    }

    /// A-12: `max-age = min(30, seconds until stale)`, floored at 0.
    pub fn cache_max_age(&self) -> i64 {
        (self.max_age_seconds - self.age_seconds).clamp(0, 30)
    }
}

pub fn utc(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn staleness_flips_after_max_age() {
        let as_of = Utc.with_ymd_and_hms(2026, 10, 13, 17, 0, 0).unwrap();
        let at = |s| as_of + chrono::Duration::seconds(s);
        let fresh = RunMeta::new(Role::Production, as_of, 900, at(900));
        let stale = RunMeta::new(Role::Production, as_of, 900, at(901));
        assert!(!fresh.stale);
        assert!(stale.stale);
        assert_eq!(fresh.served_at, "2026-10-13T17:15:00Z");
    }

    #[test]
    fn cache_max_age_is_bounded() {
        let as_of = Utc.with_ymd_and_hms(2026, 10, 13, 17, 0, 0).unwrap();
        let meta = |age| {
            RunMeta::new(
                Role::Production,
                as_of,
                900,
                as_of + chrono::Duration::seconds(age),
            )
        };
        assert_eq!(meta(0).cache_max_age(), 30);
        assert_eq!(meta(890).cache_max_age(), 10);
        assert_eq!(meta(5000).cache_max_age(), 0);
    }
}
