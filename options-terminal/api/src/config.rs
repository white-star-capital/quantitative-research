//! Env-driven configuration, validated at startup (A-22, §9.5).

use std::{net::SocketAddr, path::PathBuf, time::Duration};

use axum::http::HeaderValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// v1: no authentication (README §8 Q1). The seam exists so keys are config, not code (A-17).
    None,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub bind: SocketAddr,
    /// Must connect as `api_ro` (A-2, CONTRACTS C-12).
    pub database_url: String,
    pub db_max_connections: u32,
    pub db_acquire_timeout: Duration,
    pub cors_origins: Vec<HeaderValue>,
    pub auth_mode: AuthMode,
    pub contracts_dir: PathBuf,
    pub problem_base_url: String,
    pub rate_per_minute: u32,
    pub rate_burst: u32,
    pub request_timeout: Duration,
    pub catalog_refresh: Duration,
    pub cache_ttl: Duration,
}

#[derive(Debug, thiserror::Error)]
#[error("invalid configuration: {0}")]
pub struct ConfigError(String);

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// `lookup` abstracts the environment so tests don't mutate process state.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let get = |k: &str, default: &str| lookup(k).unwrap_or_else(|| default.to_owned());
        let parse_u32 = |k: &str, default: &str| -> Result<u32, ConfigError> {
            get(k, default)
                .parse::<u32>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(|| ConfigError(format!("{k} must be a positive integer")))
        };

        let bind = get("API_BIND", "0.0.0.0:8080")
            .parse()
            .map_err(|_| ConfigError("API_BIND must be host:port".into()))?;
        let database_url =
            lookup("DATABASE_URL").ok_or_else(|| ConfigError("DATABASE_URL is required".into()))?;

        let cors_origins = get("API_CORS_ORIGINS", "")
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|o| {
                if o == "*" {
                    return Err(ConfigError(
                        "API_CORS_ORIGINS must not be '*' (A-15)".into(),
                    ));
                }
                HeaderValue::from_str(o)
                    .map_err(|_| ConfigError(format!("API_CORS_ORIGINS: bad origin '{o}'")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let auth_mode = match get("AUTH_MODE", "none").as_str() {
            "none" => AuthMode::None,
            other => {
                return Err(ConfigError(format!(
                    "AUTH_MODE '{other}' is not supported in v1"
                )));
            }
        };

        Ok(Self {
            bind,
            database_url,
            db_max_connections: parse_u32("DB_MAX_CONNECTIONS", "10")?,
            db_acquire_timeout: Duration::from_secs(2),
            cors_origins,
            auth_mode,
            contracts_dir: PathBuf::from(get("CONTRACTS_DIR", "../contracts")),
            problem_base_url: get("API_PROBLEM_BASE_URL", "https://options-terminal/problems")
                .trim_end_matches('/')
                .to_owned(),
            rate_per_minute: parse_u32("API_RATE_PER_MINUTE", "120")?,
            rate_burst: parse_u32("API_RATE_BURST", "30")?,
            request_timeout: Duration::from_secs(5),
            catalog_refresh: Duration::from_secs(60),
            cache_ttl: Duration::from_secs(10),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cfg(pairs: &[(&str, &str)]) -> Result<Config, ConfigError> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).into(), (*v).into()))
            .collect();
        Config::from_lookup(|k| map.get(k).cloned())
    }

    #[test]
    fn requires_database_url() {
        assert!(cfg(&[]).is_err());
        assert!(cfg(&[("DATABASE_URL", "postgres://x")]).is_ok());
    }

    #[test]
    fn rejects_wildcard_cors_and_unknown_auth() {
        assert!(cfg(&[("DATABASE_URL", "x"), ("API_CORS_ORIGINS", "*")]).is_err());
        assert!(cfg(&[("DATABASE_URL", "x"), ("AUTH_MODE", "apikey")]).is_err());
    }

    #[test]
    fn parses_origins_and_limits() {
        let c = cfg(&[
            ("DATABASE_URL", "x"),
            (
                "API_CORS_ORIGINS",
                "http://localhost:5173, https://terminal.example",
            ),
            ("API_RATE_PER_MINUTE", "60"),
        ])
        .unwrap();
        assert_eq!(c.cors_origins.len(), 2);
        assert_eq!(c.rate_per_minute, 60);
        assert!(cfg(&[("DATABASE_URL", "x"), ("API_RATE_BURST", "0")]).is_err());
    }
}
