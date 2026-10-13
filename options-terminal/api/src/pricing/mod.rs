//! Evaluation math (README §3.1, API §3.2, guide §3–§6). A 1:1 port of
//! `models/terminal_models/core/quant/{black76,smile,evaluate}.py`; Python is the reference and
//! `tests/pricing_parity.rs` holds this code to `contracts/fixtures/pricing/` (A-33, P-3).
//! f64 throughout: valuation, not settlement (P-1).

pub mod black76;
pub mod evaluate;
pub mod surface;
