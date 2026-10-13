"""The only module that touches Postgres (M-3). Implemented in M1:

- connect as `runner_rw` from environment config (X-6);
- take an advisory lock on (slot_id, scope_key, role) so runs don't overlap (M-19);
- insert one model_runs row in a single transaction;
- NOTIFY model_runs '<slot_id>' on success (C-13);
- sync the slots table from registry.yaml on deploy (M-14), including shadow_implementations,
  marking removed slots retired, then NOTIFY slots (CONTRACTS C-13a).
"""
