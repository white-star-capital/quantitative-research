-- Slots can have any number of shadow implementations, and they come and go
-- (MODELS.md §5). The runner's registry sync (M-14) writes the current list here so the
-- API can address each shadow by implementation and list them, without reading
-- registry.yaml. Shape: [{"implementation": "vol_surface_sabr", "version": "0.3.0"}, ...]

ALTER TABLE slots
    ADD COLUMN shadow_implementations JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(shadow_implementations) = 'array');

-- Shadow lookups filter on implementation.
CREATE INDEX model_runs_latest_by_impl
    ON model_runs (slot_id, scope_key, role, implementation_id, status, as_of DESC);
