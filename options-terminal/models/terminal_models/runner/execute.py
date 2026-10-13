"""Per-run pipeline (MODELS.md §6), implemented in M1:

1. Resolve slot -> implementation(s) and scopes from the registry.
2. Fetch inputs via adapters; reject stale inputs (M-8) and cross-source skew (M-9b).
3. Execute `Model.run` in a subprocess with the manifest timeout.
4. Validate: payload schema, cross-field shapes (C-4), no NaN/Inf (C-3), bounds,
   sanity checks, size (C-6).
5. Build the envelope; 6. hand it to writer.py; 7. log a structured summary.

A failed run still produces an envelope (status=failed, payload=null) — M-17.
"""
