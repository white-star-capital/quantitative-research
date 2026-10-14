# Models — `models/`

Python package that holds every model implementation, the framework they plug into, and the runner that executes them and writes validated outputs. Read `CONTRACTS.md` first; this document assumes the envelope, payload kinds and slots defined there.

## 1. Goals

1. Adding, swapping or retiring a model touches only `models/` (and `contracts/` for a new slot).
2. A model is a pure function of its inputs: no network calls, no DB writes, no global state inside the model.
3. Nothing reaches users unless it passed validation.
4. Any output can be reproduced from its recorded inputs and code SHA.

## 2. Folder layout

```
models/
├── pyproject.toml
├── uv.lock
├── registry.yaml                 # which implementation serves each slot
├── terminal_models/
│   ├── core/
│   │   ├── base.py               # Model ABC, RunContext
│   │   ├── manifest.py           # manifest schema + loader
│   │   ├── payloads.py           # Pydantic types per payload kind (generated or contract-tested)
│   │   ├── envelope.py
│   │   ├── registry.py           # registry loader + validation
│   │   ├── validation.py         # schema + bounds + slot sanity checks
│   │   ├── scope.py              # canonical scope_key (shared test vectors)
│   │   └── quant/                # shared math: Black-76, interpolation, day counts, SVI helpers…
│   ├── adapters/
│   │   ├── base.py               # DataAdapter interface
│   │   └── <source>/             # one package per data source
│   ├── runner/
│   │   ├── cli.py                # `python -m terminal_models.runner …`
│   │   ├── execute.py
│   │   └── writer.py             # only module that writes to Postgres
│   └── library/
│       └── <implementation_id>/
│           ├── manifest.yaml
│           ├── model.py
│           ├── README.md         # methodology, assumptions, known limits
│           └── tests/
│               ├── fixtures/     # recorded inputs + expected outputs
│               └── test_model.py
└── tests/                        # framework tests
```

- **M-1** Python 3.12+, dependencies managed with `uv`, lockfile committed.
- **M-2** Implementations in `library/` MAY import from `core/` and from the Python standard and scientific stack. They MUST NOT import from other implementations, from `adapters/` or from `runner/`. Shared logic moves to `core/quant/`. Enforce with `import-linter` in CI.
- **M-3** Only `runner/writer.py` touches the database.

## 3. Interfaces

### 3.1 Model

```python
class Model(ABC):
    manifest: Manifest  # loaded from the implementation's manifest.yaml

    @abstractmethod
    def required_inputs(self, scope: Scope) -> list[InputSpec]:
        """Declare which data the run needs. The runner fetches it via adapters."""

    @abstractmethod
    def run(self, ctx: RunContext) -> Payload:
        """Pure computation. ctx carries as_of, scope, fetched inputs, params, logger."""
```

- **M-4** `run` MUST be deterministic: same `ctx` → same payload. Randomness (e.g. a fitter's initial guess) uses a seed from `ctx.params`.
- **M-5** `run` MUST NOT do I/O. All data arrives through `ctx.inputs`; this is what makes model tests fast and reproducible.
  - **M-5a** A model whose second fetch depends on the first (e.g. order books for screened candidates) MAY implement `followup_inputs(scope, inputs) -> list[InputSpec]`. It must be pure, like `run`. The runner fetches the declared inputs, then the follow-ups, and snapshots all of them, so replay is unchanged. Follow-ups marked `optional` that fail are left out; the model handles their absence. (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md A-1.)
- **M-6** `run` raises `ModelError(code, message)` for expected failures (e.g. insufficient strikes to fit). Any other exception is caught by the runner and recorded as `status = failed`, `error.code = "unhandled"`.

### 3.2 Data adapters

```python
class DataAdapter(ABC):
    name: str
    def fetch(self, spec: InputSpec, as_of: datetime) -> InputData: ...
```

- **M-7** Adapters own timeouts, retries with backoff, and rate-limit handling for their source. Every external call has a timeout.
- **M-8** Every fetched input records `snapshot_at` (the source timestamp, not fetch time). The runner rejects an input whose `snapshot_at` is older than the implementation's `max_input_age_seconds` (manifest) relative to `as_of`, with `error.code = "stale_input"`.
- **M-9** Adapters MAY cache raw snapshots to object storage keyed by input hash, so runs can be replayed. SHOULD be implemented by M4.

- **M-9a** If an input cannot be fetched or is stale, the run fails. Adapters and implementations MUST NOT substitute hardcoded, default or placeholder prices (e.g. `1.00`, `0.00`). No prices, no output.

Sources (README §8 Q2, decided). Both are public and keyless:

| Adapter | Source | Provides |
|---|---|---|
| `adapters/hyperliquid/` | Hyperliquid public `POST /info` | Underlying spot/perp prices (mids, candles) |
| `adapters/deribit/` | Deribit public `/api/v2/public/*` | Option chains: instruments, mark price, mark IV, greeks, underlying/index price |

- **M-9b** An implementation that combines both sources MUST check that their `snapshot_at` values are within the manifest's `max_input_skew_seconds` and fail with `error.code = "input_skew"` otherwise.

Build M1 against a file-based adapter reading recorded snapshots first; the two live adapters plug in behind the same interface. The file adapter stays for tests and replay.

### 3.3 Manifest

`library/<implementation_id>/manifest.yaml`:

```yaml
implementation_id: vol_surface_svi
version: 2.1.0                 # semver; bump on any output-affecting change
fills_slot: vol_surface
owner: alejandro
description: Raw SVI per expiry, interpolated in total variance across tenors.
schedule: "*/15 * * * *"       # cron, UTC
timeout_seconds: 120
max_input_age_seconds: 120
max_input_skew_seconds: 30        # only for implementations that combine sources (M-9b)
params:
  min_strikes_per_expiry: 7
  seed: 42
bounds:                        # hard; keyed by value key (column/item/series key, or z); abs_<key> bounds |value|
  z: { min: 0.01, max: 5.0 }
sanity_checks:                 # soft; violation = warning on a successful run
  - calendar_arbitrage
  - butterfly_arbitrage
```

- **M-10** `fills_slot` MUST reference an existing slot; the implementation's payload MUST match that slot's `payload_kind` and `payload_version`.
- **M-11** `version` MUST bump on any change that can alter output (code, params, dependency upgrade in `core/quant`). CI fails if files under an implementation changed without a version bump.

## 4. Registry

`models/registry.yaml` is the production switchboard:

```yaml
slots:
  vol_surface:
    production: { implementation: vol_surface_svi, version: 2.1.0 }
    shadow:
      - { implementation: vol_surface_sabr, version: 0.3.0 }
    scopes:
      - { underlying: BTC }
      - { underlying: ETH }
  skew_metrics:
    production: { implementation: skew_metrics_basic, version: 1.0.0 }
    scopes:
      - { underlying: BTC }
      - { underlying: ETH }
```

- **M-12** Exactly one production implementation per slot, except during a slot's initial shadow period, when `production: null` is allowed with at least one shadow. Zero or more shadows.
- **M-13** The registry loader validates on startup and in CI: referenced implementations exist, versions match manifests, scopes satisfy the slot's dimensions and allowlist.
- **M-14** Registry changes go through PR review. The runner syncs the `slots` table (production impl/version, `shadow_implementations`, status) from the registry on each deploy, then `NOTIFY slots` (CONTRACTS C-13a). A slot removed from the registry is set to `status = 'retired'`, never deleted, so its run history stays readable.

## 5. Lifecycle: adding, swapping, retiring

| Stage | How | Visible to users? |
|---|---|---|
| **dev** | Implementation exists in `library/`, not in registry | No |
| **shadow** | Listed under a slot's `shadow` | Only via `?role=shadow` on the API (`API.md` §3); runs and stores outputs on the same schedule as production |
| **production** | Set as the slot's `production` | Yes, served as latest |
| **retired** | Removed from registry; code MAY stay for reproducibility | History only |

**Promotion (swap) procedure:**

1. Add the new implementation as shadow. Run it for at least the slot's `min_shadow_period` (default 5 trading days; set per slot).
2. Generate a **shadow diff report**: `runner diff <slot> --against shadow:<impl>` compares production vs shadow outputs over the period (per-point differences, warning counts, failure rates, runtime).
3. PR changes `production` in `registry.yaml`, links the diff report, and gets owner approval.
4. Deploy. The next scheduled run writes the new implementation as production.

**Rollback:** revert the registry PR. The previous implementation's next run becomes latest. Until then, the last good production output is still served (with a stale flag if it ages out).

- **M-15** Swapping implementations within a slot MUST NOT require any API or frontend change. M4's swap drill proves this.
- **M-16** The diff report MUST be produced by the tooling, not by hand, so promotions are comparable.

## 6. Runner

```
python -m terminal_models.runner run <slot> [--scope underlying=BTC] [--as-of 2026-10-13T17:00:00Z] [--role production|shadow|all] [--dry-run]
python -m terminal_models.runner run-due            # everything whose schedule is due (scheduler entrypoint)
python -m terminal_models.runner diff <slot> --against shadow:<impl> --from … --to …
python -m terminal_models.runner validate-registry
python -m terminal_models.runner replay <run_id>    # re-run with recorded inputs; assert identical output
```

Per run, the runner MUST, in order:

1. Resolve slot → implementation(s) from the registry; resolve scopes.
2. Fetch inputs via adapters; check input staleness (M-8).
3. Execute `run` in a subprocess with the manifest `timeout_seconds`. Timeout → failed run, `error.code = "timeout"`.
4. **Validate:**
   - Payload against the JSON Schema for the slot's kind and version.
   - Cross-field shape rules the schema can't express (CONTRACTS C-4).
   - No `NaN`/`Inf` anywhere (CONTRACTS C-3).
   - Manifest `bounds` (hard: failure).
   - Manifest `sanity_checks` (soft: warning).
   - Payload size (CONTRACTS C-6).
5. Build the envelope (run_id, scope, timestamps, input hash, git SHA, warnings).
6. Write one row in a single transaction; `NOTIFY` on success.
7. Log a structured summary: `run_id`, slot, implementation, version, scope, status, duration, warnings count.

- **M-17** A failed run MUST still write a row (`status = failed`, payload null). Silence is not an acceptable failure mode.
- **M-18** `--dry-run` executes and validates but prints the envelope instead of writing it.
- **M-19** Runs of the same slot and scope MUST NOT overlap. Use a Postgres advisory lock keyed on `(slot_id, scope_key, role)`.

**Options-specific sanity checks** to provide in `core/validation.py` (implementations opt in via manifest):

| Check | Applies to | Rule |
|---|---|---|
| `calendar_arbitrage` | surface | Total variance non-decreasing in tenor at fixed moneyness |
| `butterfly_arbitrage` | surface, curve (smile) | Implied density non-negative across strikes |
| `put_call_parity` | table (chain) | Parity residual within tolerance |
| `monotonic_tenor` | curve (term structure) | x strictly increasing |
| `greeks_signs` | table (greeks) | Gamma ≥ 0, vega ≥ 0 for long options |

## 7. Testing

- **M-20** Each implementation ships tests using recorded inputs in `tests/fixtures/`: a golden-output test (within numeric tolerance), at least one edge case (sparse strikes, one expiry, crossed quotes) and one expected-failure case.
- **M-21** Every implementation's output in tests MUST validate against its slot contract (CONTRACTS §7).
- **M-22** Framework tests cover: registry validation errors, timeout handling, failed-run writing, advisory lock, `scope_key` test vectors, replay determinism.
- **M-23** Coverage SHOULD be ≥ 90% for `core/` and `runner/`; implementation coverage is judged by edge-case completeness rather than percentage.
- **M-24** Determinism check: CI runs each implementation's golden test 10 times and requires identical output.

## 8. Acceptance (M1)

- [ ] `uv sync && pytest` passes from a clean checkout.
- [ ] One reference implementation fills one slot end-to-end: fetch (file adapter) → run → validate → write to local Postgres.
- [ ] A deliberately broken implementation writes a failed run and leaves latest unchanged.
- [ ] `replay <run_id>` reproduces a stored output exactly.
- [ ] `validate-registry` catches each misconfiguration in M-13.
- [ ] `import-linter` enforces M-2.
