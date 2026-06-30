"""
Local JSON persistence for pricing-engine results.

Replaces the old `save_to_mongodb`. Stamps each result with `created_at` and
`model_version` (same fields the Mongo path stamped), then writes a timestamped
JSON file plus a `latest.json` to a local `output/` directory.

Also maintains `market_history.json`, a cross-snapshot index keyed by
Polymarket `market_id`, updated on every save.
"""

import glob
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
MARKET_HISTORY_FILENAME = "market_history.json"


def _json_default(obj):
    """JSON serializer for objects the default encoder can't handle.

    Results carry numpy scalars (e.g. liquidity_score in model_params); convert
    those to native Python via .item(), and fall back to str() for anything else.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_market_history(out_dir: str) -> Dict:
    path = os.path.join(out_dir, MARKET_HISTORY_FILENAME)
    if not os.path.isfile(path):
        return {"metadata": {}, "markets": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _result_to_observation(result: Dict, snapshot_ts: str) -> Dict:
    analysis = result.get("analysis") or {}
    pricing = result.get("pricing") or {}
    return {
        "ts": snapshot_ts,
        "volume": result.get("volume"),
        "liquidity": result.get("liquidity"),
        "active": result.get("active"),
        "closed": result.get("closed"),
        "current_prices": result.get("current_prices"),
        "pricing": pricing,
        "analysis": analysis,
        "outcome_pricings": result.get("outcome_pricings"),
        "regime": result.get("regime") or pricing.get("regime"),
    }


def _upsert_static_fields(market_record: Dict, result: Dict) -> None:
    for field in ("slug", "question", "market_type", "condition_id", "end_date"):
        value = result.get(field)
        if value:
            market_record[field] = value
    if not market_record.get("market_type"):
        market_record["market_type"] = result.get("market_type", "binary")


def _merge_results_into_history(
    history: Dict,
    results: List[Dict],
    snapshot_ts: str,
) -> None:
    markets = history.setdefault("markets", {})

    for result in results:
        market_id = result.get("market_id")
        if not market_id:
            continue
        mid = str(market_id)

        if mid not in markets:
            markets[mid] = {
                "slug": "",
                "question": "",
                "market_type": "binary",
                "condition_id": "",
                "end_date": "",
                "first_seen": snapshot_ts,
                "last_seen": snapshot_ts,
                "observations": [],
            }

        record = markets[mid]
        _upsert_static_fields(record, result)

        first_seen = record.get("first_seen") or snapshot_ts
        last_seen = record.get("last_seen") or snapshot_ts
        if snapshot_ts < first_seen:
            record["first_seen"] = snapshot_ts
        if snapshot_ts > last_seen:
            record["last_seen"] = snapshot_ts

        observation = _result_to_observation(result, snapshot_ts)
        observations = record.setdefault("observations", [])

        replaced = False
        for index, existing in enumerate(observations):
            if existing.get("ts") == snapshot_ts:
                observations[index] = observation
                replaced = True
                break
        if not replaced:
            observations.append(observation)

        observations.sort(key=lambda item: item.get("ts", ""))


def _save_market_history(out_dir: str, history: Dict) -> str:
    path = os.path.join(out_dir, MARKET_HISTORY_FILENAME)
    markets = history.get("markets") or {}
    observation_count = sum(len(m.get("observations") or []) for m in markets.values())
    history["metadata"] = {
        "updated_at": datetime.now().isoformat(),
        "market_count": len(markets),
        "observation_count": observation_count,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def update_market_history(
    results: List[Dict],
    snapshot_ts: str,
    out_dir: str = None,
) -> str:
    """Merge one snapshot's results into market_history.json keyed by market_id."""
    out_dir = out_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    history = _load_market_history(out_dir)
    _merge_results_into_history(history, results, snapshot_ts)
    return _save_market_history(out_dir, history)


def rebuild_market_history(out_dir: str = None) -> str:
    """Rebuild market_history.json from all pricing_results_*.json snapshot files."""
    out_dir = out_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    snapshot_files = []
    for path in glob.glob(os.path.join(out_dir, "pricing_results_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        created_at = (data.get("metadata") or {}).get("created_at")
        results = data.get("results") or []
        if created_at and results:
            snapshot_files.append((created_at, results))

    snapshot_files.sort(key=lambda item: item[0])

    history: Dict = {"metadata": {}, "markets": {}}
    for created_at, results in snapshot_files:
        _merge_results_into_history(history, results, created_at)

    return _save_market_history(out_dir, history)


def save_results_to_json(
    results: List[Dict],
    out_dir: str = None,
    model_version: str = "v1",
) -> str:
    """Save pricing results to a local JSON file.

    Args:
        results: List of per-market result dicts.
        out_dir: Directory to write to (defaults to ./output next to this file).
        model_version: Stamped onto each result.

    Returns:
        Path to the timestamped JSON file written.
    """
    out_dir = out_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    created_at = datetime.now().isoformat()
    for result in results:
        result["created_at"] = created_at
        result["model_version"] = model_version

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"pricing_results_{timestamp}.json"
    filepath = os.path.join(out_dir, filename)

    payload = {
        "metadata": {
            "count": len(results),
            "model_version": model_version,
            "created_at": created_at,
        },
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    # Also overwrite a stable latest.json for easy downstream access.
    latest_path = os.path.join(out_dir, "latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    update_market_history(results, created_at, out_dir)

    return filepath
