"""
Fetch TAO-denominated pool history for all subnets listed in all_subnet_metadata.csv.
Output format matches get_tao_stats_subnet_price.py; combined file is separate.
Writes ``tao_subnets_wide.parquet`` for fast loads in the RP-PCA Streamlit app.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from rp_pca.config import (  # noqa: E402
    TAO_SUBNET_DATA_DIR,
    TAO_SUBNET_EXPORT_END,
    TAO_SUBNET_EXPORT_START,
    TAO_SUBNET_WIDE_PARQUET,
)
from rp_pca.data.tao_subnet_loader import load_subnet_candles_combined  # noqa: E402


def resolve_taostats_auth() -> str | None:
    """Get TAOSTATS_AUTH from env, or fallback to .env next to this script."""
    auth = os.environ.get("TAOSTATS_AUTH")
    if auth:
        return auth

    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return None

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip() != "TAOSTATS_AUTH":
            continue
        val = val.strip().strip("\"'")
        if val:
            return val
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Taostats dTAO pool history for all subnets in metadata CSV"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {TAO_SUBNET_DATA_DIR})",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to all_subnet_metadata.csv (default: beside this script)",
    )
    p.add_argument(
        "--start-date",
        default=TAO_SUBNET_EXPORT_START,
        help="Inclusive start date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        default=TAO_SUBNET_EXPORT_END,
        help="Inclusive end date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--no-parquet",
        action="store_true",
        help=f"Skip writing {TAO_SUBNET_WIDE_PARQUET}",
    )
    return p.parse_args()


def resolve_metadata_path(arg: Path | None) -> Path:
    if arg is not None:
        p = Path(arg).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {p}")
        return p
    p = _REPO_ROOT / "all_subnet_metadata.csv"
    if p.is_file():
        return p
    raise FileNotFoundError(
        f"Could not find all_subnet_metadata.csv in {_REPO_ROOT} (use --metadata)"
    )


def main() -> None:
    args = parse_args()
    auth = resolve_taostats_auth()
    if not auth:
        raise RuntimeError(
            "Set TAOSTATS_AUTH in environment or .env to your Taostats API credentials "
            "(Authorization header value per Taostats docs)."
        )

    metadata_path = resolve_metadata_path(args.metadata)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else TAO_SUBNET_DATA_DIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = args.start_date
    end_date = args.end_date
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_unix = int(start_dt.timestamp())
    end_unix = int(end_dt.timestamp())

    base_url = "https://api.taostats.io/api/dtao/pool/history/v1"
    headers = {
        "accept": "application/json",
        "Authorization": auth,
    }

    request_interval = 14.0
    last_request_time = 0.0

    metadata_df = pd.read_csv(metadata_path)
    netuids = metadata_df["netuid"].astype(int).tolist()
    total_subnets = len(netuids)

    dfs: list[pd.DataFrame] = []
    saved_files: list[Path] = []

    print(f"\n{'='*70}")
    print(f"Fetching data for {total_subnets} subnets (from {metadata_path})")
    print(f"Output directory: {out_dir}")
    print(f"Date range: {start_date} → {end_date}")
    print(f"Rate limit: ~{request_interval:.0f}s between requests")
    print(f"{'='*70}\n")

    for current_subnet, netuid in enumerate(netuids, start=1):
        symbol = f"SN{netuid}"
        cmc_id = 0

        progress = f"[{current_subnet}/{total_subnets}]"
        print(f"{progress} Fetching {symbol:8} (netuid={netuid:3}) ", end="", flush=True)

        data_list: list[dict] = []
        page = 1
        page_count = 0
        skip_subnet = False

        while True:
            elapsed = time.time() - last_request_time
            if elapsed < request_interval:
                remaining = request_interval - elapsed
                print(f"⏳ {remaining:.1f}s ", end="", flush=True)
                time.sleep(remaining)

            params = {
                "netuid": netuid,
                "frequency": "by_day",
                "timestamp_start": start_unix,
                "timestamp_end": end_unix,
                "page": page,
            }

            try:
                response = requests.get(base_url, params=params, headers=headers, timeout=120)
                response.raise_for_status()
                last_request_time = time.time()
                page_count += 1

                json_data = response.json()
                data_list.extend(json_data["data"])

                next_page = json_data["pagination"]["next_page"]
                if next_page is None:
                    break
                page = next_page
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    print("⚠️ Rate limit! Waiting 60s... ", end="", flush=True)
                    time.sleep(60)
                    continue
                code = e.response.status_code if e.response is not None else "?"
                print(f"❌ HTTP {code}, skipping")
                skip_subnet = True
                break
            except Exception as e:
                print(f"❌ Error: {e}, skipping")
                skip_subnet = True
                break

        if skip_subnet:
            continue

        if data_list:
            print(f"✓ {len(data_list)} records ({page_count} pages)")
        else:
            print("✗ No data")
            continue

        df = pd.DataFrame(data_list)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        df["price"] = df["price"].astype(float)
        df["open"] = df["price"]
        df["high"] = df["price"]
        df["low"] = df["price"]
        df["close"] = df["price"]
        df["volume"] = 0.0
        df["market_cap"] = df["market_cap"].astype(float)
        df["cmc_id"] = cmc_id
        df["symbol"] = symbol
        df = df[
            [
                "cmc_id",
                "symbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "market_cap",
            ]
        ]

        dfs.append(df)
        csv_file = out_dir / f"{symbol.lower()}_tao_daily_candles.csv"
        df.to_csv(csv_file, index=False)
        saved_files.append(csv_file)

    if dfs:
        all_data = pd.concat(dfs, ignore_index=True)
        combined_csv = out_dir / "tao_all_subnets_daily_candles_combined.csv"
        all_data.to_csv(combined_csv, index=False)
        saved_files.append(combined_csv)

        if not args.no_parquet:
            wide = load_subnet_candles_combined(combined_csv)
            pq_path = out_dir / TAO_SUBNET_WIDE_PARQUET
            wide.to_parquet(pq_path, index=True)
            saved_files.append(pq_path)

    print("\n" + "=" * 60)
    print("FILES SAVED:")
    for f in saved_files:
        print(f"  • {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
