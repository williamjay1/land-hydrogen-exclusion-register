"""Build a compact, fully validated PVGIS cache from immutable raw responses.

Raw PVGIS JSON remains on E:.  This step writes only a D:-drive cache in MW
per MWp, a candidate-index ledger, and a status manifest.  It rejects any
year that lacks one exact, validated UTC annual series for every candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from project03_acquire_pvgis_public_panel import (
    parse_payload,
    sha256,
    validate_payload_parameters,
)
from project03_public_data_core import expected_hours, utc_index_to_epoch_nanoseconds


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cache_paths(config: dict[str, Any], year: int) -> dict[str, Path]:
    directory = Path(config["cache_root"]) / "pvgis_hourly"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"pvgis_hourly_{year}"
    return {
        "matrix": directory / f"{prefix}_mw_per_mwp.npz",
        "candidate_index": directory / f"{prefix}_candidate_index.csv",
        "manifest": directory / f"{prefix}_manifest.json",
    }


def build_year(config: dict[str, Any], year: int) -> tuple[dict[str, Any], int]:
    if year not in config["target_years"]:
        raise ValueError(f"year {year} is not configured")
    pvgis = config["pvgis"]
    candidates = pd.read_csv(Path(config["candidate_file"]), dtype={"candidate_id": str})
    candidates = candidates.sort_values(["country", "candidate_id"]).reset_index(drop=True)
    raw_dir = Path(config["raw_root"]) / pvgis["raw_subdirectory"] / str(year)
    paths = cache_paths(config, year)
    expected = expected_hours(year)
    matrix = np.empty((len(candidates), expected), dtype=np.float32)
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    canonical_index: pd.DatetimeIndex | None = None

    for row_index, row in enumerate(candidates.itertuples(index=False)):
        raw_path = raw_dir / f"{row.candidate_id}.json"
        try:
            if not raw_path.exists():
                raise FileNotFoundError(f"missing raw PVGIS response: {raw_path}")
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            validate_payload_parameters(payload, row, year, pvgis)
            raw_index, aligned_index, values_w_per_kwp, offset_minutes = parse_payload(payload, year)
            if canonical_index is None:
                canonical_index = aligned_index
            elif not canonical_index.equals(aligned_index):
                raise ValueError("candidate PVGIS response has a distinct aligned UTC index")
            matrix[row_index, :] = (values_w_per_kwp / 1000.0).astype(np.float32)
            audit_rows.append(
                {
                    "candidate_id": str(row.candidate_id),
                    "country": str(row.country),
                    "row_index": row_index,
                    "hours": int(len(values_w_per_kwp)),
                    "pvgis_timestamp_label_offset_minutes": int(offset_minutes),
                    "raw_first_timestamp_utc": raw_index[0].isoformat(),
                    "raw_last_timestamp_utc": raw_index[-1].isoformat(),
                    "market_interval_first_utc": aligned_index[0].isoformat(),
                    "market_interval_last_utc": aligned_index[-1].isoformat(),
                    "market_interval_rule": "floor UTC PVGIS timestamp to the enclosing ENTSO-E day-ahead hour",
                    "pv_annual_kwh_per_kwp": float(values_w_per_kwp.sum() / 1000.0),
                    "raw_file": str(raw_path),
                    "raw_sha256": sha256(raw_path),
                }
            )
        except Exception as exc:  # noqa: BLE001 - output an auditable no-cache status, never a partial cache
            failures.append(
                {
                    "candidate_id": str(row.candidate_id),
                    "country": str(row.country),
                    "error": repr(exc),
                }
            )

    result: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "year": year,
        "candidate_count": int(len(candidates)),
        "validated_candidates": int(len(audit_rows)),
        "failure_count": int(len(failures)),
        "raw_directory": str(raw_dir),
        "power_unit": "MW per MWp",
        "source_power_unit": pvgis["power_unit"],
        "pvgis_to_market_interval_rule": "Each UTC PVGIS timestamp is assigned to its enclosing UTC ENTSO-E hourly interval; raw timestamps and location-specific offsets are retained in the candidate audit.",
        "boundary": "Cache represents PV resource only. It contains no electricity prices, storage, grid-access estimate, or hydrogen output result.",
        "failures": failures,
    }
    if failures or canonical_index is None:
        result["status"] = "FAIL_INCOMPLETE_OR_INVALID_RAW_PANEL"
        paths["manifest"].write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result, 2

    np.savez_compressed(
        paths["matrix"],
        candidate_ids=candidates["candidate_id"].to_numpy(dtype="U"),
        timestamps_utc_ns=utc_index_to_epoch_nanoseconds(canonical_index),
        pv_mw_per_mwp=matrix,
    )
    audit = pd.DataFrame(audit_rows).sort_values("row_index")
    audit.to_csv(paths["candidate_index"], index=False)
    result.update(
        {
            "status": "PASS_COMPLETE_VALIDATED_PANEL",
            "matrix_path": str(paths["matrix"]),
            "matrix_sha256": sha256(paths["matrix"]),
            "candidate_index_path": str(paths["candidate_index"]),
            "candidate_index_sha256": sha256(paths["candidate_index"]),
            "timestamp_start_utc": canonical_index[0].isoformat(),
            "timestamp_end_utc": canonical_index[-1].isoformat(),
            "hours": int(len(canonical_index)),
            "timestamp_storage_format": "signed 64-bit Unix-epoch nanoseconds, explicitly encoded independent of pandas internal resolution",
            "offset_minutes_counts": {
                str(key): int(value)
                for key, value in audit["pvgis_timestamp_label_offset_minutes"].value_counts().sort_index().items()
            },
        }
    )
    paths["manifest"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--years", type=int, nargs="+")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    years = args.years or list(config["target_years"])
    reports: list[dict[str, Any]] = []
    exit_code = 0
    for year in years:
        report, code = build_year(config, year)
        reports.append(report)
        exit_code = max(exit_code, code)
    print(json.dumps({"reports": reports}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
