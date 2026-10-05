r"""Acquire versioned public inputs for the SETA regret-certificate pilot.

New source files are written once to F:\AcademicData and made read-only.
Derived candidate and price panels are written only to D:\MLWork.  The PVGIS
API is queried serially; the code does not assume the legacy PVGIS v5 request
limit applies to v6.  Any source/API failure is recorded and stops the run.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import stat
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from project03_build_smard_hourly_panels import read_smard_segment


PROJECT = Path(r"D:\MLWork\Europe_green_hydrogen_siting")
CROSSWALK = PROJECT / "datasets" / "intermediate" / "public_data_redesign" / "candidate_bidding_zone_full_population_20260927_v1.csv"
DEFLATOR = PROJECT / "cache" / "public_data_redesign" / "economic_deflators" / "official_eur_2023_conversion_table_20260927_v1.json"
RAW_ROOT = Path(r"F:\AcademicData\Europe_green_hydrogen_siting\raw\SETA_regret_certificate_pilot_20261002\pilot_v1")
DATA_ROOT = PROJECT / "datasets" / "intermediate" / "regret_certificate_pilot_20261002_v1"
LOG_ROOT = PROJECT / "logs" / "regret_certificate_pilot_20261002_v1"

SMARD_URL = "https://www.smard.de/nip-download-manager/nip/download/market-data"
SMARD_TERMS_URL = "https://www.smard.de/en/datennutzung"
PVGIS_URL = "https://photovoltaic-geographical-information-system.ec.europa.eu/api/v6/power/broadband"
PVGIS_DOC = Path(r"F:\AcademicData\Europe_green_hydrogen_siting\raw\SETA_regret_certificate_pilot_20261002\docs\pvgis_v6_openapi_20261002.json")
PVGIS_DOC_SHA256 = "f044105c193554bcee264c175aadf7ba61da31feb49b75d147644c8f6ecc9008"

ZONE_SERIES: dict[str, dict[str, Any]] = {
    "DE_LU": {
        "column": "Germany/Luxembourg [€/MWh] Calculated resolutions",
        "module_id": 8_004_169,
        "eic": "10Y1001A1001A82H",
    },
    "FR": {
        "column": "France [€/MWh] Calculated resolutions",
        "module_id": 8_000_254,
        "eic": "10YFR-RTE------C",
    },
    "BE": {
        "column": "Belgium [€/MWh] Calculated resolutions",
        "module_id": 8_004_996,
        "eic": "10YBE----------2",
    },
    "NL": {
        "column": "Netherlands [€/MWh] Calculated resolutions",
        "module_id": 8_000_256,
        "eic": "10YNL----------L",
    },
}
# SMARD emits this requested subset in its service-defined column order, which
# is not guaranteed to match the order of moduleIds in the POST body.
SMARD_CSV_ZONE_ORDER = ("DE_LU", "BE", "FR", "NL")
PILOT_PER_ZONE = 8
YEARS = tuple(range(2014, 2025))
PVGIS_PARAMS_BASE: dict[str, Any] = {
    "frequency": "Hourly",
    "timezone": "UTC",
    "irradiance_source": "SARAH-3",
    "surface_orientation": 180,
    "surface_tilt": 35,
    "photovoltaic_module": "cSi:Free standing 2025",
    "photovoltaic_module_type": "Mono-Facial",
    "system_efficiency": 0.86,
    "power_model": "Huld 2011",
    "peak-power": 1,
    "temperature_model": "Faiman",
    "verbose": 1,
}
ZONE_BERLIN = ZoneInfo("Europe/Berlin")
HOUR_MS = 3_600_000


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path: Path, data: bytes) -> None:
    """Write an immutable source/audit artifact without replacing prior data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) == sha256_bytes(data):
            return
        raise FileExistsError(f"refusing to replace nonidentical immutable file: {path}")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(path, stat.S_IREAD)
    except OSError:
        pass


def write_derived_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) == sha256_bytes(data):
            return
        raise FileExistsError(f"refusing to replace existing derived output: {path}")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def haversine_km(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance for latitude/longitude in radians."""
    delta = coords_a[:, None, :] - coords_b[None, :, :]
    hav = (
        np.sin(delta[..., 0] / 2) ** 2
        + np.cos(coords_a[:, None, 0])
        * np.cos(coords_b[None, :, 0])
        * np.sin(delta[..., 1] / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(hav)))


def select_sites(crosswalk_path: Path = CROSSWALK) -> pd.DataFrame:
    """Deterministic, outcome-blind maximin sample: eight sites in each zone."""
    source = pd.read_csv(crosswalk_path)
    required = {
        "candidate_id",
        "country",
        "nuts3_name",
        "lat",
        "lon",
        "bidding_zone_key",
        "bidding_zone_eic",
        "candidate_to_bus_km",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"candidate crosswalk misses columns: {sorted(missing)}")
    selected: list[pd.Series] = []
    for zone, info in ZONE_SERIES.items():
        group = source.loc[source["bidding_zone_key"].eq(zone)].copy()
        if group["bidding_zone_eic"].nunique() != 1 or group["bidding_zone_eic"].iloc[0] != info["eic"]:
            raise ValueError(f"crosswalk bidding-zone EIC mismatch in {zone}")
        group["lat"] = pd.to_numeric(group["lat"], errors="raise")
        group["lon"] = pd.to_numeric(group["lon"], errors="raise")
        if len(group) < PILOT_PER_ZONE or group[["lat", "lon"]].isna().any().any():
            raise ValueError(f"insufficient coordinates for {zone}")
        group = group.sort_values("candidate_id", kind="stable").reset_index(drop=True)
        coords = np.radians(group[["lat", "lon"]].to_numpy(dtype=float))
        distances = haversine_km(coords, coords)
        centroid = np.radians(group[["lat", "lon"]].mean().to_numpy(dtype=float)[None, :])
        first = int(np.argmin(haversine_km(coords, centroid).ravel()))
        indexes = [first]
        while len(indexes) < PILOT_PER_ZONE:
            nearest = distances[:, indexes].min(axis=1)
            nearest[indexes] = -1.0
            indexes.append(int(np.argmax(nearest)))
        for ordinal, idx in enumerate(indexes, start=1):
            row = group.iloc[idx].copy()
            row["pilot_zone_order"] = ordinal
            row["sample_design"] = "centroid-started_great-circle_farthest-point; no PV/price outcome used"
            selected.append(row)
    result = pd.DataFrame(selected).reset_index(drop=True)
    if result["candidate_id"].nunique() != len(result) or len(result) != 4 * PILOT_PER_ZONE:
        raise ValueError("pilot selection contains duplicate IDs or an unexpected sample size")
    return result


def persist_site_sample(sites: pd.DataFrame) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    path = DATA_ROOT / "pilot_sites_32.csv"
    payload = sites.to_csv(index=False, lineterminator="\n").encode("utf-8")
    write_derived_once(path, payload)
    return path


def smard_bounds_2024() -> tuple[int, int, int]:
    # Match the PVGIS/price model's UTC calendar year exactly. SMARD labels its
    # exported intervals in Europe/Berlin civil time, but the request bounds
    # are stored as UTC epoch milliseconds.
    start_utc = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end_utc = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    start_ms = round(start_utc.timestamp() * 1000)
    end_ms = round(end_utc.timestamp() * 1000)
    hours = (end_ms - start_ms) // HOUR_MS
    if hours != 8784:
        raise AssertionError(f"2024 local-market year should have 8784 intervals, got {hours}")
    return start_ms, end_ms, hours


def acquire_smard_2024(session: requests.Session, sites: pd.DataFrame) -> dict[str, Any]:
    raw_dir = RAW_ROOT / "smard_2024"
    start_ms, end_ms, expected_hours = smard_bounds_2024()
    module_ids = [ZONE_SERIES[z]["module_id"] for z in ZONE_SERIES]
    request = {
        "request_form": [
            {
                "type": "discrete",
                "region": "DE",
                "moduleIds": module_ids,
                "language": "en",
                "timestamp_to": end_ms,
                "format": "CSV",
                "timestamp_from": start_ms,
                "resolution": "hour",
            }
        ]
    }
    request_bytes = (json.dumps(request, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    request_path = raw_dir / "requests" / "smard_utc_calendar_2024_four_zones.json"
    write_immutable(request_path, request_bytes)

    terms_path = raw_dir / "terms" / "smard_data_use_20261002.html"
    if terms_path.exists():
        terms_bytes = terms_path.read_bytes()
        terms_status = 200  # Existing snapshot was first written after an HTTP-200 check.
    else:
        terms_response = session.get(SMARD_TERMS_URL, timeout=(20, 120))
        terms_bytes = terms_response.content
        write_immutable(terms_path, terms_bytes)
        terms_status = terms_response.status_code
    if terms_status != 200:
        raise RuntimeError(f"SMARD terms snapshot request returned HTTP {terms_status}")
    terms_text = terms_bytes.decode("utf-8", errors="replace").casefold()
    if "cc by 4.0" not in terms_text and "cc-by-4.0" not in terms_text:
        raise ValueError("current SMARD terms snapshot does not visibly state CC BY 4.0")

    response = session.post(
        SMARD_URL,
        data=request_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "Public-data research reproducibility pilot/1.0"},
        timeout=(20, 240),
    )
    if response.status_code != 200:
        raise RuntimeError(f"SMARD 2024 data request returned HTTP {response.status_code}")
    response_bytes = response.content
    response_path = raw_dir / "responses" / "smard_utc_calendar_2024_four_zones.csv"
    write_immutable(response_path, response_bytes)
    columns = [ZONE_SERIES[z]["column"] for z in SMARD_CSV_ZONE_ORDER]
    parsed = read_smard_segment(
        response_bytes,
        from_epoch_ms=start_ms,
        to_epoch_ms_exclusive=end_ms,
        expected_columns=columns,
    )
    if len(parsed) != expected_hours:
        raise ValueError(f"SMARD 2024 response failed coverage checks: HTTP {response.status_code}; rows={len(parsed)}")
    if parsed[0][0].isoformat() != "2024-01-01T00:00:00+00:00":
        raise ValueError("SMARD source rows do not start at the expected UTC boundary")
    if parsed[-1][0].isoformat() != "2024-12-31T23:00:00+00:00":
        raise ValueError("SMARD source rows do not end at the expected UTC boundary")

    deflator_data = json.loads(DEFLATOR.read_text(encoding="utf-8"))
    hicp = deflator_data["hicp_eu27_cp00_annual_average_index"]
    factor = float(hicp["2023"] / hicp["2024"])
    derived_rows: list[dict[str, Any]] = []
    for instant, prices in parsed:
        for zone, info in ZONE_SERIES.items():
            nominal = float(prices[info["column"]])
            derived_rows.append(
                {
                    "timestamp_utc": instant.isoformat(),
                    "bidding_zone_key": zone,
                    "bidding_zone_eic": info["eic"],
                    "price_eur_nominal_per_mwh": f"{nominal:.10f}",
                    "price_deflator_to_eur_2023": f"{factor:.12f}",
                    "price_eur_2023_per_mwh": f"{nominal * factor:.10f}",
                }
            )
    panel = pd.DataFrame(derived_rows)
    panel_path = DATA_ROOT / "smard_utc_calendar_2024_four_zone_hourly_eur2023_v1.csv"
    write_derived_once(panel_path, panel.to_csv(index=False, lineterminator="\n").encode("utf-8"))

    record = {
        "status": "PASS_COMPLETE_2024_FOUR_ZONE_PANEL",
        "year_definition": "UTC calendar year; exact overlap with PVGIS 2024 hourly series",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SMARD_URL,
        "terms_url": SMARD_TERMS_URL,
        "required_attribution": "Bundesnetzagentur | SMARD.de",
        "license_snapshot": str(terms_path),
        "license_snapshot_sha256": sha256_file(terms_path),
        "request_file": str(request_path),
        "request_sha256": sha256_file(request_path),
        "response_file": str(response_path),
        "response_sha256": sha256_file(response_path),
        "response_bytes": response_path.stat().st_size,
        "http_status": response.status_code,
        "module_ids": module_ids,
        "bidding_zones": list(ZONE_SERIES),
        "source_intervals": expected_hours,
        "source_first_utc": parsed[0][0].isoformat(),
        "source_last_utc": parsed[-1][0].isoformat(),
        "superseded_local_calendar_attempt": "The earlier smard_2024_four_zones.csv request covered a local-market year and was excluded because its UTC endpoints did not match calendar 2024.",
        "price_cells": len(panel),
        "missing_price_cells": int(panel["price_eur_nominal_per_mwh"].isna().sum()),
        "deflator_source": str(DEFLATOR),
        "deflator_sha256": sha256_file(DEFLATOR),
        "deflator_factor_2024_to_eur2023": factor,
        "derived_panel": str(panel_path),
        "derived_panel_sha256": sha256_file(panel_path),
        "pilot_sites_sha256": sha256_file(DATA_ROOT / "pilot_sites_32.csv"),
        "independence_note": "Four bidding zones are market-level units; the 32 sites are not 32 independent electricity markets.",
    }
    record_path = LOG_ROOT / "smard_utc_calendar_2024_acquisition_manifest_v1.json"
    write_derived_once(record_path, (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return record


def pvgis_request_params(site: pd.Series, year: int) -> dict[str, Any]:
    params = dict(PVGIS_PARAMS_BASE)
    params.update(
        {
            "latitude": float(site["lat"]),
            "longitude": float(site["lon"]),
            "start_time": f"{year}-01-01T00:00:00",
            "end_time": f"{year}-12-31T23:59:59",
        }
    )
    return params


def pvgis_expected_hours(year: int) -> int:
    return 8784 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 8760


def validate_pvgis_payload(payload: dict[str, Any], *, year: int, candidate_id: str) -> list[dict[str, Any]]:
    timestamps = payload.get("timestamps")
    power_w = payload.get("power")
    if not isinstance(timestamps, list) or not isinstance(power_w, list) or len(timestamps) != len(power_w):
        raise ValueError(f"PVGIS {candidate_id}/{year}: timestamp and power arrays are absent or unequal")
    expected = pvgis_expected_hours(year)
    if len(timestamps) != expected:
        raise ValueError(f"PVGIS {candidate_id}/{year}: received {len(timestamps)} rows, expected {expected}")
    parsed = pd.to_datetime(pd.Series(timestamps), errors="raise")
    # PVGIS SARAH timestamps carry a documented site-dependent sub-hour offset.
    # With the request explicitly fixed to UTC, map to the nearest UTC market hour
    # only when the reported offset remains within 15 minutes; never interpolate.
    rounded = parsed.dt.round("h")
    offset_min = (parsed - rounded).abs().dt.total_seconds() / 60
    if float(offset_min.max()) > 15.0:
        raise ValueError(f"PVGIS {candidate_id}/{year}: SARAH timestamps exceed the 15-minute UTC-hour mapping tolerance")
    if rounded.duplicated().any() or not rounded.is_monotonic_increasing:
        raise ValueError(f"PVGIS {candidate_id}/{year}: rounded UTC timestamps duplicate or are not monotonic")
    expected_start = pd.Timestamp(year=year, month=1, day=1)
    expected_end = pd.Timestamp(year=year, month=12, day=31, hour=23)
    if rounded.iloc[0] != expected_start or rounded.iloc[-1] != expected_end:
        raise ValueError(f"PVGIS {candidate_id}/{year}: UTC-hour coverage boundaries do not match calendar year")
    power = pd.to_numeric(pd.Series(power_w), errors="raise").to_numpy(dtype=float)
    if not np.isfinite(power).all() or (power < -1e-6).any() or (power > 1300).any():
        raise ValueError(f"PVGIS {candidate_id}/{year}: nonfinite or physically implausible 1-kWp power values")
    return [
        {
            "candidate_id": candidate_id,
            "year": year,
            "timestamp_utc": rounded.iloc[i].isoformat(),
            "source_timestamp": str(timestamps[i]),
            "pv_mw_per_mwp": power[i] / 1000.0,
            "pv_power_source_unit": "W per 1 kWp; divided by 1000 to obtain MW per MWp",
        }
        for i in range(expected)
    ]


def acquire_pvgis_year(session: requests.Session, site: pd.Series, year: int, *, delay_seconds: float) -> dict[str, Any]:
    if sha256_file(PVGIS_DOC).lower() != PVGIS_DOC_SHA256:
        raise ValueError("archived PVGIS v6 OpenAPI snapshot hash mismatch")
    candidate_id = str(site["candidate_id"])
    params = pvgis_request_params(site, year)
    raw_dir = RAW_ROOT / "pvgis_v6" / "responses"
    request_path = RAW_ROOT / "pvgis_v6" / "requests" / f"pvgis_{candidate_id}_{year}.json"
    response_path = raw_dir / f"pvgis_{candidate_id}_{year}.json"
    request_record = {
        "source_url": PVGIS_URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "coordinates": {"latitude": params["latitude"], "longitude": params["longitude"]},
        "data_year": year,
        "parameters": params,
        "api_schema_snapshot": str(PVGIS_DOC),
        "api_schema_sha256": PVGIS_DOC_SHA256,
        "note": "PVGIS v6 SARAH-3 hourly point output; `quiet` deliberately omitted because it can return an empty JSON object.",
    }
    request_bytes = (json.dumps(request_record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_immutable(request_path, request_bytes)
    response = session.get(PVGIS_URL, params=params, timeout=(20, 240))
    response_bytes = response.content
    write_immutable(response_path, response_bytes)
    if response.status_code != 200:
        raise RuntimeError(f"PVGIS returned HTTP {response.status_code} for {candidate_id}/{year}")
    payload = response.json()
    rows = validate_pvgis_payload(payload, year=year, candidate_id=candidate_id)
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return {
        "candidate_id": candidate_id,
        "year": year,
        "request_file": str(request_path),
        "request_sha256": sha256_file(request_path),
        "response_file": str(response_path),
        "response_sha256": sha256_file(response_path),
        "response_bytes": len(response_bytes),
        "rows": len(rows),
        "first_utc": rows[0]["timestamp_utc"],
        "last_utc": rows[-1]["timestamp_utc"],
        "max_sarah_hour_offset_minutes": float(
            (pd.to_datetime(pd.Series(payload["timestamps"]))
             - pd.to_datetime(pd.Series(payload["timestamps"])).dt.round("h"))
            .abs().dt.total_seconds().max() / 60
        ),
    }


def build_pvgis_derived_panel(sites: pd.DataFrame, *, years: tuple[int, ...]) -> Path:
    out_path = DATA_ROOT / "pvgis_v6_sarah3_hourly_2014_2024.csv.gz"
    if out_path.exists():
        raise FileExistsError(f"derived PVGIS panel already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".partial")
    if tmp_path.exists():
        raise FileExistsError(f"partial PVGIS panel already exists; preserve and inspect it: {tmp_path}")
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8", newline="") as gz:
            writer = csv.DictWriter(
                gz,
                fieldnames=["candidate_id", "year", "timestamp_utc", "source_timestamp", "pv_mw_per_mwp", "pv_power_source_unit"],
                lineterminator="\n",
            )
            writer.writeheader()
            for _, site in sites.iterrows():
                candidate_id = str(site["candidate_id"])
                for year in years:
                    path = RAW_ROOT / "pvgis_v6" / "responses" / f"pvgis_{candidate_id}_{year}.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    rows = validate_pvgis_payload(payload, year=year, candidate_id=candidate_id)
                    writer.writerows(rows)
        with tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    except Exception:
        # Keep an inspectable partial cache. Never remove acquired raw responses.
        raise
    return out_path


def acquire_pvgis(sites: pd.DataFrame, *, first_year: int, last_year: int, limit_sites: int, delay_seconds: float) -> dict[str, Any]:
    if not PVGIS_DOC.is_file() or sha256_file(PVGIS_DOC).lower() != PVGIS_DOC_SHA256:
        raise ValueError("PVGIS v6 API specification is missing or does not match the archived snapshot")
    if first_year < 2014 or last_year > 2024 or first_year > last_year:
        raise ValueError("PVGIS v6 acquisition years must lie in 2014–2024")
    chosen = sites.head(limit_sites) if limit_sites > 0 else sites
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Public-data research reproducibility pilot/1.0"})
    records: list[dict[str, Any]] = []
    for _, site in chosen.iterrows():
        for year in range(first_year, last_year + 1):
            record = acquire_pvgis_year(session, site, year, delay_seconds=delay_seconds)
            records.append(record)
            print(json.dumps({"complete": len(records), "of": len(chosen) * (last_year-first_year+1), **record}, ensure_ascii=True), flush=True)
    return {
        "status": "PASS_PVGIS_ACQUISITION",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sites": len(chosen),
        "years": [first_year, last_year],
        "series_count": len(records),
        "hourly_records": sum(r["rows"] for r in records),
        "raw_records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--acquire-smard-2024", action="store_true")
    parser.add_argument("--acquire-pvgis", action="store_true")
    parser.add_argument("--pvgis-first-year", type=int, default=2014)
    parser.add_argument("--pvgis-last-year", type=int, default=2024)
    parser.add_argument("--pvgis-limit-sites", type=int, default=0, help="0 means all 32 pilot sites")
    parser.add_argument("--pvgis-delay-seconds", type=float, default=2.0)
    parser.add_argument("--crosswalk", type=Path, default=CROSSWALK)
    args = parser.parse_args()
    if not (args.select_only or args.acquire_smard_2024 or args.acquire_pvgis):
        parser.error("choose --select-only, --acquire-smard-2024, or --acquire-pvgis")
    sites = select_sites(args.crosswalk)
    site_path = persist_site_sample(sites)
    print(json.dumps({"status": "PASS_OUTCOME_BLIND_SITE_SAMPLE", "path": str(site_path), "sha256": sha256_file(site_path), "counts": sites.groupby("bidding_zone_key").size().to_dict(), "sites": sites["candidate_id"].tolist()}, ensure_ascii=True, indent=2))
    outcomes: dict[str, Any] = {}
    if args.acquire_smard_2024:
        with requests.Session() as session:
            outcomes["smard_2024"] = acquire_smard_2024(session, sites)
    if args.acquire_pvgis:
        outcomes["pvgis"] = acquire_pvgis(sites, first_year=args.pvgis_first_year, last_year=args.pvgis_last_year, limit_sites=args.pvgis_limit_sites, delay_seconds=args.pvgis_delay_seconds)
    if args.acquire_pvgis and args.pvgis_first_year == 2014 and args.pvgis_last_year == 2024 and args.pvgis_limit_sites in (0, len(sites)):
        outcomes["derived_pvgis_panel"] = str(build_pvgis_derived_panel(sites, years=YEARS))
    if outcomes:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = LOG_ROOT / f"acquisition_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "crosswalk": str(args.crosswalk),
            "crosswalk_sha256": sha256_file(args.crosswalk),
            "site_sample": str(site_path),
            "site_sample_sha256": sha256_file(site_path),
            "outcomes": outcomes,
        }
        write_derived_once(out_path, (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print(json.dumps({"summary_file": str(out_path), "outcome_keys": list(outcomes)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
