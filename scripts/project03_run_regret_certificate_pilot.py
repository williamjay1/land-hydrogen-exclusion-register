"""Run the frozen two-year dispatch and regret-certificate pilot.

The PV inputs in this first executable model check are the locally archived,
version-consistent PVGIS-SARAH3 2020--2023 hourly panels. Dispatch uses only
prior-year PV climatology and a fixed four-week-lag day-ahead price forecast.
Observed SMARD prices are used only for ex-post settlement scoring. 2024 is
therefore an external price-year holdout with a climatological PV proxy, not a
2024 weather hindcast or measured operational result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_array


PROJECT = Path(r"D:\MLWork\Europe_green_hydrogen_siting")
DATA_ROOT = PROJECT / "datasets" / "intermediate" / "regret_certificate_pilot_20261002_v1"
RESULT_ROOT = PROJECT / "results" / "regret_certificate_pilot_20261002_v1"
LOG_ROOT = PROJECT / "logs" / "regret_certificate_pilot_20261002_v1"
SITE_FILE = DATA_ROOT / "pilot_sites_32.csv"
FULL_ZONE_MAPPING = PROJECT / "datasets" / "intermediate" / "public_data_redesign" / "candidate_bidding_zone_full_population_20260927_v1.csv"
PV_CACHE = PROJECT / "cache" / "public_data_redesign" / "pvgis_hourly"
PV_MANIFESTS = {year: PV_CACHE / f"pvgis_hourly_{year}_manifest.json" for year in (2020, 2021, 2022, 2023)}
PV_MATRICES = {year: PV_CACHE / f"pvgis_hourly_{year}_mw_per_mwp.npz" for year in (2020, 2021, 2022, 2023)}
PRICE_2023 = PROJECT / "datasets" / "intermediate" / "public_data_redesign" / "smard_fixed_nonpolish_cohort_2016_2023_v3" / "smard_ccby_hourly_2023_v2.csv"
PRICE_2022 = PROJECT / "datasets" / "intermediate" / "public_data_redesign" / "smard_fixed_nonpolish_cohort_2016_2023_v3" / "smard_ccby_hourly_2022_v2.csv"
PRICE_2024 = DATA_ROOT / "smard_utc_calendar_2024_four_zone_hourly_eur2023_v1.csv"

ZONE_EICS = {
    "DE_LU": "10Y1001A1001A82H",
    "FR": "10YFR-RTE------C",
    "BE": "10YBE----------2",
    "NL": "10YNL----------L",
}
MODEL = {
    "electrolyser_capacity_mwe": 1.0,
    "electrolyser_specific_electricity_kwh_per_kg": 55.0,
    "pv_capacity_mwp_per_mwe": 2.0,
    "annual_average_utilisation": 0.50,
    "minimum_stable_load_fraction": 0.20,
    "ramp_limit_mwe_per_hour": 1.0,
    "hydrogen_storage_hours_of_demand": 12.0,
    "dispatch_horizon_hours": 24,
    "day_end_hydrogen_inventory_kg": 0.0,
    "storage_boundary_note": "Hydrogen inventory is forced to zero at each UTC day end; storage shifts production only within each daily horizon, not across days.",
    "unit_commitment_boundary_note": "On/off, starts, and stops are represented; startup costs and minimum up/down times are not modeled.",
    "daily_decomposition_condition": "At the base ramp limit (equal to full electrolyser capacity), zero startup cost, no minimum up/down time, and zero end-of-day tank inventory, daily MILPs are separable across UTC days; changing these assumptions invalidates this decomposition argument and requires re-deriving the certificate.",
    "startup_cost_eur_per_start": 0.0,
    "startup_cost_note": "No project-specific startup cost was identified; transitions are represented and counted but not monetised in this pilot.",
    "pv_forecast_method": "mean historical PVGIS-SARAH3 output at same UTC month-day-hour; only years before dispatch year; Feb-29 fallback uses adjacent dates if absent",
    "price_forecast_method": "Mean of the four most recent observed prices at the same UTC weekday-hour (lags 7, 14, 21, and 28 days); all inputs precede the delivery day.",
    "settlement_evaluation": "Observed SMARD day-ahead prices are used only to score ex-post settlement cost; they are never supplied to the dispatch optimizer.",
    "objective_scope": "Electricity procurement cost only; no electrolyser/PV CAPEX, fixed O&M, land, water, compression, storage capex, imbalance or grid-connection charges.",
}
HOURS_PER_YEAR = {2020: 8784, 2021: 8760, 2022: 8760, 2023: 8760, 2024: 8784}
EPSILON_GRID_EUR_PER_KG = (0.0, 0.01, 0.05, 0.10, 0.25)
PRIMARY_EPSILON = 0.05


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result: {path}")
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def load_sites(full_four_zone: bool = False, custom_site_file: Path | None = None) -> tuple[pd.DataFrame, Path]:
    if custom_site_file is not None:
        if full_four_zone:
            raise ValueError("--site-file and --full-four-zone-population are mutually exclusive")
        if not custom_site_file.is_file():
            raise FileNotFoundError(custom_site_file)
        sites = pd.read_csv(custom_site_file, dtype={"candidate_id": str})
        expected_zones = set(ZONE_EICS)
        if set(sites["bidding_zone_key"].dropna().astype(str)) != expected_zones:
            raise ValueError("custom site file must include all four audited bidding zones")
        if len(sites) < 4 or sites["candidate_id"].nunique() != len(sites):
            raise ValueError("a custom candidate cohort must contain unique candidate IDs and at least one site per zone")
        if sites.groupby("bidding_zone_key").size().eq(0).any():
            raise ValueError("a custom candidate cohort must contain at least one site in every audited zone")
        if len(sites) == 128 and sites.groupby("bidding_zone_key").size().to_dict() != {zone: 32 for zone in ZONE_EICS}:
            raise ValueError("the frozen 128-site validation cohort must contain 32 candidates per zone")
        source_path = custom_site_file
    elif full_four_zone:
        if not FULL_ZONE_MAPPING.is_file():
            raise FileNotFoundError(FULL_ZONE_MAPPING)
        sites = pd.read_csv(FULL_ZONE_MAPPING, dtype={"candidate_id": str})
        sites = sites.loc[
            sites["assignment_status"].eq("VERIFIED")
            & sites["bidding_zone_eic"].isin(ZONE_EICS.values())
        ].copy()
        expected = {"DE_LU": 401, "FR": 96, "BE": 44, "NL": 40}
        if sites.groupby("bidding_zone_key").size().to_dict() != expected:
            raise ValueError("verified four-zone population no longer matches the audited 581-candidate roster")
        source_path = FULL_ZONE_MAPPING
    else:
        sites = pd.read_csv(SITE_FILE, dtype={"candidate_id": str})
        if len(sites) != 32 or sites["candidate_id"].nunique() != 32:
            raise ValueError("the pilot site file must contain 32 unique candidates")
        expected = {"DE_LU": 8, "FR": 8, "BE": 8, "NL": 8}
        if sites.groupby("bidding_zone_key").size().to_dict() != expected:
            raise ValueError("the pilot must contain eight candidates in each of four bidding zones")
        source_path = SITE_FILE
    if sites["candidate_id"].duplicated().any() or sites["candidate_id"].isna().any():
        raise ValueError("candidate roster must contain unique non-missing candidate IDs")
    return sites.reset_index(drop=True), source_path


def load_pv_cache() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    expected_ids: np.ndarray | None = None
    for year in (2020, 2021, 2022, 2023):
        manifest_path = PV_MANIFESTS[year]
        matrix_path = PV_MATRICES[year]
        if not manifest_path.is_file() or not matrix_path.is_file():
            raise FileNotFoundError(f"missing archived PVGIS cache for {year}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS_COMPLETE_VALIDATED_PANEL":
            raise ValueError(f"PV cache manifest is not a validated panel for {year}")
        archive = np.load(matrix_path, allow_pickle=False)
        ids = archive["candidate_ids"].astype(str)
        timestamps = archive["timestamps_utc_ns"]
        matrix = archive["pv_mw_per_mwp"]
        if expected_ids is None:
            expected_ids = ids
        if not np.array_equal(ids, expected_ids):
            raise ValueError("candidate row order differs across yearly PV cache matrices")
        if matrix.shape != (len(ids), HOURS_PER_YEAR[year]):
            raise ValueError(f"unexpected PV matrix shape for {year}: {matrix.shape}")
        raw_dir = Path(manifest["raw_directory"])
        raw_sample = raw_dir / f"{ids[0]}.json"
        if not raw_sample.is_file():
            raise FileNotFoundError(f"cannot independently verify PV source for {year}: {raw_sample}")
        raw_record = json.loads(raw_sample.read_text(encoding="utf-8"))
        meteo = raw_record.get("inputs", {}).get("meteo_data", {})
        if meteo.get("radiation_db") != "PVGIS-SARAH3" or int(meteo.get("year_min", -1)) != year or int(meteo.get("year_max", -1)) != year:
            raise ValueError(f"raw PVGIS response does not verify SARAH3 source/year for {year}")
        idx = pd.to_datetime(timestamps, utc=True)
        expected_idx = pd.date_range(f"{year}-01-01T00:00:00Z", periods=HOURS_PER_YEAR[year], freq="h")
        if not idx.equals(expected_idx):
            raise ValueError(f"PVGIS UTC-hour index does not match calendar year {year}")
        result[year] = {
            "ids": ids,
            "row": {candidate: i for i, candidate in enumerate(ids)},
            "timestamps": idx,
            "matrix": matrix,
            "manifest_path": manifest_path,
            "matrix_path": matrix_path,
            "manifest": manifest,
            "source_verified_by_raw_response": str(raw_sample),
        }
    return result


def historical_climatology(
    candidate_id: str,
    target_year: int,
    pv_cache: dict[int, dict[str, Any]],
) -> tuple[pd.DatetimeIndex, np.ndarray, float]:
    history_years = tuple(year for year in (2020, 2021, 2022, 2023) if year < target_year)
    if not history_years:
        raise ValueError("no historical PV years precede the requested target year")
    by_key: dict[tuple[int, int, int], list[float]] = {}
    historical_yields: list[float] = []
    for year in history_years:
        info = pv_cache[year]
        if candidate_id not in info["row"]:
            raise KeyError(f"candidate {candidate_id} is absent from the complete PVGIS cache")
        values = np.asarray(info["matrix"][info["row"][candidate_id]], dtype=float)
        if not np.isfinite(values).all() or (values < -1e-8).any():
            raise ValueError(f"invalid PV generation profile for {candidate_id}/{year}")
        historical_yields.append(float(values.sum()))
        idx = info["timestamps"]
        for month, day, hour, value in zip(idx.month, idx.day, idx.hour, values, strict=True):
            by_key.setdefault((int(month), int(day), int(hour)), []).append(float(value))
    clim = {key: float(np.mean(values)) for key, values in by_key.items()}
    target_idx = pd.date_range(f"{target_year}-01-01T00:00:00Z", periods=HOURS_PER_YEAR[target_year], freq="h")
    forecast = np.empty(len(target_idx), dtype=float)
    for i, (month, day, hour) in enumerate(zip(target_idx.month, target_idx.day, target_idx.hour, strict=True)):
        key = (int(month), int(day), int(hour))
        if key in clim:
            forecast[i] = clim[key]
        elif month == 2 and day == 29:
            feb28 = clim[(2, 28, int(hour))]
            mar01 = clim[(3, 1, int(hour))]
            forecast[i] = (feb28 + mar01) / 2
        else:
            raise ValueError(f"no prior-year PV climatology for {candidate_id} at {key}")
    return target_idx, forecast, float(np.mean(historical_yields))


def load_prices(year: int) -> dict[str, np.ndarray]:
    if year in (2022, 2023):
        path = PRICE_2022 if year == 2022 else PRICE_2023
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df = df.loc[df["bidding_zone_eic"].isin(ZONE_EICS.values())].copy()
        eic_zone = {eic: zone for zone, eic in ZONE_EICS.items()}
        df["bidding_zone_key"] = df["bidding_zone_eic"].map(eic_zone)
        price_col = "price_eur_per_mwh"
    elif year == 2024:
        path = PRICE_2024
        df = pd.read_csv(path)
        price_col = "price_eur_2023_per_mwh"
    else:
        raise ValueError("price panels are defined only for 2022-2024")
    if not path.is_file():
        raise FileNotFoundError(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="raise")
    result: dict[str, np.ndarray] = {}
    expected_idx = pd.date_range(f"{year}-01-01T00:00:00Z", periods=HOURS_PER_YEAR[year], freq="h")
    for zone in ZONE_EICS:
        rows = df.loc[df["bidding_zone_key"].eq(zone), ["timestamp_utc", price_col]].sort_values("timestamp_utc")
        if len(rows) != HOURS_PER_YEAR[year] or rows["timestamp_utc"].duplicated().any():
            raise ValueError(f"{zone}/{year} price coverage is incomplete or duplicated")
        if not pd.DatetimeIndex(rows["timestamp_utc"]).equals(expected_idx):
            raise ValueError(f"{zone}/{year} SMARD timestamps do not exactly match the UTC calendar year")
        values = pd.to_numeric(rows[price_col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{zone}/{year} has missing or nonfinite price values")
        result[zone] = values
    return result


def forecast_day_ahead_prices(
    target_year: int,
    prices_by_year: dict[int, dict[str, np.ndarray]],
    method: str = "four_week_mean",
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Forecast each delivery-hour price from four prior same-weekday hours.

    The lag rule is fixed in advance: mean prices at t-7, t-14, t-21, and
    t-28 days. Although the combined source object contains the full target
    year for ex-post scoring, every indexed predictor timestamp precedes the
    target delivery timestamp by at least seven days.
    """
    if method not in {"four_week_mean", "last_week_same_hour"}:
        raise ValueError(f"unsupported price forecast method: {method}")
    target_idx = pd.date_range(f"{target_year}-01-01T00:00:00Z", periods=HOURS_PER_YEAR[target_year], freq="h")
    forecasts: dict[str, np.ndarray] = {}
    quality: list[dict[str, Any]] = []
    source_years = (2022, 2023, target_year) if target_year == 2024 else (2022, 2023)
    source_years = tuple(dict.fromkeys(source_years))
    for zone in ZONE_EICS:
        source_parts: list[pd.Series] = []
        for year in source_years:
            if year not in prices_by_year:
                raise KeyError(f"missing historical prices for forecast construction: {year}")
            idx = pd.date_range(f"{year}-01-01T00:00:00Z", periods=HOURS_PER_YEAR[year], freq="h")
            source_parts.append(pd.Series(prices_by_year[year][zone], index=idx, dtype=float))
        history = pd.concat(source_parts).sort_index()
        if history.index.has_duplicates:
            raise ValueError(f"duplicate price-history timestamps for {zone}")
        lag_matrix = np.vstack(
            [history.reindex(target_idx - pd.Timedelta(days=7 * lag)).to_numpy(dtype=float) for lag in range(1, 5)]
        )
        if not np.isfinite(lag_matrix).all():
            raise ValueError(f"insufficient or missing 4-week price history for {zone}/{target_year}")
        four_week_mean = lag_matrix.mean(axis=0)
        forecast = four_week_mean if method == "four_week_mean" else lag_matrix[0]
        forecasts[zone] = forecast
        actual = prices_by_year[target_year][zone]
        last_week = lag_matrix[0]
        four_week_error = four_week_mean - actual
        forecast_error = forecast - actual
        last_week_error = last_week - actual
        quality.append(
            {
                "year": target_year,
                "zone": zone,
                "hours": len(actual),
                "selected_forecast_method": method,
                "selected_forecast_mae_eur_per_mwh": float(np.mean(np.abs(forecast_error))),
                "selected_forecast_rmse_eur_per_mwh": float(np.sqrt(np.mean(forecast_error**2))),
                "selected_forecast_bias_eur_per_mwh": float(np.mean(forecast_error)),
                "four_week_mean_mae_eur_per_mwh": float(np.mean(np.abs(four_week_error))),
                "four_week_mean_rmse_eur_per_mwh": float(np.sqrt(np.mean(four_week_error**2))),
                "four_week_mean_bias_eur_per_mwh": float(np.mean(four_week_error)),
                "last_week_same_hour_mae_eur_per_mwh": float(np.mean(np.abs(last_week_error))),
                "last_week_same_hour_rmse_eur_per_mwh": float(np.sqrt(np.mean(last_week_error**2))),
                "forecast_rule": "mean of t-7d, t-14d, t-21d, t-28d; same UTC hour" if method == "four_week_mean" else "t-7d; same UTC hour",
                "realized_prices_used_for_dispatch": False,
            }
        )
    return forecasts, quality


def relaxed_daily_lower_bound(price: np.ndarray, pv_available: np.ndarray, target_mwh: float) -> float:
    """Exact LP relaxation for one day, retaining hourly capacity and PV bounds.

    At negative prices, grid supply is selected before zero-cost PV. At
    nonnegative prices, PV is selected first and grid can use only the residual
    electrolyser capacity. Sorting these hourly marginal blocks solves the
    continuous daily relaxation; min-load, commitment, ramp, and inventory
    trajectory constraints are relaxed, so the result is a lower bound.
    """
    blocks: list[tuple[float, float, int]] = []
    for hour, p in enumerate(price):
        pv_cap = float(np.clip(pv_available[hour], 0.0, MODEL["electrolyser_capacity_mwe"]))
        if p < 0:
            blocks.append((float(p), MODEL["electrolyser_capacity_mwe"], hour))
        else:
            if pv_cap > 0:
                blocks.append((0.0, pv_cap, hour))
            grid_cap = MODEL["electrolyser_capacity_mwe"] - pv_cap
            if grid_cap > 0:
                blocks.append((float(p), grid_cap, hour))
    blocks.sort(key=lambda block: (block[0], block[2]))
    remaining = target_mwh
    cost = 0.0
    for unit_cost, capacity, _hour in blocks:
        use = min(remaining, capacity)
        cost += use * unit_cost
        remaining -= use
        if remaining <= 1e-9:
            break
    if remaining > 1e-7:
        raise RuntimeError("daily LP relaxation cannot satisfy the fixed hydrogen service")
    return float(cost)


def relaxed_daily_lower_bound_operational_lp(
    price: np.ndarray,
    pv_available: np.ndarray,
    utilisation: float,
) -> float:
    """LP relaxation retaining hourly PV, tank trajectory, and ramp limits.

    Binary commitment is relaxed to [0, 1]. Start/stop variables are omitted
    because they have no objective coefficient or minimum-up/down coupling in
    the stated model. The first-hour ramp constraint is omitted, which only
    enlarges the feasible set and therefore preserves a valid lower bound.
    """
    if len(price) != 24 or len(pv_available) != 24:
        raise ValueError("operational LP relaxation requires exactly 24 hourly values")
    cap = MODEL["electrolyser_capacity_mwe"]
    spec = MODEL["electrolyser_specific_electricity_kwh_per_kg"]
    minimum = MODEL["minimum_stable_load_fraction"] * cap
    ramp = MODEL["ramp_limit_mwe_per_hour"]
    hourly_demand = cap * utilisation * 1000.0 / spec
    tank_cap = MODEL["hydrogen_storage_hours_of_demand"] * hourly_demand
    load = np.arange(0, 24)
    pv = np.arange(24, 48)
    grid = np.arange(48, 72)
    inventory = np.arange(72, 96)
    on = np.arange(96, 120)
    c = np.zeros(120, dtype=float)
    c[grid] = np.asarray(price, dtype=float)

    a_eq = np.zeros((48, 120), dtype=float)
    b_eq = np.zeros(48, dtype=float)
    for h in range(24):
        # Electrical balance: load = PV + grid.
        a_eq[h, load[h]] = 1.0
        a_eq[h, pv[h]] = -1.0
        a_eq[h, grid[h]] = -1.0
        # H2 inventory balance: today's inventory = prior inventory + output - demand.
        row = 24 + h
        a_eq[row, inventory[h]] = 1.0
        a_eq[row, load[h]] = -1000.0 / spec
        if h > 0:
            a_eq[row, inventory[h - 1]] = -1.0
        b_eq[row] = -hourly_demand

    a_ub = np.zeros((94, 120), dtype=float)
    b_ub = np.zeros(94, dtype=float)
    for h in range(24):
        # Relaxed commitment envelope.
        a_ub[h, load[h]] = 1.0
        a_ub[h, on[h]] = -cap
        row = 24 + h
        a_ub[row, load[h]] = -1.0
        a_ub[row, on[h]] = minimum
    row = 48
    for h in range(1, 24):
        a_ub[row, load[h]] = 1.0
        a_ub[row, load[h - 1]] = -1.0
        b_ub[row] = ramp
        row += 1
        a_ub[row, load[h - 1]] = 1.0
        a_ub[row, load[h]] = -1.0
        b_ub[row] = ramp
        row += 1
    # Unused rows remain 0 <= 0 and are harmless; retain a stable shape.
    bounds = (
        [(0.0, cap)] * 24
        + [(0.0, float(np.clip(v, 0.0, cap))) for v in pv_available]
        + [(0.0, cap)] * 24
        + [(0.0, tank_cap)] * 23
        + [(0.0, 0.0)]
        + [(0.0, 1.0)] * 24
    )
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.fun is None or not np.isfinite(result.fun):
        raise RuntimeError(f"operational LP lower bound failed: {result.message}")
    return float(result.fun)


def constant_feasible_daily_cost(price: np.ndarray, pv_available: np.ndarray, utilisation: float) -> tuple[float, float, float]:
    """Feasible constant-load schedule used as a low-cost incumbent upper bound."""
    cap = MODEL["electrolyser_capacity_mwe"]
    load = cap * utilisation
    pv_use = np.where(price < 0.0, 0.0, np.minimum(pv_available, load))
    grid = load - pv_use
    cost = float(np.dot(price, grid))
    target_mwh = cap * utilisation * 24.0
    target_kg = target_mwh * 1000.0 / MODEL["electrolyser_specific_electricity_kwh_per_kg"]
    return cost, float(grid.sum()), float(target_kg)


def solve_day_milp(
    price: np.ndarray,
    pv_per_mwp: np.ndarray,
    *,
    settlement_price: np.ndarray | None = None,
    previous_power_mwe: float,
    previous_on: int,
    utilisation: float = 0.50,
) -> dict[str, Any]:
    """24-hour unit-commitment MILP with minimum load, ramp, and finite H2 tank."""
    if len(price) != 24 or len(pv_per_mwp) != 24:
        raise ValueError("daily MILP requires exactly 24 hourly price and PV values")
    if settlement_price is None:
        settlement_price = price
    if len(settlement_price) != 24:
        raise ValueError("daily settlement price vector must contain exactly 24 hours")
    cap = MODEL["electrolyser_capacity_mwe"]
    spec = MODEL["electrolyser_specific_electricity_kwh_per_kg"]
    min_load = MODEL["minimum_stable_load_fraction"] * cap
    ramp = MODEL["ramp_limit_mwe_per_hour"]
    hourly_demand_kg = cap * utilisation * 1000.0 / spec
    tank_cap = MODEL["hydrogen_storage_hours_of_demand"] * hourly_demand_kg
    pv_available = np.minimum(np.maximum(pv_per_mwp, 0.0) * MODEL["pv_capacity_mwp_per_mwe"], cap)

    # Variable blocks: load, PV use, grid import, on/off, H2 inventory,
    # start flags, shutdown flags. All arrays are indexed by UTC hour.
    load = np.arange(0, 24)
    pv = np.arange(24, 48)
    grid = np.arange(48, 72)
    on = np.arange(72, 96)
    inventory = np.arange(96, 120)
    starts = np.arange(120, 144)
    stops = np.arange(144, 168)
    variable_count = 168
    lower = np.zeros(variable_count, dtype=float)
    upper = np.ones(variable_count, dtype=float)
    upper[load] = cap
    upper[pv] = pv_available
    upper[grid] = cap
    upper[inventory] = tank_cap
    lower[inventory[-1]] = 0.0
    upper[inventory[-1]] = 0.0  # Daily delivery contract: cyclic end inventory.
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[on] = 1
    integrality[starts] = 1
    integrality[stops] = 1

    c = np.zeros(variable_count, dtype=float)
    c[grid] = price
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    row_lb: list[float] = []
    row_ub: list[float] = []

    def add(coefficients: dict[int, float], lb: float = -np.inf, ub: float = np.inf) -> None:
        row = len(row_lb)
        for col, value in coefficients.items():
            if value != 0:
                rows.append(row)
                cols.append(int(col))
                values.append(float(value))
        row_lb.append(float(lb))
        row_ub.append(float(ub))

    rate = 1000.0 / spec  # kg per MWh
    for h in range(24):
        # Electrical balance: electrolyser load equals PV plus grid supply.
        add({int(load[h]): 1.0, int(pv[h]): -1.0, int(grid[h]): -1.0}, 0.0, 0.0)
        # Binary commitment and stable-load envelope.
        add({int(load[h]): 1.0, int(on[h]): -cap}, ub=0.0)
        add({int(load[h]): 1.0, int(on[h]): -min_load}, lb=0.0)
        # Exact start/shutdown indicator constraints, including the carried
        # status across the day boundary.
        if h == 0:
            prev = int(previous_on)
            # starts >= on_t - prev; starts <= on_t; starts <= 1 - prev.
            add({int(on[h]): 1.0, int(starts[h]): -1.0}, ub=float(prev))
            add({int(starts[h]): 1.0, int(on[h]): -1.0}, ub=0.0)
            add({int(starts[h]): 1.0}, ub=1.0 - prev)
            # stops >= prev - on_t; stops <= prev; stops <= 1 - on_t.
            add({int(on[h]): 1.0, int(stops[h]): 1.0}, lb=float(prev))
            add({int(stops[h]): 1.0}, ub=float(prev))
            add({int(stops[h]): 1.0, int(on[h]): 1.0}, ub=1.0)
        else:
            # starts >= on_t - on_(t-1); starts <= on_t; starts <= 1-on_(t-1).
            add({int(on[h]): 1.0, int(on[h - 1]): -1.0, int(starts[h]): -1.0}, ub=0.0)
            add({int(starts[h]): 1.0, int(on[h]): -1.0}, ub=0.0)
            add({int(starts[h]): 1.0, int(on[h - 1]): 1.0}, ub=1.0)
            # stops >= on_(t-1) - on_t; stops <= on_(t-1); stops <= 1-on_t.
            add({int(on[h]): 1.0, int(stops[h]): 1.0, int(on[h - 1]): -1.0}, lb=0.0)
            add({int(stops[h]): 1.0, int(on[h - 1]): -1.0}, ub=0.0)
            add({int(stops[h]): 1.0, int(on[h]): 1.0}, ub=1.0)
        # Ramping from the preceding hour; one-hour-step, full-scale bound.
        if h == 0:
            add({int(load[h]): 1.0}, ub=ramp + previous_power_mwe)
            add({int(load[h]): -1.0}, ub=ramp - previous_power_mwe)
        else:
            add({int(load[h]): 1.0, int(load[h - 1]): -1.0}, ub=ramp)
            add({int(load[h]): -1.0, int(load[h - 1]): 1.0}, ub=ramp)
        # Fixed hourly hydrogen delivery with a bounded within-day tank.
        if h == 0:
            add({int(inventory[h]): 1.0, int(load[h]): -rate}, -hourly_demand_kg, -hourly_demand_kg)
        else:
            add(
                {int(inventory[h]): 1.0, int(inventory[h - 1]): -1.0, int(load[h]): -rate},
                -hourly_demand_kg,
                -hourly_demand_kg,
            )
    matrix = coo_array((values, (rows, cols)), shape=(len(row_lb), variable_count)).tocsr()
    constraints = LinearConstraint(matrix, np.asarray(row_lb), np.asarray(row_ub))
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"disp": False, "mip_rel_gap": 1e-9},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"daily unit-commitment MILP failed: {result.message}")
    solution = result.x
    load_values = solution[load]
    pv_values = solution[pv]
    grid_values = solution[grid]
    on_values = np.rint(solution[on]).astype(int)
    inv_values = solution[inventory]
    starts_count = int((np.diff(np.r_[int(previous_on), on_values]) == 1).sum())
    stops_count = int((np.diff(np.r_[int(previous_on), on_values]) == -1).sum())
    expected_mwh = cap * utilisation * 24.0
    if not np.isclose(load_values.sum(), expected_mwh, atol=1e-6):
        raise RuntimeError("daily MILP violates the fixed hydrogen-energy service")
    if not np.allclose(load_values, pv_values + grid_values, atol=1e-7):
        raise RuntimeError("daily MILP violates electricity balance")
    if inv_values.min() < -1e-7 or inv_values.max() > tank_cap + 1e-6 or abs(inv_values[-1]) > 1e-7:
        raise RuntimeError("daily MILP violates tank bounds or cyclic terminal inventory")
    if np.max(np.abs(np.diff(np.r_[previous_power_mwe, load_values]))) > ramp + 1e-6:
        raise RuntimeError("daily MILP violates ramp bounds")
    return {
        "forecast_objective_eur": float(np.dot(price, grid_values)),
        "realized_settlement_eur": float(np.dot(settlement_price, grid_values)),
        "load_mwh": float(load_values.sum()),
        "grid_mwh": float(grid_values.sum()),
        "pv_mwh": float(pv_values.sum()),
        "starts": starts_count,
        "stops": stops_count,
        "ending_power_mwe": float(load_values[-1]),
        "ending_on": int(on_values[-1]),
        "min_inventory_kg": float(inv_values.min()),
        "max_inventory_kg": float(inv_values.max()),
        "solver_mip_gap": float(getattr(result, "mip_gap", 0.0) or 0.0),
    }


def simulate_candidate_year(
    site: pd.Series,
    year: int,
    pv_forecast: np.ndarray,
    price_forecast_by_zone: dict[str, np.ndarray],
    settlement_price_by_zone: dict[str, np.ndarray],
    *,
    max_days: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    zone = str(site["bidding_zone_key"])
    forecast_prices = price_forecast_by_zone[zone]
    settlement_prices = settlement_price_by_zone[zone]
    n_days = HOURS_PER_YEAR[year] // 24
    if any(len(values) != HOURS_PER_YEAR[year] for values in (pv_forecast, forecast_prices, settlement_prices)):
        raise ValueError(f"{site.candidate_id}/{year}: price and PV forecast lengths do not match")
    if max_days > 0:
        n_days = min(n_days, max_days)
    previous_power = 0.0
    previous_on = 0
    daily: list[dict[str, Any]] = []
    total_forecast_cost = 0.0
    total_settlement_cost = 0.0
    total_grid = 0.0
    total_pv = 0.0
    total_starts = 0
    started = time.perf_counter()
    for day in range(n_days):
        sl = slice(day * 24, (day + 1) * 24)
        outcome = solve_day_milp(
            forecast_prices[sl],
            pv_forecast[sl],
            settlement_price=settlement_prices[sl],
            previous_power_mwe=previous_power,
            previous_on=previous_on,
            utilisation=MODEL["annual_average_utilisation"],
        )
        total_forecast_cost += outcome["forecast_objective_eur"]
        total_settlement_cost += outcome["realized_settlement_eur"]
        total_grid += outcome["grid_mwh"]
        total_pv += outcome["pv_mwh"]
        total_starts += outcome["starts"]
        utc_date = (pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day)).date().isoformat()
        daily.append({"candidate_id": str(site["candidate_id"]), "zone": zone, "year": year, "day_index": day, "utc_date": utc_date, **outcome})
        previous_power = outcome["ending_power_mwe"]
        previous_on = outcome["ending_on"]
    elapsed = time.perf_counter() - started
    kg = n_days * 24 * MODEL["electrolyser_capacity_mwe"] * MODEL["annual_average_utilisation"] * 1000.0 / MODEL["electrolyser_specific_electricity_kwh_per_kg"]
    summary = {
        "candidate_id": str(site["candidate_id"]),
        "zone": zone,
        "year": year,
        "simulated_days": n_days,
        "full_calendar_year": n_days == HOURS_PER_YEAR[year] // 24,
        "forecast_objective_eur": total_forecast_cost,
        "forecast_objective_eur_per_kg": total_forecast_cost / kg if kg else float("nan"),
        "realized_settlement_eur": total_settlement_cost,
        "realized_settlement_eur_per_kg": total_settlement_cost / kg if kg else float("nan"),
        "grid_mwh": total_grid,
        "pv_used_mwh": total_pv,
        "hydrogen_kg": kg,
        "starts": total_starts,
        "milp_wall_seconds": elapsed,
        "objective_scope": MODEL["objective_scope"],
        "interpretation": "Forecast-price objective drives dispatch; actual DA price is used only for ex-post settlement scoring.",
    }
    return summary, daily


def compute_certificate(
    sites: pd.DataFrame,
    pv_cache: dict[int, dict[str, Any]],
    forecast_prices_2023: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = time.perf_counter()
    records: list[dict[str, Any]] = []
    cap = MODEL["electrolyser_capacity_mwe"]
    util = MODEL["annual_average_utilisation"]
    target_mwh = cap * util * 24.0
    for _, site in sites.iterrows():
        candidate_id = str(site["candidate_id"])
        _, pv_forecast, pv_yield = historical_climatology(candidate_id, 2023, pv_cache)
        zone = str(site["bidding_zone_key"])
        price = forecast_prices_2023[zone]
        pv_avail = np.minimum(pv_forecast * MODEL["pv_capacity_mwp_per_mwe"], cap)
        lb_total = 0.0
        operational_lp_total = 0.0
        ub_total = 0.0
        hydrogen_total = 0.0
        for day in range(HOURS_PER_YEAR[2023] // 24):
            sl = slice(day * 24, (day + 1) * 24)
            lb_total += relaxed_daily_lower_bound(price[sl], pv_avail[sl], target_mwh)
            operational_lp_total += relaxed_daily_lower_bound_operational_lp(
                price[sl], pv_avail[sl], util
            )
            ub_cost, _grid, kg_day = constant_feasible_daily_cost(price[sl], pv_avail[sl], util)
            ub_total += ub_cost
            hydrogen_total += kg_day
        analytic_bound = lb_total / hydrogen_total
        operational_lp_bound = operational_lp_total / hydrogen_total
        if operational_lp_bound + 1e-7 < analytic_bound:
            raise RuntimeError(
                f"stronger operational LP bound fell below the analytic relaxation for {candidate_id}: "
                f"{operational_lp_bound} < {analytic_bound}"
            )
        records.append(
            {
                "candidate_id": candidate_id,
                "zone": zone,
                "prior_mean_pv_yield_2020_2022_mwh_per_mwp": pv_yield,
                "analytic_relaxation_lower_bound_2023_forecast_objective_eur_per_kg": analytic_bound,
                "operational_lp_lower_bound_2023_forecast_objective_eur_per_kg": operational_lp_bound,
                "lower_bound_2023_forecast_objective_eur_per_kg": operational_lp_bound,
                "constant_policy_upper_bound_2023_forecast_objective_eur_per_kg": ub_total / hydrogen_total,
                "upper_bound_policy": "constant 50% load; PV first at nonnegative prices; grid first and PV curtailed at negative prices",
            }
        )
    frame = pd.DataFrame(records)
    if (frame["lower_bound_2023_forecast_objective_eur_per_kg"] > frame["constant_policy_upper_bound_2023_forecast_objective_eur_per_kg"] + 1e-7).any():
        raise RuntimeError("a purported lower bound exceeds the feasible constant-policy upper bound")
    best_upper_row = frame.loc[frame["constant_policy_upper_bound_2023_forecast_objective_eur_per_kg"].idxmin()]
    metadata = {
        "bound_evaluation_wall_seconds": time.perf_counter() - start,
        "bound_formulation": "Daily LP relaxation retains electrical balance, candidate-specific hourly PV availability, finite within-day hydrogen inventory, hourly delivery, and hourly ramp constraints; binary commitment is relaxed to [0,1]. The first-hour ramp is omitted to preserve relaxation validity.",
        "comparison_relaxation": "The analytic daily merit-order bound relaxes storage trajectory, minimum load, commitment, and ramp constraints; it is reported only as a weaker comparator.",
        "best_constant_policy_candidate_id": str(best_upper_row["candidate_id"]),
        "best_constant_policy_upper_bound_eur_per_kg": float(best_upper_row["constant_policy_upper_bound_2023_forecast_objective_eur_per_kg"]),
        "primary_certificate_rule": "within each bidding zone z, discard i only if L_i > U_z + epsilon; strict inequality preserves ties",
        "global_certificate_comparator": "discard i only if L_i > min_z(U_z) + epsilon; this preserves the global optimum but may eliminate an entire price zone",
        "numerical_tolerance_eur_per_kg": 1e-7,
        "scope": "finite 2023 forecast-price objective and the specified day-ahead/climatological PV model only; no guarantee for realized 2023 settlement cost, 2024, or future years",
    }
    return frame, metadata


def certificate_keep_ids(bounds: pd.DataFrame, best_upper: float, epsilon: float) -> set[str]:
    threshold = best_upper + epsilon
    keep = bounds.loc[bounds["lower_bound_2023_forecast_objective_eur_per_kg"] <= threshold + 1e-7, "candidate_id"]
    if len(keep) == 0:
        raise RuntimeError("certificate unexpectedly eliminated every candidate")
    return set(keep.astype(str))


def certificate_keep_ids_by_zone(
    bounds: pd.DataFrame,
    best_upper_by_zone: dict[str, float],
    epsilon: float,
) -> set[str]:
    thresholds = bounds["zone"].astype(str).map(best_upper_by_zone)
    if thresholds.isna().any():
        raise ValueError("a candidate zone is missing its feasible incumbent upper bound")
    keep = bounds.loc[
        bounds["lower_bound_2023_forecast_objective_eur_per_kg"]
        <= thresholds + epsilon + 1e-7,
        "candidate_id",
    ]
    if keep.empty:
        raise RuntimeError("zone-specific certificate unexpectedly eliminated every candidate")
    return set(keep.astype(str))


def top_ids_by_zone(
    scores: dict[str, float],
    sites: pd.DataFrame,
    share: float = 0.10,
    higher_is_better: bool = False,
) -> set[str]:
    if not 0.0 < share <= 1.0:
        raise ValueError("share must be in (0, 1]")
    selected: set[str] = set()
    for _zone, group in sites.groupby("bidding_zone_key", sort=False):
        count = max(1, int(np.ceil(share * len(group))))
        ids = group["candidate_id"].astype(str).tolist()
        selected.update(
            sorted(ids, key=lambda cid: ((-scores[cid]) if higher_is_better else scores[cid], cid))[:count]
        )
    return selected


def run(args: argparse.Namespace) -> Path:
    sites, site_file = load_sites(args.full_four_zone_population, args.site_file)
    pv_cache = load_pv_cache()
    missing = set(sites["candidate_id"].astype(str)).difference(set(pv_cache[2023]["row"]))
    if missing:
        raise KeyError(f"pilot sites missing from PV cache: {sorted(missing)}")
    if args.max_sites > 0:
        maximum = len(sites) if args.full_four_zone_population else (len(sites) if args.site_file else 32)
        if args.max_sites % 4 != 0 or not 4 <= args.max_sites <= maximum:
            raise ValueError(f"--max-sites must be divisible by 4 and between 4 and {maximum}")
        sites = (
            sites.groupby("bidding_zone_key", sort=False, group_keys=False)
            .head(args.max_sites // 4)
            .reset_index(drop=True)
        )
    actual_prices = {year: load_prices(year) for year in (2022, 2023, 2024)}
    price_forecasts: dict[int, dict[str, np.ndarray]] = {}
    price_forecast_quality: list[dict[str, Any]] = []
    for year in (2023, 2024):
        price_forecasts[year], quality = forecast_day_ahead_prices(year, actual_prices, args.price_forecast_method)
        price_forecast_quality.extend(quality)
    if args.smoke:
        site = sites.iloc[0]
        _, forecast, _ = historical_climatology(str(site.candidate_id), 2023, pv_cache)
        zone = str(site.bidding_zone_key)
        price = price_forecasts[2023][zone][:24]
        start = time.perf_counter()
        sample = solve_day_milp(
            price,
            forecast[:24],
            settlement_price=actual_prices[2023][zone][:24],
            previous_power_mwe=0.0,
            previous_on=0,
        )
        print(json.dumps({"status": "PASS_DAILY_MILP_SMOKE", "candidate_id": str(site.candidate_id), "zone": str(site.bidding_zone_key), "model": MODEL, "outcome": sample, "wall_seconds": time.perf_counter() - start}, ensure_ascii=True, indent=2))
        return Path()
    utilization_tag = f"u{int(round(MODEL['annual_average_utilisation'] * 100)):02d}"
    price_method_tag = "fourweek" if args.price_forecast_method == "four_week_mean" else "lastweek"
    checkpoint_source_dir = args.reuse_checkpoint_dir.resolve() if args.reuse_checkpoint_dir else None
    checkpoint_source_manifest: dict[str, Any] | None = None
    if checkpoint_source_dir is not None:
        source_manifest_path = checkpoint_source_dir / "run_manifest.json"
        source_checkpoints = checkpoint_source_dir / "candidate_checkpoints"
        if not source_manifest_path.is_file() or not source_checkpoints.is_dir():
            raise FileNotFoundError(f"checkpoint source is incomplete: {checkpoint_source_dir}")
        checkpoint_source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if int(checkpoint_source_manifest.get("candidate_count", -1)) != len(sites):
            raise ValueError("checkpoint source candidate count differs from the requested roster")
        if checkpoint_source_manifest.get("site_roster_sha256", "").lower() != file_hash(site_file).lower():
            raise ValueError("checkpoint source roster hash differs from the requested roster")
        source_model = checkpoint_source_manifest.get("model", {})
        for key in ("annual_average_utilisation", "price_forecast_method", "electrolyser_capacity_mwe",
                    "electrolyser_specific_electricity_kwh_per_kg", "pv_capacity_mwp_per_mwe",
                    "minimum_stable_load_fraction", "ramp_limit_mwe_per_hour",
                    "hydrogen_storage_hours_of_demand", "startup_cost_eur_per_start"):
            if source_model.get(key) != MODEL.get(key):
                raise ValueError(f"checkpoint source model mismatch for {key}")
    run_id = datetime.now(timezone.utc).strftime(f"run_%Y%m%dT%H%M%SZ_{utilization_tag}_{price_method_tag}")
    out_dir = RESULT_ROOT / run_id
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    checkpoints_dir = out_dir / "candidate_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    write_new(
        out_dir / "run_manifest.json",
        (
            json.dumps(
                {
                    "status": "RUNNING",
                    "run_id": run_id,
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "candidate_count": len(sites),
                    "candidate_counts_by_zone": sites.groupby("bidding_zone_key").size().to_dict(),
                    "site_roster_path": str(site_file),
                    "site_roster_sha256": file_hash(site_file),
                    "model": MODEL,
                    "primary_epsilon_eur_per_kg": PRIMARY_EPSILON,
                    "checkpoint_format": "one JSON file per solved site-year, including daily dispatch rows",
                    "reused_checkpoint_source": str(checkpoint_source_dir) if checkpoint_source_dir else None,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )

    def save_candidate_checkpoint(year: int, candidate_id: str, summary: dict[str, Any], daily: list[dict[str, Any]], phase: str) -> None:
        checkpoint_path = checkpoints_dir / f"{year}_{candidate_id}.json"
        payload = {
            "phase": phase,
            "year": year,
            "candidate_id": candidate_id,
            "summary": summary,
            "daily_dispatch": daily,
        }
        write_new(checkpoint_path, (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        append_jsonl(
            progress_path,
            {
                "phase": phase,
                "year": year,
                "candidate_id": candidate_id,
                "milp_wall_seconds": summary.get("milp_wall_seconds"),
                "checkpoint": str(checkpoint_path),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def load_reused_checkpoint(year: int, candidate_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        if checkpoint_source_dir is None:
            return None
        checkpoint_path = checkpoint_source_dir / "candidate_checkpoints" / f"{year}_{candidate_id}.json"
        if not checkpoint_path.is_file():
            return None
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if int(payload.get("year", -1)) != year or str(payload.get("candidate_id")) != candidate_id:
            raise ValueError(f"checkpoint identity mismatch: {checkpoint_path}")
        summary = payload.get("summary", {})
        daily = payload.get("daily_dispatch", [])
        expected_days = HOURS_PER_YEAR[year] // 24
        if not summary.get("full_calendar_year") or int(summary.get("simulated_days", -1)) != expected_days:
            raise ValueError(f"checkpoint is not a complete calendar-year result: {checkpoint_path}")
        if len(daily) != expected_days or any(
            int(row.get("year", -1)) != year or str(row.get("candidate_id")) != candidate_id
            for row in daily
        ):
            raise ValueError(f"checkpoint dispatch rows are incomplete or mismatched: {checkpoint_path}")
        return summary, daily

    bounds, cert_meta = compute_certificate(sites, pv_cache, price_forecasts[2023])
    anchor_summaries: dict[str, dict[str, Any]] = {}
    anchor_dailies: dict[str, list[dict[str, Any]]] = {}
    anchor_pv_scores: dict[str, float] = {}
    anchor_ids_by_zone: dict[str, str] = {}
    for zone in ZONE_EICS:
        zone_bounds = bounds.loc[bounds["zone"].astype(str).eq(zone)]
        anchor_row = zone_bounds.loc[zone_bounds["lower_bound_2023_forecast_objective_eur_per_kg"].idxmin()]
        anchor_id = str(anchor_row["candidate_id"])
        anchor_ids_by_zone[zone] = anchor_id
        anchor_site = sites.loc[sites["candidate_id"].astype(str).eq(anchor_id)].iloc[0]
        _, anchor_pv_forecast, anchor_pv_score = historical_climatology(anchor_id, 2023, pv_cache)
        cached_anchor = load_reused_checkpoint(2023, anchor_id)
        if cached_anchor is None:
            anchor_summary, anchor_daily = simulate_candidate_year(
                anchor_site,
                2023,
                anchor_pv_forecast,
                price_forecasts[2023],
                actual_prices[2023],
            )
            anchor_phase = "train_reference_zone_anchor"
        else:
            anchor_summary, anchor_daily = cached_anchor
            anchor_phase = "train_reference_zone_anchor_reused"
        anchor_summaries[anchor_id] = anchor_summary
        anchor_dailies[anchor_id] = anchor_daily
        anchor_pv_scores[anchor_id] = anchor_pv_score
        save_candidate_checkpoint(2023, anchor_id, anchor_summary, anchor_daily, anchor_phase)

    best_upper_by_zone: dict[str, float] = {}
    incumbent_by_zone: dict[str, str] = {}
    constant_upper_by_zone = bounds.groupby("zone")[
        "constant_policy_upper_bound_2023_forecast_objective_eur_per_kg"
    ].min().to_dict()
    bounds["anchor_exact_forecast_objective_eur_per_kg"] = np.nan
    bounds["candidate_feasible_upper_bound_2023_forecast_objective_eur_per_kg"] = bounds[
        "constant_policy_upper_bound_2023_forecast_objective_eur_per_kg"
    ]
    for zone, anchor_id in anchor_ids_by_zone.items():
        anchor_summary = anchor_summaries[anchor_id]
        anchor_row = bounds.loc[bounds["candidate_id"].astype(str).eq(anchor_id)].iloc[0]
        anchor_upper = float(anchor_summary["forecast_objective_eur_per_kg"])
        constant_upper = float(constant_upper_by_zone[zone])
        best_upper_by_zone[zone] = min(anchor_upper, constant_upper)
        const_row = bounds.loc[
            bounds["zone"].astype(str).eq(zone)
            & bounds["constant_policy_upper_bound_2023_forecast_objective_eur_per_kg"].eq(constant_upper)
        ].iloc[0]
        incumbent_by_zone[zone] = anchor_id if anchor_upper <= constant_upper else str(const_row["candidate_id"])
        bounds.loc[
            bounds["candidate_id"].astype(str).eq(anchor_id),
            "anchor_exact_forecast_objective_eur_per_kg",
        ] = anchor_upper
        bounds.loc[
            bounds["candidate_id"].astype(str).eq(anchor_id),
            "candidate_feasible_upper_bound_2023_forecast_objective_eur_per_kg",
        ] = min(
            anchor_upper,
            float(anchor_row["constant_policy_upper_bound_2023_forecast_objective_eur_per_kg"]),
        )
    bounds["zone_feasible_incumbent_upper_bound_2023_forecast_objective_eur_per_kg"] = bounds[
        "zone"
    ].astype(str).map(best_upper_by_zone)
    best_upper = min(best_upper_by_zone.values())
    global_keep = certificate_keep_ids(bounds, best_upper, PRIMARY_EPSILON)
    primary_keep = certificate_keep_ids_by_zone(bounds, best_upper_by_zone, PRIMARY_EPSILON)
    anchor_solver_seconds = sum(float(summary["milp_wall_seconds"]) for summary in anchor_summaries.values())
    cert_meta.update(
        {
            "primary_screening_scope": "within each bidding zone, preserve all candidates whose LP lower bound is no more than epsilon above a feasible incumbent upper bound for that same zone",
            "anchor_candidate_id_by_zone": anchor_ids_by_zone,
            "anchor_exact_forecast_objective_eur_per_kg_by_zone": {
                zone: float(anchor_summaries[cid]["forecast_objective_eur_per_kg"])
                for zone, cid in anchor_ids_by_zone.items()
            },
            "feasible_incumbent_upper_bound_eur_per_kg_by_zone": best_upper_by_zone,
            "incumbent_candidate_id_by_zone": incumbent_by_zone,
            "anchor_solver_wall_seconds_total": anchor_solver_seconds,
            "global_comparator_upper_bound_eur_per_kg": best_upper,
            "global_comparator_retained_ids": sorted(global_keep),
            "primary_within_zone_retained_ids": sorted(primary_keep),
            "screening_wall_seconds": float(cert_meta["bound_evaluation_wall_seconds"]) + anchor_solver_seconds,
            "incumbent_policy": "per zone, the minimum of a feasible constant-load policy and one exact sequential daily-MILP schedule for the candidate with the smallest operational LP lower bound",
        }
    )
    write_new(
        out_dir / "certificate_bounds_2023.csv",
        bounds.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    write_new(
        out_dir / "certificate_definition.json",
        (json.dumps({"epsilon_eur_per_kg": PRIMARY_EPSILON, "certificate": cert_meta, "model": MODEL}, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    bound_map = bounds.set_index("candidate_id")
    print(
        json.dumps(
            {
                "phase": "certificate_train",
                "anchors_by_zone": anchor_ids_by_zone,
                "epsilon_eur_per_kg": PRIMARY_EPSILON,
                "retained_by_zone": bounds.loc[bounds["candidate_id"].astype(str).isin(primary_keep)].groupby("zone").size().to_dict(),
                "retained_total": len(primary_keep),
                "global_comparator_retained": len(global_keep),
                "seconds": cert_meta["screening_wall_seconds"],
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    # Full 2023 reference is retained for the prior-cost comparator and for
    # direct validation that every annual LP lower bound is admissible.
    train_summaries: list[dict[str, Any]] = []
    train_daily: list[dict[str, Any]] = []
    pv_score: dict[str, float] = dict(anchor_pv_scores)
    pv_forecast_quality: list[dict[str, Any]] = []
    for n, (_, site) in enumerate(sites.iterrows(), start=1):
        cid = str(site["candidate_id"])
        _idx, train_forecast, score = historical_climatology(cid, 2023, pv_cache)
        pv_score[cid] = score
        pv_actual = np.asarray(pv_cache[2023]["matrix"][pv_cache[2023]["row"][cid]], dtype=float)
        pv_error = train_forecast - pv_actual
        pv_forecast_quality.append(
            {
                "candidate_id": cid,
                "zone": str(site["bidding_zone_key"]),
                "year": 2023,
                "hours": len(pv_actual),
                "mae_mw_per_mwp": float(np.mean(np.abs(pv_error))),
                "rmse_mw_per_mwp": float(np.sqrt(np.mean(pv_error**2))),
                "bias_mw_per_mwp": float(np.mean(pv_error)),
                "forecast_rule": "prior-year mean PVGIS-SARAH3 output by UTC month-day-hour (2020-2022)",
            }
        )
        if cid in anchor_summaries:
            summary, daily = anchor_summaries[cid], anchor_dailies[cid]
        else:
            cached = load_reused_checkpoint(2023, cid)
            if cached is None:
                summary, daily = simulate_candidate_year(
                    site,
                    2023,
                    train_forecast,
                    price_forecasts[2023],
                    actual_prices[2023],
                )
                reused = False
            else:
                summary, daily = cached
                reused = True
        train_summaries.append(summary)
        train_daily.extend(daily)
        if cid not in anchor_summaries:
            save_candidate_checkpoint(2023, cid, summary, daily, "train_reference")
        lb = float(bound_map.loc[cid, "lower_bound_2023_forecast_objective_eur_per_kg"])
        ub = float(bound_map.loc[cid, "constant_policy_upper_bound_2023_forecast_objective_eur_per_kg"])
        forecast_cost = float(summary["forecast_objective_eur_per_kg"])
        if lb > forecast_cost + 1e-5 or forecast_cost > ub + 1e-5:
            raise RuntimeError(f"forecast-objective bound validation failed for {cid}: LP={lb}, MILP={forecast_cost}, feasible={ub}")
        print(json.dumps({"phase": "train_reference", "complete": n, "of": len(sites), "candidate_id": cid, "forecast_objective_eur_per_kg": forecast_cost, "realized_settlement_eur_per_kg": summary["realized_settlement_eur_per_kg"], "seconds": summary["milp_wall_seconds"], "reused_checkpoint": bool(cid not in anchor_summaries and reused) if cid not in anchor_summaries else bool(checkpoint_source_dir is not None)}, ensure_ascii=True), flush=True)

    train_costs = pd.DataFrame(train_summaries)
    train_cost_map = train_costs.set_index("candidate_id")["realized_settlement_eur_per_kg"].to_dict()
    train_forecast_cost_map = train_costs.set_index("candidate_id")["forecast_objective_eur_per_kg"].to_dict()
    top_n = max(1, int(np.ceil(0.10 * len(sites))))
    pv_top = set(sorted(pv_score, key=lambda cid: (-pv_score[cid], cid))[:top_n])
    prior_cost_top = set(sorted(train_cost_map, key=lambda cid: (train_cost_map[cid], cid))[:top_n])
    pv_top_by_zone = top_ids_by_zone(pv_score, sites, higher_is_better=True)
    prior_cost_top_by_zone = top_ids_by_zone(train_cost_map, sites)
    train_costs["prior_mean_pv_yield_2020_2022_mwh_per_mwp"] = train_costs["candidate_id"].map(pv_score)
    train_costs = train_costs.merge(bounds, on=["candidate_id", "zone", "prior_mean_pv_yield_2020_2022_mwh_per_mwp"], how="left")
    train_costs["is_pv_top_decile"] = train_costs["candidate_id"].isin(pv_top)
    train_costs["is_prior_cost_top_decile"] = train_costs["candidate_id"].isin(prior_cost_top)
    train_costs["is_pv_top_decile_within_zone"] = train_costs["candidate_id"].isin(pv_top_by_zone)
    train_costs["is_prior_cost_top_decile_within_zone"] = train_costs["candidate_id"].isin(prior_cost_top_by_zone)

    # The certificate is frozen using 2023 forecast prices. The primary 2024
    # run solves retained sites; omitted-site solves below are an evaluation
    # oracle to measure external-year settlement regret and are not counted as
    # certificate runtime.
    test_pv: dict[str, np.ndarray] = {}
    test_scores: dict[str, float] = {}
    for _, site in sites.iterrows():
        idx, forecast, score = historical_climatology(str(site.candidate_id), 2024, pv_cache)
        if not idx.equals(pd.date_range("2024-01-01T00:00:00Z", periods=8784, freq="h")):
            raise ValueError("2024 PV climatology calendar is not a complete UTC leap year")
        test_pv[str(site.candidate_id)] = forecast
        test_scores[str(site.candidate_id)] = score
    test_summaries: dict[str, dict[str, Any]] = {}
    test_daily: list[dict[str, Any]] = []
    retained_runtime = 0.0
    for n, (_, site) in enumerate(sites.loc[sites["candidate_id"].astype(str).isin(primary_keep)].iterrows(), start=1):
        cid = str(site["candidate_id"])
        cached = load_reused_checkpoint(2024, cid)
        reused = cached is not None
        if reused:
            summary, daily = cached
        else:
            summary, daily = simulate_candidate_year(site, 2024, test_pv[cid], price_forecasts[2024], actual_prices[2024])
        test_summaries[cid] = summary
        test_daily.extend(daily)
        save_candidate_checkpoint(2024, cid, summary, daily, "test_certificate_retained_reused" if reused else "test_certificate_retained")
        retained_runtime += summary["milp_wall_seconds"]
        print(json.dumps({"phase": "test_certificate_retained", "complete": n, "of": len(primary_keep), "candidate_id": cid, "seconds": summary["milp_wall_seconds"], "reused_checkpoint": reused}, ensure_ascii=True), flush=True)
    for n, (_, site) in enumerate(sites.loc[~sites["candidate_id"].astype(str).isin(primary_keep)].iterrows(), start=1):
        cid = str(site["candidate_id"])
        cached = load_reused_checkpoint(2024, cid)
        reused = cached is not None
        if reused:
            summary, daily = cached
        else:
            summary, daily = simulate_candidate_year(site, 2024, test_pv[cid], price_forecasts[2024], actual_prices[2024])
        test_summaries[cid] = summary
        test_daily.extend(daily)
        save_candidate_checkpoint(2024, cid, summary, daily, "test_oracle_omitted_reused" if reused else "test_oracle_omitted")
        print(json.dumps({"phase": "test_oracle_omitted", "complete": n, "of": len(sites)-len(primary_keep), "candidate_id": cid, "seconds": summary["milp_wall_seconds"], "reused_checkpoint": reused}, ensure_ascii=True), flush=True)

    test_cost_map = {cid: float(row["realized_settlement_eur_per_kg"]) for cid, row in test_summaries.items()}
    test_best = min(test_cost_map.values())
    test_all_best_id = min(test_cost_map, key=lambda cid: (test_cost_map[cid], cid))
    test_best_by_zone = {
        str(zone): min(test_cost_map[str(cid)] for cid in group["candidate_id"].astype(str))
        for zone, group in sites.groupby("bidding_zone_key")
    }
    test_pv_top = set(sorted(test_scores, key=lambda cid: (-test_scores[cid], cid))[:top_n])
    test_pv_top_by_zone = top_ids_by_zone(test_scores, sites, higher_is_better=True)
    sets: dict[str, set[str]] = {
        "annual_PV_top_decile_global": test_pv_top,
        "prior_2023_realized_cost_top_decile_global": prior_cost_top,
        "annual_PV_top_decile_within_zone": test_pv_top_by_zone,
        "prior_2023_realized_cost_top_decile_within_zone": prior_cost_top_by_zone,
    }
    for epsilon in EPSILON_GRID_EUR_PER_KG:
        suffix = f"{epsilon:.2f}"
        sets[f"certificate_global_epsilon_{suffix}"] = certificate_keep_ids(bounds, best_upper, epsilon)
        sets[f"certificate_within_zone_epsilon_{suffix}"] = certificate_keep_ids_by_zone(
            bounds, best_upper_by_zone, epsilon
        )

    full_test_runtime = sum(float(row["milp_wall_seconds"]) for row in test_summaries.values())
    frontier: list[dict[str, Any]] = []
    market_frontier: list[dict[str, Any]] = []
    for label, selected in sets.items():
        best_selected = min(test_cost_map[cid] for cid in selected)
        selector_runtime = sum(float(test_summaries[cid]["milp_wall_seconds"]) for cid in selected)
        frontier.append(
            {
                "selector": label,
                "retained_candidates": len(selected),
                "retained_share": len(selected) / len(sites),
                "test_2024_best_retained_realized_settlement_eur_per_kg": best_selected,
                "test_2024_all_candidates_best_realized_settlement_eur_per_kg": test_best,
                "test_2024_realized_settlement_regret_eur_per_kg": best_selected - test_best,
                "test_2024_best_candidate_id": min((cid for cid in selected if abs(test_cost_map[cid] - best_selected) < 1e-9), default=""),
                "test_2024_global_best_candidate_id": test_all_best_id,
                "test_2024_selected_solver_seconds": selector_runtime,
                "test_2024_solver_runtime_reduction_after_screening": 1.0 - (selector_runtime + float(cert_meta["screening_wall_seconds"])) / full_test_runtime if full_test_runtime else 0.0,
                "guarantee_scope": "2023 forecast-price certificate is model conditional; 2024 realized settlement regret is an external price-year assessment and is not certified",
            }
        )
        selected_by_id = set(selected)
        for zone, group in sites.groupby("bidding_zone_key"):
            zone_ids = set(group["candidate_id"].astype(str))
            zone_selected = selected_by_id & zone_ids
            all_zone_best = test_best_by_zone[str(zone)]
            if zone_selected:
                retained_zone_best = min(test_cost_map[cid] for cid in zone_selected)
                zone_regret: float | None = retained_zone_best - all_zone_best
                retained_id = min((cid for cid in zone_selected if abs(test_cost_map[cid] - retained_zone_best) < 1e-9), default="")
            else:
                retained_zone_best = None
                zone_regret = None
                retained_id = ""
            market_frontier.append(
                {
                    "selector": label,
                    "zone": str(zone),
                    "candidate_count_in_zone": len(zone_ids),
                    "retained_in_zone": len(zone_selected),
                    "retained_share_in_zone": len(zone_selected) / len(zone_ids),
                    "best_retained_2024_realized_settlement_eur_per_kg": retained_zone_best,
                    "all_candidates_2024_best_realized_settlement_eur_per_kg": all_zone_best,
                    "within_zone_2024_realized_settlement_regret_eur_per_kg": zone_regret,
                    "best_retained_candidate_id": retained_id,
                    "guarantee_scope": "2024 settlement regret is evaluated ex post and is not certified",
                }
            )
    frontier_df = pd.DataFrame(frontier)
    market_frontier_df = pd.DataFrame(market_frontier)
    primary_label = f"certificate_within_zone_epsilon_{PRIMARY_EPSILON:.2f}"
    global_label = f"certificate_global_epsilon_{PRIMARY_EPSILON:.2f}"
    primary_test_regret = float(
        frontier_df.loc[frontier_df["selector"].eq(primary_label), "test_2024_realized_settlement_regret_eur_per_kg"].iloc[0]
    )
    primary_summary = next(row for row in frontier if row["selector"] == primary_label)
    primary_market_rows = [row for row in market_frontier if row["selector"] == primary_label]
    primary_market_regrets = [row["within_zone_2024_realized_settlement_regret_eur_per_kg"] for row in primary_market_rows]
    primary_market_pruning = [1.0 - float(row["retained_share_in_zone"]) for row in primary_market_rows]
    overall_candidate_pruning = 1.0 - float(primary_summary["retained_share"])
    if any(value is None for value in primary_market_regrets):
        worst_market_test_regret: float | None = None
    else:
        worst_market_test_regret = max(float(value) for value in primary_market_regrets)
    if primary_summary["retained_candidates"] < 1 or primary_summary["retained_candidates"] > len(sites):
        raise RuntimeError("invalid within-zone certificate retention size")

    training_frontier: list[dict[str, Any]] = []
    train_best = min(train_forecast_cost_map.values())
    train_best_by_zone = {
        str(zone): min(train_forecast_cost_map[str(cid)] for cid in group["candidate_id"].astype(str))
        for zone, group in sites.groupby("bidding_zone_key")
    }
    for epsilon in EPSILON_GRID_EUR_PER_KG:
        candidate_sets = {
            "global": certificate_keep_ids(bounds, best_upper, epsilon),
            "within_zone": certificate_keep_ids_by_zone(bounds, best_upper_by_zone, epsilon),
        }
        for strategy, keep in candidate_sets.items():
            regret = min(train_forecast_cost_map[cid] for cid in keep) - train_best
            zone_regrets: dict[str, float | None] = {}
            for zone, group in sites.groupby("bidding_zone_key"):
                zone_ids = set(group["candidate_id"].astype(str))
                selected_zone = keep & zone_ids
                zone_regrets[str(zone)] = (
                    min(train_forecast_cost_map[cid] for cid in selected_zone) - train_best_by_zone[str(zone)]
                    if selected_zone
                    else None
                )
            finite_zone_regrets = [float(value) for value in zone_regrets.values() if value is not None]
            worst_zone_regret = (
                max(finite_zone_regrets) if len(finite_zone_regrets) == sites["bidding_zone_key"].nunique() else None
            )
            check_value = regret if strategy == "global" else worst_zone_regret
            if check_value is None or check_value > epsilon + 2e-5:
                raise RuntimeError(
                    f"{strategy} epsilon certificate failed at epsilon={epsilon}: regret={check_value}"
                )
            training_frontier.append(
                {
                    "strategy": strategy,
                    "epsilon_eur_per_kg": epsilon,
                    "retained_candidates": len(keep),
                    "retained_share": len(keep) / len(sites),
                    "global_2023_forecast_objective_regret_eur_per_kg": regret,
                    "worst_zone_2023_forecast_objective_regret_eur_per_kg": worst_zone_regret,
                    "zone_regret_eur_per_kg": json.dumps(zone_regrets, sort_keys=True),
                    "theoretical_regret_upper_bound_eur_per_kg": epsilon,
                    "bound_check": "PASS",
                }
            )

    screening_seconds = float(cert_meta["screening_wall_seconds"])
    retained_runtime = float(primary_summary["test_2024_selected_solver_seconds"])
    screened_pipeline_runtime = retained_runtime + screening_seconds
    runtime_reduction_after_screening = 1.0 - (screened_pipeline_runtime / full_test_runtime if full_test_runtime else 0.0)
    primary_training_row = next(
        row for row in training_frontier
        if row["strategy"] == "within_zone" and abs(row["epsilon_eur_per_kg"] - PRIMARY_EPSILON) < 1e-12
    )
    screening_value_gate = bool(
        overall_candidate_pruning >= 0.20
        and primary_market_pruning
        and min(primary_market_pruning) >= 0.20
        and runtime_reduction_after_screening > 0.0
        and primary_training_row["worst_zone_2023_forecast_objective_regret_eur_per_kg"] <= PRIMARY_EPSILON + 2e-5
        and worst_market_test_regret is not None
        and worst_market_test_regret <= PRIMARY_EPSILON + 1e-9
    )
    global_summary = next(row for row in frontier if row["selector"] == global_label)
    run_summary = {
        "status": "PASS_FROZEN_EXTERNAL_VALIDATION_GATE" if screening_value_gate else "FAIL_FROZEN_EXTERNAL_VALIDATION_GATE",
        "screening_value_gate": {
            "status": "PASS" if screening_value_gate else "FAIL",
            "rule": "Within-zone primary screen must eliminate at least 20% of candidates overall and in every market, reduce solver runtime after charging screening construction, pass the exact 2023 worst-zone regret bound, and have worst-zone 2024 realized settlement regret no greater than 0.05 EUR/kg.",
            "retained_candidates": int(primary_summary["retained_candidates"]),
            "candidate_count": len(sites),
            "overall_candidate_pruning_fraction": overall_candidate_pruning,
            "minimum_market_candidate_pruning_fraction": min(primary_market_pruning) if primary_market_pruning else 0.0,
            "retained_candidates_by_zone": {
                row["zone"]: row["retained_in_zone"] for row in primary_market_rows
            },
            "runtime_reduction_after_screening": runtime_reduction_after_screening,
            "2024_realized_settlement_regret_eur_per_kg": primary_test_regret,
            "worst_zone_2024_realized_settlement_regret_eur_per_kg": worst_market_test_regret,
            "worst_zone_2023_exact_forecast_objective_regret_eur_per_kg": primary_training_row[
                "worst_zone_2023_forecast_objective_regret_eur_per_kg"
            ],
            "primary_epsilon_eur_per_kg": PRIMARY_EPSILON,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "python": sys.version,
        "platform": platform.platform(),
        "sample": {
            "candidate_count": len(sites),
            "cohort_mode": (
                "full verified four-zone population"
                if args.full_four_zone_population
                else (
                    "frozen 128-site spatial holdout"
                    if args.site_file and len(sites) == 128
                    else ("custom candidate cohort" if args.site_file else "balanced development sample")
                )
            ),
            "balanced_integration_slice": not args.full_four_zone_population and len(sites) < 32,
            "market_level_units": sites["bidding_zone_key"].nunique(),
            "candidate_count_by_zone": sites.groupby("bidding_zone_key").size().to_dict(),
            "source_roster_path": str(site_file),
            "source_roster_sha256": file_hash(site_file),
            "nonindependence_note": f"Sites sharing a bidding zone share the same hourly price series; n={len(sites)} candidates is not n={sites['bidding_zone_key'].nunique()} independent markets.",
        },
        "input_provenance": {
            "prices_2022_history": {"path": str(PRICE_2022), "sha256": file_hash(PRICE_2022)},
            "prices_2023": {"path": str(PRICE_2023), "sha256": file_hash(PRICE_2023)},
            "prices_2024": {"path": str(PRICE_2024), "sha256": file_hash(PRICE_2024)},
            "pv_years": {
                str(year): {
                    "matrix_path": str(PV_MATRICES[year]),
                    "matrix_sha256": file_hash(PV_MATRICES[year]),
                    "manifest_path": str(PV_MANIFESTS[year]),
                    "manifest_sha256": file_hash(PV_MANIFESTS[year]),
                    "raw_response_source_check_path": pv_cache[year]["source_verified_by_raw_response"],
                    "raw_response_source_check_sha256": file_hash(Path(pv_cache[year]["source_verified_by_raw_response"])),
                }
                for year in (2020, 2021, 2022, 2023)
            },
        },
        "model": MODEL,
        "pilot_boundary": f"The dispatch objective uses only forecasts formed from information available before each delivery day using {args.price_forecast_method}; 2023 prices use prior 2022-2023 observations and 2024 prices use prior 2022-2024 observations. Actual DA prices are used only for settlement scoring. The 2023 PV proxy uses 2020-2022 climatology; 2024 uses 2020-2023 climatology. No 2024 actual weather series is used.",
        "certificate": {
            **cert_meta,
            "primary_epsilon_eur_per_kg": PRIMARY_EPSILON,
            "primary_retained_ids": sorted(primary_keep),
            "primary_worst_zone_2024_realized_settlement_regret_eur_per_kg": worst_market_test_regret,
            "global_comparator_retained_ids_at_primary_epsilon": sorted(global_keep),
            "global_comparator_test_2024_regret_eur_per_kg": global_summary[
                "test_2024_realized_settlement_regret_eur_per_kg"
            ],
            "training_frontier": training_frontier,
            "primary_2024_realized_settlement_regret_eur_per_kg": primary_test_regret,
            "bound_validation": "PASS: all exact 2023 forecast-price MILP objectives lie above the operational daily LP lower bound and below a feasible constant-load policy upper bound within 1e-5 EUR/kg.",
        },
        "forecast_quality": price_forecast_quality,
        "runtime": {
            "full_2023_reference_solver_seconds": sum(float(row["milp_wall_seconds"]) for row in train_summaries),
            "certificate_screening_seconds": screening_seconds,
            "2024_all_candidate_reference_solver_seconds": full_test_runtime,
            "2024_retained_candidate_solver_seconds_observed": retained_runtime,
            "2024_retained_plus_certificate_screen_seconds": screened_pipeline_runtime,
            "2024_solver_runtime_reduction_fraction_after_screening": runtime_reduction_after_screening,
            "global_comparator_retained_solver_seconds": global_summary["test_2024_selected_solver_seconds"],
            "global_comparator_runtime_reduction_after_screening": global_summary[
                "test_2024_solver_runtime_reduction_after_screening"
            ],
            "note": "All-candidate 2024 MILP solves are an evaluation oracle and are not counted in retained-only runtime. Screening time includes operational LP bound calculation plus one feasible exact 2023 anchor solve per zone. All candidate MILP timings use the same machine and model; these are solver, not total data-pipeline, runtimes.",
        },
        "caveats": [
            "2024 is an external price-year holdout, not a PV weather holdout; PV dispatch input is a prior-year climatology proxy, not a numerical weather forecast.",
            "The certificate is rigorous only for the finite 2023 forecast-price objective and stated operating assumptions; it does not certify realized settlement cost, future prices, or real project costs.",
            "Startup transitions are represented and counted but not monetised; sub-hour start dynamics are unresolved at hourly granularity.",
            "Hydrogen cost is variable electricity procurement cost only, not LCOH or delivered hydrogen cost.",
            "Four price zones provide a feasibility pilot, not a representative EU-27 economic sample.",
        ],
    }
    annual = train_costs.merge(
        pd.DataFrame([{"candidate_id": cid, **row} for cid, row in test_summaries.items()]),
        on=["candidate_id", "zone"],
        suffixes=("_2023", "_2024"),
        how="inner",
    )
    write_new(out_dir / "candidate_cohort.csv", sites.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    write_new(out_dir / "annual_candidate_costs_and_bounds.csv", annual.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    write_new(out_dir / "certificate_frontier_training_and_test.csv", frontier_df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    write_new(out_dir / "market_frontier_2024.csv", market_frontier_df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    write_new(out_dir / "certificate_training_regret_check.csv", pd.DataFrame(training_frontier).to_csv(index=False, lineterminator="\n").encode("utf-8"))
    write_new(out_dir / "price_forecast_quality.csv", pd.DataFrame(price_forecast_quality).to_csv(index=False, lineterminator="\n").encode("utf-8"))
    write_new(out_dir / "pv_forecast_quality_2023.csv", pd.DataFrame(pv_forecast_quality).to_csv(index=False, lineterminator="\n").encode("utf-8"))
    write_new(out_dir / "daily_dispatch_2023.csv", pd.DataFrame(train_daily).to_csv(index=False, lineterminator="\n").encode("utf-8"))
    write_new(out_dir / "daily_dispatch_2024.csv", pd.DataFrame(test_daily).to_csv(index=False, lineterminator="\n").encode("utf-8"))
    write_new(out_dir / "run_summary.json", (json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    write_new(
        out_dir / "run_completed.json",
        (json.dumps({"status": run_summary["status"], "completed_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps({"status": run_summary["status"], "screening_value_gate": run_summary["screening_value_gate"], "output_dir": str(out_dir), "training_frontier": training_frontier, "test_frontier": frontier, "runtime": run_summary["runtime"]}, ensure_ascii=True, indent=2), flush=True)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="solve one 2023 day only; no paper results are produced")
    parser.add_argument("--max-sites", type=int, default=0, help="balanced subset size; default runs all sites in the selected roster")
    parser.add_argument("--site-file", type=Path, default=None, help="run a frozen candidate roster with all four audited price zones")
    parser.add_argument("--full-four-zone-population", action="store_true", help="expand from the 32-site pilot to all verified candidate points in DE-LU, FR, BE, and NL")
    parser.add_argument("--utilisation", type=float, default=0.50, help="fixed electrolyser annual utilisation target; planned levels are 0.30, 0.50, and 0.70")
    parser.add_argument("--price-forecast-method", choices=("four_week_mean", "last_week_same_hour"), default="four_week_mean", help="frozen causal price forecast rule")
    parser.add_argument("--reuse-checkpoint-dir", type=Path, default=None, help="reuse validated full-year candidate checkpoints from a prior compatible interrupted run")
    args = parser.parse_args()
    if not 0.0 < args.utilisation <= 1.0:
        parser.error("--utilisation must be in (0, 1]")
    MODEL["annual_average_utilisation"] = float(args.utilisation)
    MODEL["price_forecast_method"] = args.price_forecast_method
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
