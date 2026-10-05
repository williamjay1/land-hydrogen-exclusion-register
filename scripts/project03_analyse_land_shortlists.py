"""Audit whether electricity-only screens lose regional land-context options.

The Land extension is retrospective. Cost outcomes evaluate screens; the
land-aware screening function can obtain MILP costs ONLY for anchors selected
from land metrics and LP bounds. Timings are charged checkpoint replays, not a
new end-to-end benchmark. No all-candidate oracle is a free screening input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

PROJECT = Path(r"D:\MLWork\Europe_green_hydrogen_siting")
ENERGY = PROJECT / "results/regret_certificate_pilot_20261002_v1"
LAND = PROJECT / "datasets/intermediate/land_context_clc2018_natura2024_20261005_v1"
OUT = PROJECT / "results/land_shortlist_assessment_20261005_v2"
RUNS = {
    "u50_fourweek": "run_20261002T063739Z_u50_fourweek",
    "u30_fourweek": "run_20261002T072226Z_u30_fourweek",
    "u70_fourweek": "run_20261002T080302Z_u70_fourweek",
    "u50_lastweek": "run_20261004T173404Z_u50_lastweek",
}
VIEWS = {
    "industrial_context": [("industrial_commercial_fraction", -1)],
    "agricultural_context": [("agricultural_fraction", 1)],
    "natural_context": [("forest_seminatural_fraction", 1)],
    "protected_context": [("natura2000_union_fraction", 1)],
    "joint_context": [("industrial_commercial_fraction", -1), ("agricultural_fraction", 1),
                      ("forest_seminatural_fraction", 1), ("natura2000_union_fraction", 1)],
}
LOWER = "lower_bound_2023_forecast_objective_eur_per_kg"
CONSTANT = "constant_policy_upper_bound_2023_forecast_objective_eur_per_kg"
MODEL_COST = "forecast_objective_eur_per_kg_2023"
EPSILON = 0.05
NUMERICAL_GUARD = 1e-7


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def nondominated(cost: np.ndarray, land: np.ndarray, epsilon: float = 0) -> np.ndarray:
    """Strict cost improvement by epsilon, no-worse land: conservative frontier.

    At zero this can retain exact cost ties; this is intentional, not a claim
    that epsilon is a multivariate distance or that land has a scalar weight.
    """
    cost_dominates = cost[:, None] + epsilon < cost[None, :] - NUMERICAL_GUARD
    # Never certify a worse context through a positive tolerance: this
    # relation must remain transitive for threshold and dependency guarantees.
    land_no_worse = np.all(land[:, None, :] <= land[None, :, :], axis=2)
    return ~np.any(cost_dominates & land_no_worse, axis=0)


def land_intervals(frame: pd.DataFrame, view: list, gap_robust: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.column_stack([frame[col].to_numpy(float) * direction for col, direction in view])
    lo, hi = values.copy(), values.copy()
    if gap_robust:
        missing = frame.clc_unknown_fraction.to_numpy(float)
        for k, (column, direction) in enumerate(view):
            if column == "natura2000_union_fraction":
                continue
            if direction == 1:
                hi[:, k] = np.minimum(1, hi[:, k] + missing)
            else:
                lo[:, k] = np.maximum(-1, lo[:, k] - missing)
    return values, lo, hi


def bound_land_screen(frame: pd.DataFrame, lo: np.ndarray, hi: np.ndarray, solve_anchor,
                      extra_land_anchors: bool) -> tuple[np.ndarray, list[dict], list[str], float]:
    """No oracle cost input: callback is invoked only for deterministic anchors."""
    began = time.perf_counter()
    ids = frame.candidate_id.astype(str).tolist()
    lower = frame[LOWER].to_numpy(float)
    upper = frame[CONSTANT].to_numpy(float).copy()
    anchor_indexes = [min(range(len(frame)), key=lambda i: (lower[i], ids[i]))]
    if extra_land_anchors:
        for k in range(hi.shape[1]):
            anchor_indexes.append(min(range(len(frame)), key=lambda i: (hi[i, k], lower[i], ids[i])))
    anchor_indexes = list(dict.fromkeys(anchor_indexes))
    charged_seconds = 0.0
    for index in anchor_indexes:
        cost, seconds = solve_anchor(ids[index])
        upper[index] = min(upper[index], cost)
        charged_seconds += seconds
    no_worse = np.all(hi[:, None, :] <= lo[None, :, :], axis=2)
    dominated = (upper[:, None] + EPSILON < lower[None, :] - NUMERICAL_GUARD) & no_worse
    witnesses = []
    keep = ~np.any(dominated, axis=0)
    for i in np.flatnonzero(~keep):
        js = np.flatnonzero(dominated[:, i])
        j = min(js, key=lambda j: (upper[j], ids[j]))
        witnesses.append({
            "removed": ids[i], "witness": ids[j], "upper_witness": float(upper[j]),
            "lower_removed": float(lower[i]), "cost_margin_eur_per_kg": float(lower[i] - upper[j]),
            "witness_land_upper": hi[j].tolist(), "removed_land_lower": lo[i].tolist(),
        })
    return keep, witnesses, [ids[i] for i in anchor_indexes], charged_seconds + time.perf_counter() - began


def self_test() -> None:
    # Cheapest region need not represent the regional land trade-off frontier.
    costs = np.array([1., 1.2, 1.4, 1.5])
    land = np.array([[.9], [.2], [.3], [.8]])
    assert nondominated(costs, land).tolist() == [True, True, False, False]
    assert nondominated(costs, land, .25).tolist() == [True, True, True, False]
    sample = pd.DataFrame({"candidate_id": list("ABCD"), LOWER: costs - .02, CONSTANT: costs + .2})
    calls = []
    def solve(cid):
        calls.append(cid)
        return costs[list("ABCD").index(cid)], .01
    keep, witnesses, anchors, _ = bound_land_screen(sample, land, land, solve, True)
    assert calls == ["A", "B"] and anchors == calls
    assert keep.tolist() == [True, True, False, False]
    assert not (nondominated(costs, land, EPSILON) & ~keep).any()
    for w in witnesses:
        assert w["cost_margin_eur_per_kg"] > EPSILON
    # Lower bounds on land prevent using an unknown area as favourable cover.
    lo = np.array([[.15], [.2]])
    hi = np.array([[.3], [.2]])
    # No robust superiority of A over B: its missing land could make it worse.
    assert not bool(np.all(hi[0] <= lo[1]))
    # A positive tolerance is not transitive. The decreasing-cost chain below
    # must NOT remove any of the increasingly worse context alternatives.
    tiny_context = np.array([[0.], [.75e-12], [1.5e-12]])
    tiny_cost = np.array([3., 2., 1.])
    assert nondominated(tiny_cost, tiny_context).all()
    tiny_frame = pd.DataFrame({"candidate_id": ["x0", "x1", "x2"],
                               LOWER: tiny_cost, CONSTANT: tiny_cost})
    tiny_keep, tiny_witnesses, _, _ = bound_land_screen(
        tiny_frame, tiny_context, tiny_context,
        lambda cid: (tiny_cost[int(cid[1:])], 0.), True)
    assert tiny_keep.all() and not tiny_witnesses
    for threshold in np.unique(tiny_context):
        eligible = tiny_context[:, 0] <= threshold
        assert (eligible & tiny_keep).any()
    # Random bound ensembles: all epsilon-relevant land options must survive.
    random = np.random.default_rng(20261005)
    for _ in range(100):
        true_cost = random.uniform(1, 3, 12)
        context = random.uniform(0, 1, (12, 3))
        f = pd.DataFrame({"candidate_id": [f"x{i}" for i in range(12)],
                          LOWER: true_cost - random.uniform(0, .1, 12),
                          CONSTANT: true_cost + random.uniform(0, .5, 12)})
        def callback(cid):
            return true_cost[int(cid[1:])], 0.
        retained, _, _, _ = bound_land_screen(f, context, context, callback, True)
        assert not (nondominated(true_cost, context, EPSILON) & ~retained).any()


def corrected_energy(frame: pd.DataFrame) -> tuple[dict, set[str]]:
    retained = set()
    zone_rows = []
    for zone, group in frame.groupby("zone", sort=True):
        anchor = group.sort_values([LOWER, "candidate_id"], kind="stable").iloc[0]
        upper = min(float(anchor[MODEL_COST]), float(group[CONSTANT].min()))
        kept = group.loc[group[LOWER] <= upper + EPSILON + NUMERICAL_GUARD]
        retained.update(kept.candidate_id.astype(str))
        zone_rows.append({"zone": zone, "n": len(group), "anchor": anchor.candidate_id,
                          "incumbent_upper_eur_per_kg": upper, "retained": len(kept),
                          "pruned_fraction": 1 - len(kept) / len(group),
                          "regret_2023": float(kept[MODEL_COST].min() - group[MODEL_COST].min()),
                          "regret_2024_settlement": float(kept.realized_settlement_eur_per_kg_2024.min() - group.realized_settlement_eur_per_kg_2024.min())})
    return {"candidate_count": len(frame), "retained": len(retained), "pruned_fraction": 1 - len(retained) / len(frame), "zones": zone_rows}, retained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="diagnostic previews only; never main manuscript evidence")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("PASS: toy counterexample, anchor isolation, coverage uncertainty, 100 random bound/frontier ensembles")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"analysis_utc": datetime.now(timezone.utc).isoformat(), "status": "retrospective Land extension",
               "script_sha256": sha(Path(__file__)), "context_comparison": "exact componentwise <=; uncertain comparisons retained; no positive land tolerance", "exclusion": "FRM02: Corsica not in mainland France procurement regime",
               "epsilon_eur_per_kg": EPSILON, "energy_correction": {}, "inputs": {}, "timing_scope": "charged checkpoint replay, not a rerun benchmark"}
    energy_frames = {}
    for name, run in RUNS.items():
        path = ENERGY / run / "annual_candidate_costs_and_bounds.csv"
        original = pd.read_csv(path)
        frame = original.loc[~original.candidate_id.str.startswith("FRM")].copy()
        if len(frame) != 127:
            raise RuntimeError("Corrected evaluation roster is not 127 candidates")
        if (frame[LOWER] > frame[MODEL_COST] + NUMERICAL_GUARD).any():
            raise RuntimeError("Energy lower bound audit failed")
        energy_frames[name] = frame
        summary["energy_correction"][name], _ = corrected_energy(frame)
        summary["inputs"][name] = {"path": str(path), "sha256": sha(path)}
    # Generate the corrected derived roster once; no post-outcome replacement.
    roster_path = PROJECT / "datasets/intermediate/regret_certificate_pilot_20261002_v1/validation_sites_128_excluding_development_v1.csv"
    roster = pd.read_csv(roster_path)
    roster = roster.loc[~roster.candidate_id.str.startswith("FRM")].copy()
    roster["land_extension_exclusion_rule"] = "Exclude Corsican FRM regions; no replacement; retrospective geographic correction"
    roster.to_csv(OUT / "land_primary_sites_127.csv", index=False, lineterminator="\n")
    (OUT / "corrected_energy_summary.json").write_bytes(encode(summary))
    land_path = LAND / "land_context_metrics.csv"
    if not land_path.exists():
        print("Energy mapping corrected; land data not yet available")
        return 0
    contexts = pd.read_csv(land_path)
    contexts = contexts.loc[~contexts.candidate_id.str.startswith("FRM")].copy()
    available = set(contexts.loc[contexts.scale.eq("nuts3"), "candidate_id"])
    target = set(roster.candidate_id)
    if not args.allow_partial and not target.issubset(available):
        print(json.dumps({"energy_correction_complete": True, "land_status": "AWAITING_COVERAGE", "covered": len(available & target), "required": len(target)}))
        return 0
    results, witnesses_all, memberships, threshold_queries = [], [], [], []
    for scenario, energy in energy_frames.items():
        _, certificate_ids = corrected_energy(energy)
        pv_ids = set()
        for zone, group in energy.groupby("zone"):
            quota = sum(cid in certificate_ids for cid in group.candidate_id)
            ordered = group.sort_values(["prior_mean_pv_yield_2020_2022_mwh_per_mwp", "candidate_id"], ascending=[False, True], kind="stable")
            pv_ids.update(ordered.head(quota).candidate_id)
        for scale, context in contexts.groupby("scale"):
            merged = energy.merge(context.drop(columns=["zone"]), on="candidate_id", validate="one_to_one")
            for zone, group in merged.groupby("zone", sort=True):
                group = group.sort_values("candidate_id", kind="stable").reset_index(drop=True)
                if not group.qa_pass.all():
                    raise RuntimeError(f"Land geometry/coverage audit failed: {scale}/{zone}")
                if not args.allow_partial and len(group) != (31 if zone == "FR" else 32):
                    raise RuntimeError("Incomplete region coverage in nominal full analysis")
                lookup = group.set_index("candidate_id")
                def anchor_callback(cid):
                    row = lookup.loc[cid]
                    return float(row[MODEL_COST]), float(row.milp_wall_seconds_2023)
                for view_name, view in VIEWS.items():
                    values, lo, hi = land_intervals(group, view, True)
                    costs = group[MODEL_COST].to_numpy(float)
                    frontier = nondominated(costs, values)
                    nearfrontier = nondominated(costs, values, EPSILON)
                    keep_energy = group.candidate_id.isin(certificate_ids).to_numpy()
                    keep_pv = group.candidate_id.isin(pv_ids).to_numpy()
                    # A strong, simple land-aware comparator requires no MILP
                    # anchors: retain the frontier of LP cost and mapped context.
                    # LP values are estimates of MILP cost, not admissible upper
                    # bounds. Audit rather than assume protection of true costs.
                    lp_values = group[LOWER].to_numpy(float)
                    lp_frontier = nondominated(lp_values, values)
                    lp_nearfrontier = nondominated(lp_values, values, EPSILON)
                    selectors = [("electricity_certificate", keep_energy, [], [], 0.),
                                 ("PV_energy_equal_quota", keep_pv, [], [], 0.),
                                 ("LP_land_frontier", lp_frontier, [], [], 0.),
                                 ("LP_land_epsilon_relevant", lp_nearfrontier, [], [], 0.)]
                    for additional, label in [(False, "land_bound_cost_anchor"), (True, "land_bound_extrema_anchors")]:
                        safe_frame = group[["candidate_id", LOWER, CONSTANT]].copy()
                        keep, witnesses, anchors, seconds = bound_land_screen(safe_frame, lo, hi, anchor_callback, additional)
                        if (nearfrontier & ~keep).any():
                            raise RuntimeError("Land-aware certificate removed an epsilon-relevant option")
                        for item in witnesses:
                            i, j = lookup.loc[item["removed"]], lookup.loc[item["witness"]]
                            if not float(j[MODEL_COST]) + EPSILON < float(i[MODEL_COST]) - NUMERICAL_GUARD:
                                raise RuntimeError("Witness does not dominate actual model objective")
                            witnesses_all.append({"scenario": scenario, "scale": scale, "zone": zone, "view": view_name, "selector": label, **item})
                        selectors.append((label, keep, witnesses, anchors, seconds))
                    for label, keep, witnesses, anchors, seconds in selectors:
                        ids = group.candidate_id[keep].tolist()
                        if not ids and not args.allow_partial:
                            raise RuntimeError("A selector removed an entire zone in this cohort")
                        missing_frontier = int(np.sum(frontier & ~keep))
                        missing_near = int(np.sum(nearfrontier & ~keep))
                        results.append({
                            "scenario": scenario, "scale": scale, "zone": zone, "view": view_name, "selector": label,
                            "candidate_count": len(group), "retained": int(np.sum(keep)), "pruned_fraction": float(np.mean(~keep)),
                            "frontier_size": int(frontier.sum()), "frontier_omitted": missing_frontier,
                            "frontier_omitted_fraction": missing_frontier / int(frontier.sum()),
                            "epsilon_relevant_size": int(nearfrontier.sum()), "epsilon_relevant_omitted": missing_near,
                            "regret_2023_model": float(np.min(costs[keep]) - np.min(costs)) if keep.any() else None,
                            "regret_2024_settlement": float(group.loc[keep, "realized_settlement_eur_per_kg_2024"].min() - group.realized_settlement_eur_per_kg_2024.min()) if keep.any() else None,
                            "anchor_count": len(anchors), "anchor_ids": ";".join(anchors), "charged_anchor_and_rule_seconds": seconds,
                            "mean_clc_coverage_fraction": float(group.clc_coverage_fraction.mean()),
                            "preview_only": args.allow_partial,
                        })
                        for cid, retained, is_frontier, is_near in zip(group.candidate_id, keep, frontier, nearfrontier):
                            memberships.append({"scenario": scenario, "scale": scale, "zone": zone, "view": view_name,
                                                "selector": label, "candidate_id": cid, "retained": bool(retained),
                                                "frontier": bool(is_frontier), "epsilon_relevant": bool(is_near)})
                        if len(view) == 1:
                            for threshold in sorted(set(values[:, 0])):
                                eligible = values[:, 0] <= threshold
                                screened_eligible = eligible & keep
                                for year, column in [(2023, MODEL_COST), (2024, "realized_settlement_eur_per_kg_2024")]:
                                    reference = float(group.loc[eligible, column].min())
                                    regret = float(group.loc[screened_eligible, column].min() - reference) if screened_eligible.any() else None
                                    if label.startswith("land_bound") and year == 2023:
                                        if regret is None or regret > NUMERICAL_GUARD:
                                            raise RuntimeError("Land certificate failed a qualifying-threshold optimum audit")
                                    threshold_queries.append({
                                        "scenario": scenario, "scale": scale, "zone": zone, "view": view_name,
                                        "selector": label, "score_year": year, "minimised_context_threshold": float(threshold),
                                        "original_fraction_threshold": float(threshold * view[0][1]),
                                        "direction": "at_least" if view[0][1] == -1 else "at_most",
                                        "eligible_full": int(eligible.sum()), "eligible_shortlist": int(screened_eligible.sum()),
                                        "lost_all_eligible_options": not bool(screened_eligible.any()),
                                        "constrained_best_cost_regret_eur_per_kg": regret, "preview_only": args.allow_partial,
                                    })
    suffix = "_preview" if args.allow_partial else ""
    pd.DataFrame(results).to_csv(OUT / f"land_shortlist_comparison{suffix}.csv", index=False, lineterminator="\n")
    pd.DataFrame(memberships).to_csv(OUT / f"land_shortlist_membership{suffix}.csv", index=False, lineterminator="\n")
    pd.DataFrame(threshold_queries).to_csv(OUT / f"land_threshold_queries{suffix}.csv", index=False, lineterminator="\n")
    (OUT / f"land_dominance_witnesses{suffix}.json").write_bytes(encode(witnesses_all))
    summary["inputs"]["land"] = {"path": str(land_path), "sha256": sha(land_path)}
    summary["land_candidate_count"] = len(available & target)
    summary["preview_only"] = args.allow_partial
    summary["interpretation"] = "Regional context alternatives, not actual impacts, vacancies, biodiversity quality, or permissions. No new temporal holdout for this post-energy extension."
    (OUT / f"analysis_manifest{suffix}.json").write_bytes(encode(summary))
    print(json.dumps({"land_candidates": len(available & target), "comparison_rows": len(results), "dominance_witnesses": len(witnesses_all), "preview_only": args.allow_partial}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
