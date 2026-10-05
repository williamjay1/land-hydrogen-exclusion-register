"""Descriptive spatial diagnostics; no causal or national-population inference."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"D:\MLWork\Europe_green_hydrogen_siting")
OUT = ROOT / "results/land_planning_revision_20261005_v1"
FIG = ROOT / "final_submission_seta/figures_land_shortlist_20261005_v1"
LAND = ROOT / "datasets/intermediate/land_context_clc2018_natura2024_20261005_v1/land_context_metrics.csv"
MEMBER = ROOT / "results/land_shortlist_assessment_20261005_v1/land_shortlist_membership.csv"
ENERGY = ROOT / "results/regret_certificate_pilot_20261002_v1/run_20261002T063739Z_u50_fourweek/annual_candidate_costs_and_bounds.csv"
COST = "forecast_objective_eur_per_kg_2023"
COLS = ["industrial_commercial_fraction", "agricultural_fraction", "forest_seminatural_fraction", "natura2000_union_fraction"]
ZONES = ["BE", "DE_LU", "FR", "NL"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    land = pd.read_csv(LAND)
    land = land.loc[~land.candidate_id.str.startswith("FRM")].copy()
    whole = land.loc[land.scale.eq("nuts3")].set_index("candidate_id")
    energy = pd.read_csv(ENERGY).set_index("candidate_id").loc[whole.index]
    members = pd.read_csv(MEMBER)
    primary = members.loc[members.scenario.eq("u50_fourweek") & members.scale.eq("nuts3")
                          & members.selector.eq("electricity_certificate")]
    rows, cases, support_rows = [], [], []
    for zone in ZONES:
        local = whole.loc[whole.zone.eq(zone)]
        row = {"zone": zone, "n": len(local), "area_median_km2": local.region_or_mask_area_km2.median()}
        for col in COLS:
            for name, quantile in [("q25", .25), ("median", .5), ("q75", .75)]:
                row[col + "_" + name + "_pct"] = local[col].quantile(quantile) * 100
        rows.append(row)
        local = local.join(energy[[COST]])
        winner = local.sort_values([COST], kind="stable").iloc[0]
        for view, col, ascending in [("protected_context", COLS[3], True), ("industrial_context", COLS[0], False)]:
            alternative = local.assign(region_id=local.index).sort_values([col, COST, "region_id"], ascending=[ascending, True, True]).iloc[0]
            chosen = primary.loc[primary.zone.eq(zone) & primary.view.eq(view)]
            kept = local.loc[chosen.loc[chosen.retained, "candidate_id"]]
            boundary = kept[col].min() if ascending else kept[col].max()
            cases.append({"zone": zone, "view": view, "alternative_id": alternative.name,
                          "alternative_name": alternative.nuts3_name, "alternative_fraction_pct": alternative[col] * 100,
                          "alternative_cost_eur_per_kg": alternative[COST], "winner_id": winner.name,
                          "winner_name": winner.nuts3_name, "winner_fraction_pct": winner[col] * 100,
                          "winner_cost_eur_per_kg": winner[COST], "cost_difference_eur_per_kg": alternative[COST] - winner[COST],
                          "energy_shortlist_context_boundary_pct": boundary * 100,
                          "alternative_retained": bool(chosen.set_index("candidate_id").loc[alternative.name, "retained"]),
                          "at_alternative_threshold_all_options_lost": bool(boundary > alternative[col] + 1e-12 if ascending else boundary < alternative[col] - 1e-12)})
    for scale, radius in [("buffer_5km_clipped", 5), ("buffer_10km_clipped", 10), ("buffer_20km_clipped", 20)]:
        local = land.loc[land.scale.eq(scale)].set_index("candidate_id").loc[whole.index]
        for zone in ZONES:
            ids = whole.index[whole.zone.eq(zone)]
            clipped = local.loc[ids, "region_or_mask_area_km2"] / (np.pi * radius ** 2)
            row = {"zone": zone, "scale": scale, "n": len(ids), "median_circle_fraction_retained": clipped.median(),
                   "n_clipped_below_95pct_circle": int((clipped < .95).sum()),
                   "median_parent_region_fraction": (local.loc[ids, "region_or_mask_area_km2"] / whole.loc[ids, "region_or_mask_area_km2"]).median()}
            for col in COLS:
                row["median_abs_change_" + col + "_pp"] = (local.loc[ids, col] - whole.loc[ids, col]).abs().median() * 100
            support_rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "zone_land_context_distributions.csv", index=False)
    pd.DataFrame(cases).to_csv(OUT / "deterministic_context_extrema_cases.csv", index=False)
    pd.DataFrame(support_rows).to_csv(OUT / "spatial_support_diagnostics.csv", index=False)

    # All four mapped dimensions, raw fractions; no normalised composite score.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.5), sharex=True, sharey=True)
    x = np.arange(4)
    labels = ["Industrial /\ncommercial", "Agricultural", "Forest /\nsemi-natural", "Natura 2000"]
    for ax, zone in zip(axes.ravel(), ZONES):
        local = whole.loc[whole.zone.eq(zone)]
        selection = primary.loc[primary.zone.eq(zone) & primary.view.eq("protected_context")].set_index("candidate_id")
        for region, data in local.iterrows():
            if not selection.loc[region, "retained"]:
                ax.plot(x, data[COLS].astype(float).to_numpy() * 100, color="#a9a9a9", alpha=.45, linewidth=.7, zorder=1)
        for region in selection.index[selection.frontier & ~selection.retained]:
            ax.plot(x, local.loc[region, COLS].astype(float).to_numpy() * 100, color="#c03d3e", alpha=.85, linewidth=1.4, marker="o", markersize=3, zorder=2)
        for region in selection.index[selection.frontier & selection.retained]:
            ax.plot(x, local.loc[region, COLS].astype(float).to_numpy() * 100, color="#196f8d", alpha=.8, linewidth=1.2, linestyle="--", zorder=3)
        ax.set_title({"BE": "Belgium", "DE_LU": "Germany/Luxembourg price zone", "FR": "Mainland France", "NL": "Netherlands"}[zone], fontsize=9, loc="left")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mapped area share (%)")
    handles = [plt.Line2D([0], [0], color="#a9a9a9", lw=1, label="Removed region"),
               plt.Line2D([0], [0], color="#c03d3e", lw=1.5, marker="o", markersize=3, label="Removed protection-cost frontier"),
               plt.Line2D([0], [0], color="#196f8d", lw=1.2, linestyle="--", label="Retained protection-cost frontier")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=1, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .875), h_pad=1.9)
    stem = FIG / "figure6_excluded_land_context_profiles"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=900, bbox_inches="tight")
    plt.close(fig)
    manifest = {"scope": "Retrospective finite-cohort descriptive diagnostics; no institutional or causal inference.",
                "inputs": {str(path): sha256(path.read_bytes()).hexdigest() for path in [LAND, MEMBER, ENERGY]},
                "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "regions": len(whole), "profiles": len(cases),
                "outputs": {str(path): sha256(path.read_bytes()).hexdigest() for path in [
                    OUT / "zone_land_context_distributions.csv", OUT / "deterministic_context_extrema_cases.csv",
                    OUT / "spatial_support_diagnostics.csv", stem.with_suffix(".pdf"), stem.with_suffix(".png")]}}
    (OUT / "planning_profiles_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(cases).to_string(index=False))
    print(pd.DataFrame(support_rows).to_string(index=False))


if __name__ == "__main__":
    main()
