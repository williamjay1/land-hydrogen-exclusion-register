"""Frozen regional exclusion certificates and hypothetical dependency reopening.

Standard-library only. Read existing analysis outputs; never run the analysis,
solve a new model, edit source data, or infer a parcel/permit decision.
All generated files are confined to root/results/land_exclusion_register_20261005_v2
on drive D. Use Python -B to avoid bytecode writes when importing this module.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile


DEFAULT_ROOT = Path(r"D:\MLWork\Europe_green_hydrogen_siting")
OUTPUT_NAME = "land_exclusion_register_20261005_v2"
SOURCE_NAME = "land_shortlist_assessment_20261005_v2"
AUDIT_SOURCE_NAME = "land_shortlist_assessment_20261005_v1"
SCHEMA = "land-regional-exclusion-register/2.0.0"
LAND_SELECTORS = {"land_bound_cost_anchor", "land_bound_extrema_anchors"}
SCOPE = ("scenario", "support", "zone", "view", "selector")
MAIN = ("u50_fourweek", "nuts3", "protected_context", "land_bound_extrema_anchors")
PERTURBATION = "model-input perturbation not empirical permit outcome"
INVALIDATORS = ("preference", "spatial_support", "model_version", "data_version")
CONTEXT_TOL = 0.0  # Exact componentwise <=; positive tolerance is non-transitive.
VALUE_TOL = 1e-12  # CSV parser round-off only; not a softened strict cost test.


class RegisterError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise RegisterError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    require(str(value).lower() in {"true", "false"}, f"Invalid boolean: {value}")
    return str(value).lower() == "true"


def number(value):
    result = float(value)
    require(math.isfinite(result), f"Nonfinite numeric input: {value}")
    return result


def close(a, b, label):
    require(math.isclose(number(a), number(b), rel_tol=0, abs_tol=VALUE_TOL),
            f"Source mismatch: {label}: {a} versus {b}")


def key(row):
    return tuple(row[k] for k in SCOPE)


def source_key(row):
    return (row["scenario"], row["scale"], row["zone"], row["view"], row["selector"])


def scope_dict(scope):
    return dict(zip(SCOPE, scope))


def in_main(row):
    return tuple(row[k] for k in ("scenario", "support", "view", "selector")) == MAIN


def output_dir(root):
    # Reject redirected drives/junctions before creating anything.
    resolved = (root / "results" / OUTPUT_NAME).resolve()
    require(resolved.drive.upper() == "D:", "Output must resolve to drive D; no C/E/F writes")
    require(resolved.is_relative_to(root.resolve()), "Output escaped the requested project root")
    return resolved


def output_file(out, name):
    candidate = (out / name).resolve()
    require(candidate.is_relative_to(out.resolve()) and candidate != out.resolve(),
            "Output filename must stay inside the register output directory")
    require(candidate.drive.upper() == "D:", "Output must resolve to drive D")
    return candidate


def write_bytes(out, name, data):
    target = output_file(out, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents partially written CSV/JSON on interruption.
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".register-", delete=False) as tmp:
        tmp.write(data)
        temp_path = Path(tmp.name)
    temp_path.replace(target)


def write_json(out, name, data):
    write_bytes(out, name, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
                           + "\n").encode("utf-8"))


def write_csv(out, name, rows):
    require(bool(rows), f"Cannot infer columns for empty CSV: {name}")
    columns = list(rows[0])
    require(all(set(r) == set(columns) for r in rows), f"Inconsistent CSV schema: {name}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    write_bytes(out, name, stream.getvalue().encode("utf-8"))


def analysis_constants(path):
    """Read literal constants from AST; importing the analysis could write pycache."""
    wanted = {"RUNS", "VIEWS", "LOWER", "CONSTANT", "MODEL_COST", "EPSILON", "NUMERICAL_GUARD"}
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = ast.literal_eval(node.value)
    require(set(values) == wanted, "Analysis constants are missing or no longer literal")
    return values


def relocate(path_text, root):
    """Rebase legacy absolute manifest paths onto any supplied project root."""
    original = Path(path_text)
    if original.is_absolute():
        require(original.is_relative_to(DEFAULT_ROOT),
                f"Input is outside the original project; relocation is ambiguous: {original}")
        return root / original.relative_to(DEFAULT_ROOT)
    return root / original


def intervals(raw, view):
    missing = number(raw["clc_unknown_fraction"])
    signed, original = {}, {}
    for column, direction in view:
        require(direction in {-1, 1}, "Unsupported context direction")
        value = number(raw[column])
        lo = hi = value * direction
        if column != "natura2000_union_fraction":
            if direction == 1:
                hi = min(1.0, hi + missing)
            else:
                lo = max(-1.0, lo - missing)
        signed[column] = {"lower": lo, "upper": hi}
        original[column] = {"lower": lo if direction == 1 else -hi,
                            "upper": hi if direction == 1 else -lo}
    return {"original_fraction": original, "minimised_signed": signed,
            "clc_unknown_fraction": missing,
            "coverage_rule": "CLC unknown area widens active CLC fractions; Natura union is point-valued",
            "uncertainty_scope": "coverage allocation only; not positional, temporal, legal, or statistical uncertainty"}


def no_worse(witness_interval, removed_interval):
    return all(witness_interval["minimised_signed"][c]["upper"]
               <= removed_interval["minimised_signed"][c]["lower"]
               for c in witness_interval["minimised_signed"])


def trace(start, edges, retained):
    chain = [start]
    while chain[-1] in edges:
        next_id = edges[chain[-1]]["witness"]
        require(next_id not in chain, f"Witness cycle: {chain + [next_id]}")
        chain.append(next_id)
    require(chain[-1] in retained, f"Chain has no retained terminal: {chain}")
    return chain


def load_sources(root):
    analysis_path = root / "scripts/project03_analyse_land_shortlists.py"
    src = root / "results" / SOURCE_NAME
    manifest_path = src / "analysis_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("preview_only") is False, "Diagnostic preview is not admissible input")
    require(manifest.get("context_comparison") == "exact componentwise <=; uncertain comparisons retained; no positive land tolerance",
            "Only the strict v2 analysis is admissible for certificate generation")
    config = analysis_constants(analysis_path)
    require(sha(analysis_path) == manifest["script_sha256"], "Analysis code version drift; recompute certificates")
    close(manifest["epsilon_eur_per_kg"], config["EPSILON"], "epsilon")
    paths = {"analysis_script": analysis_path, "analysis_manifest": manifest_path,
             "witnesses": src / "land_dominance_witnesses.json",
             "membership": src / "land_shortlist_membership.csv"}
    paths["audit_only_v1_witnesses"] = root / "results" / AUDIT_SOURCE_NAME / "land_dominance_witnesses.json"
    paths["audit_only_v1_manifest"] = root / "results" / AUDIT_SOURCE_NAME / "analysis_manifest.json"
    paths["land"] = relocate(manifest["inputs"]["land"]["path"], root)
    for scenario, run in config["RUNS"].items():
        path = root / "results/regret_certificate_pilot_20261002_v1" / run / "annual_candidate_costs_and_bounds.csv"
        require(path.resolve() == relocate(manifest["inputs"][scenario]["path"], root).resolve(),
                f"RUNS and manifest disagree for {scenario}")
        paths[scenario] = path
    sources = {name: {"relative_path": str(p.relative_to(root)), "sha256": sha(p),
                      "bytes": p.stat().st_size} for name, p in paths.items()}
    for name in ["land", *config["RUNS"]]:
        require(sources[name]["sha256"] == manifest["inputs"][name]["sha256"],
                f"Input data version drift: {name}; recompute the whole certificate set")
    witnesses = read_json(paths["witnesses"])
    require(len(witnesses) == 5631, "Expected all 5631 strict v2 witnesses; do not silently change the cohort")
    memberships = read_csv(paths["membership"])
    land_rows = read_csv(paths["land"])
    fraction_fields = [c for c in land_rows[0] if c.endswith("_fraction")]
    land = {}
    for row in land_rows:
        lk = (row["scale"], row["candidate_id"])
        require(lk not in land, f"Duplicate land context: {lk}")
        land[lk] = row
    energy = {}
    for scenario in config["RUNS"]:
        rows = [r for r in read_csv(paths[scenario]) if not r["candidate_id"].startswith("FRM")]
        energy[scenario] = {r["candidate_id"]: r for r in rows}
        require(len(rows) == len(energy[scenario]) == 127, f"Energy cohort must contain 127 distinct regions: {scenario}")
        for row in rows:
            lower, model, upper = (number(row[config[c]]) for c in ("LOWER", "MODEL_COST", "CONSTANT"))
            require(lower <= model + config["NUMERICAL_GUARD"] and model <= upper + config["NUMERICAL_GUARD"],
                    f"Cost bounds invalid: {scenario}/{row['candidate_id']}")
    binding = {"schema_version": SCHEMA, "source_sha256": {k: v["sha256"] for k, v in sources.items()},
               "context_views": config["VIEWS"], "epsilon_eur_per_kg": config["EPSILON"],
               "strict_cost_guard_eur_per_kg": config["NUMERICAL_GUARD"],
               "context_tolerance": CONTEXT_TOL, "excluded_roster_rule": manifest["exclusion"]}
    return config, manifest, sources, binding, witnesses, memberships, land, energy, fraction_fields


def make_register(root):
    config, source_manifest, sources, binding, witnesses, memberships, land, energy, fractions = load_sources(root)
    set_id = digest(binding)
    groups = defaultdict(dict)
    for m in memberships:
        if m["selector"] in LAND_SELECTORS:
            g = source_key(m)
            require(m["candidate_id"] not in groups[g], f"Duplicate membership: {g}/{m['candidate_id']}")
            groups[g][m["candidate_id"]] = boolean(m["retained"])
    require(len(groups) == 640, "Expected all 640 zone-specific certificate groups, including groups without exclusions")
    edges = defaultdict(dict)
    for w in witnesses:
        g = source_key(w)
        require(g in groups and w["removed"] not in edges[g], f"Unexpected/duplicate exclusion: {g}/{w['removed']}")
        edges[g][w["removed"]] = w
    reconstructed = {}
    for g, members in groups.items():
        scenario, support, zone, view_name, selector = g
        view = config["VIEWS"][view_name]
        require(len(members) == (31 if zone == "FR" else 32), f"Incomplete zone group: {g}")
        erows = energy[scenario]
        require(set(members) == {cid for cid, r in erows.items() if r["zone"] == zone}, f"Cohort mismatch: {g}")
        ir = {}
        for cid in members:
            raw = land[(support, cid)]
            require(raw["zone"] == zone and boolean(raw["qa_pass"]), f"Land zone/QA failure: {g}/{cid}")
            for f in fractions:
                # Pre-union diagnostic sums can exceed one because polygons overlap.
                if f == "natura2000_sum_before_union_fraction":
                    require(number(raw[f]) >= 0, f"Negative pre-union diagnostic: {cid}/{f}")
                else:
                    require(-VALUE_TOL <= number(raw[f]) <= 1 + VALUE_TOL, f"Fraction out of bounds: {cid}/{f}")
            close(number(raw["clc_coverage_fraction"]) + number(raw["clc_unknown_fraction"]), 1, f"CLC coverage: {cid}")
            ir[cid] = intervals(raw, view)
        lower = {cid: number(erows[cid][config["LOWER"]]) for cid in members}
        upper = {cid: number(erows[cid][config["CONSTANT"]]) for cid in members}
        anchors = [min(members, key=lambda cid: (lower[cid], cid))]
        if selector == "land_bound_extrema_anchors":
            anchors += [min(members, key=lambda cid: (ir[cid]["minimised_signed"][c]["upper"], lower[cid], cid))
                        for c, _ in view]
        anchors = list(dict.fromkeys(anchors))
        for cid in anchors:
            upper[cid] = min(upper[cid], number(erows[cid][config["MODEL_COST"]]))
        # Reconstruct the original pairwise rule without selecting replacement witnesses.
        for cid, kept in members.items():
            dominators = [other for other in members
                          if upper[other] + config["EPSILON"] < lower[cid] - config["NUMERICAL_GUARD"]
                          and no_worse(ir[other], ir[cid])]
            require(kept == (not dominators), f"Membership does not reproduce the frozen rule: {g}/{cid}")
            if not kept:
                require(cid in edges[g], f"Historical exclusion has no witness: {g}/{cid}")
                expected = min(dominators, key=lambda other: (upper[other], other))
                require(edges[g][cid]["witness"] == expected, f"Recorded witness does not match the frozen selector: {g}/{cid}")
        require(set(edges[g]) == {cid for cid, kept in members.items() if not kept}, f"Exclusion/membership mismatch: {g}")
        reconstructed[g] = (ir, lower, upper, anchors)
    register = []
    registry_memberships = []
    for g, members in sorted(groups.items()):
        scenario, support, zone, view_name, selector = g
        view = config["VIEWS"][view_name]
        directions = {c: {"sign": d, "preference": "at_most" if d == 1 else "at_least"}
                      for c, d in view}
        retained = {cid for cid, kept in members.items() if kept}
        ir, lower, upper, anchors = reconstructed[g]
        group_id = digest({"set_id": set_id, "scope": scope_dict(g)})
        for cid, kept in sorted(members.items()):
            registry_memberships.append({"schema_version": SCHEMA, "certificate_set_id": set_id,
                                         "certificate_group_id": group_id, **scope_dict(g),
                                         "candidate_id": cid, "retained": kept,
                                         "initial_status": "RETAINED" if kept else "EXCLUDED",
                                         "modeled_candidate_status": "frozen_roster_member",
                                         "parcel_id": "unknown", "decision_owner": "unknown",
                                         "permit_status": "not_evaluated"})
        for removed, w in sorted(edges[g].items()):
            witness = w["witness"]
            require(witness in members and witness != removed, f"Missing/self witness: {g}/{removed}")
            close(w["lower_removed"], lower[removed], "lower_removed")
            close(w["upper_witness"], upper[witness], "upper_witness")
            close(w["cost_margin_eur_per_kg"], lower[removed] - upper[witness], "cost margin")
            require(len(w["witness_land_upper"]) == len(w["removed_land_lower"]) == len(view), "Context vector dimension mismatch")
            for index, (column, _) in enumerate(view):
                close(w["witness_land_upper"][index], ir[witness]["minimised_signed"][column]["upper"], column)
                close(w["removed_land_lower"][index], ir[removed]["minimised_signed"][column]["lower"], column)
                require(number(w["witness_land_upper"][index]) <= number(w["removed_land_lower"][index]),
                        f"Recorded strict interval violation: {g}/{removed}/{column}")
            chain = trace(removed, edges[g], retained)
            terminal = chain[-1]
            margin = number(w["lower_removed"]) - number(w["upper_witness"]) - config["EPSILON"] - config["NUMERICAL_GUARD"]
            terminal_margin = lower[removed] - upper[terminal] - config["EPSILON"] - config["NUMERICAL_GUARD"]
            require(margin > 0 and terminal_margin > 0, f"Non-strict cost certificate: {g}/{removed}")
            require(no_worse(ir[witness], ir[removed]) and no_worse(ir[terminal], ir[removed]), f"Interval dominance failure: {g}/{removed}")
            model_margin = (number(energy[scenario][removed][config["MODEL_COST"]])
                            - number(energy[scenario][witness][config["MODEL_COST"]])
                            - config["EPSILON"] - config["NUMERICAL_GUARD"])
            require(model_margin > 0, f"Modeled objective improvement failure: {g}/{removed}")
            cert_id = digest({"group_id": group_id, "removed": removed, "witness": witness})
            record = {"schema_version": SCHEMA, "certificate_set_id": set_id,
                      "certificate_group_id": group_id, "certificate_id": cert_id,
                      **scope_dict(g), "removed": removed, "witness": witness,
                      "final_retained_region": terminal, "chain_region_ids": chain,
                      "chain_hops": len(chain) - 1, "dependency_region_ids": chain[1:],
                      "context_directions": directions,
                      "removed_coverage_interval": ir[removed], "witness_coverage_interval": ir[witness],
                      "final_retained_coverage_interval": ir[terminal],
                      "removed_original_context_fractions": {f: land[(support, removed)][f] for f in fractions},
                      "witness_original_context_fractions": {f: land[(support, witness)][f] for f in fractions},
                      "final_retained_original_context_fractions": {f: land[(support, terminal)][f] for f in fractions},
                      "witness_land_upper": w["witness_land_upper"], "removed_land_lower": w["removed_land_lower"],
                      "lower_removed": number(w["lower_removed"]), "upper_witness": number(w["upper_witness"]),
                      "epsilon_eur_per_kg": config["EPSILON"],
                      "numerical_guard_eur_per_kg": config["NUMERICAL_GUARD"], "context_tolerance": CONTEXT_TOL,
                      "cost_margin_eur_per_kg": number(w["cost_margin_eur_per_kg"]),
                      "strict_margin_eur_per_kg": margin, "upper_final_retained": upper[terminal],
                      "terminal_strict_margin_eur_per_kg": terminal_margin,
                      "model_objective_strict_margin_eur_per_kg": model_margin,
                      "witness_upper_basis": "min(constant_policy,anchor_model_objective)" if witness in anchors else "constant_policy",
                      "anchor_ids": anchors, "same_zone": True, "componentwise_interval_pass": True,
                      "chain_acyclic": True, "terminal_retained": True,
                      "historical_status": "EXCLUDED", "certificate_status": "VALID_FROZEN_REGIONAL_COMPARISON",
                      "parcel_id_removed": "unknown", "parcel_id_witness": "unknown", "parcel_id_final_retained": "unknown",
                      "decision_owner_removed": "unknown", "decision_owner_witness": "unknown", "decision_owner_final_retained": "unknown",
                      "permit_status_removed": "not_evaluated", "permit_status_witness": "not_evaluated", "permit_status_final_retained": "not_evaluated",
                      "legal_substitutability": "not_evaluated", "parcel_substitutability": "not_evaluated",
                      "certificate_scope": "regional_comparison_only",
                      "witness_source_sha256": sources["witnesses"]["sha256"],
                      "membership_source_sha256": sources["membership"]["sha256"],
                      "analysis_manifest_sha256": sources["analysis_manifest"]["sha256"],
                      "analysis_script_sha256": sources["analysis_script"]["sha256"],
                      "land_source_sha256": sources["land"]["sha256"],
                      "energy_source_sha256": sources[scenario]["sha256"]}
            # Flat CSV columns preserve the original CSV decimal strings verbatim.
            for role, cid in (("removed", removed), ("witness", witness), ("final_retained", terminal)):
                for fraction in fractions:
                    record[f"{role}_{fraction}"] = land[(support, cid)][fraction]
            register.append(record)
    return config, source_manifest, sources, binding, register, registry_memberships, energy, land, fractions


def transition(records, members, states=None, invalidations=()):
    """Immutable historical records; no search for a new witness or recertification."""
    states = states or {}
    kept = {r["candidate_id"] for r in members if boolean(r["retained"])}
    require(set(states) <= kept, "Perturbation target must be retained in the selected frozen scope")
    require(all(v in {"revoked", "unknown"} for v in states.values()), "Only revoked/unknown perturbations are supported")
    require(set(invalidations) <= set(INVALIDATORS), "Unknown whole-certificate invalidation reason")
    events, exclusions, affected = [], [], set()
    for r in records:
        failed = [cid for cid in r["dependency_region_ids"] if cid in states]
        status = "INVALID_RECOMPUTE" if invalidations else "REOPEN" if failed else "EXCLUDED"
        if status != "EXCLUDED":
            affected.add(r["removed"])
            events.append({"certificate_id": r["certificate_id"], "removed": r["removed"],
                           "from_status": "EXCLUDED", "to_status": status,
                           "failed_dependencies": {cid: states[cid] for cid in failed},
                           "invalidation_reasons": list(invalidations), "evidence_class": PERTURBATION})
        exclusions.append({**r, "effective_status": status,
                           "certificate_status": "INVALID_RECOMPUTE" if invalidations else
                           "SUSPENDED_DEPENDENCY_REVIEW_REQUIRED" if failed else r["certificate_status"],
                           "failed_dependencies": {cid: states[cid] for cid in failed},
                           "replacement_witness": "not_selected", "invalidation_reasons": list(invalidations)})
    roster = []
    for r in members:
        cid = r["candidate_id"]
        status = ("NOT_EVALUATED_RECOMPUTE" if invalidations else "WITNESS_" + states[cid].upper() if cid in states
                  else "REOPEN" if cid in affected else r["initial_status"])
        roster.append({**r, "retained": boolean(r["retained"]), "effective_status": status})
    return {"schema_version": SCHEMA, "evidence_class": PERTURBATION,
            "empirical_permit_outcome": False, "replacement_witness_selected": False,
            "witness_states": states, "invalidation_reasons": list(invalidations),
            "reopen_count": sum(r["effective_status"] == "REOPEN" for r in exclusions),
            "invalid_recompute_count": sum(r["effective_status"] == "INVALID_RECOMPUTE" for r in exclusions),
            "unaffected_exclusion_count": sum(r["effective_status"] == "EXCLUDED" for r in exclusions),
            "roster": roster, "exclusions": exclusions, "events": events}


def grouped(records, members):
    rr, mm = defaultdict(list), defaultdict(list)
    for r in records:
        rr[key(r)].append(r)
    for r in members:
        mm[key(r)].append(r)
    return rr, mm


def dependency_audit(records, members):
    rr, mm = grouped(records, members)
    audit_rows, dependency_rows = [], []
    for g, roster in sorted(mm.items()):
        group_records = rr[g]
        for r in group_records:
            for position, cid in enumerate(r["dependency_region_ids"], 1):
                dependency_rows.append({"schema_version": SCHEMA, **scope_dict(g),
                                        "certificate_id": r["certificate_id"], "removed": r["removed"],
                                        "dependency_region": cid, "chain_position": position,
                                        "is_final_retained": cid == r["final_retained_region"]})
        for member in sorted(roster, key=lambda r: r["candidate_id"]):
            if not boolean(member["retained"]):
                continue
            cid = member["candidate_id"]
            expected = sorted(r["removed"] for r in group_records if cid in r["dependency_region_ids"])
            revoked = transition(group_records, roster, {cid: "revoked"})
            unknown = transition(group_records, roster, {cid: "unknown"})
            for result in (revoked, unknown):
                actual = sorted(r["removed"] for r in result["exclusions"] if r["effective_status"] == "REOPEN")
                require(actual == expected, f"Reopen dependency closure failure: {g}/{cid}")
                require(result["unaffected_exclusion_count"] + result["reopen_count"] == len(group_records), "Unrelated exclusions changed")
                require(not result["replacement_witness_selected"], "Reopen selected an unverified substitute")
            audit_rows.append({"schema_version": SCHEMA, **scope_dict(g), "retained_region": cid,
                               "initial_group_retained": sum(boolean(r["retained"]) for r in roster),
                               "initial_group_excluded": len(group_records), "reopen_if_revoked": len(expected),
                               "reopen_if_unknown": len(expected), "affected_removed_ids": expected,
                               "unchanged_exclusions": len(group_records) - len(expected),
                               "new_witness_selected": False, "dependency_closure_pass": True,
                               "evidence_class": PERTURBATION})
    require(sum(r["reopen_if_revoked"] for r in audit_rows) == len(records), "Each chain must have exactly one retained terminal dependency")
    return audit_rows, dependency_rows


def energy_register(config, source_manifest, sources, energy, land, fractions, set_id):
    scenario = MAIN[0]
    erows = energy[scenario]
    zone_metadata = {z["zone"]: z for z in source_manifest["energy_correction"][scenario]["zones"]}
    rows = []
    for cid, r in sorted(erows.items()):
        zone = r["zone"]
        z = zone_metadata[zone]
        zone_candidates = {c: e for c, e in erows.items() if e["zone"] == zone}
        anchor = min(zone_candidates, key=lambda c: (number(zone_candidates[c][config["LOWER"]]), c))
        policy_owner = min(zone_candidates, key=lambda c: (number(zone_candidates[c][config["CONSTANT"]]), c))
        anchor_upper = number(zone_candidates[anchor][config["MODEL_COST"]])
        policy_upper = number(zone_candidates[policy_owner][config["CONSTANT"]])
        owner = anchor if anchor_upper <= policy_upper else policy_owner
        incumbent = min(anchor_upper, policy_upper)
        close(z["incumbent_upper_eur_per_kg"], incumbent, "energy incumbent")
        require(z["anchor"] == anchor, "Energy anchor mismatch")
        lower = number(r[config["LOWER"]])
        margin = lower - incumbent - config["EPSILON"] - config["NUMERICAL_GUARD"]
        keep = lower <= incumbent + config["EPSILON"] + config["NUMERICAL_GUARD"]
        raw = land[("nuts3", cid)]
        rows.append({"schema_version": SCHEMA, "certificate_set_id": set_id,
                     "scenario": scenario, "support": "nuts3", "zone": zone,
                     "view": "energy_only", "selector": "electricity_certificate", "candidate_id": cid,
                     "initial_status": "RETAINED" if keep else "REMOVE", "retained": keep,
                     "lower_candidate": lower, "zone_incumbent_upper": incumbent,
                     "energy_incumbent_region": owner, "cost_anchor_region": anchor,
                     "energy_upper_basis": "anchor_model_objective" if owner == anchor and anchor_upper <= policy_upper else "constant_policy",
                     "epsilon_eur_per_kg": config["EPSILON"], "numerical_guard_eur_per_kg": config["NUMERICAL_GUARD"],
                     "strict_margin_eur_per_kg": margin,
                     "context_status": "unprotected", "context_witness": "not_evaluated",
                     "context_certificate": "not_evaluated", "legal_substitutability": "not_evaluated",
                     "parcel_id": "unknown", "decision_owner": "unknown", "permit_status": "not_evaluated",
                     "certificate_scope": "electricity_objective_only", "land_source_sha256": sources["land"]["sha256"],
                     "energy_source_sha256": sources[scenario]["sha256"],
                     **{f: raw[f] for f in fractions}})
    require(len(rows) == 127 and sum(not r["retained"] for r in rows) == 79, "Main energy-only register must contain 127 rows / 79 removes")
    for zone, z in zone_metadata.items():
        require(sum(r["retained"] for r in rows if r["zone"] == zone) == z["retained"], "Energy membership mismatch")
    return rows


def audit_legacy_v1(root, config, sources, records, members):
    """Audit obsolete source only; never certify a rejected or superseded v1 record."""
    legacy_path = root / "results" / AUDIT_SOURCE_NAME / "land_dominance_witnesses.json"
    legacy = read_json(legacy_path)
    require(len(legacy) == 5635, "Audit-only v1 must preserve all 5635 historical records")
    canonical_pairs = {(key(r), r["removed"], r["witness"]) for r in records}
    roster = {(key(r), r["candidate_id"]): boolean(r["retained"]) for r in members}
    rejected = []
    valid_old_pairs = set()
    for w in legacy:
        violations = []
        for index, (column, direction) in enumerate(config["VIEWS"][w["view"]]):
            upper, lower = number(w["witness_land_upper"][index]), number(w["removed_land_lower"][index])
            if upper > lower:
                violations.append({"column": column, "sign": direction,
                                   "witness_signed_upper": upper, "removed_signed_lower": lower,
                                   "strict_violation": upper - lower})
        pair = (source_key(w), w["removed"], w["witness"])
        if violations:
            require(pair not in canonical_pairs, "Rejected v1 witness was still certified in v2")
            require(roster[(source_key(w), w["removed"])], "Rejected v1 region was not retained in strict v2")
            rejected.append({"schema_version": SCHEMA, **scope_dict(source_key(w)),
                             "removed": w["removed"], "witness": w["witness"],
                             "audit_status": "REJECTED_NONCERTIFIABLE",
                             "canonical_v2_removed_region_status": "RETAINED",
                             "strict_componentwise_violations": violations,
                             "maximum_strict_violation": max(v["strict_violation"] for v in violations),
                             "context_tolerance": 0.0, "witness_source_sha256": sources["audit_only_v1_witnesses"]["sha256"],
                             "manifest_source_sha256": sources["audit_only_v1_manifest"]["sha256"],
                             "source_role": "audit_only; never valid certificate input"})
        else:
            valid_old_pairs.add(pair)
    require(len(rejected) == 4 and valid_old_pairs == canonical_pairs,
            "v1/v2 audit must isolate exactly four rejected comparisons and 5631 unchanged witness pairs")
    return rejected, {"schema_version": SCHEMA, "source_role": "obsolete v1 audit only",
                       "v1_historical_rows": len(legacy), "strict_v2_certified_rows": len(records),
                       "rejected_v1_rows": len(rejected), "unchanged_witness_pairs": len(valid_old_pairs),
                       "rejected_never_marked_valid": True, "main_nuts3_protected_scope_affected": False,
                       "records": rejected}


def main_summary(records, members, audit):
    main_records = [r for r in records if in_main(r)]
    main_members = [r for r in members if in_main(r)]
    main_audit = [r for r in audit if in_main(r)]
    counts = Counter(r["final_retained_region"] for r in main_records)
    require(len(main_records) == 38 and sum(boolean(r["retained"]) for r in main_members) == 89, "Main scope is not 38 excluded / 89 retained")
    worst = min(main_audit, key=lambda r: (-r["reopen_if_revoked"], r["retained_region"]))
    terminal = worst["retained_region"]
    example = min((r for r in main_records if r["final_retained_region"] == terminal), key=lambda r: r["removed"])
    return {"scenario": MAIN[0], "support": MAIN[1], "view": MAIN[2], "selector": MAIN[3],
            "zone": "ALL (BE, DE_LU, FR, NL)", "initial_candidates": len(main_members),
            "initial_excluded": len(main_records), "initial_retained": 89,
            "distinct_retained_terminals_used": len(counts), "terminal_dependency_counts": dict(sorted(counts.items())),
            "maximum_single_terminal_reopen_count": worst["reopen_if_revoked"], "maximum_impact_terminal": terminal,
            "maximum_impact_zone": worst["zone"], "affected_removed_ids": worst["affected_removed_ids"],
            "example_chain": example["chain_region_ids"], "example_certificate_id": example["certificate_id"],
            "example_strict_margin_eur_per_kg": example["strict_margin_eur_per_kg"],
            "hypothetical_removed_terminal": terminal, "evidence_class": PERTURBATION,
            "empirical_permit_outcome": False}


def mechanism_self_test():
    # Synthetic topology tests are software checks, never empirical project evidence.
    edges = {"A": {"witness": "B"}, "B": {"witness": "C"}}
    require(trace("A", edges, {"C"}) == ["A", "B", "C"], "Multihop trace failed")
    for bad, retained in (({"A": {"witness": "B"}, "B": {"witness": "A"}}, set()),
                          ({"A": {"witness": "Z"}}, {"C"})):
        try:
            trace("A", bad, retained)
        except RegisterError:
            pass
        else:
            raise RegisterError("Cycle/nonretained-terminal rejection failed")
    records = [{"removed": "A", "witness": "B", "final_retained_region": "C", "dependency_region_ids": ["B", "C"],
                "certificate_id": "toy-A", "certificate_status": "VALID_FROZEN_REGIONAL_COMPARISON"},
               {"removed": "B", "witness": "C", "final_retained_region": "C", "dependency_region_ids": ["C"],
                "certificate_id": "toy-B", "certificate_status": "VALID_FROZEN_REGIONAL_COMPARISON"},
               {"removed": "D", "witness": "E", "final_retained_region": "E", "dependency_region_ids": ["E"],
                "certificate_id": "toy-D", "certificate_status": "VALID_FROZEN_REGIONAL_COMPARISON"}]
    members = [{"candidate_id": cid, "retained": cid in {"C", "E"}, "initial_status": "RETAINED" if cid in {"C", "E"} else "EXCLUDED"}
               for cid in "ABCDE"]
    before = canonical((records, members))
    for state in ("revoked", "unknown"):
        result = transition(records, members, {"C": state})
        require(result["reopen_count"] == 2 and result["unaffected_exclusion_count"] == 1, "Multihop dependency reopening failed")
    for reason in INVALIDATORS:
        result = transition(records, members, invalidations=[reason])
        require(result["invalid_recompute_count"] == 3, "Whole-set invalidation failed")
    try:
        transition(records, members, {"A": "revoked"})
    except RegisterError:
        pass
    else:
        raise RegisterError("Nonretained perturbation target accepted")
    require(canonical((records, members)) == before, "Transition mutated historical records")
    require(not no_worse(intervals({"clc_unknown_fraction": ".15", "x": ".15"}, [("x", 1)]),
                         intervals({"clc_unknown_fraction": "0", "x": ".2"}, [("x", 1)])), "Coverage uncertainty was ignored")
    industrial = intervals({"clc_unknown_fraction": ".1", "x": ".2"}, [("x", -1)])
    close(industrial["original_fraction"]["x"]["upper"], .3, "At-least context interval")
    # Positive tolerance breaks transitivity: B<=A+t and C<=B+t, but C>A+t.
    tiny_context = [0.0, .75e-12, 1.5e-12]
    tiny_cost = [3.0, 2.0, 1.0]
    tiny_intervals = [intervals({"clc_unknown_fraction": "0", "x": str(x)}, [("x", 1)]) for x in tiny_context]
    strict_keep = [not any(tiny_cost[j] + .05 < tiny_cost[i] - 1e-7
                           and no_worse(tiny_intervals[j], tiny_intervals[i]) for j in range(3)) for i in range(3)]
    tolerant_keep = [not any(tiny_cost[j] + .05 < tiny_cost[i] - 1e-7
                             and tiny_context[j] <= tiny_context[i] + 1e-12 for j in range(3)) for i in range(3)]
    require(strict_keep == [True, True, True], "Tiny-context strict counterexample must retain all three points")
    require(tolerant_keep == [False, False, True]
            and tiny_context[1] <= tiny_context[0] + 1e-12
            and tiny_context[2] <= tiny_context[1] + 1e-12
            and not tiny_context[2] <= tiny_context[0] + 1e-12,
            "Nontransitive positive-tolerance counterexample failed")
    return {"multihop_reopen": "PASS", "cycle_rejection": "PASS", "nonretained_terminal_rejection": "PASS",
            "unknown_and_revoked_equivalence": "PASS", "whole_set_invalidators": list(INVALIDATORS),
            "history_immutable": "PASS", "coverage_direction_checks": "PASS", "evidence_scope": "synthetic software tests",
            "tiny_context_nontransitivity_test": {"status": "PASS", "context": tiny_context,
                                                  "cost": tiny_cost, "strict_retained": strict_keep,
                                                  "obsolete_positive_tolerance_retained": tolerant_keep}}


def docs(out, summary, fractions, columns):
    text = f"""# Regional exclusion register, schema {SCHEMA}

This is a runnable supplement to the retrospective Land regional comparison.
It certifies all 5,631 strict v2 context witnesses. Four obsolete v1 comparisons are
rejected separately and never marked valid. These are not new siting or permit decisions.
Certificates apply only within the frozen scenario, spatial support, zone, preference view,
selector, epsilon, model and source data versions. Regional comparison does not establish
legal or parcel substitutability, land availability, permission, biodiversity quality or
actual project impacts. Parcel IDs and decision owners are unknown; permissions are not_evaluated.

## Files

- exclusion_register.csv / exclusion_register.json: 5,631 strict v2 exclusion certificates.
- rejected_v1_certificates.csv / legacy_v1_strict_audit.json: four noncertifiable v1
  FRL02 -> FRL03 comparisons, buffer_10km_clipped / joint_context / extrema anchors,
  with exact signed interval violations of 1.710293351419393e-15. Audit only.
- land_certificate_membership.csv: 20,320 regional memberships in 640 zone-specific scopes.
- energy_only_initial_register.csv: the main u50_fourweek electricity screen, all 127 rows,
  including 79 REMOVE and 48 RETAINED. Every context status is unprotected. Its energy
  incumbent is not a context witness; no context-dominance certificate is supplied.
- dependency_index.csv: historical certificate dependency edges, including chain positions.
- leave_one_retained_witness_out.csv: every retained member in every scope, including
  zero-dependency members, with both revoked and unknown perturbation results.
- main_leave_one_retained_witness_out.csv: all 89 retained members in the main scope.
- main_exclusion_dependency_impacts.csv: all 38 main-scope removals, their terminal
  dependencies and each terminal's single-loss reopening impact.
- validation_report.json, main_scenario_summary.json and manifest.json: exact validation,
  dependency statistics, source/output hashes and version binding.
- example_revoked_terminal.json / example_unknown_terminal.json: full main-scope roster,
  unchanged historical chains, effective exclusions and event logs for hypothetical {summary['maximum_impact_terminal']} loss.
- example_preference_invalidation.json: whole-scope invalidation requiring recomputation.

## Run (PowerShell)

```powershell
python -B D:\\MLWork\\Europe_green_hydrogen_siting\\scripts\\project03_build_exclusion_register.py build
python -B D:\\MLWork\\Europe_green_hydrogen_siting\\scripts\\project03_build_exclusion_register.py validate
python -B D:\\MLWork\\Europe_green_hydrogen_siting\\scripts\\project03_build_exclusion_register.py query --scenario u50_fourweek --support nuts3 --zone ALL --view protected_context --selector land_bound_extrema_anchors
python -B D:\\MLWork\\Europe_green_hydrogen_siting\\scripts\\project03_build_exclusion_register.py query --scenario u50_fourweek --support nuts3 --zone ALL --view protected_context --selector land_bound_extrema_anchors --revoke-witness {summary['maximum_impact_terminal']} --output query_revoked.json
python -B D:\\MLWork\\Europe_green_hydrogen_siting\\scripts\\project03_build_exclusion_register.py query --unknown-witness {summary['maximum_impact_terminal']} --output query_unknown.json
python -B D:\\MLWork\\Europe_green_hydrogen_siting\\scripts\\project03_build_exclusion_register.py query --invalidate preference --output query_invalid.json
```

`--root D:\\path\\to\\project` works before or after the command. The relocated project must
contain the same analysis script, input results and intermediate land CSV. Absolute legacy
input paths are rebased from the default project root. Output stays in
`<root>/results/{OUTPUT_NAME}` and must resolve to drive D; C/E/F output is rejected.
`--output` is relative to this output directory and cannot escape it. Query without it is read-only.
Only the requested script and output package need to be copied; input files are read-only.
No third-party package, analysis import, optimisation solve, network request or p-value is needed.

## Reopening semantics

`--revoke-witness` and `--unknown-witness` may be repeated or combined for different retained
members. Every historical exclusion whose recorded chain depends on a failed member changes
its effective status from EXCLUDED to REOPEN. The original record and chain remain unchanged.
No replacement witness is searched, selected or certified. A retained candidate with no
dependents changes its modeled status but reopens zero exclusions. These events are
**{PERTURBATION}**. They do not claim that any real project or permit was revoked.

REOPEN means reassess the previously excluded regional option; it does not mean approved,
feasible, newly retained or recertified. There is no automatic transition back to EXCLUDED:
new verified inputs require a separate recomputation/version of the analysis and register.
Query snapshots are hypothetical overlays, not persisted changes to the base register.
Conflicting states and perturbations of nonretained members are rejected.

Changing preference/direction/epsilon, spatial support, model version or data version invalidates
the entire certificate set and requires recomputation, including all region memberships.
Use `--invalidate preference|spatial_support|model_version|data_version` to demonstrate this
in the selected scope: all exclusions become INVALID_RECOMPUTE and all roster assessments
become NOT_EVALUATED_RECOMPUTE. Unqueried scopes bound to changed inputs are also invalid;
a query snapshot does not revalidate them. Source hash drift blocks query/validate entirely.

## Main frozen scope and perturbation

u50_fourweek / nuts3 / all zones / protected_context / land_bound_extrema_anchors:
38 EXCLUDED, 89 RETAINED, {summary['distinct_retained_terminals_used']} distinct retained terminals used.
Terminal dependency counts: {canonical(summary['terminal_dependency_counts'])}.
Maximum single-terminal impact: {summary['maximum_impact_terminal']} in {summary['maximum_impact_zone']},
{summary['maximum_single_terminal_reopen_count']} exclusions REOPEN under either revoked or unknown.
Example chain: {' -> '.join(summary['example_chain'])}.
All observed chains have one edge; the software also checks synthetic multi-edge chains and cycles.

The CLC intervals propagate unknown mapped area only. The Natura fraction is point-valued
in the existing analysis. Neither is a statistical confidence interval or a full uncertainty
assessment of map precision, time, legality or ecological impact. Directions are explicit,
not monetised; componentwise interval dominance uses signed minimised coordinates.
Strict margin = lower_removed - upper_witness - epsilon - numerical_guard and must be > 0.
Context comparison is exact componentwise <=, with no positive tolerance. A synthetic
context [0, 0.75e-12, 1.5e-12], cost [3, 2, 1] check retains all three points and demonstrates
the non-transitivity of the obsolete 1e-12 rule. Input decimal
strings in all {len(fractions)} fraction fields are preserved verbatim in flat CSV columns.

This supplement adds an executable dependency audit. It creates no new temporal holdout,
empirical permit outcome, causal claim, significance test or AI-generated data label.
"""
    write_bytes(out, "README.md", text.encode("utf-8"))
    descriptions = {
        "schema_version": "Versioned serialization contract; incompatible changes require a new major version.",
        "certificate_set_id": "SHA256 of frozen source hashes and analysis rules; binding for the complete set.",
        "certificate_group_id": "SHA256 of set ID plus scenario/support/zone/view/selector.",
        "certificate_id": "SHA256 of group ID plus recorded removed/witness pair.",
        "scenario": "Existing energy run key from analysis.RUNS.",
        "support": "Existing spatial support, called scale in the source files.",
        "zone": "Frozen procurement zone; every chain node must have this same zone.",
        "view": "Preference vector key from analysis.VIEWS.",
        "selector": "Historical bound selector; no alternative witness search occurs during reopening.",
        "removed": "Region historically excluded in this exact scope.",
        "witness": "Recorded immediate regional witness; not a parcel or legal substitute.",
        "final_retained_region": "Retained endpoint reached by the recorded directed chain.",
        "chain_region_ids": "JSON array including removed region and each witness through the retained endpoint.",
        "dependency_region_ids": "JSON array of all chain nodes except the removed region.",
        "chain_hops": "Number of directed exclusion-to-witness edges.",
        "context_directions": "JSON object: original fraction column -> sign (+1 at_most; -1 at_least).",
        "lower_removed": "Historical LP lower bound on 2023 forecast electricity objective, EUR/kg H2.",
        "upper_witness": "Historical feasible witness upper bound after only deterministic anchor solves, EUR/kg H2.",
        "epsilon_eur_per_kg": "Inherited epsilon tolerance (0.05 EUR/kg H2), not a multivariate distance.",
        "numerical_guard_eur_per_kg": "Inherited strict cost guard (1e-7 EUR/kg H2).",
        "context_tolerance": "Exactly zero; componentwise interval <= has no positive tolerance.",
        "cost_margin_eur_per_kg": "lower_removed - upper_witness, before epsilon and numerical guard.",
        "strict_margin_eur_per_kg": "lower_removed - upper_witness - epsilon - numerical guard; strictly positive.",
        "upper_final_retained": "Reconstructed feasible bound of the retained endpoint, EUR/kg H2.",
        "terminal_strict_margin_eur_per_kg": "Removed LP lower minus terminal feasible upper minus epsilon and guard; >0.",
        "model_objective_strict_margin_eur_per_kg": "Post-hoc source-model objective audit, not an oracle screening input.",
        "witness_upper_basis": "constant_policy or min(constant_policy,anchor_model_objective).",
        "anchor_ids": "JSON array of independently reconstructed deterministic cost/context anchors.",
        "witness_land_upper": "Historical signed minimised context upper vector; columns ordered by analysis.VIEWS.",
        "removed_land_lower": "Historical signed minimised context lower vector, in the same order.",
        "same_zone": "Validated chain uses one zone throughout.",
        "componentwise_interval_pass": "Immediate and terminal intervals are no worse in all active directions.",
        "chain_acyclic": "No repeated node in this historical witness chain.",
        "terminal_retained": "Endpoint is retained in the same frozen membership scope.",
        "historical_status": "Immutable initial EXCLUDED record; effective_status is supplied by a query overlay.",
        "certificate_status": "Frozen regional certificate validity; a query can suspend it or require recomputation.",
        "legal_substitutability": "not_evaluated; regional certificates do not determine legal substitution.",
        "parcel_substitutability": "not_evaluated; no parcel assessment was supplied.",
        "certificate_scope": "regional_comparison_only.",
    }
    roles = ("removed", "witness", "final_retained")
    for col in columns:
        if col in descriptions:
            continue
        if col.endswith("_sha256"):
            descriptions[col] = "SHA256 of the identified original input file; full relative paths are in manifest.json."
        elif col.startswith("parcel_id_") or col.startswith("decision_owner_"):
            descriptions[col] = "unknown: no project-level/parcel-level owner or identifier was supplied."
        elif col.startswith("permit_status_"):
            descriptions[col] = "not_evaluated: no approval, rejection, withdrawal or revocation is inferred."
        elif col.endswith("_coverage_interval"):
            descriptions[col] = "JSON: original fraction and signed minimised lower/upper intervals for active view components; CLC gap allocation only."
        elif col.endswith("_original_context_fractions"):
            descriptions[col] = "JSON object of every *_fraction source column, preserved as original decimal strings."
        elif any(col == f"{role}_{fraction}" for role in roles for fraction in fractions):
            descriptions[col] = "Original decimal string of the named fraction for the named region role; no rounding or imputation."
        else:
            raise RegisterError(f"Undocumented register column: {col}")
    codebook = {"schema_version": SCHEMA, "csv_nested_values": "JSON objects/arrays; booleans serialized True/False",
                "register_columns": {col: descriptions[col] for col in columns},
                "fraction_fields": fractions,
                "states": {"EXCLUDED": "Historical regional exclusion still supported within the frozen scope.",
                           "REOPEN": "Recorded dependency revoked/unknown in a hypothetical model-input overlay; review required.",
                           "INVALID_RECOMPUTE": "Frozen preference/support/model/data binding changed; full set recomputation required.",
                           "RETAINED": "Kept in the frozen model roster; no permit or parcel inference.",
                           "WITNESS_REVOKED": "Hypothetically withdrawn retained model input, not a permit outcome.",
                           "WITNESS_UNKNOWN": "Retained model input treated as unknown in this hypothetical overlay.",
                           "NOT_EVALUATED_RECOMPUTE": "Roster assessment invalidated by a changed frozen binding.",
                           "REMOVE": "Energy-only exclusion with unprotected context."},
                "energy_only": {"rows": 127, "remove": 79, "retained": 48, "context_status": "unprotected",
                                "energy_incumbent_region": "Feasible electricity-cost incumbent; never a context witness.",
                                "raw_fractions": "Descriptive context fields only, no dominance guarantee."},
                "dependency_audit": {"unit": "one retained region in one scenario/support/zone/view/selector",
                                     "includes_zero_dependents": True, "states_tested": ["revoked", "unknown"],
                                     "affected_removed_ids": "JSON list of all dependent historical exclusions"},
                "scope_limitations": ["regional comparison only", "coverage intervals are not confidence intervals",
                                      "legal/parcel substitutability not evaluated", PERTURBATION]}
    write_json(out, "codebook.json", codebook)
    md = "# Codebook\n\nSchema: " + SCHEMA + "\n\n" + "\n".join(f"- `{c}`: {descriptions[c]}" for c in columns)
    md += "\n\nSee codebook.json for state definitions and the energy-only/dependency-audit contracts.\n"
    write_bytes(out, "codebook.md", md.encode("utf-8"))


def build(root):
    out = output_dir(root)
    checks = mechanism_self_test()
    config, source_manifest, sources, binding, records, members, energy, land, fractions = make_register(root)
    rejected, legacy_report = audit_legacy_v1(root, config, sources, records, members)
    audit, dependencies = dependency_audit(records, members)
    summary = main_summary(records, members, audit)
    energy_rows = energy_register(config, source_manifest, sources, energy, land, fractions, digest(binding))
    main_records, main_members = [r for r in records if in_main(r)], [r for r in members if in_main(r)]
    report = {"schema_version": SCHEMA, "status": "PASS", "historical_witnesses_checked": len(records),
              "zone_specific_scopes_checked": len({key(r) for r in members}),
              "memberships_checked": len(members), "retained_memberships_tested": len(audit),
              "leave_one_retained_witness_out_state_cases": 2 * len(audit),
              "positive_impact_retained_memberships": sum(r["reopen_if_revoked"] > 0 for r in audit),
              "zero_impact_retained_memberships": sum(r["reopen_if_revoked"] == 0 for r in audit),
              "dependency_index_rows": len(dependencies), "chain_hops_histogram": dict(sorted(Counter(r["chain_hops"] for r in records).items())),
              "minimum_strict_margin_eur_per_kg": min(r["strict_margin_eur_per_kg"] for r in records),
              "maximum_reopen_in_any_one_zone_scope": max(r["reopen_if_revoked"] for r in audit),
              "all_chains_acyclic": True, "all_terminals_retained": True, "all_chains_same_zone": True,
              "strict_cost_and_componentwise_intervals": True, "frozen_selectors_reconstructed": True,
              "all_source_hashes_match_analysis_manifest": True,
              "original_fraction_decimal_strings_preserved": True, "no_replacement_witness_selected": True,
              "energy_only_rows": len(energy_rows), "energy_only_remove": 79, "energy_only_retained": 48,
              "canonical_source": SOURCE_NAME, "legacy_v1_rejected_noncertifiable": len(rejected),
              "context_comparison": "exact componentwise <=; no positive tolerance",
              "synthetic_mechanism_checks": checks, "main_scope": summary, "evidence_class": PERTURBATION,
              "empirical_permit_outcome": False}
    write_csv(out, "exclusion_register.csv", records)
    write_json(out, "exclusion_register.json", {"schema_version": SCHEMA, "binding": binding, "records": records})
    write_csv(out, "land_certificate_membership.csv", members)
    write_csv(out, "energy_only_initial_register.csv", energy_rows)
    write_csv(out, "dependency_index.csv", dependencies)
    write_csv(out, "leave_one_retained_witness_out.csv", audit)
    main_audit = [r for r in audit if in_main(r)]
    write_csv(out, "main_leave_one_retained_witness_out.csv", main_audit)
    main_impacts = [{"schema_version": SCHEMA, **scope_dict(key(r)), "certificate_id": r["certificate_id"],
                     "removed": r["removed"], "witness": r["witness"], "chain_region_ids": r["chain_region_ids"],
                     "final_retained_region": r["final_retained_region"],
                     "terminal_reopen_if_revoked": summary["terminal_dependency_counts"][r["final_retained_region"]],
                     "terminal_reopen_if_unknown": summary["terminal_dependency_counts"][r["final_retained_region"]],
                     "strict_margin_eur_per_kg": r["strict_margin_eur_per_kg"], "evidence_class": PERTURBATION}
                    for r in main_records]
    write_csv(out, "main_exclusion_dependency_impacts.csv", main_impacts)
    write_json(out, "main_scenario_summary.json", summary)
    write_json(out, "validation_report.json", report)
    write_csv(out, "rejected_v1_certificates.csv", rejected)
    write_json(out, "legacy_v1_strict_audit.json", legacy_report)
    for state in ("revoked", "unknown"):
        result = transition(main_records, main_members, {summary["maximum_impact_terminal"]: state})
        result["selected_scope"] = {**summary, "example_only": True}
        require(result["reopen_count"] == summary["maximum_single_terminal_reopen_count"], "Example reopen count mismatch")
        write_json(out, f"example_{state}_terminal.json", result)
    write_json(out, "example_preference_invalidation.json", transition(main_records, main_members, invalidations=["preference"]))
    docs(out, summary, fractions, list(records[0]))
    names = ["exclusion_register.csv", "exclusion_register.json", "land_certificate_membership.csv",
             "energy_only_initial_register.csv", "dependency_index.csv", "leave_one_retained_witness_out.csv",
             "main_scenario_summary.json", "validation_report.json", "example_revoked_terminal.json",
             "example_unknown_terminal.json", "example_preference_invalidation.json", "README.md", "codebook.json", "codebook.md",
             "rejected_v1_certificates.csv", "legacy_v1_strict_audit.json"]
    names += ["main_leave_one_retained_witness_out.csv", "main_exclusion_dependency_impacts.csv"]
    manifest = {"schema_version": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                "certificate_set_id": digest(binding), "binding": binding, "evidence_class": PERTURBATION,
                "empirical_permit_outcome": False, "certificate_scope": "regional_comparison_only",
                "legal_substitutability": "not_evaluated", "parcel_substitutability": "not_evaluated",
                "canonical_source": SOURCE_NAME, "obsolete_source_role": "v1 audit only; four comparisons REJECTED_NONCERTIFIABLE",
                "legacy_v1_rejected_noncertifiable": len(rejected), "context_comparison": "exact componentwise <=; no positive tolerance",
                "preference_support_model_data_change_policy": "entire certificate set invalid; recompute from verified inputs",
                "reopening_policy": "all historical chain dependents REOPEN; no unverified alternative witness selected",
                "sources": sources, "builder_relative_path": "scripts/project03_build_exclusion_register.py",
                "builder_sha256": sha(Path(__file__)), "main_scope": summary,
                "counts": {k: report[k] for k in ("historical_witnesses_checked", "zone_specific_scopes_checked", "memberships_checked",
                                                  "retained_memberships_tested", "leave_one_retained_witness_out_state_cases",
                                                  "positive_impact_retained_memberships", "zero_impact_retained_memberships")},
                "outputs": {name: {"sha256": sha(output_file(out, name)), "bytes": output_file(out, name).stat().st_size}
                            for name in names}}
    write_json(out, "manifest.json", manifest)  # commit manifest last
    return {"status": "PASS", "output": str(out), **manifest["counts"], "main_scope": summary}


def load_built(root):
    out = output_dir(root)
    manifest = read_json(out / "manifest.json")
    require(manifest["schema_version"] == SCHEMA, "Unsupported register schema")
    require(sha(Path(__file__)) == manifest["builder_sha256"], "Builder changed; rebuild/register version review required")
    for name, source in manifest["sources"].items():
        source_path = (root / source["relative_path"]).resolve()
        require(source_path.is_relative_to(root.resolve()), "Manifest input path escaped root")
        require(sha(source_path) == source["sha256"], f"Frozen input changed ({name}); entire certificate set invalid, recompute")
    for name, metadata in manifest["outputs"].items():
        require(sha(output_file(out, name)) == metadata["sha256"], f"Register output corrupted/changed: {name}")
    payload = read_json(out / "exclusion_register.json")
    require(payload["binding"] == manifest["binding"] and digest(payload["binding"]) == manifest["certificate_set_id"], "Certificate binding mismatch")
    return out, manifest, payload["records"], read_csv(out / "land_certificate_membership.csv")


def validate(root):
    out, manifest, records, members = load_built(root)
    config, source_manifest, sources, binding, rebuilt, remembers, energy, land, fractions = make_register(root)
    require(canonical(records) == canonical(rebuilt), "JSON register does not reconstruct from historical sources")
    # Compare both independent serializations, including nested JSON and raw decimals.
    csv_records = read_csv(out / "exclusion_register.csv")
    require(len(csv_records) == len(records), "Register CSV row count mismatch")
    for row, record in zip(csv_records, records):
        require(set(row) == set(record), "Register CSV columns mismatch")
        for col, value in record.items():
            expected = canonical(value) if isinstance(value, (dict, list)) else str(value)
            require(row[col] == expected, f"CSV/JSON mismatch: {record['certificate_id']}/{col}")
    require(len(members) == len(remembers), "Membership CSV row count mismatch")
    for row, expected in zip(members, remembers):
        require(row == {k: str(v) for k, v in expected.items()}, "Membership CSV mismatch")
    audit, dependencies = dependency_audit(records, members)
    rejected, legacy_report = audit_legacy_v1(root, config, sources, records, members)
    energy_rows = energy_register(config, source_manifest, sources, energy, land, fractions, digest(binding))
    summary = main_summary(records, members, audit)
    checks = mechanism_self_test()
    for name, expected in (("leave_one_retained_witness_out.csv", audit), ("dependency_index.csv", dependencies),
                           ("energy_only_initial_register.csv", energy_rows)):
        actual = read_csv(out / name)
        require(len(actual) == len(expected), f"Derived CSV row count mismatch: {name}")
        for row, item in zip(actual, expected):
            require(row == {k: canonical(v) if isinstance(v, (dict, list)) else str(v) for k, v in item.items()}, f"Derived CSV mismatch: {name}")
    require(read_json(out / "legacy_v1_strict_audit.json") == legacy_report, "Legacy rejection audit mismatch")
    rejected_csv = read_csv(out / "rejected_v1_certificates.csv")
    require(rejected_csv == [{k: canonical(v) if isinstance(v, (dict, list)) else str(v) for k, v in r.items()} for r in rejected],
            "Rejected v1 CSV mismatch")
    require(summary == manifest["main_scope"] == read_json(out / "main_scenario_summary.json"), "Main summary mismatch")
    for state in ("revoked", "unknown"):
        expected = transition([r for r in records if in_main(r)], [r for r in members if in_main(r)],
                              {summary["maximum_impact_terminal"]: state})
        actual = read_json(out / f"example_{state}_terminal.json")
        # Membership booleans are True/False in JSON but strings in CSV; normalise.
        for row in expected["roster"]:
            row["retained"] = boolean(row["retained"])
        require(all(actual[k] == v for k, v in expected.items()), f"Perturbation snapshot mismatch: {state}")
    return {"status": "PASS", "historical_certificates_reconstructed": len(records),
            "leave_one_retained_witness_out_state_cases": 2 * len(audit),
            "energy_only_rows": len(energy_rows), "main_scope": summary,
            "legacy_v1_rejected_noncertifiable": len(rejected),
            "synthetic_mechanism_checks": checks, "source_and_output_hashes": "PASS"}


def query(root, args):
    out, manifest, records, members = load_built(root)
    def selected(r):
        return (r["scenario"] == args.scenario and r["support"] == args.support
                and r["view"] == args.view and r["selector"] == args.selector
                and (args.zone == "ALL" or r["zone"] == args.zone))
    rr, mm = [r for r in records if selected(r)], [r for r in members if selected(r)]
    require(bool(mm), "Requested scenario/support/zone/view/selector has no frozen certificate scope")
    states = {}
    for state, targets in (("revoked", args.revoke_witness), ("unknown", args.unknown_witness)):
        for cid in targets:
            require(cid not in states or states[cid] == state, f"Conflicting witness states: {cid}")
            states[cid] = state
    result = transition(rr, mm, states, args.invalidate)
    result.update({"certificate_set_id": manifest["certificate_set_id"],
                   "selected_scope": {k: getattr(args, k) for k in SCOPE},
                   "initial_candidate_count": len(mm), "initial_retained_count": sum(boolean(r["retained"]) for r in mm),
                   "initial_excluded_count": len(rr),
                   "affected_removed_ids": sorted(r["removed"] for r in result["exclusions"] if r["effective_status"] != "EXCLUDED"),
                   "scope_limitations": "regional comparison only; legal and parcel substitutability not_evaluated",
                   "whole_set_change_policy": "preference/spatial support/model/data changes invalidate the entire set; recomputation required"})
    # Initial registers and the committed manifest are immutable to query overlays.
    if args.output:
        target = output_file(out, args.output)
        reserved = {output_file(out, n) for n in ["manifest.json", *manifest["outputs"]]}
        require(target not in reserved, "Query output cannot overwrite a frozen register, evidence file or manifest")
        write_json(out, args.output, result)
    compact = {k: v for k, v in result.items() if k not in {"roster", "exclusions", "events"}}
    if args.output:
        compact["output"] = str(output_file(out, args.output))
    if args.full:
        return result
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate", "self-test", "query"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=argparse.SUPPRESS)
        if name == "query":
            p.add_argument("--scenario", default=MAIN[0])
            p.add_argument("--support", default=MAIN[1])
            p.add_argument("--zone", default="ALL")
            p.add_argument("--view", default=MAIN[2])
            p.add_argument("--selector", choices=sorted(LAND_SELECTORS), default=MAIN[3])
            p.add_argument("--revoke-witness", action="append", default=[])
            p.add_argument("--unknown-witness", action="append", default=[])
            p.add_argument("--invalidate", choices=INVALIDATORS, action="append", default=[])
            p.add_argument("--output", help="JSON path relative to the register output directory")
            p.add_argument("--full", action="store_true", help="Print full roster and historical records instead of summary")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        output_dir(root)  # fail before any write for forbidden roots, even self-test
        if args.command == "build":
            result = build(root)
        elif args.command == "validate":
            result = validate(root)
        elif args.command == "self-test":
            result = {"status": "PASS", **mechanism_self_test()}
        else:
            result = query(root, args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (RegisterError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
