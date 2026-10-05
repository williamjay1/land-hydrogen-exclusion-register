"""Retrieve immutable public GIS responses and audit NUTS-3 land context.

This is a regional context product, NOT a parcel suitability assessment.
Raw responses/requests go to F; geometry processing and derived products go to D.
No ArcGIS simplification, feature-count truncation, or area-field shortcut is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import threading
import time

import pandas as pd
from pyproj import Transformer
import requests
from shapely import make_valid
from shapely.geometry import Point, shape
from shapely.ops import transform, unary_union

PROJECT = Path(r"D:\MLWork\Europe_green_hydrogen_siting")
DEFAULT_ROSTER = PROJECT / "datasets/intermediate/regret_certificate_pilot_20261002_v1/validation_sites_128_excluding_development_v1.csv"
NUTS = Path(r"E:\AcademicData\Project03\raw\gisco\NUTS_RG_01M_2024_4326_LEVL_3.geojson")
RAW = Path(r"F:\AcademicData\Project03\raw\land_context_clc2018_natura2024_20261005_v1")
OUT = PROJECT / "datasets/intermediate/land_context_clc2018_natura2024_20261005_v1"
CLC = "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/0"
NATURA = "https://bio.discomap.eea.europa.eu/arcgis/rest/services/ProtectedSites/Natura2000Sites/MapServer"
POLICIES = {
    "copernicus_data_policy": "https://land.copernicus.eu/en/data-policy",
    "eea_legal_notice": "https://www.eea.europa.eu/en/legal-notice",
    "clc_product": "https://land.copernicus.eu/api/en/products/corine-land-cover/clc2018",
}
TO_EQUAL_AREA = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Project03-reproducible-regional-land-context/1.0 (academic GIS audit)"
RAW_LOCK = threading.RLock()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace raw source: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)
    path.chmod(stat.S_IREAD)


def retrieve(url: str, params: dict | None = None, *, label: str, expect_json: bool = True) -> tuple[object, str]:
    request = {"method": "GET", "url": url, "params": params or {}}
    key = digest(json_bytes(request))
    target = RAW / "responses" / f"{label}_{key}.bin"
    record = RAW / "requests" / f"{label}_{key}.json"
    with RAW_LOCK:
        if target.exists() and record.exists():
            metadata = json.loads(record.read_text(encoding="utf-8"))
            data = target.read_bytes()
            if digest(data) != metadata["response_sha256"]:
                raise RuntimeError(f"Raw response changed: {target}")
            return (json.loads(data) if expect_json else data), key
        # A response without a committed request record cannot be silently overwritten.
        if target.exists() or record.exists():
            raise RuntimeError(f"Incomplete raw transaction: {key}")
    last_error = None
    for attempt in range(4):
        try:
            response = SESSION.get(url, params=params, timeout=(20, 150))
            response.raise_for_status()
            decoded = response.json() if expect_json else response.content
            if expect_json and isinstance(decoded, dict) and "error" in decoded:
                raise RuntimeError(f"ArcGIS error: {decoded['error']}")
            with RAW_LOCK:
                if target.exists() and record.exists():
                    existing = target.read_bytes()
                    return (json.loads(existing) if expect_json else existing), key
                immutable(target, response.content)
                immutable(record, json_bytes({
                    **request, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "response_url": response.url, "status": response.status_code,
                    "response_headers": dict(response.headers),
                    "response_sha256": digest(response.content), "bytes": len(response.content),
                    "response_file": str(target),
                }))
            return decoded, key
        except (requests.RequestException, ValueError, RuntimeError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Retrieval failed for {url}: {last_error}")


def fetch_layer(layer: str, bbox: tuple, label: str, out_fields: str) -> tuple[list[dict], dict]:
    ids_result, ids_key = retrieve(layer + "/query", {
        "where": "1=1", "geometry": ",".join(f"{x:.10f}" for x in bbox),
        "geometryType": "esriGeometryEnvelope", "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects", "returnIdsOnly": "true", "f": "json",
    }, label=label + "_ids")
    ids = sorted(set(ids_result.get("objectIds") or []))
    features, keys = [], [ids_key]
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        response, key = retrieve(layer + "/query", {
            "objectIds": ",".join(map(str, batch)), "outFields": out_fields,
            "returnGeometry": "true", "outSR": 4326, "f": "geojson",
        }, label=label + "_features")
        if response.get("exceededTransferLimit") or response.get("properties", {}).get("exceededTransferLimit"):
            raise RuntimeError("ArcGIS response truncated; no metrics will be calculated")
        rows = response.get("features", [])
        observed = [int(f["properties"]["OBJECTID"]) for f in rows]
        if len(observed) != len(set(observed)) or set(observed) != set(batch):
            raise RuntimeError(f"Feature IDs do not match request: {label}")
        features.extend(rows)
        keys.append(key)
    return features, {"feature_count": len(features), "request_keys": keys}


def projected_features(features: list[dict], mask) -> tuple[list[tuple[dict, object]], int]:
    output, repairs = [], 0
    for feature in features:
        if feature.get("geometry") is None:
            raise RuntimeError("A queried feature has no geometry")
        geom = shape(feature["geometry"])
        if not geom.is_valid:
            repairs += 1
            geom = make_valid(geom)
        geom = transform(TO_EQUAL_AREA, geom)
        if not geom.is_valid:
            repairs += 1
            geom = make_valid(geom)
        clipped = geom.intersection(mask)
        if not clipped.is_empty and clipped.area > 0:
            output.append((feature["properties"], clipped))
    return output, repairs


def calculate_context(clc_rows: list, protected_rows: list, masks: dict[str, object]) -> list[dict]:
    metrics = []
    for scale, mask in masks.items():
        clc, repair_clc = projected_features(clc_rows, mask)
        protected, repair_protected = projected_features(protected_rows, mask)
        by_code = defaultdict(list)
        for properties, geometry in clc:
            by_code[str(properties["Code_18"])].append(geometry)
        class_unions = {code: unary_union(geometries) for code, geometries in by_code.items()}
        coverage = unary_union(list(class_unions.values()))
        protection = unary_union([g for _, g in protected])
        area = float(mask.area)
        class_area = {code: float(g.area) for code, g in class_unions.items()}
        unknown = max(0.0, area - coverage.area) / area
        overlap = max(0.0, sum(class_area.values()) - coverage.area) / area
        def union_share(codes: list[str]) -> float:
            return unary_union([class_unions[c] for c in codes if c in class_unions]).area / area
        metrics.append({
            "scale": scale, "region_or_mask_area_km2": area / 1e6,
            "clc_coverage_fraction": coverage.area / area,
            "clc_unknown_fraction": unknown, "clc_cross_class_overlap_fraction": overlap,
            "industrial_commercial_fraction": union_share(["121"]),
            "agricultural_fraction": union_share([c for c in class_unions if c.startswith("2")]),
            "forest_seminatural_fraction": union_share([c for c in class_unions if c.startswith("3")]),
            "wetland_fraction": union_share([c for c in class_unions if c.startswith("4")]),
            "artificial_fraction": union_share([c for c in class_unions if c.startswith("1")]),
            "water_fraction": union_share([c for c in class_unions if c.startswith("5")]),
            "natura2000_union_fraction": protection.area / area,
            "natura2000_sum_before_union_fraction": sum(g.area for _, g in protected) / area,
            "class_area_km2": {c: a / 1e6 for c, a in class_area.items()},
            "clc_geometry_repairs": repair_clc, "natura_geometry_repairs": repair_protected,
            "qa_pass": coverage.area / area >= 0.95 and overlap < 0.001,
        })
    return metrics


def aggregate() -> None:
    rows = []
    for file in sorted((OUT / "candidate_checkpoints").glob("*.json")):
        checkpoint = json.loads(file.read_text(encoding="utf-8"))
        for item in checkpoint["metrics"]:
            rows.append({k: v for k, v in {**checkpoint["candidate"], **item}.items() if k != "class_area_km2"})
    if rows:
        temporary = OUT / "land_context_metrics.partial.csv"
        pd.DataFrame(rows).to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(OUT / "land_context_metrics.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--per-zone-limit", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--exclude-corsica", action="store_true", help="Exclude FRM regions: mainland FR day-ahead prices do not apply")
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--combined-natura-layer", action="store_true", help="Use layer 2 only after auditing that it covers site types A, B and C")
    parser.add_argument("--audit-cached", action="store_true", help="Offline recompute cached class/protection fractions from archived feature responses")
    args = parser.parse_args()
    if shutil.disk_usage("D:\\").free < 5 * 1024 ** 3:
        raise RuntimeError("D drive has less than 5 GB available")
    if shutil.disk_usage("F:\\").free < 10 * 1024 ** 3:
        raise RuntimeError("F drive has less than 10 GB available")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidate_checkpoints").mkdir(exist_ok=True)
    metadata = {}
    for name, url in POLICIES.items():
        _, metadata[name] = retrieve(url, label=name, expect_json=False)
    for label, url in [("clc_layer", CLC)] + [(f"natura_layer_{i}", NATURA + f"/{i}") for i in range(3)]:
        layer_meta, key = retrieve(url, {"f": "pjson"}, label=label)
        if layer_meta.get("type") != "Feature Layer":
            raise RuntimeError("Expected vector feature layer")
        metadata[label] = key
    root_meta, metadata["natura_service"] = retrieve(NATURA, {"f": "pjson"}, label="natura_service")
    if args.combined_natura_layer:
        stats = json.dumps([{"statisticType": "count", "onStatisticField": "OBJECTID", "outStatisticFieldName": "n"}])
        types, metadata["natura_site_type_counts"] = retrieve(NATURA + "/2/query", {
            "where": "1=1", "groupByFieldsForStatistics": "SITETYPE", "outStatistics": stats,
            "returnGeometry": "false", "f": "json",
        }, label="natura_combined_site_type_counts")
        if {r["attributes"]["SITETYPE"] for r in types["features"]} != {"A", "B", "C"}:
            raise RuntimeError("Combined Natura layer does not cover all designation types")
    roster = pd.read_csv(args.roster)
    if roster.candidate_id.duplicated().any():
        raise ValueError("Duplicate candidate IDs")
    if args.exclude_corsica:
        roster = roster.loc[~roster.candidate_id.str.startswith("FRM")].copy()
    if args.per_zone_limit:
        roster = roster.groupby("bidding_zone_key", sort=False).head(args.per_zone_limit)
    if args.max_candidates:
        roster = roster.head(args.max_candidates)
    registry = json.loads(NUTS.read_text(encoding="utf-8"))
    regions = {f["properties"]["NUTS_ID"]: f for f in registry["features"]}
    nuts_hash = digest(NUTS.read_bytes())
    started = datetime.now(timezone.utc).isoformat()
    def process_site(number: int, site: dict) -> None:
        cid = str(site["candidate_id"])
        checkpoint_file = OUT / "candidate_checkpoints" / f"{cid}.json"
        if checkpoint_file.exists():
            existing = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if existing["nuts_source_sha256"] != nuts_hash:
                raise RuntimeError("NUTS source changed; use a new dataset version")
            if not args.audit_cached:
                print(json.dumps({"candidate_id": cid, "number": number, "total": len(roster), "cached": True}), flush=True)
                return
            clc = []
            for key in existing["clc_query"]["request_keys"][1:]:
                clc.extend(json.loads((RAW / "responses" / f"clc_features_{key}.bin").read_bytes())["features"])
            protected = []
            layer_ids = [2] if len(existing["natura_queries"]) == 1 else [0, 1, 2]
            for layer_id, audit in zip(layer_ids, existing["natura_queries"]):
                for key in audit["request_keys"][1:]:
                    protected.extend(json.loads((RAW / "responses" / f"natura_{layer_id}_features_{key}.bin").read_bytes())["features"])
            region = shape(regions[cid]["geometry"])
            if not region.is_valid:
                region = make_valid(region)
            mask = transform(TO_EQUAL_AREA, region)
            if not mask.is_valid:
                mask = make_valid(mask)
            point = transform(TO_EQUAL_AREA, Point(float(site["lon"]), float(site["lat"])))
            masks = {"nuts3": mask, **{f"buffer_{km}km_clipped": mask.intersection(point.buffer(km * 1000)) for km in (5, 10, 20)}}
            recalculated = calculate_context(clc, protected, masks)
            for old, new in zip(existing["metrics"], recalculated):
                for key in ["clc_coverage_fraction", "industrial_commercial_fraction", "agricultural_fraction",
                            "forest_seminatural_fraction", "natura2000_union_fraction"]:
                    if abs(old[key] - new[key]) > 1e-10:
                        raise RuntimeError(f"Offline recomputation differs: {cid}/{old['scale']}/{key}")
            print(json.dumps({"candidate_id": cid, "offline_recomputation": "PASS"}), flush=True)
            return
        began = time.perf_counter()
        region = shape(regions[cid]["geometry"])
        region_invalid = not region.is_valid
        if region_invalid:
            region = make_valid(region)
        point = Point(float(site["lon"]), float(site["lat"]))
        if not region.covers(point):
            raise RuntimeError(f"Representative coordinate is outside {cid}")
        mask = transform(TO_EQUAL_AREA, region)
        if not mask.is_valid:
            mask = make_valid(mask)
        projected_point = transform(TO_EQUAL_AREA, point)
        masks = {"nuts3": mask}
        masks.update({f"buffer_{km}km_clipped": mask.intersection(projected_point.buffer(km * 1000)) for km in (5, 10, 20)})
        clc, clc_audit = fetch_layer(CLC, region.bounds, "clc", "OBJECTID,Code_18,ID")
        protected, audits = [], []
        for layer_id in ([2] if args.combined_natura_layer else range(3)):
            features, audit = fetch_layer(NATURA + f"/{layer_id}", region.bounds, f"natura_{layer_id}", "OBJECTID,SITECODE,SITETYPE,RELEASE_DATE")
            protected.extend(features)
            audits.append(audit)
        result = {
            "candidate": {"candidate_id": cid, "zone": site["bidding_zone_key"], "country": site["country"],
                          "nuts3_name": site["nuts3_name"], "lat": site["lat"], "lon": site["lon"]},
            "metrics": calculate_context(clc, protected, masks),
            "nuts_source": str(NUTS), "nuts_source_sha256": nuts_hash,
            "nuts_geometry_was_invalid": region_invalid,
            "clc_query": clc_audit, "natura_queries": audits,
            "metadata_keys": metadata,
            "natura_service_description": root_meta.get("serviceDescription"),
            "natura_query_strategy": "combined layer 2 (all site types)" if args.combined_natura_layer else "union of layers 0, 1 and 2",
            "seconds": time.perf_counter() - began,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "CLC2018 and Natura2000 service snapshot (2024); regional context, not vacant land or legal/buildable feasibility",
        }
        temporary = checkpoint_file.with_suffix(".partial.json")
        temporary.write_bytes(json_bytes(result))
        temporary.replace(checkpoint_file)
        print(json.dumps({"candidate_id": cid, "number": number, "total": len(roster),
                          "seconds": result["seconds"], "nuts3_metrics": result["metrics"][0]}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(process_site, n, site) for n, site in enumerate(roster.to_dict("records"), 1)]
        for future in as_completed(pending):
            future.result()
            aggregate()
    aggregate()
    manifest = {
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "requested_candidates": len(roster), "roster": str(args.roster), "roster_sha256": digest(args.roster.read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()), "raw_root": str(RAW), "derived_root": str(OUT),
        "metadata_keys": metadata, "response_bytes_total": sum(f.stat().st_size for f in (RAW / "responses").glob("*.bin")),
        "data_vintage": {"clc": 2018, "nuts": 2024, "natura": "service description 2024, snapshot retrieved 2026"},
        "excluded_corsica": args.exclude_corsica,
        "workers": args.workers,
        "offline_recomputation_requested": args.audit_cached,
        "no_historical_information_set_claim": True,
    }
    (OUT / "run_manifest.json").write_bytes(json_bytes(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
