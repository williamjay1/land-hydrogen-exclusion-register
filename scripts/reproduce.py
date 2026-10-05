"""Portable entry point over unchanged frozen scientific source and inputs.

Only file routing and PV row-subset loading are adapted. No model equation or
screening criterion is rewritten. Frozen result files are inputs, not outputs.
"""
from pathlib import Path, PureWindowsPath
import hashlib
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY = PureWindowsPath(r'D:\MLWork\Europe_green_hydrogen_siting')
sys.path.insert(0, str(ROOT / 'scripts'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def json_read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def confined(root, p):
    p = p.resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Generated output cannot escape the explicit checkout')
    return p


def relocate(text, root=ROOT):
    wp = PureWindowsPath(text)
    if wp.is_absolute():
        if not wp.is_relative_to(LEGACY):
            raise ValueError('An absolute legacy input is outside the recorded project')
        p = root.joinpath(*wp.relative_to(LEGACY).parts)
    else:
        p = root.joinpath(*wp.parts)
    return confined(root, p)


def register_module():
    m = importlib.import_module('project03_build_exclusion_register')
    m.DEFAULT_ROOT = ROOT
    m.relocate = relocate
    m.output_dir = lambda root: confined(root, root / 'results' / m.OUTPUT_NAME)
    def output_file(out, name):
        p = confined(out, out / name)
        if p == out.resolve():
            raise ValueError('Expected an output filename, not the directory itself')
        return p
    m.output_file = output_file
    return m


def load_subset(m):
    import numpy as np
    import pandas as pd
    info = json_read(ROOT / 'data/pv/manifest.json')
    result = {}
    expected_ids = None
    for year in range(2020, 2024):
        spec = info[str(year)]
        p, proof = ROOT / spec['path'], ROOT / spec['source_sample']
        if sha(p) != spec['sha256'] or sha(proof) != spec['source_sample_sha256']:
            raise ValueError('Public PV subset or source-version proof changed')
        meta = json_read(proof)['inputs']['meteo_data']
        if meta['radiation_db'] != 'PVGIS-SARAH3' or meta['year_min'] != year or meta['year_max'] != year:
            raise ValueError('PVGIS source database/year mismatch')
        with np.load(p, allow_pickle=False) as z:
            ids = z['candidate_ids'].astype(str).copy()
            matrix = z['pv_mw_per_mwp'].copy()
            stamps = z['timestamps_utc_ns'].copy()
        if expected_ids is None: expected_ids = ids
        if not np.array_equal(ids, expected_ids): raise ValueError('PV row order changed')
        expected = pd.date_range(f'{year}-01-01T00:00:00Z', periods=m.HOURS_PER_YEAR[year], freq='h')
        idx = pd.to_datetime(stamps, utc=True)
        if not idx.equals(expected) or matrix.shape != (128, len(expected)):
            raise ValueError('Incomplete or shifted PV UTC calendar')
        if not np.isfinite(matrix).all() or (matrix < -1e-8).any():
            raise ValueError('Nonfinite/negative PV input')
        result[year] = {'ids':ids, 'row':{cid:i for i,cid in enumerate(ids)},
            'timestamps':idx, 'matrix':matrix, 'manifest_path':ROOT/'data/pv/manifest.json',
            'matrix_path':p, 'manifest':spec, 'source_verified_by_raw_response':str(proof)}
    return result


def energy_module():
    m = importlib.import_module('project03_run_regret_certificate_pilot')
    m.PROJECT = ROOT
    m.DATA_ROOT = ROOT / 'datasets/intermediate/regret_certificate_pilot_20261002_v1'
    m.SITE_FILE = m.DATA_ROOT / 'validation_sites_128_excluding_development_v1.csv'
    m.RESULT_ROOT = ROOT / 'work/dispatch'
    m.LOG_ROOT = ROOT / 'work/logs'
    base = ROOT / 'datasets/intermediate/public_data_redesign/smard_fixed_nonpolish_cohort_2016_2023_v3'
    m.PRICE_2022 = base / 'smard_ccby_hourly_2022_v2.csv'
    m.PRICE_2023 = base / 'smard_ccby_hourly_2023_v2.csv'
    m.PRICE_2024 = m.DATA_ROOT / 'smard_utc_calendar_2024_four_zone_hourly_eur2023_v1.csv'
    m.PV_CACHE = ROOT / 'data/pv'
    m.PV_MATRICES = {y:ROOT/f'data/pv/pv_{y}_128.npz' for y in range(2020,2024)}
    m.PV_MANIFESTS = {y:ROOT/'data/pv/manifest.json' for y in range(2020,2024)}
    m.load_pv_cache = lambda: load_subset(m)
    return m


def verify_files():
    manifest = json_read(ROOT / 'provenance/file_manifest.json')
    for name, meta in manifest['files'].items():
        p = confined(ROOT, ROOT/name)
        if sha(p) != meta['sha256']:
            raise ValueError(f'Frozen release file changed: {name}')
    return len(manifest['files'])


def audit_data():
    import numpy as np
    import pandas as pd
    count = verify_files()
    m = energy_module()
    pv = load_subset(m)
    for year in (2022, 2023, 2024): m.load_prices(year)
    a = importlib.import_module('project03_analyse_land_shortlists')
    records = []
    for scenario, folder in a.RUNS.items():
        u = .3 if scenario.startswith('u30') else .7 if scenario.startswith('u70') else .5
        for year, days in [(2023,365),(2024,366)]:
            p = ROOT / 'results/regret_certificate_pilot_20261002_v1' / folder / f'daily_dispatch_{year}.csv'
            df = pd.read_csv(p)
            df = df.loc[~df.candidate_id.str.startswith('FRM')]
            assert len(df) == 127*days and df.groupby('candidate_id').utc_date.nunique().eq(days).all()
            balance = (df.load_mwh-df.grid_mwh-df.pv_mwh).abs().max()
            assert balance < 1e-6 and (df.load_mwh-24*u).abs().max() < 1e-6
            assert df.min_inventory_kg.min() >= -1e-7
            assert df.max_inventory_kg.max() <= 12*u*1000/55+1e-6
            records.append({'scenario':scenario,'year':year,'rows':len(df),'max_balance_error_mwh':float(balance)})
    assert len(json_read(ROOT/'results/land_shortlist_assessment_20261005_v2/land_dominance_witnesses.json')) == 5631
    print(json.dumps({'status':'PASS_RELEASE_INPUT_AND_DAILY_AUDIT','hashed_files':count,
        'pv_rows':len(pv[2020]['ids']), 'pv_years':[2020,2021,2022,2023],
        'complete_price_zones':4,'daily_checks':records}, indent=2))


def analyse(args):
    import pandas as pd
    m = importlib.import_module('project03_analyse_land_shortlists')
    m.PROJECT = ROOT
    m.ENERGY = ROOT / 'results/regret_certificate_pilot_20261002_v1'
    m.LAND = ROOT / 'datasets/intermediate/land_context_clc2018_natura2024_20261005_v1'
    m.OUT = ROOT / 'work/analysis'
    sys.argv = [sys.argv[0], *args]
    m.main()
    if '--self-test' in args: return
    ref = ROOT / 'results/land_shortlist_assessment_20261005_v2'
    for name in ['land_shortlist_membership.csv','land_threshold_queries.csv','land_shortlist_comparison.csv']:
        original = pd.read_csv(ref/name)
        fresh = pd.read_csv(m.OUT/name)
        if 'charged_anchor_and_rule_seconds' in original:
            original = original.drop(columns=['charged_anchor_and_rule_seconds'])
            fresh = fresh.drop(columns=['charged_anchor_and_rule_seconds'])
        pd.testing.assert_frame_equal(original, fresh, check_exact=False, atol=1e-12, rtol=1e-12)
    assert json_read(ref/'land_dominance_witnesses.json') == json_read(m.OUT/'land_dominance_witnesses.json')
    print(json.dumps({'status':'PASS_ANALYSIS_REPLAY','witnesses':5631,
        'comparison_rows':1920,'membership_rows':60960,'threshold_queries':94176,
        'timing_excluded_from_equality':True,'raw_GIS_recreated':False}))


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Use register {build,validate,query,self-test}, audit-data, analyse or dispatch [--smoke].')
    command, args = sys.argv[1], sys.argv[2:]
    if command == 'register':
        m = register_module()
        sys.argv = [str(ROOT/'scripts/project03_build_exclusion_register.py'), '--root', str(ROOT), *args]
        return m.main()
    if command == 'audit-data': audit_data(); return 0
    if command == 'analyse': analyse(args); return 0
    if command == 'dispatch':
        m = energy_module()
        if '--site-file' not in args: args = ['--site-file', str(m.SITE_FILE), *args]
        sys.argv = [str(ROOT/'scripts/project03_run_regret_certificate_pilot.py'), *args]
        return m.main()
    raise SystemExit(f'Unknown command: {command}')


if __name__ == '__main__': raise SystemExit(main())
