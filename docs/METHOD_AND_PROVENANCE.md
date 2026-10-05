# Method-to-code and data provenance

The original source files are unmodified frozen records. `scripts/reproduce.py`
adapts file locations, portable legacy-path parsing and the exact PV row subset;
it does not alter model equations, prices, forecasts, context comparisons or
the strict cost guard. A generated-register rebuild records the actual builder
and input versions. Do not execute a legacy script directly expecting its
original D/E/F paths to exist on a different machine.

| Mathematical / procedural quantity | Original implementation |
|---|---|
| Daily 24-hour load, grid/PV balance, binary commitments/transitions, ramp and tank constraints | `project03_run_regret_certificate_pilot.solve_day_milp` |
| Continuous operational LP lower bound | `relaxed_daily_lower_bound_operational_lp` in the same file |
| Feasible constant-load upper bound | `constant_feasible_daily_cost` in the same file |
| Past-only UTC PV proxy and leap-day fallback | `historical_climatology` |
| 7/14/21/28-day or 7-day same-hour price forecast | `forecast_day_ahead_prices` |
| Annual bounds/objectives and settlement scoring | `run` and yearly solve functions in the same file |
| Missing CLC coverage intervals and context direction | `project03_analyse_land_shortlists.land_intervals` |
| Bound/anchor removal criterion and named witness | `bound_land_screen`; epsilon 0.05 EUR/kg, cost guard 1e-7 EUR/kg; **exact** componentwise land comparisons |
| Frontier and near-optimal comparisons | `nondominated` |
| Certificate reconstruction and direct dependencies | `project03_build_exclusion_register.make_register`, `trace`, `dependency_audit` |
| Withdrawal/unknown overlay and whole-scope invalidators | `transition`, `query`, `mechanism_self_test` |
| Regional support, polygons, union and equal-area fractions | `project03_build_land_context.calculate_context` and associated acquisition functions; raw GIS not bundled |

The initial frozen operational roster contains 128 regions. The land extension
excludes FRM02 because the mainland French procurement zone does not apply;
there is no replacement, leaving 127. Four price zones are market contexts,
not 127 independent electricity markets. Land preferences, the geographic
correction and documentary exercise were retrospective.

The original data-source and output hashes remain intact in the manifests.
The public PV subset is additionally hashed and checked bit-for-bit against
the original cache. Prices contain original nominal values and 2023-euro
conversion factors. The release wrapper verifies complete UTC calendars,
including 8784 hours for leap years; it never fills missing test prices.

2024 scoring is another price year in the same markets using prior-year PV
climatology, not independent geography, measured operation or actual-weather
validation. Certificates concern only the finite 2023 forecast objective and
specified regional context intervals. Runtime replay uses historical solver
checkpoints, not new hardware timing. Files in the legacy v1 folder are audit
material only: four positive-tolerance comparisons are invalid under the strict
rule and cannot be used as current certificates.

Original GIS acquisition and plot code is included for methodological
transparency; its source dependencies are not silently fabricated. Five
non-map figures are included as vector PDF/SVG and native 1200-dpi PNG. The
boundary-containing map requires provider geometry and terms and is not part
of this public data repository. Institutional adoption has not been measured.
