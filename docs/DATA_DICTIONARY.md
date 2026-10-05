# Data dictionary

CSV files have a header row, UTF-8 text and no hidden spreadsheet formulas.
Region identifiers are strings, not numeric observations. Exact columns are
retained from the original output; the source code and manifests define them.

| File group | Unit / keys / interpretation |
|---|---|
| `data/pv/pv_YEAR_128.npz` | `candidate_ids`: original 128 IDs; `timestamps_utc_ns`: signed int64 Unix nanoseconds; `pv_mw_per_mwp`: float32 MW output per installed MWp. Rows subset unchanged from the original 1165-row cache. |
| SMARD inputs | `timestamp_utc`, zone EIC/key, provider identifiers; nominal and 2023-euro EUR/MWh columns and dimensionless deflation factor. The 2022/2023 economic input is `price_eur_per_mwh`; 2024 is `price_eur_2023_per_mwh`. |
| Land context metrics | One row per `candidate_id` and `scale`; 127 IDs × four supports = 508 rows. Fraction columns are dimensionless [0,1], not percent. `natura2000_sum_before_union_fraction` is a diagnostic which can exceed 1; the union fraction is the analysis variable. Area is km². `qa_pass` is a production check, not legal feasibility. |
| Annual candidate costs and bounds | `candidate_id`, `zone`; forecast model, lower-bound and feasible-policy upper-bound columns in 2023 EUR/kg hydrogen. `realized_settlement_eur_per_kg_2024` is ex-post scoring. `milp_wall_seconds_2023` and related timing columns are historical seconds. Cost is electricity procurement only, not LCOH. |
| Daily dispatch | Candidate and `utc_date`; grid/PV/load in MWh; tank minima/maxima in kg; solver gap dimensionless. 365/366 daily records per candidate in 2023/2024. Original operational tables also contain subsequently excluded FRM02. |
| Shortlist comparison | `scenario`, `scale`, `zone`, `view`, `selector`; integer counts for candidate, retained, frontier and omitted options; fractions dimensionless; regrets EUR/kg. `preview_only` must be false for the canonical data. |
| Membership | Same scenario/support/zone/view/selector keys plus candidate ID; boolean retained/frontier/near-optimal fields. 60960 rows. |
| Threshold queries | Signed minimisation threshold and original fraction threshold; `direction` records at-least/at-most. Counts are qualifying options, not probabilities or legally approved parcels. Empty eligible shortlists have null regrets and a true loss flag. 94176 rows. |
| Dominance witnesses JSON | Scenario/support/zone/view/selector, removed ID, named retained witness, exact bounds and context intervals. 5631 strict records; EUR/kg cost margin is before subtracting epsilon and guard. |
| Generated register | Schema, source/data/model/context bindings, certificate IDs, dependency chains and retained terminal. Unknown organisational owners/parcel status remain unknown. `query` creates a hypothetical read-only overlay unless an explicit output is requested. |

Views separately prefer higher industrial/commercial fraction or lower
agricultural, natural or Natura coverage; the joint view requires componentwise
compatibility. These are declared comparison preferences, not universal land
preferences. Supports are NUTS-3 and region-clipped 5/10/20 km buffers. Scenario
keys identify utilisation and the price forecast rule. Full JSON register and
CSV register are generated from the same source; do not edit either as a way
to change the frozen exclusion history.
