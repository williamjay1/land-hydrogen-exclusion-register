# Land alternatives: reproducible exclusion register

Code and supporting numerical data for **An exclusion register for land
alternatives in regional hydrogen planning**, by Junjie Zhang. This is research
software, not a permitting tool or a record of actual planning decisions.

## Reproduce the central findings

Python 3.12 is the recorded production version. The register needs only Python's
standard library; install the pinned numerical dependencies for the other steps.
Use a D-drive checkout on the author's Windows machine. Other users may use an
explicit project checkout on Linux/macOS. No API credential is required.

```sh
python -B scripts/reproduce.py register build
python -B scripts/reproduce.py register validate
python -B scripts/reproduce.py register query --revoke-witness DEA29
python -B scripts/reproduce.py register query --unknown-witness DEA29
python -B scripts/reproduce.py register query --invalidate preference
python -B -m unittest discover -s tests -v
python -m pip install -r requirements-reproduction.txt
python -B scripts/reproduce.py audit-data
python -B scripts/reproduce.py analyse
python -B scripts/reproduce.py dispatch --smoke
```

Frozen inputs are never overwritten by the wrapper. Generated register files
are ignored by Git; `register build` reconstructs them from the included source
tables. The analysis replay writes to `work/analysis` and checks memberships,
query outcomes and removal witnesses against the frozen outputs. Replay timings
are not new end-to-end solver benchmarks.

For an optional, expensive rerun of the original operational model:

```sh
python -B scripts/reproduce.py dispatch --utilisation 0.50 --price-forecast-method four_week_mean
```

Repeat with utilisation 0.30/0.70, and with 0.50 plus
`last_week_same_hour`, for the other scenarios. The wrapper uses the original
128-region frozen operational roster, including the subsequently excluded
Corsican region. The land comparison removes FRM02 without replacement, leaving
127 regions. New solver runs go to `work/dispatch`; runtimes and solver choices
can differ from the original machine. Full two-year reruns are not claimed to
have been repeated as part of public-release preparation.

## Contents and scope

- Exact 128-row subsets of the original 2020–2023 PVGIS-SARAH3 float32 hourly
  caches, timestamps and independent source-version sample responses.
- Complete SMARD UTC price inputs for 2022–2024, with original deflation factors
  and provider identifiers. Bundled 2022/2023 source panels contain additional
  zones; the model selects BE, DE-LU, mainland FR and NL only.
- Four scenario annual costs/bounds and daily dispatch tables; 508 derived land
  context rows; all strict shortlist, membership, threshold-query and witness
  outputs, and the rejected legacy version for audit only.
- Unmodified original analysis/model/register/GIS/plot source, a portable entry
  wrapper, tests, provenance hashes and five non-map figure products.

The 127 regions are **not** 127 independent electricity markets. The land
extension is retrospective. Costs are variable electricity procurement costs,
not full LCOH. PV inputs are prior-year climatologies, not actual-year weather.
Certificates cover the stated finite model and context definitions, not parcels,
rights, environmental approval or institutional improvements.

Raw GIS geometries and original public planning PDFs are **not** redistributed.
The documentary ledger gives URLs, locators, source hashes and the permit-date
conflict. The original GIS acquisition code documents retrieval and processing;
recreating that layer requires obtaining the correct source snapshots under
their terms. Thus the bundle reproduces the numerical screening from frozen
derived inputs, not complete raw-data acquisition. Unpublished manuscript
sources and PDFs are deliberately not published as a preprint in this repository.

## Licensing, citation and archiving

Original software is under the MIT licence. Author-created analytical products
are under CC BY 4.0 **only to the extent of the author's rights**; upstream
source terms and attribution are preserved in [DATA_LICENSE.md](DATA_LICENSE.md).
No open licence is conferred on GISCO boundaries or linked planning documents.

Use `CITATION.cff` for the software citation. A DOI must only be added after the
author actually completes the Zenodo deposit. Follow
[the author handover](docs/ZENODO_HANDOVER_zh.md); do not mistake a Git tag for an
archived DOI. There is no automatic Zenodo deposition in the CI workflow.

See [methods and provenance](docs/METHOD_AND_PROVENANCE.md),
[research integrity](docs/RESEARCH_INTEGRITY.md) and
[data dictionary](docs/DATA_DICTIONARY.md). Verification reports document what
was actually run, not journal acceptance or independent peer certification.
