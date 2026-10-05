# Data rights and source attribution

This is not a blanket relicensing of all upstream sources. Original software
has an MIT licence. Author-created costs, context summaries, screening outputs,
certificates and non-map analytical artwork are offered under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) to the extent of Junjie
Zhang's rights. Preserve the following upstream notices and identify processing
changes. No endorsement by a provider or the European Union is implied.

| Source / material | Version and use | Release decision and attribution |
|---|---|---|
| SMARD market data | UTC prices 2022–2024; prior-hour forecasts and ex-post scoring; deflated to 2023 euros | Included cleaned inputs retain nominal prices and factors. CC BY 4.0; credit **Bundesnetzagentur \| SMARD.de**. [Official data-use terms](https://www.smard.de/en/datennutzung). The release does not use restricted direct ENTSO-E API files. |
| JRC PVGIS | Frozen PVGIS-SARAH3 2020–2023 cache; original 1165 rows, exact released subset 128 | Included hourly numerical subset and four source-version sample responses. [PVGIS usage conditions](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/usage-conditions-data-protection_en) state that information is free with no use restriction. Credit European Commission, Joint Research Centre, PVGIS. No claim of v6 or 2024 actual-weather generation. |
| Copernicus Land Monitoring Service | CORINE Land Cover 2018; area aggregation and interval widening for unclassified coverage | Derived fractions only. **Generated using European Union's Copernicus Land Monitoring Service information.** Aggregation, clipping and union operations are author transformations, not official classifications. [Data policy](https://land.copernicus.eu/en/data-policy). |
| European Environment Agency | Natura 2000 service description 2024; actual snapshot obtained in 2026; unioned regional fractions | Derived summaries only. Acknowledge EEA; retain any source-specific exceptions. [EEA legal notice](https://www.eea.europa.eu/en/legal-notice). No claim that this snapshot was available prospectively in 2023. |
| GISCO / EuroGeographics | NUTS 2024 IDs, regional support and representative locations | **No boundary geometries or boundary-containing map artwork are released here.** Administrative boundaries: © EuroGeographics. Source-derived regional identifiers/support retain the provider terms; MIT/CC BY does not replace these terms. [Specific statistical-unit terms](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units) include non-commercial use and required attribution. Users obtaining geometry must accept the applicable terms themselves. |
| Eurostat HICP | 2023-euro conversion factors in the cleaned price tables | Provider and extraction provenance remain in the acquisition source. [Eurostat copyright policy](https://ec.europa.eu/eurostat/about-us/policies/copyright). Factors are not new observations of project-specific costs. |
| Public planning records | H2V Normandy, H2V Marseille Fos and VoltH2 Farmsum | URLs, hashes, factual crosswalk and exact page locators only. Original PDFs are not included or relicensed. No automatic permission or complete dossier claim. |

The data release is for transparent scientific replication. If your proposed
use depends on commercial re-use of source-derived geography or any additional
third-party material, obtain permission from the relevant rightsholder; the
author's analytical licence does not grant such permission. Provider licensing
may change; source URLs and retrieved policy snapshot hashes are recorded in
`provenance/policy_records.json` (HTTP success alone is not a licence audit).
