# Data used in this prototype

This is a build from real supplied datasets, not a fabricated sample. Counts below describe the local preparation on 22 October 2026. The complete input filenames, byte sizes, SHA-256 hashes and generation time are in `data/processed/manifest.json`, also downloadable from the app.

| Input | What is used | Local build |
|---|---|---|
| `Borton_Sup2_AppendixB_18000.xlsx`, `Extractables` | Every original name and identifier field; row order defines `B00001` onward | 18,266 rows |
| Three supplied FDA CLAP RRF PDFs | Per-method outcomes and the individual RRFs at 5, 10 and 20 µg/mL | 106 CLAP codes; 318 method/code slots, including not-reported slots |
| Draft `sample_data/clap_compounds.csv` | CLAP names, CAS, internal-standard flags and extracted RRF bounds used to audit the PDF parsing | 106 records |
| Draft `sample_data/structures_clap.csv` | Previously resolved PubChem SMILES; retained as team-supplied structures, not new verified lookups | 103 usable structures after withholding RM84; RM24 and RM104 unresolved |
| `1-s2.0-S0273230026001364-mmc1_1285tox.xlsx`, `Compiled Dataset` | Reported CAS, SMILES, chemical name, Tier 1 and Tier 2 classes | 1,289 rows, despite the filename |
| `plastchem_db_v1.01.xlsx`, `Full database`, second-row headers | CAS, InChIKey, structure bridges, hazard values, polymer values, functions and industrial uses | 17,932 actual data rows; duplicate source evidence preserved |
| Public `data/MassSpecGym.tsv` | Spectrum ID, InChIKey, adduct, instrument metadata; grouped without loading peak arrays into the app | 231,104 spectra; 28,929 connectivity keys |
| `data/manual/structure_overrides.csv` | Explicit hand-reviewed Borton structures, when supplied | Empty initially |
| `data/manual/clap_structure_exclusions.csv` | Documented exclusion of the unverified RM84 structure | One reviewed exclusion |

## Map coverage and missing structures

The 18,266 Borton rows are retained. This build resolves **3,661 rows**, representing **3,318 unique full source InChIKeys**. **70 rows have identity conflicts** and **14,535 remain unresolved**. All rows remain in the searchable table. Only resolved structures receive map coordinates; there are no invented positions for unresolved compounds.

Structures are obtained offline through checksum-valid CAS numbers shared with the supplied CLAP, PlastChem and Stavitskaya structure tables. A row with multiple candidate full InChIKeys is held for review. This approach is a lower bound: most Borton entries have not undergone an independent structure lookup, and source-assigned CAS numbers can themselves be wrong.

The source contains a Bisphenol A row with `2080-05-7` as well as rows carrying the actual `80-05-7` identifier. The former is not silently repaired or matched by CAS suffix. The resolved examples use the correctly identified rows.

RM84 is described by the FDA as a commercial blend. The supplied draft assigns it an unverified single structure. That structure is excluded from all fingerprints, identity bridges and projection coordinates; the actual CLAP method records remain accessible by record code. The mixture is not silently treated as a measured pure compound.

## What each layer means

- **CLAP:** measured numeric RRFs apply to the stated FDA conditions. Numeric, partially measured, not detected, below DL and not reported are preserved separately. Internal-standard flags come from the supplied team extraction. Parsed numeric bounds are checked against that extraction.
- **Stavitskaya:** the supplied supplement provides chemical classes, not a row-level toxic/non-toxic endpoint. The app does not transform dataset membership into a toxicity label.
- **PlastChem:** hazard scores and associations are displayed as reported. The companion `Hazard_information_PlastChem_v1.01.xlsx` is not separately ingested; the main workbook already contains the hazard fields used here. Hazard scores are not binary toxicity predictions. Polymer values mean released (2), present (1), used (0.5), inconclusive (0.25) and not detected (0), as explained by the source README.
- **MassSpecGym:** a connectivity-level library match establishes reference availability, not confirmation of an experimental identification or a stereoisomer. Positive/negative counts come from the actual adduct suffix; unspecified polarity stays unspecified. This is an MS/MS layer, not a GC-EI spectral library.
- **Natural occurrence:** unknown. No suitable field is present in the supplied PlastChem workbook. Bioplastic associations are not interpreted as natural occurrence.
- **NIST:** not integrated. Its raw extraction runs require compound identification and annotation before a compound-to-polymer evidence layer can be produced. PlastChem associations are shown with their own source instead.

The group chat, draft code, background brief and existing repository notebook informed the implementation. Their illustrative or previously printed aggregate statistics are not used as new measured evidence. The SharePoint files did not require account access because local copies were supplied. No private chat content is included in the application or its exports.

## Primary source links

- Borton and Coleman: [Appendix B](https://doi.org/10.14573/altex.2504231s2)
- FDA: [Chemicals List for Analytical Performance](https://cdrh-rst.fda.gov/chemicals-list-analytical-performance-clap)
- Stavitskaya et al.: [Exploring the chemical space of pharmaceutical extractables and leachables](https://doi.org/10.1016/j.yrtph.2026.106163)
- Wagner et al.: [PlastChem database v1.01](https://doi.org/10.5281/zenodo.10701706)
- MassSpecGym: [Dataset](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym), [code](https://github.com/pluskal-lab/MassSpecGym)
- NIST: [LC-MS/MS and GC-MS measurements of polymer extractables and leachables](https://data.nist.gov/od/id/ark:/88434/mds2-4147)

Raw and processed files are ignored by Git. Review each source's redistribution terms before publishing them. The app performs no model training. The default local mode makes no external database query. The optional AI agent can request a small live PubChem PUG REST property record. The UI labels that result as online evidence and keeps it separate from the prepared build.
