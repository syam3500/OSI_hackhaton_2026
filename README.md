# E&L analytical coverage checker

A prototype that checks whether an extractable or leachable (E&L) compound has measured, method-specific analytical data in the FDA CLAP list. If it has none, the tool looks for similar CLAP compounds. If there are none, it says so. It does not predict a response factor.

## Why

In E&L studies, labs screen extracts with GC/MS and LC/MS and report compounds above the Analytical Evaluation Threshold (AET). Concentrations are often estimated with a Relative Response Factor (RRF). RRFs depend strongly on the instrument and the method, and a weak responder can be underestimated and fall below the AET. Public method-specific data are scarce: FDA CLAP covers 106 compounds, while Borton Appendix B lists about 18,000 compounds reported in E&L studies with no response data.

## What it does

Input: a compound name, CAS number or SMILES.

1. Resolve and standardise the structure with RDKit.
2. Compare it with every CLAP compound (2048-bit Morgan fingerprints, radius 2, Tanimoto similarity) and return the top 3 or 5 neighbours.
3. Check whether it lies inside the CLAP descriptor range (molecular weight, logP, TPSA, H-bond donors and acceptors, rotatable bonds, formal charge).
4. Report the exact CLAP record if there is one, the neighbours' GC/MS, LC/MS positive and LC/MS negative outcomes (always shown separately), and an evidence status.

| Status | Meaning |
|---|---|
| Measured | The exact compound has a method-specific CLAP record. |
| Represented | Close CLAP neighbours exist and the compound is inside the descriptor range. |
| Unsupported | The compound is outside the reference domain. No response is predicted. |

The similarity cut-off and descriptor rules are prototype operating rules, not validated analytical thresholds.

## Planned repository layout

Nothing has been pushed yet. This is the structure we plan to use:

```
scripts/        data preparation
data/raw/       source files, not to be committed 
data/interim/   intermediate files and PubChem cache, not to be committed
data/manual/    structure_overrides.csv for hand-checked structures
data/processed/ outputs of the pipeline
app/            Streamlit interface and chemical-space map
```

## Setup *(planned)*

## Run the data preparation *(planned)*

## Run the app *(planned)*

```bash
streamlit run app/app.py
```

## Limitations

- No response factor is predicted, and structural similarity does not prove detectability.
- PubChem CAS lookups can return a wrong or ambiguous record; trade names and mixtures may need manual structures.
- Borton Appendix B names and CAS numbers were compiled as received and not verified by its authors.

## Data sources

1. U.S. FDA. [Chemicals List for Analytical Performance (CLAP)](https://cdrh-rst.fda.gov/chemicals-list-analytical-performance-clap), RST24MC05.01, 2024.
2. Borton L.K., Coleman K.P. [Material-mediated pyrogens in medical devices: Myth or reality?](https://doi.org/10.14573/altex.2504231) ALTEX 43(1), 78–97. Appendix B.

## Licence

To be decided. Check the terms of each data source before committing processed data derived from it.
