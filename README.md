# E&L Atlas

A working Streamlit prototype for exploring the Borton chemical space and checking method-specific FDA CLAP coverage. It uses RDKit Morgan fingerprints, Tanimoto similarity and descriptor ranges. There is no predictive model and no RRF is inferred for a query compound.

The existing `notebooks/borton_ms_id_coverage.ipynb` is preserved. Its previously printed aggregate results are not used as per-compound evidence.

## Run locally

Python 3.10 or 3.11 is recommended. From this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app/app.py
```

The local build has prepared data in `data/processed/`. A deployment also includes the small `deployment/evidence.zip` bundle, unpacked and verified automatically when local tables are absent. Raw workbooks, PDFs and the large MassSpecGym source are ignored by Git. The default local chat and chemistry tools run offline once dependencies are installed.

## Chat, agent and data uploads

**Chat** is the default entry point, with conversation history, prompt examples, follow-up compound context, source results and an **Open compound on map** action. Local mode answers focused evidence questions without a model or API key. It is explicitly labeled a local assistant, not an LLM agent.

An optional OpenAI Responses tool-calling agent is implemented. Add `OPENAI_API_KEY` and `OPENAI_MODEL` to server environment/secrets, then enable **Use AI agent** in Chat. It uses local read-only compound, evidence, SMILES and dataset-summary tools first. For an explicit PubChem/property request, or one unresolved local identity, it can also fetch a small live PubChem PUG REST property record. PubChem candidates stay separate from local measured evidence and never replace the chemistry rules or predict RRFs/toxicity. No real API credentials are bundled, and live model behavior needs account configuration.

In **Data & sources**, upload **CSV or XLSX** target compounds and map name/CAS/SMILES columns. This changes chat and map targets for your session while retaining the reference layers. Default runtime data is **Parquet**, prepared from the original XLSX/PDF/CSV/TSV sources. Permanent reference/default changes use the preparation pipeline and a new deployment bundle.

See [DEPLOYMENT.md](DEPLOYMENT.md) for Vercel container deployment, alternative Streamlit Community Cloud settings, optional agent secrets and updating datasets after deployment.

## What is implemented

- Every Borton source row is searchable; every resolved structure is plotted, with no downsampling. Unresolved and conflicting rows remain in the compound table.
- Interactive 2D/3D fingerprint PCA and molecular-descriptor views, with CLAP overlays and selected-neighbour connections. Click a point in 2D or select a table row to inspect its evidence.
- Search by name, CAS, source ID or InChIKey, plus an explicit SMILES input mode. Known CAS queries use identifier boundaries, not suffix matching.
- Separate GC/MS, LC/MS positive and LC/MS negative assessments. Exact RRF values at 5, 10 and 20 µg/mL remain separate from non-detection, below-DL and not-reported outcomes.
- Adjustable Tanimoto cutoff, descriptor range margin and top 3/5 CLAP neighbours. Ranges are calculated separately for the measured panel in each method.
- Source-linked evidence for Stavitskaya chemical classes, PlastChem hazard scores, polymer associations, functions and uses, and actual MassSpecGym reference-spectrum coverage.
- Chat interface with local evidence tools, an optional OpenAI tool-calling agent and session CSV/XLSX target uploads.
- Filters, compound CSV export, detailed evidence JSON export, file SHA-256 provenance and in-app methodology/source pages.

## Prepare the real source data

The inputs for this build are documented in [DATA_SOURCES.md](DATA_SOURCES.md). No synthetic sample records are used.

```bash
python scripts/prepare_data.py \
  --background "../Relevant BAckground info" \
  --draft "../first draft of prototype" \
  --massspecgym data/raw/MassSpecGym.tsv
```

The background directory must contain the Borton, Stavitskaya and PlastChem workbooks and all three FDA RRF PDFs. The draft directory must contain `sample_data/clap_compounds.csv` and `sample_data/structures_clap.csv`. The script audits parsed FDA RRF bounds against the supplied extraction and fails on disagreements rather than silently replacing them.

Optional MassSpecGym download, using the public 262 MB spectrum table rather than the multi-gigabyte training/candidate data:

```bash
mkdir -p data/raw
curl -L --fail --retry 2 \
  https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/MassSpecGym.tsv \
  -o data/raw/MassSpecGym.tsv
```

Without that file, preparation still works and the interface says MassSpecGym is not loaded. The local assistant and Explore workspace do not perform public lookups. Only the enabled AI agent can make a bounded PubChem PUG REST request when the question needs it.

## Evidence rules

| Status | Meaning, separately for each method |
|---|---|
| Measured | An exact identity or explicitly labeled CAS/source-record match has at least one numeric FDA RRF. |
| Represented | A measured CLAP neighbour meets the cutoff and the query lies within every descriptor range of that method's measured panel. |
| Unsupported | No neighbour/range support, or the exact compound was not detected / below DL under the FDA conditions. |
| Unresolved | A usable structure is missing and no direct numeric record is available. |

Measured does not guarantee reliability under other experimental conditions. Represented is an exploratory operating rule, not a validated analytical conclusion. An exact non-detection is never promoted to Represented by similar compounds.

The full, original InChIKey is retained for exact identity. RDKit cleanup, largest organic fragment selection and neutralization are used for similarity and descriptor calculation only. CAS bridges require valid checksums; conflicting structures are withheld. A cleaned parent is never silently treated as the measured salt or stereoisomer.

PCA is an unsupervised visualization, not a response model. It is computed on unique standardized 2048-bit Morgan fingerprints (radius 2), and the projection is reused for every source row and custom query. Neighbours and coverage always use the full fingerprints, never distances on the displayed map.

## Review and extend structures

Add hand-reviewed rows to `data/manual/structure_overrides.csv` with `borton_id,smiles,review_note`, then rerun preparation. Stable IDs such as `B00001` follow the original Borton sheet order. Duplicate IDs, invalid SMILES and missing review notes fail validation.

`data/manual/clap_structure_exclusions.csv` quarantines the unverified RM84 structure. The FDA notes that its commercial material is a blend. Its method records remain accessible by CLAP code, while that structure is excluded from the map, reference similarity panels and Borton bridges.

## Important limits

- The map cannot place entries without structures. The searchable full Borton list is larger than the mapped subset. Offline CAS bridges are a lower bound; no mass PubChem/name-resolution job has been run.
- Stavitskaya's supplied supplement contains Tier 1/2 classes, not a per-compound binary toxicity label. Membership is not labeled toxic.
- PlastChem hazard scores retain their reported values, including suspected/inconclusive scores. Missing values are not treated as safe. Polymer values distinguish released (2), present (1), used (0.5), inconclusive (0.25), and not detected (0).
- MassSpecGym matching uses the first InChIKey block, which does not distinguish stereoisomers. A library reference exists; no user spectrum has been matched and GC-EI coverage is not assessed by this layer.
- NIST's raw polymer extraction runs are not a curated compound-to-polymer table and are not integrated. PlastChem associations do not establish the origin of a particular sample.
- Natural occurrence is unknown: the supplied PlastChem workbook has no field that establishes it. Bioplastics are not used as a proxy.
- Source structure tables and trade-name/mixture assignments still require scientific review. The prototype does not perform a toxicological or clinical safety assessment.

## Validate

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Checks cover CAS boundaries/checksums, salt identity, conflict handling, source-row preservation, method-specific numeric/non-detection outcomes, measured-only descriptor panels, spectral polarity, PDF parsing and Streamlit interactions. Source integration tests skip when the prepared data are unavailable.

## Repository layout

```text
app/              Streamlit interface and CSS
elspace/          Chemistry, evidence rules and Plotly visualization
scripts/          Reproducible source preparation
data/raw/         Optional source downloads (ignored)
data/interim/     Intermediate data (ignored)
data/manual/      Reviewed overrides and exclusions
data/processed/   Local Parquet tables, PCA projection and manifest (ignored)
tests/            Scientific-rule regressions and app interaction checks
deployment/       Small, checksummed default evidence bundle for Cloud
notebooks/        Existing team notebook
```

## Data terms

The software licence is still a team decision. Check each source's redistribution terms before committing or publishing raw or processed datasets. Raw source files remain ignored; Cloud deployment uses only the prepared evidence bundle.
