from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elspace.chemistry import DESCRIPTORS, fingerprint_matrix, molecule_image, structure
from elspace.bootstrap import ensure_dataset
from elspace.datasets import build_target_store, read_upload, upload_digest
from elspace.evidence import EvidenceStore, METHODS, OUTCOMES, search_compounds
from elspace.plots import chemical_map
from elspace.ui_chat import render_chat

st.set_page_config(page_title="E&L Atlas · Chemical space", page_icon=":material/science:", layout="wide")
st.html(ROOT / "app/styles.css")
DATA = ensure_dataset(ROOT / "data/processed", ROOT / "deployment/evidence.zip")


@st.cache_resource(show_spinner="Loading the source evidence…")
def load_store(version: int):
    return EvidenceStore.load(DATA)


@st.cache_data(show_spinner="Checking method-specific coverage…")
def coverage_catalog(version: str, method: str, cutoff: float, margin: float, _store):
    return _store.enrich_catalog(method, cutoff, margin)


def choose_example(cas: str):
    st.session_state["search"] = cas
    st.session_state["input_mode"] = "Dataset search"
    st.session_state.pop("custom_target", None)


def safe_csv(frame: pd.DataFrame) -> bytes:
    exported = frame.copy()
    for column in exported.select_dtypes(include=["object", "string"]):
        exported[column] = exported[column].map(lambda value: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")) else value)
    return exported.to_csv(index=False).encode("utf-8")


if not (DATA / "manifest.json").exists():
    st.title("E&L Atlas")
    st.info("Prepare the supplied datasets before opening the atlas.")
    st.code("python scripts/prepare_data.py\nstreamlit run app/app.py", language="bash")
    st.stop()

base_version = (DATA / "manifest.json").stat().st_mtime_ns
base_store = load_store(base_version)
if st.session_state.get("uploaded_base_version") != base_version:
    st.session_state.pop("uploaded_store", None)
    st.session_state.pop("uploaded_dataset_version", None)
store = st.session_state.get("uploaded_store", base_store)
version = str(base_version) + st.session_state.get("uploaded_dataset_version", "")
manifest = store.manifest
counts = manifest["counts"]
target_source = manifest.get("target_source", "Borton")
target_counts = manifest.get("active_targets", {"label": "Borton", "rows": counts["borton_rows"], "resolved": counts["borton_resolved"]})
if "pending_query" in st.session_state:
    st.session_state["search"] = st.session_state.pop("pending_query")
if "pending_input_mode" in st.session_state:
    st.session_state["input_mode"] = st.session_state.pop("pending_input_mode")
if "pending_workspace" in st.session_state:
    st.session_state["workspace"] = st.session_state.pop("pending_workspace")

with st.sidebar:
    st.title("E&L Atlas")
    st.caption("Analytical coverage explorer")
    page = st.radio("Workspace", ["Chat", "Explore", "Data & sources", "Methodology"], key="workspace", label_visibility="collapsed")
    st.divider()
    with st.expander("Coverage rules", expanded=page != "Chat"):
        method = st.selectbox("Analytical method", list(METHODS), format_func=METHODS.get)
        cutoff = st.slider("Tanimoto cutoff", 0.0, 1.0, 0.50, 0.05)
        margin = st.slider("Descriptor margin (%)", 0, 25, 0, 5) / 100
        neighbours_count = st.radio("CLAP neighbours", [3, 5], index=1, horizontal=True)
        st.caption("Exploratory operating rules. Similarity and descriptor ranges do not establish detectability or reliable quantification.")
    st.divider()
    st.caption(f"{counts['clap_rows']} CLAP records · {counts['clap_resolved']} with structures")
    st.caption("Morgan radius 2 · 2048 bits\n\nNo response factor is predicted.")

if page == "Chat":
    render_chat(store, cutoff, margin, neighbours_count, version)
    st.stop()

if page == "Data & sources":
    st.title("Data & sources")
    st.subheader("Active target dataset")
    st.write(f"**{target_counts['label']}** · {target_counts['rows']:,} rows · {target_counts['resolved']:,} with usable structures")
    st.caption("The deployed reference data is stored in Parquet tables. Upload CSV or XLSX target compounds to replace the target map and chat dataset for this browser session. FDA CLAP and supporting evidence remain available as reference layers.")
    uploaded = st.file_uploader("Upload target compounds", type=["csv", "xlsx"], key="target_upload")
    if uploaded is not None:
        try:
            payload = uploaded.getvalue()
            sheet = None
            if uploaded.name.casefold().endswith(".xlsx"):
                sheets = pd.ExcelFile(BytesIO(payload)).sheet_names
                sheet = st.selectbox("Worksheet", sheets, index=sheets.index("Extractables") if "Extractables" in sheets else 0)
            frame = read_upload(payload, uploaded.name, sheet)
            st.dataframe(frame.head(5), hide_index=True, width="stretch")
            columns = frame.columns.tolist()
            optional = ["Not provided", *columns]
            mappings = st.columns(3)
            name_default = next((index for index, column in enumerate(columns) if "name" in column.casefold()), 0)
            cas_default = next((index for index, column in enumerate(optional) if "cas" in column.casefold()), 0)
            smiles_default = next((index for index, column in enumerate(optional) if "smiles" in column.casefold()), 0)
            name_column = mappings[0].selectbox("Compound name column", columns, index=name_default)
            cas_column = mappings[1].selectbox("CAS column", optional, index=cas_default)
            smiles_column = mappings[2].selectbox("SMILES column", optional, index=smiles_default)
            st.caption("Choose a name column and at least CAS or SMILES. CAS identifiers are matched offline to loaded structures. Invalid SMILES and conflicting identities stay flagged. Limit: 20,000 rows / 25 MB.")
            if st.button("Use this target dataset", type="primary"):
                with st.spinner("Validating structures and preparing the uploaded target map…"):
                    digest = upload_digest(payload)
                    uploaded_store = build_target_store(base_store, frame, uploaded.name, digest, name_column,
                                                        None if cas_column == "Not provided" else cas_column,
                                                        None if smiles_column == "Not provided" else smiles_column, DATA / "projection.npz")
                st.session_state["uploaded_store"] = uploaded_store
                st.session_state["uploaded_base_version"] = base_version
                st.session_state["uploaded_dataset_version"] = digest + json.dumps([uploaded.name, sheet, name_column, cas_column, smiles_column])
                st.session_state["pending_query"] = "UP00001"
                st.session_state["pending_input_mode"] = "Dataset search"
                st.session_state.pop("last_map_selection", None)
                st.rerun()
        except Exception as error:
            st.error("Could not apply the target dataset: " + str(error))
    if target_source != "Borton" and st.button("Restore Borton targets"):
        for key in ["uploaded_store", "uploaded_base_version", "uploaded_dataset_version", "chat_messages", "chat_context_id"]:
            st.session_state.pop(key, None)
        st.session_state["pending_query"] = "80-05-7"
        st.rerun()
    st.caption("Session uploads are temporary and do not change another visitor's data. To permanently replace the reference build, rerun preparation, rebuild deployment/evidence.zip and update the deployed repository.")
    st.divider()
    st.write("This build uses the supplied source data. No synthetic sample compounds, RRFs, hazards or spectral matches are generated.")
    st.dataframe(pd.DataFrame([
        {"Layer": "Borton Appendix B", "Records": str(counts["borton_rows"]), "What it supplies": "Names and reported identifiers; not measured response"},
        {"Layer": "FDA CLAP", "Records": str(counts["clap_rows"]), "What it supplies": "Method-specific RRFs, non-detection and below-DL outcomes"},
        {"Layer": "Stavitskaya supplement", "Records": str(counts["stavitskaya_rows"]), "What it supplies": "Tier 1 and Tier 2 chemical classes; no binary toxicity endpoint"},
        {"Layer": "PlastChem v1.01", "Records": str(counts["plastchem_rows"]), "What it supplies": "Reported hazard scores, polymer associations, functions and uses"},
        {"Layer": "MassSpecGym", "Records": str(manifest["massspecgym"].get("spectra", "Not loaded")), "What it supplies": "Reference MS/MS availability; connectivity-level matching"},
        {"Layer": "NIST polymer extracts", "Records": "Not integrated", "What it supplies": "Raw extraction runs; compound-to-polymer annotations are not available here"},
    ]), hide_index=True, width="stretch")
    st.subheader("Identity audit")
    st.write(f"Of {counts['borton_rows']:,} Borton rows, {counts['borton_resolved']:,} have usable structures, {counts['borton_conflicts']:,} have conflicting source identities, and {counts['borton_unresolved']:,} remain unresolved. The map includes every resolved row; all rows remain searchable in the compound table.")
    st.caption("Source rows are preserved. Duplicate PlastChem evidence rows are retained and aggregated only for display. Borton structures are joined through checksum-valid CAS identifiers already present in the supplied structure-bearing datasets.")
    for name, url in manifest["source_urls"].items():
        st.link_button(name, url)
    with st.expander("File provenance and checksums"):
        st.dataframe(pd.DataFrame(manifest["files"]), hide_index=True, width="stretch")
    st.download_button("Download source manifest", json.dumps(manifest, indent=2), "source_manifest.json", "application/json")
    st.stop()

if page == "Methodology":
    st.title("How evidence is assessed")
    st.markdown("""
1. **Keep identity and similarity separate.** Exact evidence uses the full source InChIKey. CAS fallback is labeled; conflicting identities require review. RDKit cleanup, largest organic fragment selection and neutralization are used for fingerprints and descriptors only. A neutralized parent is never silently treated as a measured salt.
2. **Compare structures in the original fingerprint space.** Morgan fingerprints use radius 2 and 2048 bits. Tanimoto similarity ranks CLAP neighbours. The projection never determines a neighbour or evidence status.
3. **Check each analytical method separately.** A numeric RRF for the exact identity is Measured. A non-exact compound is Represented only when a CLAP compound with a numeric RRF in that method meets the cutoff and all seven descriptors fit that method's measured panel. Exact non-detection or below-DL evidence is preserved as Unsupported. Missing structure is Unresolved.
4. **Keep missing evidence visible.** Not detected, below the detection limit and not reported are different outcomes. Absence from a dataset does not establish safety, absence from nature, or inability to identify a compound.
5. **Read supporting layers as reported.** Stavitskaya supplies chemical classes. PlastChem supplies hazard scores and graded polymer associations. MassSpecGym supplies spectrum availability at connectivity level, without stereochemical confirmation. NIST raw runs have not been interpreted.
""")
    st.subheader("Map interpretation")
    variance = manifest["projection"]["explained_variance_ratio"]
    st.write(f"PCA is computed on unique standardized Morgan fingerprints, then every resolved source row is placed on it. PC1 and PC2 retain {sum(variance[:2]):.1%} of fingerprint variance; three axes retain {sum(variance):.1%}. Nearby points can be dissimilar in the full fingerprint space. Overlapping source rows can share coordinates.")
    st.write("The alternate descriptor view plots molecular weight against Crippen logP; its third axis is TPSA. Range padding extends each panel min/max by the selected percentage of its span. A zero-width range stays fixed.")
    st.info("A measured RRF applies to the published FDA conditions. It is not a guarantee for a different method, instrument, matrix or concentration. Represented is an exploratory label, not a validated analytical claim.")
    st.stop()

catalog = coverage_catalog(version, method, cutoff, margin, store)
borton = catalog[catalog.source.eq(target_source)]

st.title("Chemical space")
st.caption("Extractables & leachables · Analytical coverage and source evidence")
metrics = st.columns(4)
metrics[0].metric("Borton records" if target_source == "Borton" else "Uploaded target rows", f"{target_counts['rows']:,}", help="Original source rows; repeated identities are preserved.")
metrics[1].metric("Borton rows on the map" if target_source == "Borton" else "Target rows on the map", f"{target_counts['resolved']:,}", help="Only resolved structures can be mapped. All rows remain in the table.")
metrics[2].metric(f"Measured · {METHODS[method]}", f"{borton.coverage.eq('Measured').sum():,}")
metrics[3].metric("Structures with MS/MS", f"{borton.loc[borton.ms_reference.eq('Reference spectra available'), 'inchikey'].nunique():,}" if manifest['massspecgym']['loaded'] else "Not loaded",
                  help="Unique full source InChIKeys among Borton rows with connectivity-level MassSpecGym matches.")

search_left, search_right = st.columns([2, 3], vertical_alignment="bottom")
with search_left:
    input_mode = st.radio("Input", ["Dataset search", "SMILES"], horizontal=True, key="input_mode", label_visibility="collapsed")
    query = st.text_input("Find a compound" if input_mode == "Dataset search" else "Enter a SMILES structure",
                          placeholder="Name, CAS, InChIKey or Borton ID" if input_mode == "Dataset search" else "e.g. CC(C)(C)c1ccc(O)cc1", key="search")
with search_right:
    examples = st.columns(3)
    for column, (name, cas) in zip(examples, [("Bisphenol A", "80-05-7"), ("Caprolactam", "105-60-2"), ("Benzoic acid", "65-85-0")]):
        column.button(name, on_click=choose_example, args=(cas,), width="stretch")

if input_mode == "SMILES":
    candidates = catalog.iloc[:0]
else:
    candidates = search_compounds(catalog, query)

target = None
if len(candidates):
    options = candidates.compound_id.tolist()[:300]
    if st.session_state.get("compound_pick") not in options:
        previous = st.session_state.get("selected_compound")
        st.session_state["compound_pick"] = previous if previous in options else options[0]
    lookup = candidates.set_index("compound_id")
    selected_id = st.selectbox("Matching source records", options, key="compound_pick",
                               format_func=lambda identifier: f"{lookup.loc[identifier, 'name']} · {identifier} · {lookup.loc[identifier, 'source']}")
    st.session_state["selected_compound"] = selected_id
    target = catalog[catalog.compound_id.eq(selected_id)].iloc[0].to_dict()
    if len(candidates) > 300:
        st.caption(f"Showing the first 300 of {len(candidates):,} matches. Refine your search to see the rest.")
else:
    properties = structure(query.strip()) if input_mode == "SMILES" else None
    if properties:
        target = {"compound_id": "QUERY", "name": "Custom SMILES query", "cas_raw": "", "valid_cas": "", "source": "User input",
                  "structure_status": "Resolved", "structure_source": "User-supplied SMILES", "structure_note": "", **properties}
        with np.load(DATA / "projection.npz") as projection_data:
            position = np.einsum("ij,kj->ik", fingerprint_matrix([properties["smiles_std"]]) - projection_data["center"], projection_data["axes"], optimize=False)
        target.update(dict(zip(["map_x", "map_y", "map_z"], position[0])))
        st.caption("Valid SMILES: evaluated locally against the loaded reference datasets.")
    else:
        st.warning("Enter a valid SMILES structure." if input_mode == "SMILES" else "No local record matches that query. Enter another name or CAS, or switch to SMILES input.")

with st.expander("Map filters", expanded=False):
    filters = st.columns(4)
    statuses = filters[0].multiselect("Coverage status", ["Measured", "Represented", "Unsupported", "Unresolved"], default=["Measured", "Represented", "Unsupported", "Unresolved"])
    hazard_filter = filters[1].selectbox("Reported hazard evidence", ["All records", "Reported flags", "No positive flags in source", "No source match"])
    ms_filter = filters[2].selectbox("MassSpecGym", ["All records", "Reference spectra available", "No reference in loaded MassSpecGym"])
    show_clap = filters[3].checkbox("Overlay FDA CLAP panel", value=True)

filtered = borton[borton.coverage.isin(statuses)]
if hazard_filter != "All records":
    filtered = filtered[filtered.hazard_evidence.eq(hazard_filter)]
if ms_filter != "All records":
    filtered = filtered[filtered.ms_reference.eq(ms_filter)]

map_tab, table_tab = st.tabs(["Chemical-space map", "Compound table"])
neighbours = store.neighbours(target, neighbours_count) if target else pd.DataFrame()
assessments = store.assess(target, cutoff, margin) if target else []
with map_tab:
    map_column, card_column = st.columns([3, 1], gap="large")
    with map_column:
        settings = st.columns([2, 2, 1])
        projection = settings[0].selectbox("Map axes", ["Fingerprint PCA", "Molecular descriptors"])
        color_label = settings[1].selectbox("Colour by", ["Analytical coverage", "Reported hazard evidence", "MS/MS reference"])
        color_by = {"Analytical coverage": "coverage", "Reported hazard evidence": "hazard_evidence", "MS/MS reference": "ms_reference"}[color_label]
        dimensions = settings[2].selectbox("Dimensions", ["2D", "3D"])
        plotted = filtered[filtered.smiles_std.ne("")]
        figure = chemical_map(plotted, store.reference, target or {}, neighbours, color_by, dimensions, projection, show_clap)
        chart = st.plotly_chart(figure, width="stretch", theme=None, key="chemical_space", on_select="rerun" if dimensions == "2D" else "ignore",
                               selection_mode="points", config={"displaylogo": False, "scrollZoom": True})
        if dimensions == "2D" and chart.selection.points:
            picked = chart.selection.points[-1].get("customdata", [None])[0]
            signature = str([(point.get("curve_number"), point.get("point_index"), point.get("customdata")) for point in chart.selection.points])
            if picked and picked != "QUERY" and signature != st.session_state.get("last_map_selection"):
                st.session_state["last_map_selection"] = signature
                st.session_state["selected_compound"] = picked
                st.session_state["pending_query"] = picked
                st.session_state["pending_input_mode"] = "Dataset search"
                st.rerun()
        st.caption(f"{len(plotted):,} target rows shown · {len(filtered) - len(plotted):,} filtered rows lack usable structures. Click a point in 2D to inspect it. Cross = selected compound; diamonds = CLAP. Neighbour lines use full-fingerprint similarity.")
        st.caption("PCA is a visual summary, not a similarity metric. The map includes every resolved target row; unresolved and conflicting rows stay in the table.")
    with card_column:
        st.subheader("Compound")
        if target:
            st.write(target["name"])
            st.caption(f"{target['compound_id']} · {target['source']}")
            st.caption("CAS: " + (str(target["cas_raw"]) or "Not supplied"))
            if target.get("smiles_raw"):
                st.image(molecule_image(target["smiles_raw"]), width="stretch")
            else:
                st.info("Structure " + target["structure_status"].lower())
            for result in assessments:
                st.write(f"**{result['label']}**")
                st.badge(result["status"], color="blue" if result["status"] == "Measured" else "gray")
            st.caption("Status is assessed separately for each analytical method.")
        else:
            st.caption("Select a source record or enter a valid SMILES.")

with table_tab:
    st.subheader(f"{len(filtered):,} target source records")
    columns = ["compound_id", "name", "cas_raw", "structure_status", "coverage", "nearest_measured_similarity", "chemical_class", "hazard_evidence", "ms_reference"]
    table_selection = st.dataframe(filtered[columns], hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                                   key="compound_table", column_config={"nearest_measured_similarity": st.column_config.NumberColumn("Nearest measured Tanimoto", format="%.3f")})
    if table_selection.selection.rows:
        selected_row = filtered.iloc[table_selection.selection.rows[0]]
        signature = (tuple(filtered.compound_id), selected_row.compound_id)
        if st.session_state.get("last_table_selection") != signature:
            st.session_state["last_table_selection"] = signature
            st.session_state["pending_query"] = selected_row.compound_id
            st.session_state["pending_input_mode"] = "Dataset search"
            st.rerun()
    st.download_button("Export filtered compounds", safe_csv(filtered), "target_compounds_filtered.csv", "text/csv")

if target:
    st.divider()
    st.subheader("Evidence for " + target["name"])
    evidence_tab, neighbour_tab, descriptor_tab, provenance_tab = st.tabs(["Source evidence", "CLAP neighbours", "Descriptor ranges", "Identity & export"])
    with evidence_tab:
        st.write("**Measured response · FDA CLAP**")
        rows = []
        for result in assessments:
            direct = result["exact_records"]
            if direct.empty:
                rows.append({"Method": result["label"], "Evidence": result["status"], "FDA outcome": "No exact record", "RRF at 5 µg/mL": "—", "RRF at 10 µg/mL": "—", "RRF at 20 µg/mL": "—"})
            for _, record in direct.iterrows():
                rows.append({"Method": result["label"], "Evidence": result["status"], "FDA outcome": OUTCOMES[record.outcome],
                             **{f"RRF at {concentration} µg/mL": record.get(f"rrf_{concentration}") if pd.notna(record.get(f"rrf_{concentration}")) else "—" for concentration in [5, 10, 20]},
                             "CLAP code": record.clap_code, "Match": result["match_basis"], "Internal standard": bool(record.get("is_internal_standard", False)), "PDF page": record.get("source_page")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("n.d. = not detected; below DL = below the detection limit. RRFs are measured under the published FDA conditions. Missing values are never converted to zero. Internal standards define the reference response in their corresponding methods.")
        for result in assessments:
            with st.expander(f"{result['label']} · {result['status']} — why?"):
                st.write(result["reason"])
                st.write(f"Measured reference panel with structures: {result['panel_size']} compounds.")
                if result["similarity"] is not None:
                    st.write(f"Nearest measured reference: {result['closest_clap']} · Tanimoto {result['similarity']:.3f}")
        first, second = st.columns(2, gap="large")
        with first:
            st.write("**Chemical classes & reported hazards**")
            stav, stav_basis = store.layer(target, "stavitskaya")
            if len(stav):
                st.dataframe(stav[["tier1", "tier2"]].drop_duplicates(), hide_index=True, width="stretch")
                st.caption("Stavitskaya supplement · " + stav_basis)
            else:
                st.caption("Stavitskaya: " + stav_basis)
            st.caption("The supplied Stavitskaya supplement reports chemical classes, not per-compound toxicity outcomes. Membership alone does not mean toxic.")
            plastics, plastic_basis = store.layer(target, "plastics")
            hazard_rows = []
            for _, row in plastics.iterrows():
                for name, score in json.loads(row.hazards).items():
                    hazard_rows.append({"Source ID": row.plastchem_id, "Reported field": name, "Source score": score})
            if hazard_rows:
                st.dataframe(pd.DataFrame(hazard_rows).drop_duplicates(), hide_index=True, width="stretch")
            else:
                st.caption("No reported PlastChem hazard values for this identity.")
            st.caption("PlastChem · " + plastic_basis + ". Scores are shown as reported; blanks mean unavailable. Positive scores can include suspected or inconclusive assessments. This is not a toxicity assessment.")
        with second:
            st.write("**MS/MS reference spectra · MassSpecGym**")
            spectrum = store.spectrum(target)
            st.write(spectrum["status"])
            if "n_spectra" in spectrum:
                st.write(f"{int(spectrum['n_spectra']):,} spectra · {int(spectrum['positive']):,} positive · {int(spectrum['negative']):,} negative · {int(spectrum['unknown']):,} unspecified")
                st.caption("Adducts: " + spectrum["adducts"])
                st.caption("Instruments: " + spectrum["instruments"])
                st.caption("Example spectrum IDs: " + spectrum["example_identifiers"])
            st.caption(spectrum["match_basis"] + ". A reference spectrum supports identification; no experimental spectrum has been compared, and GC-EI coverage is not assessed by this layer.")
            st.write("**Polymer associations · PlastChem**")
            polymer_rows = []
            meanings = {2.0: "Released", 1.0: "Present", 0.5: "Used", 0.25: "Inconclusive", 0.0: "Not detected"}
            for _, row in plastics.iterrows():
                for polymer, score in json.loads(row.polymers).items():
                    polymer_rows.append({"Polymer": polymer, "Reported association": meanings.get(score, str(score)), "Source ID": row.plastchem_id})
            if polymer_rows:
                st.dataframe(pd.DataFrame(polymer_rows).drop_duplicates(), hide_index=True, width="stretch")
            else:
                st.caption("No polymer association reported for this identity.")
            functions = sorted(set(plastics.functions) - {""})
            applications = sorted(set(plastics.applications) - {""})
            if functions:
                st.write("Functions: " + " | ".join(functions).replace("_", " "))
            if applications:
                st.write("Other applications: " + " | ".join(applications))
            st.caption("Polymer association does not establish the origin of a particular sample. NIST raw polymer runs are not integrated.")
            st.write("**Natural occurrence**")
            st.caption("Unknown. The supplied PlastChem workbook has no natural-occurrence field. A bioplastic association is not evidence of natural occurrence.")
    with neighbour_tab:
        if neighbours.empty:
            st.info("Resolve the compound structure to find CLAP neighbours.")
        else:
            neighbour_rows = []
            for _, neighbour in neighbours.iterrows():
                record = {"CLAP": neighbour.clap_code, "Name": neighbour['name'], "CAS": neighbour.cas_raw, "Tanimoto": neighbour.Tanimoto}
                for key, label in METHODS.items():
                    observations = store.methods[store.methods.clap_code.eq(neighbour.clap_code) & store.methods.method.eq(key)]
                    observation = observations.iloc[0]
                    numeric = pd.notna(observation.rrf_min)
                    record[label] = OUTCOMES[observation.outcome] + (f" · {observation.rrf_min:.3g}–{observation.rrf_max:.3g}" if numeric else "")
                neighbour_rows.append(record)
            st.dataframe(pd.DataFrame(neighbour_rows), hide_index=True, width="stretch", column_config={"Tanimoto": st.column_config.NumberColumn(format="%.3f")})
            st.caption("Top neighbours from the full fingerprint space. Exact identities are excluded; compounds sharing a cleaned parent can still be distinct identities. RRFs belong to the neighbours and are never assigned to the query.")
    with descriptor_tab:
        selected_method = st.selectbox("Descriptor reference panel", list(METHODS), format_func=METHODS.get)
        result = next(item for item in assessments if item["method"] == selected_method)
        if result["descriptor_check"].empty:
            st.info("Descriptor checks require a usable structure and measured references.")
        else:
            st.dataframe(result["descriptor_check"], hide_index=True, width="stretch")
            st.caption(f"Panel: {result['panel_size']} CLAP structures with numeric RRFs in {result['label']}. Range padding: {margin:.0%} of each descriptor span. Descriptors use the cleaned, neutralized parent.")
    with provenance_tab:
        st.write("Structure status: " + target["structure_status"])
        st.write("Structure source: " + (target.get("structure_source") or "Not resolved"))
        if target.get("structure_note"):
            st.write(target["structure_note"])
        if target.get("inchikey"):
            st.text("Source identity: " + target["inchikey"])
            st.text("Similarity identity: " + target["inchikey_std"])
            st.code(target["smiles_raw"], language=None)
        report = {"compound": {key: value for key, value in target.items() if not isinstance(value, float) or np.isfinite(value)},
                  "rules": {"cutoff": cutoff, "descriptor_margin": margin, "fingerprint": "Morgan r=2, 2048 bits"},
                  "methods": [{**{key: value for key, value in item.items() if key not in ["exact_records", "descriptor_check"]},
                               "exact_records": item["exact_records"].replace({np.nan: None}).to_dict("records"),
                               "descriptor_check": item["descriptor_check"].replace({np.nan: None}).to_dict("records")} for item in assessments],
                  "plastchem": {"match_basis": plastic_basis, "records": plastics.to_dict("records")},
                  "stavitskaya": {"match_basis": stav_basis, "records": stav.to_dict("records")},
                  "massspecgym": store.spectrum(target), "source_urls": manifest["source_urls"],
                  "limitations": ["No predictive model", "No toxicity assessment", "No sample spectrum matching", "Natural occurrence unknown", "NIST raw runs not integrated"]}
        st.download_button("Download compound evidence", json.dumps(report, indent=2, default=str), target["compound_id"] + "_evidence.json", "application/json")
