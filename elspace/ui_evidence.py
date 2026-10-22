from __future__ import annotations

import pandas as pd
import streamlit as st

from elspace.chemistry import molecular_properties, molecule_data_uri
from elspace.plots import spectrum_plot


def render_properties(target: dict, compact: bool = False):
    if not target.get("smiles_raw") or target.get("structure_status") != "Resolved":
        st.info("A reviewed structure is needed to draw the molecule and calculate properties.")
        return
    if compact:
        st.image(molecule_data_uri(target["smiles_raw"]), width="stretch")
        with st.expander("Molecular properties"):
            property_table(target)
    else:
        drawing, properties = st.columns([1, 1], gap="large")
        with drawing:
            st.image(molecule_data_uri(target["smiles_raw"]), width="stretch")
        with properties:
            property_table(target)
        st.write("Source SMILES")
        st.code(target["smiles_raw"], language=None)
        st.caption("The drawing depicts the source SMILES, including source fragments. Properties below are RDKit calculations on the cleaned, neutralized parent used for similarity.")


def property_table(target: dict):
    values = molecular_properties(target["smiles_raw"])
    st.dataframe(pd.DataFrame([{"Property": name, "RDKit value": str(value)} for name, value in values.items()]),
                 hide_index=True, width="stretch")
    st.caption("RDKit · cleaned, neutralized parent. Crippen logP is not Molinspiration miLogP. No molecular volume or uncomputed property is invented.")


def render_spectra(store, target: dict, key: str):
    summary = store.spectrum(target)
    st.write(summary["status"])
    peaks = store.reference_spectra(target)
    st.caption("Real MassSpecGym reference peaks, matched by the InChIKey connectivity block—not stereochemical identity or an experimental sample match. Intensities are normalized to the base peak = 100% for display.")
    if "n_spectra" in summary:
        st.write(f"{int(summary['n_spectra']):,} source spectra · {len(peaks):,} representative spectra bundled for plotting")
        st.caption(store.manifest["massspecgym"].get("peak_spectra", {}).get("selection", "Peak data has not been bundled."))
    positive, negative, gc = st.tabs(["LC-MS/MS positive", "LC-MS/MS negative", "GC-EI"])
    for tab, polarity, label in [(positive, "positive", "LC-MS/MS positive"), (negative, "negative", "LC-MS/MS negative")]:
        with tab:
            selected = peaks[peaks.polarity.eq(polarity)] if len(peaks) else peaks
            if selected.empty:
                st.info(f"No {polarity}-ion peak spectrum is bundled for this identity. This does not establish non-detectability.")
                continue
            records = selected.to_dict("records")

            def spectrum_label(index):
                row = records[index]
                energy = row.get("collision_energy")
                energy_text = f"{energy:g}" if pd.notna(energy) else "not reported"
                return f"{row['identifier']} · {row['adduct']} · CE {energy_text} · {row['instrument_type']}"

            choice = st.selectbox("Reference spectrum", list(range(len(records))), format_func=spectrum_label, key=f"{key}_{polarity}")
            spectrum = records[choice]
            st.plotly_chart(spectrum_plot(spectrum, label + " · " + spectrum["adduct"]), theme=None, width="stretch",
                            key=f"{key}_{polarity}_plot", config={"displaylogo": False})
            st.caption(f"Source ID: {spectrum['identifier']} · Instrument: {spectrum['instrument_type']} · Precursor m/z: {spectrum['precursor_mz']:.5f} · Formula: {spectrum['formula']}")
            st.caption("Collision energy is reproduced as reported; units and acquisition conditions are not inferred. The dashed line is the reported precursor m/z, not an invented fragment.")
            st.download_button("Download reference peak record", pd.DataFrame([spectrum]).to_json(orient="records", indent=2),
                               spectrum["identifier"] + ".json", "application/json", key=f"{key}_{polarity}_download")
    with gc:
        st.info("GC-EI reference peaks are not supplied by the loaded MassSpecGym layer. The example image's 70 eV spectrum cannot be reconstructed from SMILES or LC-MS/MS peaks. Add a licensed GC-EI reference library to enable this panel.")
