from __future__ import annotations

from dataclasses import asdict
import json
import os

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from elspace.chat import EvidenceTools, agent_reply, local_reply
from elspace.ui_evidence import render_properties, render_spectra


def setting(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        return ""


def show_reply(reply: dict, key: str):
    st.markdown(reply["text"])
    st.caption(reply.get("engine", "Local evidence tools"))
    for index, evidence in enumerate(reply.get("evidence", [])):
        compound = evidence["compound"]
        with st.expander("Structure, molecular properties and reference spectra · " + compound["name"]):
            render_properties(compound)
            store = st.session_state.get("chat_reference_store")
            if store is not None:
                render_spectra(store, compound, f"{key}_spectra_{index}")
        with st.expander("Source results · " + compound["name"]):
            method_rows = []
            for method in evidence["methods"]:
                direct = method["exact_records"]
                for observation in direct or [{}]:
                    method_rows.append({"Method": method["label"], "Evidence": method["status"], "Match": method["match_basis"],
                                        "CLAP": observation.get("clap_code", "—"), "Outcome": observation.get("outcome", "No exact record"),
                                        "RRF 5 µg/mL": observation.get("rrf_5"), "RRF 10 µg/mL": observation.get("rrf_10"), "RRF 20 µg/mL": observation.get("rrf_20")})
            st.dataframe(pd.DataFrame(method_rows), hide_index=True, width="stretch")
            if evidence["neighbours"]:
                st.write("Closest CLAP structures")
                st.dataframe(pd.DataFrame(evidence["neighbours"]), hide_index=True, width="stretch")
            for layer, label in [("plastchem", "PlastChem source evidence"), ("stavitskaya", "Stavitskaya source evidence")]:
                st.write(label + " · " + evidence[layer]["match_basis"])
                if evidence[layer]["records"]:
                    st.dataframe(pd.DataFrame(evidence[layer]["records"]), hide_index=True, width="stretch")
            st.write("MassSpecGym")
            st.json(evidence["massspecgym"], expanded=False)
            st.download_button("Download this evidence", json.dumps(evidence, indent=2, default=str),
                               compound["compound_id"] + "_chat_evidence.json", "application/json", key=f"{key}_download_{index}")
        if st.button("Open compound on map", key=f"{key}_map_{index}"):
            if compound["source"] == "PubChem lookup":
                external = st.session_state.get("external_compounds", {}).get(compound["compound_id"])
                if external is None:
                    st.warning("Resolve this PubChem identity again before opening the map.")
                    continue
                st.session_state["pubchem_lookup"] = {"query": external["lookup_query"], "candidates": [external]}
                st.session_state["pending_query"] = external["lookup_query"]
                st.session_state["pending_input_mode"] = "PubChem lookup"
            elif compound["compound_id"] == "QUERY":
                st.session_state["pending_query"] = compound["smiles_raw"]
                st.session_state["pending_input_mode"] = "SMILES"
            else:
                st.session_state["pending_query"] = compound["compound_id"]
                st.session_state["pending_input_mode"] = "Dataset search"
            st.session_state["pending_workspace"] = "Explore"
            st.rerun()
    for index, candidate in enumerate(reply.get("candidates", [])):
        if st.button(f"{candidate['name']} · {candidate['compound_id']}", key=f"{key}_candidate_{index}"):
            st.session_state["queued_chat_prompt"] = "Check " + candidate["compound_id"]
            st.rerun()
    if reply.get("tools"):
        with st.expander("Evidence tools used"):
            st.json(reply["tools"], expanded=False)
    if reply.get("evidence"):
        source_urls = reply["evidence"][0]["source_urls"]
        st.caption("Sources: " + " · ".join(f"[{name}]({source_urls[name]})" for name in ["FDA CLAP", "Stavitskaya", "PlastChem", "MassSpecGym"]))


def render_chat(store, cutoff: float, margin: float, neighbours_count: int, version: str):
    if st.session_state.get("chat_dataset_version") != version:
        st.session_state["chat_dataset_version"] = version
        st.session_state["chat_messages"] = []
        st.session_state.pop("chat_context_id", None)
    messages = st.session_state.setdefault("chat_messages", [])
    st.session_state["chat_reference_store"] = store
    external = st.session_state.setdefault("external_compounds", {})
    api_key, model = setting("OPENAI_API_KEY"), setting("OPENAI_MODEL")
    configured = bool(api_key and model)
    with st.sidebar:
        st.divider()
        st.subheader("Chat")
        if st.button("New chat", width="stretch"):
            st.session_state["chat_messages"] = []
            st.session_state.pop("chat_context_id", None)
            st.rerun()
        if not configured:
            st.session_state["use_ai_agent"] = False
        use_agent = st.toggle("Use AI agent", disabled=not configured, key="use_ai_agent",
                              help="Requires OPENAI_API_KEY and OPENAI_MODEL in server secrets. When enabled, your messages and selected source evidence are sent to OpenAI.")
        st.caption("OpenAI agent available" if configured else "Local evidence tools active. AI agent needs server-side API credentials.")
        detail_level = st.radio("Answer detail", ["Detailed", "Concise"], horizontal=True)
        allow_pubchem = st.toggle("Look up unknown compounds in PubChem", value=True,
                                  help="Only the requested CAS/name/CID is sent to NCBI PubChem. Uploaded datasets are not sent. Disable for offline-only use.")
        st.caption("Classifier notebooks pending; no predictive model is active.")
    tools = EvidenceTools(store, cutoff, margin, neighbours_count, allow_pubchem, external, detail_level == "Detailed")
    st.title("Ask about your chemical data.")
    active = store.manifest.get("active_targets", {"label": "Borton", "rows": store.manifest["counts"]["borton_rows"]})
    st.caption(f"{active['label']} · {active['rows']:,} target rows · FDA CLAP, chemical classes, hazards, polymers and MS/MS references")
    st.caption("AI agent sends messages and selected evidence to OpenAI." if use_agent else "Local evidence assistant · No API key needed · No RRF or toxicity prediction")
    st.caption("Unknown CAS/name/CID queries can be sent to PubChem when enabled. External records stay separate from Borton; lookup is not proof of detectability.")
    if not messages:
        st.write("Check a compound, including a CAS absent from Borton; compare analytical methods; inspect structures and real reference spectra. For an unknown name, use ‘PubChem: compound name’. You can also provide ‘SMILES: …’ for offline analysis.")
        starters = st.columns(3)
        for column, label, question in zip(starters, ["Check Bisphenol A", "Compare methods", "Data sources"],
                                           ["Check Bisphenol A", "Compare the CLAP methods for Caprolactam", "What datasets are loaded?"]):
            if column.button(label, width="stretch"):
                st.session_state["queued_chat_prompt"] = question
    for index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                show_reply(message["reply"], f"history_{index}")
            else:
                st.markdown(message["text"])
    prompt = st.chat_input("Ask about a compound, CAS number, method or dataset", max_chars=4000)
    prompt = prompt or st.session_state.pop("queued_chat_prompt", None)
    if prompt:
        history = list(messages)
        messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Checking source evidence…"):
                if use_agent:
                    from openai import OpenAI, OpenAIError

                    try:
                        with OpenAI(api_key=api_key, timeout=25, max_retries=0) as client:
                            reply = agent_reply(tools, prompt, history, st.session_state.get("chat_context_id"), client, model)
                    except (OpenAIError, ValueError):
                        reply = local_reply(tools, prompt, st.session_state.get("chat_context_id"))
                        reply.text = "The AI service is unavailable. Here is the local evidence result instead.\n\n" + reply.text
                else:
                    reply = local_reply(tools, prompt, st.session_state.get("chat_context_id"))
            serialized = asdict(reply)
            show_reply(serialized, f"history_{len(messages)}")
        messages.append({"role": "assistant", "text": reply.text, "reply": serialized})
        if reply.compound_id:
            st.session_state["chat_context_id"] = reply.compound_id
        if len(messages) > 40:
            st.session_state["chat_messages"] = messages[-40:]
    st.caption("Measured outcomes, not detected, below DL and not reported remain distinct. Structural similarity supports exploratory coverage, not guaranteed measurement.")
