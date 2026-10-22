# Deploy E&L Atlas on Streamlit Community Cloud

The app is prepared for deployment from `syam3500/OSI_hackhaton_2026`. Cloud hosting runs the same Streamlit application, including chat, the map and session target uploads.

## Deployment settings

| Field | Value |
|---|---|
| Repository | `syam3500/OSI_hackhaton_2026` |
| Branch | `ritwik` |
| Main file path | `app/app.py` |
| Python | `3.12` in Advanced settings |
| Dependencies | Root `requirements.txt` |
| Default dataset | `deployment/evidence.zip`, unpacked and checksum-verified on first startup |

Sign in at [Streamlit Community Cloud](https://share.streamlit.io), select the workspace for the repository owner, create an app and use the settings above. The source changes and prepared bundle must first be present on the selected GitHub branch.

The deployment bundle is about 2.8 MB. It contains only the prepared chemistry/evidence tables, PCA projection and provenance. It does not contain the raw 262 MB MassSpecGym download, source workbooks/PDFs, private chat, local environment, or credentials. Source records retain their reported limitations and identity exclusions.

## Chat and the optional AI agent

The default chat is a local evidence assistant. It works without any model credentials, makes no model API calls and routes focused questions to the same source and chemistry calculations as the map.

To enable the optional OpenAI tool-calling agent, add these values in the app's **Advanced settings → Secrets** or **Settings → Secrets**:

```toml
OPENAI_API_KEY = "your-server-side-key"
OPENAI_MODEL = "a-Responses-compatible-model-available-to-your-account"
```

Use an actual model ID supported by the API account. No model or key is assumed. Do not commit a real `secrets.toml` or paste keys into the chatbot. `.streamlit/secrets.toml.example` contains placeholders only.

After configuration, users can enable **Use AI agent** in Chat. Messages and selected tool evidence are then sent to OpenAI. Calls use the Responses API with storage disabled, a bounded tool loop and read-only tools: compound search, compound evidence, SMILES analysis and dataset summary. The model cannot change a calculated evidence status, assign a neighbour's RRF to a query, change data, run arbitrary code or deploy the app. Deterministic source results remain available below the response. On API failure, the app returns the local evidence result with a notice.

The SDK/tool-loop contract is tested with a mock client. Live model responses require a real key/model and have not been exercised when those credentials are absent.

## Change datasets after deployment

### Replace target compounds for one session

Open **Data & sources**, upload a **CSV or XLSX**, select a worksheet if applicable, map the name/CAS/SMILES columns, then click **Use this target dataset**. Provide a name and at least CAS or SMILES. Limits are 25 MB and 20,000 rows. Invalid and conflicting structures remain flagged; no unreviewed identity is silently substituted.

The chat and map then use the uploaded targets with the existing CLAP and supporting reference evidence. The upload affects the current browser session only and does not alter other visitors' data. **Restore Borton targets** returns to the default build. Session uploads and chats are not saved as a durable database and may disappear after a reload or server restart.

### Permanently replace the default/reference build

1. Update the local source files and any reviewed overrides.
2. Run the preparation pipeline with the applicable source paths.
3. Rebuild and validate the small deployment bundle:

```bash
python scripts/prepare_data.py --background "../Relevant BAckground info" --draft "../first draft of prototype"
python scripts/package_deployment.py
python -m pytest -q
```

4. Publish the updated source and `deployment/evidence.zip` to the deployed GitHub branch. Streamlit rebuilds the app from repository updates. A changed bundle replaces the previously extracted default data; active session uploads are cleared when the reference version changes.

Changing a reference layer means producing a compatible prepared build, not treating an arbitrary uploaded target CSV as measured FDA evidence. `DATA_SOURCES.md` and the manifest document the default inputs. Review source redistribution terms before publishing a different data bundle.

## Verification before publishing

- Run `python -m pytest -q` with prepared source data.
- Run `python scripts/package_deployment.py` after changing default data.
- Confirm no real API key or other credential is included in Git changes.
- On Cloud, check the app logs, submit **Check Bisphenol A**, open its compound map and test a session upload.

Official references: [Deploy to Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [update a deployed app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling).
