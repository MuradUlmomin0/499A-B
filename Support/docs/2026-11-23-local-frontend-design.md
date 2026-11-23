# PRIVATE SHIELD local frontend design

Date: 23 November 2026
Status: implemented and browser-verified on 23 November 2026; legacy test limits recorded in the plan
Target project: `/mnt/CA7ED12F7ED114D3/NSU/499A-B`

## Intent and constraints

Give the student a local browser interface for demonstrating the existing
intrusion classifier and explaining the completed federated experiments to
faculty. The user selected local operation first. Use the existing `Support/`
folder and the self-contained assets already in this project. Do not edit the
`499A-B (Copy)` research repository.

The laptop has 8 GB RAM. The research algorithms, learned parameters, class
order, preprocessing, partitions and recorded experiment results remain as
implemented. This is a presentation and inference layer, not a new study.

## Approaches

* Recommended default, subject to this approval: Streamlit. It calls the current
  Python interfaces directly and supplies tables, charts, controls and CSV
  downloads without a separate JavaScript build system.
* Alternative: Flask with HTML/CSS/JavaScript. It gives more control over a
  custom website but needs additional web code and request/state handling.

Streamlit is the proposed first implementation because the immediate goal is
a straightforward local faculty demonstration. The user has approved local
deployment; framework and scope are confirmed by approving this design.

## First-version interface

Use a simple, readable sidebar with four views:

1. **Overview:** project purpose, 46 input features, 34 traffic categories,
   simulated-client research setting, and a brief explanation of the MLP.
2. **Predict traffic:** select a saved FedAvg, adapted FedMADE, temporal-trust,
   or fusion model. Initially use the clean IID seed-42 final-round artifacts
   already referenced by the four `demo_*.yaml` files. A button classifies the
   bundled 20 real validation-derived records. Show record index, predicted
   category, predicted-class softmax score, and an optional feature preview.
   Call scores model scores, not calibrated confidence or detection accuracy.
   Provide a CSV download with labels and all class scores. Clear previous
   predictions when the selected model changes.
3. **Saved research results:** read the existing CP7 comparison, with IID and
   non-IID selections and clean/poisoned metrics. Clearly label final-test,
   three-seed summaries. Include the actual mixed conclusion: fusion reduces
   mean target-recall damage versus FedAvg and temporal trust, but adapted
   FedMADE is stronger in the recorded experiment; rare-class failures remain.
   Never calculate headline accuracy from the unlabeled 20-record demo.
4. **Training history:** plot the selected saved model's 20-round validation
   accuracy and macro F1 from its existing `round_metrics.csv`. Label these as
   saved training histories, not a training session happening in the browser.

Use plain wording and native Streamlit controls. The first version has no file
upload, live packet capture, new model training, attack execution, accounts,
public hosting, or final-test inference. These boundaries keep the initial
demo understandable and avoid guessing new data/preprocessing rules.

## Integration

Create `Support/frontend/app.py` for the interface and
`Support/frontend/service.py` for the model catalog, input loading, prediction
adapter, saved-result loading, and CSV generation. Keep service logic usable
without a Streamlit import so it can be tested independently.

Reuse `load_model_artifact`, `load_prepared_data`, and
`predict_prepared_data`, including their model/data contract checks. Use
`Support/configs/demo_predict.yaml` for the prepared demo inputs. Read model
artifact paths from the four existing evaluation demo configurations without
running their full validation workflows. Reuse the existing local path reader
for historical JSON and YAML. Preserve `main.py` as the current CLI.

Keep at most one cached model, use CPU and one numerical thread, and never
load the complete training/validation/test arrays for a prediction click.
Protect shared inference state across browser sessions and ensure changing a
model cannot reuse predictions under the wrong label. Display a clear error
for missing/incompatible inputs. If optional model/result/history assets are
absent from a smaller GitHub package, disable the unavailable choice or show
an explanatory empty view; keep available prediction functionality usable.

Add one compatible, pinned Streamlit release to `Requirements.txt` only after
checking dependency resolution against the existing Python 3.14.4, NumPy 2.5.0
and pandas 3.0.4 environment. Do not silently upgrade/downgrade pinned research
libraries. Resolve any conflict before installing.

After the new files and requirement are finalized, update the package's
`Support/runtime_manifest.json` runtime-code inventory and verification record
to include the frontend. Keep all historical research records and model/data
checksums unchanged. Recheck runtime integrity and the existing CLI.

Expected launch command after implementation:

```bash
shield_env/bin/python -m streamlit run Support/frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

## Verification and acceptance

Test the prediction adapter and result provenance, including model changes,
missing assets, incompatible feature/class contracts, and partial demo bundles.
Check every model on the bundled inputs and compare its UI labels with the
existing prediction API. Confirm CPU-only bounded inference, no optimizer or
training calls, and no test-set scoring. Browser smoke checks must cover page
navigation, model selection, prediction table, CSV download and chart labels.

Verify startup and prediction with the old project/download locations blocked.
Run appropriate existing regressions, runtime integrity verification and CLI
smoke checks. Report files changed, dependency changes, test outcomes, the
launch command, and any remaining limitation. Do not publish or push to GitHub
as part of this local frontend task.
