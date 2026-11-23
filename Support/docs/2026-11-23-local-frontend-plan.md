# Local frontend implementation plan

> For agentic workers: use superpowers:executing-plans for this approved inline execution.

**Goal:** A local Streamlit faculty dashboard for the existing saved models and results.
**Architecture:** `Support/frontend/service.py` owns bounded prediction and read-only result access. `Support/frontend/app.py` renders four views and session state. Existing CLI and research algorithms are reused unchanged.
**Tech stack:** Python 3.14.4, Streamlit 1.67.0, existing pinned PyTorch/NumPy/pandas stack; pytest 9.1.1 for development checks.
**Spec:** `Support/docs/2026-11-23-local-frontend-design.md`, approved by the user's “proceed”.

## Global constraints

- Work in `/mnt/CA7ED12F7ED114D3/NSU/499A-B`, inside its existing `Support/` structure.
- Preserve the original Copy project, historical record bytes, learned weights, class/feature order and preprocessing.
- Use CPU, one numerical thread, at most one cached model, and the bundled prediction input with at most 20 records.
- No training, optimizer, file upload, live capture, final-test inference, publishing or push.
- Keep original seven package pins; add only compatible Streamlit runtime dependency.
- Preserve dirty user work; use scoped /tmp backups and staging, no branch/index/commit operations. The approved target directory takes precedence over the skill's worktree default.
- Record the frontend runtime code and changed requirement in the runtime manifest, with fresh verification.

## Review focus

1. Model changes after a successful prediction must clear stale labels and CSV state.
2. Partial GitHub bundles must retain FedAvg prediction when optional configs, models, results or histories are missing.
3. Invalid input contracts or oversized prediction arrays must fail before expensive inference.
4. Concurrent sessions must not swap or mutate each other's cached model during inference.
5. Saved validation curves and three-seed final-test summaries must retain distinct labels, percent/percentage-point units and negative recall drops.

## Task 1: Bounded prediction and saved-results service

Files: create `Support/frontend/service.py`; create `Support/tests/test_frontend_service.py`.

Interfaces:
- `ModelChoice(key, label, artifact_path, available, reason)` for the four fixed clean IID seed-42 model variants.
- `PredictionView(model_key, model_label, model_name, artifact_sha256, frame, feature_preview, class_names, testing_only)`; `csv_bytes()` returns UTF-8 CSV including all class scores.
- `PredictionService(root=ROOT).catalog()`, `.demo_summary()`, `.predict(model_key)`: Streamlit-independent service; one cached CPU artifact, serialized inference.
- `load_research_results(root=ROOT, condition='iid') -> pandas.DataFrame`: validates the saved CP7 source and returns four mean-metric rows with percent/percentage-point conversions.
- `load_training_history(model_key, root=ROOT) -> pandas.DataFrame`: validates 20 ordered validation rounds and returns their original accuracy/macro-F1 columns.
- `FrontendError` for invalid inputs; `UnavailableError` for absent optional assets.

- [x] Write and run failing tests for catalog/mini bundle fallback, predictions matching the existing API, all-score CSV, cache replacement, wrong contract, absent/corrupt model, oversized demo, concurrent model switching, metric source/units, negative drops and history roles.
- [x] Implement the service using the existing artifact/data/prediction APIs; never call a training/evaluation runner.
- [x] Run `python -m pytest -q Support/tests/test_frontend_service.py`; fix failures before Task 2.

## Task 2: Browser dashboard

Files: create `Support/frontend/app.py`; create `Support/tests/test_frontend_app.py`; modify `Requirements.txt`.

Interfaces: Task 1 service; Streamlit session state stores only completed prediction views, tagged by model identity.

- [x] Install the successfully resolved pinned Streamlit 1.67.0 and development pytest without changing original research pins.
- [x] Write failing Streamlit AppTest cases for all four views, real prediction, model-switch state clearing, missing optional results/history, and downloadable CSV.
- [x] Implement native Streamlit sidebar navigation, global available-model selection, four designed views, source labels, errors and CSV download.
- [x] Bind loopback only at launch. Read paths relative to the script's repository, not the process working directory.
- [x] Run service and AppTest suites. Check real Chrome rendering and CSV download against a loopback server.

## Task 3: Integrity, regression and handoff

Files: modify `Support/runtime_manifest.json` and mark the design/plan implemented after evidence is collected.

- [x] Refresh runtime hashes only for approved frontend additions and Requirements.txt; retain every research-asset checksum.
- [x] Run component suites, the current repository test collection, and the full adapted research regression suite; identify any pre-existing legacy-test collection limitations explicitly.
- [x] Verify old-folder-blocked prediction for all four models, CLI dummy/prepared workflows, local HTTP health, CPU/RAM behavior and runtime bundle checksums.
- [x] Obtain one fresh whole-change review as required by executing-plans. Fix material findings with a failing regression first; re-run relevant suites.
- [x] Record final checks in the runtime manifest and report created/modified files, tests, remaining limits, exact launch command and suggested commit message. Do not create a Git commit or publish.

## Observed results

- Frontend service and Streamlit AppTest: 33 passed.
- Actual-folder reference and path checks: 14 passed.
- Chrome: overview, FedAvg inference, real CSV download (20 records, 34 scores), model-switch clearing, fusion inference, non-IID saved comparison and validation charts passed.
- All four real artifacts match the existing API with old project/download/dataset folders blocked; CPU one thread, inference-process peak RSS 373.1 MiB. This is not a whole-laptop or browser RAM measurement.
- Dummy and prepared-prediction CLI checks passed.
- All 7,660 packaged assets and 82 runtime code files verified. Historical asset checksums and all model weights remain unchanged.
- All 48 saved final-test cases verified offline; no new final-test forward passes.
- Original seven requirements pins retained; Streamlit 1.67.0 added; pip check passed.
- Fresh review: no Critical or Important findings. Three minor findings deferred below.

## Existing test limitations

Bare collection of Support/tests is blocked by its pre-existing lowercase support import. The new frontend tests are self-contained and run with --noconftest. Historical tests were case-adapted only under /tmp. The full historical run returned 1,029 passed and four failures; retry resolved the CLI timeout, leaving three preparation tests stopped by the real <1 GiB available-RAM guard (1,030 distinct cases passed). The resource guard and test timeouts were not relaxed. These failures concern the older preparation runner, which the dashboard does not call. The expected invalid-input float32 overflow warning was retained.

## Deferred minor findings

- The global model selector remains visible on the all-method three-seed results page; it affects prediction and history only.
- The demo checksum is read before archive-size rejection; a mistakenly substituted large NPZ can cause extra disk I/O.
- Synthetic concurrency fixtures share weights; distinct per-model expectations would strengthen that regression test.

## Execution decisions and review record

# SDD ledger — plan: Support/docs/2026-11-23-local-frontend-plan.md

Ruling: Execute in the user-approved existing 499A-B folder with scoped backups, not a new worktree — the user explicitly chose this folder and existing dirty work must remain intact — cost if wrong: weaker branch isolation, mitigated by backups and a fixed owned-file list.
Ruling: Keep testing artifacts in /tmp and install pytest only as a development tool; do not add it to runtime requirements — preserve the runtime-only package structure — cost if wrong: developers must install pytest separately to rerun tests.
Pre-flight: Task 2 consumes Task 1 ModelChoice/PredictionView APIs and data-frame units; plan signatures align. Task 3 owns runtime-manifest refresh after Task 2 Requirements change.
Task 1: Ruling: Existing Support/tests/conftest.py uses legacy lowercase support imports, so run the self-contained frontend tests with --noconftest and perform broad adapted research regressions separately — avoids modifying unrelated old tests — cost if wrong: default legacy test collection remains unavailable and must be reported.
Task 1: complete — missing-module RED followed by service implementation; 25 tests passed in 10.27 seconds using --noconftest (service_green.log). No research code or artifacts changed.
Task 2: UI RED was 8 missing-app failures. GREEN: 33/33 service and AppTest cases passed in 7.19 seconds; installed Streamlit 1.67.0 while retaining all original package pins. Browser verification continues.
Task 2: complete — 33/33 frontend tests; Chrome seven-flow smoke passed, including real 20-row CSV with 34 class scores. Four actual models match existing API while old project/download/dataset paths are blocked; CPU one thread, peak inference process RSS 373.1 MiB.
Final: minor (deferred): The global model selector stays visible on the three-seed comparison page even though it applies to predictions and history only; saved result data and role labels are correct.
Final: minor (deferred): Demo NPZ checksum runs before the archive size limit, so an accidentally substituted large file can incur extra disk I/O before rejection.
Final: minor (deferred): Synthetic concurrency fixtures share weights; they validate identity/locking but could be strengthened with different expected per-model scores.
Final review: fresh gpt-6.2-astra reviewer found no Critical or Important defects; ready for the approved local faculty demo.
Final: Ruling: Wider regressions initially returned 1029 passed and 4 resource-related failures (one 60-second CLI timeout; three existing <1 GiB available-RAM guards). Close only our isolated verification Chrome and retry the affected checks without changing source or test thresholds — retain real resource safeguards — cost if wrong: environmental failures remain reported rather than hidden.
Final: Ruling: Keep completed work in place without a Git menu, commit, push or workspace deletion — the user selected local in-place development and preserved existing dirty work — cost if wrong: user handles integration and /tmp backups remain until manual cleanup.
Final: Ruling: Reviewer set aside unfinished verification metadata and wider regressions; finish both before handoff. Supplied browser and parity evidence has been independently run here. Unrelated dirty changes, public hosting, training and fresh holdout scoring remain outside this approved local task — cost if wrong: those workflows need their own later verification.
Verification update: Retry resolved the CLI timeout (six checks passed), but the same three old preparation checks still stop at the unchanged 1 GiB available-RAM guard. Do not weaken that guard or alter fixtures to manufacture a pass; report the resource limit. New frontend tests and real-browser flows are green.
Task 3: complete — final frontend suite 33/33 passed in 16.21 seconds; actual-folder reference/path suite 14/14; isolated dummy/prepared CLI passed; 7,660 assets and 82 runtime code hashes verified; all 48 saved final-test cases verified offline. Initial wider suite 1029 passed/4 failed; retry resolved the CLI timeout, leaving three unrelated real-RAM-guard failures (1030 distinct historical cases passed). Fresh reviewer found no Critical/Important issues. Dashboard remains on loopback; original Copy and all historical asset/model bytes preserved.
