# DEPRECATED — legacy per-server regressor wrappers

These 15 files were `src/server{1..5}_regressor/{detect,speech,predict}_predictor.py`.
They are **superseded by `src/regressors.py`** (`TracePredictor`) as of workstream
**B1** and are no longer imported by anything.

## Why they were replaced (D-02 / D-02b / D-02c)

* **D-02** — every one of the 15 hardcoded `models/server1/…` and
  `dataset/server1.csv`. Servers 2–5 predicted computation delay with server 1's
  model on server 1's trace. The computation-latency path was not heterogeneous.
* **D-02b** — `Server._load_predictors_from_regressor_folder` did
  `sys.path.insert(0, server{i}_regressor)` then
  `importlib.import_module("detect_predictor")`. Identical module names across
  the five folders meant `sys.modules` returned server 1's already-imported
  class for servers 2–5 regardless of `sys.path`.
* **D-02c** — each import was wrapped in `except Exception: predictors[x] = None`,
  and `_predict_using_letter` then fell back to a **contention-free single-letter
  CSV row lookup**. With scikit-learn absent (D-01) every load failed silently
  and the run still produced plausible numbers.

Additionally (W-01) the wrappers were **not identical** — they carried three
different `_build_features` schemas matching three different shipped model
feature sets. `src/regressors.py` ports all three and selects by the model
bundle's own `feature_columns`.

## Why they are kept (for now)

The frozen `results/reference_v0/` was produced against this code path. Keeping
the files (unimported, out of `src/`'s import surface) lets us reproduce that
path if a discrepancy needs chasing. **Scheduled for deletion** once B1's
numbers are reconciled against `reference_v0` — track in `CHANGELOG.md`.

Do **not** import from here. Gate G4 (`_spec_source` rule is analogous) and
plain review should treat this directory as archived.
