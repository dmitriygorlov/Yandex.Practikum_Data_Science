# Reproducible teaching projects — 2026-11-18

All 15 notebooks are inventoried in `notebook-inventory.json`, including original
imports, data references, code-review candidates and recorded Python versions.
Original saved outputs/conclusions remain historical. Newly executed notebooks go to
ignored `artifacts/`; neither synthetic smoke scores nor old training logs are new
benchmark results. Exact old package versions are unknown: there was no dependency
manifest. Most notebook metadata recorded Python 3.7.8.

## Environments

| Environment | Python | Main pinned libraries |
| --- | --- | --- |
| Analytics / ML / NLP | 3.14.10 | pandas 3.1.1, NumPy 2.5.5, SciPy 1.18.1, sklearn 1.9.1, LightGBM 4.7.0, CatBoost 1.2.10, statsmodels 0.15.0, NLTK 3.10.4, phik 0.12.5, pymorphy3 2.0.6 |
| CV (separate kernel) | 3.13.18 | TensorFlow 2.21.0, Keras 3.15.1, pandas 3.1.1, NumPy 2.5.5 |

Windows x64 is the tested platform. The current CatBoost Windows wheels stop at
Python 3.14 and TensorFlow wheels at 3.13, so Python 3.15 is not forced onto either.
`requirements/*.in` select direct dependencies; `*.lock` record resolved transitive
versions. Linux/macOS and CUDA were not tested. No Dockerfile existed; none was added
solely for these local notebook exercises.

## Changes to behavior

- pandas: replace chained inplace fills/replacements with assignments to the parent,
  removed `set_axis(inplace=...)`, `fillna(method=...)`, `DataFrame.append`, and deprecated
  in-place structural operations. Hourly frequencies and weekday names use current APIs.
- Morphology: replace the Linux-only Mystem shell download/copy with a local pymorphy3
  adapter. All five Russian purpose categories have regression tests. Tokenization and
  ambiguous lemmas are not guaranteed identical to Mystem; old category results remain
  historical. Remove unused Mystem installation from English text analysis.
- NLP: prepare modern NLTK resource names explicitly, use a stopword list, and fit
  TF-IDF inside each model-selection fold; the final LightGBM uses the fitted search
  winner. Correct ignored LightGBM `depth` parameters to `max_depth` in all modules.
- Supervised learning: fit scaling only on training rows, use the selected forest's
  probabilities, and print the actual test F1 rather than an unrelated validation value.
- Numerical methods: learn category encoding only from training rows, support unseen
  categories, and remove the deleted LinearRegression `normalize` argument (numeric
  features are already explicitly scaled).
- Time series: select rolling-window size without touching the final 10% holdout and
  use the selected value, rather than the historical hard-coded 73. Lag/rolling features
  remain strictly past-only and are tested for future leakage.
- Gold recovery: define zero/zero sMAPE contribution as zero and reject empty/mismatched
  arrays, avoiding undefined metrics on legitimate zero observations.
- Telecom: durations use explicit mean-Gregorian months (365.2425/12 days), rather than
  unsupported NumPy month timedelta division or resolution-dependent integer ticks.
  Valid LogisticRegression grids use the current `l1_ratio` API. SGD uses `log_loss`;
  ROC calculations also support margin-only classifiers. Exploratory clustering is
  excluded from model features because it was fitted before the train/test split.
- CV: move executable training into `cv_model.py`, replace deprecated ImageDataGenerator
  with tf.data, use `learning_rate`, keep the original 0..1 pixel scaling and contiguous
  25% validation partition. Random weights are the safe default; ImageNet download is
  explicit. Results with random weights cannot be compared to historical pretrained runs.

Remaining methodological limits: the teaching notebooks still contain historical
analysis choices and manually specified model comparisons. Some preprocessing in the
non-text grid searches is outside inner CV folds. These are not production model
validation reports. No accuracy or speed improvement is asserted.

## Run from the repository root

```powershell
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install -r requirements/core.lock
.venv/Scripts/python -m pip check
.venv/Scripts/python prepare_nltk.py
.venv/Scripts/python validate_notebooks.py
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m pytest tests/test_notebooks.py -q
.venv/Scripts/python smoke_text.py
```

Copy the original CSV files into `data/`, or set `DATA_DIR`. The loader is local-first
and refuses implicit downloads. `ALLOW_DATA_DOWNLOAD=1` explicitly permits the original
public Yandex URLs; larger training datasets should be supplied locally when needed.
Telecom expects `DATA_DIR/final_provider/{contract,personal,internet,phone}.csv`; CV
expects `DATA_DIR/faces/labels.csv` and `faces/final_files/`. No automatic Drive downloads,
shell archive extraction, global installs, paid APIs or real messages occur.
NLTK dictionaries are explicitly prepared inside `data/nltk_data` by `prepare_nltk.py`.

```powershell
.venv/Scripts/python validate_notebooks.py Module-01_01-Data-Processing/project-1_borrower-reliability-survey.ipynb
py -3.13 -m venv .venv-cv
.venv-cv/Scripts/python -m pip install -r requirements/cv.lock
.venv-cv/Scripts/python -m pip check
$env:KERAS_HOME = "$PWD/artifacts/keras"
.venv-cv/Scripts/python -m pytest tests/test_cv.py -q
```

## Validation coverage

| Notebook/module | Actual verification | Not reproduced |
| --- | --- | --- |
| 01-01 Borrower reliability | Complete original-data notebook; morphology and CoW regressions | Historical Mystem token-by-token equivalence |
| 01-02 Apartment analysis | Complete original-data notebook | Historical numerical identity |
| 01-03 Cellular tariffs | Complete original-data notebook | Historical numerical identity |
| 01-04 Video games | Complete original-data notebook | Historical numerical identity |
| 02-01 Users behavior | Complete original-data training/evaluation notebook | Historical numerical identity |
| 02-02 Supervised learning | Syntax/API review; model-family smoke | Full Churn training/grid search |
| 02-03 Oil-region business | Revenue/index/bootstrap regression tests | Full three-region data run |
| 02-04 Gold recovery | sMAPE regression tests | Full gold dataset/grid search |
| 03-01 Linear algebra | Complete original-data notebook | Historical random-matrix identity |
| 03-02 Numerical methods | CatBoost/LightGBM synthetic train/predict, API review | Full autos data/grid search |
| 03-03 Time series | Past-only feature and statsmodels smoke tests | Full taxi grid search |
| 03-04 Text analysis | Entire notebook on synthetic comments | Full original toxic-comments training |
| 04-01 SQL | Complete notebook on original exported query CSVs | PostgreSQL query execution; no DB is configured |
| 04-02 CV | Generated PNGs, train/validation loader, ResNet50 gradient step, `.keras` reload equivalence | Real faces, ImageNet weights, full training, GPU |
| Final telecom | Duration, valid grids, margin ROC/model-family/phik checks | Full original telecom data/training |

The 12 fetched original public CSVs total about 23 MB; their sources, byte counts and
SHA256 hashes are recorded in `validation-data.json`. Data files are not committed.
Schema/syntax validation covers all 15 notebooks. CI runs core and separate CV checks
on Windows. TensorFlow/Keras emitted upstream Python/NumPy deprecation warnings during
the initial smoke; passing tests do not imply those warnings are resolved upstream.

No legacy model file is tracked. New `.keras` persistence is tested within the same
environment; cross-version old-model compatibility is unverified and requires original
trusted files. Retrain under a locked environment instead of assuming pickle portability.

No application/autodeploy configuration was found, and GitHub Pages was disabled at
inspection. New workflows are read-only tests. Rollback uses `git revert` of the
modernization commit; the baseline is `8e30956b591f1101d9c15612b2068bc76f734803`.

Primary references: [Python releases](https://www.python.org/downloads/source/),
[pandas 3.1 changes](https://pandas.pydata.org/docs/whatsnew/v3.1.0.html),
[sklearn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[deprecated ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator),
[TensorFlow package metadata](https://pypi.org/pypi/tensorflow/2.21.0/json),
[CatBoost package metadata](https://pypi.org/pypi/catboost/1.2.10/json).
