# Medical Appointment No-Show Prediction and Demand Forecasting

This project studies which scheduled appointments may be missed and how many appointment records to expect each day. The original passing run is in [ma_ml.ipynb](ma_ml.ipynb). The later SMOTENC and weather-free forecast experiments are in [ma_ml_updated_before_rollback.ipynb](ma_ml_updated_before_rollback.ipynb).

**The original run passed both requested targets.** Test ROC-AUC was **0.7822** (target > 0.75), and test R² was **0.6501** (target > 0.65). The forecast used recorded same-day weather and had MAPE of **1,007.8579%**. Classifier accuracy was **0.7010** and no-show F1 was **0.6370**.

## Reproduce the passing run

The final check in `ma_ml.ipynb`, execution `[25]`, shows both passes. The results were already recorded by **17 November 2026 at 2:26:47 p.m. IST**. The original training time is unknown because the saved cells have execution counts without timestamps.

| Winning estimator | Reproduced test score | Explicit settings |
|---|---:|---|
| `HistGradientBoostingClassifier` | ROC-AUC 0.782205085595 | `max_iter=600`, `learning_rate=0.05`, `max_leaf_nodes=63`, `min_samples_leaf=25`, `l2_regularization=1.0`, `categorical_features=cat_cols`, `random_state=42` |
| `GradientBoostingRegressor` | R² 0.650067940871 | `n_estimators=400`, `learning_rate=0.04`, `max_depth=5`, `random_state=42` |

- **Classifier.** Use 47 features, including eight native categorical columns, and a 20% stratified appointment-level holdout. Use 15% of the training portion to select the F1 cutoff of 0.33. Refit on the full training portion without oversampling. Compute ROC-AUC from positive-class probabilities.
- **Forecaster.** Train one model on seven specialty series plus the clinic total. Scale numeric features and one-hot encode specialty. Include recorded same-day weather, calendar fields, lags of 1, 2, 3, 7, 14, 21 and 28 days, shifted rolling statistics over 7, 14 and 28 days, and the four-week same-weekday mean. Train through 13 March 2021. Update actual history daily during the 14 March–12 May 2021 holdout, clip predictions at zero, and score the 60 positive-demand clinic-total days.

The verification reran both winning fits with Python 3.14.7, scikit-learn 1.9.1, NumPy 2.5.4 and pandas 3.1.1. Run the original notebook from the project directory with `Medical_appointment_data_updated.csv` (SHA-256 `2df958cca34af192fa176d3589a5c2dae5379b2008538119cfdf295d19bb109a`).

## Deliverables

- [Original notebook](ma_ml.ipynb)
- [Later experimental notebook](ma_ml_updated_before_rollback.ipynb)
- [PowerPoint presentation](deliverables/Medical_Appointment_Project.pptx), with editable charts and short speaker notes
- [Classification results](smote_classification_results.csv)
- [Forecast validation results](forecast_validation_results.csv)
- [Forecast test results](forecast_test_results.csv)
- [Daily-count audit](forecast_data_audit.csv)

## Dataset

The notebook reads the supplied `Medical_appointment_data_updated.csv`. The project brief describes rehabilitation clinic records from CER in southern Brazil. The analysis uses the local file; source dates and collection completeness have not been independently verified.

| Property | Observed value |
|---|---|
| Raw size | 109,593 rows and 29 columns |
| Extra columns removed | 3 columns containing `Unnamed` in their name |
| Exact duplicates removed | 36 |
| Clean size before derived features | 109,557 rows and 26 retained columns |
| Recorded dates | 1 January 2020 to 12 May 2021 |
| Calendar coverage | 498 days without date gaps |
| Specialty labels | 7 after normalization |
| No-show appointments | 34,831, approximately 31.79% |
| Attended appointments | 74,726, approximately 68.21% |

A row represents an appointment, not necessarily a unique patient. There is no patient identifier for reliable attendance-history features or patient-level splitting.

### Main columns

| Group | Columns |
|---|---|
| Outcome | `no_show`, mapped to `target`: 1 = no-show, 0 = attended |
| Appointment | `specialty`, `appointment_time`, `appointment_shift`, `appointment_date_continuous` |
| Patient | `gender`, `age`, `Disability`, `under_12_years_old`, `over_60_years_old`, `patient_needs_companion` |
| Location | `place`, with derived `place_clean` |
| Recorded health and support flags | `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`, `Scholarship`, `SMS_received` |
| Weather | `average_temp_day`, `average_rain_day`, `max_temp_day`, `max_rain_day`, `rainy_day_before`, `storm_day_before`, `rain_intensity`, `heat_intensity` |

Names retain the supplied spelling. Engineered sums of health flags are model features, not validated clinical scores.

## Project workflow

### 1. Cleaning and exploration

Keep the raw table, remove extra columns and exact duplicate rows, normalize label case and spaces, and parse dates using `%d/%m/%Y`. Map location labels outside a manually chosen list to `other_city`, and missing locations to `unknown`.

After structural cleaning, `place` has 11,509 missing values. The four numeric weather columns have 2,191 to 2,243 missing values each. Numeric pipelines use median imputation. Some text cleaning uses `astype(str)`, which can turn missing values into a literal `nan` category.

The duplicate rule assumes matching rows are repeated records; without appointment IDs, identical appointments cannot be ruled out. The extra-column rule matches names without independently checking their contents.

EDA covers attendance balance, specialty and disability rates, age distributions, weather associations, and daily appointment counts. These are descriptive associations, not proof of causation.

### 2. Classification features and preparation

The notebook creates 47 features: 39 numeric and 8 categorical. They include calendar parts, cyclical encodings, workload counts, patient indicators, and interactions such as specialty with shift. The answer column and its original text label stay outside the feature table.

The split reserves 20% of appointments for stratified random testing. Another 15% of the training portion forms validation data for cutoff selection. Each model is then refitted on the full training portion, and validation F1 determines its ranking.

Full-day workload counts and recorded same-day weather require a prediction-time availability check. The random split does not establish performance on future dates or ensure different patients in each subset.

### 3. SMOTE experiments

Sections 5a and 5b of the later experimental notebook contain the SMOTENC implementation and comparison.

The project uses `SMOTENC`, the SMOTE variant for mixed numeric and categorical inputs. Each imbalanced-learn pipeline prepares inputs, creates synthetic training examples, one-hot encodes categories, and fits a classifier. Validation and test records are never oversampled.

On the full training split, SMOTENC adds 31,915 synthetic no-show records. Counts change from 59,780 attended and 27,865 no-show rows to 59,780 rows in each class. SMOTE models do not also use balanced class weights.

### 4. Daily demand preparation

Demand means the number of appointment records, including both shows and no-shows. It does not measure actual attendance or unmet requests for appointments.

Create one daily series for each specialty plus `ALL_SPECIALTIES`, the clinic total. Absent date/specialty combinations receive zero, assuming they reflect no recorded appointments rather than missing collection.

Historical features include lags of 1, 2, 3, 7, 14, 21 and 28 days, shifted rolling means and standard deviations, and calendar features. Shifted features exclude the current outcome. Removing the first 28 days leaves 3,760 rows across eight series.

### 5. Chronological forecasting experiments

Section 8 of the later experimental notebook compares 14 candidates: four historical baselines, two Poisson models, four log-transformed models, and four recent-level correction models. The new experiments exclude recorded same-day weather.

| Window | Training ends | Evaluation dates |
|---|---|---|
| Validation 1 | 13 December 2020 | 14 December 2020 to 12 January 2021 |
| Validation 2 | 12 January 2021 | 13 January 2021 to 11 February 2021 |
| Validation 3 | 11 February 2021 | 12 February 2021 to 13 March 2021 |
| Final test | 13 March 2021 | 14 March 2021 to 12 May 2021 |

Selection first considers candidates with mean validation R-squared above 0.65, choosing the lowest validation MAPE among those candidates. None qualifies. The fallback chooses the lowest validation MAPE as an exploratory result, not a recommended model meeting both goals.

**Forecast horizon:** successive one-day-ahead predictions with actual prior demand updated daily. This is not a single forecast of the whole next 60 days. The holdout was inspected during earlier diagnosis; genuinely new future data is needed for independent confirmation.

## Classification results

| Model | Accuracy | No-show F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.7010 | 0.6370 | 0.5187 | 0.8252 | 0.7822 |
| Random forest, balanced | 0.6896 | 0.6340 | 0.5071 | 0.8455 | 0.7761 |
| Random forest, SMOTENC | 0.6812 | 0.6327 | 0.4992 | 0.8638 | 0.7682 |
| Logistic regression, balanced | 0.6761 | 0.6190 | 0.4943 | 0.8276 | 0.7579 |
| Logistic regression, SMOTENC | 0.6609 | 0.6131 | 0.4810 | 0.8450 | 0.7477 |

HistGradientBoosting ranks first by validation F1, with a selected cutoff of 0.33. It catches about 83% of actual no-shows, but only about 52% of its alerts are correct. SMOTENC increases recall relative to the corresponding balanced models but reduces precision and slightly reduces F1.

LightGBM is optional in the code but does not appear in the saved comparison. XGBoost availability is checked, but the classification comparison does not train it. Optional-library availability and package versions can change a fresh run.

## Forecast results and interpretation

| Metric | Original historical run | Validation-selected experiment |
|---|---:|---:|
| Model | Gradient boosting regressor | Relative MAPE mean |
| Inputs | Includes recorded weather | Calendar and prior demand |
| MAPE | 1,007.86% | 124.57% |
| R-squared | 0.6501 | -0.0558 |
| MAE, appointments | 82.49 | 117.75 |

After recorded same-day weather was removed, rolling validation initially selected **Specialty log boosting**, which returned test R² **0.2411**. The expanded search selected **Relative MAPE mean**, with test R² **−0.0558** and MAPE **124.57%**. Its MAE rose to **117.75 appointments** from **82.49** in the original run.

The audit records 1,510 appointments on 2 April 2021 and one on 3 April. Counts then remain between 1 and 11 through the end of the file. Training-period median volume is 168 per day; final-test median volume is four. A continuous calendar does not establish complete data collection.

MAPE divides each absolute error by actual demand, so errors on one- or two-appointment days become large percentages. MAE, RMSE, WAPE and R-squared expose other aspects of error. Positive-demand clinic-total dates define MAPE. The other new summary metrics include all clinic-total dates; all 60 clinic-total test dates happen to have positive demand.

### Target status

| Metric | Measured value | Target | Status |
|---|---:|---|---|
| Classification accuracy | 0.7010 | > 0.80 | Not met |
| No-show F1 | 0.6370 | > 0.70 | Not met |
| ROC-AUC | 0.7822 | > 0.75 | Met |
| Original benchmark MAPE | 1,007.8579% | < 20% | Not met |
| Original benchmark R-squared | 0.6501 | > 0.65 | Met |
| Exploratory forecast MAPE | 124.57% | < 20% | Not met |
| Exploratory forecast R-squared | -0.0558 | > 0.65 | Not met |

The original historical forecast narrowly met its R-squared goal but missed MAPE. The notebook adds the accuracy goal alongside the project brief's F1 and ROC-AUC goals.

## Running the notebook

Use Python 3.14.7 to match the verification environment. From the project folder in PowerShell:

```powershell
python -m venv .venv_project
.\.venv_project\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install jupyter ipykernel scikit-learn==1.9.1 numpy==2.5.4 pandas==3.1.1
python -m jupyter notebook ma_ml.ipynb
```

In VS Code, select the new environment as the notebook kernel, restart the kernel, and run cells from top to bottom. Later cells reuse earlier variables. The later experimental notebook also trains the SMOTENC models and can take several minutes. A fresh run can add LightGBM when installed and can select a different model.

If an existing environment points to an unavailable Python executable, create a fresh one. If a notebook tab still shows old content after an external update, close and reopen it without overwriting the updated file with an older unsaved buffer.

## Updated application interface

The rewritten `app.py` removes the model-performance score tiles and the redundant age flags, support flag, and numeric weather inputs. Disability auto-fills from the unique specialty mapping in the reference data. Age flags are derived from age; the other removed model inputs use historical defaults, explained in the interface.

The attendance result uses the saved classifier without the invalid progress-bar arguments. Demand forecasts use strict prior-day history, preserve the saved feature and category schema, and recursively fill future days. Choose a 1, 7, or 14-day forecast, then download the daily estimates. The date range stays within 30 days of the last recorded data date, rather than extrapolating years past the supplied history.

Run the app with the project environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You can also double-click `Run_App.bat`. It uses the project Python environment directly, so terminal activation and a global Streamlit command are not required. Keep its terminal window open while using the app.

The native theme lives in `.streamlit/config.toml`; pure prediction helpers live in `prediction_helpers.py`. Model schema compatibility is tested against scikit-learn 1.9.1. Run the regression checks with:

```powershell
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v
```

## Streamlit prototype and artifact lineage

`app.py` implements a no-show input form and a specialty/date demand interface. It loads separate processed CSVs and joblib bundles. With those existing artifacts present:

```powershell
python -m streamlit run app.py
```

The presentation documents this prototype from source code. It does not claim that a live app was tested or deployed. Notebook experiments do not automatically update the saved application bundles.

**Important:** `generate_processed_data.py` replaces recorded attendance labels with a hand-written risk rule and generates demand using a mathematical formula. Do not use that script to reproduce the notebook's real-label results. `train_models.py` uses separate processed files, a different classification cutoff, and a random demand split. Its metrics are not equivalent to the notebook's chronological results. The rewritten app now constructs demand inputs from strict prior history, but the artifact-generation and training limitations above remain.

## Repository guide

| File | Purpose |
|---|---|
| `ma_ml.ipynb` | Original notebook with the passing run |
| `ma_ml_updated.ipynb` | Restored copy of the original notebook |
| `ma_ml_updated_before_rollback.ipynb` | Later SMOTENC and weather-free experiments |
| `ma_ml.ipynb` | Companion copy; use the updated notebook as the reference |
| `Medical_appointment_data_updated.csv` | Raw notebook input |
| `smote_classification_results.csv` | Recorded classification comparison |
| `forecast_validation_results.csv` | Mean validation scores |
| `forecast_validation_folds.csv` | Individual fold scores |
| `forecast_validation_dates.csv` | Training and validation boundaries |
| `forecast_test_results.csv` | Test scores for all 14 forecasting candidates |
| `forecast_test_predictions.csv` | Validation-selected model predictions |
| `forecast_data_audit.csv` | Daily counts and changes |
| `forecast_target_checks.csv` | Forecast-only target checks |
| `forecast_reference_comparison.csv` | Historical and exploratory reference comparison |
| `forecast_selected_model.png` | Exploratory forecast chart |
| `app.py` | Existing Streamlit prototype |
| `models/` | Separate existing application bundles |
| `generate_processed_data.py`, `train_models.py` | Separate artifact-generation and training scripts with the caveats above |
| `deliverables/Medical_Appointment_Project.pptx` | Project deck with embedded speaker notes |

## Limitations and next steps

- Confirm source dates, data completeness, and the reason for the April volume collapse.
- Review category mappings, deduplication assumptions, and prediction-time availability.
- Obtain patient attendance history, booking lead time, and operational capacity where possible.
- Evaluate classification on future dates and separate patients when identifiers become available.
- Confirm forecasting on a new period and define the required forecasting horizon.
- Preserve original outcomes when rebuilding processed data and align app preprocessing with the evaluated notebook.
- Measure outreach and staffing effects prospectively. No no-show reduction, revenue gain, or staffing saving has been measured.

## Reproducibility

The notebook uses a seed of 42. Saved outputs and CSVs document recorded experiments; they do not replace an end-to-end rerun in the selected environment. Forecast checks verified chronological separation, finite non-negative predictions, and that changing a day's outcome does not alter that day's or earlier predictors. These checks do not establish future generalization.
