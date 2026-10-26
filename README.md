# Medical Appointment Project — simpler implementation

The analytical decisions follow industry practice. The Python uses direct pandas
operations, short notebook cells, and only a few functions for repeated work.

## Files

- `medical_appointment_project.ipynb`: the original 20 steps, rewritten simply.
- `app.py`: a separate Streamlit application. No `project_utils.py` is required.
- `requirements.txt`: dependencies for Python 3.12.

The original project and CSV could not be located when this version was created.
This revised notebook is delivered without data-dependent outputs. Its scores must
be calculated by running it on the original CSV. Previous-version scores are not
presented as results of this rewrite.

Software validation ran all 33 code cells on synthetic fixtures and checked five
Streamlit scenarios, notebook/app feature agreement, and seven-day forecast input
availability. Those checks do not measure clinical prediction accuracy. Synthetic
data, fitted test models, and synthetic model scores are excluded from delivery.

## Run on Mac

From this folder, with Python 3.12 installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Select the virtual environment's Python kernel. Put the supplied
`Medical_appointment_data (1).csv` alongside the notebook, then run all cells in
order. Change `data_path` in Step 3 if your CSV is elsewhere.

For Google Colab, upload the notebook and CSV, install the requirements in that
runtime, restart after package changes, and run the notebook in order. Download
the generated `models/` and `reports/` folders to use the application locally.

After running all 20 steps:

```bash
python -m streamlit run app.py
```

## How the design stays rigorous

- The no-show class is the positive class. Required F1 and ROC-AUC are not replaced
  by majority-class accuracy.
- Complete dates separate training, model selection, probability calibration,
  threshold selection, and final testing.
- Imputation and category encoding are fitted inside each training pipeline.
- Same-day observed weather, uncertain location codes, and untimestamped SMS
  receipt are excluded from advance prediction.
- Forecasts compare seasonal naive, exponential smoothing, SARIMA, and gradient
  boosting using seven-day blocks.
- The gradient-boosting forecast uses lags of at least seven days. Thus all inputs
  for the next seven days are known at the forecast origin; recursion is unnecessary.
- Forecast models are selected before test evaluation. Separate later validation
  dates estimate empirical error bands. Their coverage is measured on test data.
- Overall forecasts sum specialty predictions, including unknown specialty.
- The final status table compares actual test scores to the brief's strict goals.
  Scores are not guaranteed and failed goals stay visible.

F1 > 0.70, ROC-AUC > 0.75, MAPE < 20%, and R² > 0.65 remain the evaluation goals.
Simpler code does not, by itself, increase predictive accuracy.
