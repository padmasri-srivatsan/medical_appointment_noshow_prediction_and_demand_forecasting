"""Start from this folder: python -m streamlit run app.py"""
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


folder = Path(__file__).resolve().parent
st.set_page_config(page_title="Appointment Planning", layout="wide")
st.title("Medical Appointment Planning")


@st.cache_resource
def load_models():
    classifier = joblib.load(folder / "models" / "no_show_model.joblib")
    forecaster = joblib.load(folder / "models" / "demand_model.joblib")
    return classifier, forecaster


if not (folder / "models" / "no_show_model.joblib").exists() or not (folder / "models" / "demand_model.joblib").exists():
    st.info("Run all 20 notebook steps on the original CSV first. The notebook will create the models/ folder.")
    st.stop()

classifier, forecaster = load_models()
st.caption(
    f"Historical data end: {forecaster['last_date']}. "
    "Forecasts start after that date; this is not a live clinic feed."
)

status_path = folder / "reports" / "target_status.csv"
if status_path.exists():
    target_status = pd.read_csv(status_path)
    with st.expander("Measured test results against the project goals"):
        st.dataframe(target_status, hide_index=True, width="stretch")
    if not target_status["Target_met"].all():
        st.warning("Some performance goals were not achieved. Review the actual test results before operational use.")

risk_tab, demand_tab = st.tabs(["No-show prediction", "Demand forecast"])

with risk_tab:
    st.subheader("Estimate appointment no-show risk")
    st.caption("Enter information already known before the appointment. Weather, SMS receipt, and location are excluded.")

    with st.form("appointment"):
        left, right = st.columns(2)
        example_date = pd.Timestamp(classifier["last_date"]).date() + pd.Timedelta(days=1)
        appointment_date = left.date_input("Appointment date", value=example_date)
        appointment_hour = right.number_input("Appointment hour", 0, 23, 9)
        age = left.number_input("Recorded age", 0, 120, 18)
        age_unknown = right.checkbox("Age is unknown")

        categories = {}
        for column, options in classifier["category_choices"].items():
            categories[column] = st.selectbox(column.replace("_", " ").title(), options)

        st.write("Recorded flags: checked means 1, unchecked means 0.")
        st.caption("Use recorded age flags; the source flags sometimes conflict with recorded age.")
        flags = {}
        for index, column in enumerate(classifier["binary_columns"]):
            position = left if index % 2 == 0 else right
            flags[column] = int(position.checkbox(column.replace("_", " ")))
        predict_clicked = st.form_submit_button("Predict no-show risk")

    if predict_clicked:
        start = time.perf_counter()
        # These are the same two calendar features created in notebook Step 11.
        appointment = {
            "appointment_time": appointment_hour,
            "age": np.nan if age_unknown else age,
            "weekday": pd.Timestamp(appointment_date).dayofweek,
            "month": pd.Timestamp(appointment_date).month,
            **categories, **flags,
        }
        input_data = pd.DataFrame([appointment])[classifier["columns"]]
        with st.spinner("Calculating risk..."):
            raw_probability = classifier["model"].predict_proba(input_data)[:, [1]]
            probability = float(classifier["probability_adjustment"].predict_proba(raw_probability)[0, 1])

        st.metric("Estimated no-show probability", f"{probability:.1%}")
        if probability >= classifier["threshold"]:
            st.info("Above the validation-selected threshold: review for follow-up.")
        else:
            st.info("Below the validation-selected follow-up threshold.")
        st.caption(f"Threshold: {classifier['threshold']:.3f}. Calculation: {time.perf_counter() - start:.3f} seconds.")
        if pd.Timestamp(appointment_date) > pd.Timestamp(classifier["last_date"]) + pd.Timedelta(days=31):
            st.warning("This date is far beyond the historical dataset. Performance for that period has not been validated.")


# The same seven forecast features used in notebook Step 16.
def forecast_features(series):
    features = pd.DataFrame(index=series.index)
    for lag in [7, 14, 21, 28]:
        features[f"lag_{lag}"] = series.shift(lag)
    features["past_week_mean"] = series.shift(7).rolling(7).mean()
    features["weekday"] = series.index.dayofweek
    features["month"] = series.index.month
    return features


with demand_tab:
    st.subheader("Forecast daily booked appointments")
    choice = st.selectbox("Specialty", ["Overall"] + sorted(forecaster["models"]))
    days = st.slider("Days ahead", 1, 7, 7)
    dates = pd.date_range(pd.Timestamp(forecaster["last_date"]) + pd.Timedelta(days=1), periods=days)
    selected_specialties = list(forecaster["models"]) if choice == "Overall" else [choice]

    start = time.perf_counter()
    forecast_rows = []
    with st.spinner("Calculating demand forecast..."):
        for specialty in selected_specialties:
            history = forecaster["history"][specialty]
            method = forecaster["methods"][specialty]
            model = forecaster["models"][specialty]
            if method == "Seasonal naive":
                prediction = history.iloc[-7:].to_numpy()[:days]
            elif method in ["Exponential smoothing", "SARIMA"]:
                prediction = model.forecast(days)
            else:
                extended = pd.concat([history, pd.Series(np.nan, index=dates)])
                prediction = model.predict(forecast_features(extended).loc[dates])
            prediction = np.maximum(np.asarray(prediction, dtype=float), 0)
            forecast_rows.append(pd.DataFrame({"date": dates, "prediction": prediction}))

    result = pd.concat(forecast_rows).groupby("date", as_index=False)["prediction"].sum()
    width = forecaster["overall_width"] if choice == "Overall" else forecaster["widths"][choice]
    result["lower_90"] = np.maximum(0, result["prediction"] - width)
    result["upper_90"] = result["prediction"] + width
    calculation_time = time.perf_counter() - start

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result["date"], result["prediction"], "o-", label="Forecast bookings")
    ax.fill_between(result["date"], result["lower_90"], result["upper_90"], alpha=0.2, label="Empirical 90% interval")
    ax.set(xlabel="Appointment date", ylabel="Booked appointments", title=choice)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(result.round(2), hide_index=True, width="stretch")
    st.download_button("Download forecast", result.to_csv(index=False), "forecast.csv", "text/csv")
    st.caption(f"Forecast calculation: {calculation_time:.3f} seconds.")
    st.caption("Overall point forecasts equal the sum of specialties. Empirical interval coverage is measured in the notebook and is not guaranteed.")
