
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Demand Regression",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "final_xgboost_demand_model.pkl"
DATA_PATH = BASE_DIR / "ChargingRecords.csv"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    if "StartDatetime" in data.columns:
        data["StartDatetime"] = pd.to_datetime(
            data["StartDatetime"], errors="coerce"
        )

    if "TimeOfDay" not in data.columns and "StartDatetime" in data.columns:
        hour = data["StartDatetime"].dt.hour

        data["TimeOfDay"] = np.select(
            [
                hour.between(5, 11),
                hour.between(12, 16),
                hour.between(17, 20)
            ],
            ["Morning", "Afternoon", "Evening"],
            default="Night"
        )

    if "Hour" not in data.columns and "StartDatetime" in data.columns:
        data["Hour"] = data["StartDatetime"].dt.hour

    if "DayOfWeek" not in data.columns and "StartDatetime" in data.columns:
        data["DayOfWeek"] = data["StartDatetime"].dt.dayofweek

    if "Month" not in data.columns and "StartDatetime" in data.columns:
        data["Month"] = data["StartDatetime"].dt.month

    if "IsWeekend" not in data.columns and "StartDatetime" in data.columns:
        data["IsWeekend"] = (
            data["StartDatetime"].dt.dayofweek >= 5
        ).astype(int)

    return data

st.title("📈 EV Charging Demand Regression")
st.caption(
    "Predict numerical charging demand using the tuned XGBoost pipeline."
)

try:
    model = load_model(str(MODEL_PATH))
    df = load_data(str(DATA_PATH))
except Exception as e:
    st.error(f"Could not load the model or dataset: {e}")
    st.stop()

required = [
    "Duration", "Hour", "DayOfWeek", "Month", "IsWeekend",
    "ChargerCompany", "Location", "ChargerType", "TimeOfDay"
]

missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

st.subheader("Enter Charging Session Details")

with st.form("regression_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        duration = st.number_input(
            "Charging Duration",
            min_value=0.0,
            max_value=float(max(1, df["Duration"].max())),
            value=float(df["Duration"].median()),
            step=1.0
        )

        hour = st.slider("Hour of Day", 0, 23, 18)

        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ][x]
        )

    with c2:
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=8
        )

        is_weekend = int(day_of_week >= 5)

        st.text_input(
            "Weekend Status",
            value="Weekend" if is_weekend else "Weekday",
            disabled=True
        )

        charger_type = st.selectbox(
            "Charger Type",
            options=sorted(df["ChargerType"].dropna().unique().tolist())
        )

    with c3:
        location = st.selectbox(
            "Location",
            options=sorted(df["Location"].dropna().unique().tolist())
        )

        company = st.selectbox(
            "Charger Company",
            options=sorted(
                df["ChargerCompany"].dropna().unique().tolist()
            )
        )

        if 5 <= hour < 12:
            time_of_day = "Morning"
        elif 12 <= hour < 17:
            time_of_day = "Afternoon"
        elif 17 <= hour < 21:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"

        st.text_input(
            "Time of Day",
            value=time_of_day,
            disabled=True
        )

    submitted = st.form_submit_button(
        "Predict Charging Demand",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame([{
        "Duration": duration,
        "Hour": hour,
        "DayOfWeek": day_of_week,
        "Month": month,
        "IsWeekend": is_weekend,
        "ChargerCompany": company,
        "Location": location,
        "ChargerType": charger_type,
        "TimeOfDay": time_of_day
    }], columns=required)

    try:
        prediction = float(model.predict(input_df)[0])

        st.divider()
        st.subheader("Prediction Result")

        a, b = st.columns(2)

        with a:
            st.metric(
                "Predicted Charging Demand",
                f"{prediction:.2f}"
            )

        with b:
            st.metric(
                "Selected Location",
                str(location)
            )

        st.info(
            "This value is the model's numerical demand prediction. "
            "It is not a demand category or a probability."
        )

        with st.expander("View submitted model inputs"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.subheader("Regression Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", "3.30")
c2.metric("RMSE", "5.88")
c3.metric("R² Score", "0.808")

st.caption(
    "Metrics are from the tuned XGBoost regression evaluation "
    "recorded in the project notebook. They describe the evaluation "
    "dataset, not necessarily every future prediction."
)
