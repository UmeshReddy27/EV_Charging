
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

st.set_page_config(
    page_title="Demand Classification",
    page_icon="🎯",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "final_xgboost_demand_classifier.pkl"
ENCODER_PATH = BASE_DIR / "demand_label_encoder.pkl"
DATA_PATH = BASE_DIR / "ChargingRecords.csv"


# --------------------------------------------------
# LOAD MODEL, ENCODER AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)



@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    # Find the original datetime column
    datetime_col = None

    for col in ["StartDatetime", "StartDateTime"]:
        if col in data.columns:
            datetime_col = col
            break

    # Otherwise combine the original date and time columns
    if datetime_col is None:
        if {"StartDay", "StartTime"}.issubset(data.columns):
            data["StartDatetime"] = (
                data["StartDay"].astype(str) + " " +
                data["StartTime"].astype(str)
            )
            datetime_col = "StartDatetime"

    if datetime_col is not None:
        # Explicitly convert to datetime before using .dt
        data[datetime_col] = pd.to_datetime(
            data[datetime_col],
            errors="coerce"
        )

        dt = data[datetime_col]

        data["Hour"] = dt.dt.hour
        data["DayOfWeek"] = dt.dt.dayofweek
        data["Month"] = dt.dt.month
        data["IsWeekend"] = (
            dt.dt.dayofweek >= 5
        ).astype(int)

        hour = dt.dt.hour

        data["TimeOfDay"] = np.select(
            [
                hour.between(5, 11),
                hour.between(12, 16),
                hour.between(17, 20)
            ],
            ["Morning", "Afternoon", "Evening"],
            default="Night"
        )

    return data


try:
    model = load_model(str(MODEL_PATH))
    label_encoder = load_model(str(ENCODER_PATH))
    df = load_data(str(DATA_PATH))

except Exception as e:
    st.error(f"Failed to load project files: {e}")
    st.stop()


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("🎯 EV Charging Demand Classification")

st.caption(
    "Classify charging sessions into High Demand or Normal Demand."
)

features = [
    "Duration",
    "Hour",
    "DayOfWeek",
    "Month",
    "IsWeekend",
    "ChargerCompany",
    "Location",
    "ChargerType",
    "TimeOfDay"
]

missing = [col for col in features if col not in df.columns]

if missing:
    st.error(f"Required columns are missing: {missing}")
    st.stop()


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.subheader("Enter Charging Session Details")

with st.form("classification_form"):

    c1, c2, c3 = st.columns(3)

    with c1:
        duration = st.number_input(
            "Charging Duration",
            min_value=0.0,
            max_value=max(1.0, float(df["Duration"].max())),
            value=float(df["Duration"].median()),
            step=1.0
        )

        hour = st.slider("Hour of Day", 0, 23, 18)

        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
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
            options=sorted(
                df["ChargerType"].dropna().unique().tolist()
            )
        )

    with c3:
        location = st.selectbox(
            "Location",
            options=sorted(
                df["Location"].dropna().unique().tolist()
            )
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
        "Classify Demand",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# RUN PREDICTION ONLY WHEN SUBMITTED
# --------------------------------------------------

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
    }], columns=features)

    try:
        raw_prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        model_classes = np.asarray(model.classes_, dtype=int)

        probability_labels = label_encoder.inverse_transform(
            model_classes
        )

        predicted_label = label_encoder.inverse_transform(
            [int(raw_prediction)]
        )[0]

        st.session_state["classification_result"] = {
            "prediction": predicted_label,
            "probabilities": probabilities.tolist(),
            "probability_labels": probability_labels.tolist(),
            "inputs": input_df.to_dict(orient="records")[0]
        }

    except Exception as e:
        st.error(f"Classification failed: {e}")


# --------------------------------------------------
# DISPLAY PERSISTENT PREDICTION RESULT
# --------------------------------------------------

result = st.session_state.get("classification_result")

if result is not None:

    st.divider()
    st.subheader("Classification Result")

    predicted_label = result["prediction"]

    if predicted_label == "High Demand":
        st.warning("⚡ Predicted Class: HIGH DEMAND")
    else:
        st.success("✅ Predicted Class: NORMAL DEMAND")

    probability_labels = result["probability_labels"]
    probabilities = result["probabilities"]

    probability_map = dict(zip(probability_labels, probabilities))

    high_probability = probability_map.get("High Demand", 0) * 100
    normal_probability = probability_map.get("Normal Demand", 0) * 100

    p1, p2 = st.columns(2)

    p1.metric(
        "High Demand Probability",
        f"{high_probability:.2f}%"
    )

    p2.metric(
        "Normal Demand Probability",
        f"{normal_probability:.2f}%"
    )

    probability_df = pd.DataFrame({
        "Demand Category": probability_labels,
        "Probability (%)": [
            p * 100 for p in probabilities
        ]
    })

    st.bar_chart(
        probability_df.set_index("Demand Category")
    )

    with st.expander("View submitted model inputs"):
        st.dataframe(
            pd.DataFrame([result["inputs"]]),
            use_container_width=True
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()
st.subheader("Classification Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric("Accuracy", "89.94%")
m2.metric("Precision", "93.27%")
m3.metric("Recall", "86.07%")

m4, m5 = st.columns(2)

m4.metric("F1 Score", "89.52%")
m5.metric("ROC-AUC", "96.00%")

st.caption(
    "Metrics are from the tuned XGBoost classifier evaluation "
    "recorded in the project notebook."
)



# --------------------------------------------------
# CACHED CONFUSION MATRIX
# --------------------------------------------------

@st.cache_data
def get_confusion_matrix():
    return np.array([
        [6836, 452],
        [1014, 6264]
    ])


st.subheader("Confusion Matrix — Tuned XGBoost")

cm = get_confusion_matrix()

confusion_df = pd.DataFrame(
    cm,
    index=["Actual High Demand", "Actual Normal Demand"],
    columns=["Predicted High Demand", "Predicted Normal Demand"]
)

st.dataframe(
    confusion_df,
    use_container_width=True
)

st.caption(
    "This confusion matrix represents the recorded test evaluation, "
    "not the individual prediction above."
)
