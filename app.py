
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="EV Charging Analytics",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ChargingRecords.csv"

st.title("⚡ EV Charging Demand Analytics")
st.caption("Explore charging behavior, demand patterns and model insights.")

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    if "StartDatetime" in data.columns:
        data["StartDatetime"] = pd.to_datetime(
            data["StartDatetime"], errors="coerce"
        )

    if "StartDatetime" not in data.columns:
        if {"StartDay", "StartTime"}.issubset(data.columns):
            data["StartDatetime"] = pd.to_datetime(
                data["StartDay"].astype(str) + " " +
                data["StartTime"].astype(str),
                errors="coerce"
            )

    if "StartDatetime" in data.columns:
        data["Hour"] = data["StartDatetime"].dt.hour
        data["DayOfWeek"] = data["StartDatetime"].dt.dayofweek
        data["Month"] = data["StartDatetime"].dt.month
        data["IsWeekend"] = (
            data["StartDatetime"].dt.dayofweek >= 5
        ).astype(int)

        def time_of_day(hour):
            if pd.isna(hour):
                return "Unknown"
            if 5 <= hour < 12:
                return "Morning"
            elif 12 <= hour < 17:
                return "Afternoon"
            elif 17 <= hour < 21:
                return "Evening"
            return "Night"

        data["TimeOfDay"] = data["Hour"].apply(time_of_day)

    return data

try:
    df = load_data(str(DATA_PATH))
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

required_columns = {
    "Demand", "Duration", "Location", "ChargerType"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(f"Missing required columns: {sorted(missing_columns)}")
    st.stop()

if "Demand_Category" not in df.columns:
    demand_median = df["Demand"].median()
    df["Demand_Category"] = np.where(
        df["Demand"] >= demand_median,
        "High Demand",
        "Normal Demand"
    )

# Sidebar filters
st.sidebar.header("Dashboard Filters")

locations = sorted(df["Location"].dropna().unique().tolist())
selected_locations = st.sidebar.multiselect(
    "Location",
    locations,
    default=locations,
    placeholder="Select locations"
)

if selected_locations:
    filtered_df = df[df["Location"].isin(selected_locations)].copy()
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# KPI cards
st.subheader("Key Performance Indicators")

total_sessions = len(filtered_df)
average_demand = filtered_df["Demand"].mean()
average_duration = filtered_df["Duration"].mean()
high_demand_pct = (
    filtered_df["Demand_Category"].eq("High Demand").mean() * 100
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Charging Sessions", f"{total_sessions:,}")
c2.metric("Average Demand", f"{average_demand:.2f}")
c3.metric("Average Duration", f"{average_duration:.2f}")
c4.metric("High-Demand Sessions", f"{high_demand_pct:.2f}%")

st.divider()

# Demand distribution
st.subheader("Demand Distribution")

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        filtered_df["Demand"].dropna(),
        bins=30,
        kde=True,
        ax=ax
    )
    ax.set_title("Distribution of Charging Demand")
    ax.set_xlabel("Demand")
    ax.set_ylabel("Number of Sessions")
    st.pyplot(fig)
    plt.close(fig)

with right:
    demand_counts = filtered_df["Demand_Category"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    demand_counts.plot(kind="bar", ax=ax)
    ax.set_title("High vs Normal Demand")
    ax.set_xlabel("Demand Category")
    ax.set_ylabel("Number of Sessions")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

# Charger type and location analysis
st.subheader("Demand by Charger Type and Location")

left, right = st.columns(2)

with left:
    charger_demand = filtered_df.groupby(
        "ChargerType"
    )["Demand"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    charger_demand.plot(kind="bar", ax=ax)
    ax.set_title("Average Demand by Charger Type")
    ax.set_xlabel("Charger Type")
    ax.set_ylabel("Average Demand")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with right:
    location_demand = filtered_df.groupby(
        "Location"
    )["Demand"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    location_demand.plot(kind="bar", ax=ax)
    ax.set_title("Average Demand by Location")
    ax.set_xlabel("Location")
    ax.set_ylabel("Average Demand")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Time-based EDA
if "Hour" in filtered_df.columns and "StartDatetime" in filtered_df.columns:
    st.subheader("Demand Over Time")

    hourly_demand = filtered_df.groupby("Hour")["Demand"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    hourly_demand.plot(marker="o", ax=ax)
    ax.set_title("Average Demand by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Demand")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    daily_demand = (
        filtered_df.dropna(subset=["StartDatetime"])
        .set_index("StartDatetime")
        .resample("D")["Demand"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    daily_demand.plot(ax=ax)
    ax.set_title("Daily Average Charging Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Demand")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

# Key findings
st.divider()
st.subheader("Key EDA Findings")

st.markdown(
    """
    - Charging demand is right-skewed, with most sessions concentrated
      at lower-to-moderate demand values.
    - Charger type and charging duration were the two strongest
      predictors in the regression model's XGBoost feature-importance
      analysis.
    - Demand varies by location and time of day.
    - The daily demand series was stationary according to the
      Augmented Dickey-Fuller test in our notebook.
    """
)

st.caption(
    "EDA findings reflect the analysis performed on the project dataset. "
    "Use the filters to explore individual locations."
)
