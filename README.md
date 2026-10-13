# ⚡ EV Charging Demand Prediction and Usage Pattern Analysis

A machine learning project that analyzes electric vehicle (EV) charging behavior and predicts charging demand using regression and classification models. The project includes an interactive Streamlit dashboard for exploring charging patterns, generating predictions, and understanding model performance.

## 📌 Project Overview

As electric vehicle adoption grows, understanding charging demand is important for charging station planning, resource allocation, and operational efficiency.

This project uses historical EV charging records to analyze usage patterns and build machine learning models that:

- Predict numerical charging demand for a charging session.
- Classify charging sessions as **High Demand** or **Normal Demand**.
- Explore demand patterns across locations, charger types, and time periods.
- Explain model predictions using feature importance, SHAP, and LIME.
- Present results through an interactive Streamlit dashboard.

## 🎯 Objectives

- Perform data cleaning, preprocessing, and exploratory data analysis (EDA).
- Engineer useful time-based and categorical features.
- Compare multiple regression and classification algorithms.
- Tune and evaluate the best-performing models.
- Interpret model predictions using explainable AI techniques.
- Develop an interactive dashboard for model inference and data exploration.

## 📊 Dataset

**Dataset:** Charging Records

The dataset contains approximately 72,000 charging sessions.

Key variables include:

| Feature | Description |
|---|---|
| `UserID` | User identifier |
| `ChargerID` | Charging station or charger identifier |
| `ChargerCompany` | Charging company |
| `Location` | Charging location category |
| `ChargerType` | Charger type |
| `StartDatetime` | Charging session start timestamp |
| `EndDatetime` | Charging session end timestamp |
| `Duration` | Charging session duration |
| `Demand` | Numerical charging demand |

The dataset is used to analyze charging behavior and develop the regression and classification models.

## 🧹 Data Analysis and Feature Engineering

The project includes:

- Dataset inspection and data-quality checks.
- Missing-value and duplicate checks.
- Analysis of demand distributions and outliers.
- Demand comparisons by charger type and location.
- Analysis of demand by hour and other time-related features.
- Feature engineering using variables such as hour, day of week, month, weekend status, and time of day.

## 🤖 Machine Learning Models

### 1. Regression — Charging Demand Prediction

The regression model predicts the numerical demand associated with a charging session.

Multiple regression algorithms were compared, and a tuned XGBoost regression pipeline was selected for the dashboard.

**Evaluation metrics:**

| Metric | Result |
|---|---:|
| MAE | 3.30 |
| RMSE | 5.88 |
| R² Score | 0.808 |

These results indicate that the model explains approximately 80.8% of the variance in the evaluated test data.

### 2. Classification — High-Demand Session Prediction

The classification model predicts whether a charging session belongs to the High Demand or Normal Demand category.

The project compares algorithms including Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, and LightGBM. A tuned XGBoost classifier is used for the dashboard.

**Evaluation metrics:**

| Metric | Result |
|---|---:|
| Accuracy | 89.94% |
| Precision | 93.27% |
| Recall | 86.07% |
| F1-score | 89.52% |
| ROC-AUC | 96.00% |

The classifier demonstrates strong performance on the held-out test dataset.

### Confusion Matrix

| Actual / Predicted | High Demand | Normal Demand |
|---|---:|---:|
| High Demand | 6,836 | 452 |
| Normal Demand | 1,014 | 6,264 |

The confusion matrix shows the number of correctly classified sessions and the errors made by the model.

## 🔍 Explainable AI (XAI)

The project uses model interpretation techniques to understand the factors influencing predictions.

- **Feature Importance:** Identifies features that contribute most to the model.
- **SHAP:** Helps explain feature contributions to individual predictions and overall model behavior.
- **LIME:** Provides local explanations for individual predictions.

The regression feature-importance analysis identified charger type and charging duration as the two most influential features in the evaluated model.

## 📈 Streamlit Dashboard

The dashboard provides an interactive interface for exploring the dataset and using the trained machine learning models.

### Dashboard Pages

**Page 1 — Overview and EDA**
- Key performance indicators (KPIs).
- Charging demand distribution.
- Demand by charger type and location.
- Time-based demand visualizations.
- Key exploratory findings.

**Page 2 — Regression**
- Input charging-session details.
- Numerical charging-demand prediction.
- Regression model performance metrics.
- Submitted input summary.

**Page 3 — Classification**
- Input charging-session details.
- High Demand or Normal Demand prediction.
- Predicted class probabilities.
- Classification metrics and confusion matrix.
- Submitted input summary.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming and model development |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, pipelines, and evaluation |
| XGBoost | Regression and classification |
| LightGBM | Model comparison |
| SHAP | Model explainability |
| LIME | Local prediction explanations |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web dashboard |
| Jupyter Notebook | Model development and experimentation |

## 📁 Project Structure

```text
EV_Charging/
│
├── app.py
├── requirements.txt
├── ChargingRecords.csv
├── EV_Charging_Demand_Prediction.ipynb
│
├── final_xgboost_demand_model.pkl
├── final_xgboost_demand_classifier.pkl
├── demand_label_encoder.pkl
│
└── pages/
    ├── 2_Regression.py
    └── 3_Classification.py
```

The `.pkl` files contain saved model artifacts and the label encoder required by the application. Keep the filenames consistent with the loading paths used in the Python code.


## 💾 Saved Models

The project uses saved model artifacts to avoid retraining every time the dashboard starts.

- `final_xgboost_demand_model.pkl` — trained regression pipeline.
- `final_xgboost_demand_classifier.pkl` — trained classification model.
- `demand_label_encoder.pkl` — encoder for demand categories.

The application expects these files to be available at the paths specified in its code.

**Compatibility note:** Scikit-learn and related package versions should be kept compatible with the versions used to train the saved models. Loading models across different library versions may generate warnings or cause compatibility issues.

Only load trusted `.pkl` files because pickle-based files can execute code when deserialized.

## 📌 Key Findings

- Charging demand is right-skewed, with many observations concentrated at lower-to-moderate demand levels.
- Charger type and charging duration were the leading features in the regression model's feature-importance analysis.
- Demand varies across locations and time periods.
- The tuned regression model achieved an R² score of 0.808.
- The tuned classification model achieved approximately 89.94% accuracy and a 0.96 ROC-AUC score.

## ⚠️ Limitations

- Model performance is based on the available dataset and its test split.
- Predictions may differ for charging sessions with characteristics not well represented in the training data.
- Feature importance and explainability methods indicate model associations and contributions; they do not establish causation.
- Real-world deployment would require monitoring data quality and evaluating performance on new charging records.


## 👨‍💻 Author

**Umesh Eshwar Reddy**

Machine Learning Project — EV Charging Demand Prediction and Usage Pattern Analysis