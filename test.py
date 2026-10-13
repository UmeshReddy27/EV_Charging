import joblib

model = joblib.load("final_xgboost_demand_model.pkl")

print(type(model))
print(model)