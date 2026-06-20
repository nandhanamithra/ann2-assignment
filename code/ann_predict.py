import numpy as np
import joblib
import pandas as pd

model = joblib.load("obs_model.pkl")
scaler = joblib.load("obs_scaler.pkl")
encoders = joblib.load("obs_encoder.pkl")
target_encoder=joblib.load("obs_target_encoder.pkl")


age = float(input("Age: "))
gender = input("gender (male/female:) ")
height = float(input("height(m): "))
weight = float(input("weight: "))
calc = input("CALC (no/Sometimes/Frequently/Always): ")
favc = input("FAVC (yes/no): ")
fcvc = float(input("FCVC: "))
ncp = float(input("NCP: "))
scc = input("SCC (yes/no): ")
smoke = input("SMOKE (yes/no): ")
ch2o = float(input("CH2O: "))
family_history = input("family history with overweight (yes/no): ")
faf = float(input("FAF: "))
tue = float(input("TUE: "))
caec = input("CAEC (no/Sometimes/Frequently/Always): ")
mtrans = input("MTRANS (Walking/Bike/Motorbike/Public Transportation/Automobile): ")

new_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Height": height,
    "Weight": weight,
    "CALC": calc,
    "FAVC": favc,
    "FCVC": fcvc,
    "NCP": ncp,
    "SCC": scc,
    "SMOKE": smoke,
    "CH2O": ch2o,
    "family_history_with_overweight": family_history,
    "FAF": faf,
    "TUE": tue,
    "CAEC": caec,
    "MTRANS": mtrans
}])

for col in encoders:
    new_data[col] = encoders[col].transform(new_data[col])

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)[0]
class_id = np.argmax(prediction)
result = target_encoder.inverse_transform([class_id])[0]

print("\nPredicted Obesity Level:")
print(result)
