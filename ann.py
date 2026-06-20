import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input

data=pd.read_csv(r"C:\flutter_projects\INTERNSHIP\lvl 2\assignment\ann2\ObesityDataSet_raw_and_data_sinthetic.csv")

#print(data.head())
#print(data.columns)
encoders = {}
for column in data.select_dtypes(include='object').columns:
    if column != "NObeyesdad":
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        encoders[column] = le
target_encoder = LabelEncoder()
data["NObeyesdad"] = target_encoder.fit_transform(
    data["NObeyesdad"]
)
x = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42)

model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100)
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)

joblib.dump(model,"obs_model.pkl")
joblib.dump(scaler, "obs_scaler.pkl")
joblib.dump(encoders,"obs_encoder.pkl")
joblib.dump(target_encoder,"obs_target_encoder.pkl")
print("Model, scaler, and encoder saved successfully.")
