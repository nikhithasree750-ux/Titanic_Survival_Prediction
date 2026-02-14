import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
df = pd.read_csv("train.csv")

df = df.drop(["Name", "Ticket", "Cabin"], axis=1)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("Titanic Survival Prediction App")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses", 0, 8, 0)
parch = st.number_input("Number of Parents/Children", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked", ["Q", "S"])

# Convert inputs
sex = 0 if sex == "male" else 1
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = np.array([[0, pclass, sex, age, sibsp, parch, fare, embarked_Q, embarked_S]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("Passenger is likely to Survive")
    else:
        st.error("Passenger is unlikely to Survive")
