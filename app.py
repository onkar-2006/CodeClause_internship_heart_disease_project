import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load the pre-trained model and scaler
model = tf.keras.models.load_model('heart_disease_model (1).h5')
scaler = joblib.load('scaler (1).pkl')

# Function to make predictions
def predict_risk(input_data):
    input_df = pd.DataFrame([input_data])
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    return prediction[0][0]  # Return the predicted probability

# Streamlit UI
st.title("Heart Disease Risk Assessment")
st.write("Please input your health data below:")

# Input fields for the user
age = st.number_input("Age", min_value=0, max_value=120)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0)
chol = st.number_input("Cholesterol (mg/dl)", min_value=0)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

# Create a button for prediction
if st.button("Assess Risk"):
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    prediction = predict_risk(input_data)

    if prediction > 0.5:
        st.success("The model predicts a risk of heart disease.")
    else:
        st.success("The model predicts a low risk of heart disease.")

# Run the app with: streamlit run app.py



