import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Insurance Price Prediction",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 600px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1.1rem;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title-text {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-align: center;
    }
    .prediction-text {
        font-size: 1.5rem;
        color: #1f77b4;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<div class="title-text">🏥 Insurance Price Prediction</div>', unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler'], model_data['feature_columns']

# Load model
model, scaler, feature_columns = load_model()

# Simple form layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    children = st.number_input("Number of Children", 0, 10, 0)

with col2:
    sex = st.selectbox("Gender", ["male", "female"])
    smoker = st.selectbox("Smoking Status", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prepare the input data
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Make prediction
def predict_charges(input_data):
    input_processed = pd.get_dummies(input_data)
    for col in feature_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0
    input_processed = input_processed[feature_columns]
    
    input_scaled = scaler.transform(input_processed)
    prediction = model.predict(input_scaled)[0]
    return prediction

# Prediction button and result
if st.button("Predict Insurance Charges"):
    with st.spinner('Calculating...'):
        prediction = predict_charges(input_data)
        st.markdown(f'<div class="prediction-text">Predicted Insurance Charges: ${prediction:,.2f}</div>', unsafe_allow_html=True) 