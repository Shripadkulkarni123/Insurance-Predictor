import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

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

# Load and prepare data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    data = pd.read_csv(url)
    return data

# Load the data
data = load_data()

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

# Train the model
@st.cache_resource
def train_model():
    X = pd.get_dummies(data.drop('charges', axis=1))
    y = data['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns

# Train the model
model, scaler, feature_columns = train_model()

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