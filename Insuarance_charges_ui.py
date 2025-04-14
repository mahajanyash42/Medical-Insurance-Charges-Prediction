import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model.pkl')
st.title("Medical Insurance Charges Predictor 💰")

# User input
age = st.slider('Age', 18, 100)
bmi = st.number_input('BMI', value=25.0)
children = st.number_input('Number of Children', min_value=0, max_value=10, step=1)
smoker = st.selectbox('Smoker', ['Yes', 'No'])

# Prepare data for prediction
smoker_encoded = 1 if smoker == 'Yes' else 0
input_data = np.array([[age, bmi, children, smoker_encoded]])

# Prediction
if st.button('Predict Charges'):
    prediction = model.predict(input_data)
    st.success(f'Estimated Insurance Charges: ₹{prediction[0]:,.2f}')

