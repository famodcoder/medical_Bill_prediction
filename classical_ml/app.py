import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the Model and Columns ---
@st.cache_resource
def load_components():
    model = joblib.load('classical_ml/models/insurance_model.pkl')
    expected_columns = joblib.load('classical_ml/models/model_columns.pkl')
    return model, expected_columns

model, expected_columns = load_components()

# --- 2. Build the Web UI ---
st.title("🏥 Medical Insurance Bill Predictor")
st.write("Enter the patient's details below to estimate their insurance charges.")

# Create layout with columns for a cleaner look
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    bmi = st.number_input("BMI (e.g., 25.5)", min_value=10.0, max_value=60.0, value=25.0)
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5, 6])

with col2:
    sex = st.radio("Sex", ["male", "female"])
    smoker = st.radio("Smoker?", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# --- 3. Prediction Logic ---
if st.button("Calculate Estimated Bill"):
    
    # Map the UI inputs directly to the binary format the model learned
    input_data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }
    
    # Convert to DataFrame and enforce the exact column order from training
    input_df = pd.DataFrame([input_data])[expected_columns]
    
    # Make the prediction
    prediction = model.predict(input_df)
    
    # Display the result beautifully

    st.success(f"**Estimated Insurance Bill: ${prediction[0]:,.2f}**")
