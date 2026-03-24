import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config for a professional look
st.set_page_config(page_title="Telco Churn Analytics", layout="wide", page_icon="📞")

# Load assets
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

# Custom CSS to make it look modern
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("📞 Customer Retention Dashboard")
st.markdown("---")

# Layout: 1/3 for Inputs, 2/3 for Results
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Customer Profile")
    with st.container():
        tenure = st.slider("Tenure (Months with Company)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        tech_support = st.radio("Has Tech Support?", ["Yes", "No"], horizontal=True)
        
        st.markdown("---")
        st.subheader("💸 Financials")
        monthly = st.number_input("Monthly Charges ($)", value=65.0)
        total = st.number_input("Total Charges ($)", value=500.0)
        
        predict_btn = st.button("Analyze Churn Risk")

with col2:
    st.subheader("📊 Prediction Analysis")
    if predict_btn:
        # 1. Process Input
        input_df = pd.DataFrame(0, index=[0], columns=features)
        input_df['tenure'] = tenure
        input_df['MonthlyCharges'] = monthly
        input_df['TotalCharges'] = total
        
        # Match Categorical Encodings (Ensure these match your training dummies!)
        if contract == "One year": input_df['Contract_One year'] = 1
        if contract == "Two year": input_df['Contract_Two year'] = 1
        if internet == "Fiber optic": input_df['InternetService_Fiber optic'] = 1
        if internet == "No": input_df['InternetService_No'] = 1
        if tech_support == "Yes": input_df['TechSupport_Yes'] = 1

        # 2. Predict
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        # 3. Visual Display
        with st.container():
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            if prediction == 1:
                st.error(f"### ⚠️ High Risk of Churn")
                st.progress(probability)
                st.write(f"The model is **{probability:.1%}** confident that this customer will leave.")
                st.warning("Strategy: Offer a long-term contract discount or technical support bundle.")
            else:
                st.success(f"### ✅ Low Risk of Churn")
                st.progress(probability)
                st.write(f"The model is **{1-probability:.1%}** confident that this customer will stay.")
                st.info("Strategy: Monitor usage patterns but no immediate intervention needed.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.info("👈 Enter customer details and click 'Analyze' to see the prediction.")

# Sidebar info
st.sidebar.header("About the Model")
st.sidebar.info("This app uses a **Logistic Regression** model trained on the Telco Churn dataset. It achieved an accuracy of 81% and prioritizes high Recall to identify at-risk customers.")
