# Streamlit app: loads the committed model, collects inputs into a dataframe, and shows the prediction.
"""
Streamlit app for the 'Visit with Us' Wellness Tourism Package predictor.

Loads the trained model committed to this folder by the CI/CD pipeline,
collects customer details through a simple form, and shows whether the
model predicts the customer is likely to buy the Wellness package.
"""

import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧘", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🧘 Wellness Tourism Package Predictor")
st.write(
    "Enter a customer's profile and sales-interaction details to predict whether "
    "they are likely to purchase the new **Wellness Tourism Package**."
)

model = load_model()

with st.form("customer_form"):
    st.subheader("Customer details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
        occupation = st.selectbox(
            "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        designation = st.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        monthly_income = st.number_input("Monthly Income", min_value=0, value=20000, step=500)

    with col2:
        num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
        num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
        num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=30, value=2)
        preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0], index=0)
        passport = st.selectbox("Holds a Passport?", ["Yes", "No"])
        own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

    st.subheader("Sales interaction details")
    col3, col4 = st.columns(2)
    with col3:
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox(
            "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )
    with col4:
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=180, value=15)
        num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    st.subheader("Input summary")
    st.dataframe(input_df)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")
    if prediction == 1:
        st.success(f"✅ Likely to purchase the Wellness Tourism Package (probability: {probability:.1%})")
    else:
        st.warning(f"❌ Unlikely to purchase the Wellness Tourism Package (probability: {probability:.1%})")
