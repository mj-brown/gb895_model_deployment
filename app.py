# Streamlit Community Cloud app for Healthy Meals churn prediction
# Co-authored with CoCo
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Healthy Meals Churn Predictor", layout="centered")



@st.cache_resource
def load_model():
    with open("churn_rf_healthy_meals.pkl", "rb") as f:
        model = pickle.load(f)
    with open("churn_encoder_healthy_meals.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


model, encoder = load_model()

st.title("Healthy Meals Churn Predictor")
st.markdown(
    "Enter customer attributes below to predict the probability of churn "
    "(not renewing their subscription)."
)

with st.form("prediction_form"):
    st.subheader("Customer Demographics")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income_level = st.selectbox(
            "Income Level", ["Low", "Medium", "High", "Very High"]
        )
        education = st.selectbox(
            "Education", ["High School", "Graduate", "Post-Graduate", "Other"]
        )

    with col2:
        device_type = st.selectbox(
            "Device Type", ["Desktop-only", "Mobile-only", "Multi-device"]
        )
        tech_comfort_score = st.number_input(
            "Tech Comfort Score (1-10)", min_value=1, max_value=10, value=5
        )

    st.subheader("Usage Metrics (2022)")
    col3, col4 = st.columns(2)

    with col3:
        tot_num_sessions = st.number_input(
            "Total Number of Sessions", min_value=0, value=50
        )
        gross_session_length = st.number_input(
            "Gross Session Length (minutes)", min_value=0, value=500
        )
        active_days = st.number_input(
            "Active Days", min_value=0, max_value=365, value=60
        )

    with col4:
        active_quarters = st.number_input(
            "Active Quarters", min_value=0, max_value=4, value=3
        )
        avg_sessions_per_quarter = st.number_input(
            "Avg Sessions per Active Quarter", min_value=0.0, value=16.0, step=1.0
        )

    submitted = st.form_submit_button("Predict Churn Probability")

if submitted:
    cat_df = pd.DataFrame(
        {
            "INCOME_LEVEL": [income_level],
            "EDUCATION": [education],
            "DEVICE_TYPE": [device_type],
        }
    )
    encoded_cats = encoder.transform(cat_df)
    encoded_df = pd.DataFrame(
        encoded_cats, columns=encoder.get_feature_names_out(cat_df.columns)
    )

    numeric_df = pd.DataFrame(
        {
            "AGE": [age],
            "TECH_COMFORT_SCORE": [tech_comfort_score],
            "TOT_NUM_SESSIONS_2022": [tot_num_sessions],
            "GROSS_SESSION_LENGTH_2022": [gross_session_length],
            "ACTIVE_DAYS_2022": [active_days],
            "ACTIVE_QUARTERS_2022": [active_quarters],
            "AVG_SESSIONS_PER_ACTIVE_QUARTER": [avg_sessions_per_quarter],
        }
    )

    features = pd.concat([numeric_df, encoded_df], axis=1)
    renewal_prob = model.predict_proba(features)[0][1]
    churn_prob = 1 - renewal_prob

    st.divider()
    col_a, col_b = st.columns(2)
    col_a.metric("Churn Probability", f"{churn_prob:.1%}")
    col_b.metric("Renewal Probability", f"{renewal_prob:.1%}")

    if churn_prob >= 0.7:
        st.error("HIGH RISK — This customer is very likely to churn.")
    elif churn_prob >= 0.4:
        st.warning("MODERATE RISK — This customer may churn.")
    else:
        st.success("LOW RISK — This customer is likely to renew.")
