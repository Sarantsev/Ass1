import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
feature_names = data.feature_names

MODEL_PATH = '/models/model.pkl'
SCALER_PATH = '/models/scaler.pkl'
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

st.title("Breast Cancer Prediction ")

st.header("Input Features", anchor=False)
user_input = []
with st.form("input_form", clear_on_submit=False):
    for feature in feature_names:
        val = st.number_input(
            f"{feature}",
            value=float(np.median(data.data[:, list(feature_names).index(feature)])),
            step=0.01,
            format="%.4f"
        )
        user_input.append(val)
    submitted = st.form_submit_button("Predict")

if 'submitted' in locals() and submitted:
    sample = np.array(user_input).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0]

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {'Malignant' if pred == 0 else 'Benign'}")
    st.write("**Probabilities:**")
    st.json({"Malignant": float(prob[0]), "Benign": float(prob[1])})
else:
    st.info("Enter feature values above and click Predict.")