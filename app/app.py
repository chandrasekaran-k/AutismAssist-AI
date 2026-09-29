import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCREEN_MODEL = ROOT / "models/asd_screening_model.joblib"
THERAPY_MODEL = ROOT / "models/therapy_recommendation_model.joblib"

st.set_page_config(page_title="AutismAssist AI", page_icon="🧩", layout="wide")

st.title("🧩 AutismAssist AI")
st.caption("Screening support + therapy recommendation prototype")

st.warning(
    "This application is a research/portfolio prototype. "
    "It is NOT a diagnostic tool and does not replace assessment by a qualified professional."
)

tab1, tab2 = st.tabs(["ASD Screening", "Therapy Support"])

score_features = [
    "Age", "Gender", "Eye_Contact", "Response_to_Name",
    "Communication_Level", "Joint_Attention", "Sensory_Regulation",
    "Motor_Skills", "ADL_Skills", "Hyperactivity_Control",
    "Social_Interaction"
]

def scale_help():
    st.caption("Behavioral/support scores use the dataset's 1–5 scale; verify the scale definition before clinical use.")

with tab1:
    st.subheader("Screening Support")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=18, value=8)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        eye = st.slider("Eye Contact", 1, 5, 3)
        response = st.slider("Response to Name", 1, 5, 3)
    with c2:
        communication = st.slider("Communication Level", 1, 5, 3)
        joint = st.slider("Joint Attention", 1, 5, 3)
        sensory = st.slider("Sensory Regulation", 1, 5, 3)
        motor = st.slider("Motor Skills", 1, 5, 4)
    with c3:
        adl = st.slider("ADL Skills", 1, 5, 4)
        hyper = st.slider("Hyperactivity Control", 1, 5, 4)
        social = st.slider("Social Interaction", 1, 5, 3)

    scale_help()

    if st.button("Run Screening", type="primary"):
        model = joblib.load(SCREEN_MODEL)
        profile = pd.DataFrame([{
            "Age": age, "Gender": gender, "Eye_Contact": eye,
            "Response_to_Name": response, "Communication_Level": communication,
            "Joint_Attention": joint, "Sensory_Regulation": sensory,
            "Motor_Skills": motor, "ADL_Skills": adl,
            "Hyperactivity_Control": hyper, "Social_Interaction": social
        }])
        probability = float(model.predict_proba(profile)[0, 1])
        label = "Higher screening risk" if probability >= 0.5 else "Lower screening risk"
        st.metric("Model output", label)
        st.progress(min(max(probability, 0.0), 1.0))
        st.write(f"Model-estimated probability of ASD_Status=1: **{probability:.1%}**")
        st.info("Use this result only as a screening-support signal; professional assessment is required.")

with tab2:
    st.subheader("Therapy Support")
    st.write("Enter the same developmental/support profile to generate model-based support areas.")
    vals = {}
    cols = st.columns(3)
    with cols[0]:
        vals["Age"] = st.number_input("Age ", min_value=1, max_value=18, value=8)
        vals["Gender"] = st.selectbox("Gender ", ["Male", "Female", "Other"])
        vals["Eye_Contact"] = st.slider("Eye Contact ", 1, 5, 3)
        vals["Response_to_Name"] = st.slider("Response to Name ", 1, 5, 3)
    with cols[1]:
        vals["Communication_Level"] = st.slider("Communication Level ", 1, 5, 3)
        vals["Joint_Attention"] = st.slider("Joint Attention ", 1, 5, 3)
        vals["Sensory_Regulation"] = st.slider("Sensory Regulation ", 1, 5, 3)
        vals["Motor_Skills"] = st.slider("Motor Skills ", 1, 5, 4)
    with cols[2]:
        vals["ADL_Skills"] = st.slider("ADL Skills ", 1, 5, 4)
        vals["Hyperactivity_Control"] = st.slider("Hyperactivity Control ", 1, 5, 4)
        vals["Social_Interaction"] = st.slider("Social Interaction ", 1, 5, 3)
        vals["ASD_Status"] = st.selectbox("ASD Status (dataset field)", [0, 1])

    if st.button("Generate Support Areas"):
        model = joblib.load(THERAPY_MODEL)
        X = pd.DataFrame([vals])
        pred = model.predict(X)[0]
        names = [
            "Speech Therapy", "Occupational Therapy",
            "Behaviour Therapy", "Parent Training",
            "Social Skills Support"
        ]
        recommended = [n for n, v in zip(names, pred) if int(v) == 1]
        if recommended:
            st.success("Model-identified support areas:")
            for item in recommended:
                st.write(f"• {item}")
        else:
            st.info("No support area was predicted at the model threshold.")
        st.caption("These outputs are model predictions for a portfolio prototype, not treatment recommendations.")
