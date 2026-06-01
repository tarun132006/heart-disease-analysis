import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ── Page Config ───────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# ── Train Model ───────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('data/stroke_cleaned.csv')
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model, scaler

model, scaler = train_model()

# ── Header ────────────────────────────────────
st.title("🧠 Stroke Risk Predictor")
st.markdown("Answer the questions below to assess your stroke risk.")
st.divider()

# ── Page Selection ────────────────────────────
page = st.sidebar.radio("Select Mode", ["👤 Quick Check (Public)", "🔬 Clinical Assessment"])

# ══════════════════════════════════════════════
# PAGE 1 — QUICK CHECK (PUBLIC)
# ══════════════════════════════════════════════
if page == "👤 Quick Check (Public)":
    st.subheader("👤 Quick Health Check")
    st.markdown("Answer simple questions — no medical knowledge needed.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Your Age",
            min_value=1, max_value=100,
            value=None,
            placeholder="Enter your age"
        )
        weight = st.number_input(
            "Weight (kg)",
            min_value=10.0, max_value=300.0,
            value=None, step=0.5,
            placeholder="Enter your weight in kg"
        )
        height = st.number_input(
            "Height (cm)",
            min_value=50.0, max_value=250.0,
            value=None, step=0.5,
            placeholder="Enter your height in cm"
        )
        gender = st.selectbox("Gender", ["Select...", "Male", "Female"])

    with col2:
        hypertension  = st.radio("Do you have high blood pressure?", ["No", "Yes"])
        heart_disease = st.radio("Do you have any heart disease?", ["No", "Yes"])
        ever_married  = st.radio("Have you ever been married?", ["No", "Yes"])
        smoking       = st.selectbox(
            "Smoking Status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"]
        )

    # Show BMI only when weight and height are filled
    if weight and height:
        bmi = weight / ((height / 100) ** 2)
        st.info(f"📊 Your calculated BMI: **{bmi:.1f}**")
    else:
        bmi = None
        st.info("📊 BMI will be calculated automatically once you enter weight and height")

    st.divider()

    if st.button("🔍 Check My Risk", use_container_width=True):

        # Validate inputs
        if not age or not weight or not height or gender == "Select...":
            st.warning("⚠️ Please fill in all fields before checking your risk.")
        else:
            # Encode inputs
            gender_enc    = 1 if gender == "Male" else 0
            married_enc   = 1 if ever_married == "Yes" else 0
            hypert_enc    = 1 if hypertension == "Yes" else 0
            heart_enc     = 1 if heart_disease == "Yes" else 0
            smoking_map   = {"never smoked": 2, "formerly smoked": 1, "smokes": 3, "Unknown": 0}
            smoking_enc   = smoking_map[smoking]
            work_enc      = 2   # Default: Private
            residence_enc = 1   # Default: Urban
            avg_glucose   = 100.0  # Default average

            patient = pd.DataFrame([[gender_enc, age, hypert_enc, heart_enc,
                                      married_enc, work_enc, residence_enc,
                                      avg_glucose, bmi, smoking_enc]],
                                   columns=['gender', 'age', 'hypertension', 'heart_disease',
                                            'ever_married', 'work_type', 'Residence_type',
                                            'avg_glucose_level', 'bmi', 'smoking_status'])

            patient_scaled = scaler.transform(patient)
            probability    = model.predict_proba(patient_scaled)[0][1]
            prediction     = 1 if probability > 0.30 else 0

            st.subheader("Your Results")
            st.progress(float(probability))
            st.metric("Stroke Risk Score", f"{probability*100:.1f}%")

            if prediction == 1:
                st.error("⚠️ HIGHER RISK — Please consult a doctor for a proper assessment")
            else:
                st.success("✅ LOWER RISK — Keep maintaining a healthy lifestyle!")

            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Age", age)
            col2.metric("BMI", f"{bmi:.1f}")
            col3.metric("Smoking", smoking)

            st.caption("⚠️ This is not a medical diagnosis. Always consult a healthcare professional.")

# ══════════════════════════════════════════════
# PAGE 2 — CLINICAL ASSESSMENT
# ══════════════════════════════════════════════
else:
    st.subheader("🔬 Clinical Assessment")
    st.markdown("For healthcare professionals — enter full patient details.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        gender        = st.selectbox("Gender", ["Select...", "Male", "Female"])
        age           = st.number_input(
            "Patient Age",
            min_value=1, max_value=100,
            value=None,
            placeholder="Enter patient's age"
        )
        hypertension  = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married  = st.selectbox("Ever Married", ["Yes", "No"])

    with col2:
        work_type   = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt job", "children", "Never worked"]
        )
        residence   = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose = st.number_input(
            "Average Glucose Level (mg/dL)",
            min_value=50.0, max_value=300.0,
            value=None, step=0.1,
            placeholder="Enter glucose level"
        )
        bmi         = st.number_input(
            "BMI",
            min_value=10.0, max_value=70.0,
            value=None, step=0.1,
            placeholder="Enter patient BMI"
        )
        smoking     = st.selectbox(
            "Smoking Status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"]
        )

    st.divider()

    if st.button("🔍 Predict Stroke Risk", use_container_width=True):

        # Validate inputs
        if not age or not avg_glucose or not bmi or gender == "Select...":
            st.warning("⚠️ Please fill in all fields before predicting.")
        else:
            gender_enc    = 1 if gender == "Male" else 0
            married_enc   = 1 if ever_married == "Yes" else 0
            hypert_enc    = 1 if hypertension == "Yes" else 0
            heart_enc     = 1 if heart_disease == "Yes" else 0
            smoking_map   = {"never smoked": 2, "formerly smoked": 1, "smokes": 3, "Unknown": 0}
            smoking_enc   = smoking_map[smoking]
            work_map      = {"Private": 2, "Self-employed": 3, "Govt job": 0, "children": 1, "Never worked": 4}
            work_enc      = work_map[work_type]
            residence_enc = 1 if residence == "Urban" else 0

            patient = pd.DataFrame([[gender_enc, age, hypert_enc, heart_enc,
                                      married_enc, work_enc, residence_enc,
                                      avg_glucose, bmi, smoking_enc]],
                                   columns=['gender', 'age', 'hypertension', 'heart_disease',
                                            'ever_married', 'work_type', 'Residence_type',
                                            'avg_glucose_level', 'bmi', 'smoking_status'])

            patient_scaled = scaler.transform(patient)
            probability    = model.predict_proba(patient_scaled)[0][1]
            prediction     = 1 if probability > 0.30 else 0

            st.subheader("Clinical Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Probability", f"{probability*100:.1f}%")
            col2.metric("Glucose Level", avg_glucose)
            col3.metric("BMI", bmi)

            st.progress(float(probability))

            if prediction == 1:
                st.error("⚠️ HIGH RISK — Patient shows indicators of stroke risk")
            else:
                st.success("✅ LOW RISK — Patient shows no strong indicators of stroke risk")

            st.divider()
            st.caption("Model: Random Forest with SMOTE balancing · class_weight=balanced · Dataset: Kaggle Stroke Prediction")

# ── Footer ────────────────────────────────────
st.divider()
st.caption("🧠 Stroke Risk Predictor · Built with Streamlit · Not a substitute for medical advice")