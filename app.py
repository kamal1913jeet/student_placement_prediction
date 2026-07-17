import streamlit as st
import pandas as pd
import joblib

# ===========================
# Load Model and Encoders
# ===========================
model = joblib.load("model/placement_model.pkl")
branch_encoder = joblib.load("model/branch_encoder.pkl")
tier_encoder = joblib.load("model/tier_encoder.pkl")

# ===========================
# Page Configuration
# ===========================
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Placement Prediction")
st.sidebar.title("About Project")

st.sidebar.info(
    """
    🎓 Student Placement Prediction

    Machine Learning Project

    Algorithm:
    Random Forest Classifier

    Purpose:
    Predict whether a student is likely
    to get placed based on academic,
    technical and skill-based factors.
    """
)

st.info(
    """
    This application uses Machine Learning to predict whether a student
    is likely to get placed based on academic performance,
    technical skills, internships, projects and other factors.
    """
)

st.write("Fill in the student details below to predict placement status.")

# ===========================
# User Inputs
# ===========================

branch = st.selectbox(
    "Select Branch",
    list(branch_encoder.classes_)
)

college_tier = st.selectbox(
    "Select College Tier",
    list(tier_encoder.classes_)
)

cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

backlogs = st.number_input(
    "Number of Backlogs",
    min_value=0,
    max_value=10,
    value=0
)

coding_skills = st.slider(
    "Coding Skills",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

dsa_score = st.slider(
    "DSA Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

aptitude_score = st.slider(
    "Aptitude Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

communication_skills = st.slider(
    "Communication Skills",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

ml_knowledge = st.slider(
    "Machine Learning Knowledge",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

system_design = st.slider(
    "System Design",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

internships = st.number_input(
    "Number of Internships",
    min_value=0,
    max_value=10,
    value=0
)

projects_count = st.number_input(
    "Number of Projects",
    min_value=0,
    max_value=20,
    value=0
)

certifications = st.number_input(
    "Number of Certifications",
    min_value=0,
    max_value=20,
    value=0
)

hackathons = st.number_input(
    "Hackathons Participated",
    min_value=0,
    max_value=20,
    value=0
)

open_source_contributions = st.number_input(
    "Open Source Contributions",
    min_value=0,
    max_value=20,
    value=0
)

extracurriculars = st.number_input(
    "Extracurricular Activities",
    min_value=0,
    max_value=10,
    value=0
)

# ===========================
# Prediction
# ===========================

if st.button("Predict Placement"):

    # Encode categorical inputs
    branch_encoded = branch_encoder.transform([branch])[0]
    tier_encoded = tier_encoder.transform([college_tier])[0]

    input_data = pd.DataFrame({
        "branch": [branch_encoded],
        "college_tier": [tier_encoded],
        "cgpa": [cgpa],
        "backlogs": [backlogs],
        "coding_skills": [coding_skills],
        "dsa_score": [dsa_score],
        "aptitude_score": [aptitude_score],
        "communication_skills": [communication_skills],
        "ml_knowledge": [ml_knowledge],
        "system_design": [system_design],
        "internships": [internships],
        "projects_count": [projects_count],
        "certifications": [certifications],
        "hackathons": [hackathons],
        "open_source_contributions": [open_source_contributions],
        "extracurriculars": [extracurriculars]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🎉 Student is likely to get placed!")
    else:
        st.error("❌ Student is unlikely to get placed.")

    st.write(f"**Placement Probability:** {probability * 100:.2f}%")

    st.progress(float(probability))
    st.divider()

st.subheader("📊 Model Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("Confusion Matrix")
    st.image(
        "model/confusion_matrix.png"
    )

with col2:
    st.write("Feature Importance")
    st.image(
        "model/feature_importance.png"
    )
    
    st.divider()

st.caption(
    "Developed using Python, Machine Learning and Streamlit"
)