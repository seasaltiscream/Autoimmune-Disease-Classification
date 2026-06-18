import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Autoimmune Predictor",
    page_icon="🩺",
    layout="centered"
)

t.write("Python:", sys.version)
st.write("scikit-learn:", sklearn.__version__)

try:
    import sklearn._loss
    st.success("✅ sklearn._loss imported")
except Exception as e:
    st.error(f"❌ {e}")

st.stop()

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_systems():
    set1 = joblib.load("autoimmune_system_set1_final.pkl")
    set2 = joblib.load("autoimmune_system_set2_final.pkl")
    return set1, set2

set1, set2 = load_systems()

# =========================
# TITLE
# =========================
st.title("Autoimmune Disease Predictor")
# st.caption("Academic Machine Learning Prototype")

st.markdown("---")

# =========================
# SELECT DATASET
# =========================
dataset_choice = st.selectbox("Choose Model", ["Dataset 1", "Dataset 2"])

system = set1 if dataset_choice == "Dataset 1" else set2

model = system["model"]
label_encoder = system["label_encoder"]
features = system["feature_order"]
medians = system.get("medians", {})

st.write(f"**Best Model:** {system['model_name']}")

# =========================
# INPUT STORAGE
# =========================
inputs = {}

# =====================================================
# DATASET 1 (UNCHANGED LOGIC)
# =====================================================
if dataset_choice == "Dataset 1":

    st.subheader("Patient Input (Dataset 1)")

    for feature in features:

        default_value = float(medians.get(feature, 0))

        if feature in ["Gender", "HLA-B27", "ANA",
                        "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm",
                        "Joint_pain", "Fatigue_or_chronic_tiredness",
                        "Dry_eyes_and/or_mouth", "Rashes_and_skin_lesions"]:

            if feature == "Gender":
                val = st.selectbox(feature, ["Female", "Male"])
                inputs[feature] = 1 if val == "Male" else 0

            elif feature == "ANA":
                val = st.selectbox(feature, ["Negative", "Positive"])
                inputs[feature] = 1 if val == "Positive" else 0

            else:
                val = st.selectbox(feature, ["No", "Yes"])
                inputs[feature] = 1 if val == "Yes" else 0

        else:
            inputs[feature] = st.number_input(feature, value=default_value)

# =====================================================
# DATASET 2 (IMPROVED UX - CLEAR USER GUIDANCE)
# =====================================================
else:

    st.subheader("Patient Input (Dataset 2)")

    st.info("Fill in the sections below. Required fields are automatically handled if left empty.")

    # =========================
    # BASIC CLINICAL DATA
    # =========================
    with st.expander("🩸 Basic Clinical Data (REQUIRED)", expanded=True):
        st.caption("These are core blood test indicators used in all diagnoses.")

        for feature in features:

            if feature in ["Age", "Gender", "Sickness_Duration_Months",
                           "RBC_Count", "Hemoglobin", "Hematocrit",
                           "MCV", "MCH", "MCHC", "RDW",
                           "WBC_Count", "PLT_Count",
                           "Neutrophils", "Lymphocytes",
                           "Monocytes", "Eosinophils",
                           "Basophils", "MPV"]:

                default_value = float(medians.get(feature, 0))

                if feature == "Gender":
                    val = st.selectbox(
                        feature,
                        ["Female", "Male"],
                        help="Required for demographic classification"
                    )
                    inputs[feature] = 1 if val == "Male" else 0

                else:
                    inputs[feature] = st.number_input(
                        feature,
                        value=default_value,
                        help="Required clinical measurement"
                    )

    # =========================
    # AUTOIMMUNE MARKERS
    # =========================
    with st.expander("🧬 Autoimmune Markers (IMPORTANT - RECOMMENDED)"):
        st.caption("These help differentiate between autoimmune diseases.")

        for feature in features:

            if feature in ["ANA", "Rheumatoid_factor", "ACPA",
                           "Anti_dsDNA", "Anti_Sm", "Anti_Ro_SSA",
                           "Anti_La_SSB", "ANCA", "Anti_TPO", "Anti_Tg",
                           "Anti_SMA"]:

                val = st.selectbox(
                    feature,
                    ["No", "Yes"],
                    help="Recommended: improves diagnostic accuracy"
                )
                inputs[feature] = 1 if val == "Yes" else 0

    # =========================
    # SYMPTOMS
    # =========================
    with st.expander("🧠 Symptoms (SUPPORTING INFORMATION)"):
        st.caption("Symptoms help support the diagnosis but are not definitive.")

        for feature in features:

            if feature in ["Low_grade_fever", "Fatigue_or_chronic_tiredness",
                           "Dizziness", "Weight_loss",
                           "Rashes_and_skin_lesions",
                           "Stiffness_in_the_joints",
                           "Dry_eyes_and/or_mouth",
                           "General_unwell_feeling",
                           "Joint_pain"]:

                val = st.selectbox(
                    feature,
                    ["No", "Yes"],
                    help="Symptom presence (supporting evidence)"
                )
                inputs[feature] = 1 if val == "Yes" else 0

# =========================
# FILL MISSING FEATURES
# =========================
for col in features:
    if col not in inputs:
        inputs[col] = float(medians.get(col, 0))

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    input_df = pd.DataFrame([inputs])
    input_df = input_df.reindex(columns=features, fill_value=0)

    pred = model.predict(input_df)[0]
    disease = label_encoder.inverse_transform([pred])[0]

    st.success(f"### Prediction: {disease}")

    # =========================
    # PROBABILITY OUTPUT
    # =========================
    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(input_df)[0]

        prob_df = pd.DataFrame({
            "Disease": label_encoder.classes_,
            "Probability": probs
        }).sort_values("Probability", ascending=False)

        top1 = prob_df.iloc[0]["Probability"]

        if top1 >= 0.80:
            st.info(f"Confidence: 🟢 High ({top1:.2f})")
        elif top1 >= 0.60:
            st.info(f"Confidence: 🟡 Medium ({top1:.2f})")
        else:
            st.warning(f"Confidence: 🔴 Low ({top1:.2f})")

        st.subheader("Top Predictions")
        st.dataframe(prob_df.head(3), use_container_width=True)

        st.bar_chart(prob_df.set_index("Disease"))

# =========================
# FOOTER
# =========================
st.markdown("---")
# st.caption("For research purposes only. Not a medical diagnostic tool.")
