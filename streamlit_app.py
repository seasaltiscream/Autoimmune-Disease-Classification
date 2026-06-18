import streamlit as st
import sys
import sklearn
import joblib
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Debug Environment",
    page_icon="🩺",
    layout="centered"
)

st.title("Streamlit Debug")

st.subheader("Python")
st.code(sys.version)

st.subheader("Installed Packages")
st.write("scikit-learn:", sklearn.__version__)
st.write("joblib:", joblib.__version__)
st.write("pandas:", pd.__version__)

st.subheader("Testing sklearn._loss")

try:
    import sklearn._loss
    st.success("✅ sklearn._loss imported successfully")
    st.write(sklearn._loss.__file__)
except Exception as e:
    st.error("❌ Failed to import sklearn._loss")
    st.exception(e)

st.subheader("Testing GradientBoosting import")

try:
    from sklearn.ensemble import GradientBoostingClassifier
    st.success("✅ GradientBoostingClassifier imported successfully")
except Exception as e:
    st.error("❌ Failed to import GradientBoostingClassifier")
    st.exception(e)

st.subheader("Testing _loss module directly")

try:
    from sklearn._loss.loss import HalfMultinomialLoss
    st.success("✅ HalfMultinomialLoss imported successfully")
except Exception as e:
    st.error("❌ Failed to import HalfMultinomialLoss")
    st.exception(e)

st.info("Stopping before loading the pickle.")
st.stop()
