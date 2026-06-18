import streamlit as st
import joblib
import traceback

st.title("Loading Both Models")

try:
    set1 = joblib.load("autoimmune_system_set1_final.pkl")
    st.success("✅ Dataset 1 loaded")
except Exception:
    st.error("❌ Dataset 1 failed")
    st.code(traceback.format_exc())

try:
    set2 = joblib.load("autoimmune_system_set2_final.pkl")
    st.success("✅ Dataset 2 loaded")
except Exception:
    st.error("❌ Dataset 2 failed")
    st.code(traceback.format_exc())
