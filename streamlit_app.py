import streamlit as st
import joblib
import traceback

st.title("Pickle Debug")

try:
    obj = joblib.load("autoimmune_system_set1_final.pkl")
    st.success("✅ Dataset 1 loaded successfully!")
    st.write(type(obj))
    st.write(obj.keys())

except Exception as e:
    st.error("❌ Failed to load Dataset 1")
    st.code(traceback.format_exc())
