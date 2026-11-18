"""
Week 9 - California home price predictor (Streamlit).

Run:  streamlit run app.py
Needs price_model.joblib (made by train_app_model.py) in the same folder.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("price_model.joblib")

st.set_page_config(page_title="CA Home Price Predictor", page_icon="🏠")


@st.cache_resource
def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


st.title("🏠 California Home Price Predictor")
st.write(
    "Enter a few basic details for a single-family home and the model estimates "
    "its likely **sale (close) price**."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH.name}`. Run `python train_app_model.py` "
        "first and put the saved file next to app.py."
    )
    st.stop()

bundle = load_bundle()
model, features, ranges = bundle["model"], bundle["features"], bundle["ranges"]

col1, col2 = st.columns(2)
with col1:
    living_area = st.number_input("Living area (sq ft)", min_value=300, max_value=15000, value=1800, step=50)
    beds = st.number_input("Bedrooms", min_value=0, max_value=12, value=3, step=1)
with col2:
    baths = st.number_input("Bathrooms", min_value=0, max_value=12, value=2, step=1)
    lot_size = st.number_input("Lot size (sq ft)", min_value=0, max_value=500000, value=6000, step=250)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([[living_area, beds, baths, lot_size]], columns=features)
    pred = float(model.predict(row)[0])
    if bundle.get("log_target", False):
        pred = float(np.exp(pred))

    st.metric("Estimated sale price", f"${pred:,.0f}")

    # Rough range from the model's typical error on the test month.
    mdape = bundle["metrics"]["test_MdAPE_pct"] / 100
    low, high = pred * (1 - 2 * mdape), pred * (1 + 2 * mdape)
    st.write(f"Likely range: **${low:,.0f} – ${high:,.0f}**")

    # Warn when an input is outside what the model has mostly seen.
    inputs = dict(zip(features, [living_area, beds, baths, lot_size]))
    for name, val in inputs.items():
        lo, hi = ranges[name]
        if val < lo or val > hi:
            st.warning(f"{name} = {val:,} is outside the typical range ({lo:,.0f}–{hi:,.0f}); the estimate may be less reliable.")

with st.expander("About this model"):
    m = bundle["metrics"]
    st.write(
        f"- XGBoost trained on **{m['n_train']:,}** California single-family sales, using only the four inputs above.\n"
        f"- On a held-out recent month: median error **{m['test_MdAPE_pct']:.1f}%**, "
        f"**{m['test_within_20pct']:.0f}%** of homes within 20%, R² {m['test_R2']:.2f}.\n"
        "- It does **not** know the location, condition, or school district, so it is a rough guide. "
        "The full Week 8 model (R² ≈ 0.87) uses nearby sales and district data and is more accurate."
    )
