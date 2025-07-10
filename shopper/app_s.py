# app_s.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
kmeans = joblib.load('Downloads/GUVI Practise/Shopper_Spectrum/kmeans_model.pkl')
scaler = joblib.load('Downloads/GUVI Practise/Shopper_Spectrum/scaler.pkl')
similarity_df = joblib.load('Downloads/GUVI Practise/Shopper_Spectrum/similarity_matrix.pkl')  # Product similarity matrix

#  Page Setup & Custom CSS

st.set_page_config(
    page_title="🛍️ Shopper Spectrum",
    page_icon="🛒",
    layout="centered"
)

# Use Google Fonts + soft colors
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background-color: #fafafa;
    }

    h1, h2, h3 {
        color: #333333;
    }

    .stButton>button {
        background-color: #ff7f50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }

    .stButton>button:hover {
        background-color: #ff5722;
    }

    .block-container {
        padding-top: 2rem;
    }

    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/263/263115.png",
    width=80
)
st.sidebar.title("🛍️ Shopper Spectrum")
page = st.sidebar.radio("🗂️ Go to:", ["Customer Segmentation", "Product Recommendation"])
st.sidebar.markdown("---")
st.sidebar.caption("Classic Shopping Theme • Streamlit 🚀")

# Customer Segmentation

if page == "Customer Segmentation":
    st.title("🎯 Customer Segmentation")
    st.write(
        "Understand your shoppers by classifying them into segments "
        "based on **Recency, Frequency, Monetary (RFM)** scores."
    )

    with st.form("segmentation_form"):
        recency = st.number_input("🗓️ Recency (days since last purchase):", min_value=0, value=30)
        frequency = st.number_input("🔄 Frequency (number of purchases):", min_value=0, value=5)
        monetary = st.number_input("💰 Monetary (total spend):", min_value=0.0, value=500.0, step=50.0)
        submit = st.form_submit_button("🔍 Predict Segment")

    if submit:
        scaled_input = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(scaled_input)[0]

        if cluster == 0:
            label = "💎 High-Value Customer"
        elif cluster == 1:
            label = "🟢 Regular Customer"
        elif cluster == 2:
            label = "🟡 Occasional Customer"
        else:
            label = "🔴 At-Risk Customer"

        st.success(f"✅ Predicted Segment: **{label}** (Cluster #{cluster})")

# Product Recommendation

elif page == "Product Recommendation":
    st.title("🛒 Product Recommendation")
    st.write(
        "Help shoppers find products they'll love — based on "
        "**purchase similarity** with other products."
    )

    product_list = sorted(list(similarity_df.columns))
    product_name = st.selectbox("📦 Select a product:", product_list)

    if st.button("🔍 Get Recommendations"):
        if product_name in similarity_df.columns:
            similar = similarity_df[product_name].sort_values(ascending=False)[1:6]
            st.write(f"**Top 5 products similar to:** _{product_name}_")
            for i, (prod, score) in enumerate(similar.items(), start=1):
                st.info(f"{i}. **{prod}** (Similarity: {score:.2f})")
        else:
            st.warning("⚠️ Product not found in similarity matrix.")

# Footer

st.markdown("---")
st.caption("🛍️ Shopper Spectrum • E-Commerce Analytics • Streamlit Cloud Ready 🚀")
