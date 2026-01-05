import streamlit as st
import numpy as np
import pandas as pd

st.title("📊 Synthetic Marketing Dataset Generator")

st.markdown("""
This page creates a **synthetic marketing dataset** that we will use
later to train the causal deep learning model.

Each row represents **one customer**.

- some customers receive a **promotion** (`treatment = 1`)
- some do **not** (`treatment = 0`)
- we observe their **spending** (`spend`)

This dataset is designed to include **confounding** so that
causal inference is actually meaningful.
""")

# ---------------------------------------------------------
# User parameter
# ---------------------------------------------------------
n = st.slider("Number of customers", 100, 10000, 2000, step=100)

# ---------------------------------------------------------
# Generate data
# ---------------------------------------------------------
np.random.seed(42)

income = np.random.normal(50, 15, n)
loyalty = np.random.uniform(0, 1, n)

# probability of receiving promotion depends on features (→ confounding)
p_treatment = 1 / (1 + np.exp(-(0.05 * income + 3 * loyalty - 5)))

treatment = np.random.binomial(1, p_treatment)

# true treatment effect ≈ +5 on average
spend = (
    30
    + 0.8 * income
    + 10 * loyalty
    + 5 * treatment
    + np.random.normal(0, 5, n)
)

df = pd.DataFrame({
    "income": income,
    "loyalty": loyalty,
    "treatment": treatment,
    "spend": spend,
})

# save dataset for other pages
st.session_state["df"] = df

# ---------------------------------------------------------
# Display and explain dataset
# ---------------------------------------------------------
st.success("✅ Synthetic dataset generated and stored for training.")

st.write("### 📊 Preview of the dataset")
st.dataframe(df.head())

st.write("### 📐 Shape:", df.shape)

st.markdown("""
### 🧾 Column meanings

- **income** → simulated customer income  
- **loyalty** → loyalty score between 0 and 1  
- **treatment**
  - 1 → promotion sent  
  - 0 → no promotion  
- **spend** → total money spent by the customer  

👉 This dataset will automatically be used in **Page 3 – Model Training**.
""")
