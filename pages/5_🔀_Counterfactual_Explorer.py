import streamlit as st
import torch
import numpy as np
import pandas as pd
import hashlib

st.title("🔀 Counterfactual Explorer")


# -------------------------------------------------------------------
# 1) Safety checks
# -------------------------------------------------------------------
required = ["trained_model", "train_df", "features", "treat_col", "outcome_col"]

missing = [k for k in required if k not in st.session_state]

if missing:
    st.error("❌ Some required elements are missing:")
    st.write(missing)

    st.info("""
To use this page:

1️⃣ Load or generate data (Page 2 or 8)  
2️⃣ Train the model (Page 3)  
3️⃣ Come back here  
""")
    st.stop()


# -------------------------------------------------------------------
# 2) Load model and dataset
# -------------------------------------------------------------------
model = st.session_state["trained_model"]
df = st.session_state["train_df"]

features = st.session_state["features"]
treat_col = st.session_state["treat_col"]
outcome_col = st.session_state["outcome_col"]


# -------------------------------------------------------------------
# 3) 🔍 CLEAR VISUAL INDICATORS (what dataset & model are in use)
# -------------------------------------------------------------------
st.success("✅ Using dataset and model from the CURRENT session")

# dataset source indicator
if "synthetic_df" in st.session_state and st.session_state["train_df"].equals(
    st.session_state["synthetic_df"]
):
    st.info("📊 Dataset in use: **Synthetic generated dataset**")
else:
    st.info("📊 Dataset in use: **Kaggle / uploaded marketing dataset**")

# model hash identity
model_bytes = str(model.state_dict()).encode()
model_hash = hashlib.md5(model_bytes).hexdigest()[:8]
st.write(f"🧬 Model ID: `{model_hash}` (changes each time you retrain)")

# feature definitions
st.write("🧩 Features used for training:", features)
st.write("💊 Treatment column:", treat_col)
st.write("🎯 Outcome column:", outcome_col)


# -------------------------------------------------------------------
# 4) Explanation
# -------------------------------------------------------------------
st.markdown("""
## 🧠 What is a counterfactual?

A counterfactual answers:

> **What would THIS SAME customer have done if the promotion decision changed?**

- 🚫 **Ŷ₀** — predicted outcome **without** promotion  
- 💊 **Ŷ₁** — predicted outcome **with** promotion  
- 🎯 **ITE = Ŷ₁ − Ŷ₀** — individual uplift effect  

This is the basis of **uplift modeling** and **personalized targeting**.
""")


# -------------------------------------------------------------------
# 5) Select a customer
# -------------------------------------------------------------------
st.subheader("🧍 Select a customer from the dataset")

row_index = st.number_input(
    "Choose row index",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

customer = df.iloc[row_index]

st.write("### 👤 Selected customer feature values")
st.write(customer[features])


# -------------------------------------------------------------------
# 6) Predict counterfactual outcomes for this customer
# -------------------------------------------------------------------
x_tensor = torch.tensor(
    customer[features].values,
    dtype=torch.float32
).unsqueeze(0)

model.eval()
with torch.no_grad():
    # your DragonNet produces TWO heads: y0_hat, y1_hat
    y0_pred, y1_pred = model(x_tensor)

y0 = float(y0_pred.item())
y1 = float(y1_pred.item())
ite = y1 - y0


# -------------------------------------------------------------------
# 7) Display ITE results
# -------------------------------------------------------------------
st.subheader("🔮 Counterfactual outcomes for this customer")

col1, col2 = st.columns(2)

with col1:
    st.metric("🚫 No promotion (Ŷ₀)", f"{y0:.2f}")

with col2:
    st.metric("💊 With promotion (Ŷ₁)", f"{y1:.2f}")

st.metric("🎯 Individual Treatment Effect (ITE)", f"{ite:.2f}")

st.caption("""
- 🟢 Positive ITE → promotion **helps THIS customer**
- 🔴 Negative ITE → promotion **hurts or is unnecessary**
""")


# -------------------------------------------------------------------
# 8) 🎛 Interactive “what-if” sliders
# -------------------------------------------------------------------
st.subheader("🎛 Try changing customer attributes (what-if analysis)")

modified_values = []

for feat in features:
    original = float(customer[feat])

    new_val = st.slider(
        f"{feat}",
        min_value=float(df[feat].min()),
        max_value=float(df[feat].max()),
        value=original
    )

    modified_values.append(new_val)

x_mod = torch.tensor(modified_values, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    y0_mod, y1_mod = model(x_mod)

y0_mod = float(y0_mod.item())
y1_mod = float(y1_mod.item())
ite_mod = y1_mod - y0_mod


# -------------------------------------------------------------------
# 9) Show modified result
# -------------------------------------------------------------------
st.markdown("### 🧭 Counterfactual after your changes")

st.write(f"🚫 Ŷ₀ (no promotion) = **{y0_mod:.2f}**")
st.write(f"💊 Ŷ₁ (promotion) = **{y1_mod:.2f}**")
st.write(f"🎯 New ITE = **{ite_mod:.2f}**")

st.caption("""
You just simulated **personalized uplift**.

This is used in:

- targeted marketing
- churn prevention
- pricing and A/B testing
""")
