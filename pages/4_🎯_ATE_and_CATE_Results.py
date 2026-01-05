import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("🎯 Causal Effect Results — ATE, CATE, ITE")


# ----------------------------------------------------
# 1) Safety checks
# ----------------------------------------------------
required = ["trained_model", "train_df", "features", "treat_col", "outcome_col"]

missing = [k for k in required if k not in st.session_state]

if missing:
    st.error("❌ Some required elements are missing in session state:")
    st.write(missing)

    st.info("""
👉 To use this page:

1️⃣ Page 2 or 8 → load or generate data  
2️⃣ Page 3 → select variables and train the model  
3️⃣ Return here to see causal results  
""")
    st.stop()


# ----------------------------------------------------
# 2) Load items from session
# ----------------------------------------------------
model = st.session_state["trained_model"]
df = st.session_state["train_df"]

features = st.session_state["features"]
treat_col = st.session_state["treat_col"]
outcome_col = st.session_state["outcome_col"]


# ----------------------------------------------------
# 3) Explanation text
# ----------------------------------------------------
st.markdown("""
## 🧠 What are we estimating?

Your model learned to predict **two potential outcomes** for each customer:

- **Y₀** → spending **without promotion**
- **Y₁** → spending **with promotion**

From those we derive:

| Metric | Meaning |
|--------|--------|
| **ITE** | Individual Treatment Effect for each single customer |
| **ATE** | Average Treatment Effect over all customers |
| **CATE** | Effect for groups (income level, age band, etc.) |

🟢 Positive effect → promotion **increases** spending  
🔴 Negative effect → promotion **decreases** spending  
""")


# ----------------------------------------------------
# 4) Compute ITE / ATE
# ----------------------------------------------------
X = torch.tensor(df[features].values, dtype=torch.float32)

model.eval()
with torch.no_grad():
    # your model returns y0_hat and y1_hat ONLY (2 outputs)
    y0_hat, y1_hat = model(X)

ITE = (y1_hat - y0_hat).numpy().flatten()
ATE = float(np.mean(ITE))


# ----------------------------------------------------
# 5) Show ATE
# ----------------------------------------------------
st.subheader("🎯 Average Treatment Effect (ATE)")

st.metric(
    label="Average effect of promotion on outcome",
    value=f"{ATE:.3f}"
)

st.caption("""
Positive → promotions increase spending on average  
Negative → promotions decrease spending on average  
""")


# ----------------------------------------------------
# 6) Plot ITE distribution
# ----------------------------------------------------
st.subheader("📊 Distribution of Individual Treatment Effects (ITE)")

fig1, ax1 = plt.subplots()
ax1.hist(ITE, bins=30)
ax1.set_title("Distribution of ITE")
ax1.set_xlabel("Effect size")
ax1.set_ylabel("Number of customers")
st.pyplot(fig1)


# ----------------------------------------------------
# 7) CATE by subgroup
# ----------------------------------------------------
st.subheader("🧩 Conditional Average Treatment Effect (CATE)")

group_var = st.selectbox(
    "Choose a grouping variable",
    options=[c for c in df.columns if c not in [treat_col, outcome_col]],
)

df_effects = df.copy()
df_effects["ITE"] = ITE

cate_table = df_effects.groupby(group_var)["ITE"].mean()

st.write("### 📄 CATE values by group")
st.write(cate_table)

fig2, ax2 = plt.subplots()
cate_table.plot(kind="bar", ax=ax2)
ax2.set_title("CATE by group")
ax2.set_ylabel("Average treatment effect")
st.pyplot(fig2)


# ----------------------------------------------------
# 8) Interpretation helper
# ----------------------------------------------------
st.markdown("""
## 🧭 How to interpret your results

### ✔ ATE large and positive
Promotions are effective **on average**  
→ scale the campaign

### ✔ ATE ≈ 0
Little to no effect  
→ redesign or target better

### ✔ ATE negative
Promotion **hurts revenue**
→ possible cannibalization or wrong targeting

---

### 🧩 Reading CATE

- some groups → strong positive effect  
- some groups → no effect  
- some groups → negative effect  

This shows **who to target** and **who to avoid**.

---

⚠️ Reminder  
This is **causal inference**, not prediction.  
We answer:

> “What would have happened if promotion was (not) given?”
""")
