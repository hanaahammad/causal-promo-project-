import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time

st.title("🧠 Train Deep Causal Model (DragonNet-style)")


# -----------------------------
# Reload dataset if needed
# -----------------------------
if "train_df" not in st.session_state and os.path.exists("data/active_dataset.csv"):
    st.session_state["train_df"] = pd.read_csv("data/active_dataset.csv")
    st.info("📥 Reloaded dataset from previous session.")


# -----------------------------
# Select dataset
# -----------------------------
if "synthetic_df" in st.session_state:
    df = st.session_state["synthetic_df"]
    dataset_type = "synthetic"
    st.success("✔ Using synthetic dataset")

elif "train_df" in st.session_state:
    df = st.session_state["train_df"]
    dataset_type = "real"
    st.success("✔ Using Kaggle / real dataset")

else:
    st.error("❌ No dataset found. Go to Page 2 or 8 first.")
    st.stop()

st.dataframe(df.head())


# -----------------------------
# Explanation
# -----------------------------
st.markdown("""
### 🎯 What happens here

We train a **deep neural causal model** to estimate the impact of promotions:

- ATE – Average Treatment Effect  
- CATE – Effect by subgroup  
- ITE – Effect per customer  

We must select:

- **X** features (customer characteristics)
- **T** treatment (promotion yes/no)
- **Y** outcome (spending)
""")


# -----------------------------
# Column selection
# -----------------------------
numeric_cols = [c for c in df.columns if df[c].dtype != "object"]
binary_cols = [c for c in df.columns if df[c].nunique() == 2]

treat_col = st.selectbox(
    "Treatment column (binary 0/1)",
    options=binary_cols if len(binary_cols) > 0 else df.columns,
)

# total spend auto
spend_cols = [c for c in df.columns if str(c).startswith("Mnt")]

if "TotalSpend" not in df.columns and len(spend_cols) > 0:
    df["TotalSpend"] = df[spend_cols].sum(axis=1)

outcome_candidates = ["TotalSpend"] + spend_cols
outcome_candidates = [c for c in outcome_candidates if c in df.columns]

outcome_col = st.selectbox(
    "Outcome column (spending)",
    options=outcome_candidates if len(outcome_candidates) > 0 else df.columns,
)

features = st.multiselect(
    "Feature columns (X)",
    options=list(df.columns),
    default=[c for c in numeric_cols if c not in [treat_col, outcome_col]][:6],
)


# -----------------------------
# Save to session (important!)
# -----------------------------
st.session_state["features"] = features
st.session_state["treat_col"] = treat_col
st.session_state["outcome_col"] = outcome_col


# -----------------------------
# Normalize treatment
# -----------------------------
st.subheader("🔧 Treatment normalization")

if dataset_type == "synthetic":
    st.info("Synthetic dataset detected — already binary ✔")

else:
    st.info("Real dataset detected — cleaning treatment…")

    df[treat_col] = df[treat_col].astype(str).str.lower()
    df[treat_col] = df[treat_col].map({
        "yes": 1, "true": 1, "y": 1,
        "no": 0, "false": 0, "n": 0
    })

    df[treat_col] = df[treat_col].replace({-1: 0, 2: 1})

    df = df.dropna(subset=[treat_col]).copy()
    df[treat_col] = df[treat_col].astype(int)

st.write("Unique values:", df[treat_col].unique())

if df[treat_col].nunique() != 2:
    st.error("🚨 Treatment is not binary — choose Response or AcceptedCmp columns.")
    st.stop()


# -----------------------------
# Tensors
# -----------------------------
X = torch.tensor(df[features].values, dtype=torch.float32)
T = torch.tensor(df[[treat_col]].values, dtype=torch.float32)
Y = torch.tensor(df[[outcome_col]].values, dtype=torch.float32)


# -----------------------------
# DragonNet
# -----------------------------
class DragonNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.y0 = nn.Linear(32, 1)
        self.y1 = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.y0(h), self.y1(h)

model = DragonNet(len(features))


# -----------------------------
# Training controls
# -----------------------------
st.subheader("🚀 Train model with live progress")

epochs = st.slider("Epochs", 50, 500, 200, 50)
lr = st.number_input("Learning rate", 1e-4, 1e-2, 1e-3)

start = st.button("▶️ Start training")
stop_flag = st.button("🛑 Stop training")

progress_bar = st.progress(0)
status = st.empty()
eta_box = st.empty()
chart = st.empty()


# -----------------------------
# Training loop WITH animation
# -----------------------------
def train_model():
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    losses = []
    t0 = time.time()

    for epoch in range(epochs):

        if stop_flag:
            st.warning("Training stopped by user ❌")
            break

        opt.zero_grad()
        y0, y1 = model(X)
        yhat = T * y1 + (1 - T) * y0

        loss = mse(yhat, Y)
        loss.backward()
        opt.step()

        losses.append(loss.item())

        # progress
        progress_bar.progress((epoch + 1) / epochs)
        status.write(f"Epoch {epoch+1}/{epochs} | Loss={loss.item():.5f}")

        elapsed = time.time() - t0
        remaining = (epochs - (epoch + 1)) * (elapsed / (epoch + 1))
        eta_box.info(f"⏱ ETA ≈ {remaining:.1f} seconds")

        # loss curve live plot
        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.set_title("Training Loss (live)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        chart.pyplot(fig)

    return losses


if start:
    losses = train_model()

    st.success("🎉 Training finished!")

    st.session_state["trained_model"] = model
    st.session_state["losses"] = losses
    st.session_state["train_df"] = df
