import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
# try session_state first
df = st.session_state.get("train_df")

# fallback: auto-load dataset saved on disk
import os, pandas as pd
if df is None and os.path.exists("data/active_dataset.csv"):
    df = pd.read_csv("data/active_dataset.csv")
    st.session_state["train_df"] = df
    st.info("📂 Dataset restored automatically (new session detected).")

st.set_page_config(page_title="Train Deep Causal Model", page_icon="🧠")

st.title("🧠 Train Deep Causal Model (DragonNet-style)")

# -------------------------------
# 1) Load dataset safely
# -------------------------------

df = st.session_state.get("train_df", None)

# fallback: reload automatically from disk if exists
if df is None and os.path.exists("data/active_dataset.csv"):
    df = pd.read_csv("data/active_dataset.csv")
    st.session_state["train_df"] = df
    st.info("📂 Dataset reloaded automatically from disk (new session detected).")

if df is None:
    st.error("❌ No dataset found. Go to **Page 2** or **Page 8** first to load or generate data.")
    st.stop()

st.success("✅ Dataset is loaded and ready for training.")
st.write("📄 Data preview:")
st.dataframe(df.head())

# -------------------------------
# 2) Basic usage instructions
# -------------------------------

st.markdown("""
### 🧭 How to use this page

1. Select:
- feature columns (X)
- treatment column (T) — must be binary
- outcome column (Y)

2. Set training hyperparameters

3. Click **Train Model**

You will see live:
- loss curve
- progress bar
- epoch counter
""")

# -------------------------------
# 3) Variable selection
# -------------------------------

cols = list(df.columns)

features = st.multiselect("🧩 Feature columns (X)", cols)

treat_col = st.selectbox("💊 Treatment column (T) — must be binary 0/1", cols)

outcome_col = st.selectbox("🎯 Outcome column (Y)", cols)

# -------------------------------
# 4) Validate treatment as binary
# -------------------------------

# convert yes/no etc to {0,1}
df[treat_col] = df[treat_col].astype("category").cat.codes

unique_t = sorted(df[treat_col].unique().tolist())
st.write(f"🧪 Unique values in treatment after conversion: {unique_t}")

if len(unique_t) != 2:
    st.error("🚨 Treatment must be binary (exactly 2 unique values after conversion).")
    st.stop()

# Save selections for other pages
st.session_state["features"] = features
st.session_state["treat_col"] = treat_col
st.session_state["outcome_col"] = outcome_col

# -------------------------------
# 5) Prepare tensors for PyTorch
# -------------------------------

if len(features) == 0:
    st.warning("⚠️ Please select at least one feature.")
    st.stop()

X = torch.tensor(df[features].values, dtype=torch.float32)
T = torch.tensor(df[[treat_col]].values, dtype=torch.float32)
Y = torch.tensor(df[[outcome_col]].values, dtype=torch.float32)

st.success("📦 Data converted to PyTorch tensors successfully.")

# -------------------------------
# 6) Visual preview of distributions
# -------------------------------

st.subheader("📊 Feature distributions")

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df[outcome_col], bins=30)
ax.set_title("Outcome distribution")
st.pyplot(fig)

# -------------------------------
# 7) Define simplified DragonNet model
# -------------------------------

class DragonNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.y0_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.y1_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        rep = self.shared(x)
        y0 = self.y0_head(rep)
        y1 = self.y1_head(rep)
        return y0, y1

model = DragonNet(input_dim=len(features))

# -------------------------------
# 8) Training controls
# -------------------------------

st.subheader("⚙️ Training parameters")

epochs = st.slider("Epochs", 10, 300, 100)
lr = st.number_input("Learning rate", 0.0001, 0.05, 0.001)

train_button = st.button("🚀 Train Model")

# -------------------------------
# 9) Training loop with live progress
# -------------------------------

if train_button:

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    progress = st.progress(0)
    status = st.empty()

    losses = []

    for epoch in range(epochs):

        y0_pred, y1_pred = model(X)

        # factual outcome
        y_pred = torch.where(T == 1, y1_pred, y0_pred)

        loss = loss_fn(y_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        progress.progress((epoch + 1) / epochs)
        status.text(f"Epoch {epoch+1}/{epochs} — Loss = {loss.item():.4f}")

        time.sleep(0.01)

    st.success("🎉 Training complete")

    # store for later pages
    st.session_state["trained_model"] = model
    st.session_state["X_tensor"] = X
    st.session_state["T_tensor"] = T
    st.session_state["Y_tensor"] = Y
    st.session_state["train_losses"] = losses

    # plot loss
    fig2, ax2 = plt.subplots()
    ax2.plot(losses)
    ax2.set_title("Training Loss Curve")
    st.pyplot(fig2)

    st.info("📌 Model saved in session. You can now go to **Page 4 (ATE/CATE)** or **Page 5 (Counterfactual Explorer)**.")

