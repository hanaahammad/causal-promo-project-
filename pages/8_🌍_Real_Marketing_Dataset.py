import streamlit as st
import pandas as pd
import csv
import os

st.title("🌍 Real Marketing Dataset (Kaggle)")

st.markdown("""
Upload the **marketing_campaign.csv** file.

The app will automatically detect:

- delimiter (; , \t | )
- encoding
- header row

and show a clean preview.
""")

uploaded = st.file_uploader("📥 Upload CSV file", type=["csv"])

def smart_read_csv(file):
    # try most common delimiters first
    delimiters = [";", ",", "\t", "|"]

    for sep in delimiters:
        try:
            df = pd.read_csv(file, sep=sep, engine="python")
            # if only 1 column, delimiter probably wrong
            if df.shape[1] > 1:
                return df, sep
        except Exception:
            continue

    # fallback: Python sniff
    file.seek(0)
    sample = file.read(2048).decode("utf-8", errors="ignore")
    dialect = csv.Sniffer().sniff(sample)
    sep = dialect.delimiter

    file.seek(0)
    df = pd.read_csv(file, sep=sep)

    return df, sep


if uploaded:
    df, sep_used = smart_read_csv(uploaded)

    st.success(f"✔ File loaded successfully using separator `{sep_used}`")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("👀 Data preview")
    st.dataframe(df.head())

    # Save into session
    st.session_state["train_df"] = df

    # Persist to disk
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/active_dataset.csv", index=False)

    st.success("✔ Saved to session and disk — you may now go to **Page 3**")
