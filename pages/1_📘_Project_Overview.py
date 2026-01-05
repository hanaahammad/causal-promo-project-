import streamlit as st
from pathlib import Path
from PIL import Image

st.title("📘 Project Overview")

st.markdown("""
### 🎯 Objective

We want to estimate **the causal effect of marketing promotions** on customer purchases.

Instead of only predicting *who will buy*, we answer:

> **What would this customer do with vs without a promotion?**

This requires **causal inference + deep learning**.
""")

st.markdown("""
### 🧭 Pipeline Steps

1. Define treatment, outcome, confounders  
2. Generate or load data  
3. Specify causal assumptions (DAG)  
4. Train deep neural causal model (DragonNet)  
5. Estimate ATE / CATE  
6. Explore counterfactuals  
7. Export model to ONNX for deployment
""")

# pipeline image if exists
img_path = Path(__file__).parents[1] / "assets" / "pipeline_diagram.png"

if img_path.exists():
    st.image(Image.open(img_path), caption="Causal ML Pipeline")
else:
    st.info("Pipeline diagram image will appear here once added.")

st.success("Use the sidebar menu to move through the pipeline.")
