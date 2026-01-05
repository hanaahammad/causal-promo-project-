import streamlit as st
import torch
import onnx
import onnxruntime as ort
import numpy as np
import io

st.title("📦 Export & Use Model in ONNX Format")

# ----------------------------------------------------
# 1) Check prerequisites
# ----------------------------------------------------
if "trained_model" not in st.session_state or "features" not in st.session_state:
    st.error("❌ No trained model found.")
    st.info("""
Go to **Page 3 – Train Deep Model**, train a model,
then return here to export and use it.
""")
    st.stop()

model = st.session_state["trained_model"]
features = st.session_state["features"]
input_dim = len(features)

st.success("✔ Trained model detected — ready to export or use")


# ----------------------------------------------------
# 2) Why ONNX matters
# ----------------------------------------------------
st.header("🤔 Why ONNX?")

st.markdown("""
We **train** in PyTorch, but deployment may happen in:

- Java
- C#
- C++
- mobile apps
- cloud inference services

**ONNX is the universal model format** that allows that.

👉 Train once  
👉 Deploy anywhere  
""")


# ----------------------------------------------------
# 3) Show export code for learning
# ----------------------------------------------------
st.header("🧾 Export code (PyTorch → ONNX)")

export_code = r"""
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["features"],
    output_names=["y0", "y1", "propensity"],
    opset_version=17,
    dynamic_axes={"features": {0: "batch"}}
)
"""

st.code(export_code, language="python")


# ----------------------------------------------------
# 4) Perform export
# ----------------------------------------------------
st.header("📤 Export model to ONNX")

dummy_input = torch.randn(1, input_dim)

onnx_buffer = io.BytesIO()

if st.button("🚀 Export to ONNX"):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_buffer,
        input_names=["features"],
        output_names=["y0", "y1", "propensity"],
        opset_version=17,
        dynamic_axes={"features": {0: "batch"}},
    )

    onnx_bytes = onnx_buffer.getvalue()
    st.session_state["onnx_bytes"] = onnx_bytes

    st.success("✅ Model exported successfully!")

    st.download_button(
        "💾 Download ONNX file",
        onnx_bytes,
        file_name="causal_model.onnx",
        mime="application/octet-stream",
    )


# ----------------------------------------------------
# 5) Use ONNX model for inference
# ----------------------------------------------------
st.header("🧪 Use ONNX model for prediction")

if "onnx_bytes" not in st.session_state:
    st.info("👉 Export the model above first to enable ONNX inference.")
else:
    onnx_bytes = st.session_state["onnx_bytes"]

    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_bytes)

    # Random sample input
    sample = np.random.randn(1, input_dim).astype(np.float32)

    if st.button("▶️ Run prediction using ONNX model"):
        outputs = session.run(
            None, {"features": sample}
        )

        y0_onnx, y1_onnx, prop_onnx = outputs

        st.write("### ✅ ONNX inference results")
        st.write(f"ŷ₀ (no treatment): **{y0_onnx[0][0]:.4f}**")
        st.write(f"ŷ₁ (treatment): **{y1_onnx[0][0]:.4f}**")

        st.write("Propensity score:", float(prop_onnx[0][0]))


# ----------------------------------------------------
# 6) Compare PyTorch vs ONNX predictions
# ----------------------------------------------------
st.header("⚖️ Consistency check: PyTorch vs ONNX")

if st.button("🔍 Compare predictions"):
    # PyTorch prediction
    torch_input = torch.tensor(sample, dtype=torch.float32)
    with torch.no_grad():
        y0_t, y1_t, _ = model(torch_input)

    # ONNX prediction
    outputs = session.run(None, {"features": sample})
    y0_o, y1_o, _ = outputs

    st.write("### PyTorch")
    st.write(float(y0_t), float(y1_t))

    st.write("### ONNX")
    st.write(float(y0_o[0][0]), float(y1_o[0][0]))

    diff = abs(float(y0_t) - float(y0_o[0][0])) + abs(float(y1_t) - float(y1_o[0][0]))

    st.write(f"Total difference = **{diff:.6f}**")

    if diff < 1e-3:
        st.success("✔ The exported ONNX model matches PyTorch outputs.")
    else:
        st.warning("⚠ Noticeable difference detected — check opset or model layers.")


# ----------------------------------------------------
# 7) Final explanation block
# ----------------------------------------------------
st.header("📘 What we accomplished")

st.markdown("""
You have now:

✔ trained a deep causal model  
✔ exported it to ONNX  
✔ downloaded the portable model  
✔ ran predictions using ONNX Runtime  
✔ verified PyTorch vs ONNX consistency  

This is exactly what happens in **real deployment pipelines**:

1. Data scientists train in PyTorch  
2. Model exported to ONNX  
3. Engineers deploy ONNX model in apps or APIs  
""")
