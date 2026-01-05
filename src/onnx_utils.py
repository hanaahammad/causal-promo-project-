import torch
import onnx
import onnxruntime as ort

def export_to_onnx(model, path, input_dim):
    dummy = torch.randn(1, input_dim)
    torch.onnx.export(model, dummy, path,
                      input_names=["input"],
                      output_names=["y0","y1","t"])

def run_onnx(path, x):
    ort_sess = ort.InferenceSession(path)
    return ort_sess.run(None, {"input": x}) 
