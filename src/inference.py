import torch

def estimate_ate(model, X):
    with torch.no_grad():
        y0, y1, _ = model(X)
    return (y1 - y0).mean().item()

def counterfactual(model, x_row):
    with torch.no_grad():
        y0, y1, _ = model(x_row.unsqueeze(0))
    return y0.item(), y1.item(), (y1 - y0).item()
