import pandas as pd
import torch

def cate_by_group(model, df, feature, bins=3):
    df["group"] = pd.qcut(df[feature], q=bins, labels=False)

    cates = []
    for g in sorted(df["group"].unique()):
        sub = df[df["group"]==g]
        X = torch.tensor(sub[["income","loyalty"]].values, dtype=torch.float32)
        with torch.no_grad():
            y0, y1, _ = model(X)
        cates.append((g, (y1 - y0).mean().item()))
    return cates
