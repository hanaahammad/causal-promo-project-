import torch
import torch.nn as nn

def train_dragonnet(model, X, T, Y, epochs=200, lr=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    losses = []

    for _ in range(epochs):
        optimizer.zero_grad()

        y0, y1, t_logit = model(X)

        # pick correct head based on treatment
        y_pred = torch.where(T == 1, y1, y0)

        loss = mse(y_pred, Y) + bce(t_logit, T)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return model, losses
