import torch
import torch.nn as nn

class SimpleDragonNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.y0_head = nn.Linear(16, 1)
        self.y1_head = nn.Linear(16, 1)
        self.t_head = nn.Linear(16, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.y0_head(h), self.y1_head(h), self.t_head(h)
