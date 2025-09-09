# encoder.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)
