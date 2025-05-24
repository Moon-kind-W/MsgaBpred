import torch
import torch.nn as nn
import torch.nn.functional as F
class AE(nn.Module):
    def __init__(self, dim_in, dim_out, hidden, dropout = 0., bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden, bias=bias),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_out, bias=bias),
            nn.LayerNorm(dim_out),
        )
    def forward(self, x):
        return self.net(x)