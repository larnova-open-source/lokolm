import torch.nn as nn
import torch.nn.functional as F


# Position-wise Feed-Forward Network

class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=4):
        super().__init__()
        # Expand to mlp_ratio * d_model, then project back down
        self.c_fc = nn.Linear(d_model, mlp_ratio * d_model)
        self.c_proj = nn.Linear(mlp_ratio * d_model, d_model)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x
