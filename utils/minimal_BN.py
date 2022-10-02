import torch.nn as nn
import torch

# With Learnable Parameters
m = nn.BatchNorm1d(3)
m2 = nn.LayerNorm(3)
input = torch.randn(4, 3)
for name, p in m.named_parameters():
    if p.requires_grad:
        print(name, p)
for name, p in m2.named_parameters():
    if p.requires_grad:
        print(name, p)
output = m(input)
output_2 = m2(input)
print(input, output, output_2)
