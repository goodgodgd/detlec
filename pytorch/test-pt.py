import torch
x = torch.rand(2, 3, 4)
print(torch.sum(x, dim=[1, 2]))
