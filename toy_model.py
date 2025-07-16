import torch
from torch import nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"{x.dtype=}")
        x = self.fc1(x)
        print(f"after fc1 {x.dtype=}")
        x = self.relu(x)
        print(f"after relu {x.dtype=}")
        x = self.ln(x)
        print(f"after layer norm {x.dtype=}")
        x = self.fc2(x)
        print(f"after fc2 {x.dtype=}")
        return x


with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    model = ToyModel(100, 200).to("cuda")
    inputs = torch.arange(100, dtype=torch.float32, device="cuda")
    targets = torch.arange(200, dtype=torch.float32, device="cuda")
    predictions = model(inputs)
    loss = torch.nn.functional.cross_entropy(predictions, targets)
    print(f"{loss.dtype=}")
    loss.backward()
    for name, param in model.named_parameters():
        print(f"{name} grad dtype = {param.grad.dtype}")