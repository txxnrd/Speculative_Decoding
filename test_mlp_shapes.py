import torch
import torch.nn as nn

# Test SimpleMLP shapes
class TestMLP(nn.Module):
    def __init__(self, input_dim=8192, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.mlp(x)
        print(f"MLP output shape before view: {output.shape}")
        flattened = output.view(-1)
        print(f"MLP output shape after view: {flattened.shape}")
        return flattened

# Test
mlp = TestMLP()
batch_size = 16
input_tensor = torch.randn(batch_size, 8192)
target_tensor = torch.randint(0, 2, (batch_size,)).float()

print(f"Input shape: {input_tensor.shape}")
print(f"Target shape: {target_tensor.shape}")

output = mlp(input_tensor)
print(f"Final output shape: {output.shape}")

# Test loss
criterion = nn.BCEWithLogitsLoss()
try:
    loss = criterion(output, target_tensor)
    print(f"Loss computed successfully: {loss.item()}")
except Exception as e:
    print(f"Error: {e}") 