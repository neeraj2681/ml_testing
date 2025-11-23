import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration & Data Generation ---
torch.manual_seed(42)
np.random.seed(42)

# Generate data: y = x^2 between -2 and 2
X = np.linspace(-2, 2, 1000).reshape(-1, 1)
Y = X*(X**2 - 1)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# --- 2. Define the Networks ---

# N1: Single Hidden Layer (The "Wide" Network)
# We give it 500 neurons to try and brute-force the solution.
class SingleLayerNet(nn.Module):
    def __init__(self):
        super(SingleLayerNet, self).__init__()
        self.hidden = nn.Linear(1, 500)  # Massive width
        self.output = nn.Linear(500, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        return self.output(x)

# N2: Two Hidden Layers (The "Deep" Network)
# We give it only 10 neurons per layer.
# Total params is roughly (1*10) + (10*10) + (10*1) = 120 params
# vs N1 which is (1*500) + (500*1) = 1000 params.
class TwoLayerNet(nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(1, 10)   # Tiny width
        self.layer2 = nn.Linear(10, 10)  # Tiny width
        self.output = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)

# Initialize models
model_n1 = SingleLayerNet()
model_n2 = TwoLayerNet()

# --- 3. Training Loop ---
def train_model(model, name, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    losses = []
    
    print(f"Training {name}...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, Y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.5f}")
    
    return losses

losses_n1 = train_model(model_n1, "N1 (1 Layer, 500 units)")
losses_n2 = train_model(model_n2, "N2 (2 Layers, 10 units)")

# --- 4. Visualization ---
model_n1.eval()
model_n2.eval()

with torch.no_grad():
    pred_n1 = model_n1(X_tensor).numpy()
    pred_n2 = model_n2(X_tensor).numpy()

plt.figure(figsize=(12, 5))

# Plot 1: The Function Approximation
plt.subplot(1, 2, 1)
plt.title("Function Approximation: y = x^2")
plt.plot(X, Y, label="True Function (x^2)", color="black", linestyle="--", linewidth=2)
plt.plot(X, pred_n1, label="N1 (1 Layer, 500 units)", color="red", alpha=0.6)
plt.plot(X, pred_n2, label="N2 (2 Layers, 10 units)", color="blue", alpha=0.6)
plt.legend()
plt.grid(True)

# Plot 2: Loss Curves
plt.subplot(1, 2, 2)
plt.title("Training Loss (MSE)")
plt.plot(losses_n1, label="N1 Loss", color="red")
plt.plot(losses_n2, label="N2 Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log") # Log scale shows the convergence difference better
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()